// SPDX-License-Identifier: Apache-2.0
/// @file persist_distance.cpp
/// @brief Saving and loading of supplied-distance models (FittedDistanceGpr and LoadedDistanceGpr).
/// @details Writes the distance section of config.json and one lower-triangular packed tensor per distance slot
/// next to the usual training data, and restores a model by rebuilding the kernel with freshly issued slots,
/// re-validating the stored tables and either adopting the stored factor or factoring again.

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/optimizer/optimizer.hpp"
#include "cxxgpr/persist/persist.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "checkpoint_io.hpp"
#include "config_json.hpp"
#include "kernel/distance_desc.hpp"
#include "safetensors.hpp"

namespace cxxgpr::persist
{

    namespace
    {

        /// @brief Views a typed span as raw bytes.
        /// @tparam T Element type.
        /// @param[in] s Span to view.
        /// @return Byte view of the same memory.
        template <typename T> [[nodiscard]] std::span<const std::byte> bytes_of(std::span<const T> s) noexcept
        {
            return std::as_bytes(s);
        }

        /// @brief Returns whether every element of a span is finite.
        /// @tparam T Floating-point element type.
        /// @param[in] s Span to scan.
        /// @return True when no element is NaN or infinite.
        template <typename T> [[nodiscard]] bool all_finite(std::span<const T> s) noexcept
        {
            return std::ranges::all_of(s, [](T v) { return std::isfinite(v); });
        }

        /// @brief Reads a tensor of the given storage precision and widens it to double.
        /// @param[in] st Mapped safetensors file.
        /// @param[in] name Tensor name.
        /// @param[in] shape Expected shape.
        /// @param[in] as_double True to read an F64 tensor, false to read an F32 tensor.
        /// @return Values as double, or GprError when the tensor is missing or has another shape or dtype.
        [[nodiscard]] std::expected<std::vector<double>, GprError> read_widened(
            const detail::MappedSafetensors& st,
            std::string_view name,
            std::span<const std::size_t> shape,
            bool as_double
        )
        {
            if (as_double)
            {
                auto sp = st.read_f64(name, shape);
                if (!sp.has_value())
                {
                    return std::unexpected(std::move(sp).error());
                }
                return std::vector<double>(sp->begin(), sp->end());
            }
            auto sp = st.read_f32(name, shape);
            if (!sp.has_value())
            {
                return std::unexpected(std::move(sp).error());
            }
            return std::vector<double>(sp->begin(), sp->end());
        }

        /// @brief Expands a lower-triangular packed table into a full symmetric n x n matrix.
        /// @param[in] packed Packed values, column by column, diagonal included.
        /// @param[in] n Table order.
        /// @param[out] full Destination of n * n values, column-major.
        void unpack_symmetric(std::span<const double> packed, std::size_t n, double* full) noexcept
        {
            std::size_t pos = 0;
            for (std::size_t j = 0; j < n; ++j)
            {
                for (std::size_t i = j; i < n; ++i)
                {
                    const double v = packed[pos++];
                    full[(j * n) + i] = v;
                    full[(i * n) + j] = v;
                }
            }
        }

        /// @brief Converts a prediction to double.
        /// @tparam T Source scalar.
        /// @param[in] p Prediction in the model scalar.
        /// @return Prediction in double.
        template <typename T> [[nodiscard]] Prediction<double> widen(const Prediction<T>& p)
        {
            Prediction<double> out;
            out.mean.assign(p.mean.begin(), p.mean.end());
            out.variance.assign(p.variance.begin(), p.variance.end());
            out.variance_kind = p.variance_kind;
            return out;
        }

        /// @brief Converts a predictive covariance to double.
        /// @tparam T Source scalar.
        /// @param[in] p Covariance in the model scalar.
        /// @return Covariance in double.
        template <typename T> [[nodiscard]] PredictiveCovariance<double> widen(const PredictiveCovariance<T>& p)
        {
            PredictiveCovariance<double> out;
            out.mean.assign(p.mean.begin(), p.mean.end());
            out.covariance.assign(p.covariance.begin(), p.covariance.end());
            out.n_points = p.n_points;
            out.variance_kind = p.variance_kind;
            return out;
        }

        /// @brief Converts a vector of samples to double.
        /// @tparam T Source scalar.
        /// @param[in] v Samples in the model scalar.
        /// @return Samples in double.
        template <typename T> [[nodiscard]] std::vector<double> widen(const std::vector<T>& v)
        {
            return std::vector<double>(v.begin(), v.end());
        }

        /// @brief Widens the value of an expected result to double, keeping errors.
        /// @tparam T Source value type.
        /// @param[in] r Result in the model scalar.
        /// @return Result in double.
        template <typename T>
        [[nodiscard]] auto widen_result(std::expected<T, GprError> r) -> std::expected<decltype(widen(*r)), GprError>
        {
            if (!r.has_value())
            {
                return std::unexpected(std::move(r).error());
            }
            return widen(*r);
        }

    }  // namespace

    namespace detail
    {

        /// @brief Validates and atomically writes the checkpoint of a supplied-distance model.
        /// @details Checks the dimensions and the finiteness of every tensor, serializes config.json with the
        /// distance section, packs the lower triangle of every distance table into a d2.<k> tensor, and writes both
        /// files through staged temporary files.
        /// @param[in] input Borrowed model description.
        /// @param[in] dir Destination directory, created if needed.
        /// @param[in] registry Unused; kept for symmetry with the coordinate model.
        /// @return Empty expected on success, or GprError on validation, serialization, or I/O failure.
        /// @retval GprErrorCode::EmptyInput When there is no training point, or the coordinates do not match the mode.
        /// @retval GprErrorCode::NonFiniteInput When a persisted value is NaN or infinite.
        /// @retval GprErrorCode::PersistFailed When serialization, directory creation or writing fails.
        std::expected<void, GprError> save_distance_model_impl(
            const DistanceSaveInput& input,
            const std::filesystem::path& dir,
            [[maybe_unused]] const PersistRegistry& registry
        )
        {
            const bool with_points = std::holds_alternative<const DistanceKernel<WithPoints>*>(input.kernel);
            if (input.n_train == 0U || with_points != (input.n_features > 0U) || input.y_raw.size() != input.n_train
                || input.x_raw.size() != input.n_train * input.n_features)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (!std::isfinite(input.factor_jitter) || !all_finite(input.x_raw) || !all_finite(input.y_raw)
                || !all_finite(input.l_f64) || !all_finite(input.l_f32) || !all_finite(input.alpha_f64)
                || !all_finite(input.alpha_f32))
            {
                return std::unexpected(GprError::non_finite_input());
            }

            const cxxgpr::detail::DistanceKernelDesc desc = std::visit([](const auto* kernel) {
                return cxxgpr::detail::describe_distance_kernel(*kernel);
            }, input.kernel);
            auto cfg_str = serialize_distance_config_json(input, desc);
            if (!cfg_str.has_value())
            {
                return std::unexpected(std::move(cfg_str).error());
            }

            // Pack the lower triangle of every distance table; the stored scalar selects the tensor dtype.
            std::vector<std::vector<double>> packed_f64;
            std::vector<std::vector<float>> packed_f32;
            if (const auto* const* s64 = std::get_if<const CheckedDistanceStore<double>*>(&input.store))
            {
                packed_f64 = cxxgpr::detail::export_packed_distance_tables(**s64);
            }
            else
            {
                packed_f32 = cxxgpr::detail::export_packed_distance_tables(
                    *std::get<const CheckedDistanceStore<float>*>(input.store)
                );
            }

            const std::size_t n = input.n_train;
            const std::size_t packed_len = n * (n + 1U) / 2U;
            std::vector<std::string> names;
            names.reserve(desc.slots.size());
            for (std::size_t k = 0; k < desc.slots.size(); ++k)
            {
                names.push_back("d2." + std::to_string(k));
            }

            std::vector<TensorWriteView> tensors;
            if (input.n_features > 0U)
            {
                tensors.push_back(
                    TensorWriteView{
                        .name = "x",
                        .dtype = TensorDtype::F64,
                        .shape = {n, input.n_features},
                        .bytes = bytes_of(input.x_raw)
                }
                );
            }
            tensors.push_back(
                TensorWriteView{.name = "y", .dtype = TensorDtype::F64, .shape = {n}, .bytes = bytes_of(input.y_raw)}
            );
            if (input.has_factor)
            {
                const bool l_double = !input.l_f64.empty();
                tensors.push_back(
                    TensorWriteView{
                        .name = "l",
                        .dtype = l_double ? TensorDtype::F64 : TensorDtype::F32,
                        .shape = {n, n},
                        .bytes = l_double ? bytes_of(input.l_f64) : bytes_of(input.l_f32)
                }
                );
                const bool a_double = !input.alpha_f64.empty();
                tensors.push_back(
                    TensorWriteView{
                        .name = "alpha",
                        .dtype = a_double ? TensorDtype::F64 : TensorDtype::F32,
                        .shape = {n},
                        .bytes = a_double ? bytes_of(input.alpha_f64) : bytes_of(input.alpha_f32)
                    }
                );
            }
            for (std::size_t k = 0; k < desc.slots.size(); ++k)
            {
                std::vector<std::size_t> shape{packed_len};
                if (const auto ard_dims = desc.slots[k].ard_dims; ard_dims.has_value())
                {
                    shape = {*ard_dims, packed_len};
                }
                const bool as_double = !packed_f64.empty();
                tensors.push_back(
                    TensorWriteView{
                        .name = names[k],
                        .dtype = as_double ? TensorDtype::F64 : TensorDtype::F32,
                        .shape = std::move(shape),
                        .bytes = as_double ? bytes_of(std::span<const double>(packed_f64[k]))
                                           : bytes_of(std::span<const float>(packed_f32[k]))
                    }
                );
            }

            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Io, "failed to create target persistence directory")
                );
            }
            return write_checkpoint_files(dir, *cfg_str, tensors);
        }

    }  // namespace detail

    /// @brief Restored state of a supplied-distance model: the fitted model of the stored precision.
    /// @tparam C Distance mode.
    template <DistanceModeConcept C> struct LoadedDistanceGpr<C>::Impl
    {
        /// @brief Fitted model of the precision policy found in the checkpoint.
        using Model = std::variant<
            FittedDistanceGpr<optimizer::Lbfgs, DoublePrecision, C>,
            FittedDistanceGpr<optimizer::Lbfgs, SinglePrecision, C>,
            FittedDistanceGpr<optimizer::Lbfgs, MixedPrecision<PromoteStorage>, C>,
            FittedDistanceGpr<optimizer::Lbfgs, MixedPrecision<ReevaluateKernel>, C>>;
        /// @brief Fitted model of the stored precision.
        Model model;
        /// @brief Number of training points.
        std::size_t n{0};
        /// @brief Coordinate dimension.
        std::size_t d{0};
        /// @brief Stored precision policy.
        PersistPrecisionKind kind{PersistPrecisionKind::Double};
        /// @brief Whether the factor and weights came from the checkpoint.
        bool has_factor{false};
        /// @brief Accepted diagonal jitter reported by the checkpoint.
        double factor_jitter{0.0};
    };

    namespace
    {

        /// @brief Builds the fitted model of precision policy P from a parsed checkpoint.
        /// @tparam P Precision policy matching the stored kind.
        /// @tparam C Distance mode.
        /// @param[in,out] cfg Parsed configuration; its transforms are moved into the model.
        /// @param[in] kernel Rebuilt kernel carrying the fitted hyperparameters.
        /// @param[in] source Distance tables unpacked from the checkpoint, bound to the slots of kernel.
        /// @param[in] x_raw Raw training coordinates (empty for a DistanceOnly model).
        /// @param[in] y_raw Raw training targets.
        /// @param[in] st Mapped checkpoint tensors.
        /// @return The fitted model, or GprError when the stored factor or tables are unusable.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<FittedDistanceGpr<optimizer::Lbfgs, P, C>, GprError> build_fitted(
            detail::ParsedConfig& cfg,
            const DistanceKernel<C>& kernel,
            const DistanceSource& source,
            std::span<const double> x_raw,
            std::span<const double> y_raw,
            const detail::MappedSafetensors& st
        )
        {
            using Storage = typename P::Storage;
            using Refine = typename P::Refine;
            constexpr bool with_points = std::same_as<C, WithPoints>;
            const std::size_t n = cfg.n;
            const std::size_t d = cfg.d;
            if (!cfg.likelihood.has_value())
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Config, "missing likelihood configuration")
                );
            }
            const GaussianLikelihood likelihood = *cfg.likelihood;

            std::expected<cxxgpr::detail::DistanceFittedPayload<P, C>, GprError> payload =
                std::unexpected(GprError::empty_input());
            if (cfg.has_factor)
            {
                const std::array<std::size_t, 2> l_shape{n, n};
                const std::array<std::size_t, 1> a_shape{n};
                std::span<const Storage> l_span;
                std::span<const Refine> a_span;
                if constexpr (std::same_as<Storage, double>)
                {
                    auto l = st.read_f64("l", l_shape);
                    if (!l.has_value())
                    {
                        return std::unexpected(std::move(l).error());
                    }
                    l_span = *l;
                }
                else
                {
                    auto l = st.read_f32("l", l_shape);
                    if (!l.has_value())
                    {
                        return std::unexpected(std::move(l).error());
                    }
                    l_span = *l;
                }
                if constexpr (std::same_as<Refine, double>)
                {
                    auto a = st.read_f64("alpha", a_shape);
                    if (!a.has_value())
                    {
                        return std::unexpected(std::move(a).error());
                    }
                    a_span = *a;
                }
                else
                {
                    auto a = st.read_f32("alpha", a_shape);
                    if (!a.has_value())
                    {
                        return std::unexpected(std::move(a).error());
                    }
                    a_span = *a;
                }
                std::shared_ptr<const transform::FittedInputTransform> x_fit;
                if constexpr (with_points)
                {
                    x_fit = std::shared_ptr<const transform::FittedInputTransform>(std::move(cfg.x_transform));
                }
                payload = cxxgpr::detail::assemble_distance_payload<P, C>(
                    kernel, likelihood, cfg.kernel_exp, source, x_raw, d, y_raw, x_fit,
                    std::shared_ptr<const transform::FittedTargetTransform>(std::move(cfg.y_transform)), l_span, a_span,
                    cfg.factor_jitter
                );
            }
            else
            {
                // Without a stored factor, factor again at the stored hyperparameters (no optimization).
                const cxxgpr::detail::DistanceTrainConfig<C> train_cfg{
                    .kernel = kernel,
                    .likelihood = likelihood,
                    .jitter = cfg.jitter,
                    .input_transform = *cfg.x_unfitted,
                    .target_transform = *cfg.y_unfitted,
                    .kernel_exp = cfg.kernel_exp,
                    .optimize = false,
                };
                std::optional<MatrixRef<double>> x;
                if constexpr (with_points)
                {
                    x = MatrixRef<double>(x_raw.data(), n, d, n);
                }
                payload = cxxgpr::detail::train_distance_payload<P, C>(train_cfg, source, x, y_raw, nullptr, nullptr);
            }
            if (!payload.has_value())
            {
                return std::unexpected(std::move(payload).error());
            }
            return detail::DistanceModelAccess::make<optimizer::Lbfgs, P, C>(
                std::make_shared<const cxxgpr::detail::DistanceFittedPayload<P, C>>(std::move(*payload)),
                optimizer::Lbfgs{}, cfg.jitter, std::move(cfg.x_unfitted), std::move(cfg.y_unfitted), cfg.kernel_exp
            );
        }

    }  // namespace

    /// @brief Loads a supplied-distance checkpoint.
    /// @details Parses config.json, rebuilds the kernel with freshly issued slots, validates and unpacks the stored
    /// d2.<k> tables through the normal table validation, and restores the fitted model of the stored precision.
    /// @tparam C Distance mode the caller expects.
    /// @param[in] dir Directory holding config.json and model.safetensors.
    /// @param[in] registry Registry that recreates custom kernels and transforms.
    /// @return The loaded model, or GprError.
    /// @retval GprErrorCode::PersistFailed When a file is unreadable or malformed, a tensor is missing, or the
    /// checkpoint holds another kind of model.
    /// @retval GprErrorCode::UnsupportedPersistVersion When the format version is not supported.
    /// @retval GprErrorCode::InvalidDistance When a stored distance table is non-finite, negative or has a nonzero
    /// diagonal.
    template <DistanceModeConcept C>
    std::expected<LoadedDistanceGpr<C>, GprError> LoadedDistanceGpr<C>::load(
        const std::filesystem::path& dir, const PersistRegistry& registry
    )
    {
        constexpr bool with_points = std::same_as<C, WithPoints>;
        auto cfg = detail::parse_distance_config_json(dir / "config.json", registry);
        if (!cfg.has_value())
        {
            return std::unexpected(std::move(cfg).error());
        }
        if (cfg->distance_with_points != with_points)
        {
            return std::unexpected(
                GprError::persist_failed(
                    PersistErrorKind::WrongModel,
                    with_points ? "checkpoint is a distance-only model; load it with LoadedDistanceGpr<DistanceOnly>"
                                : "checkpoint is a with-points model; load it with LoadedDistanceGpr<WithPoints>"
                )
            );
        }
        auto st = detail::MappedSafetensors::open(dir / "model.safetensors");
        if (!st.has_value())
        {
            return std::unexpected(std::move(st).error());
        }
        const std::size_t n = cfg->n;
        const std::size_t d = cfg->d;
        const std::array<std::size_t, 1> y_shape{n};
        auto y_sp = (*st)->read_f64("y", y_shape);
        if (!y_sp.has_value())
        {
            return std::unexpected(std::move(y_sp).error());
        }
        std::span<const double> x_sp;
        if constexpr (with_points)
        {
            const std::array<std::size_t, 2> x_shape{n, d};
            auto x = (*st)->read_f64("x", x_shape);
            if (!x.has_value())
            {
                return std::unexpected(std::move(x).error());
            }
            x_sp = *x;
        }

        if (!cfg->distance_kernel.has_value())
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Config, "distance kernel is missing"));
        }
        const cxxgpr::detail::DistanceKernelDesc kernel_desc = *cfg->distance_kernel;
        auto kernel = cxxgpr::detail::build_distance_kernel<C>(kernel_desc);
        if (!kernel.has_value())
        {
            return std::unexpected(std::move(kernel).error());
        }

        // Unpack every stored table into full symmetric matrices and bind them to the new slots.
        const bool tables_f64 = cfg->precision_kind == PersistPrecisionKind::Double;
        const std::size_t packed_len = n * (n + 1U) / 2U;
        const auto slots = kernel->slots();
        std::vector<std::vector<double>> matrices;
        std::vector<std::vector<MatrixRef<double>>> ard_refs(slots.size());
        DistanceSource source;
        for (std::size_t k = 0; k < slots.size(); ++k)
        {
            const std::string name = "d2." + std::to_string(k);
            const std::size_t dims = slots[k].ard_dims.value_or(1U);
            std::vector<std::size_t> shape{packed_len};
            if (slots[k].ard_dims.has_value())
            {
                shape = {dims, packed_len};
            }
            auto packed = read_widened(**st, name, shape, tables_f64);
            if (!packed.has_value())
            {
                return std::unexpected(std::move(packed).error());
            }
            const std::size_t first = matrices.size();
            for (std::size_t m = 0; m < dims; ++m)
            {
                std::vector<double> full(n * n);
                unpack_symmetric(
                    std::span<const double>(packed->data() + (m * packed_len), packed_len), n, full.data()
                );
                matrices.push_back(std::move(full));
            }
            if (slots[k].ard_dims.has_value())
            {
                for (std::size_t m = 0; m < dims; ++m)
                {
                    ard_refs[k].emplace_back(matrices[first + m].data(), n, n, n);
                }
                source.bind_ard(slots[k].id, ard_refs[k]);
            }
            else
            {
                source.bind_scalar(slots[k].id, MatrixRef<double>(matrices[first].data(), n, n, n));
            }
        }

        std::optional<typename Impl::Model> model;
        switch (cfg->precision_kind)
        {
            case PersistPrecisionKind::Double:
            {
                auto m = build_fitted<DoublePrecision, C>(*cfg, *kernel, source, x_sp, *y_sp, **st);
                if (!m.has_value())
                {
                    return std::unexpected(std::move(m).error());
                }
                model = typename Impl::Model(std::move(*m));
                break;
            }
            case PersistPrecisionKind::Single:
            {
                auto m = build_fitted<SinglePrecision, C>(*cfg, *kernel, source, x_sp, *y_sp, **st);
                if (!m.has_value())
                {
                    return std::unexpected(std::move(m).error());
                }
                model = typename Impl::Model(std::move(*m));
                break;
            }
            case PersistPrecisionKind::MixedPromote:
            {
                auto m = build_fitted<MixedPrecision<PromoteStorage>, C>(*cfg, *kernel, source, x_sp, *y_sp, **st);
                if (!m.has_value())
                {
                    return std::unexpected(std::move(m).error());
                }
                model = typename Impl::Model(std::move(*m));
                break;
            }
            case PersistPrecisionKind::MixedReevaluate:
            {
                auto m = build_fitted<MixedPrecision<ReevaluateKernel>, C>(*cfg, *kernel, source, x_sp, *y_sp, **st);
                if (!m.has_value())
                {
                    return std::unexpected(std::move(m).error());
                }
                model = typename Impl::Model(std::move(*m));
                break;
            }
        }
        auto impl = std::make_unique<Impl>(Impl{
            .model = std::move(*model),
            .n = n,
            .d = d,
            .kind = cfg->precision_kind,
            .has_factor = cfg->has_factor,
            .factor_jitter = cfg->factor_jitter
        });
        return LoadedDistanceGpr(std::move(impl));
    }

    /// @brief Wraps a restored state.
    /// @tparam C Distance mode.
    /// @param[in] impl Restored state.
    template <DistanceModeConcept C>
    LoadedDistanceGpr<C>::LoadedDistanceGpr(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl))
    {
    }

    /// @brief Copy-constructs a loaded model; the copy owns independent query buffers.
    /// @tparam C Distance mode.
    /// @param[in] other Model to copy.
    template <DistanceModeConcept C>
    LoadedDistanceGpr<C>::LoadedDistanceGpr(const LoadedDistanceGpr& other) :
        impl_(std::make_unique<Impl>(*other.impl_))
    {
    }

    /// @brief Copy-assigns a loaded model; the copy owns independent query buffers.
    /// @tparam C Distance mode.
    /// @param[in] other Model to copy.
    /// @return This model.
    template <DistanceModeConcept C>
    LoadedDistanceGpr<C>& LoadedDistanceGpr<C>::operator=(const LoadedDistanceGpr& other)
    {
        if (this != &other)
        {
            impl_ = std::make_unique<Impl>(*other.impl_);
        }
        return *this;
    }

    /// @brief Move-constructs a loaded model.
    /// @tparam C Distance mode.
    /// @param[in] other Model to move from.
    template <DistanceModeConcept C>
    LoadedDistanceGpr<C>::LoadedDistanceGpr(LoadedDistanceGpr&& other) noexcept = default;

    /// @brief Move-assigns a loaded model.
    /// @tparam C Distance mode.
    /// @param[in] other Model to move from.
    /// @return This model.
    template <DistanceModeConcept C>
    LoadedDistanceGpr<C>& LoadedDistanceGpr<C>::operator=(LoadedDistanceGpr&& other) noexcept = default;

    /// @brief Destroys the loaded model.
    /// @tparam C Distance mode.
    template <DistanceModeConcept C> LoadedDistanceGpr<C>::~LoadedDistanceGpr() = default;

    /// @brief Returns the number of training points.
    /// @tparam C Distance mode.
    /// @return Training point count.
    template <DistanceModeConcept C> std::size_t LoadedDistanceGpr<C>::n() const noexcept
    {
        return impl_->n;
    }

    /// @brief Returns the coordinate dimension.
    /// @tparam C Distance mode.
    /// @return Coordinate dimension (0 for DistanceOnly).
    template <DistanceModeConcept C> std::size_t LoadedDistanceGpr<C>::d() const noexcept
    {
        return impl_->d;
    }

    /// @brief Returns the stored precision policy.
    /// @tparam C Distance mode.
    /// @return Precision kind.
    template <DistanceModeConcept C> PersistPrecisionKind LoadedDistanceGpr<C>::precision_kind() const noexcept
    {
        return impl_->kind;
    }

    /// @brief Returns whether the checkpoint held the factor and weights.
    /// @tparam C Distance mode.
    /// @return True when the factor was stored.
    template <DistanceModeConcept C> bool LoadedDistanceGpr<C>::has_factor() const noexcept
    {
        return impl_->has_factor;
    }

    /// @brief Returns the accepted diagonal jitter of the factor.
    /// @tparam C Distance mode.
    /// @return Jitter offset as reported by the checkpoint.
    template <DistanceModeConcept C> double LoadedDistanceGpr<C>::factor_jitter() const
    {
        return std::visit([](const auto& m) { return m.factor_jitter(); }, impl_->model);
    }

    /// @brief Returns the kernel with its freshly issued slots.
    /// @tparam C Distance mode.
    /// @return Kernel carrying the fitted hyperparameters.
    template <DistanceModeConcept C> const DistanceKernel<C>& LoadedDistanceGpr<C>::kernel() const
    {
        return std::visit([](const auto& m) -> const DistanceKernel<C>& { return m.kernel(); }, impl_->model);
    }

    /// @brief Returns the likelihood with the fitted noise variance.
    /// @tparam C Distance mode.
    /// @return Gaussian likelihood.
    template <DistanceModeConcept C> const GaussianLikelihood& LoadedDistanceGpr<C>::likelihood() const
    {
        return std::visit([](const auto& m) -> const GaussianLikelihood& { return m.likelihood(); }, impl_->model);
    }

    /// @brief Returns the jitter policy.
    /// @tparam C Distance mode.
    /// @return Jitter policy.
    template <DistanceModeConcept C> const JitterPolicy& LoadedDistanceGpr<C>::jitter() const
    {
        return std::visit([](const auto& m) -> const JitterPolicy& {
            return detail::DistanceModelAccess::jitter(m);
        }, impl_->model);
    }

    /// @brief Returns the exponential mode.
    /// @tparam C Distance mode.
    /// @return Kernel exponential mode.
    template <DistanceModeConcept C> KernelExp LoadedDistanceGpr<C>::kernel_exp() const
    {
        return std::visit([](const auto& m) { return detail::DistanceModelAccess::kernel_exp(m); }, impl_->model);
    }

    /// @brief Predicts the mean and variance at the queries of a DistanceOnly model.
    /// @tparam C Distance mode.
    /// @param[in] cross Training-to-query tables bound to the slots of kernel().
    /// @param[in] options Variance selection.
    /// @return Prediction in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<Prediction<double>, GprError> LoadedDistanceGpr<C>::predict(
        const DistanceSource& cross, const PredictOptions& options
    ) const
        requires std::same_as<C, DistanceOnly>
    {
        return std::visit([&](const auto& m) { return widen_result(m.predict(cross, options)); }, impl_->model);
    }

    /// @brief Predicts the mean and variance at the queries of a WithPoints model.
    /// @tparam C Distance mode.
    /// @param[in] cross Training-to-query tables bound to the slots of kernel().
    /// @param[in] xq Query coordinates, m x d.
    /// @param[in] options Variance selection.
    /// @return Prediction in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<Prediction<double>, GprError> LoadedDistanceGpr<C>::predict(
        const DistanceSource& cross, MatrixRef<double> xq, const PredictOptions& options
    ) const
        requires std::same_as<C, WithPoints>
    {
        return std::visit([&](const auto& m) { return widen_result(m.predict(cross, xq, options)); }, impl_->model);
    }

    /// @brief Computes the joint predictive covariance of a DistanceOnly model.
    /// @tparam C Distance mode.
    /// @param[in] cross Training-to-query tables.
    /// @param[in] query Query-to-query tables.
    /// @param[in] options Variance selection.
    /// @return Mean and covariance in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<PredictiveCovariance<double>, GprError> LoadedDistanceGpr<C>::predict_covariance(
        const DistanceSource& cross, const DistanceSource& query, const PredictOptions& options
    ) const
        requires std::same_as<C, DistanceOnly>
    {
        return std::visit([&](const auto& m) {
            return widen_result(m.predict_covariance(cross, query, options));
        }, impl_->model);
    }

    /// @brief Computes the joint predictive covariance of a WithPoints model.
    /// @tparam C Distance mode.
    /// @param[in] cross Training-to-query tables.
    /// @param[in] query Query-to-query tables.
    /// @param[in] xq Query coordinates, m x d.
    /// @param[in] options Variance selection.
    /// @return Mean and covariance in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<PredictiveCovariance<double>, GprError> LoadedDistanceGpr<C>::predict_covariance(
        const DistanceSource& cross, const DistanceSource& query, MatrixRef<double> xq, const PredictOptions& options
    ) const
        requires std::same_as<C, WithPoints>
    {
        return std::visit([&](const auto& m) {
            return widen_result(m.predict_covariance(cross, query, xq, options));
        }, impl_->model);
    }

    /// @brief Draws joint posterior samples of a DistanceOnly model.
    /// @tparam C Distance mode.
    /// @param[in] cross Training-to-query tables.
    /// @param[in] query Query-to-query tables.
    /// @param[in] n_draws Number of samples.
    /// @param[in] seed Seed of the deterministic generator.
    /// @param[in] options Variance selection.
    /// @return Samples in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<std::vector<double>, GprError> LoadedDistanceGpr<C>::sample(
        const DistanceSource& cross,
        const DistanceSource& query,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options
    ) const
        requires std::same_as<C, DistanceOnly>
    {
        return std::visit([&](const auto& m) {
            return widen_result(m.sample(cross, query, n_draws, seed, options));
        }, impl_->model);
    }

    /// @brief Draws joint posterior samples of a WithPoints model.
    /// @tparam C Distance mode.
    /// @param[in] cross Training-to-query tables.
    /// @param[in] query Query-to-query tables.
    /// @param[in] xq Query coordinates, m x d.
    /// @param[in] n_draws Number of samples.
    /// @param[in] seed Seed of the deterministic generator.
    /// @param[in] options Variance selection.
    /// @return Samples in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<std::vector<double>, GprError> LoadedDistanceGpr<C>::sample(
        const DistanceSource& cross,
        const DistanceSource& query,
        MatrixRef<double> xq,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options
    ) const
        requires std::same_as<C, WithPoints>
    {
        return std::visit([&](const auto& m) {
            return widen_result(m.sample(cross, query, xq, n_draws, seed, options));
        }, impl_->model);
    }

    /// @brief Computes leave-one-out predictions at the training points.
    /// @tparam C Distance mode.
    /// @param[in] options Variance selection.
    /// @return Leave-one-out prediction in double, or GprError.
    template <DistanceModeConcept C>
    std::expected<Prediction<double>, GprError> LoadedDistanceGpr<C>::loo_predict(const PredictOptions& options) const
    {
        return std::visit([&](const auto& m) { return widen_result(m.loo_predict(options)); }, impl_->model);
    }

    /// @brief Returns the shared fitted state for precision policy P.
    /// @tparam C Distance mode.
    /// @tparam P Precision policy matching precision_kind().
    /// @return Shared fitted state.
    template <DistanceModeConcept C>
    template <PrecisionPolicyConcept P>
    std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<P, C>> LoadedDistanceGpr<C>::fitted_payload() const
    {
        return detail::DistanceModelAccess::payload(std::get<FittedDistanceGpr<optimizer::Lbfgs, P, C>>(impl_->model));
    }

    /// @brief Returns a copy of the unfitted coordinate transform.
    /// @tparam C Distance mode.
    /// @return Cloned transform.
    template <DistanceModeConcept C>
    std::unique_ptr<transform::UnfittedInputTransform> LoadedDistanceGpr<C>::unfitted_input_clone() const
    {
        return std::visit([](const auto& m) {
            return detail::DistanceModelAccess::unfitted_input(m).clone();
        }, impl_->model);
    }

    /// @brief Returns a copy of the unfitted target transform.
    /// @tparam C Distance mode.
    /// @return Cloned transform.
    template <DistanceModeConcept C>
    std::unique_ptr<transform::UnfittedTargetTransform> LoadedDistanceGpr<C>::unfitted_target_clone() const
    {
        return std::visit([](const auto& m) {
            return detail::DistanceModelAccess::unfitted_target(m).clone();
        }, impl_->model);
    }

    template class LoadedDistanceGpr<DistanceOnly>;
    template class LoadedDistanceGpr<WithPoints>;

    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<DoublePrecision, DistanceOnly>>
        LoadedDistanceGpr<DistanceOnly>::fitted_payload<DoublePrecision>() const;
    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<SinglePrecision, DistanceOnly>>
        LoadedDistanceGpr<DistanceOnly>::fitted_payload<SinglePrecision>() const;
    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>>
        LoadedDistanceGpr<DistanceOnly>::fitted_payload<MixedPrecision<PromoteStorage>>() const;
    template std::shared_ptr<
        const cxxgpr::detail::DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>>
        LoadedDistanceGpr<DistanceOnly>::fitted_payload<MixedPrecision<ReevaluateKernel>>() const;
    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<DoublePrecision, WithPoints>>
        LoadedDistanceGpr<WithPoints>::fitted_payload<DoublePrecision>() const;
    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<SinglePrecision, WithPoints>>
        LoadedDistanceGpr<WithPoints>::fitted_payload<SinglePrecision>() const;
    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>>
        LoadedDistanceGpr<WithPoints>::fitted_payload<MixedPrecision<PromoteStorage>>() const;
    template std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>>
        LoadedDistanceGpr<WithPoints>::fitted_payload<MixedPrecision<ReevaluateKernel>>() const;

}  // namespace cxxgpr::persist
