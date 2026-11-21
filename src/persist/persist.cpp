// SPDX-License-Identifier: Apache-2.0
/// @file persist.cpp
/// @brief Implementation of PersistRegistry, LoadedGpr checkpoint loading/inference, and atomic model saving.
/// @details Implements persist_id validation, factory registry lookups, config.json + model.safetensors
/// loading with zero-copy mmap for F64 Cholesky factors or factor reconstruction when has_factor is false,
/// double-promoted inference methods, typed FittedPayload extraction, and staged atomic persistence writes.

#include "cxxgpr/persist/persist.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/workspace/inference.hpp"
#include "cxxgpr/workspace/mll.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "config_json.hpp"
#include "safetensors.hpp"
#include "workspace/predict_diagonal.hpp"

namespace cxxgpr::persist
{

    /// @brief Validates a custom persist_id against empty and reserved prefix rules.
    /// @details Rejects empty strings and identifiers beginning with RESERVED_PREFIX ("cxxgpr.").
    /// @param[in] persist_id Candidate custom extension identifier.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on violation.
    /// @retval GprErrorCode::PersistFailed When persist_id is empty or starts with "cxxgpr.".
    std::expected<void, GprError> validate_persist_id(std::string_view persist_id)
    {
        // Reject empty identifiers and reserved library prefixes before registration or serialization.
        if (persist_id.empty())
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::InvalidPersistId, "persist_id must not be empty")
            );
        }
        if (persist_id.starts_with(RESERVED_PREFIX))
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::InvalidPersistId, "persist_id must not start with 'cxxgpr.'")
            );
        }
        return {};
    }

    namespace
    {

        /// @brief Validates persist_id and inserts factory into map if not already registered.
        /// @tparam MapT Associative container mapping std::string to factory function.
        /// @tparam FnT Factory callable type matching MapT::mapped_type.
        /// @param[in,out] map Registry map receiving the new entry.
        /// @param[in] persist_id Candidate unique extension identifier.
        /// @param[in] factory Factory callable stored under persist_id.
        /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on error or duplicate key.
        template <typename MapT, typename FnT>
        [[nodiscard]] std::expected<void, GprError> register_impl(MapT& map, std::string persist_id, FnT factory)
        {
            if (auto v = validate_persist_id(persist_id); !v.has_value())
            {
                return v;
            }
            if (map.contains(persist_id))
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::InvalidPersistId, "duplicate persist_id registration")
                );
            }
            map.emplace(std::move(persist_id), std::move(factory));
            return {};
        }

        /// @brief Looks up persist_id in map and invokes the registered factory with state_json.
        /// @tparam MapT Associative container mapping std::string to factory function.
        /// @param[in] map Registry map queried for persist_id.
        /// @param[in] persist_id Extension identifier read from config.json.
        /// @param[in] state_json Serialized JSON state passed to the factory.
        /// @return Factory return expected, or PersistFailed(UnregisteredId) if missing.
        template <typename MapT>
        [[nodiscard]] auto create_impl(const MapT& map, std::string_view persist_id, std::string_view state_json)
            -> decltype(map.begin()->second(state_json))
        {
            const auto it = map.find(std::string(persist_id));
            if (it == map.end())
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::UnregisteredId, "unregistered persist_id in config.json")
                );
            }
            return it->second(state_json);
        }

        /// @brief Reinterprets a contiguous typed read-only span as raw byte view for Safetensors output.
        /// @tparam T Element scalar type.
        /// @param[in] s Typed read-only span.
        /// @return Non-owning byte span covering s.
        template <typename T> [[nodiscard]] std::span<const std::byte> as_bytes(std::span<const T> s) noexcept
        {
            return std::as_bytes(s);
        }

        /// @brief Precision-typed runtime inference state held inside LoadedGpr::Impl.
        /// @details Stores compiled kernel, transformed training tensors, Cholesky factor L view (zero-copy
        /// mmap when F64 and has_factor, owned otherwise), weights alpha, and reusable query workspace.
        /// @tparam P Precision policy defining Storage and Refine scalar types.
        template <typename P> struct LoadedTypedState
        {
            /// @brief Storage scalar type for training coordinates and Cholesky factor L.
            using Storage = P::Storage;
            /// @brief Refinement and output scalar type for transformed targets and weights alpha.
            using Refine = P::Refine;

            /// @brief Compiled kernel evaluator configured with restored kernel_spec and kernel_exp.
            std::shared_ptr<const CompiledKernel<Storage>> compiled_kernel;
            /// @brief Transformed column-major training features cast to Storage precision (n * d).
            std::vector<Storage> x_train;
            /// @brief Transformed training targets cast to Refine precision (n).
            std::vector<Refine> y_transformed;
            /// @brief Owned Cholesky factor L storage (empty when zero-copy mmap backs l_view).
            std::vector<Storage> l_owned;
            /// @brief Active read-only view of L (pointing to mmap for F64 has_factor, or l_owned otherwise).
            std::span<const Storage> l_view;
            /// @brief Owned copy of refined prediction weights alpha (n).
            std::vector<Refine> alpha_owned;
            /// @brief Active read-only span over alpha_owned (n).
            std::span<const Refine> alpha_view;
            /// @brief Negative log marginal likelihood computed from loaded or reconstructed factor.
            double neg_log_marginal_likelihood{0.0};
            /// @brief Mutable reusable query scratch buffers avoiding allocations on repeated predict_into().
            mutable QueryWorkspace<P> query_ws;
            /// @brief Temporary typed prediction buffer used to promote SinglePrecision outputs to double.
            mutable Prediction<Refine> tmp_pred;
        };

        /// @brief Constructs LoadedTypedState<P> from parsed config and mapped safetensors.
        /// @details Applies restored transforms to raw (x, y), checks finiteness, and either binds zero-copy
        /// mmap F64 / owned F32 factors when has_factor is true or reconstructs L and alpha via factor_and_mll()
        /// when has_factor is false.
        /// @tparam P Target precision policy matching cfg.precision_kind.
        /// @param[in] cfg Parsed checkpoint metadata and cloned transforms.
        /// @param[in] st Validated MappedSafetensors handle keeping mapped memory alive.
        /// @param[in] x_raw Raw column-major training features in double precision (n * d).
        /// @param[in] y_raw Raw training targets in double precision (n).
        /// @param[out] out_is_mmap Output flag set true iff L is backed directly by zero-copy mmap.
        /// @return Constructed LoadedTypedState<P> on success, or GprError on transform, tensor, or factor failure.
        /// @retval GprErrorCode::PersistFailed When transforms fail, tensor validation fails, or L_ii <= 0.
        /// @retval GprErrorCode::CholeskyFailed When factor reconstruction fails when has_factor is false.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<LoadedTypedState<P>, GprError> build_typed_state(
            const detail::ParsedConfig& cfg,
            const std::shared_ptr<const detail::MappedSafetensors>& st,
            std::span<const double> x_raw,
            std::span<const double> y_raw,
            bool& out_is_mmap
        )
        {
            using Storage = P::Storage;
            using Refine = P::Refine;

            const std::size_t n = cfg.n;
            const std::size_t d = cfg.d;
            if (!cfg.kernel_spec.has_value() || !cfg.likelihood.has_value())
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Config, "missing kernel or likelihood configuration")
                );
            }

            // Re-apply fitted input and target transforms to raw double tensors and verify finiteness.
            std::vector<double> x_trans(x_raw.begin(), x_raw.end());
            if (auto ap = cfg.x_transform->apply(x_trans, n, d); !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }
            std::vector<double> y_trans(y_raw.begin(), y_raw.end());
            if (auto tf = cfg.y_transform->transform(y_trans); !tf.has_value())
            {
                return std::unexpected(std::move(tf).error());
            }
            for (const double v : x_trans)
            {
                if (!std::isfinite(v))
                {
                    return std::unexpected(
                        GprError::persist_failed(
                            PersistErrorKind::Tensor, "non-finite value after applying x_transform"
                        )
                    );
                }
            }
            for (const double v : y_trans)
            {
                if (!std::isfinite(v))
                {
                    return std::unexpected(
                        GprError::persist_failed(
                            PersistErrorKind::Tensor, "non-finite value after applying y_transform"
                        )
                    );
                }
            }

            LoadedTypedState<P> state;
            {
                auto ck =
                    std::make_shared<CompiledKernel<Storage>>(CompiledKernel<Storage>::from_spec(*cfg.kernel_spec));
                ck->set_kernel_exp(cfg.kernel_exp);
                state.compiled_kernel = std::move(ck);
            }

            state.x_train.resize(x_trans.size());
            for (std::size_t i = 0; i < x_trans.size(); ++i)
            {
                state.x_train[i] = static_cast<Storage>(x_trans[i]);
            }
            state.y_transformed.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                state.y_transformed[i] = static_cast<Refine>(y_trans[i]);
            }

            if (cfg.has_factor)
            {
                // Bind persisted factor L directly via zero-copy mmap for F64 Storage, or copy F32 Storage.
                const std::array<std::size_t, 2> l_shape{n, n};
                const std::array<std::size_t, 1> a_shape{n};
                if constexpr (std::same_as<Storage, double>)
                {
                    auto l_sp = st->read_f64("l", l_shape);
                    if (!l_sp.has_value())
                    {
                        return std::unexpected(std::move(l_sp).error());
                    }
                    state.l_view = *l_sp;
                    out_is_mmap = true;
                }
                else
                {
                    auto l_sp = st->read_f32("l", l_shape);
                    if (!l_sp.has_value())
                    {
                        return std::unexpected(std::move(l_sp).error());
                    }
                    state.l_owned.assign(l_sp->begin(), l_sp->end());
                    state.l_view = state.l_owned;
                    out_is_mmap = false;
                }
                if constexpr (std::same_as<Refine, double>)
                {
                    auto a_sp = st->read_f64("alpha", a_shape);
                    if (!a_sp.has_value())
                    {
                        return std::unexpected(std::move(a_sp).error());
                    }
                    state.alpha_owned.assign(a_sp->begin(), a_sp->end());
                    state.alpha_view = state.alpha_owned;
                }
                else
                {
                    auto a_sp = st->read_f32("alpha", a_shape);
                    if (!a_sp.has_value())
                    {
                        return std::unexpected(std::move(a_sp).error());
                    }
                    state.alpha_owned.assign(a_sp->begin(), a_sp->end());
                    state.alpha_view = state.alpha_owned;
                }
                // Validate positive Cholesky diagonal entries and compute MLL for extracted payloads.
                double quad = 0.0;
                double log_det = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    quad += static_cast<double>(state.y_transformed[i]) * static_cast<double>(state.alpha_view[i]);
                    const auto l_ii = static_cast<double>(state.l_view[(i * n) + i]);
                    if (!(l_ii > 0.0) || !std::isfinite(l_ii))
                    {
                        return std::unexpected(
                            GprError::persist_failed(
                                PersistErrorKind::Tensor, "loaded Cholesky factor has non-positive diagonal"
                            )
                        );
                    }
                    log_det += std::log(l_ii);
                }
                const double log_2pi = std::log(2.0 * std::numbers::pi);
                state.neg_log_marginal_likelihood = (0.5 * quad) + log_det + (0.5 * static_cast<double>(n) * log_2pi);
                return state;
            }

            // Reconstruct L and alpha from raw data and saved configuration (has_factor == false).
            out_is_mmap = false;
            auto buffers = FitBuffers<P>::create(
                n, FitPolicy{.distance_cache = cfg.distance_cache, .cholesky_buffer = CholeskyBuffer::Retain}
            );
            if (!buffers.has_value())
            {
                return std::unexpected(std::move(buffers).error());
            }
            std::vector<Storage> y_storage(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                y_storage[i] = static_cast<Storage>(y_trans[i]);
            }
            const MatrixRef<Storage> x_ref(state.x_train.data(), n, d, n);
            // Replay exact recorded factor_jitter when positive so reconstructed L matches original save state.
            auto jitter_to_use = cfg.jitter;
            if (cfg.factor_jitter > 0.0)
            {
                auto fj = JitterPolicy::fixed(cfg.factor_jitter);
                if (fj.has_value())
                {
                    jitter_to_use = *fj;
                }
            }
            auto mll =
                factor_and_mll(*state.compiled_kernel, *cfg.likelihood, x_ref, y_storage, jitter_to_use, *buffers, 0U);
            if (!mll.has_value())
            {
                return std::unexpected(std::move(mll).error());
            }
            state.neg_log_marginal_likelihood = *mll;
            state.l_owned.resize(n * n);
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = 0; row < n; ++row)
                {
                    state.l_owned[(col * n) + row] = buffers->core.k_ref()(row, col);
                }
            }
            state.l_view = state.l_owned;

            state.alpha_owned.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                state.alpha_owned[i] = static_cast<Refine>(buffers->core.rhs[i]);
            }
            state.alpha_view = state.alpha_owned;
            return state;
        }

        /// @brief Validates, stages, transforms, and casts a query matrix into a loaded state's query workspace.
        /// @details Grows st.query_ws when needed, copies xs column-major into query_xs while checking
        /// finiteness, applies in_tf to exactly the first m * d staged values, and casts them to Storage in
        /// query_x. The workspace only grows, so every access is bounded by m * d rather than by the vector
        /// sizes; a smaller batch may therefore follow a larger one on the same loaded model.
        /// @tparam P Precision policy active in st.
        /// @param[in] st Loaded typed precision state owning the mutable query workspace.
        /// @param[in] in_tf Restored fitted input feature transform.
        /// @param[in] n Training sample count n.
        /// @param[in] d Training feature dimension d.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @return Column-major MatrixRef<Storage> (m x d) backed by st.query_ws.query_x, or GprError.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != d.
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<MatrixRef<typename P::Storage>, GprError> stage_loaded_query(
            const LoadedTypedState<P>& st,
            const transform::FittedInputTransform& in_tf,
            std::size_t n,
            std::size_t d,
            MatrixRef<double> xs
        )
        {
            using Storage = P::Storage;

            if (xs.is_empty() || xs.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            const std::size_t m = xs.rows();
            if (xs.cols() != d)
            {
                return std::unexpected(GprError::dimension_mismatch(xs.cols(), d));
            }
            // ensure() rejects an overflowing m * d before any buffer is indexed below.
            if (auto ens = st.query_ws.ensure(n, m, d, st.compiled_kernel->nested_depth()); !ens.has_value())
            {
                return std::unexpected(std::move(ens).error());
            }
            // Copy query coordinates column-major while checking finiteness before applying input transform.
            for (std::size_t col = 0; col < d; ++col)
            {
                for (std::size_t row = 0; row < m; ++row)
                {
                    const double val = xs(row, col);
                    if (!std::isfinite(val))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                    st.query_ws.query_xs[(col * m) + row] = val;
                }
            }
            // The transform checks its span length against m * d, so pass exactly the staged prefix.
            const std::size_t md = m * d;
            const std::span<double> staged(st.query_ws.query_xs.data(), md);
            if (auto ap = in_tf.apply(staged, m, d); !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }
            for (std::size_t i = 0; i < md; ++i)
            {
                st.query_ws.query_x[i] = static_cast<Storage>(st.query_ws.query_xs[i]);
            }
            return MatrixRef<Storage>(st.query_ws.query_x.data(), m, d, m);
        }

        /// @brief Evaluates double-promoted posterior mean and variance on a loaded precision state.
        /// @details Validates xs, applies in_tf, casts to Storage, invokes detail::predict_diagonal_into<P>() with
        /// FactorValidation::AlreadyValidated (the factor was checked once by load), promotes float outputs to double
        /// when Refine is float, and applies tgt_tf inverse mean and variance transforms.
        /// @tparam P Precision policy active in st.
        /// @param[in] st Loaded typed precision state owning compiled kernel, factor view, and query workspace.
        /// @param[in] lik Restored Gaussian observation noise likelihood.
        /// @param[in] in_tf Restored fitted input feature transform.
        /// @param[in] tgt_tf Restored fitted target transform.
        /// @param[in] n Training sample count n.
        /// @param[in] d Training feature dimension d.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] options Prediction variance controls.
        /// @param[out] out Destination double-precision Prediction struct populated on the original target scale.
        /// @return Empty expected on success, or GprError on query validation, prediction, or transform failure.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> loaded_predict_typed(
            const LoadedTypedState<P>& st,
            const GaussianLikelihood& lik,
            const transform::FittedInputTransform& in_tf,
            const transform::FittedTargetTransform& tgt_tf,
            std::size_t n,
            std::size_t d,
            MatrixRef<double> xs,
            const PredictOptions& options,
            Prediction<double>& out
        )
        {
            using Storage = P::Storage;
            using Refine = P::Refine;

            auto xq = stage_loaded_query(st, in_tf, n, d, xs);
            if (!xq.has_value())
            {
                return std::unexpected(std::move(xq).error());
            }
            const MatrixRef<Storage> xq_ref = *xq;
            const std::size_t m = xq_ref.rows();
            const MatrixRef<Storage> x_train_ref(st.x_train.data(), n, d, n);
            const MatrixRef<Storage> l_ref(st.l_view.data(), n, n, n);

            // Predict directly into out for double Refine, or promote via tmp_pred for SinglePrecision.
            if constexpr (std::same_as<Refine, double>)
            {
                // The factor was validated when the model was loaded (finite, positive diagonal), so skip the
                // per-call rescan of the whole lower triangle.
                if (auto pr = cxxgpr::detail::predict_diagonal_into<P>(
                        *st.compiled_kernel, lik, x_train_ref, l_ref, st.alpha_view, xq_ref, options, st.query_ws, out,
                        linalg::detail::FactorValidation::AlreadyValidated
                    );
                    !pr.has_value())
                {
                    return pr;
                }
            }
            else
            {
                if (auto pr = cxxgpr::detail::predict_diagonal_into<P>(
                        *st.compiled_kernel, lik, x_train_ref, l_ref, st.alpha_view, xq_ref, options, st.query_ws,
                        st.tmp_pred, linalg::detail::FactorValidation::AlreadyValidated
                    );
                    !pr.has_value())
                {
                    return pr;
                }
                out.mean.resize(m);
                out.variance.resize(m);
                for (std::size_t i = 0; i < m; ++i)
                {
                    out.mean[i] = static_cast<double>(st.tmp_pred.mean[i]);
                    out.variance[i] = static_cast<double>(st.tmp_pred.variance[i]);
                }
                out.variance_kind = st.tmp_pred.variance_kind;
            }
            if (auto im = tgt_tf.inverse_transform_mean(out.mean); !im.has_value())
            {
                return im;
            }
            return tgt_tf.inverse_transform_variance(out.variance);
        }

    }  // namespace

    /// @brief Registers a custom kernel factory under persist_id.
    /// @param[in] persist_id Unique extension identifier outside the reserved prefix.
    /// @param[in] factory Deserialization callback constructing CustomKernel from JSON state.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId).
    std::expected<void, GprError> PersistRegistry::register_kernel(std::string persist_id, CustomKernelFactory factory)
    {
        return register_impl(kernels_, std::move(persist_id), std::move(factory));
    }

    /// @brief Registers a custom unfitted input transform factory under persist_id.
    /// @param[in] persist_id Unique extension identifier outside the reserved prefix.
    /// @param[in] factory Deserialization callback constructing UnfittedInputTransform from JSON state.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId).
    std::expected<void, GprError> PersistRegistry::register_unfitted_input(
        std::string persist_id, UnfittedInputFactory factory
    )
    {
        return register_impl(unfitted_inputs_, std::move(persist_id), std::move(factory));
    }

    /// @brief Registers a custom fitted input transform factory under persist_id.
    /// @param[in] persist_id Unique extension identifier outside the reserved prefix.
    /// @param[in] factory Deserialization callback constructing FittedInputTransform from JSON state.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId).
    std::expected<void, GprError> PersistRegistry::register_fitted_input(
        std::string persist_id, FittedInputFactory factory
    )
    {
        return register_impl(fitted_inputs_, std::move(persist_id), std::move(factory));
    }

    /// @brief Registers a custom unfitted target transform factory under persist_id.
    /// @param[in] persist_id Unique extension identifier outside the reserved prefix.
    /// @param[in] factory Deserialization callback constructing UnfittedTargetTransform from JSON state.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId).
    std::expected<void, GprError> PersistRegistry::register_unfitted_target(
        std::string persist_id, UnfittedTargetFactory factory
    )
    {
        return register_impl(unfitted_targets_, std::move(persist_id), std::move(factory));
    }

    /// @brief Registers a custom fitted target transform factory under persist_id.
    /// @param[in] persist_id Unique extension identifier outside the reserved prefix.
    /// @param[in] factory Deserialization callback constructing FittedTargetTransform from JSON state.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId).
    std::expected<void, GprError> PersistRegistry::register_fitted_target(
        std::string persist_id, FittedTargetFactory factory
    )
    {
        return register_impl(fitted_targets_, std::move(persist_id), std::move(factory));
    }

    /// @brief Looks up and invokes a registered custom kernel factory by persist_id.
    /// @param[in] persist_id Custom kernel identifier from config.json.
    /// @param[in] state_json Serialized JSON state string passed to the factory.
    /// @return Constructed CustomKernel or PersistFailed(UnregisteredId) / factory error.
    std::expected<CustomKernel, GprError> PersistRegistry::create_kernel(
        std::string_view persist_id, std::string_view state_json
    ) const
    {
        return create_impl(kernels_, persist_id, state_json);
    }

    /// @brief Looks up and invokes a registered custom unfitted input transform factory by persist_id.
    /// @param[in] persist_id Custom transform identifier from config.json.
    /// @param[in] state_json Serialized JSON state string passed to the factory.
    /// @return Constructed UnfittedInputTransform or PersistFailed(UnregisteredId) / factory error.
    std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError> PersistRegistry::create_unfitted_input(
        std::string_view persist_id, std::string_view state_json
    ) const
    {
        return create_impl(unfitted_inputs_, persist_id, state_json);
    }

    /// @brief Looks up and invokes a registered custom fitted input transform factory by persist_id.
    /// @param[in] persist_id Custom transform identifier from config.json.
    /// @param[in] state_json Serialized JSON state string passed to the factory.
    /// @return Constructed FittedInputTransform or PersistFailed(UnregisteredId) / factory error.
    std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError> PersistRegistry::create_fitted_input(
        std::string_view persist_id, std::string_view state_json
    ) const
    {
        return create_impl(fitted_inputs_, persist_id, state_json);
    }

    /// @brief Looks up and invokes a registered custom unfitted target transform factory by persist_id.
    /// @param[in] persist_id Custom transform identifier from config.json.
    /// @param[in] state_json Serialized JSON state string passed to the factory.
    /// @return Constructed UnfittedTargetTransform or PersistFailed(UnregisteredId) / factory error.
    std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError> PersistRegistry::
        create_unfitted_target(std::string_view persist_id, std::string_view state_json) const
    {
        return create_impl(unfitted_targets_, persist_id, state_json);
    }

    /// @brief Looks up and invokes a registered custom fitted target transform factory by persist_id.
    /// @param[in] persist_id Custom transform identifier from config.json.
    /// @param[in] state_json Serialized JSON state string passed to the factory.
    /// @return Constructed FittedTargetTransform or PersistFailed(UnregisteredId) / factory error.
    std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError> PersistRegistry::create_fitted_target(
        std::string_view persist_id, std::string_view state_json
    ) const
    {
        return create_impl(fitted_targets_, persist_id, state_json);
    }

    /// @brief Immutable checkpoint implementation shared across LoadedGpr copies.
    /// @details Owns restored configuration, transforms, mapped Safetensors handle, raw training data,
    /// and the single populated precision-typed state optional matching precision_kind.
    struct LoadedGpr::Impl
    {
        /// @brief Number of training samples n.
        std::size_t n{0};
        /// @brief Number of training feature dimensions d.
        std::size_t d{0};
        /// @brief True if checkpoint included serialized l and alpha tensors.
        bool has_factor{false};
        /// @brief True if the active F64 Cholesky factor is directly backed by read-only mmap.
        bool has_mmap_factor{false};
        /// @brief Accepted diagonal jitter offset recorded during factorization.
        double factor_jitter{0.0};
        /// @brief Active precision policy tag restored from config.json.
        PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
        /// @brief Restored exponential evaluation backend.
        KernelExp kernel_exp{KernelExp::Accurate};
        /// @brief Restored training distance cache policy.
        DistanceCachePolicy distance_cache{DistanceCachePolicy::Cached};
        /// @brief Restored kernel specification and hyperparameters.
        KernelSpec kernel_spec;
        /// @brief Restored Gaussian observation noise likelihood.
        GaussianLikelihood likelihood;
        /// @brief Restored Cholesky jitter policy.
        JitterPolicy jitter;
        /// @brief Restored unfitted input transform prototype for extract_payload().
        std::unique_ptr<transform::UnfittedInputTransform> x_unfitted;
        /// @brief Restored fitted input transform for query preprocessing.
        std::unique_ptr<transform::FittedInputTransform> x_transform;
        /// @brief Restored unfitted target transform prototype for extract_payload().
        std::unique_ptr<transform::UnfittedTargetTransform> y_unfitted;
        /// @brief Restored fitted target transform for inverse prediction scaling.
        std::unique_ptr<transform::FittedTargetTransform> y_transform;
        /// @brief Shared memory-mapped Safetensors archive keeping F64 mmap spans valid.
        std::shared_ptr<const detail::MappedSafetensors> safetensors;
        /// @brief Owned raw column-major double training features (n * d).
        std::vector<double> x_raw;
        /// @brief Owned raw double training targets (n).
        std::vector<double> y_raw;
        /// @brief Active typed state when precision_kind == Double.
        std::optional<LoadedTypedState<DoublePrecision>> double_state;
        /// @brief Active typed state when precision_kind == Single.
        std::optional<LoadedTypedState<SinglePrecision>> single_state;
        /// @brief Active typed state when precision_kind == MixedPromote.
        std::optional<LoadedTypedState<MixedPrecision<PromoteStorage>>> mixed_promote_state;
        /// @brief Active typed state when precision_kind == MixedReevaluate.
        std::optional<LoadedTypedState<MixedPrecision<ReevaluateKernel>>> mixed_reeval_state;
    };

    /// @brief Destroys LoadedGpr out-of-line where LoadedGpr::Impl is complete.
    LoadedGpr::~LoadedGpr() = default;

    namespace
    {

        /// @brief Visits the active LoadedTypedState<P> inside impl matching impl.precision_kind.
        /// @tparam ImplT LoadedGpr::Impl type.
        /// @tparam Fn Polymorphic callable accepting const LoadedTypedState<P>&.
        /// @param[in] impl Immutable LoadedGpr implementation instance.
        /// @param[in] fn Generic visitor invoked with the active typed state optional.
        /// @return Result returned by fn, or PersistFailed(Config) if the expected typed state is unset.
        template <typename ImplT, typename Fn> [[nodiscard]] auto with_typed_state(const ImplT& impl, Fn&& fn)
        {
            using Ret = decltype(fn(*impl.double_state));
            auto call_opt = [&](const auto& opt) -> Ret
            {
                if (!opt.has_value())
                {
                    return std::unexpected(
                        GprError::persist_failed(PersistErrorKind::Config, "missing typed precision state")
                    );
                }
                return fn(*opt);
            };
            // Dispatch dynamically on precision_kind to the single populated typed state optional.
            switch (impl.precision_kind)
            {
                case PersistPrecisionKind::Double:
                    return call_opt(impl.double_state);
                case PersistPrecisionKind::Single:
                    return call_opt(impl.single_state);
                case PersistPrecisionKind::MixedPromote:
                    return call_opt(impl.mixed_promote_state);
                case PersistPrecisionKind::MixedReevaluate:
                    return call_opt(impl.mixed_reeval_state);
            }
            return Ret(
                std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Config, "invalid precision state in LoadedGpr")
                )
            );
        }

    }  // namespace

    /// @brief Loads and validates a persisted Exact GPR directory (config.json and model.safetensors).
    /// @details Parses config.json, memory-maps model.safetensors, verifies raw x and y F64 tensors,
    /// and initializes the active LoadedTypedState (via zero-copy mmap for F64 factors or Cholesky
    /// reconstruction when has_factor is false).
    /// @param[in] dir Directory containing config.json and model.safetensors.
    /// @param[in] registry Extension registry providing factories for custom kernels and transforms.
    /// @return LoadedGpr handle on success, or GprError on I/O, schema, version, tensor, or factor failure.
    /// @retval GprErrorCode::PersistFailed When config, safetensors, or registry resolution fails.
    /// @retval GprErrorCode::UnsupportedPersistVersion When artifact format_version != FORMAT_VERSION.
    /// @retval GprErrorCode::CholeskyFailed When factor reconstruction fails when has_factor is false.
    std::expected<LoadedGpr, GprError> LoadedGpr::load(
        const std::filesystem::path& dir, const PersistRegistry& registry
    )
    {
        // Load and validate config.json schema before memory-mapping model.safetensors.
        auto cfg = detail::parse_config_json(dir / "config.json", registry);
        if (!cfg.has_value())
        {
            return std::unexpected(std::move(cfg).error());
        }
        auto st = detail::MappedSafetensors::open(dir / "model.safetensors");
        if (!st.has_value())
        {
            return std::unexpected(std::move(st).error());
        }

        // Validate mandatory raw F64 training tensors x (n x d) and y (n).
        const std::array<std::size_t, 2> x_shape{cfg->n, cfg->d};
        const std::array<std::size_t, 1> y_shape{cfg->n};
        auto x_sp = (*st)->read_f64("x", x_shape);
        if (!x_sp.has_value())
        {
            return std::unexpected(std::move(x_sp).error());
        }
        auto y_sp = (*st)->read_f64("y", y_shape);
        if (!y_sp.has_value())
        {
            return std::unexpected(std::move(y_sp).error());
        }

        if (!cfg->kernel_spec.has_value() || !cfg->likelihood.has_value())
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Config, "missing kernel or likelihood configuration")
            );
        }

        auto impl = std::make_shared<Impl>(Impl{
            .n = cfg->n,
            .d = cfg->d,
            .has_factor = cfg->has_factor,
            .has_mmap_factor = false,
            .factor_jitter = cfg->factor_jitter,
            .precision_kind = cfg->precision_kind,
            .kernel_exp = cfg->kernel_exp,
            .distance_cache = cfg->distance_cache,
            .kernel_spec = std::move(*cfg->kernel_spec),
            .likelihood = *cfg->likelihood,
            .jitter = cfg->jitter,
            .x_unfitted = std::move(cfg->x_unfitted),
            .x_transform = std::move(cfg->x_transform),
            .y_unfitted = std::move(cfg->y_unfitted),
            .y_transform = std::move(cfg->y_transform),
            .safetensors = *st,
            .x_raw = std::vector<double>(x_sp->begin(), x_sp->end()),
            .y_raw = std::vector<double>(y_sp->begin(), y_sp->end()),
            .double_state = std::nullopt,
            .single_state = std::nullopt,
            .mixed_promote_state = std::nullopt,
            .mixed_reeval_state = std::nullopt,
        });

        detail::ParsedConfig cfg_view{
            .n = impl->n,
            .d = impl->d,
            .has_factor = impl->has_factor,
            .factor_jitter = impl->factor_jitter,
            .precision_kind = impl->precision_kind,
            .kernel_exp = impl->kernel_exp,
            .distance_cache = impl->distance_cache,
            .kernel_spec = impl->kernel_spec,
            .likelihood = impl->likelihood,
            .jitter = impl->jitter,
            .x_unfitted = impl->x_unfitted->clone(),
            .x_transform = impl->x_transform->clone(),
            .y_unfitted = impl->y_unfitted->clone(),
            .y_transform = impl->y_transform->clone(),
        };

        // Construct only the precision-typed state matching precision_kind.
        switch (impl->precision_kind)
        {
            case PersistPrecisionKind::Double:
            {
                auto s =
                    build_typed_state<DoublePrecision>(cfg_view, *st, impl->x_raw, impl->y_raw, impl->has_mmap_factor);
                if (!s.has_value())
                {
                    return std::unexpected(std::move(s).error());
                }
                impl->double_state = std::move(*s);
                break;
            }
            case PersistPrecisionKind::Single:
            {
                auto s =
                    build_typed_state<SinglePrecision>(cfg_view, *st, impl->x_raw, impl->y_raw, impl->has_mmap_factor);
                if (!s.has_value())
                {
                    return std::unexpected(std::move(s).error());
                }
                impl->single_state = std::move(*s);
                break;
            }
            case PersistPrecisionKind::MixedPromote:
            {
                auto s = build_typed_state<MixedPrecision<PromoteStorage>>(
                    cfg_view, *st, impl->x_raw, impl->y_raw, impl->has_mmap_factor
                );
                if (!s.has_value())
                {
                    return std::unexpected(std::move(s).error());
                }
                impl->mixed_promote_state = std::move(*s);
                break;
            }
            case PersistPrecisionKind::MixedReevaluate:
            {
                auto s = build_typed_state<MixedPrecision<ReevaluateKernel>>(
                    cfg_view, *st, impl->x_raw, impl->y_raw, impl->has_mmap_factor
                );
                if (!s.has_value())
                {
                    return std::unexpected(std::move(s).error());
                }
                impl->mixed_reeval_state = std::move(*s);
                break;
            }
        }

        return LoadedGpr(std::move(impl));
    }

    /// @brief Returns the number of training samples n.
    /// @return Training sample count n.
    std::size_t LoadedGpr::n() const noexcept
    {
        return impl_->n;
    }

    /// @brief Returns the number of training features d.
    /// @return Training feature dimension d.
    std::size_t LoadedGpr::d() const noexcept
    {
        return impl_->d;
    }

    /// @brief Returns the persisted precision policy tag.
    /// @return Active PersistPrecisionKind.
    PersistPrecisionKind LoadedGpr::precision_kind() const noexcept
    {
        return impl_->precision_kind;
    }

    /// @brief Returns true if checkpoint tensors l and alpha were loaded from disk.
    /// @return True when config.json recorded has_factor == true.
    bool LoadedGpr::has_factor() const noexcept
    {
        return impl_->has_factor;
    }

    /// @brief Returns true if the Cholesky factor is backed directly by a read-only mmap view.
    /// @return True for DoublePrecision checkpoints loaded with has_factor == true.
    bool LoadedGpr::has_mmap_factor() const noexcept
    {
        return impl_->has_mmap_factor;
    }

    /// @brief Returns the accepted Cholesky diagonal jitter offset.
    /// @return Recorded diagonal jitter scalar.
    double LoadedGpr::factor_jitter() const noexcept
    {
        return impl_->factor_jitter;
    }

    /// @brief Returns the restored kernel specification.
    /// @return Constant reference to KernelSpec.
    const KernelSpec& LoadedGpr::kernel_spec() const noexcept
    {
        return impl_->kernel_spec;
    }

    /// @brief Returns the restored Gaussian observation noise likelihood.
    /// @return Constant reference to GaussianLikelihood.
    const GaussianLikelihood& LoadedGpr::likelihood() const noexcept
    {
        return impl_->likelihood;
    }

    /// @brief Returns the restored Cholesky stabilization jitter policy.
    /// @return Constant reference to JitterPolicy.
    const JitterPolicy& LoadedGpr::jitter() const noexcept
    {
        return impl_->jitter;
    }

    /// @brief Returns the restored exponential evaluation backend.
    /// @return Configured KernelExp mode.
    KernelExp LoadedGpr::kernel_exp() const noexcept
    {
        return impl_->kernel_exp;
    }

    /// @brief Returns the restored distance matrix caching policy.
    /// @return Configured DistanceCachePolicy.
    DistanceCachePolicy LoadedGpr::distance_cache_policy() const noexcept
    {
        return impl_->distance_cache;
    }

    /// @brief Computes double-promoted posterior mean and variance into out.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[out] out Destination double-precision Prediction struct.
    /// @param[in] options Optional prediction variance controls.
    /// @return Empty expected on success, or GprError on query validation or transform failure.
    std::expected<void, GprError> LoadedGpr::predict_into(
        MatrixRef<double> xs, Prediction<double>& out, const PredictOptions& options
    ) const
    {
        return with_typed_state(
            *impl_, [&](const auto& st)
        {
            return loaded_predict_typed(
                st, impl_->likelihood, *impl_->x_transform, *impl_->y_transform, impl_->n, impl_->d, xs, options, out
            );
        }
        );
    }

    /// @brief Allocates and computes double-promoted posterior mean and diagonal variance.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[in] options Optional prediction variance controls.
    /// @return Double-precision Prediction on the original target scale, or GprError on failure.
    std::expected<Prediction<double>, GprError> LoadedGpr::predict(
        MatrixRef<double> xs, const PredictOptions& options
    ) const
    {
        Prediction<double> out;
        if (auto r = predict_into(xs, out, options); !r.has_value())
        {
            return std::unexpected(std::move(r).error());
        }
        return out;
    }

    /// @brief Computes double-promoted posterior mean and full m x m covariance matrix.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[in] options Optional prediction variance controls.
    /// @return Double-precision PredictiveCovariance on the original target scale, or GprError on failure.
    std::expected<PredictiveCovariance<double>, GprError> LoadedGpr::predict_covariance(
        MatrixRef<double> xs, const PredictOptions& options
    ) const
    {
        auto to_fitted_cov = [&]<PrecisionPolicyConcept P>(
                                 const LoadedTypedState<P>& st
                             ) -> std::expected<PredictiveCovariance<double>, GprError>
        {
            using Storage = P::Storage;
            auto xq = stage_loaded_query(st, *impl_->x_transform, impl_->n, impl_->d, xs);
            if (!xq.has_value())
            {
                return std::unexpected(std::move(xq).error());
            }
            const MatrixRef<Storage> xq_ref = *xq;
            const std::size_t m = xq_ref.rows();
            const MatrixRef<Storage> x_train_ref(st.x_train.data(), impl_->n, impl_->d, impl_->n);
            const MatrixRef<Storage> l_ref(st.l_view.data(), impl_->n, impl_->n, impl_->n);
            auto cov_res = cxxgpr::predict_covariance<P>(
                *st.compiled_kernel, impl_->likelihood, x_train_ref, l_ref, st.alpha_view, xq_ref, options, st.query_ws
            );
            if (!cov_res.has_value())
            {
                return std::unexpected(std::move(cov_res).error());
            }
            // Promote results to double and apply inverse target transforms on the original scale.
            PredictiveCovariance<double> out{
                .mean = std::vector<double>(cov_res->mean.begin(), cov_res->mean.end()),
                .covariance = std::vector<double>(cov_res->covariance.begin(), cov_res->covariance.end()),
                .n_points = m,
                .variance_kind = cov_res->variance_kind,
            };
            if (auto im = impl_->y_transform->inverse_transform_mean(out.mean); !im.has_value())
            {
                return std::unexpected(std::move(im).error());
            }
            if (auto ic = impl_->y_transform->inverse_transform_covariance(out.covariance); !ic.has_value())
            {
                return std::unexpected(std::move(ic).error());
            }
            return out;
        };

        return with_typed_state(*impl_, to_fitted_cov);
    }

    /// @brief Draws deterministic posterior samples promoted to double precision on the original target scale.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[in] n_draws Number of posterior draws to generate.
    /// @param[in] seed Deterministic PRNG seed.
    /// @param[in] options Optional prediction variance controls.
    /// @param[in] jitter Optional predictive covariance jitter override.
    /// @return Column-major double sample matrix (m x n_draws), or GprError on validation or Cholesky failure.
    std::expected<std::vector<double>, GprError> LoadedGpr::sample(
        MatrixRef<double> xs,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options,
        std::optional<JitterPolicy> jitter
    ) const
    {
        auto sample_typed =
            [&]<PrecisionPolicyConcept P>(const LoadedTypedState<P>& st) -> std::expected<std::vector<double>, GprError>
        {
            using Storage = P::Storage;
            auto xq = stage_loaded_query(st, *impl_->x_transform, impl_->n, impl_->d, xs);
            if (!xq.has_value())
            {
                return std::unexpected(std::move(xq).error());
            }
            const MatrixRef<Storage> xq_ref = *xq;
            const std::size_t m = xq_ref.rows();
            const MatrixRef<Storage> x_train_ref(st.x_train.data(), impl_->n, impl_->d, impl_->n);
            const MatrixRef<Storage> l_ref(st.l_view.data(), impl_->n, impl_->n, impl_->n);
            auto raw_draws = cxxgpr::sample<P>(
                *st.compiled_kernel, impl_->likelihood, x_train_ref, l_ref, st.alpha_view, xq_ref, n_draws, seed,
                options, st.query_ws, jitter
            );
            if (!raw_draws.has_value())
            {
                return std::unexpected(std::move(raw_draws).error());
            }
            // Promote draws to double and apply inverse mean transform per column draw.
            std::vector<double> out(raw_draws->begin(), raw_draws->end());
            for (std::size_t draw = 0; draw < n_draws; ++draw)
            {
                const std::span<double> slice(out.data() + (draw * m), m);
                if (auto im = impl_->y_transform->inverse_transform_mean(slice); !im.has_value())
                {
                    return std::unexpected(std::move(im).error());
                }
            }
            return out;
        };

        return with_typed_state(*impl_, sample_typed);
    }

    /// @brief Computes double-promoted closed-form Leave-One-Out predictive moments on the original scale.
    /// @param[in] options Optional prediction variance controls.
    /// @return Double-precision LOO Prediction on the original target scale, or GprError on failure.
    std::expected<Prediction<double>, GprError> LoadedGpr::loo_predict(const PredictOptions& options) const
    {
        auto loo_typed =
            [&]<PrecisionPolicyConcept P>(const LoadedTypedState<P>& st) -> std::expected<Prediction<double>, GprError>
        {
            using Storage = P::Storage;
            const MatrixRef<Storage> l_ref(st.l_view.data(), impl_->n, impl_->n, impl_->n);
            auto loo = cxxgpr::loo_predict<P>(impl_->likelihood, l_ref, st.alpha_view, st.y_transformed, options);
            if (!loo.has_value())
            {
                return std::unexpected(std::move(loo).error());
            }
            // Promote LOO outputs to double before inverse target scaling.
            Prediction<double> out{
                .mean = std::vector<double>(loo->mean.begin(), loo->mean.end()),
                .variance = std::vector<double>(loo->variance.begin(), loo->variance.end()),
                .variance_kind = loo->variance_kind,
            };
            if (auto im = impl_->y_transform->inverse_transform_mean(out.mean); !im.has_value())
            {
                return std::unexpected(std::move(im).error());
            }
            if (auto iv = impl_->y_transform->inverse_transform_variance(out.variance); !iv.has_value())
            {
                return std::unexpected(std::move(iv).error());
            }
            return out;
        };

        return with_typed_state(*impl_, loo_typed);
    }

    /// @brief Copies restored state into an owned FittedPayload<P> decoupled from mmap lifetimes.
    /// @tparam P Target precision policy matching impl_->precision_kind.
    /// @param[out] out_unfitted_in Output pointer populated with cloned unfitted input transform.
    /// @param[out] out_unfitted_tgt Output pointer populated with cloned unfitted target transform.
    /// @return Owned FittedPayload<P> populated from impl_.
    template <PrecisionPolicyConcept P>
    std::expected<cxxgpr::detail::FittedPayload<P>, GprError> LoadedGpr::extract_payload(
        std::unique_ptr<transform::UnfittedInputTransform>& out_unfitted_in,
        std::unique_ptr<transform::UnfittedTargetTransform>& out_unfitted_tgt
    ) const
    {
        out_unfitted_in = impl_->x_unfitted->clone();
        out_unfitted_tgt = impl_->y_unfitted->clone();

        // Select the typed state matching P after precision_kind() check in into_fitted().
        const auto& st = [&]() -> const LoadedTypedState<P>&
        {
            if constexpr (std::same_as<P, DoublePrecision>)
            {
                return *impl_->double_state;
            }
            else if constexpr (std::same_as<P, SinglePrecision>)
            {
                return *impl_->single_state;
            }
            else if constexpr (std::same_as<P, MixedPrecision<PromoteStorage>>)
            {
                return *impl_->mixed_promote_state;
            }
            else
            {
                return *impl_->mixed_reeval_state;
            }
        }();

        // Copy l_view and alpha_view into owned vectors so FittedGpr has no dependency on mapped memory.
        return cxxgpr::detail::FittedPayload<P>{
            .kernel_spec = impl_->kernel_spec,
            .likelihood = impl_->likelihood,
            .compiled_kernel = st.compiled_kernel,
            .input_transform = impl_->x_transform->clone(),
            .target_transform = impl_->y_transform->clone(),
            .x_raw = impl_->x_raw,
            .y_raw = impl_->y_raw,
            .x_train = st.x_train,
            .y_transformed = st.y_transformed,
            .l_factor = std::vector<typename P::Storage>(st.l_view.begin(), st.l_view.end()),
            .alpha = std::vector<typename P::Refine>(st.alpha_view.begin(), st.alpha_view.end()),
            .n_train = impl_->n,
            .n_features = impl_->d,
            .factor_jitter = impl_->factor_jitter,
            .neg_log_marginal_likelihood = st.neg_log_marginal_likelihood,
            .refinement_summary = std::nullopt,
            .optimizer_result = {},
        };
    }

    template std::expected<cxxgpr::detail::FittedPayload<DoublePrecision>, GprError> LoadedGpr::extract_payload<
        DoublePrecision>(
        std::unique_ptr<transform::UnfittedInputTransform>&, std::unique_ptr<transform::UnfittedTargetTransform>&
    ) const;
    template std::expected<cxxgpr::detail::FittedPayload<SinglePrecision>, GprError> LoadedGpr::extract_payload<
        SinglePrecision>(
        std::unique_ptr<transform::UnfittedInputTransform>&, std::unique_ptr<transform::UnfittedTargetTransform>&
    ) const;
    template std::expected<cxxgpr::detail::FittedPayload<MixedPrecision<PromoteStorage>>, GprError> LoadedGpr::
        extract_payload<MixedPrecision<PromoteStorage>>(
            std::unique_ptr<transform::UnfittedInputTransform>&, std::unique_ptr<transform::UnfittedTargetTransform>&
        ) const;
    template std::expected<cxxgpr::detail::FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError> LoadedGpr::
        extract_payload<MixedPrecision<ReevaluateKernel>>(
            std::unique_ptr<transform::UnfittedInputTransform>&, std::unique_ptr<transform::UnfittedTargetTransform>&
        ) const;

    namespace detail
    {

        /// @brief Validates inputs and atomically writes config.json and model.safetensors into dir.
        /// @details Checks dimensions and tensor finiteness, serializes config.json, builds aligned tensor
        /// descriptors, writes temporary staging files model.safetensors.tmp and config.json.tmp inside dir,
        /// and atomically renames them into final checkpoint paths with cleanup on failure.
        /// @param[in] input Borrowed configuration and raw/factor spans for serialization.
        /// @param[in] dir Destination directory path created if needed.
        /// @param[in] registry Unused extension registry parameter kept for API symmetry.
        /// @return Empty expected on success, or GprError on validation, serialization, or I/O failure.
        /// @retval GprErrorCode::EmptyInput When n_train or n_features is zero.
        /// @retval GprErrorCode::NonFiniteInput When factor_jitter or any persisted tensor element is non-finite.
        /// @retval GprErrorCode::PersistFailed When config serialization, directory creation, writes, or renames fail.
        std::expected<void, GprError> save_model_impl(
            const SaveInput& input, const std::filesystem::path& dir, [[maybe_unused]] const PersistRegistry& registry
        )
        {
            if (input.n_train == 0U || input.n_features == 0U)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (!std::isfinite(input.factor_jitter))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            // Verify finiteness across all raw and optional factor/alpha spans before creating directories.
            auto check_finite_span = [](auto sp) -> std::expected<void, GprError>
            {
                for (const auto v : sp)
                {
                    if (!std::isfinite(v))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
                return {};
            };
            if (auto ok = check_finite_span(input.x_raw); !ok.has_value())
            {
                return ok;
            }
            if (auto ok = check_finite_span(input.y_raw); !ok.has_value())
            {
                return ok;
            }
            if (input.has_factor)
            {
                if (auto ok = check_finite_span(input.l_f64); !ok.has_value())
                {
                    return ok;
                }
                if (auto ok = check_finite_span(input.l_f32); !ok.has_value())
                {
                    return ok;
                }
                if (auto ok = check_finite_span(input.alpha_f64); !ok.has_value())
                {
                    return ok;
                }
                if (auto ok = check_finite_span(input.alpha_f32); !ok.has_value())
                {
                    return ok;
                }
            }
            auto cfg_str = serialize_config_json(input);
            if (!cfg_str.has_value())
            {
                return std::unexpected(std::move(cfg_str).error());
            }

            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Io, "failed to create target persistence directory")
                );
            }

            // Assemble raw F64 tensors and optional L / alpha tensors matching precision_kind.
            std::vector<TensorWriteView> tensors;
            tensors.push_back(
                TensorWriteView{
                    .name = "x",
                    .dtype = TensorDtype::F64,
                    .shape = {input.n_train, input.n_features},
                    .bytes = as_bytes(input.x_raw),
            }
            );
            tensors.push_back(
                TensorWriteView{
                    .name = "y",
                    .dtype = TensorDtype::F64,
                    .shape = {input.n_train},
                    .bytes = as_bytes(input.y_raw),
                }
            );
            if (input.has_factor)
            {
                if (!input.l_f64.empty())
                {
                    tensors.push_back(
                        TensorWriteView{
                            .name = "l",
                            .dtype = TensorDtype::F64,
                            .shape = {input.n_train, input.n_train},
                            .bytes = as_bytes(input.l_f64),
                    }
                    );
                }
                else
                {
                    tensors.push_back(
                        TensorWriteView{
                            .name = "l",
                            .dtype = TensorDtype::F32,
                            .shape = {input.n_train, input.n_train},
                            .bytes = as_bytes(input.l_f32),
                    }
                    );
                }
                if (!input.alpha_f64.empty())
                {
                    tensors.push_back(
                        TensorWriteView{
                            .name = "alpha",
                            .dtype = TensorDtype::F64,
                            .shape = {input.n_train},
                            .bytes = as_bytes(input.alpha_f64),
                        }
                    );
                }
                else
                {
                    tensors.push_back(
                        TensorWriteView{
                            .name = "alpha",
                            .dtype = TensorDtype::F32,
                            .shape = {input.n_train},
                            .bytes = as_bytes(input.alpha_f32),
                        }
                    );
                }
            }

            // Write staged temporary files inside dir, then atomically rename into final targets.
            const std::filesystem::path tmp_st = dir / "model.safetensors.tmp";
            const std::filesystem::path tmp_cfg = dir / "config.json.tmp";

            if (auto wr = write_safetensors_file(tmp_st, tensors); !wr.has_value())
            {
                std::filesystem::remove(tmp_st, ec);
                return wr;
            }

            {
                std::ofstream ofs(tmp_cfg, std::ios::binary | std::ios::trunc);
                if (!ofs)
                {
                    std::filesystem::remove(tmp_st, ec);
                    return std::unexpected(
                        GprError::persist_failed(
                            PersistErrorKind::Io, "failed to open temporary config.json for writing"
                        )
                    );
                }
                ofs.write(cfg_str->data(), static_cast<std::streamsize>(cfg_str->size()));
                ofs.flush();
                if (!ofs)
                {
                    std::filesystem::remove(tmp_st, ec);
                    std::filesystem::remove(tmp_cfg, ec);
                    return std::unexpected(
                        GprError::persist_failed(PersistErrorKind::Io, "failed while writing temporary config.json")
                    );
                }
            }

            // Replace model.safetensors and config.json atomically and clean up leftover temporary files on error.
            std::filesystem::rename(tmp_st, dir / "model.safetensors", ec);
            if (ec)
            {
                std::filesystem::remove(tmp_st, ec);
                std::filesystem::remove(tmp_cfg, ec);
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Io, "failed to atomically replace model.safetensors")
                );
            }
            std::filesystem::rename(tmp_cfg, dir / "config.json", ec);
            if (ec)
            {
                std::filesystem::remove(tmp_cfg, ec);
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Io, "failed to atomically replace config.json")
                );
            }

            return {};
        }

    }  // namespace detail

}  // namespace cxxgpr::persist
