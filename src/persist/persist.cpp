// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/persist/persist.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/workspace/inference.hpp"
#include "cxxgpr/workspace/mll.hpp"
#include "config_json.hpp"
#include "safetensors.hpp"

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

namespace cxxgpr::persist {

std::expected<void, GprError> validate_persist_id(std::string_view persist_id) {
    if (persist_id.empty()) {
        return std::unexpected(GprError::persist_failed(PersistErrorKind::InvalidPersistId,
                                                        "persist_id must not be empty"));
    }
    if (persist_id.starts_with(RESERVED_PREFIX)) {
        return std::unexpected(GprError::persist_failed(
            PersistErrorKind::InvalidPersistId, "persist_id must not start with 'cxxgpr.'"));
    }
    return {};
}

namespace {

template <typename MapT, typename FnT>
[[nodiscard]] std::expected<void, GprError> register_impl(MapT& map, std::string persist_id,
                                                          FnT factory) {
    if (auto v = validate_persist_id(persist_id); !v.has_value()) {
        return v;
    }
    if (map.contains(persist_id)) {
        return std::unexpected(GprError::persist_failed(PersistErrorKind::InvalidPersistId,
                                                        "duplicate persist_id registration"));
    }
    map.emplace(std::move(persist_id), std::move(factory));
    return {};
}

template <typename MapT>
[[nodiscard]] auto create_impl(const MapT& map, std::string_view persist_id,
                               std::string_view state_json)
    -> decltype(map.begin()->second(state_json)) {
    const auto it = map.find(std::string(persist_id));
    if (it == map.end()) {
        return std::unexpected(GprError::persist_failed(PersistErrorKind::UnregisteredId,
                                                        "unregistered persist_id in config.json"));
    }
    return it->second(state_json);
}

template <typename T>
[[nodiscard]] std::span<const std::byte> as_bytes(std::span<const T> s) noexcept {
    return std::as_bytes(s);
}

template <typename P> struct LoadedTypedState {
    using Storage = P::Storage;
    using Refine = P::Refine;

    std::shared_ptr<const CompiledKernel<Storage>> compiled_kernel;
    std::vector<Storage> x_train;
    std::vector<Refine> y_transformed;
    std::vector<Storage> l_owned;
    std::span<const Storage> l_view;
    std::vector<Refine> alpha_owned;
    std::span<const Refine> alpha_view;
    double log_marginal_likelihood{0.0};
    mutable QueryWorkspace<P> query_ws;
};

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<LoadedTypedState<P>, GprError>
build_typed_state(const detail::ParsedConfig& cfg,
                  const std::shared_ptr<const detail::MappedSafetensors>& st,
                  std::span<const double> x_raw, std::span<const double> y_raw, bool& out_is_mmap) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    const std::size_t n = cfg.n;
    const std::size_t d = cfg.d;

    std::vector<double> x_trans(x_raw.begin(), x_raw.end());
    if (auto ap = cfg.x_transform->apply(x_trans, n, d); !ap.has_value()) {
        return std::unexpected(std::move(ap).error());
    }
    std::vector<double> y_trans(y_raw.begin(), y_raw.end());
    if (auto tf = cfg.y_transform->transform(y_trans); !tf.has_value()) {
        return std::unexpected(std::move(tf).error());
    }
    for (const double v : x_trans) {
        if (!std::isfinite(v)) {
            return std::unexpected(GprError::persist_failed(
                PersistErrorKind::Tensor, "non-finite value after applying x_transform"));
        }
    }
    for (const double v : y_trans) {
        if (!std::isfinite(v)) {
            return std::unexpected(GprError::persist_failed(
                PersistErrorKind::Tensor, "non-finite value after applying y_transform"));
        }
    }

    LoadedTypedState<P> state;
    {
        auto ck = std::make_shared<CompiledKernel<Storage>>(
            CompiledKernel<Storage>::from_spec(*cfg.kernel_spec));
        ck->set_kernel_exp(cfg.kernel_exp);
        state.compiled_kernel = std::move(ck);
    }

    state.x_train.resize(x_trans.size());
    for (std::size_t i = 0; i < x_trans.size(); ++i) {
        state.x_train[i] = static_cast<Storage>(x_trans[i]);
    }
    state.y_transformed.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        state.y_transformed[i] = static_cast<Refine>(y_trans[i]);
    }

    if (cfg.has_factor) {
        const std::array<std::size_t, 2> l_shape{n, n};
        const std::array<std::size_t, 1> a_shape{n};
        if constexpr (std::same_as<Storage, double>) {
            auto l_sp = st->read_f64("l", l_shape);
            if (!l_sp.has_value()) {
                return std::unexpected(std::move(l_sp).error());
            }
            state.l_view = *l_sp;
            out_is_mmap = true;
        } else {
            auto l_sp = st->read_f32("l", l_shape);
            if (!l_sp.has_value()) {
                return std::unexpected(std::move(l_sp).error());
            }
            state.l_owned.assign(l_sp->begin(), l_sp->end());
            state.l_view = state.l_owned;
            out_is_mmap = false;
        }
        if constexpr (std::same_as<Refine, double>) {
            auto a_sp = st->read_f64("alpha", a_shape);
            if (!a_sp.has_value()) {
                return std::unexpected(std::move(a_sp).error());
            }
            state.alpha_owned.assign(a_sp->begin(), a_sp->end());
            state.alpha_view = state.alpha_owned;
        } else {
            auto a_sp = st->read_f32("alpha", a_shape);
            if (!a_sp.has_value()) {
                return std::unexpected(std::move(a_sp).error());
            }
            state.alpha_owned.assign(a_sp->begin(), a_sp->end());
            state.alpha_view = state.alpha_owned;
        }
        return state;
    }

    // Reconstruct L and alpha from raw data and saved configuration (has_factor == false).
    out_is_mmap = false;
    auto buffers = FitBuffers<P>::create(n, FitPolicy{.distance_cache = cfg.distance_cache,
                                                      .cholesky_buffer = CholeskyBuffer::Retain});
    if (!buffers.has_value()) {
        return std::unexpected(std::move(buffers).error());
    }
    std::vector<Storage> y_storage(n);
    for (std::size_t i = 0; i < n; ++i) {
        y_storage[i] = static_cast<Storage>(y_trans[i]);
    }
    const MatrixRef<Storage> x_ref(state.x_train.data(), n, d, n);
    auto jitter_to_use = cfg.jitter;
    if (cfg.factor_jitter > 0.0) {
        auto fj = JitterPolicy::fixed(cfg.factor_jitter);
        if (fj.has_value()) {
            jitter_to_use = *fj;
        }
    }
    auto mll = factor_and_mll(*state.compiled_kernel, *cfg.likelihood, x_ref, y_storage,
                              jitter_to_use, *buffers, 0U);
    if (!mll.has_value()) {
        return std::unexpected(std::move(mll).error());
    }
    state.log_marginal_likelihood = *mll;
    state.l_owned.resize(n * n);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            state.l_owned[(col * n) + row] = buffers->core.k_ref()(row, col);
        }
    }
    state.l_view = state.l_owned;

    state.alpha_owned.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        state.alpha_owned[i] = static_cast<Refine>(buffers->core.rhs[i]);
    }
    state.alpha_view = state.alpha_owned;
    return state;
}

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError>
loaded_predict_typed(const LoadedTypedState<P>& st, const GaussianLikelihood& lik,
                     const transform::FittedInputTransform& in_tf,
                     const transform::FittedTargetTransform& tgt_tf, std::size_t n, std::size_t d,
                     MatrixRef<double> xs, const PredictOptions& options, Prediction<double>& out) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    if (xs.is_empty() || xs.data() == nullptr) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t m = xs.rows();
    if (xs.cols() != d) {
        return std::unexpected(GprError::dimension_mismatch(xs.cols(), d));
    }
    if (auto ens = st.query_ws.ensure(n, m, d, st.compiled_kernel->nested_depth());
        !ens.has_value()) {
        return ens;
    }
    for (std::size_t col = 0; col < d; ++col) {
        for (std::size_t row = 0; row < m; ++row) {
            const double val = xs(row, col);
            if (!std::isfinite(val)) {
                return std::unexpected(GprError::non_finite_input());
            }
            st.query_ws.query_xs[(col * m) + row] = val;
        }
    }
    if (auto ap = in_tf.apply(st.query_ws.query_xs, m, d); !ap.has_value()) {
        return ap;
    }
    for (std::size_t i = 0; i < m * d; ++i) {
        st.query_ws.query_x[i] = static_cast<Storage>(st.query_ws.query_xs[i]);
    }
    const MatrixRef<Storage> x_train_ref(st.x_train.data(), n, d, n);
    const MatrixRef<Storage> l_ref(st.l_view.data(), n, n, n);
    const MatrixRef<Storage> xq_ref(st.query_ws.query_x.data(), m, d, m);

    if constexpr (std::same_as<Refine, double>) {
        if (auto pr = predict_into<P>(*st.compiled_kernel, lik, x_train_ref, l_ref, st.alpha_view,
                                      xq_ref, options, st.query_ws, out);
            !pr.has_value()) {
            return pr;
        }
    } else {
        Prediction<float> tmp;
        if (auto pr = predict_into<P>(*st.compiled_kernel, lik, x_train_ref, l_ref, st.alpha_view,
                                      xq_ref, options, st.query_ws, tmp);
            !pr.has_value()) {
            return pr;
        }
        out.mean.resize(m);
        out.variance.resize(m);
        for (std::size_t i = 0; i < m; ++i) {
            out.mean[i] = static_cast<double>(tmp.mean[i]);
            out.variance[i] = static_cast<double>(tmp.variance[i]);
        }
        out.variance_kind = tmp.variance_kind;
    }
    if (auto im = tgt_tf.inverse_transform_mean(out.mean); !im.has_value()) {
        return im;
    }
    return tgt_tf.inverse_transform_variance(out.variance);
}

} // namespace

std::expected<void, GprError> PersistRegistry::register_kernel(std::string persist_id,
                                                               CustomKernelFactory factory) {
    return register_impl(kernels_, std::move(persist_id), std::move(factory));
}

std::expected<void, GprError>
PersistRegistry::register_unfitted_input(std::string persist_id, UnfittedInputFactory factory) {
    return register_impl(unfitted_inputs_, std::move(persist_id), std::move(factory));
}

std::expected<void, GprError> PersistRegistry::register_fitted_input(std::string persist_id,
                                                                     FittedInputFactory factory) {
    return register_impl(fitted_inputs_, std::move(persist_id), std::move(factory));
}

std::expected<void, GprError>
PersistRegistry::register_unfitted_target(std::string persist_id, UnfittedTargetFactory factory) {
    return register_impl(unfitted_targets_, std::move(persist_id), std::move(factory));
}

std::expected<void, GprError> PersistRegistry::register_fitted_target(std::string persist_id,
                                                                      FittedTargetFactory factory) {
    return register_impl(fitted_targets_, std::move(persist_id), std::move(factory));
}

std::expected<CustomKernel, GprError>
PersistRegistry::create_kernel(std::string_view persist_id, std::string_view state_json) const {
    return create_impl(kernels_, persist_id, state_json);
}

std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError>
PersistRegistry::create_unfitted_input(std::string_view persist_id,
                                       std::string_view state_json) const {
    return create_impl(unfitted_inputs_, persist_id, state_json);
}

std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError>
PersistRegistry::create_fitted_input(std::string_view persist_id,
                                     std::string_view state_json) const {
    return create_impl(fitted_inputs_, persist_id, state_json);
}

std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>
PersistRegistry::create_unfitted_target(std::string_view persist_id,
                                        std::string_view state_json) const {
    return create_impl(unfitted_targets_, persist_id, state_json);
}

std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError>
PersistRegistry::create_fitted_target(std::string_view persist_id,
                                      std::string_view state_json) const {
    return create_impl(fitted_targets_, persist_id, state_json);
}

struct LoadedGpr::Impl {
    std::size_t n{0};
    std::size_t d{0};
    bool has_factor{false};
    bool has_mmap_factor{false};
    double factor_jitter{0.0};
    PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
    KernelExp kernel_exp{KernelExp::Accurate};
    DistanceCachePolicy distance_cache{DistanceCachePolicy::Cached};
    KernelSpec kernel_spec;
    GaussianLikelihood likelihood;
    JitterPolicy jitter;
    std::unique_ptr<transform::UnfittedInputTransform> x_unfitted;
    std::unique_ptr<transform::FittedInputTransform> x_transform;
    std::unique_ptr<transform::UnfittedTargetTransform> y_unfitted;
    std::unique_ptr<transform::FittedTargetTransform> y_transform;
    std::shared_ptr<const detail::MappedSafetensors> safetensors;
    std::vector<double> x_raw;
    std::vector<double> y_raw;
    std::optional<LoadedTypedState<DoublePrecision>> double_state;
    std::optional<LoadedTypedState<SinglePrecision>> single_state;
    std::optional<LoadedTypedState<MixedPrecision<PromoteStorage>>> mixed_promote_state;
    std::optional<LoadedTypedState<MixedPrecision<ReevaluateKernel>>> mixed_reeval_state;
};

LoadedGpr::~LoadedGpr() = default;

std::expected<LoadedGpr, GprError> LoadedGpr::load(const std::filesystem::path& dir,
                                                   const PersistRegistry& registry) {
    auto cfg = detail::parse_config_json(dir / "config.json", registry);
    if (!cfg.has_value()) {
        return std::unexpected(std::move(cfg).error());
    }
    auto st = detail::MappedSafetensors::open(dir / "model.safetensors");
    if (!st.has_value()) {
        return std::unexpected(std::move(st).error());
    }

    const std::array<std::size_t, 2> x_shape{cfg->n, cfg->d};
    const std::array<std::size_t, 1> y_shape{cfg->n};
    auto x_sp = (*st)->read_f64("x", x_shape);
    if (!x_sp.has_value()) {
        return std::unexpected(std::move(x_sp).error());
    }
    auto y_sp = (*st)->read_f64("y", y_shape);
    if (!y_sp.has_value()) {
        return std::unexpected(std::move(y_sp).error());
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

    switch (impl->precision_kind) {
    case PersistPrecisionKind::Double: {
        auto s = build_typed_state<DoublePrecision>(cfg_view, *st, impl->x_raw, impl->y_raw,
                                                    impl->has_mmap_factor);
        if (!s.has_value()) {
            return std::unexpected(std::move(s).error());
        }
        impl->double_state = std::move(*s);
        break;
    }
    case PersistPrecisionKind::Single: {
        auto s = build_typed_state<SinglePrecision>(cfg_view, *st, impl->x_raw, impl->y_raw,
                                                    impl->has_mmap_factor);
        if (!s.has_value()) {
            return std::unexpected(std::move(s).error());
        }
        impl->single_state = std::move(*s);
        break;
    }
    case PersistPrecisionKind::MixedPromote: {
        auto s = build_typed_state<MixedPrecision<PromoteStorage>>(
            cfg_view, *st, impl->x_raw, impl->y_raw, impl->has_mmap_factor);
        if (!s.has_value()) {
            return std::unexpected(std::move(s).error());
        }
        impl->mixed_promote_state = std::move(*s);
        break;
    }
    case PersistPrecisionKind::MixedReevaluate: {
        auto s = build_typed_state<MixedPrecision<ReevaluateKernel>>(
            cfg_view, *st, impl->x_raw, impl->y_raw, impl->has_mmap_factor);
        if (!s.has_value()) {
            return std::unexpected(std::move(s).error());
        }
        impl->mixed_reeval_state = std::move(*s);
        break;
    }
    }

    return LoadedGpr(std::move(impl));
}

std::size_t LoadedGpr::n() const noexcept {
    return impl_->n;
}
std::size_t LoadedGpr::d() const noexcept {
    return impl_->d;
}
PersistPrecisionKind LoadedGpr::precision_kind() const noexcept {
    return impl_->precision_kind;
}
bool LoadedGpr::has_factor() const noexcept {
    return impl_->has_factor;
}
bool LoadedGpr::has_mmap_factor() const noexcept {
    return impl_->has_mmap_factor;
}
double LoadedGpr::factor_jitter() const noexcept {
    return impl_->factor_jitter;
}
const KernelSpec& LoadedGpr::kernel_spec() const noexcept {
    return impl_->kernel_spec;
}
const GaussianLikelihood& LoadedGpr::likelihood() const noexcept {
    return impl_->likelihood;
}
const JitterPolicy& LoadedGpr::jitter() const noexcept {
    return impl_->jitter;
}
KernelExp LoadedGpr::kernel_exp() const noexcept {
    return impl_->kernel_exp;
}
DistanceCachePolicy LoadedGpr::distance_cache_policy() const noexcept {
    return impl_->distance_cache;
}

std::expected<void, GprError> LoadedGpr::predict_into(MatrixRef<double> xs, Prediction<double>& out,
                                                      const PredictOptions& options) const {
    switch (impl_->precision_kind) {
    case PersistPrecisionKind::Double:
        return loaded_predict_typed(*impl_->double_state, impl_->likelihood, *impl_->x_transform,
                                    *impl_->y_transform, impl_->n, impl_->d, xs, options, out);
    case PersistPrecisionKind::Single:
        return loaded_predict_typed(*impl_->single_state, impl_->likelihood, *impl_->x_transform,
                                    *impl_->y_transform, impl_->n, impl_->d, xs, options, out);
    case PersistPrecisionKind::MixedPromote:
        return loaded_predict_typed(*impl_->mixed_promote_state, impl_->likelihood,
                                    *impl_->x_transform, *impl_->y_transform, impl_->n, impl_->d,
                                    xs, options, out);
    case PersistPrecisionKind::MixedReevaluate:
        return loaded_predict_typed(*impl_->mixed_reeval_state, impl_->likelihood,
                                    *impl_->x_transform, *impl_->y_transform, impl_->n, impl_->d,
                                    xs, options, out);
    }
    return std::unexpected(
        GprError::persist_failed(PersistErrorKind::Config, "invalid precision state in LoadedGpr"));
}

std::expected<Prediction<double>, GprError>
LoadedGpr::predict(MatrixRef<double> xs, const PredictOptions& options) const {
    Prediction<double> out;
    if (auto r = predict_into(xs, out, options); !r.has_value()) {
        return std::unexpected(std::move(r).error());
    }
    return out;
}

std::expected<PredictiveCovariance<double>, GprError>
LoadedGpr::predict_covariance(MatrixRef<double> xs, const PredictOptions& options) const {
    auto to_fitted_cov = [&]<PrecisionPolicyConcept P>(const LoadedTypedState<P>& st)
        -> std::expected<PredictiveCovariance<double>, GprError> {
        using Storage = P::Storage;
        if (xs.is_empty() || xs.data() == nullptr) {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t m = xs.rows();
        if (xs.cols() != impl_->d) {
            return std::unexpected(GprError::dimension_mismatch(xs.cols(), impl_->d));
        }
        if (auto ens =
                st.query_ws.ensure(impl_->n, m, impl_->d, st.compiled_kernel->nested_depth());
            !ens.has_value()) {
            return std::unexpected(std::move(ens).error());
        }
        for (std::size_t col = 0; col < impl_->d; ++col) {
            for (std::size_t row = 0; row < m; ++row) {
                const double val = xs(row, col);
                if (!std::isfinite(val)) {
                    return std::unexpected(GprError::non_finite_input());
                }
                st.query_ws.query_xs[(col * m) + row] = val;
            }
        }
        if (auto ap = impl_->x_transform->apply(st.query_ws.query_xs, m, impl_->d);
            !ap.has_value()) {
            return std::unexpected(std::move(ap).error());
        }
        for (std::size_t i = 0; i < m * impl_->d; ++i) {
            st.query_ws.query_x[i] = static_cast<Storage>(st.query_ws.query_xs[i]);
        }
        const MatrixRef<Storage> x_train_ref(st.x_train.data(), impl_->n, impl_->d, impl_->n);
        const MatrixRef<Storage> l_ref(st.l_view.data(), impl_->n, impl_->n, impl_->n);
        const MatrixRef<Storage> xq_ref(st.query_ws.query_x.data(), m, impl_->d, m);
        auto cov_res =
            cxxgpr::predict_covariance<P>(*st.compiled_kernel, impl_->likelihood, x_train_ref,
                                          l_ref, st.alpha_view, xq_ref, options, st.query_ws);
        if (!cov_res.has_value()) {
            return std::unexpected(std::move(cov_res).error());
        }
        PredictiveCovariance<double> out{
            .mean = std::vector<double>(cov_res->mean.begin(), cov_res->mean.end()),
            .covariance =
                std::vector<double>(cov_res->covariance.begin(), cov_res->covariance.end()),
            .n_points = m,
            .variance_kind = cov_res->variance_kind,
        };
        if (auto im = impl_->y_transform->inverse_transform_mean(out.mean); !im.has_value()) {
            return std::unexpected(std::move(im).error());
        }
        if (auto ic = impl_->y_transform->inverse_transform_covariance(out.covariance);
            !ic.has_value()) {
            return std::unexpected(std::move(ic).error());
        }
        return out;
    };

    switch (impl_->precision_kind) {
    case PersistPrecisionKind::Double:
        return to_fitted_cov(*impl_->double_state);
    case PersistPrecisionKind::Single:
        return to_fitted_cov(*impl_->single_state);
    case PersistPrecisionKind::MixedPromote:
        return to_fitted_cov(*impl_->mixed_promote_state);
    case PersistPrecisionKind::MixedReevaluate:
        return to_fitted_cov(*impl_->mixed_reeval_state);
    }
    return std::unexpected(
        GprError::persist_failed(PersistErrorKind::Config, "invalid precision state"));
}

std::expected<std::vector<double>, GprError>
LoadedGpr::sample(MatrixRef<double> xs, std::size_t n_draws, std::uint64_t seed,
                  const PredictOptions& options, std::optional<JitterPolicy> jitter) const {
    auto sample_typed =
        [&]<PrecisionPolicyConcept P>(
            const LoadedTypedState<P>& st) -> std::expected<std::vector<double>, GprError> {
        using Storage = P::Storage;
        if (xs.is_empty() || xs.data() == nullptr) {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t m = xs.rows();
        if (xs.cols() != impl_->d) {
            return std::unexpected(GprError::dimension_mismatch(xs.cols(), impl_->d));
        }
        if (auto ens =
                st.query_ws.ensure(impl_->n, m, impl_->d, st.compiled_kernel->nested_depth());
            !ens.has_value()) {
            return std::unexpected(std::move(ens).error());
        }
        for (std::size_t col = 0; col < impl_->d; ++col) {
            for (std::size_t row = 0; row < m; ++row) {
                const double val = xs(row, col);
                if (!std::isfinite(val)) {
                    return std::unexpected(GprError::non_finite_input());
                }
                st.query_ws.query_xs[(col * m) + row] = val;
            }
        }
        if (auto ap = impl_->x_transform->apply(st.query_ws.query_xs, m, impl_->d);
            !ap.has_value()) {
            return std::unexpected(std::move(ap).error());
        }
        for (std::size_t i = 0; i < m * impl_->d; ++i) {
            st.query_ws.query_x[i] = static_cast<Storage>(st.query_ws.query_xs[i]);
        }
        const MatrixRef<Storage> x_train_ref(st.x_train.data(), impl_->n, impl_->d, impl_->n);
        const MatrixRef<Storage> l_ref(st.l_view.data(), impl_->n, impl_->n, impl_->n);
        const MatrixRef<Storage> xq_ref(st.query_ws.query_x.data(), m, impl_->d, m);
        auto raw_draws =
            cxxgpr::sample<P>(*st.compiled_kernel, impl_->likelihood, x_train_ref, l_ref,
                              st.alpha_view, xq_ref, n_draws, seed, options, st.query_ws, jitter);
        if (!raw_draws.has_value()) {
            return std::unexpected(std::move(raw_draws).error());
        }
        std::vector<double> out(raw_draws->begin(), raw_draws->end());
        for (std::size_t draw = 0; draw < n_draws; ++draw) {
            const std::span<double> slice(out.data() + (draw * m), m);
            if (auto im = impl_->y_transform->inverse_transform_mean(slice); !im.has_value()) {
                return std::unexpected(std::move(im).error());
            }
        }
        return out;
    };

    switch (impl_->precision_kind) {
    case PersistPrecisionKind::Double:
        return sample_typed(*impl_->double_state);
    case PersistPrecisionKind::Single:
        return sample_typed(*impl_->single_state);
    case PersistPrecisionKind::MixedPromote:
        return sample_typed(*impl_->mixed_promote_state);
    case PersistPrecisionKind::MixedReevaluate:
        return sample_typed(*impl_->mixed_reeval_state);
    }
    return std::unexpected(
        GprError::persist_failed(PersistErrorKind::Config, "invalid precision state"));
}

std::expected<Prediction<double>, GprError>
LoadedGpr::loo_predict(const PredictOptions& options) const {
    auto loo_typed =
        [&]<PrecisionPolicyConcept P>(
            const LoadedTypedState<P>& st) -> std::expected<Prediction<double>, GprError> {
        using Storage = P::Storage;
        const MatrixRef<Storage> l_ref(st.l_view.data(), impl_->n, impl_->n, impl_->n);
        auto loo = cxxgpr::loo_predict<P>(impl_->likelihood, l_ref, st.alpha_view, st.y_transformed,
                                          options);
        if (!loo.has_value()) {
            return std::unexpected(std::move(loo).error());
        }
        Prediction<double> out{
            .mean = std::vector<double>(loo->mean.begin(), loo->mean.end()),
            .variance = std::vector<double>(loo->variance.begin(), loo->variance.end()),
            .variance_kind = loo->variance_kind,
        };
        if (auto im = impl_->y_transform->inverse_transform_mean(out.mean); !im.has_value()) {
            return std::unexpected(std::move(im).error());
        }
        if (auto iv = impl_->y_transform->inverse_transform_variance(out.variance);
            !iv.has_value()) {
            return std::unexpected(std::move(iv).error());
        }
        return out;
    };

    switch (impl_->precision_kind) {
    case PersistPrecisionKind::Double:
        return loo_typed(*impl_->double_state);
    case PersistPrecisionKind::Single:
        return loo_typed(*impl_->single_state);
    case PersistPrecisionKind::MixedPromote:
        return loo_typed(*impl_->mixed_promote_state);
    case PersistPrecisionKind::MixedReevaluate:
        return loo_typed(*impl_->mixed_reeval_state);
    }
    return std::unexpected(
        GprError::persist_failed(PersistErrorKind::Config, "invalid precision state"));
}

template <PrecisionPolicyConcept P>
std::expected<cxxgpr::detail::FittedPayload<P>, GprError> LoadedGpr::extract_payload(
    std::unique_ptr<transform::UnfittedInputTransform>& out_unfitted_in,
    std::unique_ptr<transform::UnfittedTargetTransform>& out_unfitted_tgt) const {
    out_unfitted_in = impl_->x_unfitted->clone();
    out_unfitted_tgt = impl_->y_unfitted->clone();

    const auto& st = [&]() -> const LoadedTypedState<P>& {
        if constexpr (std::same_as<P, DoublePrecision>) {
            return *impl_->double_state;
        } else if constexpr (std::same_as<P, SinglePrecision>) {
            return *impl_->single_state;
        } else if constexpr (std::same_as<P, MixedPrecision<PromoteStorage>>) {
            return *impl_->mixed_promote_state;
        } else {
            return *impl_->mixed_reeval_state;
        }
    }();

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
        .log_marginal_likelihood = st.log_marginal_likelihood,
        .refinement_summary = std::nullopt,
        .optimizer_result = {},
    };
}

template std::expected<cxxgpr::detail::FittedPayload<DoublePrecision>, GprError>
LoadedGpr::extract_payload<DoublePrecision>(
    std::unique_ptr<transform::UnfittedInputTransform>&,
    std::unique_ptr<transform::UnfittedTargetTransform>&) const;
template std::expected<cxxgpr::detail::FittedPayload<SinglePrecision>, GprError>
LoadedGpr::extract_payload<SinglePrecision>(
    std::unique_ptr<transform::UnfittedInputTransform>&,
    std::unique_ptr<transform::UnfittedTargetTransform>&) const;
template std::expected<cxxgpr::detail::FittedPayload<MixedPrecision<PromoteStorage>>, GprError>
LoadedGpr::extract_payload<MixedPrecision<PromoteStorage>>(
    std::unique_ptr<transform::UnfittedInputTransform>&,
    std::unique_ptr<transform::UnfittedTargetTransform>&) const;
template std::expected<cxxgpr::detail::FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError>
LoadedGpr::extract_payload<MixedPrecision<ReevaluateKernel>>(
    std::unique_ptr<transform::UnfittedInputTransform>&,
    std::unique_ptr<transform::UnfittedTargetTransform>&) const;

namespace detail {

std::expected<void, GprError> save_model_impl(const SaveInput& input,
                                              const std::filesystem::path& dir,
                                              [[maybe_unused]] const PersistRegistry& registry) {
    if (input.n_train == 0U || input.n_features == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    auto cfg_str = serialize_config_json(input);
    if (!cfg_str.has_value()) {
        return std::unexpected(std::move(cfg_str).error());
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return std::unexpected(GprError::persist_failed(
            PersistErrorKind::Io, "failed to create target persistence directory"));
    }

    std::vector<TensorWriteView> tensors;
    tensors.push_back(TensorWriteView{
        .name = "x",
        .dtype = TensorDtype::F64,
        .shape = {input.n_train, input.n_features},
        .bytes = as_bytes(input.x_raw),
    });
    tensors.push_back(TensorWriteView{
        .name = "y",
        .dtype = TensorDtype::F64,
        .shape = {input.n_train},
        .bytes = as_bytes(input.y_raw),
    });
    if (input.has_factor) {
        if (!input.l_f64.empty()) {
            tensors.push_back(TensorWriteView{
                .name = "l",
                .dtype = TensorDtype::F64,
                .shape = {input.n_train, input.n_train},
                .bytes = as_bytes(input.l_f64),
            });
        } else {
            tensors.push_back(TensorWriteView{
                .name = "l",
                .dtype = TensorDtype::F32,
                .shape = {input.n_train, input.n_train},
                .bytes = as_bytes(input.l_f32),
            });
        }
        if (!input.alpha_f64.empty()) {
            tensors.push_back(TensorWriteView{
                .name = "alpha",
                .dtype = TensorDtype::F64,
                .shape = {input.n_train},
                .bytes = as_bytes(input.alpha_f64),
            });
        } else {
            tensors.push_back(TensorWriteView{
                .name = "alpha",
                .dtype = TensorDtype::F32,
                .shape = {input.n_train},
                .bytes = as_bytes(input.alpha_f32),
            });
        }
    }

    const std::filesystem::path tmp_st = dir / "model.safetensors.tmp";
    const std::filesystem::path tmp_cfg = dir / "config.json.tmp";

    if (auto wr = write_safetensors_file(tmp_st, tensors); !wr.has_value()) {
        std::filesystem::remove(tmp_st, ec);
        return wr;
    }

    {
        std::ofstream ofs(tmp_cfg, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            std::filesystem::remove(tmp_st, ec);
            return std::unexpected(GprError::persist_failed(
                PersistErrorKind::Io, "failed to open temporary config.json for writing"));
        }
        ofs.write(cfg_str->data(), static_cast<std::streamsize>(cfg_str->size()));
        ofs.flush();
        if (!ofs) {
            std::filesystem::remove(tmp_st, ec);
            std::filesystem::remove(tmp_cfg, ec);
            return std::unexpected(GprError::persist_failed(
                PersistErrorKind::Io, "failed while writing temporary config.json"));
        }
    }

    std::filesystem::rename(tmp_st, dir / "model.safetensors", ec);
    if (ec) {
        std::filesystem::remove(tmp_st, ec);
        std::filesystem::remove(tmp_cfg, ec);
        return std::unexpected(GprError::persist_failed(
            PersistErrorKind::Io, "failed to atomically replace model.safetensors"));
    }
    std::filesystem::rename(tmp_cfg, dir / "config.json", ec);
    if (ec) {
        std::filesystem::remove(tmp_cfg, ec);
        return std::unexpected(GprError::persist_failed(
            PersistErrorKind::Io, "failed to atomically replace config.json"));
    }

    return {};
}

} // namespace detail

} // namespace cxxgpr::persist
