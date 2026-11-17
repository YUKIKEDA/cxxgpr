// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/optimizer/optimizer.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/precision/refine.hpp"
#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"
#include "cxxgpr/workspace/buffers.hpp"
#include "cxxgpr/workspace/leaf_cache.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace cxxgpr {

/// Unfitted Exact Gaussian Process Regression trainer parameterized by optimizer, precision
/// policy, and kernel specification.
template <typename O, PrecisionPolicyConcept P, typename K> class Gpr;
/// Fitted Exact Gaussian Process Regression model supporting prediction, sampling, and LOO.
template <typename O, PrecisionPolicyConcept P, typename K> class FittedGpr;

namespace persist {
class LoadedGpr;
} // namespace persist

namespace detail {

template <PrecisionPolicyConcept P> struct FittedPayload {
    using Storage = typename P::Storage;
    using Refine = typename P::Refine;

    KernelSpec kernel_spec;
    GaussianLikelihood likelihood;
    std::shared_ptr<const CompiledKernel<Storage>> compiled_kernel;
    std::unique_ptr<transform::FittedInputTransform> input_transform;
    std::unique_ptr<transform::FittedTargetTransform> target_transform;
    std::vector<double> x_raw;
    std::vector<double> y_raw;
    std::vector<Storage> x_train;
    std::vector<Refine> y_transformed;
    std::vector<Storage> l_factor;
    std::vector<Refine> alpha;
    std::size_t n_train{0};
    std::size_t n_features{0};
    double factor_jitter{0.0};
    double log_marginal_likelihood{0.0};
    std::optional<RefinementSummary> refinement_summary;
    optimizer::OptimizerResult optimizer_result;
};

template <PrecisionPolicyConcept P> struct TrainConfig {
    const KernelSpec& kernel_spec;
    const GaussianLikelihood& likelihood;
    const JitterPolicy& jitter;
    const transform::UnfittedInputTransform& input_transform;
    const transform::UnfittedTargetTransform& target_transform;
    KernelExp kernel_exp;
    FitPolicy fit_policy;
    bool optimize;
    bool use_change_indices;
};

using OptimizeCallback = std::expected<optimizer::OptimizerResult, GprError> (*)(
    void* ctx, optimizer::TwiceDifferentiable& obj, std::span<const double> initial_params);

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<FittedPayload<P>, GprError>
train_or_factor_payload(const TrainConfig<P>& cfg, MatrixRef<double> x, std::span<const double> y,
                        OptimizeCallback optimize_cb, void* optimize_ctx);

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError>
fitted_predict_into(const FittedPayload<P>& payload, MatrixRef<double> xs,
                    const PredictOptions& options, QueryWorkspace<P>& workspace,
                    Prediction<typename P::Refine>& out);

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<PredictiveCovariance<typename P::Refine>, GprError>
fitted_predict_covariance(const FittedPayload<P>& payload, MatrixRef<double> xs,
                          const PredictOptions& options, QueryWorkspace<P>& workspace);

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<std::vector<typename P::Refine>, GprError>
fitted_sample(const FittedPayload<P>& payload, MatrixRef<double> xs, std::size_t n_draws,
              std::uint64_t seed, const PredictOptions& options, QueryWorkspace<P>& workspace,
              std::optional<JitterPolicy> jitter);

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError>
fitted_loo_predict(const FittedPayload<P>& payload, const PredictOptions& options);

template <typename O>
[[nodiscard]] std::expected<optimizer::OptimizerResult, GprError>
invoke_optimizer(const O& opt, optimizer::TwiceDifferentiable& obj,
                 std::span<const double> initial_params) {
    if constexpr (std::same_as<O, optimizer::Fixed>) {
        return opt.optimize(obj, initial_params);
    } else if constexpr (std::derived_from<O, optimizer::TrustRegion> ||
                         std::same_as<O, optimizer::TrustRegion>) {
        return opt.optimize(obj, initial_params);
    } else if constexpr (std::same_as<O, optimizer::Lbfgs>) {
        return opt.optimize(obj, initial_params);
    } else {
        return opt.optimize(obj, initial_params);
    }
}

extern template std::expected<FittedPayload<DoublePrecision>, GprError>
train_or_factor_payload<DoublePrecision>(const TrainConfig<DoublePrecision>&, MatrixRef<double>,
                                         std::span<const double>, OptimizeCallback, void*);
extern template std::expected<FittedPayload<SinglePrecision>, GprError>
train_or_factor_payload<SinglePrecision>(const TrainConfig<SinglePrecision>&, MatrixRef<double>,
                                         std::span<const double>, OptimizeCallback, void*);
extern template std::expected<FittedPayload<MixedPrecision<PromoteStorage>>, GprError>
train_or_factor_payload<MixedPrecision<PromoteStorage>>(
    const TrainConfig<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, std::span<const double>,
    OptimizeCallback, void*);
extern template std::expected<FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError>
train_or_factor_payload<MixedPrecision<ReevaluateKernel>>(
    const TrainConfig<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>,
    std::span<const double>, OptimizeCallback, void*);

extern template std::expected<void, GprError>
fitted_predict_into<DoublePrecision>(const FittedPayload<DoublePrecision>&, MatrixRef<double>,
                                     const PredictOptions&, QueryWorkspace<DoublePrecision>&,
                                     Prediction<double>&);
extern template std::expected<void, GprError>
fitted_predict_into<SinglePrecision>(const FittedPayload<SinglePrecision>&, MatrixRef<double>,
                                     const PredictOptions&, QueryWorkspace<SinglePrecision>&,
                                     Prediction<float>&);
extern template std::expected<void, GprError> fitted_predict_into<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, Prediction<double>&);
extern template std::expected<void, GprError> fitted_predict_into<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>,
    const PredictOptions&, QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, Prediction<double>&);

extern template std::expected<PredictiveCovariance<double>, GprError>
fitted_predict_covariance<DoublePrecision>(const FittedPayload<DoublePrecision>&, MatrixRef<double>,
                                           const PredictOptions&, QueryWorkspace<DoublePrecision>&);
extern template std::expected<PredictiveCovariance<float>, GprError>
fitted_predict_covariance<SinglePrecision>(const FittedPayload<SinglePrecision>&, MatrixRef<double>,
                                           const PredictOptions&, QueryWorkspace<SinglePrecision>&);
extern template std::expected<PredictiveCovariance<double>, GprError>
fitted_predict_covariance<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&);
extern template std::expected<PredictiveCovariance<double>, GprError>
fitted_predict_covariance<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>,
    const PredictOptions&, QueryWorkspace<MixedPrecision<ReevaluateKernel>>&);

extern template std::expected<std::vector<double>, GprError>
fitted_sample<DoublePrecision>(const FittedPayload<DoublePrecision>&, MatrixRef<double>,
                               std::size_t, std::uint64_t, const PredictOptions&,
                               QueryWorkspace<DoublePrecision>&, std::optional<JitterPolicy>);
extern template std::expected<std::vector<float>, GprError>
fitted_sample<SinglePrecision>(const FittedPayload<SinglePrecision>&, MatrixRef<double>,
                               std::size_t, std::uint64_t, const PredictOptions&,
                               QueryWorkspace<SinglePrecision>&, std::optional<JitterPolicy>);
extern template std::expected<std::vector<double>, GprError>
fitted_sample<MixedPrecision<PromoteStorage>>(const FittedPayload<MixedPrecision<PromoteStorage>>&,
                                              MatrixRef<double>, std::size_t, std::uint64_t,
                                              const PredictOptions&,
                                              QueryWorkspace<MixedPrecision<PromoteStorage>>&,
                                              std::optional<JitterPolicy>);
extern template std::expected<std::vector<double>, GprError>
fitted_sample<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>, std::size_t,
    std::uint64_t, const PredictOptions&, QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
    std::optional<JitterPolicy>);

extern template std::expected<Prediction<double>, GprError>
fitted_loo_predict<DoublePrecision>(const FittedPayload<DoublePrecision>&, const PredictOptions&);
extern template std::expected<Prediction<float>, GprError>
fitted_loo_predict<SinglePrecision>(const FittedPayload<SinglePrecision>&, const PredictOptions&);
extern template std::expected<Prediction<double>, GprError>
fitted_loo_predict<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, const PredictOptions&);
extern template std::expected<Prediction<double>, GprError>
fitted_loo_predict<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, const PredictOptions&);

} // namespace detail

/// Unfitted Exact Gaussian Process Regression trainer parameterized by optimizer O, precision
/// policy P, and kernel specification K. Does not expose prediction or LOO operations until fitted.
template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision,
          typename K = KernelSpec>
class Gpr {
    static_assert(std::same_as<K, KernelSpec>, "Gpr<O, P, K> currently supports K = KernelSpec");

public:
    /// Constructs an unfitted GPR trainer from a kernel specification and optional optimizer.
    explicit Gpr(KernelSpec kernel_spec, O optimizer = O{})
        : kernel_spec_(std::move(kernel_spec)), optimizer_(std::move(optimizer)),
          likelihood_(GaussianLikelihood::create(1e-2).value()),
          input_transform_(std::make_unique<transform::IdentityInput>()),
          target_transform_(std::make_unique<transform::IdentityTarget>()) {}

    /// Copy-constructs a trainer cloning its polymorphic transform configurations.
    Gpr(const Gpr& other)
        : kernel_spec_(other.kernel_spec_), optimizer_(other.optimizer_),
          likelihood_(other.likelihood_), jitter_(other.jitter_),
          input_transform_(other.input_transform_->clone()),
          target_transform_(other.target_transform_->clone()), kernel_exp_(other.kernel_exp_),
          fit_policy_(other.fit_policy_) {}

    /// Copy-assigns a trainer cloning its polymorphic transform configurations.
    Gpr& operator=(const Gpr& other) {
        if (this != &other) {
            Gpr copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    Gpr(Gpr&&) noexcept = default;
    Gpr& operator=(Gpr&&) noexcept = default;
    ~Gpr() = default;

    /// Returns a copy with an updated optimizer configuration.
    [[nodiscard]] Gpr with_optimizer(O optimizer) const {
        Gpr copy(*this);
        copy.optimizer_ = std::move(optimizer);
        return copy;
    }

    /// Returns a copy with an updated Gaussian observation noise likelihood.
    [[nodiscard]] Gpr with_likelihood(GaussianLikelihood likelihood) const {
        Gpr copy(*this);
        copy.likelihood_ = likelihood;
        return copy;
    }

    /// Returns a copy with an updated observation noise variance after validating positivity.
    [[nodiscard]] std::expected<Gpr, GprError> with_noise_variance(double noise_variance) const {
        auto lik = GaussianLikelihood::create(noise_variance);
        if (!lik.has_value()) {
            return std::unexpected(std::move(lik).error());
        }
        return with_likelihood(*lik);
    }

    /// Returns a copy with an updated Cholesky jitter stabilization policy.
    [[nodiscard]] Gpr with_jitter(JitterPolicy jitter) const {
        Gpr copy(*this);
        copy.jitter_ = jitter;
        return copy;
    }

    /// Returns a copy with an updated unfitted input feature transform.
    [[nodiscard]] Gpr
    with_input_transform(const transform::UnfittedInputTransform& input_transform) const {
        Gpr copy(*this);
        copy.input_transform_ = input_transform.clone();
        return copy;
    }

    /// Returns a copy with an updated unfitted target transform.
    [[nodiscard]] Gpr
    with_target_transform(const transform::UnfittedTargetTransform& target_transform) const {
        Gpr copy(*this);
        copy.target_transform_ = target_transform.clone();
        return copy;
    }

    /// Returns a copy with an updated exponential evaluation backend.
    [[nodiscard]] Gpr with_kernel_exp(KernelExp mode) const {
        Gpr copy(*this);
        copy.kernel_exp_ = mode;
        return copy;
    }

    /// Returns a copy with an updated distance cache retention policy.
    [[nodiscard]] Gpr with_distance_cache_policy(DistanceCachePolicy policy) const {
        Gpr copy(*this);
        copy.fit_policy_.distance_cache = policy;
        return copy;
    }

    /// Returns a copy with an updated Cholesky buffer retention policy.
    [[nodiscard]] Gpr with_cholesky_buffer(CholeskyBuffer policy) const {
        Gpr copy(*this);
        copy.fit_policy_.cholesky_buffer = policy;
        return copy;
    }

    /// Returns a copy configured for minimum memory footprint (Uncached + Reuse).
    [[nodiscard]] Gpr with_prefer_memory() const {
        Gpr copy(*this);
        copy.fit_policy_ = FitPolicy::prefer_memory();
        return copy;
    }

    /// Returns a copy configured for maximum optimization speed (Cached + Retain).
    [[nodiscard]] Gpr with_prefer_speed() const {
        Gpr copy(*this);
        copy.fit_policy_ = FitPolicy::prefer_speed();
        return copy;
    }

    /// Returns the current kernel specification.
    [[nodiscard]] const KernelSpec& kernel_spec() const noexcept {
        return kernel_spec_;
    }

    /// Returns the current optimizer configuration.
    [[nodiscard]] const O& optimizer() const noexcept {
        return optimizer_;
    }

    /// Returns the current Gaussian observation noise likelihood.
    [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept {
        return likelihood_;
    }

    /// Returns the current Cholesky jitter policy.
    [[nodiscard]] const JitterPolicy& jitter() const noexcept {
        return jitter_;
    }

    /// Returns the current exponential evaluation backend.
    [[nodiscard]] KernelExp kernel_exp() const noexcept {
        return kernel_exp_;
    }

    /// Returns the current fit buffer policy.
    [[nodiscard]] FitPolicy fit_policy() const noexcept {
        return fit_policy_;
    }

    /// Fits transforms and optimizes hyperparameters on column-major training data (x, y), leaving
    /// *this unchanged on failure.
    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> fit(MatrixRef<double> x,
                                                                  std::span<const double> y) const {
        return run_impl(x, y, /*optimize=*/true);
    }

    /// Fits transforms and optimizes hyperparameters from a raw column-major feature span.
    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>
    fit(std::span<const double> x_col_major, std::size_t n_rows, std::size_t n_cols,
        std::span<const double> y) const {
        return fit(MatrixRef<double>(x_col_major.data(), n_rows, n_cols, n_rows), y);
    }

    /// Fits transforms and factors the exact system at the current hyperparameters without running
    /// optimization iterations, leaving *this unchanged on failure.
    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>
    factor(MatrixRef<double> x, std::span<const double> y) const {
        return run_impl(x, y, /*optimize=*/false);
    }

    /// Fits transforms and factors the exact system from a raw column-major feature span.
    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>
    factor(std::span<const double> x_col_major, std::size_t n_rows, std::size_t n_cols,
           std::span<const double> y) const {
        return factor(MatrixRef<double>(x_col_major.data(), n_rows, n_cols, n_rows), y);
    }

private:
    friend class FittedGpr<O, P, K>;

    Gpr(KernelSpec kernel_spec, O optimizer, GaussianLikelihood likelihood, JitterPolicy jitter,
        std::unique_ptr<transform::UnfittedInputTransform> input_transform,
        std::unique_ptr<transform::UnfittedTargetTransform> target_transform, KernelExp kernel_exp,
        FitPolicy fit_policy)
        : kernel_spec_(std::move(kernel_spec)), optimizer_(std::move(optimizer)),
          likelihood_(likelihood), jitter_(jitter), input_transform_(std::move(input_transform)),
          target_transform_(std::move(target_transform)), kernel_exp_(kernel_exp),
          fit_policy_(fit_policy) {}

    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>
    run_impl(MatrixRef<double> x, std::span<const double> y, bool optimize) const {
        const detail::TrainConfig<P> cfg{
            .kernel_spec = kernel_spec_,
            .likelihood = likelihood_,
            .jitter = jitter_,
            .input_transform = *input_transform_,
            .target_transform = *target_transform_,
            .kernel_exp = kernel_exp_,
            .fit_policy = fit_policy_,
            .optimize = optimize && !std::same_as<O, optimizer::Fixed>,
            .use_change_indices = optimizer_uses_change_indices_v<O>,
        };
        auto cb = [](void* ctx, optimizer::TwiceDifferentiable& obj, std::span<const double> init)
            -> std::expected<optimizer::OptimizerResult, GprError> {
            return detail::invoke_optimizer(*static_cast<const O*>(ctx), obj, init);
        };
        auto payload =
            detail::train_or_factor_payload<P>(cfg, x, y, cb, const_cast<O*>(&optimizer_));
        if (!payload.has_value()) {
            return std::unexpected(std::move(payload).error());
        }
        return FittedGpr<O, P, K>(std::move(*payload), optimizer_, jitter_,
                                  input_transform_->clone(), target_transform_->clone(),
                                  kernel_exp_, fit_policy_);
    }

    KernelSpec kernel_spec_;
    O optimizer_;
    GaussianLikelihood likelihood_;
    JitterPolicy jitter_;
    std::unique_ptr<transform::UnfittedInputTransform> input_transform_;
    std::unique_ptr<transform::UnfittedTargetTransform> target_transform_;
    KernelExp kernel_exp_{KernelExp::Accurate};
    FitPolicy fit_policy_{FitPolicy::prefer_speed()};
};

/// Fitted Exact Gaussian Process Regression model providing posterior prediction, zero-allocation
/// predict_into, full posterior covariance, posterior sampling, and closed-form LOO inference.
template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision,
          typename K = KernelSpec>
class FittedGpr {
public:
    /// Storage scalar type used for training features and Cholesky factors.
    using Storage = typename P::Storage;
    /// Output and weight scalar type (double for DoublePrecision and MixedPrecision).
    using Refine = typename P::Refine;

    /// Copy-constructs a fitted model with an independent QueryWorkspace.
    FittedGpr(const FittedGpr& other)
        : payload_{
              .kernel_spec = other.payload_.kernel_spec,
              .likelihood = other.payload_.likelihood,
              .compiled_kernel = other.payload_.compiled_kernel,
              .input_transform = other.payload_.input_transform->clone(),
              .target_transform = other.payload_.target_transform->clone(),
              .x_raw = other.payload_.x_raw,
              .y_raw = other.payload_.y_raw,
              .x_train = other.payload_.x_train,
              .y_transformed = other.payload_.y_transformed,
              .l_factor = other.payload_.l_factor,
              .alpha = other.payload_.alpha,
              .n_train = other.payload_.n_train,
              .n_features = other.payload_.n_features,
              .factor_jitter = other.payload_.factor_jitter,
              .log_marginal_likelihood = other.payload_.log_marginal_likelihood,
              .refinement_summary = other.payload_.refinement_summary,
              .optimizer_result = other.payload_.optimizer_result,
          },
          optimizer_(other.optimizer_), jitter_(other.jitter_),
          unfitted_input_(other.unfitted_input_->clone()),
          unfitted_target_(other.unfitted_target_->clone()), kernel_exp_(other.kernel_exp_),
          fit_policy_(other.fit_policy_) {}

    /// Copy-assigns a fitted model with an independent QueryWorkspace.
    FittedGpr& operator=(const FittedGpr& other) {
        if (this != &other) {
            FittedGpr copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    FittedGpr(FittedGpr&&) noexcept = default;
    FittedGpr& operator=(FittedGpr&&) noexcept = default;
    ~FittedGpr() = default;

    /// Computes posterior mean and diagonal variance on the original target scale.
    [[nodiscard]] std::expected<Prediction<Refine>, GprError>
    predict(MatrixRef<double> xs, const PredictOptions& options = {}) const {
        return predict_with(xs, options);
    }

    /// Computes posterior mean and diagonal variance from a raw column-major query span.
    [[nodiscard]] std::expected<Prediction<Refine>, GprError>
    predict(std::span<const double> xs, std::size_t n_rows, std::size_t n_cols,
            const PredictOptions& options = {}) const {
        return predict_with(MatrixRef<double>(xs.data(), n_rows, n_cols, n_rows), options);
    }

    /// Computes posterior mean and diagonal variance with explicit PredictOptions.
    [[nodiscard]] std::expected<Prediction<Refine>, GprError>
    predict_with(MatrixRef<double> xs, const PredictOptions& options) const {
        Prediction<Refine> out;
        if (auto res = predict_with_into(xs, options, out); !res.has_value()) {
            return std::unexpected(std::move(res).error());
        }
        return out;
    }

    /// Computes posterior mean and diagonal variance into out (zero heap allocations after warmup
    /// on matching query shape).
    [[nodiscard]] std::expected<void, GprError>
    predict_into(MatrixRef<double> xs, Prediction<Refine>& out,
                 const PredictOptions& options = {}) const {
        return predict_with_into(xs, options, out);
    }

    /// Computes posterior mean and diagonal variance into out from a raw column-major query span.
    [[nodiscard]] std::expected<void, GprError>
    predict_into(std::span<const double> xs, std::size_t n_rows, std::size_t n_cols,
                 Prediction<Refine>& out, const PredictOptions& options = {}) const {
        return predict_with_into(MatrixRef<double>(xs.data(), n_rows, n_cols, n_rows), options,
                                 out);
    }

    /// Computes posterior mean and diagonal variance into out with explicit PredictOptions.
    [[nodiscard]] std::expected<void, GprError> predict_with_into(MatrixRef<double> xs,
                                                                  const PredictOptions& options,
                                                                  Prediction<Refine>& out) const {
        return detail::fitted_predict_into<P>(payload_, xs, options, query_ws_, out);
    }

    /// Computes posterior mean and full m x m posterior covariance matrix on the original scale.
    [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError>
    predict_covariance(MatrixRef<double> xs, const PredictOptions& options = {}) const {
        return predict_covariance_with(xs, options);
    }

    /// Computes posterior mean and full m x m posterior covariance from a raw column-major span.
    [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError>
    predict_covariance(std::span<const double> xs, std::size_t n_rows, std::size_t n_cols,
                       const PredictOptions& options = {}) const {
        return predict_covariance_with(MatrixRef<double>(xs.data(), n_rows, n_cols, n_rows),
                                       options);
    }

    /// Computes posterior mean and full m x m posterior covariance with explicit PredictOptions.
    [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError>
    predict_covariance_with(MatrixRef<double> xs, const PredictOptions& options) const {
        return detail::fitted_predict_covariance<P>(payload_, xs, options, query_ws_);
    }

    /// Draws n_draws deterministic posterior samples on the original target scale.
    [[nodiscard]] std::expected<std::vector<Refine>, GprError>
    sample(MatrixRef<double> xs, std::size_t n_draws, std::uint64_t seed,
           const PredictOptions& options = {},
           std::optional<JitterPolicy> jitter = std::nullopt) const {
        return sample_with(xs, n_draws, seed, options, std::move(jitter));
    }

    /// Draws n_draws deterministic posterior samples from a raw column-major query span.
    [[nodiscard]] std::expected<std::vector<Refine>, GprError>
    sample(std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, std::size_t n_draws,
           std::uint64_t seed, const PredictOptions& options = {},
           std::optional<JitterPolicy> jitter = std::nullopt) const {
        return sample_with(MatrixRef<double>(xs.data(), n_rows, n_cols, n_rows), n_draws, seed,
                           options, std::move(jitter));
    }

    /// Draws n_draws deterministic posterior samples with explicit PredictOptions.
    [[nodiscard]] std::expected<std::vector<Refine>, GprError>
    sample_with(MatrixRef<double> xs, std::size_t n_draws, std::uint64_t seed,
                const PredictOptions& options,
                std::optional<JitterPolicy> jitter = std::nullopt) const {
        return detail::fitted_sample<P>(payload_, xs, n_draws, seed, options, query_ws_,
                                        std::move(jitter));
    }

    /// Computes closed-form Leave-One-Out predictive mean and variance on the original target
    /// scale (GPML §5.4.2).
    [[nodiscard]] std::expected<Prediction<Refine>, GprError>
    loo_predict(const PredictOptions& options = {}) const {
        return loo_predict_with(options);
    }

    /// Computes closed-form Leave-One-Out predictive mean and variance with explicit
    /// PredictOptions.
    [[nodiscard]] std::expected<Prediction<Refine>, GprError>
    loo_predict_with(const PredictOptions& options) const {
        return detail::fitted_loo_predict<P>(payload_, options);
    }

    /// Converts this fitted model back into an unfitted Gpr trainer initialized at the fitted
    /// kernel hyperparameters, noise likelihood, transforms, and policy settings.
    [[nodiscard]] Gpr<O, P, K> into_trainer() const {
        return Gpr<O, P, K>(payload_.kernel_spec, optimizer_, payload_.likelihood, jitter_,
                            unfitted_input_->clone(), unfitted_target_->clone(), kernel_exp_,
                            fit_policy_);
    }

    /// Re-optimizes a new model on (x, y) starting from this fitted model's hyperparameters and
    /// configuration.
    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>
    refit(MatrixRef<double> x, std::span<const double> y) const {
        return into_trainer().fit(x, y);
    }

    /// Re-optimizes a new model from a raw column-major feature span.
    [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>
    refit(std::span<const double> x_col_major, std::size_t n_rows, std::size_t n_cols,
          std::span<const double> y) const {
        return refit(MatrixRef<double>(x_col_major.data(), n_rows, n_cols, n_rows), y);
    }

    /// Returns the fitted kernel specification with updated hyperparameters.
    [[nodiscard]] const KernelSpec& kernel_spec() const noexcept {
        return payload_.kernel_spec;
    }

    /// Returns the fitted Gaussian observation noise likelihood.
    [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept {
        return payload_.likelihood;
    }

    /// Returns the number of training samples n.
    [[nodiscard]] std::size_t n_train() const noexcept {
        return payload_.n_train;
    }

    /// Returns the training feature dimension d.
    [[nodiscard]] std::size_t n_features() const noexcept {
        return payload_.n_features;
    }

    /// Returns the accepted diagonal jitter offset added during final Cholesky factorization.
    [[nodiscard]] double factor_jitter() const noexcept {
        return payload_.factor_jitter;
    }

    /// Returns the negative marginal log-likelihood at the final fitted parameters.
    [[nodiscard]] double log_marginal_likelihood() const noexcept {
        return payload_.log_marginal_likelihood;
    }

    /// Returns the refined prediction weight vector alpha = A^-1 y_transformed.
    [[nodiscard]] std::span<const Refine> alpha() const noexcept {
        return payload_.alpha;
    }

    /// Returns the lower-triangular Cholesky factor L of A = K + (sigma_n^2 + j) I.
    [[nodiscard]] MatrixRef<Storage> l_factor() const noexcept {
        return {payload_.l_factor.data(), payload_.n_train, payload_.n_train, payload_.n_train};
    }

    /// Returns the fitted input transform.
    [[nodiscard]] const transform::FittedInputTransform& input_transform() const noexcept {
        return *payload_.input_transform;
    }

    /// Returns the fitted target transform.
    [[nodiscard]] const transform::FittedTargetTransform& target_transform() const noexcept {
        return *payload_.target_transform;
    }

    /// Returns the iterative refinement summary when P is MixedPrecision, or std::nullopt.
    [[nodiscard]] const std::optional<RefinementSummary>& refinement_summary() const noexcept {
        return payload_.refinement_summary;
    }

    /// Returns the optimizer convergence summary.
    [[nodiscard]] const optimizer::OptimizerResult& optimizer_result() const noexcept {
        return payload_.optimizer_result;
    }

    /// Returns the exponential evaluation backend.
    [[nodiscard]] KernelExp kernel_exp() const noexcept {
        return kernel_exp_;
    }

    /// Returns the fit buffer policy.
    [[nodiscard]] FitPolicy fit_policy() const noexcept {
        return fit_policy_;
    }

    /// Returns the Cholesky jitter policy.
    [[nodiscard]] const JitterPolicy& jitter() const noexcept {
        return jitter_;
    }

    /// Returns the raw (pre-transform) column-major training features in double precision.
    [[nodiscard]] std::span<const double> x_raw() const noexcept {
        return payload_.x_raw;
    }

    /// Returns the raw (pre-transform) training targets in double precision.
    [[nodiscard]] std::span<const double> y_raw() const noexcept {
        return payload_.y_raw;
    }

    /// Returns the unfitted input transform configuration retained for persistence and refit.
    [[nodiscard]] const transform::UnfittedInputTransform&
    unfitted_input_transform() const noexcept {
        return *unfitted_input_;
    }

    /// Returns the unfitted target transform configuration retained for persistence and refit.
    [[nodiscard]] const transform::UnfittedTargetTransform&
    unfitted_target_transform() const noexcept {
        return *unfitted_target_;
    }

private:
    friend class Gpr<O, P, K>;
    friend class persist::LoadedGpr;

    FittedGpr(detail::FittedPayload<P> payload, O optimizer, JitterPolicy jitter,
              std::unique_ptr<transform::UnfittedInputTransform> unfitted_input,
              std::unique_ptr<transform::UnfittedTargetTransform> unfitted_target,
              KernelExp kernel_exp, FitPolicy fit_policy)
        : payload_(std::move(payload)), optimizer_(std::move(optimizer)), jitter_(jitter),
          unfitted_input_(std::move(unfitted_input)), unfitted_target_(std::move(unfitted_target)),
          kernel_exp_(kernel_exp), fit_policy_(fit_policy) {}

    detail::FittedPayload<P> payload_;
    O optimizer_;
    JitterPolicy jitter_;
    std::unique_ptr<transform::UnfittedInputTransform> unfitted_input_;
    std::unique_ptr<transform::UnfittedTargetTransform> unfitted_target_;
    KernelExp kernel_exp_;
    FitPolicy fit_policy_;
    mutable QueryWorkspace<P> query_ws_;
};

} // namespace cxxgpr
