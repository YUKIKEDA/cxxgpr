// SPDX-License-Identifier: Apache-2.0
/// @file gpr.hpp
/// @brief Unfitted Exact GPR trainer (Gpr) and fitted posterior inference model (FittedGpr).
/// @details Separates pre-training configuration (Gpr) from post-training inference (FittedGpr) at the
/// type level, exposing immutable builder methods, non-throwing std::expected fit/factor/predict/sample/LOO
/// operations, and internal payload orchestration.

#pragma once

#include "cxxgpr/core/checked.hpp"
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

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Unfitted Exact Gaussian Process Regression trainer parameterized by optimizer, precision
    /// policy, and kernel specification.
    /// @details Forward declaration enabling friend declarations and cross-references with FittedGpr.
    /// @tparam O Optimizer policy type (e.g., optimizer::Lbfgs, optimizer::TrustRegion, optimizer::Fixed).
    /// @tparam P Precision policy satisfying PrecisionPolicyConcept.
    /// @tparam K Kernel specification type (currently KernelSpec).
    template <typename O, PrecisionPolicyConcept P, typename K> class Gpr;

    /// @brief Fitted Exact Gaussian Process Regression model supporting prediction, sampling, and LOO.
    /// @details Forward declaration enabling Gpr::fit(), Gpr::factor(), and persistence interoperability.
    /// @tparam O Optimizer policy type retained for refit() and into_trainer().
    /// @tparam P Precision policy satisfying PrecisionPolicyConcept.
    /// @tparam K Kernel specification type (currently KernelSpec).
    template <typename O, PrecisionPolicyConcept P, typename K> class FittedGpr;

    /// @brief Persistence API namespace containing LoadedGpr for loading saved Exact GPR models.
    namespace persist
    {
        /// @brief Forward declaration of type-erased disk-loaded Exact GPR model.
        class LoadedGpr;
    }  // namespace persist

    /// @brief Internal implementation helpers decoupling public Gpr/FittedGpr templates from gpr_engine.cpp.
    namespace detail
    {

        /// @brief Type-erased training artifacts transferred into FittedGpr upon successful fit() or factor().
        /// @details Owns fitted hyperparameters, compiled kernel, fitted transforms, raw and transformed
        /// training data, Cholesky factor L, refined prediction weights alpha, and training diagnostics.
        /// @tparam P Precision policy determining Storage and Refine scalar types.
        template <PrecisionPolicyConcept P> struct FittedPayload
        {
            /// @brief Storage scalar type for training features and Cholesky factor L.
            using Storage = typename P::Storage;
            /// @brief Refinement and output scalar type for transformed targets and weights alpha.
            using Refine = typename P::Refine;

            /// @brief Kernel specification updated with final optimized hyperparameters.
            KernelSpec kernel_spec;
            /// @brief Gaussian observation noise likelihood updated with final optimized variance.
            GaussianLikelihood likelihood;
            /// @brief Compiled kernel evaluator matching Storage precision and kernel_exp mode.
            std::shared_ptr<const CompiledKernel<Storage>> compiled_kernel;
            /// @brief Fitted input transform applied to training and query feature matrices.
            std::unique_ptr<transform::FittedInputTransform> input_transform;
            /// @brief Fitted target transform used for inverse prediction mean/variance/covariance scaling.
            std::unique_ptr<transform::FittedTargetTransform> target_transform;
            /// @brief Raw pre-transform column-major training features in double precision (n_train * n_features).
            std::vector<double> x_raw;
            /// @brief Raw pre-transform training targets in double precision (n_train).
            std::vector<double> y_raw;
            /// @brief Transformed column-major training features cast to Storage precision (n_train * n_features).
            std::vector<Storage> x_train;
            /// @brief Transformed training targets cast to Refine precision (n_train).
            std::vector<Refine> y_transformed;
            /// @brief Column-major lower-triangular Cholesky factor L of K + (sigma_n^2 + j) I (n_train * n_train).
            std::vector<Storage> l_factor;
            /// @brief Linear solve weight vector alpha = A^-1 y_transformed in Refine precision (n_train).
            std::vector<Refine> alpha;
            /// @brief Number of training samples n.
            std::size_t n_train{0};
            /// @brief Number of training feature columns d.
            std::size_t n_features{0};
            /// @brief Accepted diagonal jitter offset added during final Cholesky factorization.
            double factor_jitter{0.0};
            /// @brief Negative log marginal likelihood evaluated at the final fitted parameters.
            double neg_log_marginal_likelihood{0.0};
            /// @brief Mixed-precision iterative refinement summary, or std::nullopt when P does not refine in f64.
            std::optional<RefinementSummary> refinement_summary;
            /// @brief Optimizer convergence trajectory and final parameter diagnostics.
            optimizer::OptimizerResult optimizer_result;
        };

        /// @brief Immutable configuration bundle passed from Gpr::run_impl() into train_or_factor_payload().
        /// @details Aggregates borrowed trainer policies, transform prototypes, and optimization flags so
        /// training execution stays out-of-line in gpr_engine.cpp.
        /// @tparam P Precision policy controlling storage and refinement execution paths.
        template <PrecisionPolicyConcept P> struct TrainConfig
        {
            /// @brief Initial kernel specification and parameter bounds.
            const KernelSpec& kernel_spec;
            /// @brief Initial Gaussian observation noise likelihood and bounds.
            const GaussianLikelihood& likelihood;
            /// @brief Cholesky stabilization jitter policy.
            const JitterPolicy& jitter;
            /// @brief Unfitted input transform cloned and fitted to training features.
            const transform::UnfittedInputTransform& input_transform;
            /// @brief Unfitted target transform cloned and fitted to training targets.
            const transform::UnfittedTargetTransform& target_transform;
            /// @brief Exponential evaluation backend mode (Accurate or FastApprox).
            KernelExp kernel_exp;
            /// @brief Training workspace distance-cache and Cholesky-buffer retention policy.
            FitPolicy fit_policy;
            /// @brief True to invoke the hyperparameter optimizer; false for single-shot factorization.
            bool optimize;
            /// @brief True if the optimizer utilizes incremental LeafCache updates via changed_indices.
            bool use_change_indices;
            /// @brief True for built-in optimizers that only request objective values.
            bool value_only;
        };

        /// @brief Type-erased callback invoking a concrete optimizer on TwiceDifferentiable MLL objective.
        using OptimizeCallback = std::expected<optimizer::OptimizerResult, GprError> (*)(
            void* ctx, optimizer::TwiceDifferentiable& obj, std::span<const double> initial_params
        );

        /// @brief Validates training inputs, fits transforms, optionally optimizes hyperparameters, and builds
        /// FittedPayload.
        /// @details Instantiated explicitly in gpr_engine.cpp for all four supported precision policies.
        /// @tparam P Precision policy determining storage, MLL, and refinement arithmetic.
        /// @param[in] cfg Borrowed trainer configuration and transform prototypes.
        /// @param[in] x Column-major training feature matrix (n x d).
        /// @param[in] y Training target vector of length n.
        /// @param[in] optimize_cb Callback bridging to the concrete optimizer type O.
        /// @param[in,out] optimize_ctx Opaque pointer to the concrete optimizer instance.
        /// @return Constructed FittedPayload<P> on success, or GprError on validation, optimization, or Cholesky
        /// failure.
        /// @retval GprErrorCode::EmptyInput When x or y is empty or null.
        /// @retval GprErrorCode::LengthMismatch When y.size() != x.rows().
        /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When matrix factorization fails after exhausting jitter retries.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<FittedPayload<P>, GprError> train_or_factor_payload(
            const TrainConfig<P>& cfg,
            MatrixRef<double> x,
            std::span<const double> y,
            OptimizeCallback optimize_cb,
            void* optimize_ctx
        );

        /// @brief Validates query matrix xs, runs posterior prediction into out, and applies inverse target scaling.
        /// @details Reuses workspace buffers to avoid heap allocations once warmed up on a batch at least as large.
        /// @tparam P Precision policy matching payload.
        /// @param[in] payload Fitted model payload containing factor, weights, and transforms.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] options Prediction variance options.
        /// @param[in,out] workspace Mutable query workspace resized if needed.
        /// @param[out] out Destination prediction buffers populated on the original target scale.
        /// @return Empty expected on success, or GprError on input or transform error.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> fitted_predict_into(
            const FittedPayload<P>& payload,
            MatrixRef<double> xs,
            const PredictOptions& options,
            QueryWorkspace<P>& workspace,
            Prediction<typename P::Refine>& out
        );

        /// @brief Computes posterior mean and full m x m covariance matrix on the original target scale.
        /// @details Transforms query points, evaluates posterior covariance via predict_covariance<P>(),
        /// and applies inverse target scaling to both mean and covariance.
        /// @tparam P Precision policy matching payload.
        /// @param[in] payload Fitted model payload.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] options Prediction variance options.
        /// @param[in,out] workspace Mutable query workspace reused for query transform and kernel blocks.
        /// @return PredictiveCovariance<Refine> on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<PredictiveCovariance<typename P::Refine>, GprError> fitted_predict_covariance(
            const FittedPayload<P>& payload,
            MatrixRef<double> xs,
            const PredictOptions& options,
            QueryWorkspace<P>& workspace
        );

        /// @brief Draws deterministic posterior samples at xs and scales them back to the original target space.
        /// @details Uses seed and optional jitter override for predictive covariance Cholesky decomposition.
        /// @tparam P Precision policy matching payload.
        /// @param[in] payload Fitted model payload.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] n_draws Number of posterior draws to generate.
        /// @param[in] seed Deterministic PRNG seed.
        /// @param[in] options Prediction variance options.
        /// @param[in,out] workspace Mutable query workspace.
        /// @param[in] jitter Optional predictive covariance jitter override.
        /// @return Column-major sample buffer (m x n_draws) on the original scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When predictive covariance factorization fails.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<std::vector<typename P::Refine>, GprError> fitted_sample(
            const FittedPayload<P>& payload,
            MatrixRef<double> xs,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options,
            QueryWorkspace<P>& workspace,
            std::optional<JitterPolicy> jitter
        );

        /// @brief Computes closed-form Leave-One-Out prediction on the original target scale.
        /// @details Evaluates GPML analytical LOO moments in transformed space and applies inverse target transforms.
        /// @tparam P Precision policy matching payload.
        /// @param[in] payload Fitted model payload.
        /// @param[in] options Prediction variance options.
        /// @return Prediction<Refine> containing LOO means and variances, or GprError on failure.
        /// @retval GprErrorCode::NonFiniteKernelValue When inverse scaling yields non-finite values.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError> fitted_loo_predict(
            const FittedPayload<P>& payload, const PredictOptions& options
        );

        /// @brief Dispatches hyperparameter optimization to the concrete optimizer policy O.
        /// @details Keeps all optimizer call sites uniform while preserving static dispatch per optimizer type.
        /// @tparam O Optimizer type providing optimize(TwiceDifferentiable&, std::span<const double>).
        /// @param[in] opt Configured optimizer instance.
        /// @param[in,out] obj Twice-differentiable negative marginal log-likelihood objective.
        /// @param[in] initial_params Starting unconstrained log-space hyperparameter vector.
        /// @return OptimizerResult with final parameters and convergence flags, or GprError on failure.
        /// @note Failing to converge is not an error: the result is returned with converged == false. The built-in
        /// optimizers report only invalid configuration or objective failures through GprError.
        /// @note A GprError returned by a user-defined optimizer, such as GprErrorCode::OptimizationNotConverged, is
        /// forwarded unchanged.
        template <typename O>
        [[nodiscard]] std::expected<optimizer::OptimizerResult, GprError> invoke_optimizer(
            const O& opt, optimizer::TwiceDifferentiable& obj, std::span<const double> initial_params
        )
        {
            // Dispatch statically across built-in and custom optimizer types matching the concept contract.
            if constexpr (std::same_as<O, optimizer::Fixed>)
            {
                return opt.optimize(obj, initial_params);
            }
            else if constexpr (std::derived_from<O, optimizer::TrustRegion> || std::same_as<O, optimizer::TrustRegion>)
            {
                return opt.optimize(obj, initial_params);
            }
            else if constexpr (std::same_as<O, optimizer::Lbfgs>)
            {
                return opt.optimize(obj, initial_params);
            }
            else
            {
                return opt.optimize(obj, initial_params);
            }
        }

        extern template std::expected<FittedPayload<DoublePrecision>, GprError> train_or_factor_payload<
            DoublePrecision>(
            const TrainConfig<DoublePrecision>&, MatrixRef<double>, std::span<const double>, OptimizeCallback, void*
        );
        extern template std::expected<FittedPayload<SinglePrecision>, GprError> train_or_factor_payload<
            SinglePrecision>(
            const TrainConfig<SinglePrecision>&, MatrixRef<double>, std::span<const double>, OptimizeCallback, void*
        );
        extern template std::expected<FittedPayload<MixedPrecision<PromoteStorage>>, GprError> train_or_factor_payload<
            MixedPrecision<PromoteStorage>>(
            const TrainConfig<MixedPrecision<PromoteStorage>>&,
            MatrixRef<double>,
            std::span<const double>,
            OptimizeCallback,
            void*
        );
        extern template std::expected<FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError>
            train_or_factor_payload<MixedPrecision<ReevaluateKernel>>(
                const TrainConfig<MixedPrecision<ReevaluateKernel>>&,
                MatrixRef<double>,
                std::span<const double>,
                OptimizeCallback,
                void*
            );

        extern template std::expected<void, GprError> fitted_predict_into<DoublePrecision>(
            const FittedPayload<DoublePrecision>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<DoublePrecision>&,
            Prediction<double>&
        );
        extern template std::expected<void, GprError> fitted_predict_into<SinglePrecision>(
            const FittedPayload<SinglePrecision>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<SinglePrecision>&,
            Prediction<float>&
        );
        extern template std::expected<void, GprError> fitted_predict_into<MixedPrecision<PromoteStorage>>(
            const FittedPayload<MixedPrecision<PromoteStorage>>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<MixedPrecision<PromoteStorage>>&,
            Prediction<double>&
        );
        extern template std::expected<void, GprError> fitted_predict_into<MixedPrecision<ReevaluateKernel>>(
            const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
            Prediction<double>&
        );

        extern template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
            DoublePrecision>(
            const FittedPayload<DoublePrecision>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<DoublePrecision>&
        );
        extern template std::expected<PredictiveCovariance<float>, GprError> fitted_predict_covariance<SinglePrecision>(
            const FittedPayload<SinglePrecision>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<SinglePrecision>&
        );
        extern template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
            MixedPrecision<PromoteStorage>>(
            const FittedPayload<MixedPrecision<PromoteStorage>>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<MixedPrecision<PromoteStorage>>&
        );
        extern template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
            MixedPrecision<ReevaluateKernel>>(
            const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
            MatrixRef<double>,
            const PredictOptions&,
            QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
        );

        extern template std::expected<std::vector<double>, GprError> fitted_sample<DoublePrecision>(
            const FittedPayload<DoublePrecision>&,
            MatrixRef<double>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&,
            QueryWorkspace<DoublePrecision>&,
            std::optional<JitterPolicy>
        );
        extern template std::expected<std::vector<float>, GprError> fitted_sample<SinglePrecision>(
            const FittedPayload<SinglePrecision>&,
            MatrixRef<double>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&,
            QueryWorkspace<SinglePrecision>&,
            std::optional<JitterPolicy>
        );
        extern template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<PromoteStorage>>(
            const FittedPayload<MixedPrecision<PromoteStorage>>&,
            MatrixRef<double>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&,
            QueryWorkspace<MixedPrecision<PromoteStorage>>&,
            std::optional<JitterPolicy>
        );
        extern template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<ReevaluateKernel>>(
            const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
            MatrixRef<double>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&,
            QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
            std::optional<JitterPolicy>
        );

        extern template std::expected<Prediction<double>, GprError> fitted_loo_predict<DoublePrecision>(
            const FittedPayload<DoublePrecision>&, const PredictOptions&
        );
        extern template std::expected<Prediction<float>, GprError> fitted_loo_predict<SinglePrecision>(
            const FittedPayload<SinglePrecision>&, const PredictOptions&
        );
        extern template std::expected<Prediction<double>, GprError> fitted_loo_predict<MixedPrecision<PromoteStorage>>(
            const FittedPayload<MixedPrecision<PromoteStorage>>&, const PredictOptions&
        );
        extern template std::expected<Prediction<double>, GprError> fitted_loo_predict<
            MixedPrecision<ReevaluateKernel>>(
            const FittedPayload<MixedPrecision<ReevaluateKernel>>&, const PredictOptions&
        );

    }  // namespace detail

    /// @defgroup cxxgpr_gpr Exact GPR Trainer and Inference Model
    /// @brief High-level type-safe training and posterior prediction APIs.
    /// @{

    /// @ingroup cxxgpr_gpr
    /// @brief Unfitted Exact Gaussian Process Regression trainer parameterized by optimizer O, precision
    /// policy P, and kernel specification K. Does not expose prediction or LOO operations until fitted.
    /// @details Uses immutable builder setters returning copies with updated configuration, and treats
    /// fit() and factor() as const operations that leave *this unchanged even if training fails.
    /// @code
    /// const auto kernel = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
    /// const auto model = cxxgpr::Gpr<>(kernel).fit(x_train, n_samples, n_features, y_train).value();
    /// const auto pred = model.predict(x_query, m_queries, n_features).value();
    /// @endcode
    /// @note Input coordinate buffers are interpreted strictly in column-major order.
    /// @attention Pre-1.0 versions (0.x) may refine template defaults and auxiliary diagnostic fields.
    /// @see FittedGpr, KernelSpec
    /// @tparam O Hyperparameter optimizer type (defaults to optimizer::Lbfgs).
    /// @tparam P Precision policy (defaults to DoublePrecision).
    /// @tparam K Kernel specification type (defaults to KernelSpec).
    template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision, typename K = KernelSpec>
    class Gpr
    {
        static_assert(std::same_as<K, KernelSpec>, "Gpr<O, P, K> currently supports K = KernelSpec");

    public:
        /// @brief Constructs an unfitted GPR trainer from a kernel specification and optional optimizer.
        /// @details Initializes noise variance to 1e-2, identity input/target transforms, a fixed zero
        /// jitter offset (JitterPolicy default; use with_jitter() for an adaptive schedule), accurate exponential math,
        /// and FitPolicy::prefer_speed().
        /// @param[in] kernel_spec Initial kernel expression and parameter bounds.
        /// @param[in] optimizer Optional hyperparameter optimizer configuration.
        explicit Gpr(KernelSpec kernel_spec, O optimizer = O{}) :
            kernel_spec_(std::move(kernel_spec)),
            optimizer_(std::move(optimizer)),
            likelihood_(GaussianLikelihood::create(1e-2).value()),
            input_transform_(std::make_unique<transform::IdentityInput>()),
            target_transform_(std::make_unique<transform::IdentityTarget>())
        {
        }

        /// @brief Copy-constructs a trainer cloning its polymorphic transform configurations.
        /// @details Deep-copies unfitted input and target transform unique_ptr configurations.
        /// @param[in] other Trainer instance to copy.
        Gpr(const Gpr& other) :
            kernel_spec_(other.kernel_spec_),
            optimizer_(other.optimizer_),
            likelihood_(other.likelihood_),
            jitter_(other.jitter_),
            input_transform_(other.input_transform_->clone()),
            target_transform_(other.target_transform_->clone()),
            kernel_exp_(other.kernel_exp_),
            fit_policy_(other.fit_policy_)
        {
        }

        /// @brief Copy-assigns a trainer cloning its polymorphic transform configurations.
        /// @details Guards against self-assignment and delegates via copy-and-move swap.
        /// @param[in] other Trainer instance to assign from.
        /// @return Reference to this Gpr trainer.
        Gpr& operator=(const Gpr& other)
        {
            // Use copy-and-move so cloning transforms provides strong exception/state safety.
            if (this != &other)
            {
                Gpr copy(other);
                *this = std::move(copy);
            }
            return *this;
        }

        /// @brief Move-constructs a Gpr trainer transferring transform ownership.
        Gpr(Gpr&&) noexcept = default;

        /// @brief Move-assigns a Gpr trainer transferring transform ownership.
        /// @return Reference to this Gpr trainer.
        Gpr& operator=(Gpr&&) noexcept = default;

        /// @brief Destroys the Gpr trainer and its owned unfitted transform configurations.
        ~Gpr() = default;

        /// @brief Returns a copy with an updated optimizer configuration.
        /// @details Leaves *this unmodified.
        /// @param[in] optimizer New optimizer instance.
        /// @return Copy of *this configured with optimizer.
        [[nodiscard]] Gpr with_optimizer(O optimizer) const
        {
            Gpr copy(*this);
            copy.optimizer_ = std::move(optimizer);
            return copy;
        }

        /// @brief Returns a copy with an updated Gaussian observation noise likelihood.
        /// @details Leaves *this unmodified.
        /// @param[in] likelihood Validated GaussianLikelihood instance.
        /// @return Copy of *this configured with likelihood.
        [[nodiscard]] Gpr with_likelihood(GaussianLikelihood likelihood) const
        {
            Gpr copy(*this);
            copy.likelihood_ = likelihood;
            return copy;
        }

        /// @brief Returns a copy with an updated observation noise variance after validating positivity.
        /// @details Constructs a GaussianLikelihood from noise_variance and leaves *this unchanged on failure.
        /// @param[in] noise_variance Positive finite observation noise variance.
        /// @return Copy of *this with updated noise variance, or InvalidNoiseVariance on invalid input.
        /// @retval GprErrorCode::InvalidNoiseVariance When noise_variance is non-finite or not strictly positive.
        [[nodiscard]] std::expected<Gpr, GprError> with_noise_variance(double noise_variance) const
        {
            // Validate noise_variance through GaussianLikelihood::create before cloning *this.
            auto lik = GaussianLikelihood::create(noise_variance);
            if (!lik.has_value())
            {
                return std::unexpected(std::move(lik).error());
            }
            return with_likelihood(*lik);
        }

        /// @brief Returns a copy with an updated Cholesky jitter stabilization policy.
        /// @details Leaves *this unmodified.
        /// @param[in] jitter New JitterPolicy (fixed or adaptive).
        /// @return Copy of *this configured with jitter.
        [[nodiscard]] Gpr with_jitter(JitterPolicy jitter) const
        {
            Gpr copy(*this);
            copy.jitter_ = jitter;
            return copy;
        }

        /// @brief Returns a copy with an updated unfitted input feature transform.
        /// @details Clones input_transform polymorphically and leaves *this unmodified.
        /// @param[in] input_transform Unfitted input feature transform to clone.
        /// @return Copy of *this configured with a clone of input_transform.
        [[nodiscard]] Gpr with_input_transform(const transform::UnfittedInputTransform& input_transform) const
        {
            Gpr copy(*this);
            copy.input_transform_ = input_transform.clone();
            return copy;
        }

        /// @brief Returns a copy with an updated unfitted target transform.
        /// @details Clones target_transform polymorphically and leaves *this unmodified.
        /// @param[in] target_transform Unfitted target transform to clone.
        /// @return Copy of *this configured with a clone of target_transform.
        [[nodiscard]] Gpr with_target_transform(const transform::UnfittedTargetTransform& target_transform) const
        {
            Gpr copy(*this);
            copy.target_transform_ = target_transform.clone();
            return copy;
        }

        /// @brief Returns a copy with an updated exponential evaluation backend.
        /// @details Selects KernelExp::Accurate or KernelExp::FastApprox while leaving *this unmodified.
        /// @param[in] mode Exponential evaluation mode.
        /// @return Copy of *this configured with mode.
        [[nodiscard]] Gpr with_kernel_exp(KernelExp mode) const
        {
            Gpr copy(*this);
            copy.kernel_exp_ = mode;
            return copy;
        }

        /// @brief Returns a copy with an updated distance cache retention policy.
        /// @details Leaves *this unmodified.
        /// @param[in] policy Distance matrix caching policy for training.
        /// @return Copy of *this with updated FitPolicy::distance_cache.
        [[nodiscard]] Gpr with_distance_cache_policy(DistanceCachePolicy policy) const
        {
            Gpr copy(*this);
            copy.fit_policy_.distance_cache = policy;
            return copy;
        }

        /// @brief Returns a copy with an updated Cholesky buffer retention policy.
        /// @details Controls whether intermediate Cholesky and LeafCache buffers are retained across steps.
        /// @param[in] policy Cholesky buffer retention policy.
        /// @return Copy of *this with updated FitPolicy::cholesky_buffer.
        [[nodiscard]] Gpr with_cholesky_buffer(CholeskyBuffer policy) const
        {
            Gpr copy(*this);
            copy.fit_policy_.cholesky_buffer = policy;
            return copy;
        }

        /// @brief Returns a copy configured for minimum memory footprint (Uncached + Reuse).
        /// @details Applies FitPolicy::prefer_memory() while leaving *this unmodified.
        /// @return Copy of *this configured to minimize workspace memory.
        [[nodiscard]] Gpr with_prefer_memory() const
        {
            Gpr copy(*this);
            copy.fit_policy_ = FitPolicy::prefer_memory();
            return copy;
        }

        /// @brief Returns a copy configured for maximum optimization speed (Cached + Retain).
        /// @details Applies FitPolicy::prefer_speed() while leaving *this unmodified.
        /// @return Copy of *this configured for maximum optimization throughput.
        [[nodiscard]] Gpr with_prefer_speed() const
        {
            Gpr copy(*this);
            copy.fit_policy_ = FitPolicy::prefer_speed();
            return copy;
        }

        /// @brief Returns the current kernel specification.
        /// @return Constant reference to the configured KernelSpec.
        [[nodiscard]] const KernelSpec& kernel_spec() const noexcept { return kernel_spec_; }

        /// @brief Returns the current optimizer configuration.
        /// @return Constant reference to the configured optimizer instance.
        [[nodiscard]] const O& optimizer() const noexcept { return optimizer_; }

        /// @brief Returns the current Gaussian observation noise likelihood.
        /// @return Constant reference to the configured GaussianLikelihood.
        [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept { return likelihood_; }

        /// @brief Returns the current Cholesky jitter policy.
        /// @return Constant reference to the configured JitterPolicy.
        [[nodiscard]] const JitterPolicy& jitter() const noexcept { return jitter_; }

        /// @brief Returns the current exponential evaluation backend.
        /// @return Configured KernelExp mode.
        [[nodiscard]] KernelExp kernel_exp() const noexcept { return kernel_exp_; }

        /// @brief Returns the current fit buffer policy.
        /// @return Configured FitPolicy controlling distance caching and Cholesky buffer retention.
        [[nodiscard]] FitPolicy fit_policy() const noexcept { return fit_policy_; }

        /// @brief Fits transforms and optimizes hyperparameters on column-major training data (x, y), leaving
        /// *this unchanged on failure.
        /// @details Validates column-major inputs, fits input/target transforms, optimizes negative MLL
        /// (unless O is optimizer::Fixed), re-factors at final parameters, and returns an owned FittedGpr.
        /// @param[in] x Column-major training feature matrix (n x d).
        /// @param[in] y Training target vector of length n.
        /// @return Newly constructed FittedGpr on success, or GprError leaving *this unchanged.
        /// @retval GprErrorCode::EmptyInput When x or y is empty or null.
        /// @retval GprErrorCode::LengthMismatch When y.size() != x.rows().
        /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
        /// @pre `x` must point to finite column-major data of shape `(n, d)` and `y.size() == x.rows()`.
        /// @post On error, `*this` remains completely unchanged.
        /// @see factor(), FittedGpr::predict(), FittedGpr::predict_into()
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> fit(
            MatrixRef<double> x, std::span<const double> y
        ) const
        {
            return run_impl(x, y, /*optimize=*/true);
        }

        /// @brief Fits transforms and optimizes hyperparameters from a raw column-major feature span.
        /// @details Checks the length of x_col_major against n_rows * n_cols, wraps it as a MatrixRef<double> with
        /// leading dimension n_rows, and calls fit().
        /// @param[in] x_col_major Contiguous column-major feature buffer of length n_rows * n_cols.
        /// @param[in] n_rows Number of training samples n.
        /// @param[in] n_cols Number of feature columns d.
        /// @param[in] y Training target vector of length n_rows.
        /// @return Newly constructed FittedGpr on success, or GprError leaving *this unchanged.
        /// @retval GprErrorCode::EmptyInput When x_col_major or y is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x_col_major.size() != n_rows * n_cols or y.size() != n_rows.
        /// @retval GprErrorCode::NonFiniteInput When inputs contain NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> fit(
            std::span<const double> x_col_major, std::size_t n_rows, std::size_t n_cols, std::span<const double> y
        ) const
        {
            auto view = detail::training_view_from_span(x_col_major, n_rows, n_cols, y);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return fit(*view, y);
        }

        /// @brief Fits transforms and factors the exact system at the current hyperparameters without running
        /// optimization iterations, leaving *this unchanged on failure.
        /// @details Computes Cholesky factor L, refined weights alpha, and MLL at the initial parameters.
        /// @param[in] x Column-major training feature matrix (n x d).
        /// @param[in] y Training target vector of length n.
        /// @return Newly constructed FittedGpr on success, or GprError leaving *this unchanged.
        /// @retval GprErrorCode::EmptyInput When x or y is empty or null.
        /// @retval GprErrorCode::LengthMismatch When y.size() != x.rows().
        /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> factor(
            MatrixRef<double> x, std::span<const double> y
        ) const
        {
            return run_impl(x, y, /*optimize=*/false);
        }

        /// @brief Fits transforms and factors the exact system from a raw column-major feature span.
        /// @details Checks the length of x_col_major against n_rows * n_cols, wraps it as a MatrixRef<double> with
        /// leading dimension n_rows, and calls factor().
        /// @param[in] x_col_major Contiguous column-major feature buffer of length n_rows * n_cols.
        /// @param[in] n_rows Number of training samples n.
        /// @param[in] n_cols Number of feature columns d.
        /// @param[in] y Training target vector of length n_rows.
        /// @return Newly constructed FittedGpr on success, or GprError leaving *this unchanged.
        /// @retval GprErrorCode::EmptyInput When x_col_major or y is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x_col_major.size() != n_rows * n_cols or y.size() != n_rows.
        /// @retval GprErrorCode::NonFiniteInput When inputs contain NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> factor(
            std::span<const double> x_col_major, std::size_t n_rows, std::size_t n_cols, std::span<const double> y
        ) const
        {
            auto view = detail::training_view_from_span(x_col_major, n_rows, n_cols, y);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return factor(*view, y);
        }

    private:
        friend class FittedGpr<O, P, K>;

        /// @brief Private full-state constructor used by FittedGpr::into_trainer().
        /// @param[in] kernel_spec Initial or fitted kernel specification.
        /// @param[in] optimizer Optimizer configuration.
        /// @param[in] likelihood Gaussian observation noise likelihood.
        /// @param[in] jitter Cholesky jitter stabilization policy.
        /// @param[in] input_transform Owned unfitted input transform.
        /// @param[in] target_transform Owned unfitted target transform.
        /// @param[in] kernel_exp Exponential evaluation backend.
        /// @param[in] fit_policy Workspace distance-cache and Cholesky-buffer policy.
        Gpr(KernelSpec kernel_spec,
            O optimizer,
            GaussianLikelihood likelihood,
            JitterPolicy jitter,
            std::unique_ptr<transform::UnfittedInputTransform> input_transform,
            std::unique_ptr<transform::UnfittedTargetTransform> target_transform,
            KernelExp kernel_exp,
            FitPolicy fit_policy) :
            kernel_spec_(std::move(kernel_spec)),
            optimizer_(std::move(optimizer)),
            likelihood_(likelihood),
            jitter_(jitter),
            input_transform_(std::move(input_transform)),
            target_transform_(std::move(target_transform)),
            kernel_exp_(kernel_exp),
            fit_policy_(fit_policy)
        {
        }

        /// @brief Executes fit() or factor() via train_or_factor_payload() while keeping *this const.
        /// @details Bridges the concrete optimizer O through a type-erased callback and constructs FittedGpr
        /// atomically only after training succeeds.
        /// @param[in] x Column-major training feature matrix (n x d).
        /// @param[in] y Training target vector of length n.
        /// @param[in] optimize True to optimize hyperparameters (ignored when O is optimizer::Fixed).
        /// @return Constructed FittedGpr on success, or GprError leaving *this unchanged.
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> run_impl(
            MatrixRef<double> x, std::span<const double> y, bool optimize
        ) const
        {
            // Assemble borrowed configuration without mutating any member on *this.
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
                .value_only =
                    std::same_as<O, optimizer::NelderMead> || std::same_as<O, optimizer::FastSimulatedAnnealing>,
            };
            // Bridge the statically typed optimizer to the out-of-line training engine.
            auto cb = [](void* ctx, optimizer::TwiceDifferentiable& obj,
                          std::span<const double> init) -> std::expected<optimizer::OptimizerResult, GprError>
            {
                return detail::invoke_optimizer(*static_cast<const O*>(ctx), obj, init);
            };
            auto payload = detail::train_or_factor_payload<P>(cfg, x, y, cb, const_cast<O*>(&optimizer_));
            if (!payload.has_value())
            {
                return std::unexpected(std::move(payload).error());
            }
            // Only construct FittedGpr and clone unfitted transforms once payload construction succeeds.
            return FittedGpr<O, P, K>(
                std::move(*payload), optimizer_, jitter_, input_transform_->clone(), target_transform_->clone(),
                kernel_exp_, fit_policy_
            );
        }

        /// @brief Initial kernel expression specification and hyperparameter bounds.
        KernelSpec kernel_spec_;
        /// @brief Hyperparameter optimizer policy configuration.
        O optimizer_;
        /// @brief Initial Gaussian observation noise likelihood.
        GaussianLikelihood likelihood_;
        /// @brief Cholesky stabilization jitter policy.
        JitterPolicy jitter_;
        /// @brief Owned unfitted input feature transform prototype.
        std::unique_ptr<transform::UnfittedInputTransform> input_transform_;
        /// @brief Owned unfitted target transform prototype.
        std::unique_ptr<transform::UnfittedTargetTransform> target_transform_;
        /// @brief Exponential evaluation backend mode.
        KernelExp kernel_exp_{KernelExp::Accurate};
        /// @brief Distance cache and Cholesky buffer retention policy.
        FitPolicy fit_policy_{FitPolicy::prefer_speed()};
    };

    /// @brief Fitted Exact Gaussian Process Regression model providing posterior prediction, zero-allocation
    /// predict_into, full posterior covariance, posterior sampling, and closed-form LOO inference.
    /// @details Holds immutable fitted payload state plus a reusable QueryWorkspace so predict_into()
    /// performs zero heap allocations once warmed up on a query batch at least as large.
    /// @tparam O Optimizer policy type preserved for into_trainer() and refit().
    /// @tparam P Precision policy determining Storage and Refine scalar types.
    /// @tparam K Kernel specification type (KernelSpec).
    template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision, typename K = KernelSpec>
    class FittedGpr
    {
    public:
        /// @brief Storage scalar type used for training features and Cholesky factors.
        using Storage = typename P::Storage;
        /// @brief Output and weight scalar type (double for DoublePrecision and MixedPrecision).
        using Refine = typename P::Refine;

        /// @brief Copy-constructs a fitted model with an independent QueryWorkspace.
        /// @details Deep-copies payload vectors, clones fitted and unfitted transforms, and leaves
        /// query_ws_ freshly initialized for independent use.
        /// @param[in] other Fitted model instance to copy.
        FittedGpr(const FittedGpr& other) :
            payload_{
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
                .neg_log_marginal_likelihood = other.payload_.neg_log_marginal_likelihood,
                .refinement_summary = other.payload_.refinement_summary,
                .optimizer_result = other.payload_.optimizer_result,
            },
            optimizer_(other.optimizer_),
            jitter_(other.jitter_),
            unfitted_input_(other.unfitted_input_->clone()),
            unfitted_target_(other.unfitted_target_->clone()),
            kernel_exp_(other.kernel_exp_),
            fit_policy_(other.fit_policy_)
        {
        }

        /// @brief Copy-assigns a fitted model with an independent QueryWorkspace.
        /// @details Uses copy-and-move to ensure clean resource cloning.
        /// @param[in] other Fitted model instance to assign from.
        /// @return Reference to this FittedGpr.
        FittedGpr& operator=(const FittedGpr& other)
        {
            if (this != &other)
            {
                FittedGpr copy(other);
                *this = std::move(copy);
            }
            return *this;
        }

        /// @brief Move-constructs a fitted model transferring payload and workspace buffers.
        FittedGpr(FittedGpr&&) noexcept = default;

        /// @brief Move-assigns a fitted model transferring payload and workspace buffers.
        /// @return Reference to this FittedGpr.
        FittedGpr& operator=(FittedGpr&&) noexcept = default;

        /// @brief Destroys the fitted model and releases owned buffers and transforms.
        ~FittedGpr() = default;

        /// @brief Computes posterior mean and diagonal variance on the original target scale.
        /// @details Delegates to predict_with() with default or caller-supplied PredictOptions.
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] options Optional prediction variance controls.
        /// @return Prediction<Refine> on the original target scale, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> predict(
            MatrixRef<double> xs, const PredictOptions& options = {}
        ) const
        {
            return predict_with(xs, options);
        }

        /// @brief Computes posterior mean and diagonal variance from a raw column-major query span.
        /// @details Checks the length of xs against n_rows * n_cols, wraps it as a MatrixRef<double> with leading
        /// dimension n_rows, and calls predict_with().
        /// @param[in] xs Contiguous column-major query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must equal n_features()).
        /// @param[in] options Optional prediction variance controls.
        /// @return Prediction<Refine> on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When xs.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> predict(
            std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, const PredictOptions& options = {}
        ) const
        {
            auto view = detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return predict_with(*view, options);
        }

        /// @brief Computes posterior mean and diagonal variance with explicit PredictOptions.
        /// @details Allocates a Prediction<Refine> and delegates to predict_with_into().
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] options Prediction variance controls.
        /// @return Prediction<Refine> on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> predict_with(
            MatrixRef<double> xs, const PredictOptions& options
        ) const
        {
            // Reuse predict_with_into so allocation and inverse scaling share one code path.
            Prediction<Refine> out;
            if (auto res = predict_with_into(xs, options, out); !res.has_value())
            {
                return std::unexpected(std::move(res).error());
            }
            return out;
        }

        /// @ingroup cxxgpr_gpr
        /// @brief Computes posterior mean and diagonal variance into out (zero heap allocations after warmup
        /// on matching query shape).
        /// @details Delegates to predict_with_into().
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[out] out Destination prediction buffers populated on the original target scale.
        /// @param[in] options Optional prediction variance controls.
        /// @return Empty expected on success, or GprError on validation or transform failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        /// @pre `xs.cols() == n_features()` and all query coordinates are finite.
        /// @note Calling repeatedly with `xs.rows()` at most the warmed query capacity performs zero heap allocations.
        /// @see predict(), predict_with_into()
        /// @see cxxgpr::predict_covariance_into()
        [[nodiscard]] std::expected<void, GprError> predict_into(
            MatrixRef<double> xs, Prediction<Refine>& out, const PredictOptions& options = {}
        ) const
        {
            return predict_with_into(xs, options, out);
        }

        /// @brief Computes posterior mean and diagonal variance into out from a raw column-major query span.
        /// @details Checks the length of xs against n_rows * n_cols, wraps it as a MatrixRef<double> with leading
        /// dimension n_rows, and calls predict_with_into().
        /// @param[in] xs Contiguous column-major query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must equal n_features()).
        /// @param[out] out Destination prediction buffers populated on the original target scale.
        /// @param[in] options Optional prediction variance controls.
        /// @return Empty expected on success, or GprError on validation or transform failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When xs.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        [[nodiscard]] std::expected<void, GprError> predict_into(
            std::span<const double> xs,
            std::size_t n_rows,
            std::size_t n_cols,
            Prediction<Refine>& out,
            const PredictOptions& options = {}
        ) const
        {
            auto view = detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return predict_with_into(*view, options, out);
        }

        /// @brief Computes posterior mean and diagonal variance into out with explicit PredictOptions.
        /// @details Forwards directly to detail::fitted_predict_into<P>() using mutable query_ws_.
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] options Prediction variance controls.
        /// @param[out] out Destination prediction buffers populated on the original target scale.
        /// @return Empty expected on success, or GprError on validation or transform failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        [[nodiscard]] std::expected<void, GprError> predict_with_into(
            MatrixRef<double> xs, const PredictOptions& options, Prediction<Refine>& out
        ) const
        {
            return detail::fitted_predict_into<P>(payload_, xs, options, query_ws_, out);
        }

        /// @brief Computes posterior mean and full m x m posterior covariance matrix on the original scale.
        /// @details Delegates to predict_covariance_with().
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] options Optional prediction variance controls.
        /// @return PredictiveCovariance<Refine> on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError> predict_covariance(
            MatrixRef<double> xs, const PredictOptions& options = {}
        ) const
        {
            return predict_covariance_with(xs, options);
        }

        /// @brief Computes posterior mean and full m x m posterior covariance from a raw column-major span.
        /// @details Checks the length of xs against n_rows * n_cols, wraps it as a MatrixRef<double> with leading
        /// dimension n_rows, and calls predict_covariance_with().
        /// @param[in] xs Contiguous column-major query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must equal n_features()).
        /// @param[in] options Optional prediction variance controls.
        /// @return PredictiveCovariance<Refine> on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When xs.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError> predict_covariance(
            std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, const PredictOptions& options = {}
        ) const
        {
            auto view = detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return predict_covariance_with(*view, options);
        }

        /// @brief Computes posterior mean and full m x m posterior covariance with explicit PredictOptions.
        /// @details Forwards directly to detail::fitted_predict_covariance<P>() using mutable query_ws_.
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] options Prediction variance controls.
        /// @return PredictiveCovariance<Refine> on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError> predict_covariance_with(
            MatrixRef<double> xs, const PredictOptions& options
        ) const
        {
            return detail::fitted_predict_covariance<P>(payload_, xs, options, query_ws_);
        }

        /// @brief Draws n_draws deterministic posterior samples on the original target scale.
        /// @details Delegates to sample_with().
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed.
        /// @param[in] options Optional prediction variance controls.
        /// @param[in] jitter Optional predictive covariance jitter override.
        /// @return Column-major sample matrix (m x n_draws) on the original scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When predictive covariance factorization fails.
        [[nodiscard]] std::expected<std::vector<Refine>, GprError> sample(
            MatrixRef<double> xs,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {},
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const
        {
            return sample_with(xs, n_draws, seed, options, std::move(jitter));
        }

        /// @brief Draws n_draws deterministic posterior samples from a raw column-major query span.
        /// @details Checks the length of xs against n_rows * n_cols, wraps it as a MatrixRef<double> with leading
        /// dimension n_rows, and calls sample_with().
        /// @param[in] xs Contiguous column-major query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must equal n_features()).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed.
        /// @param[in] options Optional prediction variance controls.
        /// @param[in] jitter Optional predictive covariance jitter override.
        /// @return Column-major sample matrix (m x n_draws) on the original scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When xs.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When predictive covariance factorization fails.
        [[nodiscard]] std::expected<std::vector<Refine>, GprError> sample(
            std::span<const double> xs,
            std::size_t n_rows,
            std::size_t n_cols,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {},
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const
        {
            auto view = detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return sample_with(*view, n_draws, seed, options, std::move(jitter));
        }

        /// @brief Draws n_draws deterministic posterior samples with explicit PredictOptions.
        /// @details Forwards directly to detail::fitted_sample<P>() using mutable query_ws_.
        /// @param[in] xs Column-major query feature matrix (m x n_features()).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed.
        /// @param[in] options Prediction variance controls.
        /// @param[in] jitter Optional predictive covariance jitter override.
        /// @return Column-major sample matrix (m x n_draws) on the original scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != n_features().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When predictive covariance factorization fails.
        [[nodiscard]] std::expected<std::vector<Refine>, GprError> sample_with(
            MatrixRef<double> xs,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options,
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const
        {
            return detail::fitted_sample<P>(payload_, xs, n_draws, seed, options, query_ws_, std::move(jitter));
        }

        /// @brief Computes closed-form Leave-One-Out predictive mean and variance on the original target
        /// scale (GPML §5.4.2).
        /// @details Delegates to loo_predict_with().
        /// @param[in] options Optional prediction variance controls.
        /// @return Prediction<Refine> containing LOO means and variances, or GprError on failure.
        /// @retval GprErrorCode::NonFiniteKernelValue When inverse scaling produces non-finite moments.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> loo_predict(const PredictOptions& options = {}) const
        {
            return loo_predict_with(options);
        }

        /// @brief Computes closed-form Leave-One-Out predictive mean and variance with explicit
        /// PredictOptions.
        /// @details Forwards directly to detail::fitted_loo_predict<P>().
        /// @param[in] options Prediction variance controls.
        /// @return Prediction<Refine> containing LOO means and variances, or GprError on failure.
        /// @retval GprErrorCode::NonFiniteKernelValue When inverse scaling produces non-finite moments.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> loo_predict_with(const PredictOptions& options) const
        {
            return detail::fitted_loo_predict<P>(payload_, options);
        }

        /// @brief Converts this fitted model back into an unfitted Gpr trainer initialized at the fitted
        /// kernel hyperparameters, noise likelihood, transforms, and policy settings.
        /// @details Clones retained unfitted input/target transform configurations for warm-started fitting.
        /// @return Unfitted Gpr<O, P, K> warm-started at the fitted hyperparameters.
        [[nodiscard]] Gpr<O, P, K> into_trainer() const
        {
            return Gpr<O, P, K>(
                payload_.kernel_spec, optimizer_, payload_.likelihood, jitter_, unfitted_input_->clone(),
                unfitted_target_->clone(), kernel_exp_, fit_policy_
            );
        }

        /// @brief Re-optimizes a new model on (x, y) starting from this fitted model's hyperparameters and
        /// configuration.
        /// @details Warm-starts via into_trainer().fit(x, y) without mutating *this.
        /// @param[in] x Column-major training feature matrix (n x d).
        /// @param[in] y Training target vector of length n.
        /// @return Newly fitted FittedGpr on success, or GprError leaving *this unchanged.
        /// @retval GprErrorCode::EmptyInput When x or y is empty or null.
        /// @retval GprErrorCode::LengthMismatch When y.size() != x.rows().
        /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> refit(
            MatrixRef<double> x, std::span<const double> y
        ) const
        {
            return into_trainer().fit(x, y);
        }

        /// @brief Re-optimizes a new model from a raw column-major feature span.
        /// @details Checks the length of x_col_major against n_rows * n_cols, wraps it as a MatrixRef<double> with
        /// leading dimension n_rows, and calls refit().
        /// @param[in] x_col_major Contiguous column-major feature buffer of length n_rows * n_cols.
        /// @param[in] n_rows Number of training samples n.
        /// @param[in] n_cols Number of feature columns d.
        /// @param[in] y Training target vector of length n_rows.
        /// @return Newly fitted FittedGpr on success, or GprError leaving *this unchanged.
        /// @retval GprErrorCode::EmptyInput When x_col_major or y is empty, or n_rows or n_cols is zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x_col_major.size() != n_rows * n_cols or y.size() != n_rows.
        /// @retval GprErrorCode::NonFiniteInput When inputs contain NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
        [[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError> refit(
            std::span<const double> x_col_major, std::size_t n_rows, std::size_t n_cols, std::span<const double> y
        ) const
        {
            auto view = detail::training_view_from_span(x_col_major, n_rows, n_cols, y);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return refit(*view, y);
        }

        /// @brief Returns the fitted kernel specification with updated hyperparameters.
        /// @return Constant reference to the fitted KernelSpec.
        [[nodiscard]] const KernelSpec& kernel_spec() const noexcept { return payload_.kernel_spec; }

        /// @brief Returns the fitted Gaussian observation noise likelihood.
        /// @return Constant reference to the fitted GaussianLikelihood.
        [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept { return payload_.likelihood; }

        /// @brief Returns the number of training samples n.
        /// @return Training sample count n.
        [[nodiscard]] std::size_t n_train() const noexcept { return payload_.n_train; }

        /// @brief Returns the training feature dimension d.
        /// @return Training feature dimension d.
        [[nodiscard]] std::size_t n_features() const noexcept { return payload_.n_features; }

        /// @brief Returns the accepted diagonal jitter offset added during final Cholesky factorization.
        /// @return Accepted diagonal jitter value.
        [[nodiscard]] double factor_jitter() const noexcept { return payload_.factor_jitter; }

        /// @brief Returns the negative log marginal likelihood (the minimized objective, lower is better) at the final
        /// parameters.
        /// @return Final negative marginal log-likelihood scalar.
        [[nodiscard]] double neg_log_marginal_likelihood() const noexcept
        {
            return payload_.neg_log_marginal_likelihood;
        }

        /// @brief Returns the refined prediction weight vector alpha = A^-1 y_transformed.
        /// @return Read-only span over weight vector alpha of length n_train().
        [[nodiscard]] std::span<const Refine> alpha() const noexcept { return payload_.alpha; }

        /// @brief Returns the lower-triangular Cholesky factor L of A = K + (sigma_n^2 + j) I.
        /// @return Column-major MatrixRef<Storage> view of dimensions n_train() x n_train().
        [[nodiscard]] MatrixRef<Storage> l_factor() const noexcept
        {
            return {payload_.l_factor.data(), payload_.n_train, payload_.n_train, payload_.n_train};
        }

        /// @brief Returns the fitted input transform.
        /// @return Constant reference to the fitted input feature transform.
        [[nodiscard]] const transform::FittedInputTransform& input_transform() const noexcept
        {
            return *payload_.input_transform;
        }

        /// @brief Returns the fitted target transform.
        /// @return Constant reference to the fitted target transform.
        [[nodiscard]] const transform::FittedTargetTransform& target_transform() const noexcept
        {
            return *payload_.target_transform;
        }

        /// @brief Returns the iterative refinement summary when P is MixedPrecision, or std::nullopt.
        /// @return Constant reference to optional RefinementSummary diagnostics.
        [[nodiscard]] const std::optional<RefinementSummary>& refinement_summary() const noexcept
        {
            return payload_.refinement_summary;
        }

        /// @brief Returns the optimizer convergence summary.
        /// @return Constant reference to optimizer::OptimizerResult.
        [[nodiscard]] const optimizer::OptimizerResult& optimizer_result() const noexcept
        {
            return payload_.optimizer_result;
        }

        /// @brief Returns the exponential evaluation backend.
        /// @return Configured KernelExp mode.
        [[nodiscard]] KernelExp kernel_exp() const noexcept { return kernel_exp_; }

        /// @brief Returns the fit buffer policy.
        /// @return Configured FitPolicy retained for trainer reconstruction and persistence.
        [[nodiscard]] FitPolicy fit_policy() const noexcept { return fit_policy_; }

        /// @brief Returns the Cholesky jitter policy.
        /// @return Constant reference to configured JitterPolicy.
        [[nodiscard]] const JitterPolicy& jitter() const noexcept { return jitter_; }

        /// @brief Returns the raw (pre-transform) column-major training features in double precision.
        /// @return Read-only span of raw training features of length n_train() * n_features().
        [[nodiscard]] std::span<const double> x_raw() const noexcept { return payload_.x_raw; }

        /// @brief Returns the raw (pre-transform) training targets in double precision.
        /// @return Read-only span of raw training targets of length n_train().
        [[nodiscard]] std::span<const double> y_raw() const noexcept { return payload_.y_raw; }

        /// @brief Returns the unfitted input transform configuration retained for persistence and refit.
        /// @return Constant reference to the retained UnfittedInputTransform.
        [[nodiscard]] const transform::UnfittedInputTransform& unfitted_input_transform() const noexcept
        {
            return *unfitted_input_;
        }

        /// @brief Returns the unfitted target transform configuration retained for persistence and refit.
        /// @return Constant reference to the retained UnfittedTargetTransform.
        [[nodiscard]] const transform::UnfittedTargetTransform& unfitted_target_transform() const noexcept
        {
            return *unfitted_target_;
        }

    private:
        friend class Gpr<O, P, K>;
        friend class persist::LoadedGpr;

        /// @brief Private constructor invoked by Gpr::run_impl() and LoadedGpr::into_fitted().
        /// @param[in] payload Validated fitted payload containing factor, weights, and fitted transforms.
        /// @param[in] optimizer Optimizer configuration preserved for into_trainer() and refit().
        /// @param[in] jitter Cholesky jitter stabilization policy.
        /// @param[in] unfitted_input Cloned unfitted input transform for persistence and refit.
        /// @param[in] unfitted_target Cloned unfitted target transform for persistence and refit.
        /// @param[in] kernel_exp Exponential evaluation backend mode.
        /// @param[in] fit_policy Workspace distance-cache and Cholesky-buffer policy.
        FittedGpr(
            detail::FittedPayload<P> payload,
            O optimizer,
            JitterPolicy jitter,
            std::unique_ptr<transform::UnfittedInputTransform> unfitted_input,
            std::unique_ptr<transform::UnfittedTargetTransform> unfitted_target,
            KernelExp kernel_exp,
            FitPolicy fit_policy
        ) :
            payload_(std::move(payload)),
            optimizer_(std::move(optimizer)),
            jitter_(jitter),
            unfitted_input_(std::move(unfitted_input)),
            unfitted_target_(std::move(unfitted_target)),
            kernel_exp_(kernel_exp),
            fit_policy_(fit_policy)
        {
        }

        /// @brief Owned fitted model state (kernel, factor, weights, transforms, diagnostics).
        detail::FittedPayload<P> payload_;
        /// @brief Optimizer configuration preserved for into_trainer() and refit().
        O optimizer_;
        /// @brief Cholesky stabilization jitter policy.
        JitterPolicy jitter_;
        /// @brief Unfitted input transform retained for persistence and refit().
        std::unique_ptr<transform::UnfittedInputTransform> unfitted_input_;
        /// @brief Unfitted target transform retained for persistence and refit().
        std::unique_ptr<transform::UnfittedTargetTransform> unfitted_target_;
        /// @brief Exponential evaluation backend mode.
        KernelExp kernel_exp_;
        /// @brief Distance cache and Cholesky buffer retention policy.
        FitPolicy fit_policy_;
        /// @brief Mutable reusable query scratch buffers avoiding allocations during predict_into().
        mutable QueryWorkspace<P> query_ws_;
    };

    /// @}

}  // namespace cxxgpr
