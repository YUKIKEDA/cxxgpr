// SPDX-License-Identifier: Apache-2.0
/// @file distance_gpr.hpp
/// @brief Gaussian process regression on caller-supplied squared distances: DistanceGpr and FittedDistanceGpr.
/// @details A DistanceGpr is built from a DistanceKernel and trained from a DistanceSource holding the squared
/// distances between the training points (geodesic, graph, or any other distance) instead of coordinates. The fitted
/// model predicts from rectangular DistanceSources holding the squared distances from the training points to the
/// query points. The lifecycle, error contract and numerics follow Gpr and FittedGpr.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/distance.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/optimizer/optimizer.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/precision/refine.hpp"
#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"
#include "cxxgpr/workspace/buffers.hpp"
#include "cxxgpr/workspace/leaf_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Unfitted trainer for Gaussian process regression on supplied squared distances.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (distance only, or distances with points).
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C> class DistanceGpr;
    /// @brief Fitted supplied-distance model that owns its training result.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (distance only, or distances with points).
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C> class FittedDistanceGpr;

    /// @brief Library-internal part of the persistence layer.
    namespace persist::detail
    {
        /// @brief Library-internal accessor used by the persistence layer.
        struct DistanceModelAccess;
    }  // namespace persist::detail

    /// @brief Internal training and prediction engine for supplied-distance models.
    namespace detail
    {

        /// @brief Cholesky factor storage that either owns its elements or views a read-only memory map.
        /// @details A loaded double-precision checkpoint maps the stored factor instead of copying it; the mapping
        /// is kept alive by keepalive for as long as any copy of this object exists. Every other factor is owned.
        /// @tparam Storage Storage scalar type.
        template <typename Storage> struct DistanceFactorStorage
        {
            /// @brief Owned elements, column-major n x n (empty when the factor is mapped).
            std::vector<Storage> owned;
            /// @brief Mapped elements, column-major n x n (empty when the factor is owned).
            std::span<const Storage> mapped;
            /// @brief Owner of the mapping behind mapped (null when the factor is owned).
            std::shared_ptr<const void> keepalive;

            /// @brief Returns the first element of the factor.
            /// @return Pointer to the mapped or owned elements.
            [[nodiscard]] const Storage* data() const noexcept { return mapped.empty() ? owned.data() : mapped.data(); }

            /// @brief Returns the factor elements.
            /// @return Read-only span over the mapped or owned elements.
            [[nodiscard]] std::span<const Storage> view() const noexcept
            {
                return mapped.empty() ? std::span<const Storage>(owned) : mapped;
            }

            /// @brief Reports whether the factor views a memory map.
            /// @return True when the elements are mapped rather than owned.
            [[nodiscard]] bool is_mapped() const noexcept { return !mapped.empty(); }
        };

        /// @brief Immutable fitted state of a supplied-distance model, shared between copies of the fitted model.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode (DistanceOnly or WithPoints).
        template <PrecisionPolicyConcept P, DistanceModeConcept C> struct DistanceFittedPayload
        {
            /// @brief Storage scalar type.
            using Storage = typename P::Storage;
            /// @brief Refinement and output scalar type.
            using Refine = typename P::Refine;

            /// @brief Kernel carrying the fitted hyperparameters.
            DistanceKernel<C> kernel;
            /// @brief Observation noise likelihood carrying the fitted noise variance.
            GaussianLikelihood likelihood;
            /// @brief Execution-layer kernel compiled in the storage scalar.
            std::shared_ptr<const CompiledDistanceKernel<Storage>> compiled;
            /// @brief Validated training distance tables in the storage scalar.
            std::shared_ptr<const CheckedDistanceStore<Storage>> store;
            /// @brief Fitted coordinate transform (WithPoints only, otherwise null).
            std::shared_ptr<const transform::FittedInputTransform> input_transform;
            /// @brief Fitted target transform.
            std::shared_ptr<const transform::FittedTargetTransform> target_transform;
            /// @brief Raw training coordinates, column-major n x d (WithPoints only).
            std::vector<double> x_raw;
            /// @brief Raw training targets.
            std::vector<double> y_raw;
            /// @brief Transformed training coordinates in the storage scalar (WithPoints only).
            std::vector<Storage> x_train;
            /// @brief Transformed training targets in the refinement scalar.
            std::vector<Refine> y_transformed;
            /// @brief Lower Cholesky factor of K + (noise + jitter) I, column-major n x n (owned or mapped).
            DistanceFactorStorage<Storage> l_factor;
            /// @brief Weights alpha = (K + (noise + jitter) I)^-1 y.
            std::vector<Refine> alpha;
            /// @brief Number of training points.
            std::size_t n_train{0};
            /// @brief Coordinate dimension (0 for DistanceOnly).
            std::size_t n_features{0};
            /// @brief Accepted diagonal jitter offset.
            double factor_jitter{0.0};
            /// @brief Negative log marginal likelihood at the fitted hyperparameters.
            double neg_log_marginal_likelihood{0.0};
            /// @brief Optimizer trajectory summary.
            optimizer::OptimizerResult optimizer_result;
            /// @brief Mixed-precision refinement summary, or std::nullopt when P does not refine in double.
            std::optional<RefinementSummary> refinement_summary;
        };

        /// @brief Reusable query scratch buffers owned by one fitted model object.
        /// @details Every member grows monotonically and is reused, so a prediction of the same or a smaller size
        /// than an earlier one allocates nothing. The contents are scratch: a failed call may leave them partially
        /// written, and every call rewrites what it reads.
        /// @tparam Storage Storage scalar type.
        template <typename Storage> struct DistanceQueryScratch
        {
            /// @brief Validated cross tables of the latest query, refilled in place by every call.
            std::optional<CheckedDistanceStore<Storage>> cross_store;
            /// @brief Scratch of the prior-diagonal evaluation.
            std::vector<Storage> diag_scratch;
            /// @brief Cross-covariance K(X, X*) and then L^-1 K(X, X*), n x m.
            std::vector<Storage> k_star;
            /// @brief Scratch for the kernel tree evaluation, n x m.
            std::vector<Storage> scratch;
            /// @brief Prior diagonal k(x*, x*), length m.
            std::vector<Storage> k_ss;
            /// @brief Transformed query coordinates in double, m x d.
            std::vector<double> xq_double;
            /// @brief Transformed query coordinates in the storage scalar, m x d.
            std::vector<Storage> xq;
            /// @brief Nested buffers for the cross evaluation.
            NestedBuffers<Storage> nested;
            /// @brief Double-precision staging of the mean for the inverse target transform.
            std::vector<double> mean_tmp;
            /// @brief Double-precision staging of the variances for the inverse target transform.
            std::vector<double> var_tmp;
        };

        /// @brief Borrowed training configuration passed to the out-of-line engine.
        /// @tparam C Distance mode.
        template <DistanceModeConcept C> struct DistanceTrainConfig
        {
            /// @brief Kernel to train (its parameters are the initial hyperparameters).
            const DistanceKernel<C>& kernel;
            /// @brief Initial observation noise likelihood.
            const GaussianLikelihood& likelihood;
            /// @brief Jitter policy for the Cholesky factorizations.
            const JitterPolicy& jitter;
            /// @brief Unfitted coordinate transform.
            const transform::UnfittedInputTransform& input_transform;
            /// @brief Unfitted target transform.
            const transform::UnfittedTargetTransform& target_transform;
            /// @brief Exponential evaluation backend.
            KernelExp kernel_exp;
            /// @brief True to optimize hyperparameters, false to factor at the initial values.
            bool optimize;
            /// @brief True when the caller's optimizer tracks parameter change indices (enables incremental leaf
            /// caching).
            bool use_change_indices = false;
            /// @brief True when optimization uses only objective values or no optimization runs (reuses Cholesky
            /// buffer).
            bool value_only = false;
        };

        /// @brief Trains or factors a supplied-distance model and returns its fitted payload.
        /// @details Validates the inputs, fits and applies the transforms, validates the distance tables once,
        /// optimizes the hyperparameters through the supplied callback (or evaluates once when cfg.optimize is false),
        /// and returns the factor, weights and fitted hyperparameters.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] cfg Borrowed training configuration.
        /// @param[in] train Square training distance tables bound to the kernel slots.
        /// @param[in] x Training coordinates (required for WithPoints, ignored otherwise).
        /// @param[in] y Training targets.
        /// @param[in] optimize_cb Callback running the optimizer against the objective.
        /// @param[in] optimize_ctx Opaque pointer forwarded to the callback.
        /// @return Fitted payload, or GprError describing the first failing check.
        /// @retval GprErrorCode::EmptyInput When y or the tables are empty.
        /// @retval GprErrorCode::LengthMismatch When y.size() or the coordinate rows disagree with the tables.
        /// @retval GprErrorCode::NonFiniteInput When y or x contains NaN or Inf.
        /// @retval GprErrorCode::InvalidDistance When the tables are invalid.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated or mismatched.
        /// @retval GprErrorCode::CholeskyFailed When factorization fails after the jitter policy is exhausted.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<DistanceFittedPayload<P, C>, GprError> train_distance_payload(
            const DistanceTrainConfig<C>& cfg,
            const DistanceSource& train,
            std::optional<MatrixRef<double>> x,
            std::span<const double> y,
            std::expected<optimizer::OptimizerResult, GprError> (*optimize_cb)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void* optimize_ctx
        );

        /// @brief Assembles the fitted payload of a supplied-distance model from stored parts, without optimizing.
        /// @details Used to restore a persisted model. Validates the distance tables once, applies the fitted
        /// transforms, checks the stored factor and weights, and derives the marginal likelihood from them.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] kernel Kernel carrying the fitted hyperparameters.
        /// @param[in] likelihood Likelihood carrying the fitted noise variance.
        /// @param[in] kernel_exp Exponential mode of the kernel evaluation.
        /// @param[in] train Square training distance tables bound to the kernel slots.
        /// @param[in] x_raw Raw training coordinates, column-major n x n_features (WithPoints only).
        /// @param[in] n_features Coordinate dimension (0 for DistanceOnly).
        /// @param[in] y_raw Raw training targets.
        /// @param[in] input_transform Fitted coordinate transform (WithPoints only).
        /// @param[in] target_transform Fitted target transform.
        /// @param[in] l_factor Lower Cholesky factor, column-major n x n.
        /// @param[in] alpha Weights in the transformed target space.
        /// @param[in] factor_jitter Accepted diagonal jitter.
        /// @param[in] l_keepalive Owner of a read-only mapping behind l_factor; when non-null and the storage is
        /// double, the payload views l_factor instead of copying it.
        /// @return Fitted payload, or GprError describing the first failing check.
        /// @retval GprErrorCode::PersistFailed When the factor or the weights are not usable.
        /// @retval GprErrorCode::InvalidDistance When the tables are invalid.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated or mismatched.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<DistanceFittedPayload<P, C>, GprError> assemble_distance_payload(
            const DistanceKernel<C>& kernel,
            const GaussianLikelihood& likelihood,
            KernelExp kernel_exp,
            const DistanceSource& train,
            std::span<const double> x_raw,
            std::size_t n_features,
            std::span<const double> y_raw,
            const std::shared_ptr<const transform::FittedInputTransform>& input_transform,
            const std::shared_ptr<const transform::FittedTargetTransform>& target_transform,
            std::span<const typename P::Storage> l_factor,
            std::span<const typename P::Refine> alpha,
            double factor_jitter,
            std::shared_ptr<const void> l_keepalive
        );

        /// @brief Computes the posterior mean and diagonal variance at m query points.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in,out] scratch Reusable query buffers.
        /// @param[in] cross Rectangular n x m tables of squared distances from training points to query points.
        /// @param[in] xq Query coordinates m x d (WithPoints only).
        /// @param[in] options Variance selection.
        /// @param[out] out Prediction on the original target scale.
        /// @return Empty expected on success, or GprError on invalid query input.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<void, GprError> predict_distance_into(
            const DistanceFittedPayload<P, C>& payload,
            DistanceQueryScratch<typename P::Storage>& scratch,
            const DistanceSource& cross,
            std::optional<MatrixRef<double>> xq,
            const PredictOptions& options,
            Prediction<typename P::Refine>& out
        );

        /// @brief Computes the joint posterior mean and full m x m covariance at m query points.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in,out] scratch Reusable query buffers.
        /// @param[in] cross Rectangular n x m tables of squared distances from training points to query points.
        /// @param[in] query Square m x m tables of squared distances between the query points.
        /// @param[in] xq Query coordinates m x d (WithPoints only).
        /// @param[in] options Variance selection.
        /// @return Mean and covariance on the original target scale, or GprError.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<PredictiveCovariance<typename P::Refine>, GprError> predict_distance_covariance(
            const DistanceFittedPayload<P, C>& payload,
            DistanceQueryScratch<typename P::Storage>& scratch,
            const DistanceSource& cross,
            const DistanceSource& query,
            std::optional<MatrixRef<double>> xq,
            const PredictOptions& options
        );

        /// @brief Draws posterior samples at m query points.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in,out] scratch Reusable query buffers.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[in] query Square m x m tables between the query points.
        /// @param[in] xq Query coordinates m x d (WithPoints only).
        /// @param[in] n_draws Number of samples.
        /// @param[in] seed Seed of the deterministic generator.
        /// @param[in] options Variance selection.
        /// @return Column-major m x n_draws samples on the original target scale, or GprError.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<std::vector<typename P::Refine>, GprError> sample_distance(
            const DistanceFittedPayload<P, C>& payload,
            DistanceQueryScratch<typename P::Storage>& scratch,
            const DistanceSource& cross,
            const DistanceSource& query,
            std::optional<MatrixRef<double>> xq,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options
        );

        /// @brief Computes closed-form leave-one-out means and variances at the training points.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in] options Variance selection.
        /// @return Leave-one-out prediction on the original target scale, or GprError.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError> loo_distance(
            const DistanceFittedPayload<P, C>& payload, const PredictOptions& options
        );

    }  // namespace detail

    /// @brief Fitted Gaussian process on supplied squared distances, able to predict, sample and cross-validate.
    /// @details Copies share the immutable fitted state and own independent query buffers. The query methods are
    /// `const` but write their model-owned scratch buffers, so one object must not be used from two threads at once;
    /// copy the model per thread instead.
    /// @tparam O Optimizer policy retained for refit() and into_trainer().
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (DistanceOnly or WithPoints).
    template <
        typename O = optimizer::Lbfgs,
        PrecisionPolicyConcept P = DoublePrecision,
        DistanceModeConcept C = DistanceOnly>
    class FittedDistanceGpr
    {
    public:
        /// @brief Storage scalar type.
        using Storage = typename P::Storage;
        /// @brief Refinement and output scalar type.
        using Refine = typename P::Refine;

        /// @brief Copies the model; the copy shares the fitted state and starts with empty query buffers.
        /// @param[in] other Model to copy.
        FittedDistanceGpr(const FittedDistanceGpr& other) :
            payload_(other.payload_),
            optimizer_(other.optimizer_),
            jitter_(other.jitter_),
            input_transform_(other.input_transform_ ? other.input_transform_->clone() : nullptr),
            target_transform_(other.target_transform_->clone()),
            kernel_exp_(other.kernel_exp_)
        {
        }

        /// @brief Copy-assigns the model.
        /// @param[in] other Model to assign from.
        /// @return Reference to this model.
        FittedDistanceGpr& operator=(const FittedDistanceGpr& other)
        {
            if (this != &other)
            {
                FittedDistanceGpr copy(other);
                *this = std::move(copy);
            }
            return *this;
        }

        /// @brief Move-constructs the model.
        FittedDistanceGpr(FittedDistanceGpr&&) noexcept = default;

        /// @brief Move-assigns the model.
        /// @return Reference to this model.
        FittedDistanceGpr& operator=(FittedDistanceGpr&&) noexcept = default;

        /// @brief Destroys the model.
        ~FittedDistanceGpr() = default;

        /// @brief Posterior mean and diagonal variance at m query points.
        /// @details The inputs are checked in this order, and the first failure is returned: the structure of the
        /// tables (slots, emptiness, size products, shapes), their row count, the number of ARD tables, and only
        /// then the table values. A failure leaves the fitted state unchanged.
        /// @param[in] cross Rectangular n x m tables of squared distances from training points to query points,
        /// bound to the slots of kernel().
        /// @param[in] options Variance selection.
        /// @return Prediction on the original target scale, or GprError.
        /// @retval GprErrorCode::InvalidConfig When the tidy tolerance of cross is invalid.
        /// @retval GprErrorCode::UnsupportedKernelOperation When cross binds a slot of another kernel.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated or of the wrong kind.
        /// @retval GprErrorCode::EmptyInput When a table is empty.
        /// @retval GprErrorCode::LengthMismatch When the tables differ in shape or do not have n_train() rows.
        /// @retval GprErrorCode::DimensionMismatch When an ARD slot supplies the wrong number of tables.
        /// @retval GprErrorCode::InvalidDistance When a table entry is non-finite or negative.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf before its
        /// clamp at zero.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> predict(
            const DistanceSource& cross, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>
        {
            Prediction<Refine> out;
            if (auto res = predict_into(cross, out, options); !res.has_value())
            {
                return std::unexpected(std::move(res).error());
            }
            return out;
        }

        /// @brief Posterior mean and diagonal variance at m query points of a WithPoints model.
        /// @details Checks, in this order: empty xq, the structure of the tables, their row count and the rows of
        /// xq, the number of ARD tables and the columns of xq, the finiteness of xq, and the table values. A
        /// failure leaves the fitted state unchanged.
        /// @param[in] cross Rectangular n x m tables of squared distances from training points to query points.
        /// @param[in] xq Query coordinates, m x d.
        /// @param[in] options Variance selection.
        /// @return Prediction on the original target scale, or GprError.
        /// @retval GprErrorCode::EmptyInput When xq or a table is empty.
        /// @retval GprErrorCode::UnsupportedKernelOperation When cross binds a slot of another kernel.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated or of the wrong kind.
        /// @retval GprErrorCode::LengthMismatch When the tables differ in shape, do not have n_train() rows, or
        /// xq.rows() differs from the table columns.
        /// @retval GprErrorCode::DimensionMismatch When xq.cols() != n_features() or an ARD slot supplies the wrong
        /// number of tables.
        /// @retval GprErrorCode::NonFiniteInput When xq contains NaN or Inf.
        /// @retval GprErrorCode::InvalidDistance When a table entry is non-finite or negative.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf before its
        /// clamp at zero.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> predict(
            const DistanceSource& cross, MatrixRef<double> xq, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>
        {
            Prediction<Refine> out;
            if (auto res = predict_into(cross, xq, out, options); !res.has_value())
            {
                return std::unexpected(std::move(res).error());
            }
            return out;
        }

        /// @brief Same as predict(), writing into out and reusing this object's query workspace.
        /// @details The validated cross tables, the dense query buffers and out grow to the largest query count
        /// seen. After one call with m queries, a call with the same slots and at most m queries performs no heap
        /// allocation; only the first call and a call with more queries (or more ARD dimensions) allocate.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[out] out Prediction buffers.
        /// @param[in] options Variance selection.
        /// @return Empty expected on success, or GprError (see predict()).
        /// @note Copies of a model own separate workspaces; one object must not be queried from two threads at once.
        [[nodiscard]] std::expected<void, GprError> predict_into(
            const DistanceSource& cross, Prediction<Refine>& out, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>
        {
            return detail::predict_distance_into<P, C>(*payload_, scratch_, cross, std::nullopt, options, out);
        }

        /// @brief Same as predict() for a WithPoints model, writing into out and reusing the query workspace.
        /// @details Allocates nothing after a warm-up call with the same slots and at least as many queries; see
        /// the DistanceOnly overload.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[in] xq Query coordinates, m x d.
        /// @param[out] out Prediction buffers.
        /// @param[in] options Variance selection.
        /// @return Empty expected on success, or GprError (see predict()).
        [[nodiscard]] std::expected<void, GprError> predict_into(
            const DistanceSource& cross,
            MatrixRef<double> xq,
            Prediction<Refine>& out,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>
        {
            return detail::predict_distance_into<P, C>(*payload_, scratch_, cross, xq, options, out);
        }

        /// @brief Joint posterior mean and full covariance at m query points.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[in] query Square m x m tables of squared distances between the query points.
        /// @param[in] options Variance selection (Observation adds the noise variance to the diagonal).
        /// @return Mean and covariance on the original target scale, or GprError.
        [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError> predict_covariance(
            const DistanceSource& cross, const DistanceSource& query, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>
        {
            return detail::predict_distance_covariance<P, C>(*payload_, scratch_, cross, query, std::nullopt, options);
        }

        /// @brief Joint posterior mean and full covariance for a WithPoints model.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[in] query Square m x m tables between the query points.
        /// @param[in] xq Query coordinates, m x d.
        /// @param[in] options Variance selection.
        /// @return Mean and covariance on the original target scale, or GprError.
        [[nodiscard]] std::expected<PredictiveCovariance<Refine>, GprError> predict_covariance(
            const DistanceSource& cross,
            const DistanceSource& query,
            MatrixRef<double> xq,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>
        {
            return detail::predict_distance_covariance<P, C>(*payload_, scratch_, cross, query, xq, options);
        }

        /// @brief Draws deterministic posterior samples at m query points.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[in] query Square m x m tables between the query points.
        /// @param[in] n_draws Number of samples.
        /// @param[in] seed Seed of the Xoshiro256++ generator.
        /// @param[in] options Variance selection.
        /// @return Column-major m x n_draws samples on the original target scale, or GprError.
        [[nodiscard]] std::expected<std::vector<Refine>, GprError> sample(
            const DistanceSource& cross,
            const DistanceSource& query,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>
        {
            return detail::sample_distance<P, C>(
                *payload_, scratch_, cross, query, std::nullopt, n_draws, seed, options
            );
        }

        /// @brief Draws deterministic posterior samples for a WithPoints model.
        /// @param[in] cross Rectangular n x m tables from training points to query points.
        /// @param[in] query Square m x m tables between the query points.
        /// @param[in] xq Query coordinates, m x d.
        /// @param[in] n_draws Number of samples.
        /// @param[in] seed Seed of the Xoshiro256++ generator.
        /// @param[in] options Variance selection.
        /// @return Column-major m x n_draws samples on the original target scale, or GprError.
        [[nodiscard]] std::expected<std::vector<Refine>, GprError> sample(
            const DistanceSource& cross,
            const DistanceSource& query,
            MatrixRef<double> xq,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>
        {
            return detail::sample_distance<P, C>(*payload_, scratch_, cross, query, xq, n_draws, seed, options);
        }

        /// @brief Closed-form leave-one-out means and variances at the training points.
        /// @param[in] options Variance selection.
        /// @return Leave-one-out prediction on the original target scale, or GprError.
        [[nodiscard]] std::expected<Prediction<Refine>, GprError> loo_predict(const PredictOptions& options = {}) const
        {
            return detail::loo_distance<P, C>(*payload_, options);
        }

        /// @brief Returns a trainer initialized with the fitted hyperparameters and this model's configuration.
        /// @return Unfitted DistanceGpr warm-started at the fitted hyperparameters.
        [[nodiscard]] DistanceGpr<O, P, C> into_trainer() const;

        /// @brief Fits a new model on new tables, warm-started from this model's hyperparameters.
        /// @param[in] train Square tables of squared distances between the new training points.
        /// @param[in] y New targets.
        /// @return Newly fitted model, or GprError; this model is unchanged either way.
        [[nodiscard]] std::expected<FittedDistanceGpr, GprError> refit(
            const DistanceSource& train, std::span<const double> y
        ) const
            requires std::same_as<C, DistanceOnly>;

        /// @brief Fits a new WithPoints model on new tables and coordinates, warm-started from this model.
        /// @param[in] train Square tables of squared distances between the new training points.
        /// @param[in] x New training coordinates, n x d.
        /// @param[in] y New targets.
        /// @return Newly fitted model, or GprError; this model is unchanged either way.
        [[nodiscard]] std::expected<FittedDistanceGpr, GprError> refit(
            const DistanceSource& train, MatrixRef<double> x, std::span<const double> y
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Returns the kernel carrying the fitted hyperparameters (its slots define the table bindings).
        /// @return Fitted kernel.
        [[nodiscard]] const DistanceKernel<C>& kernel() const noexcept { return payload_->kernel; }

        /// @brief Returns the fitted observation noise likelihood.
        /// @return Fitted likelihood.
        [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept { return payload_->likelihood; }

        /// @brief Returns the number of training points.
        /// @return Training point count n.
        [[nodiscard]] std::size_t n_train() const noexcept { return payload_->n_train; }

        /// @brief Returns the coordinate dimension (0 for DistanceOnly).
        /// @return Feature count d.
        [[nodiscard]] std::size_t n_features() const noexcept { return payload_->n_features; }

        /// @brief Returns the accepted diagonal jitter offset.
        /// @return Jitter added during the final factorization.
        [[nodiscard]] double factor_jitter() const noexcept { return payload_->factor_jitter; }

        /// @brief Returns the negative log marginal likelihood (the minimized objective) at the fitted values.
        /// @return Negative log marginal likelihood.
        [[nodiscard]] double neg_log_marginal_likelihood() const noexcept
        {
            return payload_->neg_log_marginal_likelihood;
        }

        /// @brief Returns the weights alpha = (K + (noise + jitter) I)^-1 y on the transformed target scale.
        /// @return Weight vector of length n.
        [[nodiscard]] std::span<const Refine> alpha() const noexcept { return payload_->alpha; }

        /// @brief Returns a view of the lower Cholesky factor, column-major n x n (strictly upper entries are unused).
        /// @return Factor view.
        [[nodiscard]] MatrixRef<Storage> l_factor() const noexcept
        {
            return MatrixRef<Storage>(
                payload_->l_factor.data(), payload_->n_train, payload_->n_train, payload_->n_train
            );
        }

        /// @brief Returns the mixed-precision refinement summary of the final weights.
        /// @return Summary, or std::nullopt when the precision policy does not refine in double.
        [[nodiscard]] const std::optional<RefinementSummary>& refinement_summary() const noexcept
        {
            return payload_->refinement_summary;
        }

        /// @brief Returns the optimizer trajectory summary of the training run.
        /// @return Optimizer result.
        [[nodiscard]] const optimizer::OptimizerResult& optimizer_result() const noexcept
        {
            return payload_->optimizer_result;
        }

    private:
        friend class DistanceGpr<O, P, C>;
        friend struct persist::detail::DistanceModelAccess;

        /// @brief Constructs a fitted model from a training payload and the trainer configuration.
        /// @param[in] payload Immutable fitted state shared between copies.
        /// @param[in] optimizer Optimizer retained for refitting.
        /// @param[in] jitter Jitter policy retained for refitting.
        /// @param[in] input_transform Unfitted input transform retained for refitting.
        /// @param[in] target_transform Unfitted target transform retained for refitting.
        /// @param[in] kernel_exp Kernel exponential evaluation mode.
        FittedDistanceGpr(
            std::shared_ptr<const detail::DistanceFittedPayload<P, C>> payload,
            O optimizer,
            JitterPolicy jitter,
            std::unique_ptr<transform::UnfittedInputTransform> input_transform,
            std::unique_ptr<transform::UnfittedTargetTransform> target_transform,
            KernelExp kernel_exp
        ) :
            payload_(std::move(payload)),
            optimizer_(std::move(optimizer)),
            jitter_(jitter),
            input_transform_(std::move(input_transform)),
            target_transform_(std::move(target_transform)),
            kernel_exp_(kernel_exp)
        {
        }

        /// @brief Immutable fitted state shared between copies.
        std::shared_ptr<const detail::DistanceFittedPayload<P, C>> payload_;
        /// @brief Optimizer retained for refitting.
        O optimizer_;
        /// @brief Jitter policy retained for refitting.
        JitterPolicy jitter_;
        /// @brief Unfitted input transform retained for refitting.
        std::unique_ptr<transform::UnfittedInputTransform> input_transform_;
        /// @brief Unfitted target transform retained for refitting.
        std::unique_ptr<transform::UnfittedTargetTransform> target_transform_;
        /// @brief Kernel exponential evaluation mode.
        KernelExp kernel_exp_{KernelExp::Accurate};
        /// @brief Per-model scratch buffers reused across prediction calls.
        mutable detail::DistanceQueryScratch<Storage> scratch_;
    };

    /// @brief Unfitted trainer for Gaussian process regression on supplied squared distances.
    /// @details All methods are `const`; builder methods return a new trainer, and a failed fit leaves the trainer and
    /// every previously fitted model unchanged.
    /// @tparam O Optimizer policy (Lbfgs, NelderMead, TrustRegion, FastSimulatedAnnealing, Fixed, or a user type).
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (DistanceOnly or WithPoints).
    template <
        typename O = optimizer::Lbfgs,
        PrecisionPolicyConcept P = DoublePrecision,
        DistanceModeConcept C = DistanceOnly>
    class DistanceGpr
    {
    public:
        /// @brief Constructs a trainer from a distance kernel and an optimizer.
        /// @details Initializes the noise variance to 1e-2, identity transforms, a fixed zero jitter offset and
        /// accurate exponential math.
        /// @param[in] kernel Kernel whose parameters are the initial hyperparameters and whose slots define the
        /// distance tables.
        /// @param[in] optimizer Hyperparameter optimizer.
        explicit DistanceGpr(DistanceKernel<C> kernel, O optimizer = O{}) :
            kernel_(std::move(kernel)),
            optimizer_(std::move(optimizer)),
            likelihood_(GaussianLikelihood::create(1e-2).value()),
            input_transform_(std::make_unique<transform::IdentityInput>()),
            target_transform_(std::make_unique<transform::IdentityTarget>())
        {
        }

        /// @brief Copies the trainer, cloning the transforms.
        /// @param[in] other Trainer to copy.
        DistanceGpr(const DistanceGpr& other) :
            kernel_(other.kernel_),
            optimizer_(other.optimizer_),
            likelihood_(other.likelihood_),
            jitter_(other.jitter_),
            input_transform_(other.input_transform_->clone()),
            target_transform_(other.target_transform_->clone()),
            kernel_exp_(other.kernel_exp_)
        {
        }

        /// @brief Move-constructs the trainer.
        DistanceGpr(DistanceGpr&&) noexcept = default;

        /// @brief Destroys the trainer.
        ~DistanceGpr() = default;

        /// @brief Copy assignment is deleted; build a new trainer instead.
        /// @return Not applicable; the operator is deleted.
        DistanceGpr& operator=(const DistanceGpr&) = delete;

        /// @brief Move assignment is deleted; build a new trainer instead.
        /// @return Not applicable; the operator is deleted.
        DistanceGpr& operator=(DistanceGpr&&) = delete;

        /// @brief Returns a trainer with a different initial observation noise variance.
        /// @param[in] noise_variance Initial noise variance inside the open interval of the likelihood.
        /// @return New trainer, or GprError.
        /// @retval GprErrorCode::InvalidNoiseVariance When the value is outside the open interval or non-finite.
        [[nodiscard]] std::expected<DistanceGpr, GprError> with_noise_variance(double noise_variance) const
        {
            auto lik = GaussianLikelihood::create(noise_variance);
            if (!lik.has_value())
            {
                return std::unexpected(std::move(lik).error());
            }
            DistanceGpr copy(*this);
            copy.likelihood_ = *lik;
            return copy;
        }

        /// @brief Returns a trainer with a different jitter policy.
        /// @param[in] jitter Jitter policy for the Cholesky factorizations.
        /// @return New trainer.
        [[nodiscard]] DistanceGpr with_jitter(JitterPolicy jitter) const
        {
            DistanceGpr copy(*this);
            copy.jitter_ = jitter;
            return copy;
        }

        /// @brief Returns a trainer with a different exponential backend.
        /// @param[in] kernel_exp Accurate or FastApprox.
        /// @return New trainer.
        [[nodiscard]] DistanceGpr with_kernel_exp(KernelExp kernel_exp) const
        {
            DistanceGpr copy(*this);
            copy.kernel_exp_ = kernel_exp;
            return copy;
        }

        /// @brief Returns a trainer with a target transform.
        /// @param[in] target_transform Unfitted target transform (cloned).
        /// @return New trainer.
        [[nodiscard]] DistanceGpr with_target_transform(
            const transform::UnfittedTargetTransform& target_transform
        ) const
        {
            DistanceGpr copy(*this);
            copy.target_transform_ = target_transform.clone();
            return copy;
        }

        /// @brief Returns a trainer with a coordinate transform; available for WithPoints models only.
        /// @param[in] input_transform Unfitted input transform (cloned). Supplied distances are never transformed.
        /// @return New trainer.
        [[nodiscard]] DistanceGpr with_input_transform(const transform::UnfittedInputTransform& input_transform) const
            requires std::same_as<C, WithPoints>
        {
            DistanceGpr copy(*this);
            copy.input_transform_ = input_transform.clone();
            return copy;
        }

        /// @brief Returns a trainer with a different optimizer.
        /// @tparam O2 New optimizer type.
        /// @param[in] optimizer New optimizer instance.
        /// @return New trainer of the new optimizer type.
        template <typename O2> [[nodiscard]] DistanceGpr<O2, P, C> with_optimizer(O2 optimizer) const
        {
            DistanceGpr<O2, P, C> next(kernel_, std::move(optimizer));
            next.likelihood_ = likelihood_;
            next.jitter_ = jitter_;
            next.input_transform_ = input_transform_->clone();
            next.target_transform_ = target_transform_->clone();
            next.kernel_exp_ = kernel_exp_;
            return next;
        }

        /// @brief Optimizes the hyperparameters and fits the model on supplied training distances.
        /// @param[in] train Square tables of squared distances between the training points, bound to the slots of
        /// the kernel.
        /// @param[in] y Training targets, one per training point.
        /// @return Fitted model, or GprError leaving this trainer unchanged.
        /// @details Checks, in this order: empty y, the structure of the tables (slots, emptiness, size products,
        /// shapes), y.size() against the table order, the number of ARD tables, the finiteness of y, and only then
        /// the table values; the table is validated once, not per objective evaluation.
        /// @retval GprErrorCode::EmptyInput When y or a table is empty.
        /// @retval GprErrorCode::InvalidConfig When the tidy tolerance of train is invalid.
        /// @retval GprErrorCode::UnsupportedKernelOperation When train binds a slot of another kernel.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated or of the wrong kind.
        /// @retval GprErrorCode::LengthMismatch When a table is not square, the tables differ in shape, or y.size()
        /// differs from the table order.
        /// @retval GprErrorCode::DimensionMismatch When an ARD slot supplies the wrong number of tables.
        /// @retval GprErrorCode::NonFiniteInput When y contains NaN or Inf.
        /// @retval GprErrorCode::InvalidDistance When a table is non-finite, negative, asymmetric or nonzero on the
        /// diagonal (unless repaired with DistanceSource::tidy()).
        /// @retval GprErrorCode::CholeskyFailed When factorization fails after the jitter policy is exhausted.
        [[nodiscard]] std::expected<FittedDistanceGpr<O, P, C>, GprError> fit(
            const DistanceSource& train, std::span<const double> y
        ) const
            requires std::same_as<C, DistanceOnly>
        {
            return run(train, std::nullopt, y, true);
        }

        /// @brief Optimizes the hyperparameters and fits a WithPoints model.
        /// @details Checks as the DistanceOnly overload, with x added: empty x with y, x.rows() with the table
        /// order, x.cols() against the coordinate leaves with the ARD table counts, and the finiteness of x with y.
        /// @param[in] train Square tables of squared distances between the training points.
        /// @param[in] x Training coordinates, n x d, for the coordinate leaves of the kernel.
        /// @param[in] y Training targets.
        /// @return Fitted model, or GprError leaving this trainer unchanged.
        /// @retval GprErrorCode::EmptyInput When y, x or a table is empty.
        /// @retval GprErrorCode::LengthMismatch When x.rows() or y.size() differs from the table order.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() does not match an ARD coordinate leaf.
        /// @retval GprErrorCode::NonFiniteInput When y or x contains NaN or Inf.
        [[nodiscard]] std::expected<FittedDistanceGpr<O, P, C>, GprError> fit(
            const DistanceSource& train, MatrixRef<double> x, std::span<const double> y
        ) const
            requires std::same_as<C, WithPoints>
        {
            return run(train, x, y, true);
        }

        /// @brief Factorizes at the current hyperparameters without optimizing.
        /// @param[in] train Square tables of squared distances between the training points.
        /// @param[in] y Training targets.
        /// @return Fitted model with a trivial optimizer trajectory, or GprError.
        [[nodiscard]] std::expected<FittedDistanceGpr<O, P, C>, GprError> factor(
            const DistanceSource& train, std::span<const double> y
        ) const
            requires std::same_as<C, DistanceOnly>
        {
            return run(train, std::nullopt, y, false);
        }

        /// @brief Factorizes a WithPoints model at the current hyperparameters without optimizing.
        /// @param[in] train Square tables of squared distances between the training points.
        /// @param[in] x Training coordinates, n x d.
        /// @param[in] y Training targets.
        /// @return Fitted model with a trivial optimizer trajectory, or GprError.
        [[nodiscard]] std::expected<FittedDistanceGpr<O, P, C>, GprError> factor(
            const DistanceSource& train, MatrixRef<double> x, std::span<const double> y
        ) const
            requires std::same_as<C, WithPoints>
        {
            return run(train, x, y, false);
        }

        /// @brief Returns the kernel carrying the initial hyperparameters.
        /// @return Kernel.
        [[nodiscard]] const DistanceKernel<C>& kernel() const noexcept { return kernel_; }

        /// @brief Returns the initial observation noise likelihood.
        /// @return Likelihood.
        [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept { return likelihood_; }

        /// @brief Returns the jitter policy.
        /// @return Jitter policy.
        [[nodiscard]] const JitterPolicy& jitter() const noexcept { return jitter_; }

        /// @brief Returns the optimizer.
        /// @return Optimizer instance.
        [[nodiscard]] const O& optimizer() const noexcept { return optimizer_; }

    private:
        template <typename O2, PrecisionPolicyConcept P2, DistanceModeConcept C2> friend class DistanceGpr;
        friend class FittedDistanceGpr<O, P, C>;

        /// @brief Trains a fitted model, optionally optimizing the hyperparameters first.
        /// @param[in] train Supplied squared distances of the training set.
        /// @param[in] x Optional training points (present in the with-points mode).
        /// @param[in] y Training targets.
        /// @param[in] optimize Whether to run the optimizer before the final fit.
        /// @return Fitted model, or GprError.
        [[nodiscard]] std::expected<FittedDistanceGpr<O, P, C>, GprError> run(
            const DistanceSource& train, std::optional<MatrixRef<double>> x, std::span<const double> y, bool optimize
        ) const
        {
            const detail::DistanceTrainConfig<C> cfg{
                .kernel = kernel_,
                .likelihood = likelihood_,
                .jitter = jitter_,
                .input_transform = *input_transform_,
                .target_transform = *target_transform_,
                .kernel_exp = kernel_exp_,
                .optimize = optimize && !std::same_as<O, optimizer::Fixed>,
                .use_change_indices = optimizer_uses_change_indices_v<O>,
                .value_only = !optimize || std::same_as<O, optimizer::Fixed> || std::same_as<O, optimizer::NelderMead>
                    || std::same_as<O, optimizer::FastSimulatedAnnealing>,
            };
            auto cb = [](void* ctx, optimizer::TwiceDifferentiable& obj,
                          std::span<const double> init) -> std::expected<optimizer::OptimizerResult, GprError>
            {
                return detail::invoke_optimizer(*static_cast<const O*>(ctx), obj, init);
            };
            auto payload = detail::train_distance_payload<P, C>(cfg, train, x, y, +cb, const_cast<O*>(&optimizer_));
            if (!payload.has_value())
            {
                return std::unexpected(std::move(payload).error());
            }
            return FittedDistanceGpr<O, P, C>(
                std::make_shared<const detail::DistanceFittedPayload<P, C>>(std::move(*payload)), optimizer_, jitter_,
                input_transform_->clone(), target_transform_->clone(), kernel_exp_
            );
        }

        /// @brief Initial kernel.
        DistanceKernel<C> kernel_;
        /// @brief Optimizer.
        O optimizer_;
        /// @brief Gaussian likelihood with the initial noise variance.
        GaussianLikelihood likelihood_;
        /// @brief Jitter policy.
        JitterPolicy jitter_;
        /// @brief Unfitted input transform.
        std::unique_ptr<transform::UnfittedInputTransform> input_transform_;
        /// @brief Unfitted target transform.
        std::unique_ptr<transform::UnfittedTargetTransform> target_transform_;
        /// @brief Kernel exponential evaluation mode.
        KernelExp kernel_exp_{KernelExp::Accurate};
    };

    /// @brief Converts the fitted model back into a trainer carrying the fitted hyperparameters.
    /// @return Trainer initialised from the fitted kernel, likelihood and configuration.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (distance only, or distances with points).
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
    DistanceGpr<O, P, C> FittedDistanceGpr<O, P, C>::into_trainer() const
    {
        DistanceGpr<O, P, C> trainer(payload_->kernel, optimizer_);
        trainer.likelihood_ = payload_->likelihood;
        trainer.jitter_ = jitter_;
        trainer.input_transform_ =
            input_transform_ ? input_transform_->clone() : std::make_unique<transform::IdentityInput>();
        trainer.target_transform_ = target_transform_->clone();
        trainer.kernel_exp_ = kernel_exp_;
        return trainer;
    }

    /// @brief Refits on new supplied distances starting from the fitted hyperparameters (distance-only mode).
    /// @param[in] train Supplied squared distances of the new training set.
    /// @param[in] y New training targets.
    /// @return Refitted model, or GprError.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (distance only, or distances with points).
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<FittedDistanceGpr<O, P, C>, GprError> FittedDistanceGpr<O, P, C>::refit(
        const DistanceSource& train, std::span<const double> y
    ) const
        requires std::same_as<C, DistanceOnly>
    {
        return into_trainer().fit(train, y);
    }

    /// @brief Refits on new supplied distances and points starting from the fitted hyperparameters (with-points mode).
    /// @param[in] train Supplied squared distances of the new training set.
    /// @param[in] x New training points.
    /// @param[in] y New training targets.
    /// @return Refitted model, or GprError.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode (distance only, or distances with points).
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<FittedDistanceGpr<O, P, C>, GprError> FittedDistanceGpr<O, P, C>::refit(
        const DistanceSource& train, MatrixRef<double> x, std::span<const double> y
    ) const
        requires std::same_as<C, WithPoints>
    {
        return into_trainer().fit(train, x, y);
    }

}  // namespace cxxgpr
