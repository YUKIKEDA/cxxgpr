// SPDX-License-Identifier: Apache-2.0
/// @file inference.hpp
/// @brief Workspace-backed posterior prediction, covariance, sampling, and LOO evaluation routines.
/// @details Declares precision-parameterized Exact GPR inference functions over pre-factored lower
/// Cholesky factors L, pre-solved weights alpha, and reusable QueryWorkspace<P> buffers.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/workspace/buffers.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Computes posterior mean and diagonal predictive variance into out using pre-allocated buffers.
    /// @details Evaluates cross-covariance K_* = k(X, X_*), forms posterior mean mu_* = K_*^T * alpha,
    /// solves V = L^-1 * K_* in place, and evaluates diagonal variance max(0, k(x_*, x_*) - ||v||_2^2)
    /// (+ sigma_n^2 when options.variance_kind == VarianceKind::Observation) with zero heap allocation
    /// after workspace and output buffers are warmed up.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation variance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[out] out Destination Prediction structure populated with length-m mean and variance vectors.
    /// @return Empty expected on success, or GprError on invalid inputs, dimensions, or triangular solve failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<void, GprError> predict_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        Prediction<typename P::Refine>& out
    );

    /// @brief Computes posterior mean and diagonal predictive variance for query points x_query.
    /// @details Allocates a fresh Prediction<P::Refine> result and delegates to predict_into().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation variance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @return Populated Prediction<P::Refine>, or GprError if predict_into() fails.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError> predict(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace
    );

    /// @brief Computes posterior mean and full m x m column-major posterior covariance matrix into out.
    /// @details Evaluates mu_* = K_*^T * alpha, solves V = L^-1 * K_*, evaluates the prior query Gram
    /// matrix K_**, and assembles the symmetric posterior covariance Sigma_* = K_** - V^T * V
    /// (+ sigma_n^2 * I when options.variance_kind == VarianceKind::Observation).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation covariance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[out] out Destination PredictiveCovariance structure populated with mean and m x m covariance.
    /// @return Empty expected on success, or GprError on invalid inputs, dimensions, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When m * m or workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<void, GprError> predict_covariance_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        PredictiveCovariance<typename P::Refine>& out
    );

    /// @brief Computes posterior mean and full m x m column-major posterior covariance matrix.
    /// @details Allocates a fresh PredictiveCovariance<P::Refine> result and delegates to predict_covariance_into().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation covariance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @return Populated PredictiveCovariance<P::Refine>, or GprError if predict_covariance_into() fails.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When m * m or workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<PredictiveCovariance<typename P::Refine>, GprError> predict_covariance(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace
    );

    /// @brief Draws n_draws deterministic posterior samples from the predictive distribution.
    /// @details Computes predictive mean and covariance via predict_covariance(), decomposes Sigma_* =
    /// L_* * L_*^T with Cholesky jitter fallback (CholeskyStage::Predict), and transforms Box-Muller
    /// standard normals from Xoshiro256PlusPlus(seed) as y_draw = mu_* + L_* * z packed at draw * m + i.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] n_draws Number of posterior function or observation draws to generate.
    /// @param[in] seed Deterministic 64-bit seed passed to Xoshiro256PlusPlus.
    /// @param[in] options Prediction configuration selecting latent or observation distribution.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[in] jitter Optional Cholesky jitter policy for Sigma_*; defaults to adaptive(1e-12, 10.0, 10, 1e-2).
    /// @return Vector of length n_draws * m (empty if n_draws == 0), or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When n_draws * m or m * m overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When posterior covariance factorization fails after jitter retries.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<std::vector<typename P::Refine>, GprError> sample(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        std::optional<JitterPolicy> jitter = std::nullopt
    );

    /// @brief Computes closed-form Leave-One-Out predictive mean and variance (GPML §5.4.2) into out.
    /// @details Computes diagonal precision entries q_ii = [A^-1]_ii = ||L^-1 e_i||_2^2 via
    /// linalg::inv_diag_from_chol_l(), then evaluates mu_LOO,i = y_i - alpha_i / q_ii and
    /// sigma^2_LOO,i = 1 / q_ii (subtracting sigma_n^2 and clamping at 0 for VarianceKind::Latent),
    /// performing zero heap allocations when out already has capacity >= n.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] y_train Transformed training target vector y in refine precision (length n).
    /// @param[in] options Prediction configuration selecting latent or observation LOO variance.
    /// @param[in,out] scratch_n_by_n Caller-provided n x n mutable workspace for L^-1 computation.
    /// @param[in,out] scratch_diag Caller-provided length-n mutable span receiving [A^-1]_ii.
    /// @param[out] out Destination Prediction structure populated with LOO means and variances.
    /// @return Empty expected on success, or GprError on invalid inputs, buffers, or non-positive q_ii.
    /// @retval GprErrorCode::EmptyInput When l_factor, alpha, or y_train is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is non-square or alpha/y_train/scratch sizes mismatch.
    /// @retval GprErrorCode::NonFiniteInput When alpha or y_train contains NaN or +/-Inf.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor is singular or yields q_ii <= 0.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<void, GprError> loo_predict_into(
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        std::span<const typename P::Refine> y_train,
        const PredictOptions& options,
        MatrixMut<typename P::Storage> scratch_n_by_n,
        std::span<typename P::Storage> scratch_diag,
        Prediction<typename P::Refine>& out
    );

    /// @brief Computes closed-form Leave-One-Out predictive mean and variance (GPML §5.4.2).
    /// @details Allocates temporary n x n and length-n scratch buffers and delegates to loo_predict_into().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] y_train Transformed training target vector y in refine precision (length n).
    /// @param[in] options Prediction configuration selecting latent or observation LOO variance.
    /// @return Populated LOO Prediction<P::Refine>, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When l_factor, alpha, or y_train is empty.
    /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l_factor is non-square or alpha/y_train lengths mismatch.
    /// @retval GprErrorCode::NonFiniteInput When alpha or y_train contains NaN or +/-Inf.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor is singular or yields q_ii <= 0.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError> loo_predict(
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        std::span<const typename P::Refine> y_train,
        const PredictOptions& options
    );

    extern template std::expected<void, GprError> predict_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        Prediction<double>&
    );
    extern template std::expected<void, GprError> predict_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        Prediction<float>&
    );
    extern template std::expected<void, GprError> predict_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        Prediction<double>&
    );
    extern template std::expected<void, GprError> predict_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        Prediction<double>&
    );

    extern template std::expected<Prediction<double>, GprError> predict<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&
    );
    extern template std::expected<Prediction<float>, GprError> predict<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&
    );
    extern template std::expected<Prediction<double>, GprError> predict<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&
    );
    extern template std::expected<Prediction<double>, GprError> predict<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
    );

    extern template std::expected<void, GprError> predict_covariance_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        PredictiveCovariance<double>&
    );
    extern template std::expected<void, GprError> predict_covariance_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        PredictiveCovariance<float>&
    );
    extern template std::expected<void, GprError> predict_covariance_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        PredictiveCovariance<double>&
    );
    extern template std::expected<void, GprError> predict_covariance_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        PredictiveCovariance<double>&
    );

    extern template std::expected<PredictiveCovariance<double>, GprError> predict_covariance<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&
    );
    extern template std::expected<PredictiveCovariance<float>, GprError> predict_covariance<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&
    );
    extern template std::expected<PredictiveCovariance<double>, GprError> predict_covariance<
        MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&
    );
    extern template std::expected<PredictiveCovariance<double>, GprError> predict_covariance<
        MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
    );

    extern template std::expected<std::vector<double>, GprError> sample<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        std::optional<JitterPolicy>
    );
    extern template std::expected<std::vector<float>, GprError> sample<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        std::optional<JitterPolicy>
    );
    extern template std::expected<std::vector<double>, GprError> sample<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        std::optional<JitterPolicy>
    );
    extern template std::expected<std::vector<double>, GprError> sample<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        std::optional<JitterPolicy>
    );

    extern template std::expected<void, GprError> loo_predict_into<DoublePrecision>(
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&,
        MatrixMut<double>,
        std::span<double>,
        Prediction<double>&
    );
    extern template std::expected<void, GprError> loo_predict_into<SinglePrecision>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        std::span<const float>,
        const PredictOptions&,
        MatrixMut<float>,
        std::span<float>,
        Prediction<float>&
    );
    extern template std::expected<void, GprError> loo_predict_into<MixedPrecision<PromoteStorage>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&,
        MatrixMut<float>,
        std::span<float>,
        Prediction<double>&
    );
    extern template std::expected<void, GprError> loo_predict_into<MixedPrecision<ReevaluateKernel>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&,
        MatrixMut<float>,
        std::span<float>,
        Prediction<double>&
    );

    extern template std::expected<Prediction<double>, GprError> loo_predict<DoublePrecision>(
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&
    );
    extern template std::expected<Prediction<float>, GprError> loo_predict<SinglePrecision>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        std::span<const float>,
        const PredictOptions&
    );
    extern template std::expected<Prediction<double>, GprError> loo_predict<MixedPrecision<PromoteStorage>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&
    );
    extern template std::expected<Prediction<double>, GprError> loo_predict<MixedPrecision<ReevaluateKernel>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&
    );

}  // namespace cxxgpr
