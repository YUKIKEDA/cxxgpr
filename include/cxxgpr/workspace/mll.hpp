// SPDX-License-Identifier: Apache-2.0
/// @file mll.hpp
/// @brief Workspace-backed negative marginal log-likelihood, gradient, and analytical Hessian evaluation.
/// @details Declares factor_and_mll(), mll_and_gradient_into(), hessian_into(), and
/// value_gradient_hessian_into() operating over pre-allocated FitBuffers<P> with deterministic
/// reductions, single-pass keep_plan gradient evaluation, and CholeskyBuffer::Reuse restoration.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/workspace/buffers.hpp"

#include <cstddef>
#include <expected>
#include <span>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Assembles A = K + sigma_n^2 I, factors L, solves alpha, and returns the negative MLL.
    /// @details Evaluates K (using DistanceCache when enabled) while retaining keep_products product
    /// factor Grams in buffers.core.weighted, adds sigma_n^2 I, computes L * L^T = A with jitter
    /// retries, solves A * alpha = y in buffers.core.rhs, and returns
    /// 0.5 * (y^T * alpha + log|A| + n * log(2 * pi)).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers receiving L, alpha, factor_jitter, and kept factor Grams.
    /// @param[in] keep_products Number of top-level product terms whose factor Grams are kept for gradients.
    /// @return Negative marginal log-likelihood, or GprError on invalid input or factorization failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When x.rows() or y.size() differs from buffers.core.order().
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When distance cache or workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization fails across the full jitter ladder.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<double, GprError> factor_and_mll(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::size_t keep_products = 0
    );

    /// @brief Computes negative MLL and analytical log-space gradients in a single pass via keep_plan.
    /// @details Factors A while keeping factor Grams, builds W = alpha * alpha^T - A^-1 in
    /// buffers.w_mut(), contracts kernel parameter derivatives in one tree walk via weighted_grads(),
    /// evaluates the log-noise gradient -0.5 * sigma_n^2 * tr(W), and restores L if CholeskyBuffer::Reuse
    /// overwrote k_matrix.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers receiving updated L, alpha, W, and scratch state.
    /// @param[out] out_grad Destination span of length kernel.num_params() + 1 receiving log-space gradients
    /// (kernel parameters followed by log(sigma_n^2)).
    /// @return Negative marginal log-likelihood, or GprError on invalid inputs, size mismatch, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_grad.size() != kernel.num_params() + 1 or n mismatches.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When workspace or cache sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization (or CholeskyBuffer::Reuse restoration) fails.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<double, GprError> mll_and_gradient_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::span<double> out_grad
    );

    /// @brief Computes the analytical log-space Hessian matrix of the negative marginal log-likelihood.
    /// @details Factors A via factor_and_mll() and computes the symmetric p x p Hessian using
    /// H_ij = -0.5 * tr(W * A_ij) - 0.5 * tr(S_i * S_j) + v_i^T * v_j, where W = alpha * alpha^T - A^-1,
    /// S_i = L^-1 * A_i * L^-T, and v_i = L^-1 * A_i * alpha.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers (core.hessian is lazily allocated on first call).
    /// @param[out] out_hess Destination span of length p * p (p = kernel.num_params() + 1) receiving the
    /// symmetric Hessian matrix in flattened column/row-major order.
    /// @return Empty expected on success, or GprError on invalid inputs, size mismatch, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_hess.size() != p * p or n mismatches.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When HessianScratch or distance cache sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization or restoration fails.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<void, GprError> hessian_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::span<double> out_hess
    );

    /// @brief Computes negative MLL, analytical gradient, and analytical Hessian together in one pass.
    /// @details Evaluates negative MLL and gradient via mll_and_gradient_into(), then populates out_hess
    /// from the already factored Cholesky decomposition and alpha solution.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers updated with factor state and Hessian scratch.
    /// @param[out] out_grad Destination span of length p = kernel.num_params() + 1 receiving log-space gradients.
    /// @param[out] out_hess Destination span of length p * p receiving the symmetric log-space Hessian.
    /// @return Negative marginal log-likelihood, or GprError on invalid inputs, size mismatch, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_grad or out_hess length disagrees with p or n mismatches.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When HessianScratch or distance cache sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization or restoration fails.
    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<double, GprError> value_gradient_hessian_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::span<double> out_grad,
        std::span<double> out_hess
    );

    extern template std::expected<double, GprError> factor_and_mll<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::size_t
    );
    extern template std::expected<double, GprError> factor_and_mll<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::size_t
    );
    extern template std::expected<double, GprError> factor_and_mll<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::size_t
    );
    extern template std::expected<double, GprError> factor_and_mll<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::size_t
    );

    extern template std::expected<double, GprError> mll_and_gradient_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::span<double>
    );
    extern template std::expected<double, GprError> mll_and_gradient_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::span<double>
    );
    extern template std::expected<double, GprError> mll_and_gradient_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::span<double>
    );
    extern template std::expected<double, GprError> mll_and_gradient_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::span<double>
    );

    extern template std::expected<void, GprError> hessian_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::span<double>
    );
    extern template std::expected<void, GprError> hessian_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::span<double>
    );
    extern template std::expected<void, GprError> hessian_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::span<double>
    );
    extern template std::expected<void, GprError> hessian_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::span<double>
    );

    extern template std::expected<double, GprError> value_gradient_hessian_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::span<double>,
        std::span<double>
    );
    extern template std::expected<double, GprError> value_gradient_hessian_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::span<double>,
        std::span<double>
    );
    extern template std::expected<double, GprError> value_gradient_hessian_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::span<double>,
        std::span<double>
    );
    extern template std::expected<double, GprError> value_gradient_hessian_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::span<double>,
        std::span<double>
    );

}  // namespace cxxgpr
