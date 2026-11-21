// SPDX-License-Identifier: Apache-2.0
/// @file predict_diagonal.hpp
/// @brief Internal prediction entry point for fitted and borrowed factors.
#pragma once

#include "cxxgpr/workspace/inference.hpp"

#include "linalg/factor_validation.hpp"

/// @brief Internal helpers for predicting with an owned Cholesky factor.
namespace cxxgpr::detail
{
    /// @brief Computes posterior mean and diagonal predictive variance into out using pre-allocated buffers.
    /// @details Calls compute_mean_and_v_into() to obtain mu_* and V = L^-1 * K_*, evaluates k(x*_j, x*_j)
    /// into workspace.query_kss without heap allocation, and forms max(0, k_** - ||v_j||_2^2) (+ sigma_n^2).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation variance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[in] validation Whether the model owns a verified factor.
    /// @param[out] out Destination Prediction structure populated with length-m mean and variance vectors.
    /// @return Empty expected on success, or GprError on invalid inputs, dimensions, or triangular solve failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> predict_diagonal_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        Prediction<typename P::Refine>& out,
        linalg::detail::FactorValidation validation
    );
}  // namespace cxxgpr::detail
