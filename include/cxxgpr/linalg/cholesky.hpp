// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <expected>
#include <span>

namespace cxxgpr::linalg {

/// Factors a symmetric positive-definite matrix A in-place into its lower Cholesky factor L
/// (A = L L^T), reading and writing only the lower triangle (uplo = Lower).
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError>
cholesky_lower_in_place(MatrixMut<T> a, CholeskyStage stage = CholeskyStage::Fit);

/// Factors A in-place trying j = 0 first and escalating via policy on failure using a
/// caller-provided n x n backup buffer (zero heap allocations). Returns the accepted factor_jitter.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<double, GprError>
cholesky_lower_with_jitter(MatrixMut<T> a, MatrixMut<T> backup, const JitterPolicy& policy,
                           CholeskyStage stage = CholeskyStage::Fit);

/// Factors A in-place trying j = 0 first and escalating via policy, allocating an internal backup.
/// Returns the accepted factor_jitter.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<double, GprError>
cholesky_lower_with_jitter(MatrixMut<T> a, const JitterPolicy& policy,
                           CholeskyStage stage = CholeskyStage::Fit);

/// Solves L * X = B in-place for a lower-triangular Cholesky factor L (uplo = Lower).
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, MatrixMut<T> rhs);

/// Solves L * x = b in-place for a lower-triangular Cholesky factor L and vector b.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, std::span<T> rhs);

/// Solves L^T * X = B in-place for a lower-triangular Cholesky factor L (uplo = Lower).
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l,
                                                                           MatrixMut<T> rhs);

/// Solves L^T * x = b in-place for a lower-triangular Cholesky factor L and vector b.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l,
                                                                           std::span<T> rhs);

/// Solves (L L^T) * X = B in-place for a lower-triangular Cholesky factor L (uplo = Lower).
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, MatrixMut<T> rhs);

/// Solves (L L^T) * x = b in-place for a lower-triangular Cholesky factor L and vector b.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, std::span<T> rhs);

/// Computes diag((L L^T)^-1) = ||L^-1 e_i||_2^2 using a caller-provided n x n scratch matrix
/// (zero heap allocations).
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError>
inv_diag_from_chol_l(MatrixRef<T> l, std::span<T> out_diag, MatrixMut<T> scratch_n_by_n);

/// Computes log|A| = sum_{i=0}^{n-1} 2 * log(L_ii) using Neumaier compensated summation.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<double, GprError> log_det_from_l(MatrixRef<T> l);

/// Computes 0.5 * log|A| = 0.5 * sum_{i=0}^{n-1} 2 * log(L_ii) using Neumaier compensated
/// summation.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<double, GprError> half_log_det_from_l(MatrixRef<T> l);

} // namespace cxxgpr::linalg
