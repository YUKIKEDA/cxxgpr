// SPDX-License-Identifier: Apache-2.0
/// @file cholesky.hpp
/// @brief In-place Cholesky factorization, jittered retries, triangular solves, and log-determinants.
/// @details Declares symmetric positive-definite Cholesky decompositions reading and writing only the lower triangle,
/// along with forward/backward triangular solves, inverse diagonal computation, and Neumaier-compensated
/// log-determinant helpers.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <expected>
#include <span>

/// @brief Dense and factorized linear algebra primitives for cxxgpr.
namespace cxxgpr::linalg
{

    /// @brief Factors a symmetric positive-definite matrix A in place into its lower Cholesky factor L.
    /// @details Computes A = L * L^T while reading and overwriting only the lower triangle (including diagonal); the
    /// strictly upper triangle is untouched.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] a Square mutable matrix view containing the lower triangle of A on input and L on success.
    /// @param[in] stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Empty expected on success, or GprError on invalid input or non-positive-definite matrix.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When factorization fails or yields non-positive/non-finite diagonal
    /// entries.
    /// @pre `a` must be square with `col_stride() >= rows()` and finite lower triangle.
    /// @warning On Cholesky failure without jitter recovery, the lower triangle of `a` may be partially overwritten.
    /// @see cholesky_lower_with_jitter(), solve_llt_in_place()
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> cholesky_lower_in_place(
        MatrixMut<T> a, CholeskyStage stage = CholeskyStage::Fit
    );

    /// @brief Factors A in place trying j = 0 first and escalating via policy using a caller-provided backup buffer.
    /// @details Saves the lower triangle of A into backup once, tries j = 0, and on each retry restores the clean lower
    /// triangle from backup before adding the candidate jitter to the diagonal (zero heap allocations).
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] a Square mutable matrix view factored in place into L.
    /// @param[out] backup Preallocated n x n square mutable matrix view used to preserve the original lower triangle.
    /// @param[in] policy Diagonal jitter retry policy evaluated if j = 0 fails.
    /// @param[in] stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Accepted diagonal jitter (0.0 if unjittered factorization succeeded), or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When a or backup is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on a or backup.
    /// @retval GprErrorCode::LengthMismatch When stride < rows, either view is non-square, or dimensions differ.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When j = 0 and all jittered retry attempts fail.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<double, GprError> cholesky_lower_with_jitter(
        MatrixMut<T> a, MatrixMut<T> backup, const JitterPolicy& policy, CholeskyStage stage = CholeskyStage::Fit
    );

    /// @brief Factors A in place trying j = 0 first and escalating via policy with an internal backup buffer.
    /// @details Allocates a temporary n x n backup matrix internally and delegates to the zero-allocation overload.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] a Square mutable matrix view factored in place into L.
    /// @param[in] policy Diagonal jitter retry policy evaluated if j = 0 fails.
    /// @param[in] stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Accepted diagonal jitter (0.0 if unjittered factorization succeeded), or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols or rows * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When j = 0 and all jittered retry attempts fail.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<double, GprError> cholesky_lower_with_jitter(
        MatrixMut<T> a, const JitterPolicy& policy, CholeskyStage stage = CholeskyStage::Fit
    );

    /// @brief Solves L * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Reads only the lower triangle of L and overwrites rhs with the solution X without heap allocation.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and the solution X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, MatrixMut<T> rhs);

    /// @brief Solves L * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Wraps rhs as an n x 1 matrix view and delegates to the matrix overload.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable vector span holding b on entry and the solution x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, std::span<T> rhs);

    /// @brief Solves L^T * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Reads only the lower triangle of L (interpreted transposed as upper triangular) and overwrites rhs with
    /// X.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and the solution X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, MatrixMut<T> rhs);

    /// @brief Solves L^T * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Wraps rhs as an n x 1 matrix view and delegates to the matrix overload.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable vector span holding b on entry and the solution x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, std::span<T> rhs);

    /// @brief Solves (L * L^T) * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Executes forward substitution L * Y = B followed by backward substitution L^T * X = Y in place.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and the solution X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, MatrixMut<T> rhs);

    /// @brief Solves (L * L^T) * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Wraps rhs as an n x 1 matrix view and delegates to the matrix overload.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable vector span holding b on entry and the solution x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, std::span<T> rhs);

    /// @brief Computes diag((L * L^T)^-1) using a caller-provided n x n scratch matrix.
    /// @details Solves L * X = I_n in scratch_n_by_n and accumulates column squared norms ||L^-1 e_i||_2^2 in double
    /// precision with zero heap allocations.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[out] out_diag Preallocated output span of length l.rows() receiving the inverse diagonal entries.
    /// @param[in,out] scratch_n_by_n Caller-provided n x n mutable workspace overwritten during computation.
    /// @return Empty expected on success, or GprError on invalid factor, output length, or scratch buffer.
    /// @retval GprErrorCode::EmptyInput When l, out_diag, or scratch_n_by_n is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or scratch_n_by_n.
    /// @retval GprErrorCode::LengthMismatch When l or scratch_n_by_n is non-square, strides are invalid,
    /// out_diag.size() != l.rows(), or scratch_n_by_n.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> inv_diag_from_chol_l(
        MatrixRef<T> l, std::span<T> out_diag, MatrixMut<T> scratch_n_by_n
    );

    /// @brief Computes log|A| = sum_{i=0}^{n-1} 2 * log(L_ii) using Neumaier compensated summation.
    /// @details Accumulates in double precision to reduce floating-point cancellation across large factors.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @return Log-determinant of A = L * L^T in double precision, or GprError on invalid factor.
    /// @retval GprErrorCode::EmptyInput When l is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square or has invalid stride.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T> [[nodiscard]] std::expected<double, GprError> log_det_from_l(MatrixRef<T> l);

    /// @brief Computes 0.5 * log|A| from a lower Cholesky factor L using Neumaier compensated summation.
    /// @details Evaluates 0.5 * log_det_from_l(l) for log marginal likelihood calculations.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @return Half log-determinant of A = L * L^T in double precision, or GprError on invalid factor.
    /// @retval GprErrorCode::EmptyInput When l is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square or has invalid stride.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is non-positive.
    template <KernelScalarConcept T> [[nodiscard]] std::expected<double, GprError> half_log_det_from_l(MatrixRef<T> l);

}  // namespace cxxgpr::linalg
