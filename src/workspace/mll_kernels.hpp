// SPDX-License-Identifier: Apache-2.0
/// @file mll_kernels.hpp
/// @brief Dense marginal-likelihood linear-algebra helper kernels backed by external BLAS.
/// @details Provides zero-allocation routines to construct lower-triangular W = alpha * alpha^T - A^-1,
/// compute symmetric matrix-vector products, and evaluate deterministic column-partitioned Frobenius products.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <span>
#include <utility>

#include "linalg/blas_backend.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

/// @brief Internal marginal-likelihood linear-algebra routines backed by external BLAS.
namespace cxxgpr::mll_kernels
{

    /// @brief Computes lower triangle of W = alpha * alpha^T - (L L^T)^-1 with zero heap allocations.
    /// @details Delegates to `blas_compute_w_lower` across BuiltIn SIMD, MKL LAPACKE potri, or blocked OpenBLAS.
    /// @tparam T Storage floating-point scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square n x n lower Cholesky factor view with strictly positive finite diagonal.
    /// @param[in] alpha Precomputed solve vector A^-1 y of length n.
    /// @param[out] z_scratch Preallocated n x n scratch matrix view overwritten with Z = L^-1.
    /// @param[out] w_out Preallocated n x n output matrix view receiving lower-triangular W.
    /// @return Empty expected on success, or GprError on empty views, shape mismatch, or invalid L diagonal.
    /// @retval GprErrorCode::EmptyInput When any view or span is empty or null.
    /// @retval GprErrorCode::LengthMismatch When matrices are non-square or dimensions disagree with l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any diagonal entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> compute_w_lower_from_cholesky(
        MatrixRef<T> l, std::span<const T> alpha, MatrixMut<T> z_scratch, MatrixMut<T> w_out
    )
    {
        if (l.is_empty() || l.data() == nullptr || z_scratch.is_empty() || z_scratch.data() == nullptr
            || w_out.is_empty() || w_out.data() == nullptr || alpha.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = l.rows();
        if (!l.is_square() || z_scratch.rows() != n || z_scratch.cols() != n || w_out.rows() != n || w_out.cols() != n
            || alpha.size() != n)
        {
            return std::unexpected(GprError::length_mismatch("matrix dimension mismatch when computing W"));
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const T diag = l(i, i);
            if (!std::isfinite(diag))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            if (diag <= T{0})
            {
                return std::unexpected(GprError::non_positive_definite_matrix());
            }
        }

        cxxgpr::detail::blas_compute_w_lower(
            n, l.data(), l.col_stride(), alpha.data(), z_scratch.data(), z_scratch.col_stride(), w_out.data(),
            w_out.col_stride()
        );
        return {};
    }

    /// @brief Fills diagonal entries of square matrix mat with diag_val and zeros the strictly lower triangle.
    /// @details Writes diag_val on the diagonal and 0 strictly below it, representing dA/d(log sigma_n^2)
    /// and d^2A/d(log sigma_n^2)^2 = sigma_n^2 * I_n.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[out] mat Square mutable n x n column-major matrix view.
    /// @param[in] diag_val Scalar multiplier applied to main diagonal entries.
    template <KernelScalarConcept T> void fill_diagonal_scaled(MatrixMut<T> mat, T diag_val) noexcept
    {
        const std::size_t n = mat.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col; row < n; ++row)
            {
                mat(row, col) = (row == col) ? diag_val : T{0};
            }
        }
    }

    /// @brief Computes symmetric matrix-vector product u = A * alpha reading only the lower triangle of A.
    /// @details Delegates to CBLAS symmetric matrix-vector multiply (SYMV) with uplo = Lower.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a_lower Square symmetric matrix view with entries populated in its lower triangle.
    /// @param[in] alpha Input vector of length a_lower.rows().
    /// @param[out] u Output vector of length a_lower.rows() receiving A * alpha.
    template <KernelScalarConcept T>
    void gemv_sym_lower(MatrixRef<T> a_lower, std::span<const T> alpha, std::span<T> u) noexcept
    {
        const std::size_t n = alpha.size();
        cxxgpr::detail::blas_symv_lower(n, T{1}, a_lower.data(), a_lower.col_stride(), alpha.data(), T{0}, u.data());
    }

    /// @brief Transposes a square column-major matrix in place by swapping off-diagonal pairs.
    /// @details Swaps mat(row, col) and mat(col, row) for all col < row with zero scratch storage.
    /// @tparam T Matrix element scalar type satisfying KernelScalarConcept.
    /// @param[in,out] mat Square mutable matrix view transposed in place.
    template <KernelScalarConcept T> void transpose_square_in_place(MatrixMut<T> mat) noexcept
    {
        const std::size_t n = mat.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col + 1U; row < n; ++row)
            {
                const std::size_t r1 = row;
                const std::size_t r2 = col;
                std::swap(mat(r1, r2), mat(r2, r1));
            }
        }
    }

    /// @brief Computes deterministic symmetric Frobenius inner product <a, b>_F from lower triangles.
    /// @details Evaluates column sums diag + 2 * strictly_lower in double precision in parallel into
    /// col_partials, then folds columns in fixed ascending order for bitwise thread-count invariance.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a First symmetric n x n matrix view (lower triangle read).
    /// @param[in] b Second symmetric n x n matrix view (lower triangle read).
    /// @param[out] col_partials Scratch span of length >= a.rows() for per-column partial sums.
    /// @return Trace(a * b) computed deterministically in double precision.
    template <KernelScalarConcept T>
    [[nodiscard]] double frobenius_lower_det(MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials) noexcept
    {
        const std::size_t n = a.rows();
        const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
        for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
        {
            const auto col = static_cast<std::size_t>(col_s);
            const auto diag = static_cast<double>(a(col, col)) * static_cast<double>(b(col, col));
            const std::size_t tail_len = n - (col + 1U);
            const double off = (tail_len > 0U)
                ? cxxgpr::detail::dot_promoted_f64<T>(
                      a.col(col).data() + col + 1U, b.col(col).data() + col + 1U, tail_len
                  )
                : 0.0;
            col_partials[col] = diag + (2.0 * off);
        }
        // Fold column sums in ascending index order for bitwise thread-count invariance.
        double total = 0.0;
        for (std::size_t col = 0; col < n; ++col)
        {
            total += col_partials[col];
        }
        return total;
    }

    /// @brief Computes deterministic full-matrix Frobenius inner product <a, b>_F = tr(a^T * b).
    /// @details Accumulates each column inner product in double precision into col_partials in parallel,
    /// then sums col_partials sequentially in ascending column order.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a Left n x n matrix view.
    /// @param[in] b Right n x n matrix view.
    /// @param[out] col_partials Scratch span of length >= a.rows() for per-column partial sums.
    /// @return Full Frobenius inner product in double precision.
    template <KernelScalarConcept T>
    [[nodiscard]] double frobenius_full_det(MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials) noexcept
    {
        const std::size_t n = a.rows();
        const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
        for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
        {
            const auto col = static_cast<std::size_t>(col_s);
            col_partials[col] = cxxgpr::detail::dot_promoted_f64<T>(a.col(col).data(), b.col(col).data(), n);
        }
        // Fold column sums in ascending index order for bitwise thread-count invariance.
        double total = 0.0;
        for (std::size_t col = 0; col < n; ++col)
        {
            total += col_partials[col];
        }
        return total;
    }

}  // namespace cxxgpr::mll_kernels
