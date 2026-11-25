// SPDX-License-Identifier: Apache-2.0
/// @file packed_blas3.hpp
/// @brief Internal GotoBLAS/BLIS-style cache-packed SIMD BLAS-3 kernels and blocked Cholesky/TRSM helpers.
/// @details Provides stack-packed micro-panel routines, FMA-saturated register-blocked GEMM/SYRK microkernels,
/// lower-triangular column-strip inversion, and blocked panel/multi-RHS triangular solvers with strict ascending-k
/// accumulation order and zero heap allocation.
#pragma once

#include "cxxgpr/core/views.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>

#include "linalg/simd_blas3.hpp"

/// @brief Internal implementation helpers for cache-packed SIMD linear algebra.
namespace cxxgpr::detail
{

    /// @brief Row panel height MC chosen to keep packed A resident in L1/L2 cache and divide by all ISA MR widths.
    inline constexpr std::size_t k_blas3_mc = 48U;

    /// @brief Column panel width NC divisible by all ISA NR widths (8, 6, 4) so B packing fits without overflow.
    inline constexpr std::size_t k_blas3_nc = 48U;

    /// @brief Shared inner-dimension block size KC selected for zero L1D set-conflict thrashing inside microkernels.
    inline constexpr std::size_t k_blas3_kc = 64U;

    /// @brief Stack element capacity required for packing an MC x KC panel of scalar type T across all x86-64 ISAs.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    template <SimdBlasScalar T>
    inline constexpr std::size_t k_blas3_pack_a_cap = (std::same_as<T, float> ? 64U : k_blas3_mc) * k_blas3_kc;

    /// @brief Stack element capacity required for packing a KC x NC panel of scalar type T across all x86-64 ISAs.
    inline constexpr std::size_t k_blas3_pack_b_cap = k_blas3_nc * k_blas3_kc;

    /// @brief Block coordinates indexing a symmetric lower-triangular tile (`row_block >= col_block`).
    struct LowerTileCoords
    {
        /// @brief Zero-based block row index satisfying `row_block >= col_block`.
        std::size_t row_block{0U};
        /// @brief Zero-based block column index in `[0, n_blocks)`.
        std::size_t col_block{0U};
    };

    /// @brief Maps a column-major lower-triangular linear tile index to `(row_block, col_block)` coordinates.
    /// @details Orders tiles down each block column `col_block` so consecutive work-stealing indices maximize packed-B
    /// panel reuse while keeping tile geometry independent of OpenMP thread count.
    /// @param[in] linear_idx Linear tile index in `[0, n_blocks * (n_blocks + 1) / 2)`.
    /// @param[in] n_blocks Number of blocks along the matrix dimension.
    /// @return Matching `(row_block, col_block)` block coordinates with `row_block >= col_block`.
    [[nodiscard]] inline LowerTileCoords lower_tile_coords(std::size_t linear_idx, std::size_t n_blocks) noexcept
    {
        // Binary search cumulative column-major tile counts before column cb: cb * (2*n_blocks - cb + 1) / 2.
        std::size_t lo = 0U;
        std::size_t hi = n_blocks;
        while (lo + 1U < hi)
        {
            const std::size_t mid = lo + ((hi - lo) / 2U);
            const std::size_t start = (mid * ((2U * n_blocks) - mid + 1U)) / 2U;
            if (start <= linear_idx)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        const std::size_t col_start = (lo * ((2U * n_blocks) - lo + 1U)) / 2U;
        const std::size_t row_block = lo + (linear_idx - col_start);
        return {.row_block = row_block, .col_block = lo};
    }

    /// @brief Packs an `m x k` column-major matrix A into ISA-aligned MR-tall micro-panels.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] src Pointer to top-left element `(0, 0)` of the column-major `A` block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] m Number of rows to pack (`m <= k_blas3_mc`).
    /// @param[in] k Number of columns to pack (`k <= k_blas3_kc`).
    /// @param[out] dst Output buffer with capacity at least `k_blas3_pack_a_cap<T>`.
    template <SimdBlasScalar T>
    void pack_panel_a_colmajor(const T* src, std::size_t ld, std::size_t m, std::size_t k, T* dst) noexcept;

    /// @brief Packs the transpose of a `k x m` column-major matrix into ISA-aligned MR-tall micro-panels.
    /// @details Reads `src[r * ld + p]` as `A(r, p)` without reading strictly upper entries when `diag_mask_rows > 0`.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `k x m` column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] m Number of rows of `A` (columns of `src`), at most `k_blas3_mc`.
    /// @param[in] k Number of columns of `A` (rows of `src`), at most `k_blas3_kc`.
    /// @param[out] dst Output buffer with capacity at least `k_blas3_pack_a_cap<T>`.
    /// @param[in] diag_mask_rows If non-zero, masks entries with `p < r` for `r < diag_mask_rows` to zero without
    /// reading `src`.
    template <SimdBlasScalar T>
    void pack_panel_a_transpose(
        const T* src, std::size_t ld, std::size_t m, std::size_t k, T* dst, std::size_t diag_mask_rows = 0U
    ) noexcept;

    /// @brief Packs the transpose of a `k x m` float matrix directly into `double` MR-tall micro-panels.
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `k x m` float column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] m Number of output rows (columns of `src`), at most `k_blas3_mc`.
    /// @param[in] k Number of inner columns (rows of `src`), at most `k_blas3_kc`.
    /// @param[out] dst Output double buffer with capacity at least `k_blas3_pack_a_cap<double>`.
    void pack_panel_a_transpose_promote_f32_to_f64(
        const float* src, std::size_t ld, std::size_t m, std::size_t k, double* dst
    ) noexcept;

    /// @brief Packs the transpose of an `n x k` column-major matrix into ISA-aligned NR-wide B micro-panels.
    /// @details Reads `src[p * ld + c]` as `B(p, c)` for symmetric rank-k `L * L^T` updates.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `n x k` column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] k Number of rows of `B` (columns of `src`), at most `k_blas3_kc`.
    /// @param[in] n Number of columns of `B` (rows of `src`), at most `k_blas3_nc`.
    /// @param[out] dst Output buffer with capacity at least `k_blas3_pack_b_cap`.
    template <SimdBlasScalar T>
    void pack_panel_b_colmajor_t(const T* src, std::size_t ld, std::size_t k, std::size_t n, T* dst) noexcept;

    /// @brief Packs a `k x n` column-major matrix into ISA-aligned NR-wide B micro-panels.
    /// @details Optionally masks strictly upper entries (`p < c` for `c < diag_mask_cols`) to zero without reading
    /// `src`.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `k x n` column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] k Number of rows of `B` to pack (`k <= k_blas3_kc`).
    /// @param[in] n Number of columns of `B` to pack (`n <= k_blas3_nc`).
    /// @param[out] dst Output buffer with capacity at least `k_blas3_pack_b_cap`.
    /// @param[in] diag_mask_cols If non-zero, masks entries with `p < c` for `c < diag_mask_cols` to zero without
    /// reading `src`.
    template <SimdBlasScalar T>
    void pack_panel_b_colmajor(
        const T* src, std::size_t ld, std::size_t k, std::size_t n, T* dst, std::size_t diag_mask_cols = 0U
    ) noexcept;

    /// @brief Packs a `k x n` column-major float matrix directly into `double` NR-wide B micro-panels.
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `k x n` float column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] k Number of inner rows to pack (`k <= k_blas3_kc`).
    /// @param[in] n Number of output columns to pack (`n <= k_blas3_nc`).
    /// @param[out] dst Output double buffer with capacity at least `k_blas3_pack_b_cap`.
    void pack_panel_b_colmajor_promote_f32_to_f64(
        const float* src, std::size_t ld, std::size_t k, std::size_t n, double* dst
    ) noexcept;

    /// @brief Computes `C -= A_packed * B_packed` on an `m x n` column-major matrix block in ascending `k` order.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] m Number of rows in `C` (`m <= k_blas3_mc`).
    /// @param[in] n Number of columns in `C` (`n <= k_blas3_nc`).
    /// @param[in] k Inner reduction dimension (`k <= k_blas3_kc`).
    /// @param[in] packed_a Micro-panel packed `A` block produced by a `pack_panel_a*` routine.
    /// @param[in] packed_b Micro-panel packed `B` block produced by a `pack_panel_b*` routine.
    /// @param[in,out] c Pointer to top-left element `(0, 0)` of the `m x n` destination block.
    /// @param[in] ldc Leading dimension (column stride in elements) of `c`.
    template <SimdBlasScalar T>
    void gemm_packed_sub(
        std::size_t m, std::size_t n, std::size_t k, const T* packed_a, const T* packed_b, T* c, std::size_t ldc
    ) noexcept;

    /// @brief Computes `C += A_packed * B_packed` on an `m x n` column-major matrix block in ascending `k` order.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] m Number of rows in `C` (`m <= k_blas3_mc`).
    /// @param[in] n Number of columns in `C` (`n <= k_blas3_nc`).
    /// @param[in] k Inner reduction dimension (`k <= k_blas3_kc`).
    /// @param[in] packed_a Micro-panel packed `A` block produced by a `pack_panel_a*` routine.
    /// @param[in] packed_b Micro-panel packed `B` block produced by a `pack_panel_b*` routine.
    /// @param[in,out] c Pointer to top-left element `(0, 0)` of the `m x n` destination block.
    /// @param[in] ldc Leading dimension (column stride in elements) of `c`.
    template <SimdBlasScalar T>
    void gemm_packed_add(
        std::size_t m, std::size_t n, std::size_t k, const T* packed_a, const T* packed_b, T* c, std::size_t ldc
    ) noexcept;

    /// @brief Computes lower-triangle-only `C -= A_packed * B_packed` on a square diagonal block without touching upper
    /// entries.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] m Square diagonal tile order (`m <= k_blas3_mc`).
    /// @param[in] k Inner reduction dimension (`k <= k_blas3_kc`).
    /// @param[in] packed_a Micro-panel packed `A` block for rows `[0, m)`.
    /// @param[in] packed_b Micro-panel packed `B` block for columns `[0, m)`.
    /// @param[in,out] c Pointer to top-left element `(0, 0)` of the square diagonal destination block.
    /// @param[in] ldc Leading dimension (column stride in elements) of `c`.
    template <SimdBlasScalar T>
    void syrk_packed_lower_diag_sub(
        std::size_t m, std::size_t k, const T* packed_a, const T* packed_b, T* c, std::size_t ldc
    ) noexcept;

    /// @brief Solves trailing Cholesky panel rows `A[r_begin:r_end, k:k+bs]` in stack-packed L1 strips.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in,out] a Target symmetric matrix view holding factored `L[k:k+bs, k:k+bs]` and trailing panel rows.
    /// @param[in] k Starting row and column index of the active diagonal panel.
    /// @param[in] bs Panel width (at most `k_blas3_kc`).
    /// @param[in] r_begin Inclusive starting row owned by the caller.
    /// @param[in] r_end Exclusive ending row owned by the caller.
    template <SimdBlasScalar T>
    void solve_panel_rows_packed(
        MatrixMut<T> a, std::size_t k, std::size_t bs, std::size_t r_begin, std::size_t r_end
    ) noexcept;

    /// @brief Solves `L[row_start:n, row_start:n] * X[row_start:n, col_begin:col_end] = B` in place via cache-packed
    /// BLAS-3.
    /// @details Executes diagonal solves via division + ascending-`p` FMA matching single-RHS rounding bitwise while
    /// updating trailing rows with L1-packed `gemm_packed_sub` microkernels. Never reads the strictly upper triangle of
    /// `l`.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] l Square read-only lower Cholesky factor matrix view.
    /// @param[in,out] rhs Mutable right-hand side matrix view updated in place.
    /// @param[in] row_start Inclusive leading row and column index where the triangular solve begins.
    /// @param[in] col_begin Inclusive starting right-hand side column index.
    /// @param[in] col_end Exclusive ending right-hand side column index.
    template <SimdBlasScalar T>
    void solve_lower_block_packed(
        MatrixRef<T> l, MatrixMut<T> rhs, std::size_t row_start, std::size_t col_begin, std::size_t col_end
    ) noexcept;

    /// @brief Computes columns `[col_begin, col_end)` of lower-triangular `Z = L^-1` via cache-packed BLAS-3.
    /// @details Inverts the diagonal lower block in stack scratch, forms `-L21 * Z11`, and delegates `L22 * Z21 = -L21
    /// * Z11` to `solve_lower_block_packed`. Never reads or writes strictly upper entries of `l` or `z`.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] l Square read-only lower Cholesky factor matrix view.
    /// @param[out] z Mutable `n x n` destination matrix view receiving lower-triangular `Z = L^-1` in `[col_begin,
    /// col_end)`.
    /// @param[in] col_begin Inclusive starting column index in `z`.
    /// @param[in] col_end Exclusive ending column index in `z` (`col_end - col_begin <= k_blas3_nc`).
    template <SimdBlasScalar T>
    void invert_lower_col_strip_packed(
        MatrixRef<T> l, MatrixMut<T> z, std::size_t col_begin, std::size_t col_end
    ) noexcept;

}  // namespace cxxgpr::detail
