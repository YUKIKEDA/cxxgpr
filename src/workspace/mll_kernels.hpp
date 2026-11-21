// SPDX-License-Identifier: Apache-2.0
/// @file mll_kernels.hpp
/// @brief Private dense kernels shared by the coordinate and the supplied-distance marginal-likelihood engines.
/// @details Holds the parallel construction of W = alpha * alpha^T - A^-1 from a Cholesky factor, the symmetric
/// lower-triangle helpers, and the deterministic Frobenius contractions used by gradients and Hessians.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/linalg/parallel.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <numbers>
#include <span>
#include <utility>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "linalg/factor_validation.hpp"
#include "linalg/packed_blas3.hpp"
#include "linalg/simd_blas3.hpp"

/// @brief Dense marginal-likelihood helper kernels shared by the coordinate and supplied-distance engines.
namespace cxxgpr::mll_kernels
{

    /// @brief Computes lower-triangular W = alpha * alpha^T - A^-1 directly from Cholesky factor L.
    /// @details Executes two area-balanced OpenMP stages with contiguous column-slice vectorization:
    /// 1. Forward-substitutes Z = L^-1 into z_scratch in parallel (n^3 / 3 FLOPs, unit-stride column AXPYs).
    /// 2. Forms lower-triangular W_ij = alpha_i * alpha_j - dot(Z_{i:, i}, Z_{i:, j}) in parallel with 4-row
    /// unrolling (n^3 / 3 FLOPs). Because Stage 2 reads only z_scratch and alpha, w_out may safely alias l
    /// under CholeskyBuffer::Reuse.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor L of A = L * L^T.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y (length n).
    /// @param[out] z_scratch Mutable n x n workspace receiving lower-triangular Z = L^-1.
    /// @param[out] w_out Mutable n x n destination matrix receiving lower-triangular W (may alias l).
    /// @return Empty expected on success, or GprError on invalid dimensions, non-finite inputs, or singular L.
    /// @retval GprErrorCode::EmptyInput When any matrix view or alpha span is empty or null.
    /// @retval GprErrorCode::LengthMismatch When dimensions are non-square or inconsistent across arguments.
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

        // Stage 1: Invert lower-triangular L into Z = L^-1 (n^3 / 3 FLOPs, column-independent) using blocked
        // 4-column SIMD panel/trailing updates matching 1-column remainders bit-for-bit across thread partitions.
        auto invert_range = [n, l, z_scratch](std::size_t col_begin, std::size_t col_end) mutable noexcept
        {
            constexpr std::size_t k_block = 32U;
            constexpr std::size_t row_tile = 256U;
            std::array<const T*, k_block> l_ptrs{};
            std::array<T, k_block> v0_arr{};
            std::array<T, k_block> v1_arr{};
            std::array<T, k_block> v2_arr{};
            std::array<T, k_block> v3_arr{};

            std::size_t j = col_begin;
            for (; j + 4U <= col_end; j += 4U)
            {
                const std::size_t j0 = j;
                const std::size_t j1 = j + 1U;
                const std::size_t j2 = j + 2U;
                const std::size_t j3 = j + 3U;
                T* const z0 = z_scratch.col(j0).data();
                T* const z1 = z_scratch.col(j1).data();
                T* const z2 = z_scratch.col(j2).data();
                T* const z3 = z_scratch.col(j3).data();

                for (std::size_t c = 0U; c < 4U; ++c)
                {
                    const std::size_t jc = j + c;
                    T* const zc = z_scratch.col(jc).data();
                    const T* const l_col_jc = l.col(jc).data();
                    const T z_jj = T{1} / l_col_jc[jc];
                    zc[jc] = z_jj;
                    detail::scale_neg_set_simd(zc, l_col_jc, z_jj, jc + 1U, n);
                }

                // Resolve columns [j0..j2] through row j3 so all 4 columns share synchronized kb blocks from j3
                // + 1.
                for (std::size_t c = 0U; c < 3U; ++c)
                {
                    const std::size_t jc = j + c;
                    T* const zc = z_scratch.col(jc).data();
                    const std::size_t lead_w = j3 - jc;
                    for (std::size_t p = 0U; p < lead_w; ++p)
                    {
                        const std::size_t k = jc + 1U + p;
                        const T* const lk = l.col(k).data();
                        const T zck = zc[k] / lk[k];
                        zc[k] = zck;
                        l_ptrs[p] = lk;
                        v0_arr[p] = zck;
                        for (std::size_t r = k + 1U; r <= j3; ++r)
                        {
                            zc[r] = detail::scalar_fnmadd(lk[r], zck, zc[r]);
                        }
                    }
                    for (std::size_t r_start = j3 + 1U; r_start < n; r_start += row_tile)
                    {
                        const std::size_t r_stop = std::min(r_start + row_tile, n);
                        detail::rank_k_update_1col_simd(zc, l_ptrs.data(), lead_w, v0_arr.data(), r_start, r_stop);
                    }
                }

                for (std::size_t kb = j3 + 1U; kb < n; kb += k_block)
                {
                    const std::size_t ke = std::min(kb + k_block, n);
                    const std::size_t bs = ke - kb;
                    for (std::size_t p = 0U; p < bs; ++p)
                    {
                        const std::size_t k = kb + p;
                        const T* const lk = l.col(k).data();
                        const T lkk = lk[k];
                        const T v0p = z0[k] / lkk;
                        const T v1p = z1[k] / lkk;
                        const T v2p = z2[k] / lkk;
                        const T v3p = z3[k] / lkk;
                        z0[k] = v0p;
                        z1[k] = v1p;
                        z2[k] = v2p;
                        z3[k] = v3p;
                        l_ptrs[p] = lk;
                        v0_arr[p] = v0p;
                        v1_arr[p] = v1p;
                        v2_arr[p] = v2p;
                        v3_arr[p] = v3p;
                        for (std::size_t r = k + 1U; r < ke; ++r)
                        {
                            const T lrk = lk[r];
                            z0[r] = detail::scalar_fnmadd(lrk, v0p, z0[r]);
                            z1[r] = detail::scalar_fnmadd(lrk, v1p, z1[r]);
                            z2[r] = detail::scalar_fnmadd(lrk, v2p, z2[r]);
                            z3[r] = detail::scalar_fnmadd(lrk, v3p, z3[r]);
                        }
                    }
                    if (ke < n)
                    {
                        for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            detail::rank_k_update_4cols_simd(
                                z0, z1, z2, z3, l_ptrs.data(), bs, v0_arr.data(), v1_arr.data(), v2_arr.data(),
                                v3_arr.data(), r_start, r_stop
                            );
                        }
                    }
                }
            }

            for (; j < col_end; ++j)
            {
                T* const z_col = z_scratch.col(j).data();
                const T* const l_col_j = l.col(j).data();
                const T z_jj = T{1} / l_col_j[j];
                z_col[j] = z_jj;
                detail::scale_neg_set_simd(z_col, l_col_j, z_jj, j + 1U, n);
                for (std::size_t kb = j + 1U; kb < n; kb += k_block)
                {
                    const std::size_t ke = std::min(kb + k_block, n);
                    const std::size_t bs = ke - kb;
                    for (std::size_t p = 0U; p < bs; ++p)
                    {
                        const std::size_t k = kb + p;
                        const T* const lk = l.col(k).data();
                        const T z_kj = z_col[k] / lk[k];
                        z_col[k] = z_kj;
                        l_ptrs[p] = lk;
                        v0_arr[p] = z_kj;
                        for (std::size_t r = k + 1U; r < ke; ++r)
                        {
                            z_col[r] = detail::scalar_fnmadd(lk[r], z_kj, z_col[r]);
                        }
                    }
                    if (ke < n)
                    {
                        for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            detail::rank_k_update_1col_simd(z_col, l_ptrs.data(), bs, v0_arr.data(), r_start, r_stop);
                        }
                    }
                }
            }
        };
        if (n >= 384U)
        {
            const std::size_t col_block = (n >= 768U) ? 24U : 12U;
            const std::size_t n_col_blocks = (n + col_block - 1U) / col_block;
            const auto n_col_blocks_s = static_cast<std::ptrdiff_t>(n_col_blocks);
            const std::size_t tile = (n >= 768U) ? detail::k_blas3_mc : 24U;
            const std::size_t n_tile_blocks = (n + tile - 1U) / tile;
            const std::size_t total_tiles = (n_tile_blocks * (n_tile_blocks + 1U)) / 2U;
            const auto total_tiles_s = static_cast<std::ptrdiff_t>(total_tiles);
#ifdef _OPENMP
#    pragma omp parallel if (linalg::openmp_max_threads() > 1U)
#endif
            {
                // Stage 1: Strip-wise packed inversion of Z = L^-1 (implicit barrier ensures Z is complete).
#ifdef _OPENMP
#    pragma omp for schedule(dynamic, 1)
#endif
                for (std::ptrdiff_t b = 0; b < n_col_blocks_s; ++b)
                {
                    const std::size_t col_begin = static_cast<std::size_t>(b) * col_block;
                    const std::size_t col_end = std::min(col_begin + col_block, n);
                    detail::invert_lower_col_strip_packed(l, z_scratch, col_begin, col_end);
                }

                // Stage 2: Lower-triangular GEMM contraction W = alpha * alpha^T - Z^T * Z.
                alignas(64) std::array<T, detail::k_blas3_mc * detail::k_blas3_nc> c_tile{};
                alignas(64) std::array<T, detail::k_blas3_pack_a_cap<T>> packed_a{};
                alignas(64) std::array<T, detail::k_blas3_pack_b_cap> packed_b{};
#ifdef _OPENMP
#    pragma omp for schedule(dynamic, 1)
#endif
                for (std::ptrdiff_t t_s = 0; t_s < total_tiles_s; ++t_s)
                {
                    const auto coords = detail::lower_tile_coords(static_cast<std::size_t>(t_s), n_tile_blocks);
                    const std::size_t bi = coords.row_block;
                    const std::size_t bj = coords.col_block;
                    const std::size_t ib = bi * tile;
                    const std::size_t jb = bj * tile;
                    const std::size_t rm = std::min(tile, n - ib);
                    const std::size_t cn = std::min(tile, n - jb);
                    std::fill_n(c_tile.data(), rm * cn, T{0});

                    for (std::size_t kb = ib; kb < n; kb += detail::k_blas3_kc)
                    {
                        const std::size_t bs = std::min(detail::k_blas3_kc, n - kb);
                        detail::pack_panel_a_transpose(
                            &z_scratch(kb, ib), z_scratch.col_stride(), rm, bs, packed_a.data(), (kb == ib) ? rm : 0U
                        );
                        detail::pack_panel_b_colmajor(
                            &z_scratch(kb, jb), z_scratch.col_stride(), bs, cn, packed_b.data(), (kb == jb) ? cn : 0U
                        );
                        detail::gemm_packed_add(rm, cn, bs, packed_a.data(), packed_b.data(), c_tile.data(), rm);
                    }

                    // Subtract accumulated Z^T * Z from alpha * alpha^T in explicit FMA order over the lower tile.
                    for (std::size_t c = 0U; c < cn; ++c)
                    {
                        const std::size_t col = jb + c;
                        const T alpha_j = alpha[col];
                        const std::size_t r_start = (bi == bj) ? c : 0U;
                        const T* const tile_col = c_tile.data() + (c * rm);
                        T* const w_col = w_out.col(col).data();
                        for (std::size_t r = r_start; r < rm; ++r)
                        {
                            const std::size_t row = ib + r;
                            w_col[row] = detail::scalar_fmadd(alpha[row], alpha_j, -tile_col[r]);
                        }
                    }
                }
            }
            return {};
        }
        if (n >= 96U)
        {
            linalg::parallel_for_lower_blocks(n, 0U, invert_range);
        }
        else
        {
            invert_range(0U, n);
        }

        // Stage 2: Contract lower-triangular W_ij = alpha_i * alpha_j - <Z_{i:, i}, Z_{i:, j}>.
        // Row-block boundaries are anchored strictly to absolute 4-row multiples (row % 4 == 0) so any OpenMP
        // column split yields bitwise-identical rounding across all thread counts while using 4x2 SIMD kernels.
        auto contract_range = [n, alpha, z_scratch, w_out](std::size_t col_begin, std::size_t col_end) mutable noexcept
        {
            auto eval_single = [n, &alpha, &z_scratch, &w_out](std::size_t row, std::size_t col_idx) noexcept
            {
                const T* const z_row = z_scratch.col(row).data();
                const T* const z_col = z_scratch.col(col_idx).data();
                const T acc = detail::dot_1x1_simd(z_row + row, z_col + row, n - row);
                w_out.col(col_idx)[row] = detail::scalar_fmadd(alpha[row], alpha[col_idx], -acc);
            };

            auto finish_4rows = [&alpha](
                                    T* w_col, const T* z0, const T* z1, const T* z2, const T* zj, std::size_t i0,
                                    T alpha_j, T tail0, T tail1, T tail2, T tail3
                                ) noexcept
            {
                const std::size_t i1 = i0 + 1U;
                const std::size_t i2 = i0 + 2U;
                const std::size_t i3 = i0 + 3U;
                const T sum0 = detail::scalar_fmadd(
                    z0[i0], zj[i0], detail::scalar_fmadd(z0[i1], zj[i1], detail::scalar_fmadd(z0[i2], zj[i2], tail0))
                );
                const T sum1 = detail::scalar_fmadd(z1[i1], zj[i1], detail::scalar_fmadd(z1[i2], zj[i2], tail1));
                const T sum2 = detail::scalar_fmadd(z2[i2], zj[i2], tail2);
                w_col[i0] = detail::scalar_fmadd(alpha[i0], alpha_j, -sum0);
                w_col[i1] = detail::scalar_fmadd(alpha[i1], alpha_j, -sum1);
                w_col[i2] = detail::scalar_fmadd(alpha[i2], alpha_j, -sum2);
                w_col[i3] = detail::scalar_fmadd(alpha[i3], alpha_j, -tail3);
            };

            auto eval_4x1 =
                [n, &alpha, &z_scratch, &w_out, &finish_4rows](std::size_t row, std::size_t col_idx) noexcept
            {
                const std::size_t i0 = row;
                const std::size_t i1 = row + 1U;
                const std::size_t i2 = row + 2U;
                const std::size_t i3 = row + 3U;
                const T* const z0 = z_scratch.col(i0).data();
                const T* const z1 = z_scratch.col(i1).data();
                const T* const z2 = z_scratch.col(i2).data();
                const T* const z3 = z_scratch.col(i3).data();
                const T* const zj = z_scratch.col(col_idx).data();
                std::array<T, 4> tail_dots{};
                detail::dot_4x1_simd(z0 + i3, z1 + i3, z2 + i3, z3 + i3, zj + i3, n - i3, tail_dots);
                finish_4rows(
                    w_out.col(col_idx).data(), z0, z1, z2, zj, i0, alpha[col_idx], tail_dots[0], tail_dots[1],
                    tail_dots[2], tail_dots[3]
                );
            };

            std::size_t j = col_begin;
            for (; j + 2U <= col_end; j += 2U)
            {
                const std::size_t j0 = j;
                const std::size_t j1 = j + 1U;
                const std::size_t r4_0 = std::min(n, (j0 + 3U) & ~std::size_t{3U});
                const std::size_t r4_1 = std::min(n, (j1 + 3U) & ~std::size_t{3U});

                for (std::size_t row = j0; row < r4_0; ++row)
                {
                    eval_single(row, j0);
                }
                std::size_t row_pair_start = r4_1;
                if (r4_0 < r4_1)
                {
                    if (r4_0 + 4U <= n)
                    {
                        eval_4x1(r4_0, j0);
                    }
                    else
                    {
                        for (std::size_t row = r4_0; row < n; ++row)
                        {
                            eval_single(row, j0);
                        }
                        row_pair_start = n;
                    }
                }
                for (std::size_t row = j1; row < std::min(n, row_pair_start); ++row)
                {
                    eval_single(row, j1);
                }

                const T* const z_j0 = z_scratch.col(j0).data();
                const T* const z_j1 = z_scratch.col(j1).data();
                T* const w_col0 = w_out.col(j0).data();
                T* const w_col1 = w_out.col(j1).data();
                const T alpha_j0 = alpha[j0];
                const T alpha_j1 = alpha[j1];

                std::size_t row = row_pair_start;
                for (; row + 4U <= n; row += 4U)
                {
                    const std::size_t i0 = row;
                    const std::size_t i1 = row + 1U;
                    const std::size_t i2 = row + 2U;
                    const std::size_t i3 = row + 3U;
                    const T* const z0 = z_scratch.col(i0).data();
                    const T* const z1 = z_scratch.col(i1).data();
                    const T* const z2 = z_scratch.col(i2).data();
                    const T* const z3 = z_scratch.col(i3).data();

                    std::array<std::array<T, 2>, 4> tail_dots{};
                    detail::dot_4x2_simd(z0 + i3, z1 + i3, z2 + i3, z3 + i3, z_j0 + i3, z_j1 + i3, n - i3, tail_dots);

                    finish_4rows(
                        w_col0, z0, z1, z2, z_j0, i0, alpha_j0, tail_dots[0][0], tail_dots[1][0], tail_dots[2][0],
                        tail_dots[3][0]
                    );
                    finish_4rows(
                        w_col1, z0, z1, z2, z_j1, i0, alpha_j1, tail_dots[0][1], tail_dots[1][1], tail_dots[2][1],
                        tail_dots[3][1]
                    );
                }
                for (; row < n; ++row)
                {
                    eval_single(row, j0);
                    eval_single(row, j1);
                }
            }

            for (; j < col_end; ++j)
            {
                const std::size_t r4 = std::min(n, (j + 3U) & ~std::size_t{3U});
                for (std::size_t row = j; row < r4; ++row)
                {
                    eval_single(row, j);
                }
                std::size_t row = r4;
                for (; row + 4U <= n; row += 4U)
                {
                    eval_4x1(row, j);
                }
                for (; row < n; ++row)
                {
                    eval_single(row, j);
                }
            }
        };
        if (n >= 96U)
        {
            linalg::parallel_for_lower_blocks(n, 0U, contract_range);
        }
        else
        {
            contract_range(0U, n);
        }

        return {};
    }

    /// @brief Populates the lower triangle of a square matrix view with diag_val * I_n.
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

    /// @brief Computes u = A_sym * alpha reading only the lower triangle of symmetric matrix a_lower.
    /// @details Traverses lower-triangular entries once to accumulate both row and symmetric column
    /// contributions into u without touching the strictly upper triangle of a_lower.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a_lower Square read-only view whose lower triangle stores symmetric matrix A_i.
    /// @param[in] alpha Input vector of length n.
    /// @param[out] u Mutable output vector of length n receiving A_i * alpha.
    template <KernelScalarConcept T>
    void gemv_sym_lower(MatrixRef<T> a_lower, std::span<const T> alpha, std::span<T> u) noexcept
    {
        const std::size_t n = alpha.size();
        std::ranges::fill(u, T{0});
        for (std::size_t col = 0; col < n; ++col)
        {
            const T a_col = alpha[col];
            u[col] += a_lower(col, col) * a_col;
            T acc = T{0};
            for (std::size_t row = col + 1U; row < n; ++row)
            {
                const T v = a_lower(row, col);
                // Apply lower entry (row, col) and mirrored upper entry (col, row) simultaneously.
                u[row] += v * a_col;
                acc += v * alpha[row];
            }
            u[col] += acc;
        }
    }

    /// @brief Transposes a square column-major matrix view in place.
    /// @details Swaps strictly lower and strictly upper entries for col < row.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] mat Square mutable n x n column-major matrix view.
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
    /// @param[in,out] col_partials Pre-allocated length-n scratch span for deterministic per-column reduction.
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
                ? detail::dot_storage_to_refine_simd<T, double>(
                      a.col(col).data() + col + 1U, b.col(col).data() + col + 1U, tail_len
                  )
                : 0.0;
            col_partials[col] = diag + (2.0 * off);
        }
        // Sum per-column partials sequentially in ascending column order for deterministic OpenMP results.
        double total = 0.0;
        for (std::size_t col = 0; col < n; ++col)
        {
            total += col_partials[col];
        }
        return total;
    }

    /// @brief Computes deterministic full-matrix Frobenius inner product <a, b>_F = tr(a^T * b).
    /// @details Accumulates each column inner product in double precision into col_partials in parallel,
    /// then folds columns sequentially in ascending order for bitwise determinism.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a First n x n column-major matrix view.
    /// @param[in] b Second n x n column-major matrix view.
    /// @param[in,out] col_partials Pre-allocated length-n scratch span for deterministic per-column reduction.
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
            col_partials[col] = detail::dot_storage_to_refine_simd<T, double>(a.col(col).data(), b.col(col).data(), n);
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
