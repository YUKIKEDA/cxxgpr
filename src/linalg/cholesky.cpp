// SPDX-License-Identifier: Apache-2.0
/// @file cholesky.cpp
/// @brief Implementations of in-place Cholesky factorization, jitter retries, solves, and log-determinants.
/// @details Wraps Eigen in-place LLT decomposition and deterministic SIMD triangular solvers over column-major
/// MatrixRef/MatrixMut views, with backup-restoring jitter escalation and Neumaier compensated log-determinant
/// summation.

#include "cxxgpr/linalg/cholesky.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/linalg/parallel.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "linalg/eigen_map.hpp"
#include "linalg/factor_validation.hpp"
#include "linalg/packed_blas3.hpp"
#include "linalg/simd_blas3.hpp"

namespace cxxgpr::linalg
{

    namespace
    {

        /// @brief Column count shared by direct Cholesky panels and their scratch arrays.
        constexpr std::size_t k_direct_panel_block = 32U;

        /// @brief Validates a square lower-triangular Cholesky factor L prior to solves or determinant queries.
        /// @details Enforces square view geometry, lower-triangle finiteness, and strictly positive diagonal entries.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] l Square read-only Cholesky factor view.
        /// @return Empty expected on success, or GprError if l is invalid or singular.
        /// @retval GprErrorCode::EmptyInput When l is empty or null.
        /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When col_stride < rows or l is non-square.
        /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
        /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_lower_factor(MatrixRef<T> l)
        {
            // Enforce validation order: square view geometry -> lower finiteness -> positive diagonal.
            if (const auto check = detail::validate_matrix_view(l, true); !check.has_value())
            {
                return check;
            }
            if (const auto fin = detail::require_lower_finite(l); !fin.has_value())
            {
                return fin;
            }
            const std::size_t n = l.rows();
            for (std::size_t i = 0; i < n; ++i)
            {
                if (l(i, i) <= T{0})
                {
                    return std::unexpected(GprError::non_positive_definite_matrix());
                }
            }
            return {};
        }

        /// @brief Validates a Cholesky factor L and a mutable right-hand side matrix B before in-place solves.
        /// @details Checks L via validate_lower_factor, verifies rhs geometry and row count parity, and checks rhs
        /// finiteness.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] l Square read-only lower Cholesky factor view.
        /// @param[in] validation Whether the factor was already checked by the model producer.
        /// @param[in] rhs Mutable right-hand side matrix view.
        /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
        /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
        /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
        /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
        /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
        /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_solve_inputs(
            MatrixRef<T> l, MatrixMut<T> rhs, detail::FactorValidation validation = detail::FactorValidation::Check
        )
        {
            // Validate factor first, then right-hand side geometry, row parity, and finiteness.
            if (validation == detail::FactorValidation::Check)
            {
                if (const auto l_check = validate_lower_factor(l); !l_check.has_value())
                {
                    return l_check;
                }
            }
            if (const auto rhs_check = detail::validate_matrix_view(rhs, false); !rhs_check.has_value())
            {
                return rhs_check;
            }
            if (rhs.rows() != l.rows())
            {
                return std::unexpected(
                    GprError::length_mismatch("right-hand side row count must match factor dimension")
                );
            }
            if (const auto fin = detail::require_matrix_finite(detail::as_ref(rhs)); !fin.has_value())
            {
                return fin;
            }
            return {};
        }

        /// @brief Factors diagonal submatrix a[k:k+bs, k:k+bs] in place via Eigen::LLT and validates diagonal pivots.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] a Target n x n matrix view whose submatrix [k, k + bs) is factored in place.
        /// @param[in] k Starting row and column index of the submatrix.
        /// @param[in] bs Submatrix dimension (number of rows and columns).
        /// @return True on success, false if Eigen::LLT fails or produces a non-positive/non-finite pivot.
        template <KernelScalarConcept T>
        [[nodiscard]] bool factor_subblock_eigen(MatrixMut<T> a, std::size_t k, std::size_t bs)
        {
            const auto bs_idx = static_cast<Eigen::Index>(bs);
            detail::MutEigenMap<T> sub_map(
                &a(k, k), bs_idx, bs_idx, Eigen::OuterStride<>(static_cast<Eigen::Index>(a.col_stride()))
            );
            using RefType = Eigen::Ref<detail::ColMajorDynamic<T>, 0, Eigen::OuterStride<>>;
            Eigen::LLT<RefType, Eigen::Lower> llt(sub_map);
            if (llt.info() != Eigen::Success)
            {
                return false;
            }
            for (std::size_t i = 0; i < bs; ++i)
            {
                const T diag = a(k + i, k + i);
                if (!std::isfinite(diag) || diag <= T{0})
                {
                    return false;
                }
            }
            return true;
        }

        /// @brief Solves lower-panel rows [r_begin, r_end) x cols [k, k + bs) against the factored diagonal block.
        /// @details Tiles trailing panel rows in cache-friendly chunks of 64 and executes fused SIMD column updates
        /// via rank_k_update_1col_simd() and scale_in_place_simd() while preserving ascending p order.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] a Target n x n matrix view owning factored diagonal block and trailing panel rows.
        /// @param[in] k Starting column of the active panel.
        /// @param[in] bs Width of the active panel (at most 32).
        /// @param[in] r_begin Inclusive starting row owned by the calling thread.
        /// @param[in] r_end Exclusive ending row owned by the calling thread.
        template <KernelScalarConcept T>
        void solve_panel_rows(
            MatrixMut<T> a, std::size_t k, std::size_t bs, std::size_t r_begin, std::size_t r_end
        ) noexcept
        {
            constexpr std::size_t row_tile = 64U;
            std::array<const T*, k_direct_panel_block> l_ptrs{};
            std::array<T, k_direct_panel_block> coeffs{};
            for (std::size_t r_tile = r_begin; r_tile < r_end; r_tile += row_tile)
            {
                const std::size_t r_stop = std::min(r_tile + row_tile, r_end);
                const std::size_t tile_len = r_stop - r_tile;
                for (std::size_t col = 0U; col < bs; ++col)
                {
                    const std::size_t gc = k + col;
                    T* const col_c = a.col(gc).data();
                    for (std::size_t p = 0U; p < col; ++p)
                    {
                        l_ptrs[p] = a.col(k + p).data();
                        coeffs[p] = a(gc, k + p);
                    }
                    if (col > 0U)
                    {
                        cxxgpr::detail::rank_k_update_1col_simd(
                            col_c, l_ptrs.data(), col, coeffs.data(), r_tile, r_stop
                        );
                    }
                    const T inv_diag = T{1} / col_c[gc];
                    cxxgpr::detail::scale_in_place_simd(col_c + r_tile, tile_len, inv_diag);
                }
            }
        }

        /// @brief Applies the trailing symmetric rank-bs update to lower-triangular columns [j_begin, j_end).
        /// @details Tiles rows in cache-resident blocks of 256 and delegates 4-column blocks and trailing 1-column
        /// remainders to 8-accumulator register-blocked SIMD kernels while preserving exact ascending p = 0..bs-1
        /// accumulation order for OpenMP thread-count bitwise invariance.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] a Target n x n matrix view whose trailing lower columns [j_begin, j_end) are updated.
        /// @param[in] k Starting column of the finished panel.
        /// @param[in] bs Panel width (at most 32).
        /// @param[in] j_begin Inclusive first trailing column assigned to this thread.
        /// @param[in] j_end Exclusive last trailing column assigned to this thread.
        /// @param[in] n Matrix order.
        template <KernelScalarConcept T>
        void rank_k_lower_update(
            MatrixMut<T> a, std::size_t k, std::size_t bs, std::size_t j_begin, std::size_t j_end, std::size_t n
        ) noexcept
        {
            constexpr std::size_t row_tile = 256U;
            std::array<const T*, k_direct_panel_block> l_ptrs{};
            std::array<T, k_direct_panel_block> v0_arr{};
            std::array<T, k_direct_panel_block> v1_arr{};
            std::array<T, k_direct_panel_block> v2_arr{};
            std::array<T, k_direct_panel_block> v3_arr{};

            std::size_t j = j_begin;
            for (; j + 4U <= j_end; j += 4U)
            {
                const std::size_t j0 = j;
                const std::size_t j1 = j + 1U;
                const std::size_t j2 = j + 2U;
                const std::size_t j3 = j + 3U;
                T* const col0 = a.col(j0).data();
                T* const col1 = a.col(j1).data();
                T* const col2 = a.col(j2).data();
                T* const col3 = a.col(j3).data();

                // Update the 6 upper entries in rows [j0, j3) and pack panel multipliers for SIMD trailing blocks.
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const T* const lp = a.col(k + p).data();
                    const T v0 = lp[j0];
                    const T v1 = lp[j1];
                    const T v2 = lp[j2];
                    l_ptrs[p] = lp;
                    v0_arr[p] = v0;
                    v1_arr[p] = v1;
                    v2_arr[p] = v2;
                    v3_arr[p] = lp[j3];
                    col0[j0] = cxxgpr::detail::scalar_fnmadd(v0, v0, col0[j0]);
                    col0[j1] = cxxgpr::detail::scalar_fnmadd(v1, v0, col0[j1]);
                    col0[j2] = cxxgpr::detail::scalar_fnmadd(v2, v0, col0[j2]);
                    col1[j1] = cxxgpr::detail::scalar_fnmadd(v1, v1, col1[j1]);
                    col1[j2] = cxxgpr::detail::scalar_fnmadd(v2, v1, col1[j2]);
                    col2[j2] = cxxgpr::detail::scalar_fnmadd(v2, v2, col2[j2]);
                }

                // Update rows [j3, n) across all 4 columns in register-blocked SIMD row tiles.
                for (std::size_t r_start = j3; r_start < n; r_start += row_tile)
                {
                    const std::size_t r_stop = std::min(r_start + row_tile, n);
                    cxxgpr::detail::rank_k_update_4cols_simd(
                        col0, col1, col2, col3, l_ptrs.data(), bs, v0_arr.data(), v1_arr.data(), v2_arr.data(),
                        v3_arr.data(), r_start, r_stop
                    );
                }
            }
            for (; j < j_end; ++j)
            {
                T* const col_j = a.col(j).data();
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const T* const lp = a.col(k + p).data();
                    l_ptrs[p] = lp;
                    v0_arr[p] = lp[j];
                }
                for (std::size_t r_start = j; r_start < n; r_start += row_tile)
                {
                    const std::size_t r_stop = std::min(r_start + row_tile, n);
                    cxxgpr::detail::rank_k_update_1col_simd(col_j, l_ptrs.data(), bs, v0_arr.data(), r_start, r_stop);
                }
            }
        }

        /// @brief Selects triangular system solve direction or full symmetric L * L^T solve.
        enum class TriSolveMode : std::uint8_t
        {
            /// @brief Solves L * X = B in place.
            Forward,
            /// @brief Solves L^T * X = B in place.
            Backward,
            /// @brief Solves (L * L^T) * X = B in place.
            Llt,
        };

        /// @brief Solves `L * X = B` in place across column range `[col_begin, col_end)` via blocked SIMD panel/rank-k
        /// updates.
        /// @details Processes 4 RHS columns simultaneously (`rank_k_update_4cols_simd`) followed by 1-column
        /// remainders (`rank_k_update_1col_simd`) in fixed `k_block = 32` panels and `row_tile = 256` L1 strips without
        /// runtime CPU cache queries or reading the strictly upper triangle of `l`.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] l Square read-only lower Cholesky factor matrix view.
        /// @param[in,out] rhs Mutable right-hand side matrix view updated in place.
        /// @param[in] col_begin Inclusive starting column index in `rhs`.
        /// @param[in] col_end Exclusive ending column index in `rhs`.
        template <KernelScalarConcept T>
        void solve_lower_cols(MatrixRef<T> l, MatrixMut<T> rhs, std::size_t col_begin, std::size_t col_end) noexcept
        {
            const std::size_t n = l.rows();
            if (n >= 64U && (col_end - col_begin) >= 6U)
            {
                cxxgpr::detail::solve_lower_block_packed(l, rhs, 0U, col_begin, col_end);
                return;
            }
            constexpr std::size_t k_block = 32U;
            constexpr std::size_t row_tile = 256U;
            std::array<const T*, k_block> l_ptrs{};
            std::array<T, k_block> v0_arr{};
            std::array<T, k_block> v1_arr{};
            std::array<T, k_block> v2_arr{};
            std::array<T, k_block> v3_arr{};

            std::size_t c = col_begin;
            for (; c + 4U <= col_end; c += 4U)
            {
                T* const x0 = rhs.col(c).data();
                T* const x1 = rhs.col(c + 1U).data();
                T* const x2 = rhs.col(c + 2U).data();
                T* const x3 = rhs.col(c + 3U).data();

                for (std::size_t kb = 0U; kb < n; kb += k_block)
                {
                    const std::size_t ke = std::min(kb + k_block, n);
                    const std::size_t bs = ke - kb;
                    for (std::size_t p = 0U; p < bs; ++p)
                    {
                        const std::size_t k = kb + p;
                        const T* const lk = l.col(k).data();
                        const T lkk = lk[k];
                        const T v0p = x0[k] / lkk;
                        const T v1p = x1[k] / lkk;
                        const T v2p = x2[k] / lkk;
                        const T v3p = x3[k] / lkk;
                        x0[k] = v0p;
                        x1[k] = v1p;
                        x2[k] = v2p;
                        x3[k] = v3p;
                        l_ptrs[p] = lk;
                        v0_arr[p] = v0p;
                        v1_arr[p] = v1p;
                        v2_arr[p] = v2p;
                        v3_arr[p] = v3p;
                        for (std::size_t r = k + 1U; r < ke; ++r)
                        {
                            const T lrk = lk[r];
                            x0[r] = cxxgpr::detail::scalar_fnmadd(lrk, v0p, x0[r]);
                            x1[r] = cxxgpr::detail::scalar_fnmadd(lrk, v1p, x1[r]);
                            x2[r] = cxxgpr::detail::scalar_fnmadd(lrk, v2p, x2[r]);
                            x3[r] = cxxgpr::detail::scalar_fnmadd(lrk, v3p, x3[r]);
                        }
                    }
                    if (ke < n)
                    {
                        for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            cxxgpr::detail::rank_k_update_4cols_simd(
                                x0, x1, x2, x3, l_ptrs.data(), bs, v0_arr.data(), v1_arr.data(), v2_arr.data(),
                                v3_arr.data(), r_start, r_stop
                            );
                        }
                    }
                }
            }

            for (; c < col_end; ++c)
            {
                T* const xc = rhs.col(c).data();
                for (std::size_t kb = 0U; kb < n; kb += k_block)
                {
                    const std::size_t ke = std::min(kb + k_block, n);
                    const std::size_t bs = ke - kb;
                    for (std::size_t p = 0U; p < bs; ++p)
                    {
                        const std::size_t k = kb + p;
                        const T* const lk = l.col(k).data();
                        const T v0p = xc[k] / lk[k];
                        xc[k] = v0p;
                        l_ptrs[p] = lk;
                        v0_arr[p] = v0p;
                        for (std::size_t r = k + 1U; r < ke; ++r)
                        {
                            xc[r] = cxxgpr::detail::scalar_fnmadd(lk[r], v0p, xc[r]);
                        }
                    }
                    if (ke < n)
                    {
                        for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            cxxgpr::detail::rank_k_update_1col_simd(
                                xc, l_ptrs.data(), bs, v0_arr.data(), r_start, r_stop
                            );
                        }
                    }
                }
            }
        }

        /// @brief Resolves one 4-row backward-substitution step given precomputed tail dot products.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] x Destination solution column pointer.
        /// @param[in] l0 Column `i0` of lower factor `l`.
        /// @param[in] l1 Column `i1 = i0 + 1` of lower factor `l`.
        /// @param[in] l2 Column `i2 = i0 + 2` of lower factor `l`.
        /// @param[in] l3 Column `i3 = i0 + 3` of lower factor `l`.
        /// @param[in] i0 Starting row index of the 4-row block.
        /// @param[in] d0 Precomputed tail dot product `dot(l0[i0+4:], x[i0+4:])`.
        /// @param[in] d1 Precomputed tail dot product `dot(l1[i0+4:], x[i0+4:])`.
        /// @param[in] d2 Precomputed tail dot product `dot(l2[i0+4:], x[i0+4:])`.
        /// @param[in] d3 Precomputed tail dot product `dot(l3[i0+4:], x[i0+4:])`.
        template <KernelScalarConcept T>
        inline void finish_backward_4rows(
            T* x, const T* l0, const T* l1, const T* l2, const T* l3, std::size_t i0, T d0, T d1, T d2, T d3
        ) noexcept
        {
            const std::size_t i1 = i0 + 1U;
            const std::size_t i2 = i0 + 2U;
            const std::size_t i3 = i0 + 3U;

            const T s3 = (x[i3] - d3) / l3[i3];
            x[i3] = s3;

            const T a2 = cxxgpr::detail::scalar_fmadd(l2[i3], s3, d2);
            const T s2 = (x[i2] - a2) / l2[i2];
            x[i2] = s2;

            const T a1 = cxxgpr::detail::scalar_fmadd(l1[i2], s2, cxxgpr::detail::scalar_fmadd(l1[i3], s3, d1));
            const T s1 = (x[i1] - a1) / l1[i1];
            x[i1] = s1;

            const T a0 = cxxgpr::detail::scalar_fmadd(
                l0[i1], s1, cxxgpr::detail::scalar_fmadd(l0[i2], s2, cxxgpr::detail::scalar_fmadd(l0[i3], s3, d0))
            );
            x[i0] = (x[i0] - a0) / l0[i0];
        }

        /// @brief Solves `L^T * X = B` in place across column range `[col_begin, col_end)` via deterministic SIMD dots.
        /// @details Uses 4x2 (`dot_4x2_simd`), 4x1 (`dot_4x1_simd`), and 1x1 (`dot_1x1_simd`) dot kernels reading
        /// contiguous lower-triangular column slices of `l`, guaranteeing bitwise identical rounding across single-RHS,
        /// multi-RHS, OpenMP thread counts, and hybrid CPU cache topologies.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] l Square read-only lower Cholesky factor matrix view.
        /// @param[in,out] rhs Mutable right-hand side matrix view updated in place.
        /// @param[in] col_begin Inclusive starting column index in `rhs`.
        /// @param[in] col_end Exclusive ending column index in `rhs`.
        template <KernelScalarConcept T>
        void solve_lower_transpose_cols(
            MatrixRef<T> l, MatrixMut<T> rhs, std::size_t col_begin, std::size_t col_end
        ) noexcept
        {
            const std::size_t n = l.rows();
            std::size_t c = col_begin;
            for (; c + 2U <= col_end; c += 2U)
            {
                T* const x0 = rhs.col(c).data();
                T* const x1 = rhs.col(c + 1U).data();
                std::size_t rem = n;
                while (rem >= 4U)
                {
                    const std::size_t i0 = rem - 4U;
                    const std::size_t i1 = rem - 3U;
                    const std::size_t i2 = rem - 2U;
                    const std::size_t i3 = rem - 1U;
                    const std::size_t tail_start = rem;
                    const std::size_t tail_len = n - tail_start;
                    const T* const l0 = l.col(i0).data();
                    const T* const l1 = l.col(i1).data();
                    const T* const l2 = l.col(i2).data();
                    const T* const l3 = l.col(i3).data();
                    std::array<std::array<T, 2>, 4> dots{};
                    if (tail_len > 0U)
                    {
                        cxxgpr::detail::dot_4x2_simd(
                            l0 + tail_start, l1 + tail_start, l2 + tail_start, l3 + tail_start, x0 + tail_start,
                            x1 + tail_start, tail_len, dots
                        );
                    }
                    finish_backward_4rows(x0, l0, l1, l2, l3, i0, dots[0][0], dots[1][0], dots[2][0], dots[3][0]);
                    finish_backward_4rows(x1, l0, l1, l2, l3, i0, dots[0][1], dots[1][1], dots[2][1], dots[3][1]);
                    rem -= 4U;
                }
                while (rem > 0U)
                {
                    --rem;
                    const std::size_t i = rem;
                    const T* const li = l.col(i).data();
                    const std::size_t tail_start = i + 1U;
                    const std::size_t tail_len = n - tail_start;
                    const T acc0 = (tail_len > 0U)
                        ? cxxgpr::detail::dot_1x1_simd(li + tail_start, x0 + tail_start, tail_len)
                        : T{0};
                    const T acc1 = (tail_len > 0U)
                        ? cxxgpr::detail::dot_1x1_simd(li + tail_start, x1 + tail_start, tail_len)
                        : T{0};
                    x0[i] = (x0[i] - acc0) / li[i];
                    x1[i] = (x1[i] - acc1) / li[i];
                }
            }

            for (; c < col_end; ++c)
            {
                T* const xc = rhs.col(c).data();
                std::size_t rem = n;
                while (rem >= 4U)
                {
                    const std::size_t i0 = rem - 4U;
                    const std::size_t i1 = rem - 3U;
                    const std::size_t i2 = rem - 2U;
                    const std::size_t i3 = rem - 1U;
                    const std::size_t tail_start = rem;
                    const std::size_t tail_len = n - tail_start;
                    const T* const l0 = l.col(i0).data();
                    const T* const l1 = l.col(i1).data();
                    const T* const l2 = l.col(i2).data();
                    const T* const l3 = l.col(i3).data();
                    std::array<T, 4> dots{};
                    if (tail_len > 0U)
                    {
                        cxxgpr::detail::dot_4x1_simd(
                            l0 + tail_start, l1 + tail_start, l2 + tail_start, l3 + tail_start, xc + tail_start,
                            tail_len, dots
                        );
                    }
                    finish_backward_4rows(xc, l0, l1, l2, l3, i0, dots[0], dots[1], dots[2], dots[3]);
                    rem -= 4U;
                }
                while (rem > 0U)
                {
                    --rem;
                    const std::size_t i = rem;
                    const T* const li = l.col(i).data();
                    const std::size_t tail_start = i + 1U;
                    const std::size_t tail_len = n - tail_start;
                    const T acc = (tail_len > 0U)
                        ? cxxgpr::detail::dot_1x1_simd(li + tail_start, xc + tail_start, tail_len)
                        : T{0};
                    xc[i] = (xc[i] - acc) / li[i];
                }
            }
        }

        /// @brief Executes forward, backward, or two-stage L * L^T triangular substitution over rhs columns.
        /// @details Uses cache-packed BLAS-3 solvers in fixed column blocks for large multi-RHS forward solves and
        /// deterministic fixed 4-column SIMD blocks for backward solves across OpenMP threads.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] l Square read-only lower Cholesky factor matrix view.
        /// @param[in,out] rhs Mutable right-hand side matrix view overwritten with the solution.
        /// @param[in] mode Triangular substitution direction or combined LLT solve.
        template <KernelScalarConcept T>
        void apply_tri_solve(MatrixRef<T> l, MatrixMut<T> rhs, TriSolveMode mode) noexcept
        {
            if (mode == TriSolveMode::Llt)
            {
                apply_tri_solve(l, rhs, TriSolveMode::Forward);
                apply_tri_solve(l, rhs, TriSolveMode::Backward);
                return;
            }

            const std::size_t n_rows = l.rows();
            const std::size_t n_cols = rhs.cols();
            if (mode == TriSolveMode::Forward)
            {
#ifdef _OPENMP
                const double approx_flops =
                    static_cast<double>(n_rows) * static_cast<double>(n_rows) * static_cast<double>(n_cols);
                if (n_cols >= 12U && approx_flops >= 4194304.0)
                {
                    // Fixed column blocks keep block boundaries independent of thread count while saturating OpenMP
                    // cores.
                    const std::size_t col_block = (n_cols >= 512U) ? 24U : 12U;
                    const std::size_t n_blocks = (n_cols + col_block - 1U) / col_block;
                    const auto n_blocks_s = static_cast<std::ptrdiff_t>(n_blocks);
#    pragma omp parallel for schedule(dynamic, 1) if (openmp_max_threads() > 1U)
                    for (std::ptrdiff_t b = 0; b < n_blocks_s; ++b)
                    {
                        const std::size_t col_begin = static_cast<std::size_t>(b) * col_block;
                        const std::size_t col_end = std::min(col_begin + col_block, n_cols);
                        solve_lower_cols(l, rhs, col_begin, col_end);
                    }
                    return;
                }
#else
                (void)n_rows;
#endif
                solve_lower_cols(l, rhs, 0U, n_cols);
                return;
            }

#ifdef _OPENMP
            const double approx_flops =
                static_cast<double>(n_rows) * static_cast<double>(n_rows) * static_cast<double>(n_cols);
            if (n_cols >= 4U && approx_flops >= 16777216.0)
            {
                // Use fixed 4-column blocks so backward SIMD grouping stays invariant to thread count.
                constexpr std::size_t col_block = 4U;
                const std::size_t n_blocks = (n_cols + col_block - 1U) / col_block;
                const auto n_blocks_s = static_cast<std::ptrdiff_t>(n_blocks);
#    pragma omp parallel for schedule(dynamic, 1) if (openmp_max_threads() > 1U)
                for (std::ptrdiff_t b = 0; b < n_blocks_s; ++b)
                {
                    const std::size_t col_begin = static_cast<std::size_t>(b) * col_block;
                    const std::size_t col_end = std::min(col_begin + col_block, n_cols);
                    solve_lower_transpose_cols(l, rhs, col_begin, col_end);
                }
                return;
            }
#else
            (void)n_rows;
#endif
            solve_lower_transpose_cols(l, rhs, 0U, n_cols);
        }

        /// @brief Applies one trailing symmetric rank-bs lower tile update inside blocked Cholesky factorization.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] a Target n x n symmetric matrix view updated in place.
        /// @param[in] k Starting column of the factored panel.
        /// @param[in] bs Factored panel width.
        /// @param[in] row_start Starting row and column index of the trailing submatrix (`k + bs`).
        /// @param[in] tail Dimension of the trailing submatrix (`n - row_start`).
        /// @param[in] tile Block tile extent (`24` or `k_blas3_mc`).
        /// @param[in] n_blocks Number of lower-triangular blocks along the trailing dimension.
        /// @param[in] tile_idx Linear column-major lower tile index in `[0, n_blocks * (n_blocks + 1) / 2)`.
        /// @param[in,out] last_bj Cached block column index corresponding to current contents of `packed_b`.
        /// @param[out] packed_a Per-thread scratch buffer of length `k_blas3_pack_a_cap<T>`.
        /// @param[in,out] packed_b Per-thread scratch buffer of length `k_blas3_pack_b_cap` reused across equal `bj`.
        template <KernelScalarConcept T>
        inline void apply_cholesky_trailing_tile(
            MatrixMut<T> a,
            std::size_t k,
            std::size_t bs,
            std::size_t row_start,
            std::size_t tail,
            std::size_t tile,
            std::size_t n_blocks,
            std::size_t tile_idx,
            std::size_t& last_bj,
            T* packed_a,
            T* packed_b
        ) noexcept
        {
            const auto coords = cxxgpr::detail::lower_tile_coords(tile_idx, n_blocks);
            const std::size_t bi = coords.row_block;
            const std::size_t bj = coords.col_block;
            const std::size_t ib = bi * tile;
            const std::size_t jb = bj * tile;
            const std::size_t rm = std::min(tile, tail - ib);
            const std::size_t cn = std::min(tile, tail - jb);
            if (bj != last_bj)
            {
                cxxgpr::detail::pack_panel_b_colmajor_t(&a(row_start + jb, k), a.col_stride(), bs, cn, packed_b);
                last_bj = bj;
            }
            cxxgpr::detail::pack_panel_a_colmajor(&a(row_start + ib, k), a.col_stride(), rm, bs, packed_a);
            T* const c_dst = &a(row_start + ib, row_start + jb);
            if (bi == bj)
            {
                cxxgpr::detail::syrk_packed_lower_diag_sub(rm, bs, packed_a, packed_b, c_dst, a.col_stride());
            }
            else
            {
                cxxgpr::detail::gemm_packed_sub(rm, cn, bs, packed_a, packed_b, c_dst, a.col_stride());
            }
        }

        /// @brief Executes Eigen::LLT, direct L2 SIMD blocked Cholesky, or cache-packed BLAS-3 Cholesky in place.
        /// @details Uses a three-tier cache-aware schedule:
        /// - `n >= 384`: cache-packed BLAS-3 (`KC = 64`, `MC/NC = 48`) with dynamic tile scheduling inside OpenMP.
        /// - `256 <= n < 384`: direct column-SIMD blocked Cholesky (`k_direct_panel_block`).
        /// - `n < 256`: single-threaded in-place `Eigen::LLT`.
        /// All tiers keep single-thread and multi-thread runs bitwise identical.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] a Square mutable matrix view containing A's lower triangle on entry.
        /// @param[in] stage Factorization stage recorded in CholeskyFailed diagnostics.
        /// @return Empty expected on success, or GprErrorCode::CholeskyFailed on breakdown.
        /// @retval GprErrorCode::CholeskyFailed When factorization fails or yields a non-positive/non-finite pivot.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> run_eigen_llt_in_place(MatrixMut<T> a, CholeskyStage stage)
        {
            const std::size_t n = a.rows();
            if (n >= 384U)
            {
                bool failed = false;
                constexpr std::size_t panel_block = cxxgpr::detail::k_blas3_kc;
                constexpr std::size_t tail_cutoff = 64U;
#ifdef _OPENMP
#    pragma omp parallel if (openmp_max_threads() > 1U) shared(a, failed, n)
                {
                    alignas(64) std::array<T, cxxgpr::detail::k_blas3_pack_a_cap<T>> packed_a{};
                    alignas(64) std::array<T, cxxgpr::detail::k_blas3_pack_b_cap> packed_b{};
                    for (std::size_t k = 0U; k < n; k += panel_block)
                    {
                        const std::size_t rem = n - k;
                        if (rem <= tail_cutoff)
                        {
#    pragma omp single
                            {
                                if (!factor_subblock_eigen(a, k, rem))
                                {
                                    failed = true;
                                }
                            }
                            break;
                        }
                        const std::size_t bs = std::min(panel_block, rem);
#    pragma omp single
                        {
                            if (!factor_subblock_eigen(a, k, bs))
                            {
                                failed = true;
                            }
                        }
                        if (failed)
                        {
                            break;
                        }
                        const std::size_t row_start = k + bs;
                        const std::size_t tail = n - row_start;
                        const std::size_t n_strips =
                            (tail + cxxgpr::detail::k_blas3_mc - 1U) / cxxgpr::detail::k_blas3_mc;
                        const auto n_strips_s = static_cast<std::ptrdiff_t>(n_strips);
#    pragma omp for schedule(dynamic, 1)
                        for (std::ptrdiff_t s = 0; s < n_strips_s; ++s)
                        {
                            const std::size_t r_begin =
                                row_start + (static_cast<std::size_t>(s) * cxxgpr::detail::k_blas3_mc);
                            const std::size_t r_end = std::min(r_begin + cxxgpr::detail::k_blas3_mc, n);
                            cxxgpr::detail::solve_panel_rows_packed(a, k, bs, r_begin, r_end);
                        }

                        const std::size_t tile = (tail >= 384U) ? cxxgpr::detail::k_blas3_mc : 24U;
                        const std::size_t n_blocks = (tail + tile - 1U) / tile;
                        const std::size_t total_tiles = (n_blocks * (n_blocks + 1U)) / 2U;
                        const auto total_tiles_s = static_cast<std::ptrdiff_t>(total_tiles);
                        std::size_t last_bj = n_blocks;
#    pragma omp for schedule(dynamic, 1)
                        for (std::ptrdiff_t t_s = 0; t_s < total_tiles_s; ++t_s)
                        {
                            apply_cholesky_trailing_tile(
                                a, k, bs, row_start, tail, tile, n_blocks, static_cast<std::size_t>(t_s), last_bj,
                                packed_a.data(), packed_b.data()
                            );
                        }
                    }
                }
#else
                alignas(64) std::array<T, cxxgpr::detail::k_blas3_pack_a_cap<T>> packed_a{};
                alignas(64) std::array<T, cxxgpr::detail::k_blas3_pack_b_cap> packed_b{};
                for (std::size_t k = 0U; k < n; k += panel_block)
                {
                    const std::size_t rem = n - k;
                    if (rem <= tail_cutoff)
                    {
                        if (!factor_subblock_eigen(a, k, rem))
                        {
                            failed = true;
                        }
                        break;
                    }
                    const std::size_t bs = std::min(panel_block, rem);
                    if (!factor_subblock_eigen(a, k, bs))
                    {
                        failed = true;
                        break;
                    }
                    const std::size_t row_start = k + bs;
                    const std::size_t tail = n - row_start;
                    cxxgpr::detail::solve_panel_rows_packed(a, k, bs, row_start, n);

                    const std::size_t tile = (tail >= 384U) ? cxxgpr::detail::k_blas3_mc : 24U;
                    const std::size_t n_blocks = (tail + tile - 1U) / tile;
                    const std::size_t total_tiles = (n_blocks * (n_blocks + 1U)) / 2U;
                    std::size_t last_bj = n_blocks;
                    for (std::size_t t = 0U; t < total_tiles; ++t)
                    {
                        apply_cholesky_trailing_tile(
                            a, k, bs, row_start, tail, tile, n_blocks, t, last_bj, packed_a.data(), packed_b.data()
                        );
                    }
                }
#endif
                if (failed)
                {
                    return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
                }
                return {};
            }

            if (n >= 256U)
            {
                bool failed = false;
                constexpr std::size_t tail_cutoff = 64U;
                constexpr std::size_t row_granule = 16U;
#ifdef _OPENMP
#    pragma omp parallel if (openmp_max_threads() > 1U) shared(a, failed, n)
                {
                    const auto tid = static_cast<std::size_t>(omp_get_thread_num());
                    const auto nthreads = static_cast<std::size_t>(omp_get_num_threads());
                    for (std::size_t k = 0U; k < n; k += k_direct_panel_block)
                    {
                        const std::size_t rem = n - k;
                        if (rem <= tail_cutoff)
                        {
#    pragma omp single
                            {
                                if (!factor_subblock_eigen(a, k, rem))
                                {
                                    failed = true;
                                }
                            }
                            break;
                        }
                        const std::size_t bs = std::min(k_direct_panel_block, rem);
#    pragma omp single
                        {
                            if (!factor_subblock_eigen(a, k, bs))
                            {
                                failed = true;
                            }
                        }
                        if (failed)
                        {
                            break;
                        }
                        const std::size_t row_start = k + bs;
                        const std::size_t tail = n - row_start;
                        const std::size_t n_gran = (tail + row_granule - 1U) / row_granule;
                        const std::size_t chunk_begin =
                            row_start + std::min(tail, ((n_gran * tid) / nthreads) * row_granule);
                        const std::size_t chunk_end =
                            row_start + std::min(tail, ((n_gran * (tid + 1U)) / nthreads) * row_granule);
                        if (chunk_begin < chunk_end)
                        {
                            solve_panel_rows(a, k, bs, chunk_begin, chunk_end);
                        }
#    pragma omp barrier
                        const std::size_t rel_begin = lower_block_start(tail, tid, nthreads);
                        const std::size_t rel_end = lower_block_start(tail, tid + 1U, nthreads);
                        rank_k_lower_update(a, k, bs, row_start + rel_begin, row_start + rel_end, n);
#    pragma omp barrier
                    }
                }
#else
                for (std::size_t k = 0U; k < n; k += k_direct_panel_block)
                {
                    const std::size_t rem = n - k;
                    if (rem <= tail_cutoff)
                    {
                        if (!factor_subblock_eigen(a, k, rem))
                        {
                            failed = true;
                        }
                        break;
                    }
                    const std::size_t bs = std::min(k_direct_panel_block, rem);
                    if (!factor_subblock_eigen(a, k, bs))
                    {
                        failed = true;
                        break;
                    }
                    const std::size_t row_start = k + bs;
                    const std::size_t tail = n - row_start;
                    (void)row_granule;
                    solve_panel_rows(a, k, bs, row_start, n);
                    rank_k_lower_update(a, k, bs, row_start, row_start + tail, n);
                }
#endif
                if (failed)
                {
                    return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
                }
                return {};
            }

            // Wrap caller storage via Eigen::Ref (tightly packed when contiguous) so Eigen overwrites in place.
            bool ok = false;
            if (a.is_contiguous())
            {
                Eigen::Map<detail::ColMajorDynamic<T>> dense_map(
                    a.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)
                );
                Eigen::LLT<Eigen::Ref<detail::ColMajorDynamic<T>>, Eigen::Lower> llt(dense_map);
                ok = (llt.info() == Eigen::Success);
            }
            else
            {
                auto map = detail::to_eigen_map(a);
                using RefType = Eigen::Ref<detail::ColMajorDynamic<T>, 0, Eigen::OuterStride<>>;
                Eigen::LLT<RefType, Eigen::Lower> llt(map);
                ok = (llt.info() == Eigen::Success);
            }
            if (!ok)
            {
                return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
            }
            // Guard against numerical underflow or silent NaNs on the diagonal after decomposition.
            for (std::size_t i = 0; i < n; ++i)
            {
                const T diag = a(i, i);
                if (!std::isfinite(diag) || diag <= T{0})
                {
                    return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
                }
            }
            return {};
        }

    }  // namespace

    /// @brief Factors a symmetric positive-definite matrix A in place into its lower Cholesky factor L.
    /// @details Validates square matrix geometry and lower-triangle finiteness before invoking Eigen in-place LLT.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] a Square mutable matrix view containing the lower triangle of A on entry and L on success.
    /// @param[in] stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Empty expected on success, or GprError on invalid input or non-positive-definite matrix.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When factorization fails or yields non-positive/non-finite pivots.
    template <KernelScalarConcept T>
    std::expected<void, GprError> cholesky_lower_in_place(MatrixMut<T> a, CholeskyStage stage)
    {
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return check;
        }
        if (const auto fin = detail::require_lower_finite(detail::as_ref(a)); !fin.has_value())
        {
            return fin;
        }
        return run_eigen_llt_in_place(a, stage);
    }

    /// @brief Factors A in place trying j = 0 first and escalating via policy using a caller-provided backup buffer.
    /// @details Copies A's lower triangle to backup before j = 0; on each jittered retry, restores A's lower triangle
    /// from backup before adding j to the diagonal so failed partial factorizations never accumulate.
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
    std::expected<double, GprError> cholesky_lower_with_jitter(
        MatrixMut<T> a, MatrixMut<T> backup, const JitterPolicy& policy, CholeskyStage stage
    )
    {
        // Validate target view, backup view, dimension match, and lower-triangle finiteness.
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        if (const auto check = detail::validate_matrix_view(backup, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        if (a.rows() != backup.rows())
        {
            return std::unexpected(GprError::length_mismatch("backup matrix dimension must match target matrix"));
        }
        if (const auto fin = detail::require_lower_finite(detail::as_ref(a)); !fin.has_value())
        {
            return std::unexpected(fin.error());
        }

        // Skip backing up the lower triangle when no jitter retry steps are configured (e.g., Fixed(0.0)).
        const std::size_t n = a.rows();
        JitterSchedule schedule = policy.retry_schedule();
        const std::optional<double> first_retry = schedule.next();
        if (first_retry.has_value())
        {
            if (const auto cp = copy_lower_triangle(detail::as_ref(a), backup); !cp.has_value())
            {
                return std::unexpected(cp.error());
            }
        }

        // Attempt unjittered factorization (j = 0) first; propagate non-Cholesky failures immediately.
        const auto first = run_eigen_llt_in_place(a, stage);
        if (first.has_value())
        {
            return 0.0;
        }
        if (first.error().code() != GprErrorCode::CholeskyFailed)
        {
            return std::unexpected(first.error());
        }

        // Retry with escalating jitter, restoring the uncorrupted lower triangle before each attempt.
        double last_j = 0.0;
        for (std::optional<double> next_j = first_retry; next_j.has_value(); next_j = schedule.next())
        {
            const double j = *next_j;
            last_j = j;
            if (const auto restore = copy_lower_triangle(detail::as_ref(backup), a); !restore.has_value())
            {
                return std::unexpected(restore.error());
            }
            if (const auto add = add_to_diag(a, static_cast<T>(j)); !add.has_value())
            {
                return std::unexpected(add.error());
            }
            const auto attempt = run_eigen_llt_in_place(a, stage);
            if (attempt.has_value())
            {
                return j;
            }
            if (attempt.error().code() != GprErrorCode::CholeskyFailed)
            {
                return std::unexpected(attempt.error());
            }
        }

        return std::unexpected(GprError::cholesky_failed(last_j, n, stage));
    }

    /// @brief Factors A in place trying j = 0 first and escalating via policy with an internal backup buffer.
    /// @details Allocates an n x n temporary backup buffer after validating a and delegates to the backup overload.
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
    std::expected<double, GprError> cholesky_lower_with_jitter(
        MatrixMut<T> a, const JitterPolicy& policy, CholeskyStage stage
    )
    {
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        const auto len = column_major_len(a.rows(), a.cols());
        if (!len.has_value())
        {
            return std::unexpected(len.error());
        }
        // Allocate a packed n x n backup buffer once and reuse the zero-allocation retry loop.
        std::vector<T> backup_storage(*len);
        MatrixMut<T> backup(backup_storage.data(), a.rows(), a.cols());
        return cholesky_lower_with_jitter(a, backup, policy, stage);
    }

    /// @brief Solves L * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Validates factor and right-hand side views, then performs deterministic blocked lower-triangular
    /// forward substitution.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in] validation Whether the caller owns an already validated factor.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> detail::solve_lower(MatrixRef<T> l, MatrixMut<T> rhs, FactorValidation validation)
    {
        if (const auto check = validate_solve_inputs(l, rhs, validation); !check.has_value())
        {
            return check;
        }
        apply_tri_solve(l, rhs, TriSolveMode::Forward);
        return {};
    }

    template std::expected<void, GprError> detail::solve_lower(
        MatrixRef<float>, MatrixMut<float>, detail::FactorValidation
    );
    template std::expected<void, GprError> detail::solve_lower(
        MatrixRef<double>, MatrixMut<double>, detail::FactorValidation
    );

    /// @brief Solves L * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Validates factor and right-hand side views, then performs deterministic blocked lower-triangular
    /// forward substitution.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, MatrixMut<T> rhs)
    {
        return detail::solve_lower(l, rhs, detail::FactorValidation::Check);
    }

    /// @brief Solves L * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Guards empty spans before forwarding rhs as an n x 1 MatrixMut view.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable vector span holding b on entry and x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, std::span<T> rhs)
    {
        if (rhs.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        return solve_lower_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
    }

    /// @brief Solves L^T * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Interprets l.transpose() as an upper-triangular factor so strictly upper entries of l are never read.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, MatrixMut<T> rhs)
    {
        if (const auto check = validate_solve_inputs(l, rhs); !check.has_value())
        {
            return check;
        }
        apply_tri_solve(l, rhs, TriSolveMode::Backward);
        return {};
    }

    /// @brief Solves L^T * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Guards empty spans before forwarding rhs as an n x 1 MatrixMut view.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable vector span holding b on entry and x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, std::span<T> rhs)
    {
        if (rhs.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        return solve_lower_transpose_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
    }

    /// @brief Solves (L * L^T) * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Performs forward substitution L * Y = B followed by backward substitution L^T * X = Y in place.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, MatrixMut<T> rhs)
    {
        if (const auto check = validate_solve_inputs(l, rhs); !check.has_value())
        {
            return check;
        }
        // Perform two-stage triangular solve: L * Y = B, then L^T * X = Y.
        apply_tri_solve(l, rhs, TriSolveMode::Llt);
        return {};
    }

    /// @brief Solves (L * L^T) * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Guards empty spans before forwarding rhs as an n x 1 MatrixMut view.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param[in,out] rhs Mutable vector span holding b on entry and x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, std::span<T> rhs)
    {
        if (rhs.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        return solve_llt_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
    }

    /// @brief Computes diag((L * L^T)^-1) using a caller-provided n x n scratch matrix.
    /// @details Initializes scratch_n_by_n to I_n, solves L * X = I_n in place to obtain L^-1 in the lower triangle,
    /// and sums squared column norms in double precision.
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
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> inv_diag_from_chol_l(
        MatrixRef<T> l, std::span<T> out_diag, MatrixMut<T> scratch_n_by_n
    )
    {
        // Validate factor, output span length, and square scratch matrix dimensions.
        if (const auto l_check = validate_lower_factor(l); !l_check.has_value())
        {
            return l_check;
        }
        if (out_diag.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (const auto cnt = require_count(out_diag.size(), l.rows(), "inv_diag output length"); !cnt.has_value())
        {
            return cnt;
        }
        if (const auto sc_check = detail::validate_matrix_view(scratch_n_by_n, true); !sc_check.has_value())
        {
            return sc_check;
        }
        if (scratch_n_by_n.rows() != l.rows())
        {
            return std::unexpected(GprError::length_mismatch("scratch matrix dimension must match factor dimension"));
        }

        // Use packed lower-triangular inversion for larger matrices, otherwise seed I_n and solve forward.
        const std::size_t n = l.rows();
        if (n >= 192U)
        {
            const std::size_t col_block = (n >= 768U) ? 24U : 12U;
            const std::size_t n_blocks = (n + col_block - 1U) / col_block;
            const auto n_blocks_s = static_cast<std::ptrdiff_t>(n_blocks);
#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic, 1) if (openmp_max_threads() > 1U)
#endif
            for (std::ptrdiff_t b = 0; b < n_blocks_s; ++b)
            {
                const std::size_t col_begin = static_cast<std::size_t>(b) * col_block;
                const std::size_t col_end = std::min(col_begin + col_block, n);
                cxxgpr::detail::invert_lower_col_strip_packed(l, scratch_n_by_n, col_begin, col_end);
            }
            const auto n_cols_s = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic, 1) if (openmp_max_threads() > 1U && n >= 256U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const auto sum_sq =
                    cxxgpr::detail::sq_norm_mixed_simd<T, double>(scratch_n_by_n.col(col).data() + col, n - col);
                out_diag[col] = static_cast<T>(sum_sq);
            }
            return {};
        }

        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                scratch_n_by_n(row, col) = (row == col) ? T{1} : T{0};
            }
        }

        apply_tri_solve(l, scratch_n_by_n, TriSolveMode::Forward);

        // Since (L L^T)^-1 = (L^-1)^T L^-1, the col-th diagonal equals ||L^-1 e_col||_2^2.
        for (std::size_t col = 0; col < n; ++col)
        {
            const auto sum_sq =
                cxxgpr::detail::sq_norm_mixed_simd<T, double>(scratch_n_by_n.col(col).data() + col, n - col);
            out_diag[col] = static_cast<T>(sum_sq);
        }
        return {};
    }

    /// @brief Computes log|A| = sum_{i=0}^{n-1} 2 * log(L_ii) using Neumaier compensated summation.
    /// @details Validates L and sums 2 * log(L_ii) in double precision using Neumaier error compensation to mitigate
    /// rounding loss across varying magnitudes.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @return Log-determinant of A = L * L^T in double precision, or GprError on invalid factor.
    /// @retval GprErrorCode::EmptyInput When l is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square or has invalid stride.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<double, GprError> log_det_from_l(MatrixRef<T> l)
    {
        if (const auto check = validate_lower_factor(l); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        // Accumulate 2 * log(L_ii) with Neumaier compensation, ordering subtraction by larger magnitude.
        const std::size_t n = l.rows();
        double sum = 0.0;
        double comp = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double term = 2.0 * std::log(static_cast<double>(l(i, i)));
            const double t = sum + term;
            if (std::abs(sum) >= std::abs(term))
            {
                comp += (sum - t) + term;
            }
            else
            {
                comp += (term - t) + sum;
            }
            sum = t;
        }
        return sum + comp;
    }

    /// @brief Computes 0.5 * log|A| from a lower Cholesky factor L using Neumaier compensated summation.
    /// @details Halves the result of log_det_from_l(l) after propagating any validation failure.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @return Half log-determinant of A = L * L^T in double precision, or GprError on invalid factor.
    /// @retval GprErrorCode::EmptyInput When l is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square or has invalid stride.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<double, GprError> half_log_det_from_l(MatrixRef<T> l)
    {
        const auto ld = log_det_from_l(l);
        if (!ld.has_value())
        {
            return std::unexpected(ld.error());
        }
        return 0.5 * (*ld);
    }

    template std::expected<void, GprError> cholesky_lower_in_place<float>(MatrixMut<float>, CholeskyStage);
    template std::expected<void, GprError> cholesky_lower_in_place<double>(MatrixMut<double>, CholeskyStage);

    template std::expected<double, GprError> cholesky_lower_with_jitter<float>(
        MatrixMut<float>, MatrixMut<float>, const JitterPolicy&, CholeskyStage
    );
    template std::expected<double, GprError> cholesky_lower_with_jitter<double>(
        MatrixMut<double>, MatrixMut<double>, const JitterPolicy&, CholeskyStage
    );

    template std::expected<double, GprError> cholesky_lower_with_jitter<float>(
        MatrixMut<float>, const JitterPolicy&, CholeskyStage
    );
    template std::expected<double, GprError> cholesky_lower_with_jitter<double>(
        MatrixMut<double>, const JitterPolicy&, CholeskyStage
    );

    template std::expected<void, GprError> solve_lower_in_place<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> solve_lower_in_place<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> solve_lower_in_place<float>(MatrixRef<float>, std::span<float>);
    template std::expected<void, GprError> solve_lower_in_place<double>(MatrixRef<double>, std::span<double>);

    template std::expected<void, GprError> solve_lower_transpose_in_place<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> solve_lower_transpose_in_place<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> solve_lower_transpose_in_place<float>(MatrixRef<float>, std::span<float>);
    template std::expected<void, GprError> solve_lower_transpose_in_place<double>(MatrixRef<double>, std::span<double>);

    template std::expected<void, GprError> solve_llt_in_place<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> solve_llt_in_place<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> solve_llt_in_place<float>(MatrixRef<float>, std::span<float>);
    template std::expected<void, GprError> solve_llt_in_place<double>(MatrixRef<double>, std::span<double>);

    template std::expected<void, GprError> inv_diag_from_chol_l<float>(
        MatrixRef<float>, std::span<float>, MatrixMut<float>
    );
    template std::expected<void, GprError> inv_diag_from_chol_l<double>(
        MatrixRef<double>, std::span<double>, MatrixMut<double>
    );

    template std::expected<double, GprError> log_det_from_l<float>(MatrixRef<float>);
    template std::expected<double, GprError> log_det_from_l<double>(MatrixRef<double>);

    template std::expected<double, GprError> half_log_det_from_l<float>(MatrixRef<float>);
    template std::expected<double, GprError> half_log_det_from_l<double>(MatrixRef<double>);

}  // namespace cxxgpr::linalg
