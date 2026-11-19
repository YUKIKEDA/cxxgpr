// SPDX-License-Identifier: Apache-2.0
/// @file packed_blas3.cpp
/// @brief Implementations and explicit instantiations of stack-packed SIMD BLAS-3 microkernels.
/// @details Implements AVX-512, AVX2+FMA, SSE2, and portable C++23 register-blocked GEMM/SYRK microkernels,
/// cache-aligned A/B micro-panel packers, lower-triangular column-strip inversion, and blocked triangular solvers
/// without internal heap allocations.

#include "linalg/packed_blas3.hpp"

#include "cxxgpr/core/views.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

#include "linalg/simd_blas3.hpp"

/// @brief Internal implementation helpers for cache-packed SIMD linear algebra.
namespace cxxgpr::detail
{

    namespace
    {

        /// @brief Selects whether microkernels accumulate `+A*B` or `-A*B` into `C`.
        enum class GemmOp : std::uint8_t
        {
            /// @brief Updates destination tile via `C += A * B`.
            Add,
            /// @brief Updates destination tile via `C -= A * B`.
            Sub,
        };

        /// @brief Compile-time register-block microkernel geometry (`MR x NR`) for scalar type `T` on active ISA.
        /// @tparam T Floating-point scalar type (`double` or `float`).
        template <SimdBlasScalar T> struct Blas3Layout
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            /// @brief Micro-panel row height matched to 2 AVX-512 vectors.
            static constexpr std::size_t mr = std::same_as<T, double> ? 16U : 32U;
            /// @brief Micro-panel column width matched to 19/32 live ZMM registers (16 accumulators + 2 A + 1 B).
            static constexpr std::size_t nr = 8U;
#elifdef __AVX2__
            /// @brief Micro-panel row height matched to 2 AVX2 vectors.
            static constexpr std::size_t mr = std::same_as<T, double> ? 8U : 16U;
            /// @brief Micro-panel column width matched to 15/16 live YMM registers (12 accumulators + 2 A + 1 B).
            static constexpr std::size_t nr = 6U;
#elif defined(_M_X64) || defined(__x86_64__)
            /// @brief Micro-panel row height matched to 2 SSE2 vectors.
            static constexpr std::size_t mr = std::same_as<T, double> ? 4U : 8U;
            /// @brief Micro-panel column width matched to 15/16 live XMM registers (12 accumulators + 2 A + 1 B).
            static constexpr std::size_t nr = 6U;
#else
            /// @brief Portable scalar fallback row height.
            static constexpr std::size_t mr = 4U;
            /// @brief Portable scalar fallback column width.
            static constexpr std::size_t nr = 4U;
#endif
        };

        static_assert(
            ((k_blas3_mc + Blas3Layout<double>::mr - 1U) / Blas3Layout<double>::mr) * Blas3Layout<double>::mr
                * k_blas3_kc
            <= k_blas3_pack_a_cap<double>
        );
        static_assert(
            ((k_blas3_mc + Blas3Layout<float>::mr - 1U) / Blas3Layout<float>::mr) * Blas3Layout<float>::mr * k_blas3_kc
            <= k_blas3_pack_a_cap<float>
        );
        static_assert(
            ((k_blas3_nc + Blas3Layout<double>::nr - 1U) / Blas3Layout<double>::nr) * Blas3Layout<double>::nr
                * k_blas3_kc
            <= k_blas3_pack_b_cap
        );
        static_assert(
            ((k_blas3_nc + Blas3Layout<float>::nr - 1U) / Blas3Layout<float>::nr) * Blas3Layout<float>::nr * k_blas3_kc
            <= k_blas3_pack_b_cap
        );

        /// @brief Copies a rectangular `rows x cols` column-major submatrix between two arbitrary strides.
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] src Source top-left element pointer.
        /// @param[in] src_ld Column stride of `src`.
        /// @param[out] dst Destination top-left element pointer.
        /// @param[in] dst_ld Column stride of `dst`.
        /// @param[in] rows Number of rows to copy.
        /// @param[in] cols Number of columns to copy.
        template <SimdBlasScalar T>
        inline void copy_block_strided(
            const T* src, std::size_t src_ld, T* dst, std::size_t dst_ld, std::size_t rows, std::size_t cols
        ) noexcept
        {
            for (std::size_t c = 0U; c < cols; ++c)
            {
                std::copy_n(src + (c * src_ld), rows, dst + (c * dst_ld));
            }
        }

        /// @brief Solves a contiguous diagonal block `L[kb:kb+bs, kb:kb+bs] * X = X` inside `x_diag` in ascending-`p`
        /// order.
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] l Square read-only lower Cholesky factor view.
        /// @param[in] kb Starting row and column index in `l`.
        /// @param[in] bs Block order (`1 <= bs <= k_blas3_kc`).
        /// @param[in] cn Number of packed right-hand side columns in `x_diag`.
        /// @param[in,out] x_diag Contiguous `bs x cn` stack matrix (`ld = bs`) overwritten in place with the solution.
        template <SimdBlasScalar T>
        void solve_diag_lower_block(MatrixRef<T> l, std::size_t kb, std::size_t bs, std::size_t cn, T* x_diag) noexcept
        {
            std::size_t c = 0U;
            // Process 4 columns at a time so L column reads are reused while matching single-RHS FMA rounding bitwise.
            for (; c + 4U <= cn; c += 4U)
            {
                T* const x0 = x_diag + (c * bs);
                T* const x1 = x_diag + ((c + 1U) * bs);
                T* const x2 = x_diag + ((c + 2U) * bs);
                T* const x3 = x_diag + ((c + 3U) * bs);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const T* const lk = l.col(kb + p).data() + kb;
                    const T lkk = lk[p];
                    const T v0p = x0[p] / lkk;
                    const T v1p = x1[p] / lkk;
                    const T v2p = x2[p] / lkk;
                    const T v3p = x3[p] / lkk;
                    x0[p] = v0p;
                    x1[p] = v1p;
                    x2[p] = v2p;
                    x3[p] = v3p;
                    for (std::size_t r = p + 1U; r < bs; ++r)
                    {
                        const T lrp = lk[r];
                        x0[r] = scalar_fnmadd(lrp, v0p, x0[r]);
                        x1[r] = scalar_fnmadd(lrp, v1p, x1[r]);
                        x2[r] = scalar_fnmadd(lrp, v2p, x2[r]);
                        x3[r] = scalar_fnmadd(lrp, v3p, x3[r]);
                    }
                }
            }
            for (; c < cn; ++c)
            {
                T* const xc = x_diag + (c * bs);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const T* const lk = l.col(kb + p).data() + kb;
                    const T v0p = xc[p] / lk[p];
                    xc[p] = v0p;
                    for (std::size_t r = p + 1U; r < bs; ++r)
                    {
                        xc[r] = scalar_fnmadd(lk[r], v0p, xc[r]);
                    }
                }
            }
        }

#if defined(__AVX512F__) && defined(__AVX512DQ__)
        /// @brief Applies one AVX-512 binary64 FMA accumulation (`c + a*b` or `c - a*b`).
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @param[in] a Packed `A` vector.
        /// @param[in] b Broadcast `B` vector.
        /// @param[in] acc Current accumulator vector.
        /// @return Updated accumulator vector.
        template <GemmOp Op> [[nodiscard]] inline __m512d blas3_acc_pd_512(__m512d a, __m512d b, __m512d acc) noexcept
        {
            if constexpr (Op == GemmOp::Add)
            {
                return blas3_fmadd_pd(a, b, acc);
            }
            else
            {
                return blas3_fnmadd_pd(a, b, acc);
            }
        }

        /// @brief Applies one AVX-512 binary32 FMA accumulation (`c + a*b` or `c - a*b`).
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @param[in] a Packed `A` vector.
        /// @param[in] b Broadcast `B` vector.
        /// @param[in] acc Current accumulator vector.
        /// @return Updated accumulator vector.
        template <GemmOp Op> [[nodiscard]] inline __m512 blas3_acc_ps_512(__m512 a, __m512 b, __m512 acc) noexcept
        {
            if constexpr (Op == GemmOp::Add)
            {
                return blas3_fmadd_ps(a, b, acc);
            }
            else
            {
                return blas3_fnmadd_ps(a, b, acc);
            }
        }
#elifdef __AVX2__
        /// @brief Applies one AVX2 binary64 FMA accumulation (`c + a*b` or `c - a*b`).
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @param[in] a Packed `A` vector.
        /// @param[in] b Broadcast `B` vector.
        /// @param[in] acc Current accumulator vector.
        /// @return Updated accumulator vector.
        template <GemmOp Op> [[nodiscard]] inline __m256d blas3_acc_pd_256(__m256d a, __m256d b, __m256d acc) noexcept
        {
            if constexpr (Op == GemmOp::Add)
            {
                return blas3_fmadd_pd(a, b, acc);
            }
            else
            {
                return blas3_fnmadd_pd(a, b, acc);
            }
        }

        /// @brief Applies one AVX2 binary32 FMA accumulation (`c + a*b` or `c - a*b`).
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @param[in] a Packed `A` vector.
        /// @param[in] b Broadcast `B` vector.
        /// @param[in] acc Current accumulator vector.
        /// @return Updated accumulator vector.
        template <GemmOp Op> [[nodiscard]] inline __m256 blas3_acc_ps_256(__m256 a, __m256 b, __m256 acc) noexcept
        {
            if constexpr (Op == GemmOp::Add)
            {
                return blas3_fmadd_ps(a, b, acc);
            }
            else
            {
                return blas3_fnmadd_ps(a, b, acc);
            }
        }
#elif defined(_M_X64) || defined(__x86_64__)
        /// @brief Applies one SSE2 binary64 accumulation (`c + a*b` or `c - a*b`).
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @param[in] a Packed `A` vector.
        /// @param[in] b Broadcast `B` vector.
        /// @param[in] acc Current accumulator vector.
        /// @return Updated accumulator vector.
        template <GemmOp Op> [[nodiscard]] inline __m128d blas3_acc_pd_128(__m128d a, __m128d b, __m128d acc) noexcept
        {
            if constexpr (Op == GemmOp::Add)
            {
                return _mm_add_pd(acc, _mm_mul_pd(a, b));
            }
            else
            {
                return _mm_sub_pd(acc, _mm_mul_pd(a, b));
            }
        }

        /// @brief Applies one SSE2 binary32 accumulation (`c + a*b` or `c - a*b`).
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @param[in] a Packed `A` vector.
        /// @param[in] b Broadcast `B` vector.
        /// @param[in] acc Current accumulator vector.
        /// @return Updated accumulator vector.
        template <GemmOp Op> [[nodiscard]] inline __m128 blas3_acc_ps_128(__m128 a, __m128 b, __m128 acc) noexcept
        {
            if constexpr (Op == GemmOp::Add)
            {
                return _mm_add_ps(acc, _mm_mul_ps(a, b));
            }
            else
            {
                return _mm_sub_ps(acc, _mm_mul_ps(a, b));
            }
        }
#endif

        /// @brief Updates a single `MR x NR` output tile `C (+/-)= A_panel * B_panel` in ascending `p = 0..k-1` order.
        /// @details Holds every `MR x NR` accumulator in dedicated scalar SIMD register variables (preventing compiler
        /// stack spills) and issues one ascending `p` step per iteration.
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] k Reduction depth (`0 <= k <= k_blas3_kc`).
        /// @param[in] pa Packed `A` micro-panel (`MR` stride per `p` step).
        /// @param[in] pb Packed `B` micro-panel (`NR` stride per `p` step).
        /// @param[in,out] c Destination block pointer `(0, 0)` with row height `>= MR` and column count `>= NR`.
        /// @param[in] ldc Column stride of `c` in elements.
        template <GemmOp Op, SimdBlasScalar T>
        inline void microkernel_mr_nr(std::size_t k, const T* pa, const T* pb, T* c, std::size_t ldc) noexcept
        {
            if (k == 0U)
            {
                return;
            }
            constexpr std::size_t mr = Blas3Layout<T>::mr;
            constexpr std::size_t nr = Blas3Layout<T>::nr;

            if constexpr (std::same_as<T, double>)
            {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
                double* const col0 = c;
                double* const col1 = c + ldc;
                double* const col2 = c + (2U * ldc);
                double* const col3 = c + (3U * ldc);
                double* const col4 = c + (4U * ldc);
                double* const col5 = c + (5U * ldc);
                double* const col6 = c + (6U * ldc);
                double* const col7 = c + (7U * ldc);

                __m512d c00 = _mm512_loadu_pd(col0);
                __m512d c10 = _mm512_loadu_pd(col0 + 8U);
                __m512d c01 = _mm512_loadu_pd(col1);
                __m512d c11 = _mm512_loadu_pd(col1 + 8U);
                __m512d c02 = _mm512_loadu_pd(col2);
                __m512d c12 = _mm512_loadu_pd(col2 + 8U);
                __m512d c03 = _mm512_loadu_pd(col3);
                __m512d c13 = _mm512_loadu_pd(col3 + 8U);
                __m512d c04 = _mm512_loadu_pd(col4);
                __m512d c14 = _mm512_loadu_pd(col4 + 8U);
                __m512d c05 = _mm512_loadu_pd(col5);
                __m512d c15 = _mm512_loadu_pd(col5 + 8U);
                __m512d c06 = _mm512_loadu_pd(col6);
                __m512d c16 = _mm512_loadu_pd(col6 + 8U);
                __m512d c07 = _mm512_loadu_pd(col7);
                __m512d c17 = _mm512_loadu_pd(col7 + 8U);

                // Stream packed A and B step by step so all 16 accumulators stay pin-resident in ZMM registers.
                for (std::size_t p = 0U; p < k; ++p)
                {
                    const __m512d a0 = _mm512_loadu_pd(pa);
                    const __m512d a1 = _mm512_loadu_pd(pa + 8U);
                    __m512d vb = _mm512_set1_pd(pb[0]);
                    c00 = blas3_acc_pd_512<Op>(a0, vb, c00);
                    c10 = blas3_acc_pd_512<Op>(a1, vb, c10);
                    vb = _mm512_set1_pd(pb[1]);
                    c01 = blas3_acc_pd_512<Op>(a0, vb, c01);
                    c11 = blas3_acc_pd_512<Op>(a1, vb, c11);
                    vb = _mm512_set1_pd(pb[2]);
                    c02 = blas3_acc_pd_512<Op>(a0, vb, c02);
                    c12 = blas3_acc_pd_512<Op>(a1, vb, c12);
                    vb = _mm512_set1_pd(pb[3]);
                    c03 = blas3_acc_pd_512<Op>(a0, vb, c03);
                    c13 = blas3_acc_pd_512<Op>(a1, vb, c13);
                    vb = _mm512_set1_pd(pb[4]);
                    c04 = blas3_acc_pd_512<Op>(a0, vb, c04);
                    c14 = blas3_acc_pd_512<Op>(a1, vb, c14);
                    vb = _mm512_set1_pd(pb[5]);
                    c05 = blas3_acc_pd_512<Op>(a0, vb, c05);
                    c15 = blas3_acc_pd_512<Op>(a1, vb, c15);
                    vb = _mm512_set1_pd(pb[6]);
                    c06 = blas3_acc_pd_512<Op>(a0, vb, c06);
                    c16 = blas3_acc_pd_512<Op>(a1, vb, c16);
                    vb = _mm512_set1_pd(pb[7]);
                    c07 = blas3_acc_pd_512<Op>(a0, vb, c07);
                    c17 = blas3_acc_pd_512<Op>(a1, vb, c17);
                    pa += mr;
                    pb += nr;
                }

                _mm512_storeu_pd(col0, c00);
                _mm512_storeu_pd(col0 + 8U, c10);
                _mm512_storeu_pd(col1, c01);
                _mm512_storeu_pd(col1 + 8U, c11);
                _mm512_storeu_pd(col2, c02);
                _mm512_storeu_pd(col2 + 8U, c12);
                _mm512_storeu_pd(col3, c03);
                _mm512_storeu_pd(col3 + 8U, c13);
                _mm512_storeu_pd(col4, c04);
                _mm512_storeu_pd(col4 + 8U, c14);
                _mm512_storeu_pd(col5, c05);
                _mm512_storeu_pd(col5 + 8U, c15);
                _mm512_storeu_pd(col6, c06);
                _mm512_storeu_pd(col6 + 8U, c16);
                _mm512_storeu_pd(col7, c07);
                _mm512_storeu_pd(col7 + 8U, c17);
#elifdef __AVX2__
                double* const col0 = c;
                double* const col1 = c + ldc;
                double* const col2 = c + (2U * ldc);
                double* const col3 = c + (3U * ldc);
                double* const col4 = c + (4U * ldc);
                double* const col5 = c + (5U * ldc);

                __m256d c00 = _mm256_loadu_pd(col0);
                __m256d c10 = _mm256_loadu_pd(col0 + 4U);
                __m256d c01 = _mm256_loadu_pd(col1);
                __m256d c11 = _mm256_loadu_pd(col1 + 4U);
                __m256d c02 = _mm256_loadu_pd(col2);
                __m256d c12 = _mm256_loadu_pd(col2 + 4U);
                __m256d c03 = _mm256_loadu_pd(col3);
                __m256d c13 = _mm256_loadu_pd(col3 + 4U);
                __m256d c04 = _mm256_loadu_pd(col4);
                __m256d c14 = _mm256_loadu_pd(col4 + 4U);
                __m256d c05 = _mm256_loadu_pd(col5);
                __m256d c15 = _mm256_loadu_pd(col5 + 4U);

                // Keep exactly 12 accumulator + 2 A + 1 B = 15 live YMM registers per step to eliminate stack spills.
                for (std::size_t p = 0U; p < k; ++p)
                {
                    const __m256d a0 = _mm256_loadu_pd(pa);
                    const __m256d a1 = _mm256_loadu_pd(pa + 4U);
                    __m256d vb = _mm256_broadcast_sd(pb);
                    c00 = blas3_acc_pd_256<Op>(a0, vb, c00);
                    c10 = blas3_acc_pd_256<Op>(a1, vb, c10);
                    vb = _mm256_broadcast_sd(pb + 1U);
                    c01 = blas3_acc_pd_256<Op>(a0, vb, c01);
                    c11 = blas3_acc_pd_256<Op>(a1, vb, c11);
                    vb = _mm256_broadcast_sd(pb + 2U);
                    c02 = blas3_acc_pd_256<Op>(a0, vb, c02);
                    c12 = blas3_acc_pd_256<Op>(a1, vb, c12);
                    vb = _mm256_broadcast_sd(pb + 3U);
                    c03 = blas3_acc_pd_256<Op>(a0, vb, c03);
                    c13 = blas3_acc_pd_256<Op>(a1, vb, c13);
                    vb = _mm256_broadcast_sd(pb + 4U);
                    c04 = blas3_acc_pd_256<Op>(a0, vb, c04);
                    c14 = blas3_acc_pd_256<Op>(a1, vb, c14);
                    vb = _mm256_broadcast_sd(pb + 5U);
                    c05 = blas3_acc_pd_256<Op>(a0, vb, c05);
                    c15 = blas3_acc_pd_256<Op>(a1, vb, c15);
                    pa += mr;
                    pb += nr;
                }

                _mm256_storeu_pd(col0, c00);
                _mm256_storeu_pd(col0 + 4U, c10);
                _mm256_storeu_pd(col1, c01);
                _mm256_storeu_pd(col1 + 4U, c11);
                _mm256_storeu_pd(col2, c02);
                _mm256_storeu_pd(col2 + 4U, c12);
                _mm256_storeu_pd(col3, c03);
                _mm256_storeu_pd(col3 + 4U, c13);
                _mm256_storeu_pd(col4, c04);
                _mm256_storeu_pd(col4 + 4U, c14);
                _mm256_storeu_pd(col5, c05);
                _mm256_storeu_pd(col5 + 4U, c15);
#elif defined(_M_X64) || defined(__x86_64__)
                double* const col0 = c;
                double* const col1 = c + ldc;
                double* const col2 = c + (2U * ldc);
                double* const col3 = c + (3U * ldc);
                double* const col4 = c + (4U * ldc);
                double* const col5 = c + (5U * ldc);

                __m128d c00 = _mm_loadu_pd(col0);
                __m128d c10 = _mm_loadu_pd(col0 + 2U);
                __m128d c01 = _mm_loadu_pd(col1);
                __m128d c11 = _mm_loadu_pd(col1 + 2U);
                __m128d c02 = _mm_loadu_pd(col2);
                __m128d c12 = _mm_loadu_pd(col2 + 2U);
                __m128d c03 = _mm_loadu_pd(col3);
                __m128d c13 = _mm_loadu_pd(col3 + 2U);
                __m128d c04 = _mm_loadu_pd(col4);
                __m128d c14 = _mm_loadu_pd(col4 + 2U);
                __m128d c05 = _mm_loadu_pd(col5);
                __m128d c15 = _mm_loadu_pd(col5 + 2U);

                for (std::size_t p = 0U; p < k; ++p)
                {
                    const __m128d a0 = _mm_loadu_pd(pa);
                    const __m128d a1 = _mm_loadu_pd(pa + 2U);
                    __m128d vb = _mm_set1_pd(pb[0]);
                    c00 = blas3_acc_pd_128<Op>(a0, vb, c00);
                    c10 = blas3_acc_pd_128<Op>(a1, vb, c10);
                    vb = _mm_set1_pd(pb[1]);
                    c01 = blas3_acc_pd_128<Op>(a0, vb, c01);
                    c11 = blas3_acc_pd_128<Op>(a1, vb, c11);
                    vb = _mm_set1_pd(pb[2]);
                    c02 = blas3_acc_pd_128<Op>(a0, vb, c02);
                    c12 = blas3_acc_pd_128<Op>(a1, vb, c12);
                    vb = _mm_set1_pd(pb[3]);
                    c03 = blas3_acc_pd_128<Op>(a0, vb, c03);
                    c13 = blas3_acc_pd_128<Op>(a1, vb, c13);
                    vb = _mm_set1_pd(pb[4]);
                    c04 = blas3_acc_pd_128<Op>(a0, vb, c04);
                    c14 = blas3_acc_pd_128<Op>(a1, vb, c14);
                    vb = _mm_set1_pd(pb[5]);
                    c05 = blas3_acc_pd_128<Op>(a0, vb, c05);
                    c15 = blas3_acc_pd_128<Op>(a1, vb, c15);
                    pa += mr;
                    pb += nr;
                }

                _mm_storeu_pd(col0, c00);
                _mm_storeu_pd(col0 + 2U, c10);
                _mm_storeu_pd(col1, c01);
                _mm_storeu_pd(col1 + 2U, c11);
                _mm_storeu_pd(col2, c02);
                _mm_storeu_pd(col2 + 2U, c12);
                _mm_storeu_pd(col3, c03);
                _mm_storeu_pd(col3 + 2U, c13);
                _mm_storeu_pd(col4, c04);
                _mm_storeu_pd(col4 + 2U, c14);
                _mm_storeu_pd(col5, c05);
                _mm_storeu_pd(col5 + 2U, c15);
#else
                std::array<std::array<double, mr>, nr> acc{};
                for (std::size_t j = 0U; j < nr; ++j)
                {
                    for (std::size_t i = 0U; i < mr; ++i)
                    {
                        acc[j][i] = c[(j * ldc) + i];
                    }
                }
                for (std::size_t p = 0U; p < k; ++p)
                {
                    for (std::size_t j = 0U; j < nr; ++j)
                    {
                        const double bpj = pb[j];
                        for (std::size_t i = 0U; i < mr; ++i)
                        {
                            acc[j][i] = (Op == GemmOp::Add) ? scalar_fmadd(pa[i], bpj, acc[j][i])
                                                            : scalar_fnmadd(pa[i], bpj, acc[j][i]);
                        }
                    }
                    pa += mr;
                    pb += nr;
                }
                for (std::size_t j = 0U; j < nr; ++j)
                {
                    for (std::size_t i = 0U; i < mr; ++i)
                    {
                        c[(j * ldc) + i] = acc[j][i];
                    }
                }
#endif
            }
            else
            {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
                float* const col0 = c;
                float* const col1 = c + ldc;
                float* const col2 = c + (2U * ldc);
                float* const col3 = c + (3U * ldc);
                float* const col4 = c + (4U * ldc);
                float* const col5 = c + (5U * ldc);
                float* const col6 = c + (6U * ldc);
                float* const col7 = c + (7U * ldc);

                __m512 c00 = _mm512_loadu_ps(col0);
                __m512 c10 = _mm512_loadu_ps(col0 + 16U);
                __m512 c01 = _mm512_loadu_ps(col1);
                __m512 c11 = _mm512_loadu_ps(col1 + 16U);
                __m512 c02 = _mm512_loadu_ps(col2);
                __m512 c12 = _mm512_loadu_ps(col2 + 16U);
                __m512 c03 = _mm512_loadu_ps(col3);
                __m512 c13 = _mm512_loadu_ps(col3 + 16U);
                __m512 c04 = _mm512_loadu_ps(col4);
                __m512 c14 = _mm512_loadu_ps(col4 + 16U);
                __m512 c05 = _mm512_loadu_ps(col5);
                __m512 c15 = _mm512_loadu_ps(col5 + 16U);
                __m512 c06 = _mm512_loadu_ps(col6);
                __m512 c16 = _mm512_loadu_ps(col6 + 16U);
                __m512 c07 = _mm512_loadu_ps(col7);
                __m512 c17 = _mm512_loadu_ps(col7 + 16U);

                for (std::size_t p = 0U; p < k; ++p)
                {
                    const __m512 a0 = _mm512_loadu_ps(pa);
                    const __m512 a1 = _mm512_loadu_ps(pa + 16U);
                    __m512 vb = _mm512_set1_ps(pb[0]);
                    c00 = blas3_acc_ps_512<Op>(a0, vb, c00);
                    c10 = blas3_acc_ps_512<Op>(a1, vb, c10);
                    vb = _mm512_set1_ps(pb[1]);
                    c01 = blas3_acc_ps_512<Op>(a0, vb, c01);
                    c11 = blas3_acc_ps_512<Op>(a1, vb, c11);
                    vb = _mm512_set1_ps(pb[2]);
                    c02 = blas3_acc_ps_512<Op>(a0, vb, c02);
                    c12 = blas3_acc_ps_512<Op>(a1, vb, c12);
                    vb = _mm512_set1_ps(pb[3]);
                    c03 = blas3_acc_ps_512<Op>(a0, vb, c03);
                    c13 = blas3_acc_ps_512<Op>(a1, vb, c13);
                    vb = _mm512_set1_ps(pb[4]);
                    c04 = blas3_acc_ps_512<Op>(a0, vb, c04);
                    c14 = blas3_acc_ps_512<Op>(a1, vb, c14);
                    vb = _mm512_set1_ps(pb[5]);
                    c05 = blas3_acc_ps_512<Op>(a0, vb, c05);
                    c15 = blas3_acc_ps_512<Op>(a1, vb, c15);
                    vb = _mm512_set1_ps(pb[6]);
                    c06 = blas3_acc_ps_512<Op>(a0, vb, c06);
                    c16 = blas3_acc_ps_512<Op>(a1, vb, c16);
                    vb = _mm512_set1_ps(pb[7]);
                    c07 = blas3_acc_ps_512<Op>(a0, vb, c07);
                    c17 = blas3_acc_ps_512<Op>(a1, vb, c17);
                    pa += mr;
                    pb += nr;
                }

                _mm512_storeu_ps(col0, c00);
                _mm512_storeu_ps(col0 + 16U, c10);
                _mm512_storeu_ps(col1, c01);
                _mm512_storeu_ps(col1 + 16U, c11);
                _mm512_storeu_ps(col2, c02);
                _mm512_storeu_ps(col2 + 16U, c12);
                _mm512_storeu_ps(col3, c03);
                _mm512_storeu_ps(col3 + 16U, c13);
                _mm512_storeu_ps(col4, c04);
                _mm512_storeu_ps(col4 + 16U, c14);
                _mm512_storeu_ps(col5, c05);
                _mm512_storeu_ps(col5 + 16U, c15);
                _mm512_storeu_ps(col6, c06);
                _mm512_storeu_ps(col6 + 16U, c16);
                _mm512_storeu_ps(col7, c07);
                _mm512_storeu_ps(col7 + 16U, c17);
#elifdef __AVX2__
                float* const col0 = c;
                float* const col1 = c + ldc;
                float* const col2 = c + (2U * ldc);
                float* const col3 = c + (3U * ldc);
                float* const col4 = c + (4U * ldc);
                float* const col5 = c + (5U * ldc);

                __m256 c00 = _mm256_loadu_ps(col0);
                __m256 c10 = _mm256_loadu_ps(col0 + 8U);
                __m256 c01 = _mm256_loadu_ps(col1);
                __m256 c11 = _mm256_loadu_ps(col1 + 8U);
                __m256 c02 = _mm256_loadu_ps(col2);
                __m256 c12 = _mm256_loadu_ps(col2 + 8U);
                __m256 c03 = _mm256_loadu_ps(col3);
                __m256 c13 = _mm256_loadu_ps(col3 + 8U);
                __m256 c04 = _mm256_loadu_ps(col4);
                __m256 c14 = _mm256_loadu_ps(col4 + 8U);
                __m256 c05 = _mm256_loadu_ps(col5);
                __m256 c15 = _mm256_loadu_ps(col5 + 8U);

                // Keep exactly 12 accumulator + 2 A + 1 B = 15 live YMM registers per step to eliminate stack spills.
                for (std::size_t p = 0U; p < k; ++p)
                {
                    const __m256 a0 = _mm256_loadu_ps(pa);
                    const __m256 a1 = _mm256_loadu_ps(pa + 8U);
                    __m256 vb = _mm256_broadcast_ss(pb);
                    c00 = blas3_acc_ps_256<Op>(a0, vb, c00);
                    c10 = blas3_acc_ps_256<Op>(a1, vb, c10);
                    vb = _mm256_broadcast_ss(pb + 1U);
                    c01 = blas3_acc_ps_256<Op>(a0, vb, c01);
                    c11 = blas3_acc_ps_256<Op>(a1, vb, c11);
                    vb = _mm256_broadcast_ss(pb + 2U);
                    c02 = blas3_acc_ps_256<Op>(a0, vb, c02);
                    c12 = blas3_acc_ps_256<Op>(a1, vb, c12);
                    vb = _mm256_broadcast_ss(pb + 3U);
                    c03 = blas3_acc_ps_256<Op>(a0, vb, c03);
                    c13 = blas3_acc_ps_256<Op>(a1, vb, c13);
                    vb = _mm256_broadcast_ss(pb + 4U);
                    c04 = blas3_acc_ps_256<Op>(a0, vb, c04);
                    c14 = blas3_acc_ps_256<Op>(a1, vb, c14);
                    vb = _mm256_broadcast_ss(pb + 5U);
                    c05 = blas3_acc_ps_256<Op>(a0, vb, c05);
                    c15 = blas3_acc_ps_256<Op>(a1, vb, c15);
                    pa += mr;
                    pb += nr;
                }

                _mm256_storeu_ps(col0, c00);
                _mm256_storeu_ps(col0 + 8U, c10);
                _mm256_storeu_ps(col1, c01);
                _mm256_storeu_ps(col1 + 8U, c11);
                _mm256_storeu_ps(col2, c02);
                _mm256_storeu_ps(col2 + 8U, c12);
                _mm256_storeu_ps(col3, c03);
                _mm256_storeu_ps(col3 + 8U, c13);
                _mm256_storeu_ps(col4, c04);
                _mm256_storeu_ps(col4 + 8U, c14);
                _mm256_storeu_ps(col5, c05);
                _mm256_storeu_ps(col5 + 8U, c15);
#elif defined(_M_X64) || defined(__x86_64__)
                float* const col0 = c;
                float* const col1 = c + ldc;
                float* const col2 = c + (2U * ldc);
                float* const col3 = c + (3U * ldc);
                float* const col4 = c + (4U * ldc);
                float* const col5 = c + (5U * ldc);

                __m128 c00 = _mm_loadu_ps(col0);
                __m128 c10 = _mm_loadu_ps(col0 + 4U);
                __m128 c01 = _mm_loadu_ps(col1);
                __m128 c11 = _mm_loadu_ps(col1 + 4U);
                __m128 c02 = _mm_loadu_ps(col2);
                __m128 c12 = _mm_loadu_ps(col2 + 4U);
                __m128 c03 = _mm_loadu_ps(col3);
                __m128 c13 = _mm_loadu_ps(col3 + 4U);
                __m128 c04 = _mm_loadu_ps(col4);
                __m128 c14 = _mm_loadu_ps(col4 + 4U);
                __m128 c05 = _mm_loadu_ps(col5);
                __m128 c15 = _mm_loadu_ps(col5 + 4U);

                for (std::size_t p = 0U; p < k; ++p)
                {
                    const __m128 a0 = _mm_loadu_ps(pa);
                    const __m128 a1 = _mm_loadu_ps(pa + 4U);
                    __m128 vb = _mm_set1_ps(pb[0]);
                    c00 = blas3_acc_ps_128<Op>(a0, vb, c00);
                    c10 = blas3_acc_ps_128<Op>(a1, vb, c10);
                    vb = _mm_set1_ps(pb[1]);
                    c01 = blas3_acc_ps_128<Op>(a0, vb, c01);
                    c11 = blas3_acc_ps_128<Op>(a1, vb, c11);
                    vb = _mm_set1_ps(pb[2]);
                    c02 = blas3_acc_ps_128<Op>(a0, vb, c02);
                    c12 = blas3_acc_ps_128<Op>(a1, vb, c12);
                    vb = _mm_set1_ps(pb[3]);
                    c03 = blas3_acc_ps_128<Op>(a0, vb, c03);
                    c13 = blas3_acc_ps_128<Op>(a1, vb, c13);
                    vb = _mm_set1_ps(pb[4]);
                    c04 = blas3_acc_ps_128<Op>(a0, vb, c04);
                    c14 = blas3_acc_ps_128<Op>(a1, vb, c14);
                    vb = _mm_set1_ps(pb[5]);
                    c05 = blas3_acc_ps_128<Op>(a0, vb, c05);
                    c15 = blas3_acc_ps_128<Op>(a1, vb, c15);
                    pa += mr;
                    pb += nr;
                }

                _mm_storeu_ps(col0, c00);
                _mm_storeu_ps(col0 + 4U, c10);
                _mm_storeu_ps(col1, c01);
                _mm_storeu_ps(col1 + 4U, c11);
                _mm_storeu_ps(col2, c02);
                _mm_storeu_ps(col2 + 4U, c12);
                _mm_storeu_ps(col3, c03);
                _mm_storeu_ps(col3 + 4U, c13);
                _mm_storeu_ps(col4, c04);
                _mm_storeu_ps(col4 + 4U, c14);
                _mm_storeu_ps(col5, c05);
                _mm_storeu_ps(col5 + 4U, c15);
#else
                std::array<std::array<float, mr>, nr> acc{};
                for (std::size_t j = 0U; j < nr; ++j)
                {
                    for (std::size_t i = 0U; i < mr; ++i)
                    {
                        acc[j][i] = c[(j * ldc) + i];
                    }
                }
                for (std::size_t p = 0U; p < k; ++p)
                {
                    for (std::size_t j = 0U; j < nr; ++j)
                    {
                        const float bpj = pb[j];
                        for (std::size_t i = 0U; i < mr; ++i)
                        {
                            acc[j][i] = (Op == GemmOp::Add) ? scalar_fmadd(pa[i], bpj, acc[j][i])
                                                            : scalar_fnmadd(pa[i], bpj, acc[j][i]);
                        }
                    }
                    pa += mr;
                    pb += nr;
                }
                for (std::size_t j = 0U; j < nr; ++j)
                {
                    for (std::size_t i = 0U; i < mr; ++i)
                    {
                        c[(j * ldc) + i] = acc[j][i];
                    }
                }
#endif
            }
        }

        /// @brief Invokes microkernel_mr_nr directly for full tiles or via an aligned stack edge buffer otherwise.
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] rm Active height (`1 <= rm <= MR`).
        /// @param[in] cn Active width (`1 <= cn <= NR`).
        /// @param[in] k Reduction depth.
        /// @param[in] pa Packed `A` micro-panel pointer.
        /// @param[in] pb Packed `B` micro-panel pointer.
        /// @param[in,out] c Pointer to top-left element of the destination sub-block.
        /// @param[in] ldc Column stride of `c`.
        template <GemmOp Op, SimdBlasScalar T>
        inline void update_micro_tile(
            std::size_t rm, std::size_t cn, std::size_t k, const T* pa, const T* pb, T* c, std::size_t ldc
        ) noexcept
        {
            constexpr std::size_t mr = Blas3Layout<T>::mr;
            constexpr std::size_t nr = Blas3Layout<T>::nr;
            if (rm == mr && cn == nr)
            {
                microkernel_mr_nr<Op, T>(k, pa, pb, c, ldc);
                return;
            }
            // Use the full-width microkernel on a zero-padded stack scratch block to preserve identical FMA rounding.
            alignas(64) std::array<T, mr * nr> c_edge{};
            copy_block_strided(c, ldc, c_edge.data(), mr, rm, cn);
            microkernel_mr_nr<Op, T>(k, pa, pb, c_edge.data(), mr);
            copy_block_strided(c_edge.data(), mr, c, ldc, rm, cn);
        }

        /// @brief Shared implementation for `gemm_packed_add` and `gemm_packed_sub`.
        /// @tparam Op Accumulation sign selector (`GemmOp::Add` or `GemmOp::Sub`).
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] m Destination row count (`m <= k_blas3_mc`).
        /// @param[in] n Destination column count (`n <= k_blas3_nc`).
        /// @param[in] k Inner reduction depth (`k <= k_blas3_kc`).
        /// @param[in] packed_a Micro-panel packed `A` matrix.
        /// @param[in] packed_b Micro-panel packed `B` matrix.
        /// @param[in,out] c Destination `m x n` column-major matrix pointer.
        /// @param[in] ldc Column stride of `c` in elements.
        template <GemmOp Op, SimdBlasScalar T>
        void gemm_packed_impl(
            std::size_t m, std::size_t n, std::size_t k, const T* packed_a, const T* packed_b, T* c, std::size_t ldc
        ) noexcept
        {
            if (m == 0U || n == 0U || k == 0U)
            {
                return;
            }
            constexpr std::size_t mr = Blas3Layout<T>::mr;
            constexpr std::size_t nr = Blas3Layout<T>::nr;
            const std::size_t a_panel_stride = mr * k;
            const std::size_t b_panel_stride = nr * k;

            for (std::size_t jb = 0U; jb < n; jb += nr)
            {
                const std::size_t cn = std::min(nr, n - jb);
                const T* const pb = packed_b + ((jb / nr) * b_panel_stride);
                for (std::size_t ib = 0U; ib < m; ib += mr)
                {
                    const std::size_t rm = std::min(mr, m - ib);
                    const T* const pa = packed_a + ((ib / mr) * a_panel_stride);
                    update_micro_tile<Op, T>(rm, cn, k, pa, pb, c + (jb * ldc) + ib, ldc);
                }
            }
        }

        /// @brief Packs strided length-`k` source vectors into `PanelWidth`-wide micro-panels with optional promotion
        /// and masking.
        /// @tparam PanelWidth Compile-time micro-panel width (`mr` for `A`, `nr` for `B`).
        /// @tparam Src Scalar type of source matrix (`float` or `double`).
        /// @tparam Dst Destination packed scalar type (`float` or `double`).
        /// @param[in] src Pointer to `(0, 0)` of column-major matrix whose columns/rows supply length-`k` slices.
        /// @param[in] ld Element stride between adjacent source vectors.
        /// @param[in] count Number of source vectors (`m` for `A^T`, `n` for `B`).
        /// @param[in] k Inner reduction length along each source vector.
        /// @param[out] dst Output micro-panel buffer.
        /// @param[in] diag_mask Vector index threshold below which entries with `p < global_idx` are zeroed without
        /// reading.
        template <std::size_t PanelWidth, SimdBlasScalar Src, SimdBlasScalar Dst>
        void pack_strided_panels(
            const Src* src, std::size_t ld, std::size_t count, std::size_t k, Dst* dst, std::size_t diag_mask
        ) noexcept
        {
            const std::size_t panel_stride = PanelWidth * k;
            for (std::size_t ib = 0U; ib < count; ib += PanelWidth)
            {
                const std::size_t rm = std::min(PanelWidth, count - ib);
                Dst* const dst_panel = dst + ((ib / PanelWidth) * panel_stride);
                // Stream full unmasked panels with compile-time fixed width and no branch per element.
                if (rm == PanelWidth && ib >= diag_mask)
                {
                    for (std::size_t rr = 0U; rr < PanelWidth; ++rr)
                    {
                        const Src* const vec_src = src + ((ib + rr) * ld);
                        for (std::size_t p = 0U; p < k; ++p)
                        {
                            dst_panel[(p * PanelWidth) + rr] = static_cast<Dst>(vec_src[p]);
                        }
                    }
                    continue;
                }
                if (ib >= diag_mask)
                {
                    for (std::size_t rr = 0U; rr < rm; ++rr)
                    {
                        const Src* const vec_src = src + ((ib + rr) * ld);
                        for (std::size_t p = 0U; p < k; ++p)
                        {
                            dst_panel[(p * PanelWidth) + rr] = static_cast<Dst>(vec_src[p]);
                        }
                    }
                }
                else
                {
                    // Zero strictly upper positions first so lower-triangular factors never leak NaNs into GEMM.
                    for (std::size_t rr = 0U; rr < rm; ++rr)
                    {
                        const std::size_t global_idx = ib + rr;
                        const Src* const vec_src = src + (global_idx * ld);
                        const std::size_t p_start = (global_idx < diag_mask) ? std::min(k, global_idx) : 0U;
                        for (std::size_t p = 0U; p < p_start; ++p)
                        {
                            dst_panel[(p * PanelWidth) + rr] = Dst{0};
                        }
                        for (std::size_t p = p_start; p < k; ++p)
                        {
                            dst_panel[(p * PanelWidth) + rr] = static_cast<Dst>(vec_src[p]);
                        }
                    }
                }
                for (std::size_t rr = rm; rr < PanelWidth; ++rr)
                {
                    for (std::size_t p = 0U; p < k; ++p)
                    {
                        dst_panel[(p * PanelWidth) + rr] = Dst{0};
                    }
                }
            }
        }

        /// @brief Packs contiguous column slices of a column-major matrix into `PanelWidth`-wide micro-panels.
        /// @tparam PanelWidth Compile-time micro-panel width (`mr` for `A`, `nr` for `B^T`).
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] src Pointer to `(0, 0)` of column-major matrix (`count x k`).
        /// @param[in] ld Column stride in elements of `src`.
        /// @param[in] count Number of rows in `src` to pack.
        /// @param[in] k Inner reduction dimension (columns of `src`).
        /// @param[out] dst Output micro-panel buffer.
        template <std::size_t PanelWidth, SimdBlasScalar T>
        void pack_contiguous_col_panels(const T* src, std::size_t ld, std::size_t count, std::size_t k, T* dst) noexcept
        {
            const std::size_t panel_stride = PanelWidth * k;
            for (std::size_t ib = 0U; ib < count; ib += PanelWidth)
            {
                const std::size_t rm = std::min(PanelWidth, count - ib);
                T* const dst_panel = dst + ((ib / PanelWidth) * panel_stride);
                // Copy full panels with fixed compile-time width so compilers vectorize without calling memcpy.
                if (rm == PanelWidth)
                {
                    for (std::size_t p = 0U; p < k; ++p)
                    {
                        const T* const src_col = src + (p * ld) + ib;
                        T* const dst_col = dst_panel + (p * PanelWidth);
                        for (std::size_t rr = 0U; rr < PanelWidth; ++rr)
                        {
                            dst_col[rr] = src_col[rr];
                        }
                    }
                    continue;
                }
                for (std::size_t p = 0U; p < k; ++p)
                {
                    const T* const src_col = src + (p * ld) + ib;
                    T* const dst_col = dst_panel + (p * PanelWidth);
                    for (std::size_t rr = 0U; rr < rm; ++rr)
                    {
                        dst_col[rr] = src_col[rr];
                    }
                    for (std::size_t rr = rm; rr < PanelWidth; ++rr)
                    {
                        dst_col[rr] = T{0};
                    }
                }
            }
        }

    }  // namespace

    /// @brief Packs an `m x k` column-major matrix A into ISA-aligned MR-tall micro-panels.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] src Pointer to top-left element `(0, 0)` of the column-major `A` block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] m Number of rows to pack (`m <= k_blas3_mc`).
    /// @param[in] k Number of columns to pack (`k <= k_blas3_kc`).
    /// @param[out] dst Output buffer with capacity at least `k_blas3_pack_a_cap<T>`.
    template <SimdBlasScalar T>
    void pack_panel_a_colmajor(const T* src, std::size_t ld, std::size_t m, std::size_t k, T* dst) noexcept
    {
        pack_contiguous_col_panels<Blas3Layout<T>::mr, T>(src, ld, m, k, dst);
    }

    /// @brief Packs the transpose of a `k x m` column-major matrix into ISA-aligned MR-tall micro-panels.
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
        const T* src, std::size_t ld, std::size_t m, std::size_t k, T* dst, std::size_t diag_mask_rows
    ) noexcept
    {
        pack_strided_panels<Blas3Layout<T>::mr, T, T>(src, ld, m, k, dst, diag_mask_rows);
    }

    /// @brief Packs the transpose of a `k x m` float matrix directly into `double` MR-tall micro-panels.
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `k x m` float column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] m Number of output rows (columns of `src`), at most `k_blas3_mc`.
    /// @param[in] k Number of inner columns (rows of `src`), at most `k_blas3_kc`.
    /// @param[out] dst Output double buffer with capacity at least `k_blas3_pack_a_cap<double>`.
    void pack_panel_a_transpose_promote_f32_to_f64(
        const float* src, std::size_t ld, std::size_t m, std::size_t k, double* dst
    ) noexcept
    {
        pack_strided_panels<Blas3Layout<double>::mr, float, double>(src, ld, m, k, dst, 0U);
    }

    /// @brief Packs the transpose of an `n x k` column-major matrix into ISA-aligned NR-wide B micro-panels.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `n x k` column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] k Number of rows of `B` (columns of `src`), at most `k_blas3_kc`.
    /// @param[in] n Number of columns of `B` (rows of `src`), at most `k_blas3_nc`.
    /// @param[out] dst Output buffer with capacity at least `k_blas3_pack_b_cap`.
    template <SimdBlasScalar T>
    void pack_panel_b_colmajor_t(const T* src, std::size_t ld, std::size_t k, std::size_t n, T* dst) noexcept
    {
        pack_contiguous_col_panels<Blas3Layout<T>::nr, T>(src, ld, n, k, dst);
    }

    /// @brief Packs a `k x n` column-major matrix into ISA-aligned NR-wide B micro-panels.
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
        const T* src, std::size_t ld, std::size_t k, std::size_t n, T* dst, std::size_t diag_mask_cols
    ) noexcept
    {
        pack_strided_panels<Blas3Layout<T>::nr, T, T>(src, ld, n, k, dst, diag_mask_cols);
    }

    /// @brief Packs a `k x n` column-major float matrix directly into `double` NR-wide B micro-panels.
    /// @param[in] src Pointer to top-left element `(0, 0)` of the `k x n` float column-major block.
    /// @param[in] ld Leading dimension (column stride in elements) of `src`.
    /// @param[in] k Number of inner rows to pack (`k <= k_blas3_kc`).
    /// @param[in] n Number of output columns to pack (`n <= k_blas3_nc`).
    /// @param[out] dst Output double buffer with capacity at least `k_blas3_pack_b_cap`.
    void pack_panel_b_colmajor_promote_f32_to_f64(
        const float* src, std::size_t ld, std::size_t k, std::size_t n, double* dst
    ) noexcept
    {
        pack_strided_panels<Blas3Layout<double>::nr, float, double>(src, ld, n, k, dst, 0U);
    }

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
    ) noexcept
    {
        gemm_packed_impl<GemmOp::Sub, T>(m, n, k, packed_a, packed_b, c, ldc);
    }

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
    ) noexcept
    {
        gemm_packed_impl<GemmOp::Add, T>(m, n, k, packed_a, packed_b, c, ldc);
    }

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
    ) noexcept
    {
        if (m == 0U || k == 0U)
        {
            return;
        }
        constexpr std::size_t mr = Blas3Layout<T>::mr;
        constexpr std::size_t nr = Blas3Layout<T>::nr;
        const std::size_t a_panel_stride = mr * k;
        const std::size_t b_panel_stride = nr * k;

        for (std::size_t jb = 0U; jb < m; jb += nr)
        {
            const std::size_t cn = std::min(nr, m - jb);
            const T* const pb = packed_b + ((jb / nr) * b_panel_stride);
            for (std::size_t ib = 0U; ib < m; ib += mr)
            {
                if (ib + mr <= jb)
                {
                    continue;
                }
                const std::size_t rm = std::min(mr, m - ib);
                const T* const pa = packed_a + ((ib / mr) * a_panel_stride);
                if (ib >= (jb + cn) - 1U)
                {
                    // Micro-tile lies completely on or below the diagonal: safe to use full BLAS-3 microkernel.
                    update_micro_tile<GemmOp::Sub, T>(rm, cn, k, pa, pb, c + (jb * ldc) + ib, ldc);
                }
                else
                {
                    // Diagonal-straddling tile: update only lower entries in ascending p order via scalar_fnmadd.
                    for (std::size_t cc = 0U; cc < cn; ++cc)
                    {
                        const std::size_t col = jb + cc;
                        const std::size_t rr_start = (col > ib) ? (col - ib) : 0U;
                        T* const dst_col = c + (col * ldc);
                        for (std::size_t rr = rr_start; rr < rm; ++rr)
                        {
                            const std::size_t row = ib + rr;
                            T acc = dst_col[row];
                            for (std::size_t p = 0U; p < k; ++p)
                            {
                                acc = scalar_fnmadd(pa[(p * mr) + rr], pb[(p * nr) + cc], acc);
                            }
                            dst_col[row] = acc;
                        }
                    }
                }
            }
        }
    }

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
    ) noexcept
    {
        if (bs == 0U || r_begin >= r_end)
        {
            return;
        }
        constexpr std::size_t sub_w = 16U;
        alignas(64) std::array<T, k_blas3_mc * k_blas3_kc> tile_buf{};
        alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
        alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};
        std::array<const T*, sub_w> p_ptrs{};
        std::array<T, sub_w> coeffs{};

        // Copy each row strip into contiguous stack scratch to avoid power-of-2 L1D cache-set conflicts.
        for (std::size_t rb = r_begin; rb < r_end; rb += k_blas3_mc)
        {
            const std::size_t rm = std::min(k_blas3_mc, r_end - rb);
            copy_block_strided(&a(rb, k), a.col_stride(), tile_buf.data(), rm, rm, bs);

            for (std::size_t cb = 0U; cb < bs; cb += sub_w)
            {
                const std::size_t cw = std::min(sub_w, bs - cb);
                for (std::size_t col = 0U; col < cw; ++col)
                {
                    const std::size_t gc = cb + col;
                    T* const col_ptr = tile_buf.data() + (gc * rm);
                    for (std::size_t p = 0U; p < col; ++p)
                    {
                        p_ptrs[p] = tile_buf.data() + ((cb + p) * rm);
                        coeffs[p] = a(k + gc, k + cb + p);
                    }
                    if (col > 0U)
                    {
                        rank_k_update_1col_simd(col_ptr, p_ptrs.data(), col, coeffs.data(), 0U, rm);
                    }
                    const T inv_diag = T{1} / a(k + gc, k + gc);
                    scale_in_place_simd(col_ptr, rm, inv_diag);
                }

                const std::size_t next_cb = cb + cw;
                if (next_cb < bs)
                {
                    const std::size_t trail = bs - next_cb;
                    pack_panel_a_colmajor(tile_buf.data() + (cb * rm), rm, rm, cw, packed_a.data());
                    pack_panel_b_colmajor_t(&a(k + next_cb, k + cb), a.col_stride(), cw, trail, packed_b.data());
                    gemm_packed_sub(
                        rm, trail, cw, packed_a.data(), packed_b.data(), tile_buf.data() + (next_cb * rm), rm
                    );
                }
            }

            copy_block_strided(tile_buf.data(), rm, &a(rb, k), a.col_stride(), rm, bs);
        }
    }

    /// @brief Solves `L[row_start:n, row_start:n] * X[row_start:n, col_begin:col_end] = B` in place via cache-packed
    /// BLAS-3.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] l Square read-only lower Cholesky factor matrix view.
    /// @param[in,out] rhs Mutable right-hand side matrix view updated in place.
    /// @param[in] row_start Inclusive leading row and column index where the triangular solve begins.
    /// @param[in] col_begin Inclusive starting right-hand side column index.
    /// @param[in] col_end Exclusive ending right-hand side column index.
    template <SimdBlasScalar T>
    void solve_lower_block_packed(
        MatrixRef<T> l, MatrixMut<T> rhs, std::size_t row_start, std::size_t col_begin, std::size_t col_end
    ) noexcept
    {
        const std::size_t n = l.rows();
        if (row_start >= n || col_begin >= col_end)
        {
            return;
        }
        alignas(64) std::array<T, k_blas3_kc * k_blas3_nc> x_diag{};
        alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
        alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};

        for (std::size_t cb = col_begin; cb < col_end; cb += k_blas3_nc)
        {
            const std::size_t cn = std::min(k_blas3_nc, col_end - cb);
            for (std::size_t kb = row_start; kb < n; kb += k_blas3_kc)
            {
                const std::size_t ke = std::min(kb + k_blas3_kc, n);
                const std::size_t bs = ke - kb;
                copy_block_strided(&rhs(kb, cb), rhs.col_stride(), x_diag.data(), bs, bs, cn);
                solve_diag_lower_block(l, kb, bs, cn, x_diag.data());
                copy_block_strided(x_diag.data(), bs, &rhs(kb, cb), rhs.col_stride(), bs, cn);

                if (ke < n)
                {
                    pack_panel_b_colmajor(x_diag.data(), bs, bs, cn, packed_b.data(), 0U);
                    for (std::size_t ib = ke; ib < n; ib += k_blas3_mc)
                    {
                        const std::size_t rm = std::min(k_blas3_mc, n - ib);
                        pack_panel_a_colmajor(&l(ib, kb), l.col_stride(), rm, bs, packed_a.data());
                        gemm_packed_sub(rm, cn, bs, packed_a.data(), packed_b.data(), &rhs(ib, cb), rhs.col_stride());
                    }
                }
            }
        }
    }

    /// @brief Computes columns `[col_begin, col_end)` of lower-triangular `Z = L^-1` via cache-packed BLAS-3.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] l Square read-only lower Cholesky factor matrix view.
    /// @param[out] z Mutable `n x n` destination matrix view receiving lower-triangular `Z = L^-1` in `[col_begin,
    /// col_end)`.
    /// @param[in] col_begin Inclusive starting column index in `z`.
    /// @param[in] col_end Exclusive ending column index in `z` (`col_end - col_begin <= k_blas3_nc`).
    template <SimdBlasScalar T>
    void invert_lower_col_strip_packed(
        MatrixRef<T> l, MatrixMut<T> z, std::size_t col_begin, std::size_t col_end
    ) noexcept
    {
        const std::size_t n = l.rows();
        const std::size_t cw = col_end - col_begin;
        if (cw == 0U || col_end > n)
        {
            return;
        }

        const std::size_t row_start = col_end;
        {
            alignas(64) std::array<T, k_blas3_nc * k_blas3_nc> z_diag{};
            for (std::size_t c = 0U; c < cw; ++c)
            {
                const std::size_t jc = col_begin + c;
                const T* const l_col = l.col(jc).data();
                T* const z_col = z_diag.data() + (c * cw);
                const T z_jj = T{1} / l_col[jc];
                z_col[c] = z_jj;
                for (std::size_t r = c + 1U; r < cw; ++r)
                {
                    z_col[r] = -(l_col[col_begin + r] * z_jj);
                }
                for (std::size_t p = c + 1U; p < cw; ++p)
                {
                    const T* const lp = l.col(col_begin + p).data();
                    const T z_pc = z_col[p] / lp[col_begin + p];
                    z_col[p] = z_pc;
                    for (std::size_t r = p + 1U; r < cw; ++r)
                    {
                        z_col[r] = scalar_fnmadd(lp[col_begin + r], z_pc, z_col[r]);
                    }
                }
            }

            // Copy only the lower triangle of Z11 to z without touching strictly upper entries.
            for (std::size_t c = 0U; c < cw; ++c)
            {
                const T* const src_col = z_diag.data() + (c * cw);
                T* const dst_col = z.col(col_begin + c).data();
                std::copy_n(src_col + c, cw - c, dst_col + col_begin + c);
            }

            if (row_start < n)
            {
                alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
                alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};
                for (std::size_t c = 0U; c < cw; ++c)
                {
                    std::fill_n(z.col(col_begin + c).data() + row_start, n - row_start, T{0});
                }
                pack_panel_b_colmajor(z_diag.data(), cw, cw, cw, packed_b.data(), 0U);
                for (std::size_t ib = row_start; ib < n; ib += k_blas3_mc)
                {
                    const std::size_t rm = std::min(k_blas3_mc, n - ib);
                    pack_panel_a_colmajor(&l(ib, col_begin), l.col_stride(), rm, cw, packed_a.data());
                    gemm_packed_sub(rm, cw, cw, packed_a.data(), packed_b.data(), &z(ib, col_begin), z.col_stride());
                }
            }
        }

        if (row_start < n)
        {
            solve_lower_block_packed(l, z, row_start, col_begin, col_end);
        }
    }

    template void pack_panel_a_colmajor<double>(const double*, std::size_t, std::size_t, std::size_t, double*) noexcept;
    template void pack_panel_a_colmajor<float>(const float*, std::size_t, std::size_t, std::size_t, float*) noexcept;

    template void pack_panel_a_transpose<double>(
        const double*, std::size_t, std::size_t, std::size_t, double*, std::size_t
    ) noexcept;
    template void pack_panel_a_transpose<float>(
        const float*, std::size_t, std::size_t, std::size_t, float*, std::size_t
    ) noexcept;

    template void pack_panel_b_colmajor_t<double>(
        const double*, std::size_t, std::size_t, std::size_t, double*
    ) noexcept;
    template void pack_panel_b_colmajor_t<float>(const float*, std::size_t, std::size_t, std::size_t, float*) noexcept;

    template void pack_panel_b_colmajor<double>(
        const double*, std::size_t, std::size_t, std::size_t, double*, std::size_t
    ) noexcept;
    template void pack_panel_b_colmajor<float>(
        const float*, std::size_t, std::size_t, std::size_t, float*, std::size_t
    ) noexcept;

    template void gemm_packed_sub<double>(
        std::size_t, std::size_t, std::size_t, const double*, const double*, double*, std::size_t
    ) noexcept;
    template void gemm_packed_sub<float>(
        std::size_t, std::size_t, std::size_t, const float*, const float*, float*, std::size_t
    ) noexcept;

    template void gemm_packed_add<double>(
        std::size_t, std::size_t, std::size_t, const double*, const double*, double*, std::size_t
    ) noexcept;
    template void gemm_packed_add<float>(
        std::size_t, std::size_t, std::size_t, const float*, const float*, float*, std::size_t
    ) noexcept;

    template void syrk_packed_lower_diag_sub<double>(
        std::size_t, std::size_t, const double*, const double*, double*, std::size_t
    ) noexcept;
    template void syrk_packed_lower_diag_sub<float>(
        std::size_t, std::size_t, const float*, const float*, float*, std::size_t
    ) noexcept;

    template void solve_panel_rows_packed<double>(
        MatrixMut<double>, std::size_t, std::size_t, std::size_t, std::size_t
    ) noexcept;
    template void solve_panel_rows_packed<float>(
        MatrixMut<float>, std::size_t, std::size_t, std::size_t, std::size_t
    ) noexcept;

    template void solve_lower_block_packed<double>(
        MatrixRef<double>, MatrixMut<double>, std::size_t, std::size_t, std::size_t
    ) noexcept;
    template void solve_lower_block_packed<float>(
        MatrixRef<float>, MatrixMut<float>, std::size_t, std::size_t, std::size_t
    ) noexcept;

    template void invert_lower_col_strip_packed<double>(
        MatrixRef<double>, MatrixMut<double>, std::size_t, std::size_t
    ) noexcept;
    template void invert_lower_col_strip_packed<float>(
        MatrixRef<float>, MatrixMut<float>, std::size_t, std::size_t
    ) noexcept;

}  // namespace cxxgpr::detail
