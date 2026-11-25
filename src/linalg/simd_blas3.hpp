// SPDX-License-Identifier: Apache-2.0
/// @file simd_blas3.hpp
/// @brief Internal 3-tier x86-64 SIMD register-blocked BLAS-3 microkernels and dot products.
/// @details Implements AVX-512, AVX2+FMA, SSE2, and portable C++23 scalar kernels for Cholesky rank-k
/// panel/trailing updates, triangular solves, symmetric W = alpha * alpha^T - A^-1 inner products,
/// posterior variance/covariance projections, and mixed-precision float-to-double reductions while
/// enforcing deterministic accumulation order invariants across OpenMP thread counts.
#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

/// @brief Internal register-blocked SIMD linear algebra microkernels.
namespace cxxgpr::detail
{

    /// @brief Floating-point scalar type supported by SIMD BLAS-3 microkernels (`double` or `float`).
    /// @tparam T Candidate scalar type.
    template <typename T>
    concept SimdBlasScalar = std::same_as<T, double> || std::same_as<T, float>;

#if defined(__AVX512F__) && defined(__AVX512DQ__)

    /// @brief Computes `a * b + c` across 8 binary64 lanes using AVX-512 FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m512d blas3_fmadd_pd(__m512d a, __m512d b, __m512d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fmadd_pd(a, b, c);
#    else
        return _mm512_add_pd(_mm512_mul_pd(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 8 binary64 lanes using AVX-512 FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m512d blas3_fnmadd_pd(__m512d a, __m512d b, __m512d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fnmadd_pd(a, b, c);
#    else
        return _mm512_sub_pd(c, _mm512_mul_pd(a, b));
#    endif
    }

    /// @brief Computes `a * b + c` across 16 binary32 lanes using AVX-512 FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m512 blas3_fmadd_ps(__m512 a, __m512 b, __m512 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fmadd_ps(a, b, c);
#    else
        return _mm512_add_ps(_mm512_mul_ps(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 16 binary32 lanes using AVX-512 FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m512 blas3_fnmadd_ps(__m512 a, __m512 b, __m512 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fnmadd_ps(a, b, c);
#    else
        return _mm512_sub_ps(c, _mm512_mul_ps(a, b));
#    endif
    }

    /// @brief Reduces 8 binary64 AVX-512 lanes in fixed binary-tree lane order.
    /// @param[in] v Input 512-bit packed binary64 vector.
    /// @return Deterministic scalar sum `((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7))`.
    [[nodiscard]] inline double hsum_zmm_pd(__m512d v) noexcept
    {
        alignas(64) std::array<double, 8> tmp{};
        _mm512_store_pd(tmp.data(), v);
        return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
    }

    /// @brief Reduces 16 binary32 AVX-512 lanes in fixed binary-tree lane order.
    /// @param[in] v Input 512-bit packed binary32 vector.
    /// @return Deterministic scalar sum across all 16 lanes in fixed pairwise order.
    [[nodiscard]] inline float hsum_zmm_ps(__m512 v) noexcept
    {
        alignas(64) std::array<float, 16> tmp{};
        _mm512_store_ps(tmp.data(), v);
        const float lo = ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
        const float hi = ((tmp[8] + tmp[9]) + (tmp[10] + tmp[11])) + ((tmp[12] + tmp[13]) + (tmp[14] + tmp[15]));
        return lo + hi;
    }

#elifdef __AVX2__

    /// @brief Computes `a * b + c` across 4 binary64 lanes using AVX2 FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m256d blas3_fmadd_pd(__m256d a, __m256d b, __m256d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fmadd_pd(a, b, c);
#    else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 4 binary64 lanes using AVX2 FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m256d blas3_fnmadd_pd(__m256d a, __m256d b, __m256d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fnmadd_pd(a, b, c);
#    else
        return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#    endif
    }

    /// @brief Computes `a * b + c` across 8 binary32 lanes using AVX2 FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m256 blas3_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fmadd_ps(a, b, c);
#    else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 8 binary32 lanes using AVX2 FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m256 blas3_fnmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fnmadd_ps(a, b, c);
#    else
        return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#    endif
    }

    /// @brief Reduces 4 binary64 AVX2 lanes in fixed binary-tree lane order.
    /// @param[in] v Input 256-bit packed binary64 vector.
    /// @return Deterministic scalar sum `(v0 + v1) + (v2 + v3)`.
    [[nodiscard]] inline double hsum_ymm_pd(__m256d v) noexcept
    {
        alignas(32) std::array<double, 4> tmp{};
        _mm256_store_pd(tmp.data(), v);
        return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    }

    /// @brief Reduces 8 binary32 AVX2 lanes in fixed binary-tree lane order.
    /// @param[in] v Input 256-bit packed binary32 vector.
    /// @return Deterministic scalar sum across all 8 lanes in fixed pairwise order.
    [[nodiscard]] inline float hsum_ymm_ps(__m256 v) noexcept
    {
        alignas(32) std::array<float, 8> tmp{};
        _mm256_store_ps(tmp.data(), v);
        return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
    }

#elif defined(_M_X64) || defined(__x86_64__)

    /// @brief Reduces 2 binary64 SSE2 lanes in fixed ascending lane order.
    /// @param[in] v Input 128-bit packed binary64 vector.
    /// @return Deterministic scalar sum `v0 + v1`.
    [[nodiscard]] inline double hsum_xmm_pd(__m128d v) noexcept
    {
        alignas(16) std::array<double, 2> tmp{};
        _mm_store_pd(tmp.data(), v);
        return tmp[0] + tmp[1];
    }

    /// @brief Reduces 4 binary32 SSE2 lanes in fixed binary-tree lane order.
    /// @param[in] v Input 128-bit packed binary32 vector.
    /// @return Deterministic scalar sum `(v0 + v1) + (v2 + v3)`.
    [[nodiscard]] inline float hsum_xmm_ps(__m128 v) noexcept
    {
        alignas(16) std::array<float, 4> tmp{};
        _mm_store_ps(tmp.data(), v);
        return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    }

#endif

    /// @brief Executes scalar FMA subtraction `c - a * b` matching SIMD FMA contraction when enabled.
    /// @tparam T Scalar type (`double` or `float`).
    /// @param[in] a First multiplicand.
    /// @param[in] b Second multiplicand.
    /// @param[in] c Minuend.
    /// @return `c - a * b` computed with fused multiply-add on FMA targets.
    template <SimdBlasScalar T> [[nodiscard]] inline T scalar_fnmadd(T a, T b, T c) noexcept
    {
#if ((defined(__AVX512F__) && defined(__AVX512DQ__)) || defined(__AVX2__)) && (defined(__FMA__) || defined(_MSC_VER))
        return std::fma(-a, b, c);
#else
        return c - (a * b);
#endif
    }

    /// @brief Executes scalar FMA addition `a * b + c` matching SIMD FMA contraction when enabled.
    /// @tparam T Scalar type (`double` or `float`).
    /// @param[in] a First multiplicand.
    /// @param[in] b Second multiplicand.
    /// @param[in] c Addend.
    /// @return `a * b + c` computed with fused multiply-add on FMA targets.
    template <SimdBlasScalar T> [[nodiscard]] inline T scalar_fmadd(T a, T b, T c) noexcept
    {
#if ((defined(__AVX512F__) && defined(__AVX512DQ__)) || defined(__AVX2__)) && (defined(__FMA__) || defined(_MSC_VER))
        return std::fma(a, b, c);
#else
        return c + (a * b);
#endif
    }

    /// @brief Applies a fused rank-bs subtraction update `col_c[r] -= sum_p l_cols[p][r] * v_c[p]` across 4 columns.
    /// @details Uses 8-accumulator register tiling (`2 * VEC` rows by 4 columns) so destination rows are loaded once,
    /// updated across all `p = 0..bs-1` in registers in ascending `p` order, and stored once, followed by single-vector
    /// and scalar tail cleanups.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in,out] col0 Pointer to destination column 0 base (index 0 corresponds to row 0).
    /// @param[in,out] col1 Pointer to destination column 1 base (index 0 corresponds to row 0).
    /// @param[in,out] col2 Pointer to destination column 2 base (index 0 corresponds to row 0).
    /// @param[in,out] col3 Pointer to destination column 3 base (index 0 corresponds to row 0).
    /// @param[in] l_cols Array of `bs` read-only panel column base pointers.
    /// @param[in] bs Panel width (number of rank-1 terms accumulated in ascending order).
    /// @param[in] v0 Array of `bs` multipliers for column 0.
    /// @param[in] v1 Array of `bs` multipliers for column 1.
    /// @param[in] v2 Array of `bs` multipliers for column 2.
    /// @param[in] v3 Array of `bs` multipliers for column 3.
    /// @param[in] r_begin Inclusive starting row index to update.
    /// @param[in] r_end Exclusive ending row index to update.
    template <SimdBlasScalar T>
    inline void rank_k_update_4cols_simd(
        T* col0,
        T* col1,
        T* col2,
        T* col3,
        const T* const* l_cols,
        std::size_t bs,
        const T* v0,
        const T* v1,
        const T* v2,
        const T* v3,
        std::size_t r_begin,
        std::size_t r_end
    ) noexcept
    {
        if (bs == 0U || r_begin >= r_end)
        {
            return;
        }
        std::size_t r = r_begin;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            for (; r + 16U <= r_end; r += 16U)
            {
                __m512d a00 = _mm512_loadu_pd(col0 + r);
                __m512d a01 = _mm512_loadu_pd(col0 + r + 8U);
                __m512d a10 = _mm512_loadu_pd(col1 + r);
                __m512d a11 = _mm512_loadu_pd(col1 + r + 8U);
                __m512d a20 = _mm512_loadu_pd(col2 + r);
                __m512d a21 = _mm512_loadu_pd(col2 + r + 8U);
                __m512d a30 = _mm512_loadu_pd(col3 + r);
                __m512d a31 = _mm512_loadu_pd(col3 + r + 8U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const double* const lp = l_cols[p] + r;
                    const __m512d lp0 = _mm512_loadu_pd(lp);
                    const __m512d lp1 = _mm512_loadu_pd(lp + 8U);
                    const __m512d vb0 = _mm512_set1_pd(v0[p]);
                    const __m512d vb1 = _mm512_set1_pd(v1[p]);
                    const __m512d vb2 = _mm512_set1_pd(v2[p]);
                    const __m512d vb3 = _mm512_set1_pd(v3[p]);
                    a00 = blas3_fnmadd_pd(lp0, vb0, a00);
                    a10 = blas3_fnmadd_pd(lp0, vb1, a10);
                    a20 = blas3_fnmadd_pd(lp0, vb2, a20);
                    a30 = blas3_fnmadd_pd(lp0, vb3, a30);
                    a01 = blas3_fnmadd_pd(lp1, vb0, a01);
                    a11 = blas3_fnmadd_pd(lp1, vb1, a11);
                    a21 = blas3_fnmadd_pd(lp1, vb2, a21);
                    a31 = blas3_fnmadd_pd(lp1, vb3, a31);
                }
                _mm512_storeu_pd(col0 + r, a00);
                _mm512_storeu_pd(col0 + r + 8U, a01);
                _mm512_storeu_pd(col1 + r, a10);
                _mm512_storeu_pd(col1 + r + 8U, a11);
                _mm512_storeu_pd(col2 + r, a20);
                _mm512_storeu_pd(col2 + r + 8U, a21);
                _mm512_storeu_pd(col3 + r, a30);
                _mm512_storeu_pd(col3 + r + 8U, a31);
            }
            for (; r + 8U <= r_end; r += 8U)
            {
                __m512d a0 = _mm512_loadu_pd(col0 + r);
                __m512d a1 = _mm512_loadu_pd(col1 + r);
                __m512d a2 = _mm512_loadu_pd(col2 + r);
                __m512d a3 = _mm512_loadu_pd(col3 + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const __m512d lp = _mm512_loadu_pd(l_cols[p] + r);
                    a0 = blas3_fnmadd_pd(lp, _mm512_set1_pd(v0[p]), a0);
                    a1 = blas3_fnmadd_pd(lp, _mm512_set1_pd(v1[p]), a1);
                    a2 = blas3_fnmadd_pd(lp, _mm512_set1_pd(v2[p]), a2);
                    a3 = blas3_fnmadd_pd(lp, _mm512_set1_pd(v3[p]), a3);
                }
                _mm512_storeu_pd(col0 + r, a0);
                _mm512_storeu_pd(col1 + r, a1);
                _mm512_storeu_pd(col2 + r, a2);
                _mm512_storeu_pd(col3 + r, a3);
            }
#elifdef __AVX2__
            for (; r + 8U <= r_end; r += 8U)
            {
                __m256d a00 = _mm256_loadu_pd(col0 + r);
                __m256d a01 = _mm256_loadu_pd(col0 + r + 4U);
                __m256d a10 = _mm256_loadu_pd(col1 + r);
                __m256d a11 = _mm256_loadu_pd(col1 + r + 4U);
                __m256d a20 = _mm256_loadu_pd(col2 + r);
                __m256d a21 = _mm256_loadu_pd(col2 + r + 4U);
                __m256d a30 = _mm256_loadu_pd(col3 + r);
                __m256d a31 = _mm256_loadu_pd(col3 + r + 4U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const double* const lp = l_cols[p] + r;
                    const __m256d lp0 = _mm256_loadu_pd(lp);
                    const __m256d vb0 = _mm256_set1_pd(v0[p]);
                    const __m256d vb1 = _mm256_set1_pd(v1[p]);
                    const __m256d vb2 = _mm256_set1_pd(v2[p]);
                    const __m256d vb3 = _mm256_set1_pd(v3[p]);
                    a00 = blas3_fnmadd_pd(lp0, vb0, a00);
                    a10 = blas3_fnmadd_pd(lp0, vb1, a10);
                    a20 = blas3_fnmadd_pd(lp0, vb2, a20);
                    a30 = blas3_fnmadd_pd(lp0, vb3, a30);
                    const __m256d lp1 = _mm256_loadu_pd(lp + 4U);
                    a01 = blas3_fnmadd_pd(lp1, vb0, a01);
                    a11 = blas3_fnmadd_pd(lp1, vb1, a11);
                    a21 = blas3_fnmadd_pd(lp1, vb2, a21);
                    a31 = blas3_fnmadd_pd(lp1, vb3, a31);
                }
                _mm256_storeu_pd(col0 + r, a00);
                _mm256_storeu_pd(col0 + r + 4U, a01);
                _mm256_storeu_pd(col1 + r, a10);
                _mm256_storeu_pd(col1 + r + 4U, a11);
                _mm256_storeu_pd(col2 + r, a20);
                _mm256_storeu_pd(col2 + r + 4U, a21);
                _mm256_storeu_pd(col3 + r, a30);
                _mm256_storeu_pd(col3 + r + 4U, a31);
            }
            for (; r + 4U <= r_end; r += 4U)
            {
                __m256d a0 = _mm256_loadu_pd(col0 + r);
                __m256d a1 = _mm256_loadu_pd(col1 + r);
                __m256d a2 = _mm256_loadu_pd(col2 + r);
                __m256d a3 = _mm256_loadu_pd(col3 + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const __m256d lp = _mm256_loadu_pd(l_cols[p] + r);
                    a0 = blas3_fnmadd_pd(lp, _mm256_set1_pd(v0[p]), a0);
                    a1 = blas3_fnmadd_pd(lp, _mm256_set1_pd(v1[p]), a1);
                    a2 = blas3_fnmadd_pd(lp, _mm256_set1_pd(v2[p]), a2);
                    a3 = blas3_fnmadd_pd(lp, _mm256_set1_pd(v3[p]), a3);
                }
                _mm256_storeu_pd(col0 + r, a0);
                _mm256_storeu_pd(col1 + r, a1);
                _mm256_storeu_pd(col2 + r, a2);
                _mm256_storeu_pd(col3 + r, a3);
            }
#elif defined(_M_X64) || defined(__x86_64__)
            for (; r + 4U <= r_end; r += 4U)
            {
                __m128d a00 = _mm_loadu_pd(col0 + r);
                __m128d a01 = _mm_loadu_pd(col0 + r + 2U);
                __m128d a10 = _mm_loadu_pd(col1 + r);
                __m128d a11 = _mm_loadu_pd(col1 + r + 2U);
                __m128d a20 = _mm_loadu_pd(col2 + r);
                __m128d a21 = _mm_loadu_pd(col2 + r + 2U);
                __m128d a30 = _mm_loadu_pd(col3 + r);
                __m128d a31 = _mm_loadu_pd(col3 + r + 2U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const double* const lp = l_cols[p] + r;
                    const __m128d lp0 = _mm_loadu_pd(lp);
                    const __m128d lp1 = _mm_loadu_pd(lp + 2U);
                    const __m128d vb0 = _mm_set1_pd(v0[p]);
                    const __m128d vb1 = _mm_set1_pd(v1[p]);
                    const __m128d vb2 = _mm_set1_pd(v2[p]);
                    const __m128d vb3 = _mm_set1_pd(v3[p]);
                    a00 = _mm_sub_pd(a00, _mm_mul_pd(lp0, vb0));
                    a01 = _mm_sub_pd(a01, _mm_mul_pd(lp1, vb0));
                    a10 = _mm_sub_pd(a10, _mm_mul_pd(lp0, vb1));
                    a11 = _mm_sub_pd(a11, _mm_mul_pd(lp1, vb1));
                    a20 = _mm_sub_pd(a20, _mm_mul_pd(lp0, vb2));
                    a21 = _mm_sub_pd(a21, _mm_mul_pd(lp1, vb2));
                    a30 = _mm_sub_pd(a30, _mm_mul_pd(lp0, vb3));
                    a31 = _mm_sub_pd(a31, _mm_mul_pd(lp1, vb3));
                }
                _mm_storeu_pd(col0 + r, a00);
                _mm_storeu_pd(col0 + r + 2U, a01);
                _mm_storeu_pd(col1 + r, a10);
                _mm_storeu_pd(col1 + r + 2U, a11);
                _mm_storeu_pd(col2 + r, a20);
                _mm_storeu_pd(col2 + r + 2U, a21);
                _mm_storeu_pd(col3 + r, a30);
                _mm_storeu_pd(col3 + r + 2U, a31);
            }
            for (; r + 2U <= r_end; r += 2U)
            {
                __m128d a0 = _mm_loadu_pd(col0 + r);
                __m128d a1 = _mm_loadu_pd(col1 + r);
                __m128d a2 = _mm_loadu_pd(col2 + r);
                __m128d a3 = _mm_loadu_pd(col3 + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const __m128d lp = _mm_loadu_pd(l_cols[p] + r);
                    a0 = _mm_sub_pd(a0, _mm_mul_pd(lp, _mm_set1_pd(v0[p])));
                    a1 = _mm_sub_pd(a1, _mm_mul_pd(lp, _mm_set1_pd(v1[p])));
                    a2 = _mm_sub_pd(a2, _mm_mul_pd(lp, _mm_set1_pd(v2[p])));
                    a3 = _mm_sub_pd(a3, _mm_mul_pd(lp, _mm_set1_pd(v3[p])));
                }
                _mm_storeu_pd(col0 + r, a0);
                _mm_storeu_pd(col1 + r, a1);
                _mm_storeu_pd(col2 + r, a2);
                _mm_storeu_pd(col3 + r, a3);
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            for (; r + 32U <= r_end; r += 32U)
            {
                __m512 a00 = _mm512_loadu_ps(col0 + r);
                __m512 a01 = _mm512_loadu_ps(col0 + r + 16U);
                __m512 a10 = _mm512_loadu_ps(col1 + r);
                __m512 a11 = _mm512_loadu_ps(col1 + r + 16U);
                __m512 a20 = _mm512_loadu_ps(col2 + r);
                __m512 a21 = _mm512_loadu_ps(col2 + r + 16U);
                __m512 a30 = _mm512_loadu_ps(col3 + r);
                __m512 a31 = _mm512_loadu_ps(col3 + r + 16U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const float* const lp = l_cols[p] + r;
                    const __m512 lp0 = _mm512_loadu_ps(lp);
                    const __m512 lp1 = _mm512_loadu_ps(lp + 16U);
                    const __m512 vb0 = _mm512_set1_ps(v0[p]);
                    const __m512 vb1 = _mm512_set1_ps(v1[p]);
                    const __m512 vb2 = _mm512_set1_ps(v2[p]);
                    const __m512 vb3 = _mm512_set1_ps(v3[p]);
                    a00 = blas3_fnmadd_ps(lp0, vb0, a00);
                    a10 = blas3_fnmadd_ps(lp0, vb1, a10);
                    a20 = blas3_fnmadd_ps(lp0, vb2, a20);
                    a30 = blas3_fnmadd_ps(lp0, vb3, a30);
                    a01 = blas3_fnmadd_ps(lp1, vb0, a01);
                    a11 = blas3_fnmadd_ps(lp1, vb1, a11);
                    a21 = blas3_fnmadd_ps(lp1, vb2, a21);
                    a31 = blas3_fnmadd_ps(lp1, vb3, a31);
                }
                _mm512_storeu_ps(col0 + r, a00);
                _mm512_storeu_ps(col0 + r + 16U, a01);
                _mm512_storeu_ps(col1 + r, a10);
                _mm512_storeu_ps(col1 + r + 16U, a11);
                _mm512_storeu_ps(col2 + r, a20);
                _mm512_storeu_ps(col2 + r + 16U, a21);
                _mm512_storeu_ps(col3 + r, a30);
                _mm512_storeu_ps(col3 + r + 16U, a31);
            }
            for (; r + 16U <= r_end; r += 16U)
            {
                __m512 a0 = _mm512_loadu_ps(col0 + r);
                __m512 a1 = _mm512_loadu_ps(col1 + r);
                __m512 a2 = _mm512_loadu_ps(col2 + r);
                __m512 a3 = _mm512_loadu_ps(col3 + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const __m512 lp = _mm512_loadu_ps(l_cols[p] + r);
                    a0 = blas3_fnmadd_ps(lp, _mm512_set1_ps(v0[p]), a0);
                    a1 = blas3_fnmadd_ps(lp, _mm512_set1_ps(v1[p]), a1);
                    a2 = blas3_fnmadd_ps(lp, _mm512_set1_ps(v2[p]), a2);
                    a3 = blas3_fnmadd_ps(lp, _mm512_set1_ps(v3[p]), a3);
                }
                _mm512_storeu_ps(col0 + r, a0);
                _mm512_storeu_ps(col1 + r, a1);
                _mm512_storeu_ps(col2 + r, a2);
                _mm512_storeu_ps(col3 + r, a3);
            }
#elifdef __AVX2__
            for (; r + 16U <= r_end; r += 16U)
            {
                __m256 a00 = _mm256_loadu_ps(col0 + r);
                __m256 a01 = _mm256_loadu_ps(col0 + r + 8U);
                __m256 a10 = _mm256_loadu_ps(col1 + r);
                __m256 a11 = _mm256_loadu_ps(col1 + r + 8U);
                __m256 a20 = _mm256_loadu_ps(col2 + r);
                __m256 a21 = _mm256_loadu_ps(col2 + r + 8U);
                __m256 a30 = _mm256_loadu_ps(col3 + r);
                __m256 a31 = _mm256_loadu_ps(col3 + r + 8U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const float* const lp = l_cols[p] + r;
                    const __m256 lp0 = _mm256_loadu_ps(lp);
                    const __m256 vb0 = _mm256_set1_ps(v0[p]);
                    const __m256 vb1 = _mm256_set1_ps(v1[p]);
                    const __m256 vb2 = _mm256_set1_ps(v2[p]);
                    const __m256 vb3 = _mm256_set1_ps(v3[p]);
                    a00 = blas3_fnmadd_ps(lp0, vb0, a00);
                    a10 = blas3_fnmadd_ps(lp0, vb1, a10);
                    a20 = blas3_fnmadd_ps(lp0, vb2, a20);
                    a30 = blas3_fnmadd_ps(lp0, vb3, a30);
                    const __m256 lp1 = _mm256_loadu_ps(lp + 8U);
                    a01 = blas3_fnmadd_ps(lp1, vb0, a01);
                    a11 = blas3_fnmadd_ps(lp1, vb1, a11);
                    a21 = blas3_fnmadd_ps(lp1, vb2, a21);
                    a31 = blas3_fnmadd_ps(lp1, vb3, a31);
                }
                _mm256_storeu_ps(col0 + r, a00);
                _mm256_storeu_ps(col0 + r + 8U, a01);
                _mm256_storeu_ps(col1 + r, a10);
                _mm256_storeu_ps(col1 + r + 8U, a11);
                _mm256_storeu_ps(col2 + r, a20);
                _mm256_storeu_ps(col2 + r + 8U, a21);
                _mm256_storeu_ps(col3 + r, a30);
                _mm256_storeu_ps(col3 + r + 8U, a31);
            }
            for (; r + 8U <= r_end; r += 8U)
            {
                __m256 a0 = _mm256_loadu_ps(col0 + r);
                __m256 a1 = _mm256_loadu_ps(col1 + r);
                __m256 a2 = _mm256_loadu_ps(col2 + r);
                __m256 a3 = _mm256_loadu_ps(col3 + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const __m256 lp = _mm256_loadu_ps(l_cols[p] + r);
                    a0 = blas3_fnmadd_ps(lp, _mm256_set1_ps(v0[p]), a0);
                    a1 = blas3_fnmadd_ps(lp, _mm256_set1_ps(v1[p]), a1);
                    a2 = blas3_fnmadd_ps(lp, _mm256_set1_ps(v2[p]), a2);
                    a3 = blas3_fnmadd_ps(lp, _mm256_set1_ps(v3[p]), a3);
                }
                _mm256_storeu_ps(col0 + r, a0);
                _mm256_storeu_ps(col1 + r, a1);
                _mm256_storeu_ps(col2 + r, a2);
                _mm256_storeu_ps(col3 + r, a3);
            }
#elif defined(_M_X64) || defined(__x86_64__)
            for (; r + 8U <= r_end; r += 8U)
            {
                __m128 a00 = _mm_loadu_ps(col0 + r);
                __m128 a01 = _mm_loadu_ps(col0 + r + 4U);
                __m128 a10 = _mm_loadu_ps(col1 + r);
                __m128 a11 = _mm_loadu_ps(col1 + r + 4U);
                __m128 a20 = _mm_loadu_ps(col2 + r);
                __m128 a21 = _mm_loadu_ps(col2 + r + 4U);
                __m128 a30 = _mm_loadu_ps(col3 + r);
                __m128 a31 = _mm_loadu_ps(col3 + r + 4U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const float* const lp = l_cols[p] + r;
                    const __m128 lp0 = _mm_loadu_ps(lp);
                    const __m128 lp1 = _mm_loadu_ps(lp + 4U);
                    const __m128 vb0 = _mm_set1_ps(v0[p]);
                    const __m128 vb1 = _mm_set1_ps(v1[p]);
                    const __m128 vb2 = _mm_set1_ps(v2[p]);
                    const __m128 vb3 = _mm_set1_ps(v3[p]);
                    a00 = _mm_sub_ps(a00, _mm_mul_ps(lp0, vb0));
                    a01 = _mm_sub_ps(a01, _mm_mul_ps(lp1, vb0));
                    a10 = _mm_sub_ps(a10, _mm_mul_ps(lp0, vb1));
                    a11 = _mm_sub_ps(a11, _mm_mul_ps(lp1, vb1));
                    a20 = _mm_sub_ps(a20, _mm_mul_ps(lp0, vb2));
                    a21 = _mm_sub_ps(a21, _mm_mul_ps(lp1, vb2));
                    a30 = _mm_sub_ps(a30, _mm_mul_ps(lp0, vb3));
                    a31 = _mm_sub_ps(a31, _mm_mul_ps(lp1, vb3));
                }
                _mm_storeu_ps(col0 + r, a00);
                _mm_storeu_ps(col0 + r + 4U, a01);
                _mm_storeu_ps(col1 + r, a10);
                _mm_storeu_ps(col1 + r + 4U, a11);
                _mm_storeu_ps(col2 + r, a20);
                _mm_storeu_ps(col2 + r + 4U, a21);
                _mm_storeu_ps(col3 + r, a30);
                _mm_storeu_ps(col3 + r + 4U, a31);
            }
            for (; r + 4U <= r_end; r += 4U)
            {
                __m128 a0 = _mm_loadu_ps(col0 + r);
                __m128 a1 = _mm_loadu_ps(col1 + r);
                __m128 a2 = _mm_loadu_ps(col2 + r);
                __m128 a3 = _mm_loadu_ps(col3 + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const __m128 lp = _mm_loadu_ps(l_cols[p] + r);
                    a0 = _mm_sub_ps(a0, _mm_mul_ps(lp, _mm_set1_ps(v0[p])));
                    a1 = _mm_sub_ps(a1, _mm_mul_ps(lp, _mm_set1_ps(v1[p])));
                    a2 = _mm_sub_ps(a2, _mm_mul_ps(lp, _mm_set1_ps(v2[p])));
                    a3 = _mm_sub_ps(a3, _mm_mul_ps(lp, _mm_set1_ps(v3[p])));
                }
                _mm_storeu_ps(col0 + r, a0);
                _mm_storeu_ps(col1 + r, a1);
                _mm_storeu_ps(col2 + r, a2);
                _mm_storeu_ps(col3 + r, a3);
            }
#endif
        }
        for (; r < r_end; ++r)
        {
            T s0 = col0[r];
            T s1 = col1[r];
            T s2 = col2[r];
            T s3 = col3[r];
            for (std::size_t p = 0U; p < bs; ++p)
            {
                const T lrp = l_cols[p][r];
                s0 = scalar_fnmadd(lrp, v0[p], s0);
                s1 = scalar_fnmadd(lrp, v1[p], s1);
                s2 = scalar_fnmadd(lrp, v2[p], s2);
                s3 = scalar_fnmadd(lrp, v3[p], s3);
            }
            col0[r] = s0;
            col1[r] = s1;
            col2[r] = s2;
            col3[r] = s3;
        }
    }

    /// @brief Applies a fused rank-bs subtraction update `col[r] -= sum_p l_cols[p][r] * v[p]` to a single column.
    /// @details Unrolls 4 vector widths per iteration in registers over `p = 0..bs-1` in ascending order, followed by
    /// single-vector and scalar tail passes matching rank_k_update_4cols_simd() lane-wise.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in,out] col Pointer to destination column base (index 0 corresponds to row 0).
    /// @param[in] l_cols Array of `bs` read-only panel column base pointers.
    /// @param[in] bs Panel width (number of rank-1 terms accumulated in ascending order).
    /// @param[in] v Array of `bs` scalar multipliers for `col`.
    /// @param[in] r_begin Inclusive starting row index to update.
    /// @param[in] r_end Exclusive ending row index to update.
    template <SimdBlasScalar T>
    inline void rank_k_update_1col_simd(
        T* col, const T* const* l_cols, std::size_t bs, const T* v, std::size_t r_begin, std::size_t r_end
    ) noexcept
    {
        if (bs == 0U || r_begin >= r_end)
        {
            return;
        }
        std::size_t r = r_begin;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            for (; r + 32U <= r_end; r += 32U)
            {
                __m512d a0 = _mm512_loadu_pd(col + r);
                __m512d a1 = _mm512_loadu_pd(col + r + 8U);
                __m512d a2 = _mm512_loadu_pd(col + r + 16U);
                __m512d a3 = _mm512_loadu_pd(col + r + 24U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const double* const lp = l_cols[p] + r;
                    const __m512d vb = _mm512_set1_pd(v[p]);
                    a0 = blas3_fnmadd_pd(_mm512_loadu_pd(lp), vb, a0);
                    a1 = blas3_fnmadd_pd(_mm512_loadu_pd(lp + 8U), vb, a1);
                    a2 = blas3_fnmadd_pd(_mm512_loadu_pd(lp + 16U), vb, a2);
                    a3 = blas3_fnmadd_pd(_mm512_loadu_pd(lp + 24U), vb, a3);
                }
                _mm512_storeu_pd(col + r, a0);
                _mm512_storeu_pd(col + r + 8U, a1);
                _mm512_storeu_pd(col + r + 16U, a2);
                _mm512_storeu_pd(col + r + 24U, a3);
            }
            for (; r + 8U <= r_end; r += 8U)
            {
                __m512d a0 = _mm512_loadu_pd(col + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    a0 = blas3_fnmadd_pd(_mm512_loadu_pd(l_cols[p] + r), _mm512_set1_pd(v[p]), a0);
                }
                _mm512_storeu_pd(col + r, a0);
            }
#elifdef __AVX2__
            for (; r + 16U <= r_end; r += 16U)
            {
                __m256d a0 = _mm256_loadu_pd(col + r);
                __m256d a1 = _mm256_loadu_pd(col + r + 4U);
                __m256d a2 = _mm256_loadu_pd(col + r + 8U);
                __m256d a3 = _mm256_loadu_pd(col + r + 12U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const double* const lp = l_cols[p] + r;
                    const __m256d vb = _mm256_set1_pd(v[p]);
                    a0 = blas3_fnmadd_pd(_mm256_loadu_pd(lp), vb, a0);
                    a1 = blas3_fnmadd_pd(_mm256_loadu_pd(lp + 4U), vb, a1);
                    a2 = blas3_fnmadd_pd(_mm256_loadu_pd(lp + 8U), vb, a2);
                    a3 = blas3_fnmadd_pd(_mm256_loadu_pd(lp + 12U), vb, a3);
                }
                _mm256_storeu_pd(col + r, a0);
                _mm256_storeu_pd(col + r + 4U, a1);
                _mm256_storeu_pd(col + r + 8U, a2);
                _mm256_storeu_pd(col + r + 12U, a3);
            }
            for (; r + 4U <= r_end; r += 4U)
            {
                __m256d a0 = _mm256_loadu_pd(col + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    a0 = blas3_fnmadd_pd(_mm256_loadu_pd(l_cols[p] + r), _mm256_set1_pd(v[p]), a0);
                }
                _mm256_storeu_pd(col + r, a0);
            }
#elif defined(_M_X64) || defined(__x86_64__)
            for (; r + 8U <= r_end; r += 8U)
            {
                __m128d a0 = _mm_loadu_pd(col + r);
                __m128d a1 = _mm_loadu_pd(col + r + 2U);
                __m128d a2 = _mm_loadu_pd(col + r + 4U);
                __m128d a3 = _mm_loadu_pd(col + r + 6U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const double* const lp = l_cols[p] + r;
                    const __m128d vb = _mm_set1_pd(v[p]);
                    a0 = _mm_sub_pd(a0, _mm_mul_pd(_mm_loadu_pd(lp), vb));
                    a1 = _mm_sub_pd(a1, _mm_mul_pd(_mm_loadu_pd(lp + 2U), vb));
                    a2 = _mm_sub_pd(a2, _mm_mul_pd(_mm_loadu_pd(lp + 4U), vb));
                    a3 = _mm_sub_pd(a3, _mm_mul_pd(_mm_loadu_pd(lp + 6U), vb));
                }
                _mm_storeu_pd(col + r, a0);
                _mm_storeu_pd(col + r + 2U, a1);
                _mm_storeu_pd(col + r + 4U, a2);
                _mm_storeu_pd(col + r + 6U, a3);
            }
            for (; r + 2U <= r_end; r += 2U)
            {
                __m128d a0 = _mm_loadu_pd(col + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    a0 = _mm_sub_pd(a0, _mm_mul_pd(_mm_loadu_pd(l_cols[p] + r), _mm_set1_pd(v[p])));
                }
                _mm_storeu_pd(col + r, a0);
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            for (; r + 64U <= r_end; r += 64U)
            {
                __m512 a0 = _mm512_loadu_ps(col + r);
                __m512 a1 = _mm512_loadu_ps(col + r + 16U);
                __m512 a2 = _mm512_loadu_ps(col + r + 32U);
                __m512 a3 = _mm512_loadu_ps(col + r + 48U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const float* const lp = l_cols[p] + r;
                    const __m512 vb = _mm512_set1_ps(v[p]);
                    a0 = blas3_fnmadd_ps(_mm512_loadu_ps(lp), vb, a0);
                    a1 = blas3_fnmadd_ps(_mm512_loadu_ps(lp + 16U), vb, a1);
                    a2 = blas3_fnmadd_ps(_mm512_loadu_ps(lp + 32U), vb, a2);
                    a3 = blas3_fnmadd_ps(_mm512_loadu_ps(lp + 48U), vb, a3);
                }
                _mm512_storeu_ps(col + r, a0);
                _mm512_storeu_ps(col + r + 16U, a1);
                _mm512_storeu_ps(col + r + 32U, a2);
                _mm512_storeu_ps(col + r + 48U, a3);
            }
            for (; r + 16U <= r_end; r += 16U)
            {
                __m512 a0 = _mm512_loadu_ps(col + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    a0 = blas3_fnmadd_ps(_mm512_loadu_ps(l_cols[p] + r), _mm512_set1_ps(v[p]), a0);
                }
                _mm512_storeu_ps(col + r, a0);
            }
#elifdef __AVX2__
            for (; r + 32U <= r_end; r += 32U)
            {
                __m256 a0 = _mm256_loadu_ps(col + r);
                __m256 a1 = _mm256_loadu_ps(col + r + 8U);
                __m256 a2 = _mm256_loadu_ps(col + r + 16U);
                __m256 a3 = _mm256_loadu_ps(col + r + 24U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const float* const lp = l_cols[p] + r;
                    const __m256 vb = _mm256_set1_ps(v[p]);
                    a0 = blas3_fnmadd_ps(_mm256_loadu_ps(lp), vb, a0);
                    a1 = blas3_fnmadd_ps(_mm256_loadu_ps(lp + 8U), vb, a1);
                    a2 = blas3_fnmadd_ps(_mm256_loadu_ps(lp + 16U), vb, a2);
                    a3 = blas3_fnmadd_ps(_mm256_loadu_ps(lp + 24U), vb, a3);
                }
                _mm256_storeu_ps(col + r, a0);
                _mm256_storeu_ps(col + r + 8U, a1);
                _mm256_storeu_ps(col + r + 16U, a2);
                _mm256_storeu_ps(col + r + 24U, a3);
            }
            for (; r + 8U <= r_end; r += 8U)
            {
                __m256 a0 = _mm256_loadu_ps(col + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    a0 = blas3_fnmadd_ps(_mm256_loadu_ps(l_cols[p] + r), _mm256_set1_ps(v[p]), a0);
                }
                _mm256_storeu_ps(col + r, a0);
            }
#elif defined(_M_X64) || defined(__x86_64__)
            for (; r + 16U <= r_end; r += 16U)
            {
                __m128 a0 = _mm_loadu_ps(col + r);
                __m128 a1 = _mm_loadu_ps(col + r + 4U);
                __m128 a2 = _mm_loadu_ps(col + r + 8U);
                __m128 a3 = _mm_loadu_ps(col + r + 12U);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const float* const lp = l_cols[p] + r;
                    const __m128 vb = _mm_set1_ps(v[p]);
                    a0 = _mm_sub_ps(a0, _mm_mul_ps(_mm_loadu_ps(lp), vb));
                    a1 = _mm_sub_ps(a1, _mm_mul_ps(_mm_loadu_ps(lp + 4U), vb));
                    a2 = _mm_sub_ps(a2, _mm_mul_ps(_mm_loadu_ps(lp + 8U), vb));
                    a3 = _mm_sub_ps(a3, _mm_mul_ps(_mm_loadu_ps(lp + 12U), vb));
                }
                _mm_storeu_ps(col + r, a0);
                _mm_storeu_ps(col + r + 4U, a1);
                _mm_storeu_ps(col + r + 8U, a2);
                _mm_storeu_ps(col + r + 12U, a3);
            }
            for (; r + 4U <= r_end; r += 4U)
            {
                __m128 a0 = _mm_loadu_ps(col + r);
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    a0 = _mm_sub_ps(a0, _mm_mul_ps(_mm_loadu_ps(l_cols[p] + r), _mm_set1_ps(v[p])));
                }
                _mm_storeu_ps(col + r, a0);
            }
#endif
        }
        for (; r < r_end; ++r)
        {
            T s = col[r];
            for (std::size_t p = 0U; p < bs; ++p)
            {
                s = scalar_fnmadd(l_cols[p][r], v[p], s);
            }
            col[r] = s;
        }
    }

    /// @brief Scales a contiguous vector slice `ptr[0..len)` in place by `factor` using 4-vector unrolling.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in,out] ptr Contiguous destination slice updated in place (`ptr[i] *= factor`).
    /// @param[in] len Number of contiguous elements to scale.
    /// @param[in] factor Scalar multiplier applied to every element.
    template <SimdBlasScalar T> inline void scale_in_place_simd(T* ptr, std::size_t len, T factor) noexcept
    {
        std::size_t i = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512d vf = _mm512_set1_pd(factor);
            for (; i + 32U <= len; i += 32U)
            {
                _mm512_storeu_pd(ptr + i, _mm512_mul_pd(_mm512_loadu_pd(ptr + i), vf));
                _mm512_storeu_pd(ptr + i + 8U, _mm512_mul_pd(_mm512_loadu_pd(ptr + i + 8U), vf));
                _mm512_storeu_pd(ptr + i + 16U, _mm512_mul_pd(_mm512_loadu_pd(ptr + i + 16U), vf));
                _mm512_storeu_pd(ptr + i + 24U, _mm512_mul_pd(_mm512_loadu_pd(ptr + i + 24U), vf));
            }
            for (; i + 8U <= len; i += 8U)
            {
                _mm512_storeu_pd(ptr + i, _mm512_mul_pd(_mm512_loadu_pd(ptr + i), vf));
            }
#elifdef __AVX2__
            const __m256d vf = _mm256_set1_pd(factor);
            for (; i + 16U <= len; i += 16U)
            {
                _mm256_storeu_pd(ptr + i, _mm256_mul_pd(_mm256_loadu_pd(ptr + i), vf));
                _mm256_storeu_pd(ptr + i + 4U, _mm256_mul_pd(_mm256_loadu_pd(ptr + i + 4U), vf));
                _mm256_storeu_pd(ptr + i + 8U, _mm256_mul_pd(_mm256_loadu_pd(ptr + i + 8U), vf));
                _mm256_storeu_pd(ptr + i + 12U, _mm256_mul_pd(_mm256_loadu_pd(ptr + i + 12U), vf));
            }
            for (; i + 4U <= len; i += 4U)
            {
                _mm256_storeu_pd(ptr + i, _mm256_mul_pd(_mm256_loadu_pd(ptr + i), vf));
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128d vf = _mm_set1_pd(factor);
            for (; i + 8U <= len; i += 8U)
            {
                _mm_storeu_pd(ptr + i, _mm_mul_pd(_mm_loadu_pd(ptr + i), vf));
                _mm_storeu_pd(ptr + i + 2U, _mm_mul_pd(_mm_loadu_pd(ptr + i + 2U), vf));
                _mm_storeu_pd(ptr + i + 4U, _mm_mul_pd(_mm_loadu_pd(ptr + i + 4U), vf));
                _mm_storeu_pd(ptr + i + 6U, _mm_mul_pd(_mm_loadu_pd(ptr + i + 6U), vf));
            }
            for (; i + 2U <= len; i += 2U)
            {
                _mm_storeu_pd(ptr + i, _mm_mul_pd(_mm_loadu_pd(ptr + i), vf));
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512 vf = _mm512_set1_ps(factor);
            for (; i + 64U <= len; i += 64U)
            {
                _mm512_storeu_ps(ptr + i, _mm512_mul_ps(_mm512_loadu_ps(ptr + i), vf));
                _mm512_storeu_ps(ptr + i + 16U, _mm512_mul_ps(_mm512_loadu_ps(ptr + i + 16U), vf));
                _mm512_storeu_ps(ptr + i + 32U, _mm512_mul_ps(_mm512_loadu_ps(ptr + i + 32U), vf));
                _mm512_storeu_ps(ptr + i + 48U, _mm512_mul_ps(_mm512_loadu_ps(ptr + i + 48U), vf));
            }
            for (; i + 16U <= len; i += 16U)
            {
                _mm512_storeu_ps(ptr + i, _mm512_mul_ps(_mm512_loadu_ps(ptr + i), vf));
            }
#elifdef __AVX2__
            const __m256 vf = _mm256_set1_ps(factor);
            for (; i + 32U <= len; i += 32U)
            {
                _mm256_storeu_ps(ptr + i, _mm256_mul_ps(_mm256_loadu_ps(ptr + i), vf));
                _mm256_storeu_ps(ptr + i + 8U, _mm256_mul_ps(_mm256_loadu_ps(ptr + i + 8U), vf));
                _mm256_storeu_ps(ptr + i + 16U, _mm256_mul_ps(_mm256_loadu_ps(ptr + i + 16U), vf));
                _mm256_storeu_ps(ptr + i + 24U, _mm256_mul_ps(_mm256_loadu_ps(ptr + i + 24U), vf));
            }
            for (; i + 8U <= len; i += 8U)
            {
                _mm256_storeu_ps(ptr + i, _mm256_mul_ps(_mm256_loadu_ps(ptr + i), vf));
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128 vf = _mm_set1_ps(factor);
            for (; i + 16U <= len; i += 16U)
            {
                _mm_storeu_ps(ptr + i, _mm_mul_ps(_mm_loadu_ps(ptr + i), vf));
                _mm_storeu_ps(ptr + i + 4U, _mm_mul_ps(_mm_loadu_ps(ptr + i + 4U), vf));
                _mm_storeu_ps(ptr + i + 8U, _mm_mul_ps(_mm_loadu_ps(ptr + i + 8U), vf));
                _mm_storeu_ps(ptr + i + 12U, _mm_mul_ps(_mm_loadu_ps(ptr + i + 12U), vf));
            }
            for (; i + 4U <= len; i += 4U)
            {
                _mm_storeu_ps(ptr + i, _mm_mul_ps(_mm_loadu_ps(ptr + i), vf));
            }
#endif
        }
        for (; i < len; ++i)
        {
            ptr[i] *= factor;
        }
    }

    /// @brief Assigns `dst[r] = -(src[r] * factor)` over contiguous range `[r_begin, r_end)` with 4-vector unrolling.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[out] dst Destination column base pointer.
    /// @param[in] src Source column base pointer.
    /// @param[in] factor Positive/arbitrary scalar multiplier negated before assignment.
    /// @param[in] r_begin Inclusive starting index.
    /// @param[in] r_end Exclusive ending index.
    template <SimdBlasScalar T>
    inline void scale_neg_set_simd(T* dst, const T* src, T factor, std::size_t r_begin, std::size_t r_end) noexcept
    {
        const T neg_factor = -factor;
        std::size_t i = r_begin;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512d vf = _mm512_set1_pd(neg_factor);
            for (; i + 32U <= r_end; i += 32U)
            {
                _mm512_storeu_pd(dst + i, _mm512_mul_pd(_mm512_loadu_pd(src + i), vf));
                _mm512_storeu_pd(dst + i + 8U, _mm512_mul_pd(_mm512_loadu_pd(src + i + 8U), vf));
                _mm512_storeu_pd(dst + i + 16U, _mm512_mul_pd(_mm512_loadu_pd(src + i + 16U), vf));
                _mm512_storeu_pd(dst + i + 24U, _mm512_mul_pd(_mm512_loadu_pd(src + i + 24U), vf));
            }
            for (; i + 8U <= r_end; i += 8U)
            {
                _mm512_storeu_pd(dst + i, _mm512_mul_pd(_mm512_loadu_pd(src + i), vf));
            }
#elifdef __AVX2__
            const __m256d vf = _mm256_set1_pd(neg_factor);
            for (; i + 16U <= r_end; i += 16U)
            {
                _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(src + i), vf));
                _mm256_storeu_pd(dst + i + 4U, _mm256_mul_pd(_mm256_loadu_pd(src + i + 4U), vf));
                _mm256_storeu_pd(dst + i + 8U, _mm256_mul_pd(_mm256_loadu_pd(src + i + 8U), vf));
                _mm256_storeu_pd(dst + i + 12U, _mm256_mul_pd(_mm256_loadu_pd(src + i + 12U), vf));
            }
            for (; i + 4U <= r_end; i += 4U)
            {
                _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(src + i), vf));
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128d vf = _mm_set1_pd(neg_factor);
            for (; i + 8U <= r_end; i += 8U)
            {
                _mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(src + i), vf));
                _mm_storeu_pd(dst + i + 2U, _mm_mul_pd(_mm_loadu_pd(src + i + 2U), vf));
                _mm_storeu_pd(dst + i + 4U, _mm_mul_pd(_mm_loadu_pd(src + i + 4U), vf));
                _mm_storeu_pd(dst + i + 6U, _mm_mul_pd(_mm_loadu_pd(src + i + 6U), vf));
            }
            for (; i + 2U <= r_end; i += 2U)
            {
                _mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(src + i), vf));
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512 vf = _mm512_set1_ps(neg_factor);
            for (; i + 64U <= r_end; i += 64U)
            {
                _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(src + i), vf));
                _mm512_storeu_ps(dst + i + 16U, _mm512_mul_ps(_mm512_loadu_ps(src + i + 16U), vf));
                _mm512_storeu_ps(dst + i + 32U, _mm512_mul_ps(_mm512_loadu_ps(src + i + 32U), vf));
                _mm512_storeu_ps(dst + i + 48U, _mm512_mul_ps(_mm512_loadu_ps(src + i + 48U), vf));
            }
            for (; i + 16U <= r_end; i += 16U)
            {
                _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(src + i), vf));
            }
#elifdef __AVX2__
            const __m256 vf = _mm256_set1_ps(neg_factor);
            for (; i + 32U <= r_end; i += 32U)
            {
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vf));
                _mm256_storeu_ps(dst + i + 8U, _mm256_mul_ps(_mm256_loadu_ps(src + i + 8U), vf));
                _mm256_storeu_ps(dst + i + 16U, _mm256_mul_ps(_mm256_loadu_ps(src + i + 16U), vf));
                _mm256_storeu_ps(dst + i + 24U, _mm256_mul_ps(_mm256_loadu_ps(src + i + 24U), vf));
            }
            for (; i + 8U <= r_end; i += 8U)
            {
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vf));
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128 vf = _mm_set1_ps(neg_factor);
            for (; i + 16U <= r_end; i += 16U)
            {
                _mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(src + i), vf));
                _mm_storeu_ps(dst + i + 4U, _mm_mul_ps(_mm_loadu_ps(src + i + 4U), vf));
                _mm_storeu_ps(dst + i + 8U, _mm_mul_ps(_mm_loadu_ps(src + i + 8U), vf));
                _mm_storeu_ps(dst + i + 12U, _mm_mul_ps(_mm_loadu_ps(src + i + 12U), vf));
            }
            for (; i + 4U <= r_end; i += 4U)
            {
                _mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(src + i), vf));
            }
#endif
        }
        for (; i < r_end; ++i)
        {
            dst[i] = src[i] * neg_factor;
        }
    }

    /// @brief Computes a 4x2 block of dot products `out[r][c] = dot(a_r, b_c)` in a single pass.
    /// @details Uses 8 vector accumulators with single-vector strides matching dot_4x1_simd() and dot_1x1_simd(),
    /// guaranteeing zero register spills under AVX2 (11 live YMM registers) and exact bitwise equality between
    /// 4x2, 4x1, and 1x1 dot products on identical inputs.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] a0 First left-hand vector.
    /// @param[in] a1 Second left-hand vector.
    /// @param[in] a2 Third left-hand vector.
    /// @param[in] a3 Fourth left-hand vector.
    /// @param[in] b0 First right-hand vector.
    /// @param[in] b1 Second right-hand vector.
    /// @param[in] len Number of elements in each vector.
    /// @param[out] out Output array populated with `out[row][col] = dot(a_row, b_col)`.
    template <SimdBlasScalar T>
    inline void dot_4x2_simd(
        const T* a0,
        const T* a1,
        const T* a2,
        const T* a3,
        const T* b0,
        const T* b1,
        std::size_t len,
        std::array<std::array<T, 2>, 4>& out
    ) noexcept
    {
        std::size_t k = 0U;
        T s00{0};
        T s01{0};
        T s10{0};
        T s11{0};
        T s20{0};
        T s21{0};
        T s30{0};
        T s31{0};

        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v00 = _mm512_setzero_pd();
            __m512d v01 = _mm512_setzero_pd();
            __m512d v10 = _mm512_setzero_pd();
            __m512d v11 = _mm512_setzero_pd();
            __m512d v20 = _mm512_setzero_pd();
            __m512d v21 = _mm512_setzero_pd();
            __m512d v30 = _mm512_setzero_pd();
            __m512d v31 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d vb0 = _mm512_loadu_pd(b0 + k);
                const __m512d vb1 = _mm512_loadu_pd(b1 + k);
                const __m512d va0 = _mm512_loadu_pd(a0 + k);
                v00 = blas3_fmadd_pd(va0, vb0, v00);
                v01 = blas3_fmadd_pd(va0, vb1, v01);
                const __m512d va1 = _mm512_loadu_pd(a1 + k);
                v10 = blas3_fmadd_pd(va1, vb0, v10);
                v11 = blas3_fmadd_pd(va1, vb1, v11);
                const __m512d va2 = _mm512_loadu_pd(a2 + k);
                v20 = blas3_fmadd_pd(va2, vb0, v20);
                v21 = blas3_fmadd_pd(va2, vb1, v21);
                const __m512d va3 = _mm512_loadu_pd(a3 + k);
                v30 = blas3_fmadd_pd(va3, vb0, v30);
                v31 = blas3_fmadd_pd(va3, vb1, v31);
            }
            s00 = hsum_zmm_pd(v00);
            s01 = hsum_zmm_pd(v01);
            s10 = hsum_zmm_pd(v10);
            s11 = hsum_zmm_pd(v11);
            s20 = hsum_zmm_pd(v20);
            s21 = hsum_zmm_pd(v21);
            s30 = hsum_zmm_pd(v30);
            s31 = hsum_zmm_pd(v31);
#elifdef __AVX2__
            __m256d v00 = _mm256_setzero_pd();
            __m256d v01 = _mm256_setzero_pd();
            __m256d v10 = _mm256_setzero_pd();
            __m256d v11 = _mm256_setzero_pd();
            __m256d v20 = _mm256_setzero_pd();
            __m256d v21 = _mm256_setzero_pd();
            __m256d v30 = _mm256_setzero_pd();
            __m256d v31 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d vb0 = _mm256_loadu_pd(b0 + k);
                const __m256d vb1 = _mm256_loadu_pd(b1 + k);
                const __m256d va0 = _mm256_loadu_pd(a0 + k);
                v00 = blas3_fmadd_pd(va0, vb0, v00);
                v01 = blas3_fmadd_pd(va0, vb1, v01);
                const __m256d va1 = _mm256_loadu_pd(a1 + k);
                v10 = blas3_fmadd_pd(va1, vb0, v10);
                v11 = blas3_fmadd_pd(va1, vb1, v11);
                const __m256d va2 = _mm256_loadu_pd(a2 + k);
                v20 = blas3_fmadd_pd(va2, vb0, v20);
                v21 = blas3_fmadd_pd(va2, vb1, v21);
                const __m256d va3 = _mm256_loadu_pd(a3 + k);
                v30 = blas3_fmadd_pd(va3, vb0, v30);
                v31 = blas3_fmadd_pd(va3, vb1, v31);
            }
            s00 = hsum_ymm_pd(v00);
            s01 = hsum_ymm_pd(v01);
            s10 = hsum_ymm_pd(v10);
            s11 = hsum_ymm_pd(v11);
            s20 = hsum_ymm_pd(v20);
            s21 = hsum_ymm_pd(v21);
            s30 = hsum_ymm_pd(v30);
            s31 = hsum_ymm_pd(v31);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v00 = _mm_setzero_pd();
            __m128d v01 = _mm_setzero_pd();
            __m128d v10 = _mm_setzero_pd();
            __m128d v11 = _mm_setzero_pd();
            __m128d v20 = _mm_setzero_pd();
            __m128d v21 = _mm_setzero_pd();
            __m128d v30 = _mm_setzero_pd();
            __m128d v31 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vb0 = _mm_loadu_pd(b0 + k);
                const __m128d vb1 = _mm_loadu_pd(b1 + k);
                const __m128d va0 = _mm_loadu_pd(a0 + k);
                v00 = _mm_add_pd(v00, _mm_mul_pd(va0, vb0));
                v01 = _mm_add_pd(v01, _mm_mul_pd(va0, vb1));
                const __m128d va1 = _mm_loadu_pd(a1 + k);
                v10 = _mm_add_pd(v10, _mm_mul_pd(va1, vb0));
                v11 = _mm_add_pd(v11, _mm_mul_pd(va1, vb1));
                const __m128d va2 = _mm_loadu_pd(a2 + k);
                v20 = _mm_add_pd(v20, _mm_mul_pd(va2, vb0));
                v21 = _mm_add_pd(v21, _mm_mul_pd(va2, vb1));
                const __m128d va3 = _mm_loadu_pd(a3 + k);
                v30 = _mm_add_pd(v30, _mm_mul_pd(va3, vb0));
                v31 = _mm_add_pd(v31, _mm_mul_pd(va3, vb1));
            }
            s00 = hsum_xmm_pd(v00);
            s01 = hsum_xmm_pd(v01);
            s10 = hsum_xmm_pd(v10);
            s11 = hsum_xmm_pd(v11);
            s20 = hsum_xmm_pd(v20);
            s21 = hsum_xmm_pd(v21);
            s30 = hsum_xmm_pd(v30);
            s31 = hsum_xmm_pd(v31);
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512 v00 = _mm512_setzero_ps();
            __m512 v01 = _mm512_setzero_ps();
            __m512 v10 = _mm512_setzero_ps();
            __m512 v11 = _mm512_setzero_ps();
            __m512 v20 = _mm512_setzero_ps();
            __m512 v21 = _mm512_setzero_ps();
            __m512 v30 = _mm512_setzero_ps();
            __m512 v31 = _mm512_setzero_ps();
            for (; k + 16U <= len; k += 16U)
            {
                const __m512 vb0 = _mm512_loadu_ps(b0 + k);
                const __m512 vb1 = _mm512_loadu_ps(b1 + k);
                const __m512 va0 = _mm512_loadu_ps(a0 + k);
                v00 = blas3_fmadd_ps(va0, vb0, v00);
                v01 = blas3_fmadd_ps(va0, vb1, v01);
                const __m512 va1 = _mm512_loadu_ps(a1 + k);
                v10 = blas3_fmadd_ps(va1, vb0, v10);
                v11 = blas3_fmadd_ps(va1, vb1, v11);
                const __m512 va2 = _mm512_loadu_ps(a2 + k);
                v20 = blas3_fmadd_ps(va2, vb0, v20);
                v21 = blas3_fmadd_ps(va2, vb1, v21);
                const __m512 va3 = _mm512_loadu_ps(a3 + k);
                v30 = blas3_fmadd_ps(va3, vb0, v30);
                v31 = blas3_fmadd_ps(va3, vb1, v31);
            }
            s00 = hsum_zmm_ps(v00);
            s01 = hsum_zmm_ps(v01);
            s10 = hsum_zmm_ps(v10);
            s11 = hsum_zmm_ps(v11);
            s20 = hsum_zmm_ps(v20);
            s21 = hsum_zmm_ps(v21);
            s30 = hsum_zmm_ps(v30);
            s31 = hsum_zmm_ps(v31);
#elifdef __AVX2__
            __m256 v00 = _mm256_setzero_ps();
            __m256 v01 = _mm256_setzero_ps();
            __m256 v10 = _mm256_setzero_ps();
            __m256 v11 = _mm256_setzero_ps();
            __m256 v20 = _mm256_setzero_ps();
            __m256 v21 = _mm256_setzero_ps();
            __m256 v30 = _mm256_setzero_ps();
            __m256 v31 = _mm256_setzero_ps();
            for (; k + 8U <= len; k += 8U)
            {
                const __m256 vb0 = _mm256_loadu_ps(b0 + k);
                const __m256 vb1 = _mm256_loadu_ps(b1 + k);
                const __m256 va0 = _mm256_loadu_ps(a0 + k);
                v00 = blas3_fmadd_ps(va0, vb0, v00);
                v01 = blas3_fmadd_ps(va0, vb1, v01);
                const __m256 va1 = _mm256_loadu_ps(a1 + k);
                v10 = blas3_fmadd_ps(va1, vb0, v10);
                v11 = blas3_fmadd_ps(va1, vb1, v11);
                const __m256 va2 = _mm256_loadu_ps(a2 + k);
                v20 = blas3_fmadd_ps(va2, vb0, v20);
                v21 = blas3_fmadd_ps(va2, vb1, v21);
                const __m256 va3 = _mm256_loadu_ps(a3 + k);
                v30 = blas3_fmadd_ps(va3, vb0, v30);
                v31 = blas3_fmadd_ps(va3, vb1, v31);
            }
            s00 = hsum_ymm_ps(v00);
            s01 = hsum_ymm_ps(v01);
            s10 = hsum_ymm_ps(v10);
            s11 = hsum_ymm_ps(v11);
            s20 = hsum_ymm_ps(v20);
            s21 = hsum_ymm_ps(v21);
            s30 = hsum_ymm_ps(v30);
            s31 = hsum_ymm_ps(v31);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128 v00 = _mm_setzero_ps();
            __m128 v01 = _mm_setzero_ps();
            __m128 v10 = _mm_setzero_ps();
            __m128 v11 = _mm_setzero_ps();
            __m128 v20 = _mm_setzero_ps();
            __m128 v21 = _mm_setzero_ps();
            __m128 v30 = _mm_setzero_ps();
            __m128 v31 = _mm_setzero_ps();
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 vb0 = _mm_loadu_ps(b0 + k);
                const __m128 vb1 = _mm_loadu_ps(b1 + k);
                const __m128 va0 = _mm_loadu_ps(a0 + k);
                v00 = _mm_add_ps(v00, _mm_mul_ps(va0, vb0));
                v01 = _mm_add_ps(v01, _mm_mul_ps(va0, vb1));
                const __m128 va1 = _mm_loadu_ps(a1 + k);
                v10 = _mm_add_ps(v10, _mm_mul_ps(va1, vb0));
                v11 = _mm_add_ps(v11, _mm_mul_ps(va1, vb1));
                const __m128 va2 = _mm_loadu_ps(a2 + k);
                v20 = _mm_add_ps(v20, _mm_mul_ps(va2, vb0));
                v21 = _mm_add_ps(v21, _mm_mul_ps(va2, vb1));
                const __m128 va3 = _mm_loadu_ps(a3 + k);
                v30 = _mm_add_ps(v30, _mm_mul_ps(va3, vb0));
                v31 = _mm_add_ps(v31, _mm_mul_ps(va3, vb1));
            }
            s00 = hsum_xmm_ps(v00);
            s01 = hsum_xmm_ps(v01);
            s10 = hsum_xmm_ps(v10);
            s11 = hsum_xmm_ps(v11);
            s20 = hsum_xmm_ps(v20);
            s21 = hsum_xmm_ps(v21);
            s30 = hsum_xmm_ps(v30);
            s31 = hsum_xmm_ps(v31);
#endif
        }
        for (; k < len; ++k)
        {
            const T b0k = b0[k];
            const T b1k = b1[k];
            const T a0k = a0[k];
            const T a1k = a1[k];
            const T a2k = a2[k];
            const T a3k = a3[k];
            s00 = scalar_fmadd(a0k, b0k, s00);
            s01 = scalar_fmadd(a0k, b1k, s01);
            s10 = scalar_fmadd(a1k, b0k, s10);
            s11 = scalar_fmadd(a1k, b1k, s11);
            s20 = scalar_fmadd(a2k, b0k, s20);
            s21 = scalar_fmadd(a2k, b1k, s21);
            s30 = scalar_fmadd(a3k, b0k, s30);
            s31 = scalar_fmadd(a3k, b1k, s31);
        }
        out[0] = {s00, s01};
        out[1] = {s10, s11};
        out[2] = {s20, s21};
        out[3] = {s30, s31};
    }

    /// @brief Computes 4 single-column dot products `out[r] = dot(a_r, b0)` matching dot_4x2_simd() bitwise.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] a0 First left-hand vector.
    /// @param[in] a1 Second left-hand vector.
    /// @param[in] a2 Third left-hand vector.
    /// @param[in] a3 Fourth left-hand vector.
    /// @param[in] b0 Right-hand vector.
    /// @param[in] len Number of elements in each vector.
    /// @param[out] out Output array populated with `out[row] = dot(a_row, b0)`.
    template <SimdBlasScalar T>
    inline void dot_4x1_simd(
        const T* a0, const T* a1, const T* a2, const T* a3, const T* b0, std::size_t len, std::array<T, 4>& out
    ) noexcept
    {
        std::size_t k = 0U;
        T s0{0};
        T s1{0};
        T s2{0};
        T s3{0};

        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v0 = _mm512_setzero_pd();
            __m512d v1 = _mm512_setzero_pd();
            __m512d v2 = _mm512_setzero_pd();
            __m512d v3 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d vb = _mm512_loadu_pd(b0 + k);
                v0 = blas3_fmadd_pd(_mm512_loadu_pd(a0 + k), vb, v0);
                v1 = blas3_fmadd_pd(_mm512_loadu_pd(a1 + k), vb, v1);
                v2 = blas3_fmadd_pd(_mm512_loadu_pd(a2 + k), vb, v2);
                v3 = blas3_fmadd_pd(_mm512_loadu_pd(a3 + k), vb, v3);
            }
            s0 = hsum_zmm_pd(v0);
            s1 = hsum_zmm_pd(v1);
            s2 = hsum_zmm_pd(v2);
            s3 = hsum_zmm_pd(v3);
#elifdef __AVX2__
            __m256d v0 = _mm256_setzero_pd();
            __m256d v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd();
            __m256d v3 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d vb = _mm256_loadu_pd(b0 + k);
                v0 = blas3_fmadd_pd(_mm256_loadu_pd(a0 + k), vb, v0);
                v1 = blas3_fmadd_pd(_mm256_loadu_pd(a1 + k), vb, v1);
                v2 = blas3_fmadd_pd(_mm256_loadu_pd(a2 + k), vb, v2);
                v3 = blas3_fmadd_pd(_mm256_loadu_pd(a3 + k), vb, v3);
            }
            s0 = hsum_ymm_pd(v0);
            s1 = hsum_ymm_pd(v1);
            s2 = hsum_ymm_pd(v2);
            s3 = hsum_ymm_pd(v3);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v0 = _mm_setzero_pd();
            __m128d v1 = _mm_setzero_pd();
            __m128d v2 = _mm_setzero_pd();
            __m128d v3 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vb = _mm_loadu_pd(b0 + k);
                v0 = _mm_add_pd(v0, _mm_mul_pd(_mm_loadu_pd(a0 + k), vb));
                v1 = _mm_add_pd(v1, _mm_mul_pd(_mm_loadu_pd(a1 + k), vb));
                v2 = _mm_add_pd(v2, _mm_mul_pd(_mm_loadu_pd(a2 + k), vb));
                v3 = _mm_add_pd(v3, _mm_mul_pd(_mm_loadu_pd(a3 + k), vb));
            }
            s0 = hsum_xmm_pd(v0);
            s1 = hsum_xmm_pd(v1);
            s2 = hsum_xmm_pd(v2);
            s3 = hsum_xmm_pd(v3);
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512 v0 = _mm512_setzero_ps();
            __m512 v1 = _mm512_setzero_ps();
            __m512 v2 = _mm512_setzero_ps();
            __m512 v3 = _mm512_setzero_ps();
            for (; k + 16U <= len; k += 16U)
            {
                const __m512 vb = _mm512_loadu_ps(b0 + k);
                v0 = blas3_fmadd_ps(_mm512_loadu_ps(a0 + k), vb, v0);
                v1 = blas3_fmadd_ps(_mm512_loadu_ps(a1 + k), vb, v1);
                v2 = blas3_fmadd_ps(_mm512_loadu_ps(a2 + k), vb, v2);
                v3 = blas3_fmadd_ps(_mm512_loadu_ps(a3 + k), vb, v3);
            }
            s0 = hsum_zmm_ps(v0);
            s1 = hsum_zmm_ps(v1);
            s2 = hsum_zmm_ps(v2);
            s3 = hsum_zmm_ps(v3);
#elifdef __AVX2__
            __m256 v0 = _mm256_setzero_ps();
            __m256 v1 = _mm256_setzero_ps();
            __m256 v2 = _mm256_setzero_ps();
            __m256 v3 = _mm256_setzero_ps();
            for (; k + 8U <= len; k += 8U)
            {
                const __m256 vb = _mm256_loadu_ps(b0 + k);
                v0 = blas3_fmadd_ps(_mm256_loadu_ps(a0 + k), vb, v0);
                v1 = blas3_fmadd_ps(_mm256_loadu_ps(a1 + k), vb, v1);
                v2 = blas3_fmadd_ps(_mm256_loadu_ps(a2 + k), vb, v2);
                v3 = blas3_fmadd_ps(_mm256_loadu_ps(a3 + k), vb, v3);
            }
            s0 = hsum_ymm_ps(v0);
            s1 = hsum_ymm_ps(v1);
            s2 = hsum_ymm_ps(v2);
            s3 = hsum_ymm_ps(v3);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128 v0 = _mm_setzero_ps();
            __m128 v1 = _mm_setzero_ps();
            __m128 v2 = _mm_setzero_ps();
            __m128 v3 = _mm_setzero_ps();
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 vb = _mm_loadu_ps(b0 + k);
                v0 = _mm_add_ps(v0, _mm_mul_ps(_mm_loadu_ps(a0 + k), vb));
                v1 = _mm_add_ps(v1, _mm_mul_ps(_mm_loadu_ps(a1 + k), vb));
                v2 = _mm_add_ps(v2, _mm_mul_ps(_mm_loadu_ps(a2 + k), vb));
                v3 = _mm_add_ps(v3, _mm_mul_ps(_mm_loadu_ps(a3 + k), vb));
            }
            s0 = hsum_xmm_ps(v0);
            s1 = hsum_xmm_ps(v1);
            s2 = hsum_xmm_ps(v2);
            s3 = hsum_xmm_ps(v3);
#endif
        }
        for (; k < len; ++k)
        {
            const T b0k = b0[k];
            s0 = scalar_fmadd(a0[k], b0k, s0);
            s1 = scalar_fmadd(a1[k], b0k, s1);
            s2 = scalar_fmadd(a2[k], b0k, s2);
            s3 = scalar_fmadd(a3[k], b0k, s3);
        }
        out = {s0, s1, s2, s3};
    }

    /// @brief Computes single dot product `dot(a0, b0)` matching dot_4x2_simd() and dot_4x1_simd() bitwise.
    /// @tparam T Floating-point scalar type (`double` or `float`).
    /// @param[in] a0 Left-hand vector.
    /// @param[in] b0 Right-hand vector.
    /// @param[in] len Number of elements in each vector.
    /// @return Scalar inner product accumulated in fixed SIMD width and pairwise horizontal lane order.
    template <SimdBlasScalar T> [[nodiscard]] inline T dot_1x1_simd(const T* a0, const T* b0, std::size_t len) noexcept
    {
        std::size_t k = 0U;
        T s{0};

        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v0 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                v0 = blas3_fmadd_pd(_mm512_loadu_pd(a0 + k), _mm512_loadu_pd(b0 + k), v0);
            }
            s = hsum_zmm_pd(v0);
#elifdef __AVX2__
            __m256d v0 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                v0 = blas3_fmadd_pd(_mm256_loadu_pd(a0 + k), _mm256_loadu_pd(b0 + k), v0);
            }
            s = hsum_ymm_pd(v0);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v0 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                v0 = _mm_add_pd(v0, _mm_mul_pd(_mm_loadu_pd(a0 + k), _mm_loadu_pd(b0 + k)));
            }
            s = hsum_xmm_pd(v0);
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512 v0 = _mm512_setzero_ps();
            for (; k + 16U <= len; k += 16U)
            {
                v0 = blas3_fmadd_ps(_mm512_loadu_ps(a0 + k), _mm512_loadu_ps(b0 + k), v0);
            }
            s = hsum_zmm_ps(v0);
#elifdef __AVX2__
            __m256 v0 = _mm256_setzero_ps();
            for (; k + 8U <= len; k += 8U)
            {
                v0 = blas3_fmadd_ps(_mm256_loadu_ps(a0 + k), _mm256_loadu_ps(b0 + k), v0);
            }
            s = hsum_ymm_ps(v0);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128 v0 = _mm_setzero_ps();
            for (; k + 4U <= len; k += 4U)
            {
                v0 = _mm_add_ps(v0, _mm_mul_ps(_mm_loadu_ps(a0 + k), _mm_loadu_ps(b0 + k)));
            }
            s = hsum_xmm_ps(v0);
#endif
        }
        for (; k < len; ++k)
        {
            s = scalar_fmadd(a0[k], b0[k], s);
        }
        return s;
    }

    /// @brief Computes dot product `sum_k static_cast<Refine>(a[k]) * b[k]` across Storage and Refine vectors.
    /// @tparam Storage Storage scalar type (`double` or `float`).
    /// @tparam Refine Refinement/output scalar type (`double` or `float`).
    /// @param[in] a Left-hand vector in Storage precision.
    /// @param[in] b Right-hand vector in Refine precision.
    /// @param[in] len Number of elements in each vector.
    /// @return Inner product in Refine precision.
    template <SimdBlasScalar Storage, SimdBlasScalar Refine>
    [[nodiscard]] inline Refine dot_mixed_simd(const Storage* a, const Refine* b, std::size_t len) noexcept
    {
        if constexpr (std::same_as<Storage, Refine>)
        {
            return dot_1x1_simd<Refine>(a, b, len);
        }
        else if constexpr (std::same_as<Storage, float> && std::same_as<Refine, double>)
        {
            std::size_t k = 0U;
            double s = 0.0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v0 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d va = _mm512_cvtps_pd(_mm256_loadu_ps(a + k));
                const __m512d vb = _mm512_loadu_pd(b + k);
                v0 = blas3_fmadd_pd(va, vb, v0);
            }
            s = hsum_zmm_pd(v0);
#elifdef __AVX2__
            __m256d v0 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d va = _mm256_cvtps_pd(_mm_loadu_ps(a + k));
                const __m256d vb = _mm256_loadu_pd(b + k);
                v0 = blas3_fmadd_pd(va, vb, v0);
            }
            s = hsum_ymm_pd(v0);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v0 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                const __m128 fa = _mm_set_ps(0.0F, 0.0F, a[k + 1U], a[k]);
                const __m128d va = _mm_cvtps_pd(fa);
                const __m128d vb = _mm_loadu_pd(b + k);
                v0 = _mm_add_pd(v0, _mm_mul_pd(va, vb));
            }
            s = hsum_xmm_pd(v0);
#endif
            for (; k < len; ++k)
            {
                s = scalar_fmadd(static_cast<double>(a[k]), b[k], s);
            }
            return s;
        }
        else
        {
            Refine s{0};
            for (std::size_t k = 0U; k < len; ++k)
            {
                s = scalar_fmadd(static_cast<Refine>(a[k]), b[k], s);
            }
            return s;
        }
    }

    /// @brief Computes `sum_k static_cast<Refine>(a[k]) * static_cast<Refine>(b[k])` for two Storage vectors.
    /// @tparam Storage Storage scalar type (`double` or `float`).
    /// @tparam Refine Accumulation/output scalar type (`double` or `float`).
    /// @param[in] a First Storage vector.
    /// @param[in] b Second Storage vector.
    /// @param[in] len Number of elements in each vector.
    /// @return Inner product accumulated in Refine precision with deterministic SIMD lane folding.
    template <SimdBlasScalar Storage, SimdBlasScalar Refine>
    [[nodiscard]] inline Refine dot_storage_to_refine_simd(const Storage* a, const Storage* b, std::size_t len) noexcept
    {
        if constexpr (std::same_as<Storage, Refine>)
        {
            return dot_1x1_simd<Refine>(a, b, len);
        }
        else if constexpr (std::same_as<Storage, float> && std::same_as<Refine, double>)
        {
            std::size_t k = 0U;
            double s = 0.0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v0 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d va = _mm512_cvtps_pd(_mm256_loadu_ps(a + k));
                const __m512d vb = _mm512_cvtps_pd(_mm256_loadu_ps(b + k));
                v0 = blas3_fmadd_pd(va, vb, v0);
            }
            s = hsum_zmm_pd(v0);
#elifdef __AVX2__
            __m256d v0 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d va = _mm256_cvtps_pd(_mm_loadu_ps(a + k));
                const __m256d vb = _mm256_cvtps_pd(_mm_loadu_ps(b + k));
                v0 = blas3_fmadd_pd(va, vb, v0);
            }
            s = hsum_ymm_pd(v0);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v0 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d va = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a[k + 1U], a[k]));
                const __m128d vb = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, b[k + 1U], b[k]));
                v0 = _mm_add_pd(v0, _mm_mul_pd(va, vb));
            }
            s = hsum_xmm_pd(v0);
#endif
            for (; k < len; ++k)
            {
                s = scalar_fmadd(static_cast<double>(a[k]), static_cast<double>(b[k]), s);
            }
            return s;
        }
        else
        {
            Refine s{0};
            for (std::size_t k = 0U; k < len; ++k)
            {
                s = scalar_fmadd(static_cast<Refine>(a[k]), static_cast<Refine>(b[k]), s);
            }
            return s;
        }
    }

    /// @brief Computes a 4x2 block of Storage-to-Refine promoted dot products `out[r][c] = dot(a_r, b_c)`.
    /// @details Delegates to dot_4x2_simd() when `Storage == Refine`, and uses promoted SIMD FMA lanes matching
    /// dot_storage_to_refine_simd() bitwise when `Storage == float` and `Refine == double`.
    /// @tparam Storage Storage scalar type (`double` or `float`).
    /// @tparam Refine Accumulation/output scalar type (`double` or `float`).
    /// @param[in] a0 First Storage left-hand vector.
    /// @param[in] a1 Second Storage left-hand vector.
    /// @param[in] a2 Third Storage left-hand vector.
    /// @param[in] a3 Fourth Storage left-hand vector.
    /// @param[in] b0 First Storage right-hand vector.
    /// @param[in] b1 Second Storage right-hand vector.
    /// @param[in] len Number of elements in each vector.
    /// @param[out] out Output array populated with `out[row][col]` in Refine precision.
    template <SimdBlasScalar Storage, SimdBlasScalar Refine>
    inline void dot_4x2_storage_to_refine_simd(
        const Storage* a0,
        const Storage* a1,
        const Storage* a2,
        const Storage* a3,
        const Storage* b0,
        const Storage* b1,
        std::size_t len,
        std::array<std::array<Refine, 2>, 4>& out
    ) noexcept
    {
        if constexpr (std::same_as<Storage, Refine>)
        {
            dot_4x2_simd<Refine>(a0, a1, a2, a3, b0, b1, len, out);
        }
        else if constexpr (std::same_as<Storage, float> && std::same_as<Refine, double>)
        {
            std::size_t k = 0U;
            double s00 = 0.0;
            double s01 = 0.0;
            double s10 = 0.0;
            double s11 = 0.0;
            double s20 = 0.0;
            double s21 = 0.0;
            double s30 = 0.0;
            double s31 = 0.0;

#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v00 = _mm512_setzero_pd();
            __m512d v01 = _mm512_setzero_pd();
            __m512d v10 = _mm512_setzero_pd();
            __m512d v11 = _mm512_setzero_pd();
            __m512d v20 = _mm512_setzero_pd();
            __m512d v21 = _mm512_setzero_pd();
            __m512d v30 = _mm512_setzero_pd();
            __m512d v31 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d vb0 = _mm512_cvtps_pd(_mm256_loadu_ps(b0 + k));
                const __m512d vb1 = _mm512_cvtps_pd(_mm256_loadu_ps(b1 + k));
                const __m512d va0 = _mm512_cvtps_pd(_mm256_loadu_ps(a0 + k));
                v00 = blas3_fmadd_pd(va0, vb0, v00);
                v01 = blas3_fmadd_pd(va0, vb1, v01);
                const __m512d va1 = _mm512_cvtps_pd(_mm256_loadu_ps(a1 + k));
                v10 = blas3_fmadd_pd(va1, vb0, v10);
                v11 = blas3_fmadd_pd(va1, vb1, v11);
                const __m512d va2 = _mm512_cvtps_pd(_mm256_loadu_ps(a2 + k));
                v20 = blas3_fmadd_pd(va2, vb0, v20);
                v21 = blas3_fmadd_pd(va2, vb1, v21);
                const __m512d va3 = _mm512_cvtps_pd(_mm256_loadu_ps(a3 + k));
                v30 = blas3_fmadd_pd(va3, vb0, v30);
                v31 = blas3_fmadd_pd(va3, vb1, v31);
            }
            s00 = hsum_zmm_pd(v00);
            s01 = hsum_zmm_pd(v01);
            s10 = hsum_zmm_pd(v10);
            s11 = hsum_zmm_pd(v11);
            s20 = hsum_zmm_pd(v20);
            s21 = hsum_zmm_pd(v21);
            s30 = hsum_zmm_pd(v30);
            s31 = hsum_zmm_pd(v31);
#elifdef __AVX2__
            __m256d v00 = _mm256_setzero_pd();
            __m256d v01 = _mm256_setzero_pd();
            __m256d v10 = _mm256_setzero_pd();
            __m256d v11 = _mm256_setzero_pd();
            __m256d v20 = _mm256_setzero_pd();
            __m256d v21 = _mm256_setzero_pd();
            __m256d v30 = _mm256_setzero_pd();
            __m256d v31 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d vb0 = _mm256_cvtps_pd(_mm_loadu_ps(b0 + k));
                const __m256d vb1 = _mm256_cvtps_pd(_mm_loadu_ps(b1 + k));
                const __m256d va0 = _mm256_cvtps_pd(_mm_loadu_ps(a0 + k));
                v00 = blas3_fmadd_pd(va0, vb0, v00);
                v01 = blas3_fmadd_pd(va0, vb1, v01);
                const __m256d va1 = _mm256_cvtps_pd(_mm_loadu_ps(a1 + k));
                v10 = blas3_fmadd_pd(va1, vb0, v10);
                v11 = blas3_fmadd_pd(va1, vb1, v11);
                const __m256d va2 = _mm256_cvtps_pd(_mm_loadu_ps(a2 + k));
                v20 = blas3_fmadd_pd(va2, vb0, v20);
                v21 = blas3_fmadd_pd(va2, vb1, v21);
                const __m256d va3 = _mm256_cvtps_pd(_mm_loadu_ps(a3 + k));
                v30 = blas3_fmadd_pd(va3, vb0, v30);
                v31 = blas3_fmadd_pd(va3, vb1, v31);
            }
            s00 = hsum_ymm_pd(v00);
            s01 = hsum_ymm_pd(v01);
            s10 = hsum_ymm_pd(v10);
            s11 = hsum_ymm_pd(v11);
            s20 = hsum_ymm_pd(v20);
            s21 = hsum_ymm_pd(v21);
            s30 = hsum_ymm_pd(v30);
            s31 = hsum_ymm_pd(v31);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v00 = _mm_setzero_pd();
            __m128d v01 = _mm_setzero_pd();
            __m128d v10 = _mm_setzero_pd();
            __m128d v11 = _mm_setzero_pd();
            __m128d v20 = _mm_setzero_pd();
            __m128d v21 = _mm_setzero_pd();
            __m128d v30 = _mm_setzero_pd();
            __m128d v31 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vb0 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, b0[k + 1U], b0[k]));
                const __m128d vb1 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, b1[k + 1U], b1[k]));
                const __m128d va0 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a0[k + 1U], a0[k]));
                v00 = _mm_add_pd(v00, _mm_mul_pd(va0, vb0));
                v01 = _mm_add_pd(v01, _mm_mul_pd(va0, vb1));
                const __m128d va1 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a1[k + 1U], a1[k]));
                v10 = _mm_add_pd(v10, _mm_mul_pd(va1, vb0));
                v11 = _mm_add_pd(v11, _mm_mul_pd(va1, vb1));
                const __m128d va2 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a2[k + 1U], a2[k]));
                v20 = _mm_add_pd(v20, _mm_mul_pd(va2, vb0));
                v21 = _mm_add_pd(v21, _mm_mul_pd(va2, vb1));
                const __m128d va3 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a3[k + 1U], a3[k]));
                v30 = _mm_add_pd(v30, _mm_mul_pd(va3, vb0));
                v31 = _mm_add_pd(v31, _mm_mul_pd(va3, vb1));
            }
            s00 = hsum_xmm_pd(v00);
            s01 = hsum_xmm_pd(v01);
            s10 = hsum_xmm_pd(v10);
            s11 = hsum_xmm_pd(v11);
            s20 = hsum_xmm_pd(v20);
            s21 = hsum_xmm_pd(v21);
            s30 = hsum_xmm_pd(v30);
            s31 = hsum_xmm_pd(v31);
#endif
            for (; k < len; ++k)
            {
                const auto b0k = static_cast<double>(b0[k]);
                const auto b1k = static_cast<double>(b1[k]);
                const auto a0k = static_cast<double>(a0[k]);
                const auto a1k = static_cast<double>(a1[k]);
                const auto a2k = static_cast<double>(a2[k]);
                const auto a3k = static_cast<double>(a3[k]);
                s00 = scalar_fmadd(a0k, b0k, s00);
                s01 = scalar_fmadd(a0k, b1k, s01);
                s10 = scalar_fmadd(a1k, b0k, s10);
                s11 = scalar_fmadd(a1k, b1k, s11);
                s20 = scalar_fmadd(a2k, b0k, s20);
                s21 = scalar_fmadd(a2k, b1k, s21);
                s30 = scalar_fmadd(a3k, b0k, s30);
                s31 = scalar_fmadd(a3k, b1k, s31);
            }
            out[0] = {s00, s01};
            out[1] = {s10, s11};
            out[2] = {s20, s21};
            out[3] = {s30, s31};
        }
        else
        {
            out[0] = {
                dot_storage_to_refine_simd<Storage, Refine>(a0, b0, len),
                dot_storage_to_refine_simd<Storage, Refine>(a0, b1, len)
            };
            out[1] = {
                dot_storage_to_refine_simd<Storage, Refine>(a1, b0, len),
                dot_storage_to_refine_simd<Storage, Refine>(a1, b1, len)
            };
            out[2] = {
                dot_storage_to_refine_simd<Storage, Refine>(a2, b0, len),
                dot_storage_to_refine_simd<Storage, Refine>(a2, b1, len)
            };
            out[3] = {
                dot_storage_to_refine_simd<Storage, Refine>(a3, b0, len),
                dot_storage_to_refine_simd<Storage, Refine>(a3, b1, len)
            };
        }
    }

    /// @brief Computes 4 Storage-to-Refine promoted dot products `out[r] = dot(a_r, b0)`.
    /// @tparam Storage Storage scalar type (`double` or `float`).
    /// @tparam Refine Accumulation/output scalar type (`double` or `float`).
    /// @param[in] a0 First Storage left-hand vector.
    /// @param[in] a1 Second Storage left-hand vector.
    /// @param[in] a2 Third Storage left-hand vector.
    /// @param[in] a3 Fourth Storage left-hand vector.
    /// @param[in] b0 Storage right-hand vector.
    /// @param[in] len Number of elements in each vector.
    /// @param[out] out Output array populated with `out[row] = dot(a_row, b0)` in Refine precision.
    template <SimdBlasScalar Storage, SimdBlasScalar Refine>
    inline void dot_4x1_storage_to_refine_simd(
        const Storage* a0,
        const Storage* a1,
        const Storage* a2,
        const Storage* a3,
        const Storage* b0,
        std::size_t len,
        std::array<Refine, 4>& out
    ) noexcept
    {
        if constexpr (std::same_as<Storage, Refine>)
        {
            dot_4x1_simd<Refine>(a0, a1, a2, a3, b0, len, out);
        }
        else if constexpr (std::same_as<Storage, float> && std::same_as<Refine, double>)
        {
            std::size_t k = 0U;
            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;

#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d v0 = _mm512_setzero_pd();
            __m512d v1 = _mm512_setzero_pd();
            __m512d v2 = _mm512_setzero_pd();
            __m512d v3 = _mm512_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d vb = _mm512_cvtps_pd(_mm256_loadu_ps(b0 + k));
                v0 = blas3_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(a0 + k)), vb, v0);
                v1 = blas3_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(a1 + k)), vb, v1);
                v2 = blas3_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(a2 + k)), vb, v2);
                v3 = blas3_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(a3 + k)), vb, v3);
            }
            s0 = hsum_zmm_pd(v0);
            s1 = hsum_zmm_pd(v1);
            s2 = hsum_zmm_pd(v2);
            s3 = hsum_zmm_pd(v3);
#elifdef __AVX2__
            __m256d v0 = _mm256_setzero_pd();
            __m256d v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd();
            __m256d v3 = _mm256_setzero_pd();
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d vb = _mm256_cvtps_pd(_mm_loadu_ps(b0 + k));
                v0 = blas3_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a0 + k)), vb, v0);
                v1 = blas3_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a1 + k)), vb, v1);
                v2 = blas3_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a2 + k)), vb, v2);
                v3 = blas3_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a3 + k)), vb, v3);
            }
            s0 = hsum_ymm_pd(v0);
            s1 = hsum_ymm_pd(v1);
            s2 = hsum_ymm_pd(v2);
            s3 = hsum_ymm_pd(v3);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d v0 = _mm_setzero_pd();
            __m128d v1 = _mm_setzero_pd();
            __m128d v2 = _mm_setzero_pd();
            __m128d v3 = _mm_setzero_pd();
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vb = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, b0[k + 1U], b0[k]));
                v0 = _mm_add_pd(v0, _mm_mul_pd(_mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a0[k + 1U], a0[k])), vb));
                v1 = _mm_add_pd(v1, _mm_mul_pd(_mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a1[k + 1U], a1[k])), vb));
                v2 = _mm_add_pd(v2, _mm_mul_pd(_mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a2[k + 1U], a2[k])), vb));
                v3 = _mm_add_pd(v3, _mm_mul_pd(_mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, a3[k + 1U], a3[k])), vb));
            }
            s0 = hsum_xmm_pd(v0);
            s1 = hsum_xmm_pd(v1);
            s2 = hsum_xmm_pd(v2);
            s3 = hsum_xmm_pd(v3);
#endif
            for (; k < len; ++k)
            {
                const auto bk = static_cast<double>(b0[k]);
                s0 = scalar_fmadd(static_cast<double>(a0[k]), bk, s0);
                s1 = scalar_fmadd(static_cast<double>(a1[k]), bk, s1);
                s2 = scalar_fmadd(static_cast<double>(a2[k]), bk, s2);
                s3 = scalar_fmadd(static_cast<double>(a3[k]), bk, s3);
            }
            out = {s0, s1, s2, s3};
        }
        else
        {
            out[0] = dot_storage_to_refine_simd<Storage, Refine>(a0, b0, len);
            out[1] = dot_storage_to_refine_simd<Storage, Refine>(a1, b0, len);
            out[2] = dot_storage_to_refine_simd<Storage, Refine>(a2, b0, len);
            out[3] = dot_storage_to_refine_simd<Storage, Refine>(a3, b0, len);
        }
    }

    /// @brief Computes squared Euclidean norm `sum_k static_cast<Refine>(a[k])^2` with 4-way SIMD unrolling.
    /// @tparam Storage Storage scalar type (`double` or `float`).
    /// @tparam Refine Accumulation/output scalar type (`double` or `float`).
    /// @param[in] a Input vector in Storage precision.
    /// @param[in] len Number of contiguous elements.
    /// @return Squared l2 norm in Refine precision.
    template <SimdBlasScalar Storage, SimdBlasScalar Refine>
    [[nodiscard]] inline Refine sq_norm_mixed_simd(const Storage* a, std::size_t len) noexcept
    {
        if constexpr (std::same_as<Storage, double> && std::same_as<Refine, double>)
        {
            std::size_t k = 0U;
            double tail = 0.0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512d s0 = _mm512_setzero_pd();
            __m512d s1 = _mm512_setzero_pd();
            __m512d s2 = _mm512_setzero_pd();
            __m512d s3 = _mm512_setzero_pd();
            for (; k + 32U <= len; k += 32U)
            {
                const __m512d v0 = _mm512_loadu_pd(a + k);
                const __m512d v1 = _mm512_loadu_pd(a + k + 8U);
                const __m512d v2 = _mm512_loadu_pd(a + k + 16U);
                const __m512d v3 = _mm512_loadu_pd(a + k + 24U);
                s0 = blas3_fmadd_pd(v0, v0, s0);
                s1 = blas3_fmadd_pd(v1, v1, s1);
                s2 = blas3_fmadd_pd(v2, v2, s2);
                s3 = blas3_fmadd_pd(v3, v3, s3);
            }
            __m512d s = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
            for (; k + 8U <= len; k += 8U)
            {
                const __m512d v = _mm512_loadu_pd(a + k);
                s = blas3_fmadd_pd(v, v, s);
            }
            tail = hsum_zmm_pd(s);
#elifdef __AVX2__
            __m256d s0 = _mm256_setzero_pd();
            __m256d s1 = _mm256_setzero_pd();
            __m256d s2 = _mm256_setzero_pd();
            __m256d s3 = _mm256_setzero_pd();
            for (; k + 16U <= len; k += 16U)
            {
                const __m256d v0 = _mm256_loadu_pd(a + k);
                const __m256d v1 = _mm256_loadu_pd(a + k + 4U);
                const __m256d v2 = _mm256_loadu_pd(a + k + 8U);
                const __m256d v3 = _mm256_loadu_pd(a + k + 12U);
                s0 = blas3_fmadd_pd(v0, v0, s0);
                s1 = blas3_fmadd_pd(v1, v1, s1);
                s2 = blas3_fmadd_pd(v2, v2, s2);
                s3 = blas3_fmadd_pd(v3, v3, s3);
            }
            __m256d s = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
            for (; k + 4U <= len; k += 4U)
            {
                const __m256d v = _mm256_loadu_pd(a + k);
                s = blas3_fmadd_pd(v, v, s);
            }
            tail = hsum_ymm_pd(s);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128d s0 = _mm_setzero_pd();
            __m128d s1 = _mm_setzero_pd();
            __m128d s2 = _mm_setzero_pd();
            __m128d s3 = _mm_setzero_pd();
            for (; k + 8U <= len; k += 8U)
            {
                const __m128d v0 = _mm_loadu_pd(a + k);
                const __m128d v1 = _mm_loadu_pd(a + k + 2U);
                const __m128d v2 = _mm_loadu_pd(a + k + 4U);
                const __m128d v3 = _mm_loadu_pd(a + k + 6U);
                s0 = _mm_add_pd(s0, _mm_mul_pd(v0, v0));
                s1 = _mm_add_pd(s1, _mm_mul_pd(v1, v1));
                s2 = _mm_add_pd(s2, _mm_mul_pd(v2, v2));
                s3 = _mm_add_pd(s3, _mm_mul_pd(v3, v3));
            }
            __m128d s = _mm_add_pd(_mm_add_pd(s0, s1), _mm_add_pd(s2, s3));
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d v = _mm_loadu_pd(a + k);
                s = _mm_add_pd(s, _mm_mul_pd(v, v));
            }
            tail = hsum_xmm_pd(s);
#endif
            for (; k < len; ++k)
            {
                tail = scalar_fmadd(a[k], a[k], tail);
            }
            return tail;
        }
        else if constexpr (std::same_as<Storage, float> && std::same_as<Refine, float>)
        {
            std::size_t k = 0U;
            float tail = 0.0F;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512 s0 = _mm512_setzero_ps();
            __m512 s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps();
            __m512 s3 = _mm512_setzero_ps();
            for (; k + 64U <= len; k += 64U)
            {
                const __m512 v0 = _mm512_loadu_ps(a + k);
                const __m512 v1 = _mm512_loadu_ps(a + k + 16U);
                const __m512 v2 = _mm512_loadu_ps(a + k + 32U);
                const __m512 v3 = _mm512_loadu_ps(a + k + 48U);
                s0 = blas3_fmadd_ps(v0, v0, s0);
                s1 = blas3_fmadd_ps(v1, v1, s1);
                s2 = blas3_fmadd_ps(v2, v2, s2);
                s3 = blas3_fmadd_ps(v3, v3, s3);
            }
            __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
            for (; k + 16U <= len; k += 16U)
            {
                const __m512 v = _mm512_loadu_ps(a + k);
                s = blas3_fmadd_ps(v, v, s);
            }
            tail = hsum_zmm_ps(s);
#elifdef __AVX2__
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            for (; k + 32U <= len; k += 32U)
            {
                const __m256 v0 = _mm256_loadu_ps(a + k);
                const __m256 v1 = _mm256_loadu_ps(a + k + 8U);
                const __m256 v2 = _mm256_loadu_ps(a + k + 16U);
                const __m256 v3 = _mm256_loadu_ps(a + k + 24U);
                s0 = blas3_fmadd_ps(v0, v0, s0);
                s1 = blas3_fmadd_ps(v1, v1, s1);
                s2 = blas3_fmadd_ps(v2, v2, s2);
                s3 = blas3_fmadd_ps(v3, v3, s3);
            }
            __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
            for (; k + 8U <= len; k += 8U)
            {
                const __m256 v = _mm256_loadu_ps(a + k);
                s = blas3_fmadd_ps(v, v, s);
            }
            tail = hsum_ymm_ps(s);
#elif defined(_M_X64) || defined(__x86_64__)
            __m128 s0 = _mm_setzero_ps();
            __m128 s1 = _mm_setzero_ps();
            __m128 s2 = _mm_setzero_ps();
            __m128 s3 = _mm_setzero_ps();
            for (; k + 16U <= len; k += 16U)
            {
                const __m128 v0 = _mm_loadu_ps(a + k);
                const __m128 v1 = _mm_loadu_ps(a + k + 4U);
                const __m128 v2 = _mm_loadu_ps(a + k + 8U);
                const __m128 v3 = _mm_loadu_ps(a + k + 12U);
                s0 = _mm_add_ps(s0, _mm_mul_ps(v0, v0));
                s1 = _mm_add_ps(s1, _mm_mul_ps(v1, v1));
                s2 = _mm_add_ps(s2, _mm_mul_ps(v2, v2));
                s3 = _mm_add_ps(s3, _mm_mul_ps(v3, v3));
            }
            __m128 s = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 v = _mm_loadu_ps(a + k);
                s = _mm_add_ps(s, _mm_mul_ps(v, v));
            }
            tail = hsum_xmm_ps(s);
#endif
            for (; k < len; ++k)
            {
                tail = scalar_fmadd(a[k], a[k], tail);
            }
            return tail;
        }
        else
        {
            return dot_storage_to_refine_simd<Storage, Refine>(a, a, len);
        }
    }

}  // namespace cxxgpr::detail
