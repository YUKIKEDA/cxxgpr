// SPDX-License-Identifier: Apache-2.0
/// @file exp_simd.hpp
/// @brief Internal AVX-512, AVX2+FMA, SSE2, and portable SIMD batch helpers for kernel evaluation.
/// @details Provides contiguous column-slice primitives (`scale_set_slice`, `axpy_slice`, `set_scaled_sq_diff_slice`,
/// `add_scaled_sq_diff_slice`, `set_dot_slice`, `add_dot_slice`, `exp_scaled_nonpos_in_place`, and
/// `matern_s2_in_place`) dispatching across AVX-512, AVX2+FMA, SSE2, and portable C++23 scalar fallbacks.
#pragma once

#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/leaves.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

/// @brief Internal SIMD and batch kernel evaluation helpers.
namespace cxxgpr::detail
{

#if defined(__AVX512F__) && defined(__AVX512DQ__)

    /// @brief Computes `a * b + c` across 8 binary64 lanes using FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m512d simd_fmadd_pd(__m512d a, __m512d b, __m512d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fmadd_pd(a, b, c);
#    else
        return _mm512_add_pd(_mm512_mul_pd(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 8 binary64 lanes using FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m512d simd_fnmadd_pd(__m512d a, __m512d b, __m512d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fnmadd_pd(a, b, c);
#    else
        return _mm512_sub_pd(c, _mm512_mul_pd(a, b));
#    endif
    }

    /// @brief Computes `a * b + c` across 16 binary32 lanes using FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m512 simd_fmadd_ps(__m512 a, __m512 b, __m512 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fmadd_ps(a, b, c);
#    else
        return _mm512_add_ps(_mm512_mul_ps(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 16 binary32 lanes using FMA when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m512 simd_fnmadd_ps(__m512 a, __m512 b, __m512 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm512_fnmadd_ps(a, b, c);
#    else
        return _mm512_sub_ps(c, _mm512_mul_ps(a, b));
#    endif
    }
#endif

#ifdef __AVX2__

    /// @brief Computes `a * b + c` across 4 binary64 lanes using FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m256d simd_fmadd_pd(__m256d a, __m256d b, __m256d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fmadd_pd(a, b, c);
#    else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 4 binary64 lanes using FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m256d simd_fnmadd_pd(__m256d a, __m256d b, __m256d c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fnmadd_pd(a, b, c);
#    else
        return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#    endif
    }

    /// @brief Computes `a * b + c` across 8 binary32 lanes using FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Addend vector.
    /// @return Lane-wise `a * b + c`.
    [[nodiscard]] inline __m256 simd_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fmadd_ps(a, b, c);
#    else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#    endif
    }

    /// @brief Computes `c - a * b` across 8 binary32 lanes using FMA3 when enabled.
    /// @param[in] a First multiplicand vector.
    /// @param[in] b Second multiplicand vector.
    /// @param[in] c Minuend vector.
    /// @return Lane-wise `c - a * b`.
    [[nodiscard]] inline __m256 simd_fnmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    {
#    if defined(__FMA__) || defined(_MSC_VER)
        return _mm256_fnmadd_ps(a, b, c);
#    else
        return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#    endif
    }
#endif

    /// @brief Checks whether every binary64 entry in contiguous slice `ptr[0..len)` is finite.
    /// @details Uses 4-way unrolled branchless SIMD exponent-mask checks (AVX-512 / AVX2 / SSE2) with IEEE 754
    /// `std::isfinite` tail handling so normal, zero, and subnormal values pass while NaN and +/-Inf return false.
    /// @param[in] ptr Pointer to the start of the contiguous binary64 slice (may be nullptr only when `len == 0`).
    /// @param[in] len Number of contiguous elements to inspect.
    /// @return True iff all elements in `ptr[0..len)` are finite.
    [[nodiscard]] bool all_finite_slice(const double* ptr, std::size_t len) noexcept;

    /// @brief Checks whether every binary32 entry in contiguous slice `ptr[0..len)` is finite.
    /// @details Uses 4-way unrolled branchless SIMD exponent-mask checks (AVX-512 / AVX2 / SSE2) with IEEE 754
    /// `std::isfinite` tail handling so normal, zero, and subnormal values pass while NaN and +/-Inf return false.
    /// @param[in] ptr Pointer to the start of the contiguous binary32 slice (may be nullptr only when `len == 0`).
    /// @param[in] len Number of contiguous elements to inspect.
    /// @return True iff all elements in `ptr[0..len)` are finite.
    [[nodiscard]] bool all_finite_slice(const float* ptr, std::size_t len) noexcept;

    /// @brief Assigns `dst[k] = src[k] * weight` across a contiguous slice of length `len`.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[out] dst Output buffer of length at least `len`.
    /// @param[in] src Input buffer of length at least `len`.
    /// @param[in] weight Scalar multiplier applied to each entry.
    /// @param[in] len Number of contiguous elements to process.
    template <KernelScalarConcept T>
    inline void scale_set_slice(T* dst, const T* src, T weight, std::size_t len) noexcept
    {
        std::size_t k = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 8U <= len)
            {
                const __m512d vw = _mm512_set1_pd(weight);
                for (; k + 8U <= len; k += 8U)
                {
                    _mm512_storeu_pd(dst + k, _mm512_mul_pd(_mm512_loadu_pd(src + k), vw));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 4U <= len)
            {
                const __m256d vw = _mm256_set1_pd(weight);
                for (; k + 4U <= len; k += 4U)
                {
                    _mm256_storeu_pd(dst + k, _mm256_mul_pd(_mm256_loadu_pd(src + k), vw));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 2U <= len)
            {
                const __m128d vw = _mm_set1_pd(weight);
                for (; k + 2U <= len; k += 2U)
                {
                    _mm_storeu_pd(dst + k, _mm_mul_pd(_mm_loadu_pd(src + k), vw));
                }
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 16U <= len)
            {
                const __m512 vw = _mm512_set1_ps(weight);
                for (; k + 16U <= len; k += 16U)
                {
                    _mm512_storeu_ps(dst + k, _mm512_mul_ps(_mm512_loadu_ps(src + k), vw));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 8U <= len)
            {
                const __m256 vw = _mm256_set1_ps(weight);
                for (; k + 8U <= len; k += 8U)
                {
                    _mm256_storeu_ps(dst + k, _mm256_mul_ps(_mm256_loadu_ps(src + k), vw));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 4U <= len)
            {
                const __m128 vw = _mm_set1_ps(weight);
                for (; k + 4U <= len; k += 4U)
                {
                    _mm_storeu_ps(dst + k, _mm_mul_ps(_mm_loadu_ps(src + k), vw));
                }
            }
#endif
        }
        for (; k < len; ++k)
        {
            dst[k] = src[k] * weight;
        }
    }

    /// @brief Accumulates `dst[k] += src[k] * weight` across a contiguous slice of length `len`.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in,out] dst Accumulator buffer of length at least `len`.
    /// @param[in] src Summand buffer of length at least `len`.
    /// @param[in] weight Scalar multiplier applied to `src`.
    /// @param[in] len Number of contiguous elements to accumulate.
    template <KernelScalarConcept T> inline void axpy_slice(T* dst, const T* src, T weight, std::size_t len) noexcept
    {
        std::size_t k = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 8U <= len)
            {
                const __m512d vw = _mm512_set1_pd(weight);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vd = _mm512_loadu_pd(dst + k);
                    const __m512d vs = _mm512_loadu_pd(src + k);
                    _mm512_storeu_pd(dst + k, simd_fmadd_pd(vs, vw, vd));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 4U <= len)
            {
                const __m256d vw = _mm256_set1_pd(weight);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vd = _mm256_loadu_pd(dst + k);
                    const __m256d vs = _mm256_loadu_pd(src + k);
                    _mm256_storeu_pd(dst + k, simd_fmadd_pd(vs, vw, vd));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 2U <= len)
            {
                const __m128d vw = _mm_set1_pd(weight);
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vd = _mm_loadu_pd(dst + k);
                    const __m128d vs = _mm_loadu_pd(src + k);
                    _mm_storeu_pd(dst + k, _mm_add_pd(vd, _mm_mul_pd(vs, vw)));
                }
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 16U <= len)
            {
                const __m512 vw = _mm512_set1_ps(weight);
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vd = _mm512_loadu_ps(dst + k);
                    const __m512 vs = _mm512_loadu_ps(src + k);
                    _mm512_storeu_ps(dst + k, simd_fmadd_ps(vs, vw, vd));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 8U <= len)
            {
                const __m256 vw = _mm256_set1_ps(weight);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vd = _mm256_loadu_ps(dst + k);
                    const __m256 vs = _mm256_loadu_ps(src + k);
                    _mm256_storeu_ps(dst + k, simd_fmadd_ps(vs, vw, vd));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 4U <= len)
            {
                const __m128 vw = _mm_set1_ps(weight);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vd = _mm_loadu_ps(dst + k);
                    const __m128 vs = _mm_loadu_ps(src + k);
                    _mm_storeu_ps(dst + k, _mm_add_ps(vd, _mm_mul_ps(vs, vw)));
                }
            }
#endif
        }
        for (; k < len; ++k)
        {
            dst[k] += src[k] * weight;
        }
    }

    /// @brief Assigns `dst[k] = (x[k] - x_ref)^2 * inv_ell2` along one contiguous feature column slice.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[out] dst Output buffer of length at least `len`.
    /// @param[in] x Contiguous column slice of feature coordinates.
    /// @param[in] x_ref Reference coordinate subtracted from each entry in `x`.
    /// @param[in] inv_ell2 Non-negative scalar multiplier `1 / ell^2` (or `1` for unweighted squared distances).
    /// @param[in] len Number of elements in the slice.
    template <KernelScalarConcept T>
    inline void set_scaled_sq_diff_slice(T* dst, const T* x, T x_ref, T inv_ell2, std::size_t len) noexcept
    {
        std::size_t k = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 8U <= len)
            {
                const __m512d vref = _mm512_set1_pd(x_ref);
                const __m512d vw = _mm512_set1_pd(inv_ell2);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(x + k), vref);
                    _mm512_storeu_pd(dst + k, _mm512_mul_pd(_mm512_mul_pd(diff, diff), vw));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 4U <= len)
            {
                const __m256d vref = _mm256_set1_pd(x_ref);
                const __m256d vw = _mm256_set1_pd(inv_ell2);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(x + k), vref);
                    _mm256_storeu_pd(dst + k, _mm256_mul_pd(_mm256_mul_pd(diff, diff), vw));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 2U <= len)
            {
                const __m128d vref = _mm_set1_pd(x_ref);
                const __m128d vw = _mm_set1_pd(inv_ell2);
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d diff = _mm_sub_pd(_mm_loadu_pd(x + k), vref);
                    _mm_storeu_pd(dst + k, _mm_mul_pd(_mm_mul_pd(diff, diff), vw));
                }
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 16U <= len)
            {
                const __m512 vref = _mm512_set1_ps(x_ref);
                const __m512 vw = _mm512_set1_ps(inv_ell2);
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(x + k), vref);
                    _mm512_storeu_ps(dst + k, _mm512_mul_ps(_mm512_mul_ps(diff, diff), vw));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 8U <= len)
            {
                const __m256 vref = _mm256_set1_ps(x_ref);
                const __m256 vw = _mm256_set1_ps(inv_ell2);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(x + k), vref);
                    _mm256_storeu_ps(dst + k, _mm256_mul_ps(_mm256_mul_ps(diff, diff), vw));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 4U <= len)
            {
                const __m128 vref = _mm_set1_ps(x_ref);
                const __m128 vw = _mm_set1_ps(inv_ell2);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 diff = _mm_sub_ps(_mm_loadu_ps(x + k), vref);
                    _mm_storeu_ps(dst + k, _mm_mul_ps(_mm_mul_ps(diff, diff), vw));
                }
            }
#endif
        }
        for (; k < len; ++k)
        {
            const T diff = x[k] - x_ref;
            dst[k] = (diff * diff) * inv_ell2;
        }
    }

    /// @brief Accumulates `dst[k] += (x[k] - x_ref)^2 * inv_ell2` along one contiguous feature column slice.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in,out] dst Accumulator buffer of length at least `len`.
    /// @param[in] x Contiguous column slice of feature coordinates.
    /// @param[in] x_ref Reference coordinate subtracted from each entry in `x`.
    /// @param[in] inv_ell2 Non-negative scalar multiplier `1 / ell^2` (or `1` for unweighted squared distances).
    /// @param[in] len Number of elements in the slice.
    template <KernelScalarConcept T>
    inline void add_scaled_sq_diff_slice(T* dst, const T* x, T x_ref, T inv_ell2, std::size_t len) noexcept
    {
        std::size_t k = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 8U <= len)
            {
                const __m512d vref = _mm512_set1_pd(x_ref);
                const __m512d vw = _mm512_set1_pd(inv_ell2);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vd = _mm512_loadu_pd(dst + k);
                    const __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(x + k), vref);
                    const __m512d sq = _mm512_mul_pd(diff, diff);
                    _mm512_storeu_pd(dst + k, simd_fmadd_pd(sq, vw, vd));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 4U <= len)
            {
                const __m256d vref = _mm256_set1_pd(x_ref);
                const __m256d vw = _mm256_set1_pd(inv_ell2);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vd = _mm256_loadu_pd(dst + k);
                    const __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(x + k), vref);
                    const __m256d sq = _mm256_mul_pd(diff, diff);
                    _mm256_storeu_pd(dst + k, simd_fmadd_pd(sq, vw, vd));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 2U <= len)
            {
                const __m128d vref = _mm_set1_pd(x_ref);
                const __m128d vw = _mm_set1_pd(inv_ell2);
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vd = _mm_loadu_pd(dst + k);
                    const __m128d diff = _mm_sub_pd(_mm_loadu_pd(x + k), vref);
                    const __m128d sq = _mm_mul_pd(_mm_mul_pd(diff, diff), vw);
                    _mm_storeu_pd(dst + k, _mm_add_pd(vd, sq));
                }
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 16U <= len)
            {
                const __m512 vref = _mm512_set1_ps(x_ref);
                const __m512 vw = _mm512_set1_ps(inv_ell2);
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vd = _mm512_loadu_ps(dst + k);
                    const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(x + k), vref);
                    const __m512 sq = _mm512_mul_ps(diff, diff);
                    _mm512_storeu_ps(dst + k, simd_fmadd_ps(sq, vw, vd));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 8U <= len)
            {
                const __m256 vref = _mm256_set1_ps(x_ref);
                const __m256 vw = _mm256_set1_ps(inv_ell2);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vd = _mm256_loadu_ps(dst + k);
                    const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(x + k), vref);
                    const __m256 sq = _mm256_mul_ps(diff, diff);
                    _mm256_storeu_ps(dst + k, simd_fmadd_ps(sq, vw, vd));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 4U <= len)
            {
                const __m128 vref = _mm_set1_ps(x_ref);
                const __m128 vw = _mm_set1_ps(inv_ell2);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vd = _mm_loadu_ps(dst + k);
                    const __m128 diff = _mm_sub_ps(_mm_loadu_ps(x + k), vref);
                    const __m128 sq = _mm_mul_ps(_mm_mul_ps(diff, diff), vw);
                    _mm_storeu_ps(dst + k, _mm_add_ps(vd, sq));
                }
            }
#endif
        }
        for (; k < len; ++k)
        {
            const T diff = x[k] - x_ref;
            dst[k] += (diff * diff) * inv_ell2;
        }
    }

    /// @brief Assigns `dst[k] = x[k] * x_ref` for the initial feature dimension of a dot-product slice.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[out] dst Output buffer of length at least `len`.
    /// @param[in] x Contiguous column slice of feature coordinates.
    /// @param[in] x_ref Reference coordinate multiplied with each entry in `x`.
    /// @param[in] len Number of elements in the slice.
    template <KernelScalarConcept T> inline void set_dot_slice(T* dst, const T* x, T x_ref, std::size_t len) noexcept
    {
        scale_set_slice(dst, x, x_ref, len);
    }

    /// @brief Accumulates `dst[k] += x[k] * x_ref` for subsequent feature dimensions of a dot-product slice.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in,out] dst Accumulator buffer of length at least `len`.
    /// @param[in] x Contiguous column slice of feature coordinates.
    /// @param[in] x_ref Reference coordinate multiplied with each entry in `x`.
    /// @param[in] len Number of elements in the slice.
    template <KernelScalarConcept T> inline void add_dot_slice(T* dst, const T* x, T x_ref, std::size_t len) noexcept
    {
        axpy_slice(dst, x, x_ref, len);
    }

    /// @brief Evaluates `buf[k] = fast_exp(buf[k])` across a non-positive contiguous binary64 slice via herumi/fmath.
    /// @param[in,out] buf Mutable binary64 buffer of non-positive inputs overwritten with exponentials.
    /// @param[in] len Number of contiguous elements in `buf`.
    void fast_exp_nonpos_slice_f64(double* buf, std::size_t len) noexcept;

    /// @brief Evaluates `buf[k] = fast_exp(buf[k])` across a non-positive contiguous binary32 slice via herumi/fmath.
    /// @param[in,out] buf Mutable binary32 buffer of non-positive inputs overwritten with exponentials.
    /// @param[in] len Number of contiguous elements in `buf`.
    void fast_exp_nonpos_slice_f32(float* buf, std::size_t len) noexcept;

    /// @brief Evaluates `out[k] = exp(neg_scale * max(out[k], 0))` in place across a contiguous slice.
    /// @details Dispatches outside the inner loop on `exp_mode`: under `KernelExp::FastApprox`, scales
    /// clamped non-negative distances by `neg_scale` and invokes herumi/fmath vectorized routines.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in,out] buf Buffer containing non-negative values overwritten with kernel exponentials.
    /// @param[in] len Number of contiguous entries to transform.
    /// @param[in] neg_scale Non-positive multiplier applied before exponentiation (e.g., `-0.5 / ell^2`).
    /// @param[in] exp_mode Exponential evaluation backend (`Accurate` or `FastApprox`).
    template <KernelScalarConcept T>
    inline void exp_scaled_nonpos_in_place(T* buf, std::size_t len, T neg_scale, KernelExp exp_mode) noexcept
    {
        if (exp_mode == KernelExp::Accurate)
        {
            for (std::size_t k = 0U; k < len; ++k)
            {
                const T d = buf[k] < T{0} ? T{0} : buf[k];
                buf[k] = std::exp(d * neg_scale);
            }
            return;
        }

        for (std::size_t k = 0U; k < len; ++k)
        {
            const T d = buf[k] < T{0} ? T{0} : buf[k];
            buf[k] = d * neg_scale;
        }
        if constexpr (std::same_as<T, double>)
        {
            fast_exp_nonpos_slice_f64(buf, len);
        }
        else
        {
            fast_exp_nonpos_slice_f32(buf, len);
        }
    }

    /// @brief Evaluates the radial Matern correlation `k(u)` over a squared-distance buffer `buf` in place.
    /// @details Overwrites `buf[k]` (containing `s^2 >= 0`) with Matern `nu` evaluated at `u = sqrt(scale_sq * s^2)`,
    /// using `std::exp` under `Accurate` and herumi/fmath L1-blocked evaluation under `FastApprox`.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in,out] buf Buffer containing non-negative `s^2` entries overwritten with Matern kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] scale_sq Positive Matern factor `2 * nu` (`1`, `3`, or `5`).
    /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
    /// @param[in] exp_mode Exponential evaluation backend (`Accurate` or `FastApprox`).
    template <KernelScalarConcept T>
    inline void matern_s2_in_place(T* buf, std::size_t len, T scale_sq, MaternNu nu, KernelExp exp_mode) noexcept
    {
        if (exp_mode == KernelExp::Accurate)
        {
            for (std::size_t k = 0U; k < len; ++k)
            {
                const T raw = buf[k] < T{0} ? T{0} : buf[k];
                const T u2 = raw * scale_sq;
                const T u = std::sqrt(u2);
                const T e = std::exp(-u);
                if (nu == MaternNu::OneHalf)
                {
                    buf[k] = e;
                }
                else if (nu == MaternNu::ThreeHalves)
                {
                    buf[k] = (e > T{0}) ? ((T{1} + u) * e) : T{0};
                }
                else
                {
                    buf[k] = (e > T{0}) ? ((T{1} + u + (u2 / T{3})) * e) : T{0};
                }
            }
            return;
        }

        if (nu == MaternNu::OneHalf)
        {
            for (std::size_t k = 0U; k < len; ++k)
            {
                const T raw = buf[k] < T{0} ? T{0} : buf[k];
                buf[k] = -std::sqrt(raw * scale_sq);
            }
            if constexpr (std::same_as<T, double>)
            {
                fast_exp_nonpos_slice_f64(buf, len);
            }
            else
            {
                fast_exp_nonpos_slice_f32(buf, len);
            }
            return;
        }

        constexpr std::size_t kChunk = 256U;
        alignas(32) std::array<T, kChunk> u_buf{};
        alignas(32) std::array<T, kChunk> e_buf{};
        for (std::size_t base = 0U; base < len; base += kChunk)
        {
            const std::size_t count = std::min(kChunk, len - base);
            for (std::size_t i = 0U; i < count; ++i)
            {
                const T raw = buf[base + i] < T{0} ? T{0} : buf[base + i];
                const T u = std::sqrt(raw * scale_sq);
                u_buf[i] = u;
                e_buf[i] = -u;
            }
            if constexpr (std::same_as<T, double>)
            {
                fast_exp_nonpos_slice_f64(e_buf.data(), count);
            }
            else
            {
                fast_exp_nonpos_slice_f32(e_buf.data(), count);
            }
            if (nu == MaternNu::ThreeHalves)
            {
                for (std::size_t i = 0U; i < count; ++i)
                {
                    const T e = e_buf[i];
                    buf[base + i] = std::isnan(e) ? e : ((e > T{0}) ? ((T{1} + u_buf[i]) * e) : T{0});
                }
            }
            else
            {
                for (std::size_t i = 0U; i < count; ++i)
                {
                    const T e = e_buf[i];
                    const T u = u_buf[i];
                    buf[base + i] = std::isnan(e) ? e : ((e > T{0}) ? ((T{1} + u + ((u * u) / T{3})) * e) : T{0});
                }
            }
        }
    }

}  // namespace cxxgpr::detail
