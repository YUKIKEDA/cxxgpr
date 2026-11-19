// SPDX-License-Identifier: Apache-2.0
/// @file exp_simd.hpp
/// @brief Internal SSE2-accelerated and portable SIMD batch helpers for kernel distance and exponential evaluation.
/// @details Provides contiguous column-slice primitives (`scale_set_slice`, `axpy_slice`, `set_scaled_sq_diff_slice`,
/// `add_scaled_sq_diff_slice`, `set_dot_slice`, `add_dot_slice`, `exp_scaled_nonpos_in_place`, and
/// `matern_s2_in_place`) using baseline x86_64 SSE2 intrinsics with portable C++23 fallbacks.
#pragma once

#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/leaves.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>

#if defined(_M_X64) || defined(__x86_64__)
#    include <emmintrin.h>
#endif

/// @brief Internal SIMD and batch kernel evaluation helpers.
namespace cxxgpr::detail
{

#if defined(_M_X64) || defined(__x86_64__)

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on two packed non-positive binary64 lanes in SSE2 registers.
    /// @details Clamps arguments below `-708.0` cleanly and masks underflow lanes to zero without branches or libm
    /// calls. Assumes finite `vx_raw <= 0`.
    /// @param[in] vx_raw Two packed non-positive binary64 exponents.
    /// @return Two packed binary64 exponentials matching scalar `fast_exp_f64`.
    [[nodiscard]] inline __m128d fast_exp_nonpos_sse2_f64(__m128d vx_raw) noexcept
    {
        const __m128d vmin_x = _mm_set1_pd(-708.0);
        const __m128d vlog2e = _mm_set1_pd(std::numbers::log2e);
        const __m128d vmagic = _mm_set1_pd(6755399441055744.0);
        // NOLINTNEXTLINE(modernize-use-std-numbers)
        const __m128d vln2_hi = _mm_set1_pd(6.93147180369123816490e-1);
        const __m128d vln2_lo = _mm_set1_pd(1.90821492927058770002e-10);
        const __m128d vc7 = _mm_set1_pd(k_taylor_f64[7]);
        const __m128d vc6 = _mm_set1_pd(k_taylor_f64[6]);
        const __m128d vc5 = _mm_set1_pd(k_taylor_f64[5]);
        const __m128d vc4 = _mm_set1_pd(k_taylor_f64[4]);
        const __m128d vc3 = _mm_set1_pd(k_taylor_f64[3]);
        const __m128d vc2 = _mm_set1_pd(k_taylor_f64[2]);
        const __m128d vc1 = _mm_set1_pd(k_taylor_f64[1]);
        const __m128d vc0 = _mm_set1_pd(k_taylor_f64[0]);
        const __m128i v1023 = _mm_set1_epi64x(1023LL);

        // Mask lanes below normal threshold -708.0 to +0.0 after clamping so mul+magic stay finite.
        const __m128d vmask = _mm_cmpge_pd(vx_raw, vmin_x);
        const __m128d vx = _mm_max_pd(vx_raw, vmin_x);
        const __m128d vd_int = _mm_add_pd(_mm_mul_pd(vx, vlog2e), vmagic);
        const __m128d vn = _mm_sub_pd(vd_int, vmagic);
        const __m128d vg = _mm_sub_pd(_mm_sub_pd(vx, _mm_mul_pd(vn, vln2_hi)), _mm_mul_pd(vn, vln2_lo));

        __m128d vp = _mm_add_pd(_mm_mul_pd(vc7, vg), vc6);
        vp = _mm_add_pd(_mm_mul_pd(vp, vg), vc5);
        vp = _mm_add_pd(_mm_mul_pd(vp, vg), vc4);
        vp = _mm_add_pd(_mm_mul_pd(vp, vg), vc3);
        vp = _mm_add_pd(_mm_mul_pd(vp, vg), vc2);
        vp = _mm_add_pd(_mm_mul_pd(vp, vg), vc1);
        vp = _mm_add_pd(_mm_mul_pd(vp, vg), vc0);

        const __m128i vexp_bits = _mm_slli_epi64(_mm_add_epi64(_mm_castpd_si128(vd_int), v1023), 52);
        const __m128d vpow2 = _mm_castsi128_pd(vexp_bits);
        return _mm_and_pd(_mm_mul_pd(vp, vpow2), vmask);
    }

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on four packed non-positive binary32 lanes in SSE2 registers.
    /// @details Clamps arguments below `-87.0F` cleanly and masks underflow lanes to zero without branches or libm
    /// calls. Assumes finite `vx_raw <= 0`.
    /// @param[in] vx_raw Four packed non-positive binary32 exponents.
    /// @return Four packed binary32 exponentials matching scalar `fast_exp_f32`.
    [[nodiscard]] inline __m128 fast_exp_nonpos_sse2_f32(__m128 vx_raw) noexcept
    {
        const __m128 vmin_x = _mm_set1_ps(-87.0F);
        const __m128 vlog2e = _mm_set1_ps(std::bit_cast<float>(0x3fb8aa3bU));
        const __m128 vmagic = _mm_set1_ps(12582912.0F);
        const __m128 vln2_hi = _mm_set1_ps(std::bit_cast<float>(0x3f317180U));
        const __m128 vln2_lo = _mm_set1_ps(std::bit_cast<float>(0x3717f7d1U));
        const __m128 vc7 = _mm_set1_ps(k_taylor_f32[7]);
        const __m128 vc6 = _mm_set1_ps(k_taylor_f32[6]);
        const __m128 vc5 = _mm_set1_ps(k_taylor_f32[5]);
        const __m128 vc4 = _mm_set1_ps(k_taylor_f32[4]);
        const __m128 vc3 = _mm_set1_ps(k_taylor_f32[3]);
        const __m128 vc2 = _mm_set1_ps(k_taylor_f32[2]);
        const __m128 vc1 = _mm_set1_ps(k_taylor_f32[1]);
        const __m128 vc0 = _mm_set1_ps(k_taylor_f32[0]);
        const __m128i v127 = _mm_set1_epi32(127);

        const __m128 vmask = _mm_cmpge_ps(vx_raw, vmin_x);
        const __m128 vx = _mm_max_ps(vx_raw, vmin_x);
        const __m128 vd_int = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic);
        const __m128 vn = _mm_sub_ps(vd_int, vmagic);
        const __m128 vg = _mm_sub_ps(_mm_sub_ps(vx, _mm_mul_ps(vn, vln2_hi)), _mm_mul_ps(vn, vln2_lo));

        __m128 vp = _mm_add_ps(_mm_mul_ps(vc7, vg), vc6);
        vp = _mm_add_ps(_mm_mul_ps(vp, vg), vc5);
        vp = _mm_add_ps(_mm_mul_ps(vp, vg), vc4);
        vp = _mm_add_ps(_mm_mul_ps(vp, vg), vc3);
        vp = _mm_add_ps(_mm_mul_ps(vp, vg), vc2);
        vp = _mm_add_ps(_mm_mul_ps(vp, vg), vc1);
        vp = _mm_add_ps(_mm_mul_ps(vp, vg), vc0);

        const __m128i vexp_bits = _mm_slli_epi32(_mm_add_epi32(_mm_castps_si128(vd_int), v127), 23);
        const __m128 vpow2 = _mm_castsi128_ps(vexp_bits);
        return _mm_and_ps(_mm_mul_ps(vp, vpow2), vmask);
    }

#endif

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
#if defined(_M_X64) || defined(__x86_64__)
        // Vectorize contiguous scaling in 128-bit SSE2 lanes to bypass scalar loop latency.
        if constexpr (std::same_as<T, double>)
        {
            const __m128d vw = _mm_set1_pd(weight);
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vs = _mm_loadu_pd(src + k);
                _mm_storeu_pd(dst + k, _mm_mul_pd(vs, vw));
            }
        }
        else
        {
            const __m128 vw = _mm_set1_ps(weight);
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 vs = _mm_loadu_ps(src + k);
                _mm_storeu_ps(dst + k, _mm_mul_ps(vs, vw));
            }
        }
#endif
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
#if defined(_M_X64) || defined(__x86_64__)
        // Stream column slices through SSE2 registers for unit-stride AXPY accumulation.
        if constexpr (std::same_as<T, double>)
        {
            const __m128d vw = _mm_set1_pd(weight);
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vd = _mm_loadu_pd(dst + k);
                const __m128d vs = _mm_loadu_pd(src + k);
                _mm_storeu_pd(dst + k, _mm_add_pd(vd, _mm_mul_pd(vs, vw)));
            }
        }
        else
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
#if defined(_M_X64) || defined(__x86_64__)
        if constexpr (std::same_as<T, double>)
        {
            const __m128d vref = _mm_set1_pd(x_ref);
            const __m128d vw = _mm_set1_pd(inv_ell2);
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vx = _mm_loadu_pd(x + k);
                const __m128d diff = _mm_sub_pd(vx, vref);
                const __m128d sq = _mm_mul_pd(diff, diff);
                _mm_storeu_pd(dst + k, _mm_mul_pd(sq, vw));
            }
        }
        else
        {
            const __m128 vref = _mm_set1_ps(x_ref);
            const __m128 vw = _mm_set1_ps(inv_ell2);
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 vx = _mm_loadu_ps(x + k);
                const __m128 diff = _mm_sub_ps(vx, vref);
                const __m128 sq = _mm_mul_ps(diff, diff);
                _mm_storeu_ps(dst + k, _mm_mul_ps(sq, vw));
            }
        }
#endif
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
#if defined(_M_X64) || defined(__x86_64__)
        if constexpr (std::same_as<T, double>)
        {
            const __m128d vref = _mm_set1_pd(x_ref);
            const __m128d vw = _mm_set1_pd(inv_ell2);
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vx = _mm_loadu_pd(x + k);
                const __m128d vd = _mm_loadu_pd(dst + k);
                const __m128d diff = _mm_sub_pd(vx, vref);
                const __m128d sq = _mm_mul_pd(_mm_mul_pd(diff, diff), vw);
                _mm_storeu_pd(dst + k, _mm_add_pd(vd, sq));
            }
        }
        else
        {
            const __m128 vref = _mm_set1_ps(x_ref);
            const __m128 vw = _mm_set1_ps(inv_ell2);
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 vx = _mm_loadu_ps(x + k);
                const __m128 vd = _mm_loadu_ps(dst + k);
                const __m128 diff = _mm_sub_ps(vx, vref);
                const __m128 sq = _mm_mul_ps(_mm_mul_ps(diff, diff), vw);
                _mm_storeu_ps(dst + k, _mm_add_ps(vd, sq));
            }
        }
#endif
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

    /// @brief Evaluates `out[k] = exp(neg_scale * max(out[k], 0))` in place across a contiguous slice.
    /// @details Dispatches outside the inner loop on `exp_mode`: under `KernelExp::FastApprox`, uses branchless
    /// SSE2 magic-bias range reduction (`1.5 * 2^52` / `1.5 * 2^23`), Cody-Waite subtraction, degree-7 Horner
    /// polynomial evaluation, and bit-shifted powers of two with clean underflow masking below normal range.
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

        // FastApprox path: clamp negative noise to 0 so x = d * neg_scale <= 0, then vectorize with SSE2.
        std::size_t k = 0U;
#if defined(_M_X64) || defined(__x86_64__)
        if constexpr (std::same_as<T, double>)
        {
            const __m128d vzero = _mm_setzero_pd();
            const __m128d vscale_in = _mm_set1_pd(neg_scale);
            for (; k + 2U <= len; k += 2U)
            {
                const __m128d vd_raw = _mm_loadu_pd(buf + k);
                const __m128d vd = _mm_max_pd(vd_raw, vzero);
                _mm_storeu_pd(buf + k, fast_exp_nonpos_sse2_f64(_mm_mul_pd(vd, vscale_in)));
            }
        }
        else
        {
            const __m128 vzero = _mm_setzero_ps();
            const __m128 vscale_in = _mm_set1_ps(neg_scale);
            for (; k + 4U <= len; k += 4U)
            {
                const __m128 vd_raw = _mm_loadu_ps(buf + k);
                const __m128 vd = _mm_max_ps(vd_raw, vzero);
                _mm_storeu_ps(buf + k, fast_exp_nonpos_sse2_f32(_mm_mul_ps(vd, vscale_in)));
            }
        }
#endif
        for (; k < len; ++k)
        {
            const T d = buf[k] < T{0} ? T{0} : buf[k];
            buf[k] = eval_kernel_exp(d * neg_scale, KernelExp::FastApprox);
        }
    }

    /// @brief Evaluates the Matern kernel in place over a contiguous buffer holding `u^2 = scale_sq * max(s^2, 0)`.
    /// @details Dispatches `nu` and `exp_mode` once outside the slice loop; uses hardware SSE2 `sqrt` +
    /// `fast_exp_nonpos_sse2` under `KernelExp::FastApprox` with zero branches per element.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in,out] buf Buffer holding non-negative distance-squared values overwritten with Matern kernel values.
    /// @param[in] len Number of contiguous entries to transform.
    /// @param[in] scale_sq Positive radial scaling factor `(2 * nu)` (ARD) or `(2 * nu / ell^2)` (isotropic).
    /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
    /// @param[in] exp_mode Exponential evaluation backend (`Accurate` or `FastApprox`).
    template <KernelScalarConcept T>
    inline void matern_s2_in_place(T* buf, std::size_t len, T scale_sq, MaternNu nu, KernelExp exp_mode) noexcept
    {
        if (exp_mode == KernelExp::Accurate)
        {
            if (nu == MaternNu::OneHalf)
            {
                for (std::size_t k = 0U; k < len; ++k)
                {
                    const T u2 = buf[k] > T{0} ? (scale_sq * buf[k]) : T{0};
                    buf[k] = u2 > T{0} ? std::exp(-std::sqrt(u2)) : T{1};
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (std::size_t k = 0U; k < len; ++k)
                {
                    const T u2 = buf[k] > T{0} ? (scale_sq * buf[k]) : T{0};
                    if (u2 <= T{0})
                    {
                        buf[k] = T{1};
                    }
                    else
                    {
                        const T u = std::sqrt(u2);
                        buf[k] = (T{1} + u) * std::exp(-u);
                    }
                }
            }
            else
            {
                for (std::size_t k = 0U; k < len; ++k)
                {
                    const T u2 = buf[k] > T{0} ? (scale_sq * buf[k]) : T{0};
                    if (u2 <= T{0})
                    {
                        buf[k] = T{1};
                    }
                    else
                    {
                        const T u = std::sqrt(u2);
                        buf[k] = (T{1} + u + (u2 / T{3})) * std::exp(-u);
                    }
                }
            }
            return;
        }

        // FastApprox path: vectorize u = sqrt(scale_sq * max(s^2, 0)), polynomial, and exp(-u) in SSE2.
        std::size_t k = 0U;
#if defined(_M_X64) || defined(__x86_64__)
        if constexpr (std::same_as<T, double>)
        {
            const __m128d vzero = _mm_setzero_pd();
            const __m128d vc = _mm_set1_pd(scale_sq);
            const __m128d vone = _mm_set1_pd(1.0);
            const __m128d vinv3 = _mm_set1_pd(1.0 / 3.0);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(_mm_loadu_pd(buf + k), vzero), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    _mm_storeu_pd(buf + k, fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(_mm_loadu_pd(buf + k), vzero), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    const __m128d ve = fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu));
                    _mm_storeu_pd(buf + k, _mm_mul_pd(_mm_add_pd(vone, vu), ve));
                }
            }
            else
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(_mm_loadu_pd(buf + k), vzero), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    const __m128d ve = fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu));
                    const __m128d vpoly = _mm_add_pd(_mm_add_pd(vone, vu), _mm_mul_pd(vu2, vinv3));
                    _mm_storeu_pd(buf + k, _mm_mul_pd(vpoly, ve));
                }
            }
        }
        else
        {
            const __m128 vzero = _mm_setzero_ps();
            const __m128 vc = _mm_set1_ps(scale_sq);
            const __m128 vone = _mm_set1_ps(1.0F);
            const __m128 vinv3 = _mm_set1_ps(1.0F / 3.0F);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(_mm_loadu_ps(buf + k), vzero), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    _mm_storeu_ps(buf + k, fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(_mm_loadu_ps(buf + k), vzero), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    const __m128 ve = fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu));
                    _mm_storeu_ps(buf + k, _mm_mul_ps(_mm_add_ps(vone, vu), ve));
                }
            }
            else
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(_mm_loadu_ps(buf + k), vzero), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    const __m128 ve = fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu));
                    const __m128 vpoly = _mm_add_ps(_mm_add_ps(vone, vu), _mm_mul_ps(vu2, vinv3));
                    _mm_storeu_ps(buf + k, _mm_mul_ps(vpoly, ve));
                }
            }
        }
#endif
        for (; k < len; ++k)
        {
            const T u2 = buf[k] > T{0} ? (scale_sq * buf[k]) : T{0};
            if (u2 <= T{0})
            {
                buf[k] = T{1};
                continue;
            }
            const T u = std::sqrt(u2);
            const T e = eval_kernel_exp(-u, KernelExp::FastApprox);
            if (nu == MaternNu::OneHalf)
            {
                buf[k] = e;
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                buf[k] = (T{1} + u) * e;
            }
            else
            {
                buf[k] = (T{1} + u + (u2 / T{3})) * e;
            }
        }
    }

}  // namespace cxxgpr::detail
