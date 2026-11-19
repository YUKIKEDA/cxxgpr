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

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on eight packed non-positive binary64 lanes in AVX-512 registers.
    /// @details Clamps arguments below `-708.0` and zero-masks underflow lanes via opmask without branches or libm
    /// calls while propagating NaN lanes cleanly. Assumes finite `vx_raw <= 0` or NaN.
    /// @param[in] vx_raw Eight packed non-positive binary64 exponents.
    /// @return Eight packed binary64 exponentials matching scalar `fast_exp_f64`.
    [[nodiscard]] inline __m512d fast_exp_nonpos_avx512_f64(__m512d vx_raw) noexcept
    {
        const __m512d vmin_x = _mm512_set1_pd(-708.0);
        const __m512d vlog2e = _mm512_set1_pd(std::numbers::log2e);
        const __m512d vmagic = _mm512_set1_pd(6755399441055744.0);
        // NOLINTNEXTLINE(modernize-use-std-numbers)
        const __m512d vln2_hi = _mm512_set1_pd(6.93147180369123816490e-1);
        const __m512d vln2_lo = _mm512_set1_pd(1.90821492927058770002e-10);
        const __m512d vc7 = _mm512_set1_pd(k_taylor_f64[7]);
        const __m512d vc6 = _mm512_set1_pd(k_taylor_f64[6]);
        const __m512d vc5 = _mm512_set1_pd(k_taylor_f64[5]);
        const __m512d vc4 = _mm512_set1_pd(k_taylor_f64[4]);
        const __m512d vc3 = _mm512_set1_pd(k_taylor_f64[3]);
        const __m512d vc2 = _mm512_set1_pd(k_taylor_f64[2]);
        const __m512d vc1 = _mm512_set1_pd(k_taylor_f64[1]);
        const __m512d vc0 = _mm512_set1_pd(k_taylor_f64[0]);
        const __m512i v1023 = _mm512_set1_epi64(1023LL);

        // _CMP_NLT_UQ keeps NaN unmasked, and putting vmin_x first in max_pd preserves NaN in src2.
        const __mmask8 vmask = _mm512_cmp_pd_mask(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m512d vx = _mm512_max_pd(vmin_x, vx_raw);
        const __m512d vd_int = simd_fmadd_pd(vx, vlog2e, vmagic);
        const __m512d vn = _mm512_sub_pd(vd_int, vmagic);
        const __m512d vg = simd_fnmadd_pd(vn, vln2_lo, simd_fnmadd_pd(vn, vln2_hi, vx));

        __m512d vp = simd_fmadd_pd(vc7, vg, vc6);
        vp = simd_fmadd_pd(vp, vg, vc5);
        vp = simd_fmadd_pd(vp, vg, vc4);
        vp = simd_fmadd_pd(vp, vg, vc3);
        vp = simd_fmadd_pd(vp, vg, vc2);
        vp = simd_fmadd_pd(vp, vg, vc1);
        vp = simd_fmadd_pd(vp, vg, vc0);

        const __m512i vexp_bits = _mm512_slli_epi64(_mm512_add_epi64(_mm512_castpd_si512(vd_int), v1023), 52);
        const __m512d vpow2 = _mm512_castsi512_pd(vexp_bits);
        return _mm512_maskz_mul_pd(vmask, vp, vpow2);
    }

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on sixteen packed non-positive binary32 lanes in AVX-512
    /// registers.
    /// @details Clamps arguments below `-87.0F` and zero-masks underflow lanes via opmask without branches or libm
    /// calls while propagating NaN lanes cleanly. Assumes finite `vx_raw <= 0` or NaN.
    /// @param[in] vx_raw Sixteen packed non-positive binary32 exponents.
    /// @return Sixteen packed binary32 exponentials matching scalar `fast_exp_f32`.
    [[nodiscard]] inline __m512 fast_exp_nonpos_avx512_f32(__m512 vx_raw) noexcept
    {
        const __m512 vmin_x = _mm512_set1_ps(-87.0F);
        const __m512 vlog2e = _mm512_set1_ps(std::bit_cast<float>(0x3fb8aa3bU));
        const __m512 vmagic = _mm512_set1_ps(12582912.0F);
        const __m512 vln2_hi = _mm512_set1_ps(std::bit_cast<float>(0x3f317180U));
        const __m512 vln2_lo = _mm512_set1_ps(std::bit_cast<float>(0x3717f7d1U));
        const __m512 vc7 = _mm512_set1_ps(k_taylor_f32[7]);
        const __m512 vc6 = _mm512_set1_ps(k_taylor_f32[6]);
        const __m512 vc5 = _mm512_set1_ps(k_taylor_f32[5]);
        const __m512 vc4 = _mm512_set1_ps(k_taylor_f32[4]);
        const __m512 vc3 = _mm512_set1_ps(k_taylor_f32[3]);
        const __m512 vc2 = _mm512_set1_ps(k_taylor_f32[2]);
        const __m512 vc1 = _mm512_set1_ps(k_taylor_f32[1]);
        const __m512 vc0 = _mm512_set1_ps(k_taylor_f32[0]);
        const __m512i v127 = _mm512_set1_epi32(127);

        const __mmask16 vmask = _mm512_cmp_ps_mask(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m512 vx = _mm512_max_ps(vmin_x, vx_raw);
        const __m512 vd_int = simd_fmadd_ps(vx, vlog2e, vmagic);
        const __m512 vn = _mm512_sub_ps(vd_int, vmagic);
        const __m512 vg = simd_fnmadd_ps(vn, vln2_lo, simd_fnmadd_ps(vn, vln2_hi, vx));

        __m512 vp = simd_fmadd_ps(vc7, vg, vc6);
        vp = simd_fmadd_ps(vp, vg, vc5);
        vp = simd_fmadd_ps(vp, vg, vc4);
        vp = simd_fmadd_ps(vp, vg, vc3);
        vp = simd_fmadd_ps(vp, vg, vc2);
        vp = simd_fmadd_ps(vp, vg, vc1);
        vp = simd_fmadd_ps(vp, vg, vc0);

        const __m512i vexp_bits = _mm512_slli_epi32(_mm512_add_epi32(_mm512_castps_si512(vd_int), v127), 23);
        const __m512 vpow2 = _mm512_castsi512_ps(vexp_bits);
        return _mm512_maskz_mul_ps(vmask, vp, vpow2);
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

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on four packed non-positive binary64 lanes in AVX2+FMA registers.
    /// @details Clamps arguments below `-708.0` and masks underflow lanes to zero without branches or libm calls while
    /// propagating NaN lanes cleanly. Assumes finite `vx_raw <= 0` or NaN.
    /// @param[in] vx_raw Four packed non-positive binary64 exponents.
    /// @return Four packed binary64 exponentials matching scalar `fast_exp_f64`.
    [[nodiscard]] inline __m256d fast_exp_nonpos_avx2_f64(__m256d vx_raw) noexcept
    {
        const __m256d vmin_x = _mm256_set1_pd(-708.0);
        const __m256d vlog2e = _mm256_set1_pd(std::numbers::log2e);
        const __m256d vmagic = _mm256_set1_pd(6755399441055744.0);
        // NOLINTNEXTLINE(modernize-use-std-numbers)
        const __m256d vln2_hi = _mm256_set1_pd(6.93147180369123816490e-1);
        const __m256d vln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);
        const __m256d vc7 = _mm256_set1_pd(k_taylor_f64[7]);
        const __m256d vc6 = _mm256_set1_pd(k_taylor_f64[6]);
        const __m256d vc5 = _mm256_set1_pd(k_taylor_f64[5]);
        const __m256d vc4 = _mm256_set1_pd(k_taylor_f64[4]);
        const __m256d vc3 = _mm256_set1_pd(k_taylor_f64[3]);
        const __m256d vc2 = _mm256_set1_pd(k_taylor_f64[2]);
        const __m256d vc1 = _mm256_set1_pd(k_taylor_f64[1]);
        const __m256d vc0 = _mm256_set1_pd(k_taylor_f64[0]);
        const __m256i v1023 = _mm256_set1_epi64x(1023LL);

        const __m256d vmask = _mm256_cmp_pd(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m256d vx = _mm256_max_pd(vmin_x, vx_raw);
        const __m256d vd_int = simd_fmadd_pd(vx, vlog2e, vmagic);
        const __m256d vn = _mm256_sub_pd(vd_int, vmagic);
        const __m256d vg = simd_fnmadd_pd(vn, vln2_lo, simd_fnmadd_pd(vn, vln2_hi, vx));

        __m256d vp = simd_fmadd_pd(vc7, vg, vc6);
        vp = simd_fmadd_pd(vp, vg, vc5);
        vp = simd_fmadd_pd(vp, vg, vc4);
        vp = simd_fmadd_pd(vp, vg, vc3);
        vp = simd_fmadd_pd(vp, vg, vc2);
        vp = simd_fmadd_pd(vp, vg, vc1);
        vp = simd_fmadd_pd(vp, vg, vc0);

        const __m256i vexp_bits = _mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(vd_int), v1023), 52);
        const __m256d vpow2 = _mm256_castsi256_pd(vexp_bits);
        return _mm256_and_pd(_mm256_mul_pd(vp, vpow2), vmask);
    }

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on eight packed non-positive binary32 lanes in AVX2+FMA registers.
    /// @details Clamps arguments below `-87.0F` and masks underflow lanes to zero without branches or libm calls while
    /// propagating NaN lanes cleanly. Assumes finite `vx_raw <= 0` or NaN.
    /// @param[in] vx_raw Eight packed non-positive binary32 exponents.
    /// @return Eight packed binary32 exponentials matching scalar `fast_exp_f32`.
    [[nodiscard]] inline __m256 fast_exp_nonpos_avx2_f32(__m256 vx_raw) noexcept
    {
        const __m256 vmin_x = _mm256_set1_ps(-87.0F);
        const __m256 vlog2e = _mm256_set1_ps(std::bit_cast<float>(0x3fb8aa3bU));
        const __m256 vmagic = _mm256_set1_ps(12582912.0F);
        const __m256 vln2_hi = _mm256_set1_ps(std::bit_cast<float>(0x3f317180U));
        const __m256 vln2_lo = _mm256_set1_ps(std::bit_cast<float>(0x3717f7d1U));
        const __m256 vc7 = _mm256_set1_ps(k_taylor_f32[7]);
        const __m256 vc6 = _mm256_set1_ps(k_taylor_f32[6]);
        const __m256 vc5 = _mm256_set1_ps(k_taylor_f32[5]);
        const __m256 vc4 = _mm256_set1_ps(k_taylor_f32[4]);
        const __m256 vc3 = _mm256_set1_ps(k_taylor_f32[3]);
        const __m256 vc2 = _mm256_set1_ps(k_taylor_f32[2]);
        const __m256 vc1 = _mm256_set1_ps(k_taylor_f32[1]);
        const __m256 vc0 = _mm256_set1_ps(k_taylor_f32[0]);
        const __m256i v127 = _mm256_set1_epi32(127);

        const __m256 vmask = _mm256_cmp_ps(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m256 vx = _mm256_max_ps(vmin_x, vx_raw);
        const __m256 vd_int = simd_fmadd_ps(vx, vlog2e, vmagic);
        const __m256 vn = _mm256_sub_ps(vd_int, vmagic);
        const __m256 vg = simd_fnmadd_ps(vn, vln2_lo, simd_fnmadd_ps(vn, vln2_hi, vx));

        __m256 vp = simd_fmadd_ps(vc7, vg, vc6);
        vp = simd_fmadd_ps(vp, vg, vc5);
        vp = simd_fmadd_ps(vp, vg, vc4);
        vp = simd_fmadd_ps(vp, vg, vc3);
        vp = simd_fmadd_ps(vp, vg, vc2);
        vp = simd_fmadd_ps(vp, vg, vc1);
        vp = simd_fmadd_ps(vp, vg, vc0);

        const __m256i vexp_bits = _mm256_slli_epi32(_mm256_add_epi32(_mm256_castps_si256(vd_int), v127), 23);
        const __m256 vpow2 = _mm256_castsi256_ps(vexp_bits);
        return _mm256_and_ps(_mm256_mul_ps(vp, vpow2), vmask);
    }

#endif

#if defined(_M_X64) || defined(__x86_64__)

    /// @brief Evaluates degree-7 FastApprox `exp(x)` on two packed non-positive binary64 lanes in SSE2 registers.
    /// @details Clamps arguments below `-708.0` cleanly and masks underflow lanes to zero without branches or libm
    /// calls while propagating NaN lanes cleanly. Assumes finite `vx_raw <= 0` or NaN.
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

        // cmpnlt keeps NaN unmasked, and putting vmin_x first in max_pd preserves NaN in src2.
        const __m128d vmask = _mm_cmpnlt_pd(vx_raw, vmin_x);
        const __m128d vx = _mm_max_pd(vmin_x, vx_raw);
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
    /// calls while propagating NaN lanes cleanly. Assumes finite `vx_raw <= 0` or NaN.
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

        const __m128 vmask = _mm_cmpnlt_ps(vx_raw, vmin_x);
        const __m128 vx = _mm_max_ps(vmin_x, vx_raw);
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

    /// @brief Evaluates `out[k] = exp(neg_scale * max(out[k], 0))` in place across a contiguous slice.
    /// @details Dispatches outside the inner loop on `exp_mode`: under `KernelExp::FastApprox`, uses branchless
    /// AVX-512 / AVX2+FMA / SSE2 magic-bias range reduction (`1.5 * 2^52` / `1.5 * 2^23`), Cody-Waite subtraction,
    /// degree-7 Horner polynomial evaluation, and bit-shifted powers of two with clean underflow masking while
    /// propagating NaN entries identically to the scalar path.
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

        // Clamp negative noise to 0 with vzero as src1 so NaN in src2 passes through instead of becoming 0.
        std::size_t k = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 8U <= len)
            {
                const __m512d vzero = _mm512_setzero_pd();
                const __m512d vscale_in = _mm512_set1_pd(neg_scale);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vd = _mm512_max_pd(vzero, _mm512_loadu_pd(buf + k));
                    _mm512_storeu_pd(buf + k, fast_exp_nonpos_avx512_f64(_mm512_mul_pd(vd, vscale_in)));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 4U <= len)
            {
                const __m256d vzero = _mm256_setzero_pd();
                const __m256d vscale_in = _mm256_set1_pd(neg_scale);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vd = _mm256_max_pd(vzero, _mm256_loadu_pd(buf + k));
                    _mm256_storeu_pd(buf + k, fast_exp_nonpos_avx2_f64(_mm256_mul_pd(vd, vscale_in)));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 2U <= len)
            {
                const __m128d vzero = _mm_setzero_pd();
                const __m128d vscale_in = _mm_set1_pd(neg_scale);
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vd = _mm_max_pd(vzero, _mm_loadu_pd(buf + k));
                    _mm_storeu_pd(buf + k, fast_exp_nonpos_sse2_f64(_mm_mul_pd(vd, vscale_in)));
                }
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (k + 16U <= len)
            {
                const __m512 vzero = _mm512_setzero_ps();
                const __m512 vscale_in = _mm512_set1_ps(neg_scale);
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vd = _mm512_max_ps(vzero, _mm512_loadu_ps(buf + k));
                    _mm512_storeu_ps(buf + k, fast_exp_nonpos_avx512_f32(_mm512_mul_ps(vd, vscale_in)));
                }
            }
#endif
#ifdef __AVX2__
            if (k + 8U <= len)
            {
                const __m256 vzero = _mm256_setzero_ps();
                const __m256 vscale_in = _mm256_set1_ps(neg_scale);
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vd = _mm256_max_ps(vzero, _mm256_loadu_ps(buf + k));
                    _mm256_storeu_ps(buf + k, fast_exp_nonpos_avx2_f32(_mm256_mul_ps(vd, vscale_in)));
                }
            }
#endif
#if defined(_M_X64) || defined(__x86_64__)
            if (k + 4U <= len)
            {
                const __m128 vzero = _mm_setzero_ps();
                const __m128 vscale_in = _mm_set1_ps(neg_scale);
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vd = _mm_max_ps(vzero, _mm_loadu_ps(buf + k));
                    _mm_storeu_ps(buf + k, fast_exp_nonpos_sse2_f32(_mm_mul_ps(vd, vscale_in)));
                }
            }
#endif
        }
        for (; k < len; ++k)
        {
            const T d = buf[k] < T{0} ? T{0} : buf[k];
            buf[k] = eval_kernel_exp(d * neg_scale, KernelExp::FastApprox);
        }
    }

    /// @brief Evaluates FastApprox binary64 Matern values across AVX-512, AVX2+FMA, SSE2, and scalar tail.
    /// @details Guards `poly * exp(-u)` against `(+Inf) * 0 = NaN` when `scale_sq * s^2` overflows while preserving
    /// any input NaN lanes cleanly.
    /// @param[in,out] buf Buffer holding non-negative distance-squared values overwritten with Matern values.
    /// @param[in] len Number of contiguous entries to transform.
    /// @param[in] scale_sq Positive radial scaling factor `(2 * nu)` or `(2 * nu / ell^2)`.
    /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
    inline void matern_s2_fast_approx_f64(double* buf, std::size_t len, double scale_sq, MaternNu nu) noexcept
    {
        std::size_t k = 0U;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        if (k + 8U <= len)
        {
            const __m512d vzero = _mm512_setzero_pd();
            const __m512d vc = _mm512_set1_pd(scale_sq);
            const __m512d vone = _mm512_set1_pd(1.0);
            const __m512d vinv3 = _mm512_set1_pd(1.0 / 3.0);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vu2 = _mm512_mul_pd(_mm512_max_pd(vzero, _mm512_loadu_pd(buf + k)), vc);
                    const __m512d vu = _mm512_sqrt_pd(vu2);
                    _mm512_storeu_pd(buf + k, fast_exp_nonpos_avx512_f64(_mm512_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vu2 = _mm512_mul_pd(_mm512_max_pd(vzero, _mm512_loadu_pd(buf + k)), vc);
                    const __m512d vu = _mm512_sqrt_pd(vu2);
                    const __m512d ve = fast_exp_nonpos_avx512_f64(_mm512_sub_pd(vzero, vu));
                    const __mmask8 ve_nz = _mm512_cmp_pd_mask(ve, vzero, _CMP_NEQ_UQ);
                    _mm512_storeu_pd(buf + k, _mm512_maskz_mul_pd(ve_nz, _mm512_add_pd(vone, vu), ve));
                }
            }
            else
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vu2 = _mm512_mul_pd(_mm512_max_pd(vzero, _mm512_loadu_pd(buf + k)), vc);
                    const __m512d vu = _mm512_sqrt_pd(vu2);
                    const __m512d ve = fast_exp_nonpos_avx512_f64(_mm512_sub_pd(vzero, vu));
                    const __mmask8 ve_nz = _mm512_cmp_pd_mask(ve, vzero, _CMP_NEQ_UQ);
                    const __m512d vpoly = simd_fmadd_pd(vu2, vinv3, _mm512_add_pd(vone, vu));
                    _mm512_storeu_pd(buf + k, _mm512_maskz_mul_pd(ve_nz, vpoly, ve));
                }
            }
        }
#endif
#ifdef __AVX2__
        if (k + 4U <= len)
        {
            const __m256d vzero = _mm256_setzero_pd();
            const __m256d vc = _mm256_set1_pd(scale_sq);
            const __m256d vone = _mm256_set1_pd(1.0);
            const __m256d vinv3 = _mm256_set1_pd(1.0 / 3.0);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vu2 = _mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc);
                    const __m256d vu = _mm256_sqrt_pd(vu2);
                    _mm256_storeu_pd(buf + k, fast_exp_nonpos_avx2_f64(_mm256_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vu2 = _mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc);
                    const __m256d vu = _mm256_sqrt_pd(vu2);
                    const __m256d ve = fast_exp_nonpos_avx2_f64(_mm256_sub_pd(vzero, vu));
                    const __m256d ve_nz = _mm256_cmp_pd(ve, vzero, _CMP_NEQ_UQ);
                    _mm256_storeu_pd(buf + k, _mm256_and_pd(_mm256_mul_pd(_mm256_add_pd(vone, vu), ve), ve_nz));
                }
            }
            else
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vu2 = _mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc);
                    const __m256d vu = _mm256_sqrt_pd(vu2);
                    const __m256d ve = fast_exp_nonpos_avx2_f64(_mm256_sub_pd(vzero, vu));
                    const __m256d ve_nz = _mm256_cmp_pd(ve, vzero, _CMP_NEQ_UQ);
                    const __m256d vpoly = simd_fmadd_pd(vu2, vinv3, _mm256_add_pd(vone, vu));
                    _mm256_storeu_pd(buf + k, _mm256_and_pd(_mm256_mul_pd(vpoly, ve), ve_nz));
                }
            }
        }
#endif
#if defined(_M_X64) || defined(__x86_64__)
        if (k + 2U <= len)
        {
            const __m128d vzero = _mm_setzero_pd();
            const __m128d vc = _mm_set1_pd(scale_sq);
            const __m128d vone = _mm_set1_pd(1.0);
            const __m128d vinv3 = _mm_set1_pd(1.0 / 3.0);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(vzero, _mm_loadu_pd(buf + k)), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    _mm_storeu_pd(buf + k, fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(vzero, _mm_loadu_pd(buf + k)), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    const __m128d ve = fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu));
                    const __m128d ve_nz = _mm_cmpneq_pd(ve, vzero);
                    _mm_storeu_pd(buf + k, _mm_and_pd(_mm_mul_pd(_mm_add_pd(vone, vu), ve), ve_nz));
                }
            }
            else
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(vzero, _mm_loadu_pd(buf + k)), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    const __m128d ve = fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu));
                    const __m128d ve_nz = _mm_cmpneq_pd(ve, vzero);
                    const __m128d vpoly = _mm_add_pd(_mm_add_pd(vone, vu), _mm_mul_pd(vu2, vinv3));
                    _mm_storeu_pd(buf + k, _mm_and_pd(_mm_mul_pd(vpoly, ve), ve_nz));
                }
            }
        }
#endif
        for (; k < len; ++k)
        {
            const double clamped = buf[k] < 0.0 ? 0.0 : buf[k];
            const double u2 = scale_sq * clamped;
            if (u2 == 0.0)
            {
                buf[k] = 1.0;
                continue;
            }
            const double u = std::sqrt(u2);
            const double e = eval_kernel_exp(-u, KernelExp::FastApprox);
            if (nu == MaternNu::OneHalf)
            {
                buf[k] = e;
            }
            else if (e == 0.0)
            {
                buf[k] = 0.0;
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                buf[k] = (1.0 + u) * e;
            }
            else
            {
                buf[k] = (1.0 + u + (u2 / 3.0)) * e;
            }
        }
    }

    /// @brief Evaluates FastApprox binary32 Matern values across AVX-512, AVX2+FMA, SSE2, and scalar tail.
    /// @details Guards `poly * exp(-u)` against `(+Inf) * 0 = NaN` when `scale_sq * s^2` overflows while preserving
    /// any input NaN lanes cleanly.
    /// @param[in,out] buf Buffer holding non-negative distance-squared values overwritten with Matern values.
    /// @param[in] len Number of contiguous entries to transform.
    /// @param[in] scale_sq Positive radial scaling factor `(2 * nu)` or `(2 * nu / ell^2)`.
    /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
    inline void matern_s2_fast_approx_f32(float* buf, std::size_t len, float scale_sq, MaternNu nu) noexcept
    {
        std::size_t k = 0U;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        if (k + 16U <= len)
        {
            const __m512 vzero = _mm512_setzero_ps();
            const __m512 vc = _mm512_set1_ps(scale_sq);
            const __m512 vone = _mm512_set1_ps(1.0F);
            const __m512 vinv3 = _mm512_set1_ps(1.0F / 3.0F);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vu2 = _mm512_mul_ps(_mm512_max_ps(vzero, _mm512_loadu_ps(buf + k)), vc);
                    const __m512 vu = _mm512_sqrt_ps(vu2);
                    _mm512_storeu_ps(buf + k, fast_exp_nonpos_avx512_f32(_mm512_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vu2 = _mm512_mul_ps(_mm512_max_ps(vzero, _mm512_loadu_ps(buf + k)), vc);
                    const __m512 vu = _mm512_sqrt_ps(vu2);
                    const __m512 ve = fast_exp_nonpos_avx512_f32(_mm512_sub_ps(vzero, vu));
                    const __mmask16 ve_nz = _mm512_cmp_ps_mask(ve, vzero, _CMP_NEQ_UQ);
                    _mm512_storeu_ps(buf + k, _mm512_maskz_mul_ps(ve_nz, _mm512_add_ps(vone, vu), ve));
                }
            }
            else
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vu2 = _mm512_mul_ps(_mm512_max_ps(vzero, _mm512_loadu_ps(buf + k)), vc);
                    const __m512 vu = _mm512_sqrt_ps(vu2);
                    const __m512 ve = fast_exp_nonpos_avx512_f32(_mm512_sub_ps(vzero, vu));
                    const __mmask16 ve_nz = _mm512_cmp_ps_mask(ve, vzero, _CMP_NEQ_UQ);
                    const __m512 vpoly = simd_fmadd_ps(vu2, vinv3, _mm512_add_ps(vone, vu));
                    _mm512_storeu_ps(buf + k, _mm512_maskz_mul_ps(ve_nz, vpoly, ve));
                }
            }
        }
#endif
#ifdef __AVX2__
        if (k + 8U <= len)
        {
            const __m256 vzero = _mm256_setzero_ps();
            const __m256 vc = _mm256_set1_ps(scale_sq);
            const __m256 vone = _mm256_set1_ps(1.0F);
            const __m256 vinv3 = _mm256_set1_ps(1.0F / 3.0F);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vu2 = _mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc);
                    const __m256 vu = _mm256_sqrt_ps(vu2);
                    _mm256_storeu_ps(buf + k, fast_exp_nonpos_avx2_f32(_mm256_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vu2 = _mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc);
                    const __m256 vu = _mm256_sqrt_ps(vu2);
                    const __m256 ve = fast_exp_nonpos_avx2_f32(_mm256_sub_ps(vzero, vu));
                    const __m256 ve_nz = _mm256_cmp_ps(ve, vzero, _CMP_NEQ_UQ);
                    _mm256_storeu_ps(buf + k, _mm256_and_ps(_mm256_mul_ps(_mm256_add_ps(vone, vu), ve), ve_nz));
                }
            }
            else
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vu2 = _mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc);
                    const __m256 vu = _mm256_sqrt_ps(vu2);
                    const __m256 ve = fast_exp_nonpos_avx2_f32(_mm256_sub_ps(vzero, vu));
                    const __m256 ve_nz = _mm256_cmp_ps(ve, vzero, _CMP_NEQ_UQ);
                    const __m256 vpoly = simd_fmadd_ps(vu2, vinv3, _mm256_add_ps(vone, vu));
                    _mm256_storeu_ps(buf + k, _mm256_and_ps(_mm256_mul_ps(vpoly, ve), ve_nz));
                }
            }
        }
#endif
#if defined(_M_X64) || defined(__x86_64__)
        if (k + 4U <= len)
        {
            const __m128 vzero = _mm_setzero_ps();
            const __m128 vc = _mm_set1_ps(scale_sq);
            const __m128 vone = _mm_set1_ps(1.0F);
            const __m128 vinv3 = _mm_set1_ps(1.0F / 3.0F);
            if (nu == MaternNu::OneHalf)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(vzero, _mm_loadu_ps(buf + k)), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    _mm_storeu_ps(buf + k, fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(vzero, _mm_loadu_ps(buf + k)), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    const __m128 ve = fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu));
                    const __m128 ve_nz = _mm_cmpneq_ps(ve, vzero);
                    _mm_storeu_ps(buf + k, _mm_and_ps(_mm_mul_ps(_mm_add_ps(vone, vu), ve), ve_nz));
                }
            }
            else
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(vzero, _mm_loadu_ps(buf + k)), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    const __m128 ve = fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu));
                    const __m128 ve_nz = _mm_cmpneq_ps(ve, vzero);
                    const __m128 vpoly = _mm_add_ps(_mm_add_ps(vone, vu), _mm_mul_ps(vu2, vinv3));
                    _mm_storeu_ps(buf + k, _mm_and_ps(_mm_mul_ps(vpoly, ve), ve_nz));
                }
            }
        }
#endif
        for (; k < len; ++k)
        {
            const float clamped = buf[k] < 0.0F ? 0.0F : buf[k];
            const float u2 = scale_sq * clamped;
            if (u2 == 0.0F)
            {
                buf[k] = 1.0F;
                continue;
            }
            const float u = std::sqrt(u2);
            const float e = eval_kernel_exp(-u, KernelExp::FastApprox);
            if (nu == MaternNu::OneHalf)
            {
                buf[k] = e;
            }
            else if (e == 0.0F)
            {
                buf[k] = 0.0F;
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                buf[k] = (1.0F + u) * e;
            }
            else
            {
                buf[k] = (1.0F + u + (u2 / 3.0F)) * e;
            }
        }
    }

    /// @brief Evaluates the Matern kernel in place over a contiguous buffer holding `u^2 = scale_sq * max(s^2, 0)`.
    /// @details Dispatches `nu` and `exp_mode` once outside the slice loop; uses hardware AVX-512 / AVX2+FMA / SSE2
    /// `sqrt` and FastApprox exp under `KernelExp::FastApprox` with zero branches per element, and masks `poly * 0`
    /// underflows to `0` while keeping NaN inputs intact.
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
                    const T clamped = buf[k] < T{0} ? T{0} : buf[k];
                    const T u2 = scale_sq * clamped;
                    buf[k] = (u2 == T{0}) ? T{1} : std::exp(-std::sqrt(u2));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (std::size_t k = 0U; k < len; ++k)
                {
                    const T clamped = buf[k] < T{0} ? T{0} : buf[k];
                    const T u2 = scale_sq * clamped;
                    if (u2 == T{0})
                    {
                        buf[k] = T{1};
                    }
                    else
                    {
                        const T u = std::sqrt(u2);
                        const T e = std::exp(-u);
                        buf[k] = (e == T{0}) ? T{0} : (T{1} + u) * e;
                    }
                }
            }
            else
            {
                for (std::size_t k = 0U; k < len; ++k)
                {
                    const T clamped = buf[k] < T{0} ? T{0} : buf[k];
                    const T u2 = scale_sq * clamped;
                    if (u2 == T{0})
                    {
                        buf[k] = T{1};
                    }
                    else
                    {
                        const T u = std::sqrt(u2);
                        const T e = std::exp(-u);
                        buf[k] = (e == T{0}) ? T{0} : (T{1} + u + (u2 / T{3})) * e;
                    }
                }
            }
            return;
        }

        if constexpr (std::same_as<T, double>)
        {
            matern_s2_fast_approx_f64(buf, len, scale_sq, nu);
        }
        else
        {
            matern_s2_fast_approx_f32(buf, len, scale_sq, nu);
        }
    }

}  // namespace cxxgpr::detail
