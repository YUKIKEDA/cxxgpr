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

    /// @brief Evaluates full-accuracy degree-11 minimax `exp(x)` on eight non-positive binary64 lanes in AVX-512.
    /// @param[in] vx_raw Eight packed non-positive binary64 exponents.
    /// @return Eight packed binary64 exponentials matching `fast_exp_f64`.
    [[nodiscard]] inline __m512d fast_exp_nonpos_avx512_f64(__m512d vx_raw) noexcept
    {
        const __m512d vmin_x = _mm512_set1_pd(-708.0);
        const __m512d vlog2e = _mm512_set1_pd(std::numbers::log2e);
        const __m512d vmagic = _mm512_set1_pd(6755399441055744.0);
        const __m512d vln2_hi = _mm512_set1_pd(k_fast_exp_ln2_hi_f64);
        const __m512d vln2_lo = _mm512_set1_pd(k_fast_exp_ln2_lo_f64);
        const __m512i v1023 = _mm512_set1_epi64(1023LL);
        const auto& c = k_fast_exp_poly_f64;

        const __mmask8 vmask = _mm512_cmp_pd_mask(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m512d vx = _mm512_max_pd(vmin_x, vx_raw);
        const __m512d vd_int = simd_fmadd_pd(vx, vlog2e, vmagic);
        const __m512d vn = _mm512_sub_pd(vd_int, vmagic);
        const __m512d vg = simd_fnmadd_pd(vn, vln2_lo, simd_fnmadd_pd(vn, vln2_hi, vx));

        const __m512d vg2 = _mm512_mul_pd(vg, vg);
        const __m512d vg4 = _mm512_mul_pd(vg2, vg2);
        const __m512d vg8 = _mm512_mul_pd(vg4, vg4);

        const __m512d p01 = _mm512_add_pd(_mm512_set1_pd(c[0]), vg);
        const __m512d p23 = simd_fmadd_pd(_mm512_set1_pd(c[3]), vg, _mm512_set1_pd(c[2]));
        const __m512d p45 = simd_fmadd_pd(_mm512_set1_pd(c[5]), vg, _mm512_set1_pd(c[4]));
        const __m512d p67 = simd_fmadd_pd(_mm512_set1_pd(c[7]), vg, _mm512_set1_pd(c[6]));
        const __m512d p89 = simd_fmadd_pd(_mm512_set1_pd(c[9]), vg, _mm512_set1_pd(c[8]));
        const __m512d pab = simd_fmadd_pd(_mm512_set1_pd(c[11]), vg, _mm512_set1_pd(c[10]));

        const __m512d p03 = simd_fmadd_pd(p23, vg2, p01);
        const __m512d p47 = simd_fmadd_pd(p67, vg2, p45);
        const __m512d p8b = simd_fmadd_pd(pab, vg2, p89);
        const __m512d p07 = simd_fmadd_pd(p47, vg4, p03);
        const __m512d vp = simd_fmadd_pd(p8b, vg8, p07);

        const __m512i vexp_bits = _mm512_slli_epi64(_mm512_add_epi64(_mm512_castpd_si512(vd_int), v1023), 52);
        return _mm512_maskz_mul_pd(vmask, vp, _mm512_castsi512_pd(vexp_bits));
    }

    /// @brief Evaluates full-accuracy degree-6 minimax `exp(x)` on sixteen non-positive binary32 lanes in AVX-512.
    /// @param[in] vx_raw Sixteen packed non-positive binary32 exponents.
    /// @return Sixteen packed binary32 exponentials matching `fast_exp_f32`.
    [[nodiscard]] inline __m512 fast_exp_nonpos_avx512_f32(__m512 vx_raw) noexcept
    {
        const __m512 vmin_x = _mm512_set1_ps(-87.0F);
        const __m512 vlog2e = _mm512_set1_ps(k_fast_exp_log2e_f32);
        const __m512 vmagic = _mm512_set1_ps(12582912.0F);
        const __m512 vln2_hi = _mm512_set1_ps(k_fast_exp_ln2_hi_f32);
        const __m512 vln2_lo = _mm512_set1_ps(k_fast_exp_ln2_lo_f32);
        const __m512i v127 = _mm512_set1_epi32(127);
        const auto& cf = k_fast_exp_poly_f32;

        const __mmask16 vmask = _mm512_cmp_ps_mask(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m512 vx = _mm512_max_ps(vmin_x, vx_raw);
        const __m512 vd_int = simd_fmadd_ps(vx, vlog2e, vmagic);
        const __m512 vn = _mm512_sub_ps(vd_int, vmagic);
        const __m512 vg = simd_fnmadd_ps(vn, vln2_lo, simd_fnmadd_ps(vn, vln2_hi, vx));

        const __m512 vg2 = _mm512_mul_ps(vg, vg);
        const __m512 vg4 = _mm512_mul_ps(vg2, vg2);

        const __m512 p01 = _mm512_add_ps(_mm512_set1_ps(cf[0]), vg);
        const __m512 p23 = simd_fmadd_ps(_mm512_set1_ps(cf[3]), vg, _mm512_set1_ps(cf[2]));
        const __m512 p45 = simd_fmadd_ps(_mm512_set1_ps(cf[5]), vg, _mm512_set1_ps(cf[4]));

        const __m512 p03 = simd_fmadd_ps(p23, vg2, p01);
        const __m512 p46 = simd_fmadd_ps(_mm512_set1_ps(cf[6]), vg2, p45);
        const __m512 vp = simd_fmadd_ps(p46, vg4, p03);

        const __m512i vexp_bits = _mm512_slli_epi32(_mm512_add_epi32(_mm512_castps_si512(vd_int), v127), 23);
        return _mm512_maskz_mul_ps(vmask, vp, _mm512_castsi512_ps(vexp_bits));
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

    /// @brief Evaluates full-accuracy degree-11 minimax `exp(x)` on four non-positive binary64 lanes in AVX2+FMA.
    /// @param[in] vx_raw Four packed non-positive binary64 exponents.
    /// @return Four packed binary64 exponentials matching `fast_exp_f64`.
    [[nodiscard]] inline __m256d fast_exp_nonpos_avx2_f64(__m256d vx_raw) noexcept
    {
        const __m256d vmin_x = _mm256_set1_pd(-708.0);
        const __m256d vlog2e = _mm256_set1_pd(std::numbers::log2e);
        const __m256d vmagic = _mm256_set1_pd(6755399441055744.0);
        const __m256d vln2_hi = _mm256_set1_pd(k_fast_exp_ln2_hi_f64);
        const __m256d vln2_lo = _mm256_set1_pd(k_fast_exp_ln2_lo_f64);
        const __m256i v1023 = _mm256_set1_epi64x(1023LL);
        const auto& c = k_fast_exp_poly_f64;

        const __m256d vmask = _mm256_cmp_pd(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m256d vx = _mm256_max_pd(vmin_x, vx_raw);
        const __m256d vd_int = simd_fmadd_pd(vx, vlog2e, vmagic);
        const __m256d vn = _mm256_sub_pd(vd_int, vmagic);
        const __m256d vg = simd_fnmadd_pd(vn, vln2_lo, simd_fnmadd_pd(vn, vln2_hi, vx));

        const __m256d vg2 = _mm256_mul_pd(vg, vg);
        const __m256d vg4 = _mm256_mul_pd(vg2, vg2);
        const __m256d vg8 = _mm256_mul_pd(vg4, vg4);

        const __m256d p01 = _mm256_add_pd(_mm256_set1_pd(c[0]), vg);
        const __m256d p23 = simd_fmadd_pd(_mm256_set1_pd(c[3]), vg, _mm256_set1_pd(c[2]));
        const __m256d p45 = simd_fmadd_pd(_mm256_set1_pd(c[5]), vg, _mm256_set1_pd(c[4]));
        const __m256d p67 = simd_fmadd_pd(_mm256_set1_pd(c[7]), vg, _mm256_set1_pd(c[6]));
        const __m256d p89 = simd_fmadd_pd(_mm256_set1_pd(c[9]), vg, _mm256_set1_pd(c[8]));
        const __m256d pab = simd_fmadd_pd(_mm256_set1_pd(c[11]), vg, _mm256_set1_pd(c[10]));

        const __m256d p03 = simd_fmadd_pd(p23, vg2, p01);
        const __m256d p47 = simd_fmadd_pd(p67, vg2, p45);
        const __m256d p8b = simd_fmadd_pd(pab, vg2, p89);
        const __m256d p07 = simd_fmadd_pd(p47, vg4, p03);
        const __m256d vp = simd_fmadd_pd(p8b, vg8, p07);

        const __m256i vexp_bits = _mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(vd_int), v1023), 52);
        return _mm256_and_pd(_mm256_mul_pd(vp, _mm256_castsi256_pd(vexp_bits)), vmask);
    }

    /// @brief Evaluates full-accuracy degree-11 minimax `exp(x)` on two interleaved AVX2 binary64 vectors.
    /// @param[in] vx_raw0 First packed non-positive binary64 exponent vector.
    /// @param[in] vx_raw1 Second packed non-positive binary64 exponent vector.
    /// @param[out] out0 Exponentials corresponding to `vx_raw0`.
    /// @param[out] out1 Exponentials corresponding to `vx_raw1`.
    inline void fast_exp_nonpos_avx2_f64_x2(__m256d vx_raw0, __m256d vx_raw1, __m256d& out0, __m256d& out1) noexcept
    {
        const __m256d vmin_x = _mm256_set1_pd(-708.0);
        const __m256d vlog2e = _mm256_set1_pd(std::numbers::log2e);
        const __m256d vmagic = _mm256_set1_pd(6755399441055744.0);
        const __m256d vln2_hi = _mm256_set1_pd(k_fast_exp_ln2_hi_f64);
        const __m256d vln2_lo = _mm256_set1_pd(k_fast_exp_ln2_lo_f64);
        const __m256i v1023 = _mm256_set1_epi64x(1023LL);
        const auto& c = k_fast_exp_poly_f64;

        const __m256d vmask0 = _mm256_cmp_pd(vx_raw0, vmin_x, _CMP_NLT_UQ);
        const __m256d vmask1 = _mm256_cmp_pd(vx_raw1, vmin_x, _CMP_NLT_UQ);
        const __m256d vx0 = _mm256_max_pd(vmin_x, vx_raw0);
        const __m256d vx1 = _mm256_max_pd(vmin_x, vx_raw1);

        const __m256d vd0 = simd_fmadd_pd(vx0, vlog2e, vmagic);
        const __m256d vd1 = simd_fmadd_pd(vx1, vlog2e, vmagic);
        const __m256d vn0 = _mm256_sub_pd(vd0, vmagic);
        const __m256d vn1 = _mm256_sub_pd(vd1, vmagic);
        const __m256d vg0 = simd_fnmadd_pd(vn0, vln2_lo, simd_fnmadd_pd(vn0, vln2_hi, vx0));
        const __m256d vg1 = simd_fnmadd_pd(vn1, vln2_lo, simd_fnmadd_pd(vn1, vln2_hi, vx1));

        const __m256d vg2_0 = _mm256_mul_pd(vg0, vg0);
        const __m256d vg2_1 = _mm256_mul_pd(vg1, vg1);
        const __m256d vg4_0 = _mm256_mul_pd(vg2_0, vg2_0);
        const __m256d vg4_1 = _mm256_mul_pd(vg2_1, vg2_1);
        const __m256d vg8_0 = _mm256_mul_pd(vg4_0, vg4_0);
        const __m256d vg8_1 = _mm256_mul_pd(vg4_1, vg4_1);

        const __m256d vc0 = _mm256_set1_pd(c[0]);
        const __m256d p01_0 = _mm256_add_pd(vc0, vg0);
        const __m256d p01_1 = _mm256_add_pd(vc0, vg1);
        const __m256d vc2 = _mm256_set1_pd(c[2]);
        const __m256d vc3 = _mm256_set1_pd(c[3]);
        const __m256d p23_0 = simd_fmadd_pd(vc3, vg0, vc2);
        const __m256d p23_1 = simd_fmadd_pd(vc3, vg1, vc2);
        const __m256d vc4 = _mm256_set1_pd(c[4]);
        const __m256d vc5 = _mm256_set1_pd(c[5]);
        const __m256d p45_0 = simd_fmadd_pd(vc5, vg0, vc4);
        const __m256d p45_1 = simd_fmadd_pd(vc5, vg1, vc4);
        const __m256d vc6 = _mm256_set1_pd(c[6]);
        const __m256d vc7 = _mm256_set1_pd(c[7]);
        const __m256d p67_0 = simd_fmadd_pd(vc7, vg0, vc6);
        const __m256d p67_1 = simd_fmadd_pd(vc7, vg1, vc6);
        const __m256d vc8 = _mm256_set1_pd(c[8]);
        const __m256d vc9 = _mm256_set1_pd(c[9]);
        const __m256d p89_0 = simd_fmadd_pd(vc9, vg0, vc8);
        const __m256d p89_1 = simd_fmadd_pd(vc9, vg1, vc8);
        const __m256d vc10 = _mm256_set1_pd(c[10]);
        const __m256d vc11 = _mm256_set1_pd(c[11]);
        const __m256d pab_0 = simd_fmadd_pd(vc11, vg0, vc10);
        const __m256d pab_1 = simd_fmadd_pd(vc11, vg1, vc10);

        const __m256d p03_0 = simd_fmadd_pd(p23_0, vg2_0, p01_0);
        const __m256d p03_1 = simd_fmadd_pd(p23_1, vg2_1, p01_1);
        const __m256d p47_0 = simd_fmadd_pd(p67_0, vg2_0, p45_0);
        const __m256d p47_1 = simd_fmadd_pd(p67_1, vg2_1, p45_1);
        const __m256d p8b_0 = simd_fmadd_pd(pab_0, vg2_0, p89_0);
        const __m256d p8b_1 = simd_fmadd_pd(pab_1, vg2_1, p89_1);

        const __m256d p07_0 = simd_fmadd_pd(p47_0, vg4_0, p03_0);
        const __m256d p07_1 = simd_fmadd_pd(p47_1, vg4_1, p03_1);
        const __m256d vp0 = simd_fmadd_pd(p8b_0, vg8_0, p07_0);
        const __m256d vp1 = simd_fmadd_pd(p8b_1, vg8_1, p07_1);

        const __m256i vexp0 = _mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(vd0), v1023), 52);
        const __m256i vexp1 = _mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(vd1), v1023), 52);
        out0 = _mm256_and_pd(_mm256_mul_pd(vp0, _mm256_castsi256_pd(vexp0)), vmask0);
        out1 = _mm256_and_pd(_mm256_mul_pd(vp1, _mm256_castsi256_pd(vexp1)), vmask1);
    }

    /// @brief Evaluates full-accuracy degree-6 minimax `exp(x)` on eight non-positive binary32 lanes in AVX2+FMA.
    /// @param[in] vx_raw Eight packed non-positive binary32 exponents.
    /// @return Eight packed binary32 exponentials matching `fast_exp_f32`.
    [[nodiscard]] inline __m256 fast_exp_nonpos_avx2_f32(__m256 vx_raw) noexcept
    {
        const __m256 vmin_x = _mm256_set1_ps(-87.0F);
        const __m256 vlog2e = _mm256_set1_ps(k_fast_exp_log2e_f32);
        const __m256 vmagic = _mm256_set1_ps(12582912.0F);
        const __m256 vln2_hi = _mm256_set1_ps(k_fast_exp_ln2_hi_f32);
        const __m256 vln2_lo = _mm256_set1_ps(k_fast_exp_ln2_lo_f32);
        const __m256i v127 = _mm256_set1_epi32(127);
        const auto& cf = k_fast_exp_poly_f32;

        const __m256 vmask = _mm256_cmp_ps(vx_raw, vmin_x, _CMP_NLT_UQ);
        const __m256 vx = _mm256_max_ps(vmin_x, vx_raw);
        const __m256 vd_int = simd_fmadd_ps(vx, vlog2e, vmagic);
        const __m256 vn = _mm256_sub_ps(vd_int, vmagic);
        const __m256 vg = simd_fnmadd_ps(vn, vln2_lo, simd_fnmadd_ps(vn, vln2_hi, vx));

        const __m256 vg2 = _mm256_mul_ps(vg, vg);
        const __m256 vg4 = _mm256_mul_ps(vg2, vg2);

        const __m256 p01 = _mm256_add_ps(_mm256_set1_ps(cf[0]), vg);
        const __m256 p23 = simd_fmadd_ps(_mm256_set1_ps(cf[3]), vg, _mm256_set1_ps(cf[2]));
        const __m256 p45 = simd_fmadd_ps(_mm256_set1_ps(cf[5]), vg, _mm256_set1_ps(cf[4]));

        const __m256 p03 = simd_fmadd_ps(p23, vg2, p01);
        const __m256 p46 = simd_fmadd_ps(_mm256_set1_ps(cf[6]), vg2, p45);
        const __m256 vp = simd_fmadd_ps(p46, vg4, p03);

        const __m256i vexp_bits = _mm256_slli_epi32(_mm256_add_epi32(_mm256_castps_si256(vd_int), v127), 23);
        return _mm256_and_ps(_mm256_mul_ps(vp, _mm256_castsi256_ps(vexp_bits)), vmask);
    }

    /// @brief Evaluates full-accuracy degree-6 minimax `exp(x)` on two interleaved AVX2 binary32 vectors.
    /// @param[in] vx_raw0 First packed non-positive binary32 exponent vector.
    /// @param[in] vx_raw1 Second packed non-positive binary32 exponent vector.
    /// @param[out] out0 Exponentials corresponding to `vx_raw0`.
    /// @param[out] out1 Exponentials corresponding to `vx_raw1`.
    inline void fast_exp_nonpos_avx2_f32_x2(__m256 vx_raw0, __m256 vx_raw1, __m256& out0, __m256& out1) noexcept
    {
        const __m256 vmin_x = _mm256_set1_ps(-87.0F);
        const __m256 vlog2e = _mm256_set1_ps(k_fast_exp_log2e_f32);
        const __m256 vmagic = _mm256_set1_ps(12582912.0F);
        const __m256 vln2_hi = _mm256_set1_ps(k_fast_exp_ln2_hi_f32);
        const __m256 vln2_lo = _mm256_set1_ps(k_fast_exp_ln2_lo_f32);
        const __m256i v127 = _mm256_set1_epi32(127);
        const auto& cf = k_fast_exp_poly_f32;

        const __m256 vmask0 = _mm256_cmp_ps(vx_raw0, vmin_x, _CMP_NLT_UQ);
        const __m256 vmask1 = _mm256_cmp_ps(vx_raw1, vmin_x, _CMP_NLT_UQ);
        const __m256 vx0 = _mm256_max_ps(vmin_x, vx_raw0);
        const __m256 vx1 = _mm256_max_ps(vmin_x, vx_raw1);

        const __m256 vd0 = simd_fmadd_ps(vx0, vlog2e, vmagic);
        const __m256 vd1 = simd_fmadd_ps(vx1, vlog2e, vmagic);
        const __m256 vn0 = _mm256_sub_ps(vd0, vmagic);
        const __m256 vn1 = _mm256_sub_ps(vd1, vmagic);
        const __m256 vg0 = simd_fnmadd_ps(vn0, vln2_lo, simd_fnmadd_ps(vn0, vln2_hi, vx0));
        const __m256 vg1 = simd_fnmadd_ps(vn1, vln2_lo, simd_fnmadd_ps(vn1, vln2_hi, vx1));

        const __m256 vg2_0 = _mm256_mul_ps(vg0, vg0);
        const __m256 vg2_1 = _mm256_mul_ps(vg1, vg1);
        const __m256 vg4_0 = _mm256_mul_ps(vg2_0, vg2_0);
        const __m256 vg4_1 = _mm256_mul_ps(vg2_1, vg2_1);

        const __m256 vc0 = _mm256_set1_ps(cf[0]);
        const __m256 p01_0 = _mm256_add_ps(vc0, vg0);
        const __m256 p01_1 = _mm256_add_ps(vc0, vg1);
        const __m256 vc2 = _mm256_set1_ps(cf[2]);
        const __m256 vc3 = _mm256_set1_ps(cf[3]);
        const __m256 p23_0 = simd_fmadd_ps(vc3, vg0, vc2);
        const __m256 p23_1 = simd_fmadd_ps(vc3, vg1, vc2);
        const __m256 vc4 = _mm256_set1_ps(cf[4]);
        const __m256 vc5 = _mm256_set1_ps(cf[5]);
        const __m256 p45_0 = simd_fmadd_ps(vc5, vg0, vc4);
        const __m256 p45_1 = simd_fmadd_ps(vc5, vg1, vc4);
        const __m256 vc6 = _mm256_set1_ps(cf[6]);

        const __m256 p03_0 = simd_fmadd_ps(p23_0, vg2_0, p01_0);
        const __m256 p03_1 = simd_fmadd_ps(p23_1, vg2_1, p01_1);
        const __m256 p46_0 = simd_fmadd_ps(vc6, vg2_0, p45_0);
        const __m256 p46_1 = simd_fmadd_ps(vc6, vg2_1, p45_1);
        const __m256 vp0 = simd_fmadd_ps(p46_0, vg4_0, p03_0);
        const __m256 vp1 = simd_fmadd_ps(p46_1, vg4_1, p03_1);

        const __m256i vexp0 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_castps_si256(vd0), v127), 23);
        const __m256i vexp1 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_castps_si256(vd1), v127), 23);
        out0 = _mm256_and_ps(_mm256_mul_ps(vp0, _mm256_castsi256_ps(vexp0)), vmask0);
        out1 = _mm256_and_ps(_mm256_mul_ps(vp1, _mm256_castsi256_ps(vexp1)), vmask1);
    }
#endif

#if defined(_M_X64) || defined(__x86_64__)

    /// @brief Evaluates full-accuracy degree-11 minimax `exp(x)` on two non-positive binary64 lanes in SSE2.
    /// @param[in] vx_raw Two packed non-positive binary64 exponents.
    /// @return Two packed binary64 exponentials matching `fast_exp_f64`.
    [[nodiscard]] inline __m128d fast_exp_nonpos_sse2_f64(__m128d vx_raw) noexcept
    {
        const __m128d vmin_x = _mm_set1_pd(-708.0);
        const __m128d vlog2e = _mm_set1_pd(std::numbers::log2e);
        const __m128d vmagic = _mm_set1_pd(6755399441055744.0);
        const __m128d vln2_hi = _mm_set1_pd(k_fast_exp_ln2_hi_f64);
        const __m128d vln2_lo = _mm_set1_pd(k_fast_exp_ln2_lo_f64);
        const __m128i v1023 = _mm_set1_epi64x(1023LL);
        const auto& c = k_fast_exp_poly_f64;

        const __m128d vmask = _mm_cmpnlt_pd(vx_raw, vmin_x);
        const __m128d vx = _mm_max_pd(vmin_x, vx_raw);
        const __m128d vd_int = _mm_add_pd(_mm_mul_pd(vx, vlog2e), vmagic);
        const __m128d vn = _mm_sub_pd(vd_int, vmagic);
        const __m128d vg = _mm_sub_pd(_mm_sub_pd(vx, _mm_mul_pd(vn, vln2_hi)), _mm_mul_pd(vn, vln2_lo));

        const __m128d vg2 = _mm_mul_pd(vg, vg);
        const __m128d vg4 = _mm_mul_pd(vg2, vg2);
        const __m128d vg8 = _mm_mul_pd(vg4, vg4);

        const __m128d p01 = _mm_add_pd(_mm_set1_pd(c[0]), vg);
        const __m128d p23 = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(c[3]), vg), _mm_set1_pd(c[2]));
        const __m128d p45 = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(c[5]), vg), _mm_set1_pd(c[4]));
        const __m128d p67 = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(c[7]), vg), _mm_set1_pd(c[6]));
        const __m128d p89 = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(c[9]), vg), _mm_set1_pd(c[8]));
        const __m128d pab = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(c[11]), vg), _mm_set1_pd(c[10]));

        const __m128d p03 = _mm_add_pd(_mm_mul_pd(p23, vg2), p01);
        const __m128d p47 = _mm_add_pd(_mm_mul_pd(p67, vg2), p45);
        const __m128d p8b = _mm_add_pd(_mm_mul_pd(pab, vg2), p89);
        const __m128d p07 = _mm_add_pd(_mm_mul_pd(p47, vg4), p03);
        const __m128d vp = _mm_add_pd(_mm_mul_pd(p8b, vg8), p07);

        const __m128i vexp_bits = _mm_slli_epi64(_mm_add_epi64(_mm_castpd_si128(vd_int), v1023), 52);
        return _mm_and_pd(_mm_mul_pd(vp, _mm_castsi128_pd(vexp_bits)), vmask);
    }

    /// @brief Evaluates full-accuracy degree-6 minimax `exp(x)` on four non-positive binary32 lanes in SSE2.
    /// @param[in] vx_raw Four packed non-positive binary32 exponents.
    /// @return Four packed binary32 exponentials matching `fast_exp_f32`.
    [[nodiscard]] inline __m128 fast_exp_nonpos_sse2_f32(__m128 vx_raw) noexcept
    {
        const __m128 vmin_x = _mm_set1_ps(-87.0F);
        const __m128 vlog2e = _mm_set1_ps(k_fast_exp_log2e_f32);
        const __m128 vmagic = _mm_set1_ps(12582912.0F);
        const __m128 vln2_hi = _mm_set1_ps(k_fast_exp_ln2_hi_f32);
        const __m128 vln2_lo = _mm_set1_ps(k_fast_exp_ln2_lo_f32);
        const __m128i v127 = _mm_set1_epi32(127);
        const auto& cf = k_fast_exp_poly_f32;

        const __m128 vmask = _mm_cmpnlt_ps(vx_raw, vmin_x);
        const __m128 vx = _mm_max_ps(vmin_x, vx_raw);
        const __m128 vd_int = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic);
        const __m128 vn = _mm_sub_ps(vd_int, vmagic);
        const __m128 vg = _mm_sub_ps(_mm_sub_ps(vx, _mm_mul_ps(vn, vln2_hi)), _mm_mul_ps(vn, vln2_lo));

        const __m128 vg2 = _mm_mul_ps(vg, vg);
        const __m128 vg4 = _mm_mul_ps(vg2, vg2);

        const __m128 p01 = _mm_add_ps(_mm_set1_ps(cf[0]), vg);
        const __m128 p23 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cf[3]), vg), _mm_set1_ps(cf[2]));
        const __m128 p45 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cf[5]), vg), _mm_set1_ps(cf[4]));

        const __m128 p03 = _mm_add_ps(_mm_mul_ps(p23, vg2), p01);
        const __m128 p46 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cf[6]), vg2), p45);
        const __m128 vp = _mm_add_ps(_mm_mul_ps(p46, vg4), p03);

        const __m128i vexp_bits = _mm_slli_epi32(_mm_add_epi32(_mm_castps_si128(vd_int), v127), 23);
        return _mm_and_ps(_mm_mul_ps(vp, _mm_castsi128_ps(vexp_bits)), vmask);
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
    /// @details Dispatches outside the inner loop on `exp_mode`; under `KernelExp::FastApprox`, uses fused
    /// inline AVX-512 / 2x-unrolled AVX2+FMA / SSE2 Estrin minimax exponential evaluation without branches or
    /// libm calls.
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
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256d vd0 = _mm256_max_pd(vzero, _mm256_loadu_pd(buf + k));
                    const __m256d vd1 = _mm256_max_pd(vzero, _mm256_loadu_pd(buf + k + 4U));
                    __m256d vo0{};
                    __m256d vo1{};
                    fast_exp_nonpos_avx2_f64_x2(_mm256_mul_pd(vd0, vscale_in), _mm256_mul_pd(vd1, vscale_in), vo0, vo1);
                    _mm256_storeu_pd(buf + k, vo0);
                    _mm256_storeu_pd(buf + k + 4U, vo1);
                }
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
            for (; k < len; ++k)
            {
                const double d = buf[k] < 0.0 ? 0.0 : buf[k];
                buf[k] = fast_exp_f64(d * neg_scale);
            }
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
                for (; k + 16U <= len; k += 16U)
                {
                    const __m256 vd0 = _mm256_max_ps(vzero, _mm256_loadu_ps(buf + k));
                    const __m256 vd1 = _mm256_max_ps(vzero, _mm256_loadu_ps(buf + k + 8U));
                    __m256 vo0{};
                    __m256 vo1{};
                    fast_exp_nonpos_avx2_f32_x2(_mm256_mul_ps(vd0, vscale_in), _mm256_mul_ps(vd1, vscale_in), vo0, vo1);
                    _mm256_storeu_ps(buf + k, vo0);
                    _mm256_storeu_ps(buf + k + 8U, vo1);
                }
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
            for (; k < len; ++k)
            {
                const float d = buf[k] < 0.0F ? 0.0F : buf[k];
                buf[k] = fast_exp_f32(d * neg_scale);
            }
        }
    }

    /// @brief Evaluates radial Matern correlation `k(u)` over a squared-distance slice in FastApprox binary64 mode.
    /// @details Fuses hardware `sqrt`, 2x-unrolled Estrin minimax `exp(-u)`, and polynomial multiplication in
    /// registers.
    /// @param[in,out] buf Buffer containing non-negative `s^2` entries overwritten with Matern kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] scale_sq Positive Matern factor `2 * nu` (`1`, `3`, or `5`).
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
                    const __m512d vu =
                        _mm512_sqrt_pd(_mm512_mul_pd(_mm512_max_pd(vzero, _mm512_loadu_pd(buf + k)), vc));
                    _mm512_storeu_pd(buf + k, fast_exp_nonpos_avx512_f64(_mm512_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vu =
                        _mm512_sqrt_pd(_mm512_mul_pd(_mm512_max_pd(vzero, _mm512_loadu_pd(buf + k)), vc));
                    const __m512d ve = fast_exp_nonpos_avx512_f64(_mm512_sub_pd(vzero, vu));
                    const __mmask8 mask_nz = _mm512_cmp_pd_mask(ve, vzero, _CMP_NEQ_UQ);
                    _mm512_storeu_pd(buf + k, _mm512_maskz_mul_pd(mask_nz, _mm512_add_pd(vone, vu), ve));
                }
            }
            else
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m512d vu2 = _mm512_mul_pd(_mm512_max_pd(vzero, _mm512_loadu_pd(buf + k)), vc);
                    const __m512d vu = _mm512_sqrt_pd(vu2);
                    const __m512d ve = fast_exp_nonpos_avx512_f64(_mm512_sub_pd(vzero, vu));
                    const __mmask8 mask_nz = _mm512_cmp_pd_mask(ve, vzero, _CMP_NEQ_UQ);
                    const __m512d vpoly = simd_fmadd_pd(vu2, vinv3, _mm512_add_pd(vone, vu));
                    _mm512_storeu_pd(buf + k, _mm512_maskz_mul_pd(mask_nz, vpoly, ve));
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
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256d vu0 =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc));
                    const __m256d vu1 =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k + 4U)), vc));
                    __m256d ve0{};
                    __m256d ve1{};
                    fast_exp_nonpos_avx2_f64_x2(_mm256_sub_pd(vzero, vu0), _mm256_sub_pd(vzero, vu1), ve0, ve1);
                    _mm256_storeu_pd(buf + k, ve0);
                    _mm256_storeu_pd(buf + k + 4U, ve1);
                }
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vu =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc));
                    _mm256_storeu_pd(buf + k, fast_exp_nonpos_avx2_f64(_mm256_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256d vu0 =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc));
                    const __m256d vu1 =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k + 4U)), vc));
                    __m256d ve0{};
                    __m256d ve1{};
                    fast_exp_nonpos_avx2_f64_x2(_mm256_sub_pd(vzero, vu0), _mm256_sub_pd(vzero, vu1), ve0, ve1);
                    const __m256d nz0 = _mm256_cmp_pd(ve0, vzero, _CMP_NEQ_UQ);
                    const __m256d nz1 = _mm256_cmp_pd(ve1, vzero, _CMP_NEQ_UQ);
                    _mm256_storeu_pd(buf + k, _mm256_and_pd(_mm256_mul_pd(_mm256_add_pd(vone, vu0), ve0), nz0));
                    _mm256_storeu_pd(buf + k + 4U, _mm256_and_pd(_mm256_mul_pd(_mm256_add_pd(vone, vu1), ve1), nz1));
                }
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vu =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc));
                    const __m256d ve = fast_exp_nonpos_avx2_f64(_mm256_sub_pd(vzero, vu));
                    const __m256d nz = _mm256_cmp_pd(ve, vzero, _CMP_NEQ_UQ);
                    _mm256_storeu_pd(buf + k, _mm256_and_pd(_mm256_mul_pd(_mm256_add_pd(vone, vu), ve), nz));
                }
            }
            else
            {
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256d vu0 =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc));
                    const __m256d vu1 =
                        _mm256_sqrt_pd(_mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k + 4U)), vc));
                    __m256d ve0{};
                    __m256d ve1{};
                    fast_exp_nonpos_avx2_f64_x2(_mm256_sub_pd(vzero, vu0), _mm256_sub_pd(vzero, vu1), ve0, ve1);
                    const __m256d nz0 = _mm256_cmp_pd(ve0, vzero, _CMP_NEQ_UQ);
                    const __m256d nz1 = _mm256_cmp_pd(ve1, vzero, _CMP_NEQ_UQ);
                    const __m256d poly0 = simd_fmadd_pd(_mm256_mul_pd(vu0, vu0), vinv3, _mm256_add_pd(vone, vu0));
                    const __m256d poly1 = simd_fmadd_pd(_mm256_mul_pd(vu1, vu1), vinv3, _mm256_add_pd(vone, vu1));
                    _mm256_storeu_pd(buf + k, _mm256_and_pd(_mm256_mul_pd(poly0, ve0), nz0));
                    _mm256_storeu_pd(buf + k + 4U, _mm256_and_pd(_mm256_mul_pd(poly1, ve1), nz1));
                }
                for (; k + 4U <= len; k += 4U)
                {
                    const __m256d vu2 = _mm256_mul_pd(_mm256_max_pd(vzero, _mm256_loadu_pd(buf + k)), vc);
                    const __m256d vu = _mm256_sqrt_pd(vu2);
                    const __m256d ve = fast_exp_nonpos_avx2_f64(_mm256_sub_pd(vzero, vu));
                    const __m256d nz = _mm256_cmp_pd(ve, vzero, _CMP_NEQ_UQ);
                    const __m256d vpoly = simd_fmadd_pd(vu2, vinv3, _mm256_add_pd(vone, vu));
                    _mm256_storeu_pd(buf + k, _mm256_and_pd(_mm256_mul_pd(vpoly, ve), nz));
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
                    const __m128d vu = _mm_sqrt_pd(_mm_mul_pd(_mm_max_pd(vzero, _mm_loadu_pd(buf + k)), vc));
                    _mm_storeu_pd(buf + k, fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu = _mm_sqrt_pd(_mm_mul_pd(_mm_max_pd(vzero, _mm_loadu_pd(buf + k)), vc));
                    const __m128d ve = fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu));
                    const __m128d nz = _mm_cmpneq_pd(ve, vzero);
                    _mm_storeu_pd(buf + k, _mm_and_pd(_mm_mul_pd(_mm_add_pd(vone, vu), ve), nz));
                }
            }
            else
            {
                for (; k + 2U <= len; k += 2U)
                {
                    const __m128d vu2 = _mm_mul_pd(_mm_max_pd(vzero, _mm_loadu_pd(buf + k)), vc);
                    const __m128d vu = _mm_sqrt_pd(vu2);
                    const __m128d ve = fast_exp_nonpos_sse2_f64(_mm_sub_pd(vzero, vu));
                    const __m128d nz = _mm_cmpneq_pd(ve, vzero);
                    const __m128d vpoly = _mm_add_pd(_mm_add_pd(vone, vu), _mm_mul_pd(vu2, vinv3));
                    _mm_storeu_pd(buf + k, _mm_and_pd(_mm_mul_pd(vpoly, ve), nz));
                }
            }
        }
#endif
        for (; k < len; ++k)
        {
            const double raw = buf[k] < 0.0 ? 0.0 : buf[k];
            const double u2 = raw * scale_sq;
            const double u = std::sqrt(u2);
            const double e = fast_exp_f64(-u);
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

    /// @brief Evaluates radial Matern correlation `k(u)` over a squared-distance slice in FastApprox binary32 mode.
    /// @details Fuses hardware `sqrt`, 2x-unrolled Estrin minimax `exp(-u)`, and polynomial multiplication in
    /// registers.
    /// @param[in,out] buf Buffer containing non-negative `s^2` entries overwritten with Matern kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] scale_sq Positive Matern factor `2 * nu` (`1`, `3`, or `5`).
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
                    const __m512 vu = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_max_ps(vzero, _mm512_loadu_ps(buf + k)), vc));
                    _mm512_storeu_ps(buf + k, fast_exp_nonpos_avx512_f32(_mm512_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vu = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_max_ps(vzero, _mm512_loadu_ps(buf + k)), vc));
                    const __m512 ve = fast_exp_nonpos_avx512_f32(_mm512_sub_ps(vzero, vu));
                    const __mmask16 mask_nz = _mm512_cmp_ps_mask(ve, vzero, _CMP_NEQ_UQ);
                    _mm512_storeu_ps(buf + k, _mm512_maskz_mul_ps(mask_nz, _mm512_add_ps(vone, vu), ve));
                }
            }
            else
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m512 vu2 = _mm512_mul_ps(_mm512_max_ps(vzero, _mm512_loadu_ps(buf + k)), vc);
                    const __m512 vu = _mm512_sqrt_ps(vu2);
                    const __m512 ve = fast_exp_nonpos_avx512_f32(_mm512_sub_ps(vzero, vu));
                    const __mmask16 mask_nz = _mm512_cmp_ps_mask(ve, vzero, _CMP_NEQ_UQ);
                    const __m512 vpoly = simd_fmadd_ps(vu2, vinv3, _mm512_add_ps(vone, vu));
                    _mm512_storeu_ps(buf + k, _mm512_maskz_mul_ps(mask_nz, vpoly, ve));
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
                for (; k + 16U <= len; k += 16U)
                {
                    const __m256 vu0 =
                        _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc));
                    const __m256 vu1 =
                        _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k + 8U)), vc));
                    __m256 ve0{};
                    __m256 ve1{};
                    fast_exp_nonpos_avx2_f32_x2(_mm256_sub_ps(vzero, vu0), _mm256_sub_ps(vzero, vu1), ve0, ve1);
                    _mm256_storeu_ps(buf + k, ve0);
                    _mm256_storeu_ps(buf + k + 8U, ve1);
                }
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vu = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc));
                    _mm256_storeu_ps(buf + k, fast_exp_nonpos_avx2_f32(_mm256_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m256 vu0 =
                        _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc));
                    const __m256 vu1 =
                        _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k + 8U)), vc));
                    __m256 ve0{};
                    __m256 ve1{};
                    fast_exp_nonpos_avx2_f32_x2(_mm256_sub_ps(vzero, vu0), _mm256_sub_ps(vzero, vu1), ve0, ve1);
                    const __m256 nz0 = _mm256_cmp_ps(ve0, vzero, _CMP_NEQ_UQ);
                    const __m256 nz1 = _mm256_cmp_ps(ve1, vzero, _CMP_NEQ_UQ);
                    _mm256_storeu_ps(buf + k, _mm256_and_ps(_mm256_mul_ps(_mm256_add_ps(vone, vu0), ve0), nz0));
                    _mm256_storeu_ps(buf + k + 8U, _mm256_and_ps(_mm256_mul_ps(_mm256_add_ps(vone, vu1), ve1), nz1));
                }
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vu = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc));
                    const __m256 ve = fast_exp_nonpos_avx2_f32(_mm256_sub_ps(vzero, vu));
                    const __m256 nz = _mm256_cmp_ps(ve, vzero, _CMP_NEQ_UQ);
                    _mm256_storeu_ps(buf + k, _mm256_and_ps(_mm256_mul_ps(_mm256_add_ps(vone, vu), ve), nz));
                }
            }
            else
            {
                for (; k + 16U <= len; k += 16U)
                {
                    const __m256 vu0 =
                        _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc));
                    const __m256 vu1 =
                        _mm256_sqrt_ps(_mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k + 8U)), vc));
                    __m256 ve0{};
                    __m256 ve1{};
                    fast_exp_nonpos_avx2_f32_x2(_mm256_sub_ps(vzero, vu0), _mm256_sub_ps(vzero, vu1), ve0, ve1);
                    const __m256 nz0 = _mm256_cmp_ps(ve0, vzero, _CMP_NEQ_UQ);
                    const __m256 nz1 = _mm256_cmp_ps(ve1, vzero, _CMP_NEQ_UQ);
                    const __m256 poly0 = simd_fmadd_ps(_mm256_mul_ps(vu0, vu0), vinv3, _mm256_add_ps(vone, vu0));
                    const __m256 poly1 = simd_fmadd_ps(_mm256_mul_ps(vu1, vu1), vinv3, _mm256_add_ps(vone, vu1));
                    _mm256_storeu_ps(buf + k, _mm256_and_ps(_mm256_mul_ps(poly0, ve0), nz0));
                    _mm256_storeu_ps(buf + k + 8U, _mm256_and_ps(_mm256_mul_ps(poly1, ve1), nz1));
                }
                for (; k + 8U <= len; k += 8U)
                {
                    const __m256 vu2 = _mm256_mul_ps(_mm256_max_ps(vzero, _mm256_loadu_ps(buf + k)), vc);
                    const __m256 vu = _mm256_sqrt_ps(vu2);
                    const __m256 ve = fast_exp_nonpos_avx2_f32(_mm256_sub_ps(vzero, vu));
                    const __m256 nz = _mm256_cmp_ps(ve, vzero, _CMP_NEQ_UQ);
                    const __m256 vpoly = simd_fmadd_ps(vu2, vinv3, _mm256_add_ps(vone, vu));
                    _mm256_storeu_ps(buf + k, _mm256_and_ps(_mm256_mul_ps(vpoly, ve), nz));
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
                    const __m128 vu = _mm_sqrt_ps(_mm_mul_ps(_mm_max_ps(vzero, _mm_loadu_ps(buf + k)), vc));
                    _mm_storeu_ps(buf + k, fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu)));
                }
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu = _mm_sqrt_ps(_mm_mul_ps(_mm_max_ps(vzero, _mm_loadu_ps(buf + k)), vc));
                    const __m128 ve = fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu));
                    const __m128 nz = _mm_cmpneq_ps(ve, vzero);
                    _mm_storeu_ps(buf + k, _mm_and_ps(_mm_mul_ps(_mm_add_ps(vone, vu), ve), nz));
                }
            }
            else
            {
                for (; k + 4U <= len; k += 4U)
                {
                    const __m128 vu2 = _mm_mul_ps(_mm_max_ps(vzero, _mm_loadu_ps(buf + k)), vc);
                    const __m128 vu = _mm_sqrt_ps(vu2);
                    const __m128 ve = fast_exp_nonpos_sse2_f32(_mm_sub_ps(vzero, vu));
                    const __m128 nz = _mm_cmpneq_ps(ve, vzero);
                    const __m128 vpoly = _mm_add_ps(_mm_add_ps(vone, vu), _mm_mul_ps(vu2, vinv3));
                    _mm_storeu_ps(buf + k, _mm_and_ps(_mm_mul_ps(vpoly, ve), nz));
                }
            }
        }
#endif
        for (; k < len; ++k)
        {
            const float raw = buf[k] < 0.0F ? 0.0F : buf[k];
            const float u2 = raw * scale_sq;
            const float u = std::sqrt(u2);
            const float e = fast_exp_f32(-u);
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

    /// @brief Evaluates the radial Matern correlation `k(u)` over a squared-distance buffer `buf` in place.
    /// @details Overwrites `buf[k]` (containing `s^2 >= 0`) with Matern `nu` evaluated at `u = sqrt(scale_sq * s^2)`,
    /// using `std::exp` under `Accurate` and fused vectorized Estrin minimax evaluation under `FastApprox`.
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
