// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for batch FastApprox exponential evaluation and SIMD finiteness checks.
/// @details Inline scalar `eval_kernel_exp` and `eval_kernel_exp_jet` definitions live in `cxxgpr/kernel/exp.hpp`
/// to avoid cross-TU call overhead, while this unit compiles AVX-512/AVX2/SSE2 batch kernels and finiteness scans.
#include "cxxgpr/kernel/exp.hpp"

#include "kernel/exp_simd.hpp"

/// @brief Internal SIMD batch exponential and finiteness verification kernels.
namespace cxxgpr::detail
{
    /// @brief Check whether every binary64 element in a contiguous slice is finite.
    /// @details Scans IEEE-754 exponent bits via 4-way unrolled AVX-512, AVX2, or SSE2 loads on x86-64 and finishes
    /// any scalar tail with `std::isfinite`.
    /// @param[in] ptr Contiguous input buffer (`nullptr` allowed only when `len == 0`).
    /// @param[in] len Number of elements in the slice.
    /// @return `true` when all elements are finite; `false` on any `NaN` or `+/-Inf`.
    bool all_finite_slice(const double* ptr, std::size_t len) noexcept
    {
        if (len == 0U)
        {
            return true;
        }
        std::size_t i = 0U;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        const __m512i exp_mask = _mm512_set1_epi64(0x7FF0000000000000LL);
        unsigned int bad0 = 0U;
        unsigned int bad1 = 0U;
        for (; i + 32U <= len; i += 32U)
        {
            const __m512i v0 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i));
            const __m512i v1 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i + 8U));
            const __m512i v2 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i + 16U));
            const __m512i v3 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i + 24U));
            bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v0, exp_mask), exp_mask));
            bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v1, exp_mask), exp_mask));
            bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v2, exp_mask), exp_mask));
            bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v3, exp_mask), exp_mask));
        }
        unsigned int bad = bad0 | bad1;
        for (; i + 8U <= len; i += 8U)
        {
            const __m512i v = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i));
            bad |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v, exp_mask), exp_mask));
        }
        if (bad != 0U)
        {
            return false;
        }
#elifdef __AVX2__
        // Exponent bits [62:52] == 0x7FF identify NaN and +/-Inf across 4 binary64 lanes at once.
        const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000LL);
        __m256i bad0 = _mm256_setzero_si256();
        __m256i bad1 = _mm256_setzero_si256();
        for (; i + 16U <= len; i += 16U)
        {
            const __m256i v0 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i));
            const __m256i v1 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 4U));
            const __m256i v2 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 8U));
            const __m256i v3 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 12U));
            bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi64(_mm256_and_si256(v0, exp_mask), exp_mask));
            bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi64(_mm256_and_si256(v1, exp_mask), exp_mask));
            bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi64(_mm256_and_si256(v2, exp_mask), exp_mask));
            bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi64(_mm256_and_si256(v3, exp_mask), exp_mask));
        }
        __m256i bad = _mm256_or_si256(bad0, bad1);
        for (; i + 4U <= len; i += 4U)
        {
            const __m256i v = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i));
            bad = _mm256_or_si256(bad, _mm256_cmpeq_epi64(_mm256_and_si256(v, exp_mask), exp_mask));
        }
        if (_mm256_movemask_epi8(bad) != 0)
        {
            return false;
        }
#elif defined(_M_X64) || defined(__x86_64__)
        // Exponent bits [62:52] == 0x7FF identify NaN and +/-Inf without floating-point exceptions.
        const __m128i exp_mask = _mm_set_epi32(0x7FF00000, 0, 0x7FF00000, 0);
        const __m128i exp_target = _mm_set_epi32(0x7FF00000, -1, 0x7FF00000, -1);
        __m128i bad0 = _mm_setzero_si128();
        __m128i bad1 = _mm_setzero_si128();
        for (; i + 8U <= len; i += 8U)
        {
            const __m128i v0 = _mm_castpd_si128(_mm_loadu_pd(ptr + i));
            const __m128i v1 = _mm_castpd_si128(_mm_loadu_pd(ptr + i + 2U));
            const __m128i v2 = _mm_castpd_si128(_mm_loadu_pd(ptr + i + 4U));
            const __m128i v3 = _mm_castpd_si128(_mm_loadu_pd(ptr + i + 6U));
            bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v0, exp_mask), exp_target));
            bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v1, exp_mask), exp_target));
            bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v2, exp_mask), exp_target));
            bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v3, exp_mask), exp_target));
        }
        __m128i bad = _mm_or_si128(bad0, bad1);
        for (; i + 2U <= len; i += 2U)
        {
            const __m128i v = _mm_castpd_si128(_mm_loadu_pd(ptr + i));
            bad = _mm_or_si128(bad, _mm_cmpeq_epi32(_mm_and_si128(v, exp_mask), exp_target));
        }
        if (_mm_movemask_epi8(bad) != 0)
        {
            return false;
        }
#endif
        for (; i < len; ++i)
        {
            if (!std::isfinite(ptr[i]))
            {
                return false;
            }
        }
        return true;
    }

    /// @brief Check whether every binary32 element in a contiguous slice is finite.
    /// @details Scans IEEE-754 exponent bits via 4-way unrolled AVX-512, AVX2, or SSE2 on x86-64 and finishes any
    /// scalar tail with `std::isfinite`.
    /// @param[in] ptr Contiguous input buffer (`nullptr` allowed only when `len == 0`).
    /// @param[in] len Number of elements in the slice.
    /// @return `true` when all elements are finite; `false` on any `NaN` or `+/-Inf`.
    bool all_finite_slice(const float* ptr, std::size_t len) noexcept
    {
        if (len == 0U)
        {
            return true;
        }
        std::size_t i = 0U;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        const __m512i exp_mask = _mm512_set1_epi32(0x7F800000);
        unsigned int bad0 = 0U;
        unsigned int bad1 = 0U;
        for (; i + 64U <= len; i += 64U)
        {
            const __m512i v0 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i));
            const __m512i v1 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i + 16U));
            const __m512i v2 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i + 32U));
            const __m512i v3 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i + 48U));
            bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v0, exp_mask), exp_mask));
            bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v1, exp_mask), exp_mask));
            bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v2, exp_mask), exp_mask));
            bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v3, exp_mask), exp_mask));
        }
        unsigned int bad = bad0 | bad1;
        for (; i + 16U <= len; i += 16U)
        {
            const __m512i v = _mm512_castps_si512(_mm512_loadu_ps(ptr + i));
            bad |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v, exp_mask), exp_mask));
        }
        if (bad != 0U)
        {
            return false;
        }
#elifdef __AVX2__
        // Exponent bits [30:23] == 0xFF identify NaN and +/-Inf across 8 binary32 lanes at once.
        const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
        __m256i bad0 = _mm256_setzero_si256();
        __m256i bad1 = _mm256_setzero_si256();
        for (; i + 32U <= len; i += 32U)
        {
            const __m256i v0 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i));
            const __m256i v1 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i + 8U));
            const __m256i v2 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i + 16U));
            const __m256i v3 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i + 24U));
            bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi32(_mm256_and_si256(v0, exp_mask), exp_mask));
            bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi32(_mm256_and_si256(v1, exp_mask), exp_mask));
            bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi32(_mm256_and_si256(v2, exp_mask), exp_mask));
            bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi32(_mm256_and_si256(v3, exp_mask), exp_mask));
        }
        __m256i bad = _mm256_or_si256(bad0, bad1);
        for (; i + 8U <= len; i += 8U)
        {
            const __m256i v = _mm256_castps_si256(_mm256_loadu_ps(ptr + i));
            bad = _mm256_or_si256(bad, _mm256_cmpeq_epi32(_mm256_and_si256(v, exp_mask), exp_mask));
        }
        if (_mm256_movemask_epi8(bad) != 0)
        {
            return false;
        }
#elif defined(_M_X64) || defined(__x86_64__)
        // Exponent bits [30:23] == 0xFF identify NaN and +/-Inf across 4 binary32 lanes at once.
        const __m128i exp_mask = _mm_set1_epi32(0x7F800000);
        __m128i bad0 = _mm_setzero_si128();
        __m128i bad1 = _mm_setzero_si128();
        for (; i + 16U <= len; i += 16U)
        {
            const __m128i v0 = _mm_castps_si128(_mm_loadu_ps(ptr + i));
            const __m128i v1 = _mm_castps_si128(_mm_loadu_ps(ptr + i + 4U));
            const __m128i v2 = _mm_castps_si128(_mm_loadu_ps(ptr + i + 8U));
            const __m128i v3 = _mm_castps_si128(_mm_loadu_ps(ptr + i + 12U));
            bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v0, exp_mask), exp_mask));
            bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v1, exp_mask), exp_mask));
            bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v2, exp_mask), exp_mask));
            bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v3, exp_mask), exp_mask));
        }
        __m128i bad = _mm_or_si128(bad0, bad1);
        for (; i + 4U <= len; i += 4U)
        {
            const __m128i v = _mm_castps_si128(_mm_loadu_ps(ptr + i));
            bad = _mm_or_si128(bad, _mm_cmpeq_epi32(_mm_and_si128(v, exp_mask), exp_mask));
        }
        if (_mm_movemask_epi8(bad) != 0)
        {
            return false;
        }
#endif
        for (; i < len; ++i)
        {
            if (!std::isfinite(ptr[i]))
            {
                return false;
            }
        }
        return true;
    }

    template void exp_scaled_nonpos_in_place<double>(double*, std::size_t, double, KernelExp) noexcept;
    template void exp_scaled_nonpos_in_place<float>(float*, std::size_t, float, KernelExp) noexcept;
}  // namespace cxxgpr::detail
