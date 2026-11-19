// SPDX-License-Identifier: Apache-2.0
/// @file finite_simd.hpp
/// @brief Internal 3-tier SIMD finiteness checks shared across core-dependent modules.
/// @details Scans IEEE-754 exponent bits via 4-way unrolled AVX-512, AVX2, or SSE2 loads on x86-64 and finishes
/// any scalar tail with `std::isfinite` so normal, zero, and subnormal values pass while NaN and +/-Inf fail.
#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

/// @brief Internal shared SIMD validation helpers.
namespace cxxgpr::detail
{

    /// @brief Checks whether every entry in a contiguous scalar slice `ptr[0..len)` is finite.
    /// @details Uses branchless 4-way unrolled AVX-512, AVX2, or SSE2 exponent-mask checks on x86-64 with scalar
    /// `std::isfinite` tail handling; preserves subnormal values and returns false on any NaN or +/-Inf.
    /// @tparam T Floating-point storage scalar type (`double` or `float`).
    /// @param[in] ptr Contiguous input buffer (`nullptr` allowed only when `len == 0`).
    /// @param[in] len Number of contiguous elements to inspect.
    /// @return `true` when all elements are finite; `false` on any `NaN` or `+/-Inf`.
    template <typename T>
        requires(std::same_as<T, double> || std::same_as<T, float>)
    [[nodiscard]] inline bool all_finite_simd(const T* ptr, std::size_t len) noexcept
    {
        if (len == 0U)
        {
            return true;
        }
        std::size_t i = 0U;
        if constexpr (std::same_as<T, double>)
        {
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
        }
        else
        {
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
        }
        for (; i < len; ++i)
        {
            if (!std::isfinite(ptr[i]))
            {
                return false;
            }
        }
        return true;
    }

}  // namespace cxxgpr::detail
