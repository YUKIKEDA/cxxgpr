// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for batch FastApprox exponential evaluation.
/// @details Inline scalar `eval_kernel_exp` and `eval_kernel_exp_jet` definitions live in `cxxgpr/kernel/exp.hpp`
/// to avoid cross-TU call overhead, while this unit compiles and validates SSE2 batch exponential kernels.
#include "cxxgpr/kernel/exp.hpp"

#include "kernel/exp_simd.hpp"

/// @brief Internal SIMD batch exponential and finiteness verification kernels.
namespace cxxgpr::detail
{
    /// @brief Check whether every binary64 element in a contiguous slice is finite.
    /// @details Scans IEEE-754 exponent bits via 4-way unrolled SSE2 loads on x86-64 and finishes any scalar tail
    /// with `std::isfinite`.
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
#if defined(_M_X64) || defined(__x86_64__)
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
    /// @details Scans IEEE-754 exponent bits across four lanes per load via 4-way unrolled SSE2 on x86-64 and finishes
    /// any scalar tail with `std::isfinite`.
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
#if defined(_M_X64) || defined(__x86_64__)
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
