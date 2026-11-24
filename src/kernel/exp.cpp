// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for herumi/fmath fast exponential evaluations and SIMD finiteness checks.
/// @details Keeps third-party herumi/fmath headers (`fmath.h`, `old_fmath.hpp`, `xbyak_util.h`) isolated from
/// public headers while exporting `fast_exp_*`, `fast_exp_jet_*`, `fast_exp_nonpos_slice_*`, and `all_finite_slice`.

#include "cxxgpr/kernel/exp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "core/finite_simd.hpp"
#include "fmath.h"
#include "kernel/exp_simd.hpp"
#include "old_fmath.hpp"

#if defined(CXXGPR_HAS_FMATH_ASM) && CXXGPR_HAS_FMATH_ASM
#    define XBYAK_ONLY_CLASS_CPU
#    include "xbyak_util.h"
#endif

namespace cxxgpr::detail
{

    namespace
    {

        /// @brief Lower finite argument threshold for double-precision `fmath::expd`.
        constexpr double k_expd_min_x = -708.39641853226408;
        /// @brief Upper finite argument threshold before double-precision overflow.
        constexpr double k_expd_max_x = 709.78271289338397;
        /// @brief Lower finite argument threshold for single-precision `fmath_expf_v`.
        constexpr float k_expf_min_x = -87.336544F;
        /// @brief Upper finite argument threshold before single-precision overflow.
        constexpr float k_expf_max_x = 88.722839F;

        /// @brief Detects once at runtime whether the CPU supports both AVX2 and FMA.
        /// @return True if AVX2 and FMA are available, false otherwise.
        [[nodiscard]] bool cpu_has_avx2_fma() noexcept
        {
#if defined(CXXGPR_HAS_FMATH_ASM) && CXXGPR_HAS_FMATH_ASM
            static const bool supported = []() noexcept
            {
                const Xbyak::util::Cpu cpu;
                return cpu.has(Xbyak::util::Cpu::tAVX2 | Xbyak::util::Cpu::tFMA);
            }();
            return supported;
#else
            return false;
#endif
        }

    }  // namespace

    /// @brief Checks whether all `len` double elements starting at `ptr` are finite.
    /// @param[in] ptr Contiguous input buffer pointer.
    /// @param[in] len Number of double elements to check.
    /// @return True if every element is finite, false otherwise.
    bool all_finite_slice(const double* ptr, std::size_t len) noexcept
    {
        return all_finite_simd(ptr, len);
    }

    /// @brief Checks whether all `len` float elements starting at `ptr` are finite.
    /// @param[in] ptr Contiguous input buffer pointer.
    /// @param[in] len Number of float elements to check.
    /// @return True if every element is finite, false otherwise.
    bool all_finite_slice(const float* ptr, std::size_t len) noexcept
    {
        return all_finite_simd(ptr, len);
    }

    /// @brief Evaluates approximate `exp(x)` in double precision with IEEE-754 boundary guards.
    /// @param[in] x Scalar input argument.
    /// @return Approximated exponential value preserving NaN, 0, and infinity limits.
    double fast_exp_f64(double x) noexcept
    {
        if (std::isnan(x))
        {
            return x;
        }
        if (x == 0.0)
        {
            return 1.0;
        }
        if (x <= k_expd_min_x)
        {
            return 0.0;
        }
        if (x >= k_expd_max_x)
        {
            return std::numeric_limits<double>::infinity();
        }
        return fmath::expd(x);
    }

    /// @brief Evaluates approximate `exp(x)` in single precision with AVX2+FMA dispatch and scalar fallback.
    /// @param[in] x Scalar input argument.
    /// @return Approximated exponential value preserving NaN, 0, and infinity limits.
    float fast_exp_f32(float x) noexcept
    {
        if (std::isnan(x))
        {
            return x;
        }
        if (x == 0.0F)
        {
            return 1.0F;
        }
        if (x <= k_expf_min_x)
        {
            return 0.0F;
        }
        if (x >= k_expf_max_x)
        {
            return std::numeric_limits<float>::infinity();
        }
        if (cpu_has_avx2_fma())
        {
            float out = 0.0F;
            fmath_expf_v(&out, &x, 1U);
            return out;
        }
        return static_cast<float>(fmath::expd(static_cast<double>(x)));
    }

    /// @brief Evaluates approximate `exp(x)` and its first two derivatives in double precision.
    /// @param[in] x Scalar input argument.
    /// @return `ExpJet<double>` populated consistently with `fast_exp_f64(x)`.
    ExpJet<double> fast_exp_jet_f64(double x) noexcept
    {
        const double e = fast_exp_f64(x);
        return {.v = e, .d1 = e, .d2 = e};
    }

    /// @brief Evaluates approximate `exp(x)` and its first two derivatives in single precision.
    /// @param[in] x Scalar input argument.
    /// @return `ExpJet<float>` populated consistently with `fast_exp_f32(x)`.
    ExpJet<float> fast_exp_jet_f32(float x) noexcept
    {
        const float e = fast_exp_f32(x);
        return {.v = e, .d1 = e, .d2 = e};
    }

    /// @brief Evaluates in-place vectorized `exp(x)` over a double buffer on `(k_expd_min_x, 0)` runs.
    /// @details Aligns to 32-byte boundaries before calling `fmath::expd_v` on 4-wide chunks.
    /// @param[in,out] buf Contiguous double buffer overwritten in place.
    /// @param[in] len Number of elements in `buf`.
    void fast_exp_nonpos_slice_f64(double* buf, std::size_t len) noexcept
    {
        std::size_t i = 0U;
        while (i < len)
        {
            // Advance linearly with scalar path until 32-byte aligned and inside fmath::expd_v domain.
            if (((reinterpret_cast<std::uintptr_t>(buf + i) & std::uintptr_t{31}) != 0U) || !(buf[i] > k_expd_min_x)
                || !(buf[i] < 0.0))
            {
                buf[i] = fast_exp_f64(buf[i]);
                ++i;
                continue;
            }
            std::size_t j = i;
            while (j < len && buf[j] > k_expd_min_x && buf[j] < 0.0)
            {
                ++j;
            }
            const std::size_t vec_len = (j - i) & ~std::size_t{3};
            if (vec_len >= 4U)
            {
                fmath::expd_v(buf + i, vec_len);
                i += vec_len;
            }
            else
            {
                while (i < j)
                {
                    buf[i] = fast_exp_f64(buf[i]);
                    ++i;
                }
            }
        }
    }

    /// @brief Evaluates in-place vectorized `exp(x)` over a float buffer using `fmath_expf_v` when supported.
    /// @param[in,out] buf Contiguous float buffer overwritten in place.
    /// @param[in] len Number of elements in `buf`.
    void fast_exp_nonpos_slice_f32(float* buf, std::size_t len) noexcept
    {
        if (cpu_has_avx2_fma())
        {
            std::size_t i = 0U;
            while (i < len)
            {
                if (!(buf[i] > k_expf_min_x) || !(buf[i] < 0.0F))
                {
                    buf[i] = fast_exp_f32(buf[i]);
                    ++i;
                    continue;
                }
                std::size_t j = i;
                while (j < len && buf[j] > k_expf_min_x && buf[j] < 0.0F)
                {
                    ++j;
                }
                const std::size_t seg_len = j - i;
                if (seg_len >= 16U)
                {
                    fmath_expf_v(buf + i, buf + i, seg_len);
                    i = j;
                }
                else
                {
                    while (i < j)
                    {
                        buf[i] = fast_exp_f32(buf[i]);
                        ++i;
                    }
                }
            }
            return;
        }
        for (std::size_t i = 0U; i < len; ++i)
        {
            buf[i] = fast_exp_f32(buf[i]);
        }
    }

}  // namespace cxxgpr::detail
