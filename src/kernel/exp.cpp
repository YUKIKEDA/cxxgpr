// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for herumi/fmath fast exponential evaluations and SIMD finiteness checks.
/// @details Keeps third-party herumi/fmath headers (`fmath.h`, `old_fmath.hpp`, `xbyak_util.h`) isolated from
/// public headers while exporting `fast_exp_*`, `fast_exp_jet_*`, `fast_exp_scaled_nonpos_in_place`,
/// `fast_matern_s2_in_place`, and `all_finite_slice`.

#include "cxxgpr/kernel/exp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

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

#ifdef __AVX2__

        /// @brief Evaluates `a * b + c` across 8 binary32 lanes using AVX2 FMA3 when enabled.
        /// @param[in] a First multiplicand vector.
        /// @param[in] b Second multiplicand vector.
        /// @param[in] c Addend vector.
        /// @return Lane-wise `a * b + c`.
        [[nodiscard]] inline __m256 exp_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
        {
#    if defined(__FMA__) || defined(_MSC_VER)
            return _mm256_fmadd_ps(a, b, c);
#    else
            return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#    endif
        }

        /// @brief Evaluates `a * b + c` across 4 binary64 lanes using AVX2 FMA3 when enabled.
        /// @param[in] a First multiplicand vector.
        /// @param[in] b Second multiplicand vector.
        /// @param[in] c Addend vector.
        /// @return Lane-wise `a * b + c`.
        [[nodiscard]] inline __m256d exp_fmadd_pd(__m256d a, __m256d b, __m256d c) noexcept
        {
#    if defined(__FMA__) || defined(_MSC_VER)
            return _mm256_fmadd_pd(a, b, c);
#    else
            return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#    endif
        }

        /// @brief Chunk size tuned for L1 data cache residency during vectorized kernel evaluation.
        constexpr std::size_t k_exp_chunk = 256U;

        /// @brief Applies per-element underflow zeroing on `buf[0..count)` when `uf_masks` indicates `x <= min_x`.
        /// @tparam Width SIMD vector width in elements (4 for binary64, 8 for binary32).
        /// @tparam T Scalar type (`double` or `float`).
        /// @param[in,out] buf Target buffer containing evaluated exponentials.
        /// @param[in] count Number of valid elements in `buf`.
        /// @param[in] uf_masks Per-vector bitmask storing lanes whose input was `<= min_x` or NaN.
        template <std::size_t Width, typename T>
        inline void apply_underflow_masks(
            T* buf, std::size_t count, const std::array<std::uint8_t, k_exp_chunk / Width>& uf_masks
        ) noexcept
        {
            const std::size_t n_vecs = (count + Width - 1U) / Width;
            for (std::size_t v = 0U; v < n_vecs; ++v)
            {
                unsigned int m = uf_masks[v];
                if (m == 0U)
                {
                    continue;
                }
                const std::size_t base = v * Width;
                const std::size_t lane_end = std::min(Width, count - base);
                for (std::size_t lane = 0U; lane < lane_end; ++lane)
                {
                    if ((m & (1U << lane)) != 0U)
                    {
                        buf[base + lane] = T{0};
                    }
                }
            }
        }

#endif  // __AVX2__

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

    /// @brief Evaluates `out[k] = fast_exp(neg_scale * max(out[k], 0))` in place for binary64.
    /// @param[in,out] buf Mutable contiguous buffer of non-negative distances overwritten with kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] neg_scale Non-positive multiplier applied before exponentiation.
    void fast_exp_scaled_nonpos_in_place(double* buf, std::size_t len, double neg_scale) noexcept
    {
#ifdef __AVX2__
        const __m256d vzero = _mm256_setzero_pd();
        const __m256d vneg = _mm256_set1_pd(neg_scale);
        const __m256d vmin = _mm256_set1_pd(k_expd_min_x);

        for (std::size_t base = 0U; base < len; base += k_exp_chunk)
        {
            const std::size_t count = std::min(k_exp_chunk, len - base);
            double* const dst = buf + base;
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst);

            // Execute completely in-place when dst is 32-byte aligned and covers at least one 4-wide vector.
            if ((addr & std::uintptr_t{31}) == 0U && count >= 4U)
            {
                const std::size_t vec_count = count & ~std::size_t{3U};
                std::array<std::uint8_t, k_exp_chunk / 4U> uf_masks{};
                int any_uf = 0;
                for (std::size_t i = 0U; i < vec_count; i += 4U)
                {
                    const __m256d vx = _mm256_load_pd(dst + i);
                    const __m256d vd = _mm256_max_pd(vzero, vx);
                    const __m256d varg = _mm256_mul_pd(vd, vneg);
                    const int uf = _mm256_movemask_pd(_mm256_cmp_pd(varg, vmin, _CMP_LE_OQ));
                    uf_masks[i / 4U] = static_cast<std::uint8_t>(uf);
                    any_uf |= uf;
                    _mm256_store_pd(dst + i, varg);
                }
                fmath::expd_v(dst, vec_count);
                if (any_uf != 0)
                {
                    apply_underflow_masks<4U>(dst, vec_count, uf_masks);
                }
                const std::size_t rem = count - vec_count;
                if (rem > 0U)
                {
                    alignas(32) std::array<double, 4U> tail{0.0, 0.0, 0.0, 0.0};
                    int tail_uf = 0;
                    for (std::size_t r = 0U; r < rem; ++r)
                    {
                        const double d = dst[vec_count + r] < 0.0 ? 0.0 : dst[vec_count + r];
                        const double arg = d * neg_scale;
                        if (!(arg > k_expd_min_x))
                        {
                            tail_uf |= (1 << static_cast<int>(r));
                        }
                        tail[r] = arg;
                    }
                    fmath::expd_v(tail.data(), 4U);
                    for (std::size_t r = 0U; r < rem; ++r)
                    {
                        dst[vec_count + r] = ((tail_uf & (1 << static_cast<int>(r))) != 0) ? 0.0 : tail[r];
                    }
                }
                continue;
            }

            // Stage through a 32-byte aligned L1 scratch buffer when dst is unaligned or shorter than 4 elements.
            alignas(32) std::array<double, k_exp_chunk> arg_buf;
            std::array<std::uint8_t, k_exp_chunk / 4U> uf_masks{};
            int any_uf = 0;
            std::size_t i = 0U;
            for (; i + 4U <= count; i += 4U)
            {
                const __m256d vx = _mm256_loadu_pd(dst + i);
                const __m256d vd = _mm256_max_pd(vzero, vx);
                const __m256d varg = _mm256_mul_pd(vd, vneg);
                const int uf = _mm256_movemask_pd(_mm256_cmp_pd(varg, vmin, _CMP_LE_OQ));
                uf_masks[i / 4U] = static_cast<std::uint8_t>(uf);
                any_uf |= uf;
                _mm256_store_pd(arg_buf.data() + i, varg);
            }
            const std::size_t aligned_count = (count + 3U) & ~std::size_t{3U};
            if (i < aligned_count)
            {
                int rem_uf = 0;
                for (std::size_t k = i; k < count; ++k)
                {
                    const double d = dst[k] < 0.0 ? 0.0 : dst[k];
                    const double arg = d * neg_scale;
                    if (!(arg > k_expd_min_x))
                    {
                        rem_uf |= (1 << static_cast<int>(k - i));
                    }
                    arg_buf[k] = arg;
                }
                for (std::size_t k = count; k < aligned_count; ++k)
                {
                    arg_buf[k] = 0.0;
                }
                uf_masks[i / 4U] = static_cast<std::uint8_t>(rem_uf);
                any_uf |= rem_uf;
            }
            fmath::expd_v(arg_buf.data(), aligned_count);
            if (any_uf != 0)
            {
                apply_underflow_masks<4U>(arg_buf.data(), count, uf_masks);
            }
            std::copy_n(arg_buf.data(), count, dst);
        }
#else
        for (std::size_t k = 0U; k < len; ++k)
        {
            const double d = buf[k] < 0.0 ? 0.0 : buf[k];
            buf[k] = fast_exp_f64(d * neg_scale);
        }
#endif
    }

    /// @brief Evaluates `out[k] = fast_exp(neg_scale * max(out[k], 0))` in place for binary32.
    /// @param[in,out] buf Mutable contiguous buffer of non-negative distances overwritten with kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] neg_scale Non-positive multiplier applied before exponentiation.
    void fast_exp_scaled_nonpos_in_place(float* buf, std::size_t len, float neg_scale) noexcept
    {
#ifdef __AVX2__
        if (cpu_has_avx2_fma())
        {
            const __m256 vzero = _mm256_setzero_ps();
            const __m256 vneg = _mm256_set1_ps(neg_scale);
            const __m256 vmin = _mm256_set1_ps(k_expf_min_x);

            for (std::size_t base = 0U; base < len; base += k_exp_chunk)
            {
                const std::size_t count = std::min(k_exp_chunk, len - base);
                float* const dst = buf + base;
                std::array<std::uint8_t, k_exp_chunk / 8U> uf_masks{};
                int any_uf = 0;

                std::size_t i = 0U;
                for (; i + 8U <= count; i += 8U)
                {
                    const __m256 vx = _mm256_loadu_ps(dst + i);
                    const __m256 vd = _mm256_max_ps(vzero, vx);
                    const __m256 varg = _mm256_mul_ps(vd, vneg);
                    const int uf = _mm256_movemask_ps(_mm256_cmp_ps(varg, vmin, _CMP_LE_OQ));
                    uf_masks[i / 8U] = static_cast<std::uint8_t>(uf);
                    any_uf |= uf;
                    _mm256_storeu_ps(dst + i, varg);
                }
                if (i < count)
                {
                    int rem_uf = 0;
                    for (std::size_t k = i; k < count; ++k)
                    {
                        const float d = dst[k] < 0.0F ? 0.0F : dst[k];
                        const float arg = d * neg_scale;
                        if (!(arg > k_expf_min_x))
                        {
                            rem_uf |= (1 << static_cast<int>(k - i));
                        }
                        dst[k] = arg;
                    }
                    uf_masks[i / 8U] = static_cast<std::uint8_t>(rem_uf);
                    any_uf |= rem_uf;
                }

                fmath_expf_v(dst, dst, count);
                if (any_uf != 0)
                {
                    apply_underflow_masks<8U>(dst, count, uf_masks);
                }
            }
            return;
        }
#endif
        for (std::size_t k = 0U; k < len; ++k)
        {
            const float d = buf[k] < 0.0F ? 0.0F : buf[k];
            buf[k] = fast_exp_f32(d * neg_scale);
        }
    }

    /// @brief Evaluates radial Matern correlation `k(u)` over a squared-distance binary64 slice in place.
    /// @param[in,out] buf Buffer containing non-negative `s^2` entries overwritten with Matern kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] scale_sq Positive Matern factor `2 * nu` (`1`, `3`, or `5`).
    /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
    void fast_matern_s2_in_place(double* buf, std::size_t len, double scale_sq, MaternNu nu) noexcept
    {
#ifdef __AVX2__
        const __m256d vzero = _mm256_setzero_pd();
        const __m256d vscale = _mm256_set1_pd(scale_sq);
        const __m256d vone = _mm256_set1_pd(1.0);
        const __m256d vinv3 = _mm256_set1_pd(1.0 / 3.0);
        const __m256d vmin = _mm256_set1_pd(k_expd_min_x);

        alignas(32) std::array<double, k_exp_chunk> u_buf;
        alignas(32) std::array<double, k_exp_chunk> e_buf;

        for (std::size_t base = 0U; base < len; base += k_exp_chunk)
        {
            const std::size_t count = std::min(k_exp_chunk, len - base);
            double* const dst = buf + base;
            std::array<std::uint8_t, k_exp_chunk / 4U> uf_masks{};
            int any_uf = 0;

            std::size_t i = 0U;
            for (; i + 4U <= count; i += 4U)
            {
                const __m256d raw = _mm256_max_pd(vzero, _mm256_loadu_pd(dst + i));
                const __m256d vu2 = _mm256_mul_pd(raw, vscale);
                const __m256d vu = _mm256_sqrt_pd(vu2);
                const __m256d vneg_u = _mm256_sub_pd(vzero, vu);
                const int uf = _mm256_movemask_pd(_mm256_cmp_pd(vneg_u, vmin, _CMP_LE_OQ));
                uf_masks[i / 4U] = static_cast<std::uint8_t>(uf);
                any_uf |= uf;
                _mm256_store_pd(u_buf.data() + i, vu);
                _mm256_store_pd(e_buf.data() + i, vneg_u);
            }
            const std::size_t aligned_count = (count + 3U) & ~std::size_t{3U};
            if (i < aligned_count)
            {
                int rem_uf = 0;
                for (std::size_t k = i; k < count; ++k)
                {
                    const double raw = dst[k] < 0.0 ? 0.0 : dst[k];
                    const double u = std::sqrt(raw * scale_sq);
                    const double neg_u = -u;
                    if (!(neg_u > k_expd_min_x))
                    {
                        rem_uf |= (1 << static_cast<int>(k - i));
                    }
                    u_buf[k] = u;
                    e_buf[k] = neg_u;
                }
                for (std::size_t k = count; k < aligned_count; ++k)
                {
                    u_buf[k] = 0.0;
                    e_buf[k] = 0.0;
                }
                uf_masks[i / 4U] = static_cast<std::uint8_t>(rem_uf);
                any_uf |= rem_uf;
            }

            fmath::expd_v(e_buf.data(), aligned_count);
            if (any_uf != 0)
            {
                apply_underflow_masks<4U>(e_buf.data(), count, uf_masks);
            }

            if (nu == MaternNu::OneHalf)
            {
                std::copy_n(e_buf.data(), count, dst);
                continue;
            }

            std::size_t j = 0U;
            if (nu == MaternNu::ThreeHalves)
            {
                for (; j + 4U <= count; j += 4U)
                {
                    const __m256d vu = _mm256_load_pd(u_buf.data() + j);
                    const __m256d ve = _mm256_load_pd(e_buf.data() + j);
                    const __m256d vmask = _mm256_cmp_pd(ve, vzero, _CMP_GT_OQ);
                    const __m256d vprod = _mm256_mul_pd(_mm256_add_pd(vone, vu), ve);
                    _mm256_storeu_pd(dst + j, _mm256_and_pd(vmask, vprod));
                }
                for (; j < count; ++j)
                {
                    const double e = e_buf[j];
                    dst[j] = (e > 0.0) ? ((1.0 + u_buf[j]) * e) : 0.0;
                }
            }
            else
            {
                for (; j + 4U <= count; j += 4U)
                {
                    const __m256d vu = _mm256_load_pd(u_buf.data() + j);
                    const __m256d ve = _mm256_load_pd(e_buf.data() + j);
                    const __m256d vmask = _mm256_cmp_pd(ve, vzero, _CMP_GT_OQ);
                    const __m256d vu2 = _mm256_mul_pd(vu, vu);
                    const __m256d vpoly = exp_fmadd_pd(vu2, vinv3, _mm256_add_pd(vone, vu));
                    const __m256d vprod = _mm256_mul_pd(vpoly, ve);
                    _mm256_storeu_pd(dst + j, _mm256_and_pd(vmask, vprod));
                }
                for (; j < count; ++j)
                {
                    const double e = e_buf[j];
                    const double u = u_buf[j];
                    dst[j] = (e > 0.0) ? ((1.0 + u + ((u * u) / 3.0)) * e) : 0.0;
                }
            }
        }
#else
        for (std::size_t k = 0U; k < len; ++k)
        {
            const double raw = buf[k] < 0.0 ? 0.0 : buf[k];
            const double u2 = raw * scale_sq;
            const double u = std::sqrt(u2);
            const double e = fast_exp_f64(-u);
            if (nu == MaternNu::OneHalf)
            {
                buf[k] = e;
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                buf[k] = (e > 0.0) ? ((1.0 + u) * e) : 0.0;
            }
            else
            {
                buf[k] = (e > 0.0) ? ((1.0 + u + (u2 / 3.0)) * e) : 0.0;
            }
        }
#endif
    }

    /// @brief Evaluates radial Matern correlation `k(u)` over a squared-distance binary32 slice in place.
    /// @param[in,out] buf Buffer containing non-negative `s^2` entries overwritten with Matern kernel values.
    /// @param[in] len Number of contiguous elements in `buf`.
    /// @param[in] scale_sq Positive Matern factor `2 * nu` (`1`, `3`, or `5`).
    /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
    void fast_matern_s2_in_place(float* buf, std::size_t len, float scale_sq, MaternNu nu) noexcept
    {
#ifdef __AVX2__
        if (cpu_has_avx2_fma())
        {
            const __m256 vzero = _mm256_setzero_ps();
            const __m256 vscale = _mm256_set1_ps(scale_sq);
            const __m256 vone = _mm256_set1_ps(1.0F);
            const __m256 vinv3 = _mm256_set1_ps(1.0F / 3.0F);
            const __m256 vmin = _mm256_set1_ps(k_expf_min_x);

            alignas(32) std::array<float, k_exp_chunk> u_buf;
            alignas(32) std::array<float, k_exp_chunk> e_buf;

            for (std::size_t base = 0U; base < len; base += k_exp_chunk)
            {
                const std::size_t count = std::min(k_exp_chunk, len - base);
                float* const dst = buf + base;
                std::array<std::uint8_t, k_exp_chunk / 8U> uf_masks{};
                int any_uf = 0;

                if (nu == MaternNu::OneHalf)
                {
                    std::size_t i = 0U;
                    for (; i + 8U <= count; i += 8U)
                    {
                        const __m256 raw = _mm256_max_ps(vzero, _mm256_loadu_ps(dst + i));
                        const __m256 vu = _mm256_sqrt_ps(_mm256_mul_ps(raw, vscale));
                        const __m256 vneg_u = _mm256_sub_ps(vzero, vu);
                        const int uf = _mm256_movemask_ps(_mm256_cmp_ps(vneg_u, vmin, _CMP_LE_OQ));
                        uf_masks[i / 8U] = static_cast<std::uint8_t>(uf);
                        any_uf |= uf;
                        _mm256_storeu_ps(dst + i, vneg_u);
                    }
                    if (i < count)
                    {
                        int rem_uf = 0;
                        for (std::size_t k = i; k < count; ++k)
                        {
                            const float raw = dst[k] < 0.0F ? 0.0F : dst[k];
                            const float neg_u = -std::sqrt(raw * scale_sq);
                            if (!(neg_u > k_expf_min_x))
                            {
                                rem_uf |= (1 << static_cast<int>(k - i));
                            }
                            dst[k] = neg_u;
                        }
                        uf_masks[i / 8U] = static_cast<std::uint8_t>(rem_uf);
                        any_uf |= rem_uf;
                    }
                    fmath_expf_v(dst, dst, count);
                    if (any_uf != 0)
                    {
                        apply_underflow_masks<8U>(dst, count, uf_masks);
                    }
                    continue;
                }

                std::size_t i = 0U;
                for (; i + 8U <= count; i += 8U)
                {
                    const __m256 raw = _mm256_max_ps(vzero, _mm256_loadu_ps(dst + i));
                    const __m256 vu = _mm256_sqrt_ps(_mm256_mul_ps(raw, vscale));
                    const __m256 vneg_u = _mm256_sub_ps(vzero, vu);
                    const int uf = _mm256_movemask_ps(_mm256_cmp_ps(vneg_u, vmin, _CMP_LE_OQ));
                    uf_masks[i / 8U] = static_cast<std::uint8_t>(uf);
                    any_uf |= uf;
                    _mm256_store_ps(u_buf.data() + i, vu);
                    _mm256_store_ps(e_buf.data() + i, vneg_u);
                }
                if (i < count)
                {
                    int rem_uf = 0;
                    for (std::size_t k = i; k < count; ++k)
                    {
                        const float raw = dst[k] < 0.0F ? 0.0F : dst[k];
                        const float u = std::sqrt(raw * scale_sq);
                        const float neg_u = -u;
                        if (!(neg_u > k_expf_min_x))
                        {
                            rem_uf |= (1 << static_cast<int>(k - i));
                        }
                        u_buf[k] = u;
                        e_buf[k] = neg_u;
                    }
                    uf_masks[i / 8U] = static_cast<std::uint8_t>(rem_uf);
                    any_uf |= rem_uf;
                }

                fmath_expf_v(e_buf.data(), e_buf.data(), count);
                if (any_uf != 0)
                {
                    apply_underflow_masks<8U>(e_buf.data(), count, uf_masks);
                }

                std::size_t j = 0U;
                if (nu == MaternNu::ThreeHalves)
                {
                    for (; j + 8U <= count; j += 8U)
                    {
                        const __m256 vu = _mm256_load_ps(u_buf.data() + j);
                        const __m256 ve = _mm256_load_ps(e_buf.data() + j);
                        const __m256 vmask = _mm256_cmp_ps(ve, vzero, _CMP_GT_OQ);
                        const __m256 vprod = _mm256_mul_ps(_mm256_add_ps(vone, vu), ve);
                        _mm256_storeu_ps(dst + j, _mm256_and_ps(vmask, vprod));
                    }
                    for (; j < count; ++j)
                    {
                        const float e = e_buf[j];
                        dst[j] = (e > 0.0F) ? ((1.0F + u_buf[j]) * e) : 0.0F;
                    }
                }
                else
                {
                    for (; j + 8U <= count; j += 8U)
                    {
                        const __m256 vu = _mm256_load_ps(u_buf.data() + j);
                        const __m256 ve = _mm256_load_ps(e_buf.data() + j);
                        const __m256 vmask = _mm256_cmp_ps(ve, vzero, _CMP_GT_OQ);
                        const __m256 vu2 = _mm256_mul_ps(vu, vu);
                        const __m256 vpoly = exp_fmadd_ps(vu2, vinv3, _mm256_add_ps(vone, vu));
                        const __m256 vprod = _mm256_mul_ps(vpoly, ve);
                        _mm256_storeu_ps(dst + j, _mm256_and_ps(vmask, vprod));
                    }
                    for (; j < count; ++j)
                    {
                        const float e = e_buf[j];
                        const float u = u_buf[j];
                        dst[j] = (e > 0.0F) ? ((1.0F + u + ((u * u) / 3.0F)) * e) : 0.0F;
                    }
                }
            }
            return;
        }
#endif
        for (std::size_t k = 0U; k < len; ++k)
        {
            const float raw = buf[k] < 0.0F ? 0.0F : buf[k];
            const float u2 = raw * scale_sq;
            const float u = std::sqrt(u2);
            const float e = fast_exp_f32(-u);
            if (nu == MaternNu::OneHalf)
            {
                buf[k] = e;
            }
            else if (nu == MaternNu::ThreeHalves)
            {
                buf[k] = (e > 0.0F) ? ((1.0F + u) * e) : 0.0F;
            }
            else
            {
                buf[k] = (e > 0.0F) ? ((1.0F + u + (u2 / 3.0F)) * e) : 0.0F;
            }
        }
    }

}  // namespace cxxgpr::detail
