// SPDX-License-Identifier: Apache-2.0
/// @file exp.hpp
/// @brief Kernel exponential evaluation selector, derivative jet, and full-accuracy range-reduced approximations.
/// @details Declares KernelExp, ExpJet<T>, and inline eval_kernel_exp() / eval_kernel_exp_jet() entry points
/// dispatching KernelExp::Accurate to std::exp and KernelExp::FastApprox to built-in 1-2 ULP range-reduced
/// Estrin minimax polynomial evaluations using Cody-Waite subtraction and IEEE-754 magic-bias bit manipulation.

#pragma once

#include "cxxgpr/precision/policy.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Selects exact vs fast-approximate exponential evaluation inside built-in kernels.
    /// @details Accurate evaluates std::exp in the active scalar type; FastApprox uses built-in vectorized
    /// 1-2 ULP minimax polynomial approximation while preserving parameter transforms and Custom leaves.
    enum class KernelExp : std::uint8_t
    {
        /// @brief Standard library std::exp in the storage scalar (default).
        Accurate,
        /// @brief Vectorized 1-2 ULP Estrin minimax exponential approximation in the storage scalar.
        FastApprox,
    };

    /// @brief Zeroth, first, and second derivatives wrt scalar argument x for exp(x).
    /// @details Holds function value `v = f(x)`, first derivative `d1 = df/dx`, and second derivative
    /// `d2 = d^2f/dx^2` paired consistently for exact MLL gradients and Hessians under KernelExp::FastApprox.
    /// @tparam T Storage floating-point scalar type satisfying KernelScalarConcept.
    template <KernelScalarConcept T> struct ExpJet
    {
        /// @brief Function value f(x).
        T v = T{0};
        /// @brief First derivative f'(x) wrt x.
        T d1 = T{0};
        /// @brief Second derivative f''(x) wrt x.
        T d2 = T{0};
    };

    /// @brief Internal implementation helpers for inline full-accuracy fast exponential approximation.
    namespace detail
    {

        /// @brief Compile-time pinned degree-11 minimax polynomial coefficients on `[-ln(2)/2, ln(2)/2]` in binary64.
        inline constexpr std::array<double, 12U> k_fast_exp_poly_f64{
            1.0,
            1.0,
            5.00000000000000111e-01,
            1.66666666666666713e-01,
            4.16666666666241567e-02,
            8.33333333332614105e-03,
            1.38888889171985878e-03,
            1.98412698748021734e-04,
            2.48015213198796721e-05,
            2.75572554238798413e-06,
            2.76200769161133283e-07,
            2.51052070646442396e-08,
        };

        /// @brief Compile-time pinned degree-6 minimax polynomial coefficients on `[-ln(2)/2, ln(2)/2]` in binary32.
        inline constexpr std::array<float, 7U> k_fast_exp_poly_f32{
            1.0F,
            1.0F,
            5.000000000e-01F,
            1.666657627e-01F,
            4.166646674e-02F,
            8.363175206e-03F,
            1.393364393e-03F,
        };

        /// @brief High part of `ln(2)` in binary64 with 11 trailing zero mantissa bits so `n * hi` is exact for `|n| <
        /// 2048`.
        inline constexpr double k_fast_exp_ln2_hi_f64 = std::bit_cast<double>(0x3fe62e42fefa3800ULL);
        /// @brief Low correction part of `ln(2)` in binary64 paired with `k_fast_exp_ln2_hi_f64`.
        inline constexpr double k_fast_exp_ln2_lo_f64 = std::bit_cast<double>(0x3d2ef35793c76730ULL);
        /// @brief Single-precision `1 / ln(2)` multiplier for integer-power range reduction.
        inline constexpr float k_fast_exp_log2e_f32 = std::bit_cast<float>(0x3fb8aa3bU);
        /// @brief High part of `ln(2)` in binary32 with 8 trailing zero mantissa bits so `n * hi` is exact for `|n| <
        /// 256`.
        inline constexpr float k_fast_exp_ln2_hi_f32 = std::bit_cast<float>(0x3f317200U);
        /// @brief Low correction part of `ln(2)` in binary32 paired with `k_fast_exp_ln2_hi_f32`.
        inline constexpr float k_fast_exp_ln2_lo_f32 = std::bit_cast<float>(0x35bfbe8eU);

        /// @brief Evaluates full-accuracy (<= 2 ULP) fast approximate `exp(x)` in binary64.
        /// @details Uses Cody-Waite range reduction, IEEE-754 magic-bias scaling `2^n`, and a pinned degree-11
        /// minimax polynomial evaluated via Estrin's binary tree on `[-708.0, 709.0]`, handling extremes cleanly.
        /// @param[in] x Scalar exponent in binary64 precision.
        /// @return Approximate `exp(x)` preserving exact `1.0` at `0.0`, underflow to `0.0`, overflow to `+inf`, and
        /// `NaN`.
        [[nodiscard]] inline double fast_exp_f64(double x) noexcept
        {
            if (x >= -708.0 && x <= 709.0)
            {
                constexpr double k_magic = 6755399441055744.0;
                const double d = std::fma(x, std::numbers::log2e, k_magic);
                const double n = d - k_magic;
                const double g = std::fma(-n, k_fast_exp_ln2_lo_f64, std::fma(-n, k_fast_exp_ln2_hi_f64, x));

                const double g2 = g * g;
                const double g4 = g2 * g2;
                const double g8 = g4 * g4;

                const auto& c = k_fast_exp_poly_f64;
                const double p01 = 1.0 + g;
                const double p23 = std::fma(c[3], g, c[2]);
                const double p45 = std::fma(c[5], g, c[4]);
                const double p67 = std::fma(c[7], g, c[6]);
                const double p89 = std::fma(c[9], g, c[8]);
                const double p_ab = std::fma(c[11], g, c[10]);

                const double p03 = std::fma(p23, g2, p01);
                const double p47 = std::fma(p67, g2, p45);
                const double p8b = std::fma(p_ab, g2, p89);
                const double p07 = std::fma(p47, g4, p03);
                const double vp = std::fma(p8b, g8, p07);

                const std::uint64_t exp_bits = (std::bit_cast<std::uint64_t>(d) + 1023ULL) << 52U;
                return vp * std::bit_cast<double>(exp_bits);
            }
            if (std::isnan(x))
            {
                return x;
            }
            if (x < -746.0)
            {
                return 0.0;
            }
            if (x > 709.78271289338397)
            {
                return std::numeric_limits<double>::infinity();
            }
            return std::exp(x);
        }

        /// @brief Evaluates full-accuracy (<= 2 ULP) fast approximate `exp(x)` in binary32.
        /// @details Uses Cody-Waite range reduction, IEEE-754 magic-bias scaling `2^n`, and a pinned degree-6
        /// minimax polynomial evaluated via Estrin's binary tree on `[-87.0F, 88.0F]`, handling extremes cleanly.
        /// @param[in] x Scalar exponent in binary32 precision.
        /// @return Approximate `exp(x)` preserving exact `1.0F` at `0.0F`, underflow to `0.0F`, overflow to `+inf`, and
        /// `NaN`.
        [[nodiscard]] inline float fast_exp_f32(float x) noexcept
        {
            if (x >= -87.0F && x <= 88.0F)
            {
                constexpr float k_magic = 12582912.0F;
                const float d = std::fma(x, k_fast_exp_log2e_f32, k_magic);
                const float n = d - k_magic;
                const float g = std::fma(-n, k_fast_exp_ln2_lo_f32, std::fma(-n, k_fast_exp_ln2_hi_f32, x));

                const float g2 = g * g;
                const float g4 = g2 * g2;

                const auto& cf = k_fast_exp_poly_f32;
                const float p01 = 1.0F + g;
                const float p23 = std::fma(cf[3], g, cf[2]);
                const float p45 = std::fma(cf[5], g, cf[4]);

                const float p03 = std::fma(p23, g2, p01);
                const float p46 = std::fma(cf[6], g2, p45);
                const float vp = std::fma(p46, g4, p03);

                const std::uint32_t exp_bits = (std::bit_cast<std::uint32_t>(d) + 127U) << 23U;
                return vp * std::bit_cast<float>(exp_bits);
            }
            if (std::isnan(x))
            {
                return x;
            }
            if (x < -104.0F)
            {
                return 0.0F;
            }
            if (x > 88.722839F)
            {
                return std::numeric_limits<float>::infinity();
            }
            return std::exp(x);
        }

        /// @brief Evaluates binary64 fast approximate `exp(x)` value and first two derivatives.
        /// @param[in] x Scalar exponent argument.
        /// @return `ExpJet<double>` populated consistently with `fast_exp_f64(x)`.
        [[nodiscard]] inline ExpJet<double> fast_exp_jet_f64(double x) noexcept
        {
            const double e = fast_exp_f64(x);
            return {.v = e, .d1 = e, .d2 = e};
        }

        /// @brief Evaluates binary32 fast approximate `exp(x)` value and first two derivatives.
        /// @param[in] x Scalar exponent argument.
        /// @return `ExpJet<float>` populated consistently with `fast_exp_f32(x)`.
        [[nodiscard]] inline ExpJet<float> fast_exp_jet_f32(float x) noexcept
        {
            const float e = fast_exp_f32(x);
            return {.v = e, .d1 = e, .d2 = e};
        }

    }  // namespace detail

    /// @brief Evaluates exp(x) under the active KernelExp mode in scalar T.
    /// @details Uses standard library `std::exp` under `KernelExp::Accurate` and built-in full-accuracy
    /// range-reduced Estrin minimax polynomial evaluation under `KernelExp::FastApprox`.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in] x Scalar exponent argument.
    /// @param[in] mode Exponential evaluation backend selector.
    /// @return Evaluated kernel exponential exp(x).
    template <KernelScalarConcept T> [[nodiscard]] inline T eval_kernel_exp(T x, KernelExp mode) noexcept
    {
        if (mode == KernelExp::Accurate)
        {
            return std::exp(x);
        }
        if constexpr (std::same_as<T, double>)
        {
            return detail::fast_exp_f64(x);
        }
        else
        {
            return detail::fast_exp_f32(x);
        }
    }

    /// @brief Evaluates (v, d1, d2) for exp(x) under the active KernelExp mode in scalar T.
    /// @details Under `KernelExp::FastApprox`, returns the full-accuracy approximated exponential and its
    /// consistent derivatives `{e, e, e}`. Under `KernelExp::Accurate`, sets `v = d1 = d2 = std::exp(x)`.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in] x Scalar exponent argument.
    /// @param[in] mode Exponential evaluation backend selector.
    /// @return ExpJet<T> holding value, first derivative, and second derivative wrt x.
    template <KernelScalarConcept T> [[nodiscard]] inline ExpJet<T> eval_kernel_exp_jet(T x, KernelExp mode) noexcept
    {
        if (mode == KernelExp::Accurate)
        {
            const T e = std::exp(x);
            return {.v = e, .d1 = e, .d2 = e};
        }
        if constexpr (std::same_as<T, double>)
        {
            return detail::fast_exp_jet_f64(x);
        }
        else
        {
            return detail::fast_exp_jet_f32(x);
        }
    }

}  // namespace cxxgpr
