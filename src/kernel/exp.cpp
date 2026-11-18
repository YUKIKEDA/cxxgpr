// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Implementation of exact and FastApprox range-reduced degree-7 exponential evaluation.
/// @details Evaluates exp(x) and its first two derivatives using Cody-Waite range reduction modulo ln(2),
/// degree-7 Horner polynomial evaluation on [-ln(2)/2, ln(2)/2], and IEEE-754 power-of-two scaling.
#include "cxxgpr/kernel/exp.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace cxxgpr
{

    namespace
    {

        /// @brief Polynomial degree used by the range-reduced FastApprox exponential kernel.
        constexpr std::size_t k_poly_degree = 7U;

        /// @brief Builds the degree-7 Taylor coefficient table [1/0!, 1/1!, ..., 1/7!] in double.
        /// @details Evaluated at compile time for Horner evaluation of exp(g) over the reduced interval.
        /// @return Array of 8 polynomial coefficients in ascending power order.
        consteval std::array<double, k_poly_degree + 1U> make_taylor_f64() noexcept
        {
            std::array<double, k_poly_degree + 1U> c{};
            c[0] = 1.0;
            double fact = 1.0;
            // Accumulate k! sequentially so c[k] = 1 / k!.
            for (std::size_t k = 1U; k <= k_poly_degree; ++k)
            {
                fact *= static_cast<double>(k);
                c[k] = 1.0 / fact;
            }
            return c;
        }

        /// @brief Builds the degree-7 Taylor coefficient table [1/0!, 1/1!, ..., 1/7!] in float.
        /// @details Evaluated at compile time for single-precision Horner evaluation over the reduced interval.
        /// @return Array of 8 polynomial coefficients in ascending power order.
        consteval std::array<float, k_poly_degree + 1U> make_taylor_f32() noexcept
        {
            std::array<float, k_poly_degree + 1U> c{};
            c[0] = 1.0F;
            float fact = 1.0F;
            // Accumulate k! sequentially in float so c[k] = 1 / k!.
            for (std::size_t k = 1U; k <= k_poly_degree; ++k)
            {
                fact *= static_cast<float>(k);
                c[k] = 1.0F / fact;
            }
            return c;
        }

        /// @brief Precomputed degree-7 polynomial coefficients for double-precision FastApprox.
        constexpr std::array<double, k_poly_degree + 1U> k_taylor_f64 = make_taylor_f64();

        /// @brief Precomputed degree-7 polynomial coefficients for single-precision FastApprox.
        constexpr std::array<float, k_poly_degree + 1U> k_taylor_f32 = make_taylor_f32();

        /// @brief High-order truncated bits of ln(2) for double-precision Cody-Waite range reduction.
        /// @details Exact truncated significand bits prevent catastrophic cancellation in x - n * ln(2).
        // NOLINTNEXTLINE(modernize-use-std-numbers)
        constexpr double k_ln2_hi_f64 = 6.93147180369123816490e-1;

        /// @brief Low-order compensation tail of ln(2) paired with k_ln2_hi_f64.
        constexpr double k_ln2_lo_f64 = 1.90821492927058770002e-10;

        /// @brief Evaluates a polynomial sum_{i=0}^{len-1} c[i] * g^i using Horner's method.
        /// @details Traverses coefficients from highest degree down to constant term c[0].
        /// @tparam T Floating-point scalar type.
        /// @param g Range-reduced exponent argument in [-ln(2)/2, ln(2)/2].
        /// @param c Pointer to ascending-degree coefficient array of length len.
        /// @param len Number of coefficients to evaluate (must be >= 1).
        /// @return Polynomial value evaluated at g.
        template <typename T> [[nodiscard]] constexpr T horner(T g, const T* c, std::size_t len) noexcept
        {
            // Start from the leading coefficient and fold downward via Horner's rule.
            T acc = c[len - 1U];
            for (std::size_t i = len - 1U; i > 0U; --i)
            {
                acc = (acc * g) + c[i - 1U];
            }
            return acc;
        }

        /// @brief Computes exact IEEE-754 power of two 2^n in double from an integer-valued double n_real.
        /// @details Clamps extreme exponents to +0.0 or +infinity and constructs normal or subnormal bit patterns.
        /// @param n_real Rounded binary exponent n as a double.
        /// @return Value 2^n in IEEE-754 binary64 representation.
        [[nodiscard]] double pow2_f64(double n_real) noexcept
        {
            // Subnormal binary64 underflows below 2^-1075; exponents >= 1024 overflow to +infinity.
            if (n_real <= -1075.0)
            {
                return 0.0;
            }
            if (n_real >= 1024.0)
            {
                return std::numeric_limits<double>::infinity();
            }
            const auto n = static_cast<std::int64_t>(n_real);
            // Normal range [-1022, 1023]: encode biased exponent directly in bits [62:52].
            if (n >= -1022)
            {
                const auto bits = static_cast<std::uint64_t>(n + 1023) << 52U;
                return std::bit_cast<double>(bits);
            }
            // Subnormal range [-1074, -1023]: shift a single significand bit.
            const auto bits = 1ULL << static_cast<unsigned>(n + 1074);
            return std::bit_cast<double>(bits);
        }

        /// @brief Computes exact IEEE-754 power of two 2^n in float from an integer-valued float n_real.
        /// @details Clamps extreme exponents to +0.0f or +infinity and constructs normal or subnormal bit patterns.
        /// @param n_real Rounded binary exponent n as a float.
        /// @return Value 2^n in IEEE-754 binary32 representation.
        [[nodiscard]] float pow2_f32(float n_real) noexcept
        {
            // Subnormal binary32 underflows below 2^-150; exponents >= 128 overflow to +infinity.
            if (n_real <= -150.0F)
            {
                return 0.0F;
            }
            if (n_real >= 128.0F)
            {
                return std::numeric_limits<float>::infinity();
            }
            const auto n = static_cast<std::int32_t>(n_real);
            // Normal range [-126, 127]: encode biased exponent directly in bits [30:23].
            if (n >= -126)
            {
                const auto bits = static_cast<std::uint32_t>(n + 127) << 23U;
                return std::bit_cast<float>(bits);
            }
            // Subnormal range [-149, -127]: shift a single significand bit.
            const auto bits = 1U << static_cast<unsigned>(n + 149);
            return std::bit_cast<float>(bits);
        }

        /// @brief Performs Cody-Waite range reduction x = n * ln(2) + g for double precision.
        /// @details Splits ln(2) into high and low parts so g lies in [-ln(2)/2, ln(2)/2] with full accuracy.
        /// @param x Finite input exponent.
        /// @return Pair {reduced_argument_g, power_of_two_scale_2_to_n}.
        [[nodiscard]] std::pair<double, double> reduce_f64(double x) noexcept
        {
            // Express x = n * ln(2) + g with |g| <= ln(2) / 2 using two-step Cody-Waite subtraction.
            const double n = std::round(x * std::numbers::log2e);
            const double g = (x - (n * k_ln2_hi_f64)) - (n * k_ln2_lo_f64);
            return {g, pow2_f64(n)};
        }

        /// @brief Performs Cody-Waite range reduction x = n * ln(2) + g for single precision.
        /// @details Uses exact IEEE-754 binary32 high/low ln(2) constants so n * ln2_hi is exact in float.
        /// @param x Finite input exponent.
        /// @return Pair {reduced_argument_g, power_of_two_scale_2_to_n}.
        [[nodiscard]] std::pair<float, float> reduce_f32(float x) noexcept
        {
            // Load binary32 high/low ln(2) split and log2(e) constants by exact bit patterns.
            const auto ln2_hi = std::bit_cast<float>(0x3f317180U);
            const auto ln2_lo = std::bit_cast<float>(0x3717f7d1U);
            const auto log2_e = std::bit_cast<float>(0x3fb8aa3bU);
            const float n = std::round(x * log2_e);
            const float g = (x - (n * ln2_hi)) - (n * ln2_lo);
            return {g, pow2_f32(n)};
        }

    }  // namespace

    /// @brief Evaluates exp(x) under the selected KernelExp backend.
    /// @details Dispatches to std::exp for Accurate or non-finite arguments, and to Cody-Waite
    /// range reduction with degree-7 Horner evaluation for FastApprox.
    /// @tparam T Scalar type (double or float).
    /// @param x Exponent value.
    /// @param mode Exponential evaluation backend selector.
    /// @return Exact or approximated value of exp(x).
    template <KernelScalarConcept T> T eval_kernel_exp(T x, KernelExp mode) noexcept
    {
        // Delegate Accurate mode and non-finite inputs (NaN, +-inf) directly to std::exp.
        if (mode == KernelExp::Accurate || !std::isfinite(x))
        {
            return std::exp(x);
        }
        if constexpr (std::same_as<T, double>)
        {
            const auto [g, scale] = reduce_f64(x);
            return horner(g, k_taylor_f64.data(), k_taylor_f64.size()) * scale;
        }
        else
        {
            const auto [g, scale] = reduce_f32(x);
            return horner(g, k_taylor_f32.data(), k_taylor_f32.size()) * scale;
        }
    }

    /// @brief Evaluates exp(x) and its first two derivatives under the selected KernelExp backend.
    /// @details For FastApprox, the first and second derivatives of the degree-7 Taylor polynomial
    /// reduce cleanly to degree-6 and degree-5 truncations of the same coefficient array.
    /// @tparam T Scalar type (double or float).
    /// @param x Exponent value.
    /// @param mode Exponential evaluation backend selector.
    /// @return Value and first two derivatives with respect to x.
    template <KernelScalarConcept T> ExpJet<T> eval_kernel_exp_jet(T x, KernelExp mode) noexcept
    {
        // Preserve exact IEEE semantics for Accurate mode and non-finite inputs.
        if (mode == KernelExp::Accurate || !std::isfinite(x))
        {
            const T e = std::exp(x);
            return {.v = e, .d1 = e, .d2 = e};
        }
        // Differentiating sum_{k=0}^7 g^k/k! yields sum_{k=0}^6 g^k/k! and sum_{k=0}^5 g^k/k!.
        if constexpr (std::same_as<T, double>)
        {
            const auto [g, scale] = reduce_f64(x);
            return {
                .v = horner(g, k_taylor_f64.data(), k_taylor_f64.size()) * scale,
                .d1 = horner(g, k_taylor_f64.data(), k_taylor_f64.size() - 1U) * scale,
                .d2 = horner(g, k_taylor_f64.data(), k_taylor_f64.size() - 2U) * scale,
            };
        }
        else
        {
            const auto [g, scale] = reduce_f32(x);
            return {
                .v = horner(g, k_taylor_f32.data(), k_taylor_f32.size()) * scale,
                .d1 = horner(g, k_taylor_f32.data(), k_taylor_f32.size() - 1U) * scale,
                .d2 = horner(g, k_taylor_f32.data(), k_taylor_f32.size() - 2U) * scale,
            };
        }
    }

    template double eval_kernel_exp<double>(double, KernelExp) noexcept;
    template float eval_kernel_exp<float>(float, KernelExp) noexcept;
    template ExpJet<double> eval_kernel_exp_jet<double>(double, KernelExp) noexcept;
    template ExpJet<float> eval_kernel_exp_jet<float>(float, KernelExp) noexcept;

}  // namespace cxxgpr
