// SPDX-License-Identifier: Apache-2.0
/// @file rng.cpp
/// @brief Implementation of Xoshiro256++ floating-point, bounded-integer, and normal samplers.
/// @details Implements 53-bit uniform scaling, portable 128-bit Lemire rejection sampling,
/// open-interval clamping, and the Box-Muller normal transform.

#include "cxxgpr/core/rng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace cxxgpr
{

    namespace
    {

        /// @brief Normalization factor 2^-53 mapping a 53-bit unsigned integer to [0, 1).
        constexpr double unit_scale = 1.0 / static_cast<double>(1ULL << 53U);

        /// @brief Unsigned 128-bit integer represented as high and low 64-bit words.
        /// @details Enables portable 64x64->128 multiplication without non-standard __uint128_t.
        struct U128
        {
            /// @brief Upper 64 bits of the 128-bit product.
            std::uint64_t hi;
            /// @brief Lower 64 bits of the 128-bit product.
            std::uint64_t lo;
        };

        /// @brief Computes the exact 128-bit unsigned product of two 64-bit integers.
        /// @details Splits each operand into 32-bit halves and propagates carries across partial products.
        /// @param a First 64-bit unsigned factor.
        /// @param b Second 64-bit unsigned factor.
        /// @return Full 128-bit unsigned product split into hi and lo 64-bit words.
        constexpr U128 mul_u64_u64(std::uint64_t a, std::uint64_t b) noexcept
        {
            // Decompose 64-bit inputs into 32-bit limbs so all partial products fit in uint64_t.
            const std::uint64_t a_lo = a & 0xffffffffULL;
            const std::uint64_t a_hi = a >> 32U;
            const std::uint64_t b_lo = b & 0xffffffffULL;
            const std::uint64_t b_hi = b >> 32U;

            const std::uint64_t p00 = a_lo * b_lo;
            const std::uint64_t p01 = a_lo * b_hi;
            const std::uint64_t p10 = a_hi * b_lo;
            const std::uint64_t p11 = a_hi * b_hi;

            // Combine middle 32-bit cross terms with the upper half of p00 and propagate carries.
            const std::uint64_t mid = (p00 >> 32U) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
            const std::uint64_t lo = (p00 & 0xffffffffULL) | (mid << 32U);
            const std::uint64_t hi = p11 + (p01 >> 32U) + (p10 >> 32U) + (mid >> 32U);
            return {.hi = hi, .lo = lo};
        }

    }  // namespace

    /// @brief Returns a double uniformly distributed in [0, 1) from the top 53 bits.
    /// @details Discards the lowest 11 bits of next_u64() and multiplies by 2^-53.
    /// @return Uniform double in [0, 1).
    double Xoshiro256PlusPlus::unit() noexcept
    {
        // Use the upper 53 bits to populate the IEEE-754 double significand cleanly.
        return static_cast<double>(next_u64() >> 11U) * unit_scale;
    }

    /// @brief Returns an integer uniformly distributed in the closed interval [0, max].
    /// @details Uses Lemire's divisionless nearly-divisionless rejection algorithm on [0, max].
    /// @param max Inclusive upper bound of the integer range.
    /// @return Unbiased uniform integer in [0, max].
    std::size_t Xoshiro256PlusPlus::up_to(std::size_t max) noexcept
    {
        // Lemire (2019) unbiased rejection sampling over 0..=max, matching gprx/src/rng.rs.
        const std::uint64_t range = static_cast<std::uint64_t>(max) + 1ULL;
        if (range == 0ULL)
        {
            // When max == UINT64_MAX, range wraps to 0 so every 64-bit output is valid directly.
            return static_cast<std::size_t>(next_u64());
        }
        // Compute (2^64) mod range as (-range) mod range to reject biased low residues.
        const std::uint64_t threshold = ((~range) + 1ULL) % range;
        for (;;)
        {
            const U128 wide = mul_u64_u64(next_u64(), range);
            if (wide.lo >= threshold)
            {
                return static_cast<std::size_t>(wide.hi);
            }
        }
    }

    /// @brief Returns a double clamped strictly inside (0, 1) away from exact 0 and 1 endpoints.
    /// @details Clamps unit() into [unit_scale, 1 - unit_scale].
    /// @return Double value strictly inside (0, 1).
    double Xoshiro256PlusPlus::open_unit() noexcept
    {
        // Clamp away from exact 0.0 and 1.0 by 2^-53 so downstream logarithms remain finite.
        const double u = unit();
        constexpr double eps = unit_scale;
        if (u <= eps)
        {
            return eps;
        }
        if (u >= 1.0 - eps)
        {
            return 1.0 - eps;
        }
        return u;
    }

    /// @brief Returns a standard normal variate via Box-Muller transform.
    /// @details Evaluates sqrt(-2 * log(u1)) * cos(2 * pi * u2) from two open_unit() samples.
    /// @return Standard normal variate distributed as N(0, 1).
    double Xoshiro256PlusPlus::unit_normal() noexcept
    {
        // Guard u1 against subnormal/zero floor before taking log in Box-Muller transform.
        const double u1 = std::max(open_unit(), std::numeric_limits<double>::min());
        const double u2 = open_unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
    }

}  // namespace cxxgpr
