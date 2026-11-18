// SPDX-License-Identifier: Apache-2.0
/// @file rng.hpp
/// @brief Deterministic Xoshiro256++ pseudorandom number generator seeded via SplitMix64.
/// @details Provides a reproducible, platform-independent PRNG and uniform/normal sampling
/// helpers for tests, initialization, and randomized optimization restarts.

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Deterministic Xoshiro256++ PRNG seeded via SplitMix64 for cross-platform reproducibility.
    /// @details Uses fixed SplitMix64 expansion from a 64-bit seed to populate a 256-bit state
    /// array and generates 64-bit outputs, [0, 1) doubles, bounded integers, (0, 1) doubles, and
    /// Box-Muller standard normals.
    class Xoshiro256PlusPlus
    {
    public:
        /// @brief Initializes the 256-bit generator state from a 64-bit seed using SplitMix64.
        /// @details Steps a SplitMix64 state four times to derive independent initial words s_[0..3].
        /// @param seed Deterministic 64-bit seed value.
        explicit constexpr Xoshiro256PlusPlus(std::uint64_t seed) noexcept
        {
            std::uint64_t x = seed;
            // Expand the 64-bit seed into four well-mixed 64-bit words via SplitMix64.
            auto next_splitmix = [&x]() noexcept -> std::uint64_t
            {
                // Add the 64-bit golden ratio constant and apply Staffords Mix13 avalanche step.
                x += 0x9e3779b97f4a7c15ULL;
                std::uint64_t z = x;
                z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
                return z ^ (z >> 31U);
            };
            s_ = {next_splitmix(), next_splitmix(), next_splitmix(), next_splitmix()};
        }

        /// @brief Advances the generator and returns the next 64-bit pseudorandom word.
        /// @details Applies the Xoshiro256++ scrambler rotl(s0 + s3, 23) + s0 and linear F_2
        /// transition across the 256-bit state.
        /// @return Next 64-bit unsigned pseudorandom integer.
        [[nodiscard]] constexpr std::uint64_t next_u64() noexcept
        {
            // Compute the ++ output scrambler before advancing the linear state.
            const std::uint64_t result = std::rotl(s_[0] + s_[3], 23) + s_[0];
            // Advance the 256-bit Xoshiro state with shift-XOR-rotate transitions.
            const std::uint64_t t = s_[1] << 17U;
            s_[2] ^= s_[0];
            s_[3] ^= s_[1];
            s_[1] ^= s_[2];
            s_[0] ^= s_[3];
            s_[2] ^= t;
            s_[3] = std::rotl(s_[3], 45);
            return result;
        }

        /// @brief Returns a double uniformly distributed in [0, 1) from the top 53 bits.
        /// @details Shifts next_u64() right by 11 bits and scales by 2^-53.
        /// @return Uniform double in the half-open interval [0, 1).
        [[nodiscard]] double unit() noexcept;

        /// @brief Returns an integer uniformly distributed in the closed interval [0, max].
        /// @details Uses unbiased 128-bit rejection sampling without division bias.
        /// @param max Inclusive upper bound of the sampled integer range.
        /// @return Uniformly sampled index in [0, max].
        [[nodiscard]] std::size_t up_to(std::size_t max) noexcept;

        /// @brief Returns a double clamped strictly inside (0, 1) away from exact 0 and 1 endpoints.
        /// @details Clamps unit() to [2^-53, 1 - 2^-53] so downstream log(u) evaluations stay finite.
        /// @return Double value strictly inside (0, 1).
        [[nodiscard]] double open_unit() noexcept;

        /// @brief Returns a standard normal variate via Box-Muller transform.
        /// @details Consumes two open_unit() draws to compute sqrt(-2 * ln(u1)) * cos(2 * pi * u2).
        /// @return Pseudorandom sample from N(0, 1).
        [[nodiscard]] double unit_normal() noexcept;

        /// @brief Checks internal state equality between two generator instances.
        /// @details Compares all four 64-bit state words in s_.
        /// @param other Generator instance to compare against.
        /// @return True if both generators hold identical 256-bit internal states.
        [[nodiscard]] constexpr bool operator==(const Xoshiro256PlusPlus& other) const noexcept = default;

    private:
        /// @brief 256-bit Xoshiro internal state stored as four 64-bit words.
        std::array<std::uint64_t, 4> s_{};
    };

    /// @brief Canonical alias for seeded deterministic random number generation.
    /// @details Aliases Xoshiro256PlusPlus across the library.
    using SeededRng = Xoshiro256PlusPlus;

}  // namespace cxxgpr
