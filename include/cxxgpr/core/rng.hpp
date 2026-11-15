// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace cxxgpr {

/// Deterministic Xoshiro256++ PRNG seeded via SplitMix64 for cross-platform reproducibility.
class Xoshiro256PlusPlus {
public:
    /// Initializes the 256-bit generator state from a 64-bit seed using SplitMix64.
    explicit constexpr Xoshiro256PlusPlus(std::uint64_t seed) noexcept {
        std::uint64_t x = seed;
        auto next_splitmix = [&x]() noexcept -> std::uint64_t {
            x += 0x9e3779b97f4a7c15ULL;
            std::uint64_t z = x;
            z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31U);
        };
        s_ = {next_splitmix(), next_splitmix(), next_splitmix(), next_splitmix()};
    }

    /// Advances the generator and returns the next 64-bit pseudorandom word.
    [[nodiscard]] constexpr std::uint64_t next_u64() noexcept {
        const std::uint64_t result = std::rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17U;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);
        return result;
    }

    /// Returns a double uniformly distributed in [0, 1) from the top 53 bits.
    [[nodiscard]] double unit() noexcept;

    /// Returns an integer in [0, max), or 0 when max <= 1.
    [[nodiscard]] std::size_t up_to(std::size_t max) noexcept;

    /// Returns a double clamped strictly inside (0, 1) away from exact 0 and 1 endpoints.
    [[nodiscard]] double open_unit() noexcept;

    /// Returns a standard normal variate via Box-Muller transform.
    [[nodiscard]] double unit_normal() noexcept;

    /// Checks internal state equality between two generator instances.
    [[nodiscard]] constexpr bool
    operator==(const Xoshiro256PlusPlus& other) const noexcept = default;

private:
    std::array<std::uint64_t, 4> s_{};
};

/// Canonical alias for seeded deterministic random number generation.
using SeededRng = Xoshiro256PlusPlus;

} // namespace cxxgpr
