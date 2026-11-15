// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/rng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace cxxgpr {

namespace {

constexpr double unit_scale = 1.0 / static_cast<double>(1ULL << 53U);

struct U128 {
    std::uint64_t hi;
    std::uint64_t lo;
};

constexpr U128 mul_u64_u64(std::uint64_t a, std::uint64_t b) noexcept {
    const std::uint64_t a_lo = a & 0xffffffffULL;
    const std::uint64_t a_hi = a >> 32U;
    const std::uint64_t b_lo = b & 0xffffffffULL;
    const std::uint64_t b_hi = b >> 32U;

    const std::uint64_t p00 = a_lo * b_lo;
    const std::uint64_t p01 = a_lo * b_hi;
    const std::uint64_t p10 = a_hi * b_lo;
    const std::uint64_t p11 = a_hi * b_hi;

    const std::uint64_t mid = (p00 >> 32U) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    const std::uint64_t lo = (p00 & 0xffffffffULL) | (mid << 32U);
    const std::uint64_t hi = p11 + (p01 >> 32U) + (p10 >> 32U) + (mid >> 32U);
    return {.hi = hi, .lo = lo};
}

} // namespace

double Xoshiro256PlusPlus::unit() noexcept {
    return static_cast<double>(next_u64() >> 11U) * unit_scale;
}

std::size_t Xoshiro256PlusPlus::up_to(std::size_t max) noexcept {
    // Lemire (2019) unbiased rejection sampling over 0..=max, matching gprx/src/rng.rs.
    const std::uint64_t range = static_cast<std::uint64_t>(max) + 1ULL;
    if (range == 0ULL) {
        return static_cast<std::size_t>(next_u64());
    }
    const std::uint64_t threshold = ((~range) + 1ULL) % range;
    for (;;) {
        const U128 wide = mul_u64_u64(next_u64(), range);
        if (wide.lo >= threshold) {
            return static_cast<std::size_t>(wide.hi);
        }
    }
}

double Xoshiro256PlusPlus::open_unit() noexcept {
    const double u = unit();
    constexpr double eps = unit_scale;
    if (u <= eps) {
        return eps;
    }
    if (u >= 1.0 - eps) {
        return 1.0 - eps;
    }
    return u;
}

double Xoshiro256PlusPlus::unit_normal() noexcept {
    const double u1 = std::max(open_unit(), std::numeric_limits<double>::min());
    const double u2 = open_unit();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

} // namespace cxxgpr
