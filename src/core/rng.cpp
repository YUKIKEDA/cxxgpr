// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/rng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace cxxgpr {

namespace {

constexpr double unit_scale = 1.0 / static_cast<double>(1ULL << 53U);

} // namespace

double Xoshiro256PlusPlus::unit() noexcept {
    return static_cast<double>(next_u64() >> 11U) * unit_scale;
}

std::size_t Xoshiro256PlusPlus::up_to(std::size_t max) noexcept {
    if (max <= 1) {
        return 0;
    }
    return static_cast<std::size_t>(next_u64() % static_cast<std::uint64_t>(max));
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
