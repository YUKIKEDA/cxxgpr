// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/exp.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace cxxgpr {

namespace {

constexpr std::size_t k_poly_degree = 7U;

consteval std::array<double, k_poly_degree + 1U> make_taylor_f64() noexcept {
    std::array<double, k_poly_degree + 1U> c{};
    c[0] = 1.0;
    double fact = 1.0;
    for (std::size_t k = 1U; k <= k_poly_degree; ++k) {
        fact *= static_cast<double>(k);
        c[k] = 1.0 / fact;
    }
    return c;
}

consteval std::array<float, k_poly_degree + 1U> make_taylor_f32() noexcept {
    std::array<float, k_poly_degree + 1U> c{};
    c[0] = 1.0F;
    float fact = 1.0F;
    for (std::size_t k = 1U; k <= k_poly_degree; ++k) {
        fact *= static_cast<float>(k);
        c[k] = 1.0F / fact;
    }
    return c;
}

constexpr std::array<double, k_poly_degree + 1U> k_taylor_f64 = make_taylor_f64();
constexpr std::array<float, k_poly_degree + 1U> k_taylor_f32 = make_taylor_f32();

// Cody-Waite high/low split of ln(2) requires exact truncated significand bits.
// NOLINTNEXTLINE(modernize-use-std-numbers)
constexpr double k_ln2_hi_f64 = 6.93147180369123816490e-1;
constexpr double k_ln2_lo_f64 = 1.90821492927058770002e-10;

template <typename T> [[nodiscard]] constexpr T horner(T g, const T* c, std::size_t len) noexcept {
    T acc = c[len - 1U];
    for (std::size_t i = len - 1U; i > 0U; --i) {
        acc = (acc * g) + c[i - 1U];
    }
    return acc;
}

[[nodiscard]] double pow2_f64(double n_real) noexcept {
    if (n_real <= -1075.0) {
        return 0.0;
    }
    if (n_real >= 1024.0) {
        return std::numeric_limits<double>::infinity();
    }
    const auto n = static_cast<std::int64_t>(n_real);
    if (n >= -1022) {
        const auto bits = static_cast<std::uint64_t>(n + 1023) << 52U;
        return std::bit_cast<double>(bits);
    }
    const auto bits = 1ULL << static_cast<unsigned>(n + 1074);
    return std::bit_cast<double>(bits);
}

[[nodiscard]] float pow2_f32(float n_real) noexcept {
    if (n_real <= -150.0F) {
        return 0.0F;
    }
    if (n_real >= 128.0F) {
        return std::numeric_limits<float>::infinity();
    }
    const auto n = static_cast<std::int32_t>(n_real);
    if (n >= -126) {
        const auto bits = static_cast<std::uint32_t>(n + 127) << 23U;
        return std::bit_cast<float>(bits);
    }
    const auto bits = 1U << static_cast<unsigned>(n + 149);
    return std::bit_cast<float>(bits);
}

[[nodiscard]] std::pair<double, double> reduce_f64(double x) noexcept {
    const double n = std::round(x * std::numbers::log2e);
    const double g = (x - (n * k_ln2_hi_f64)) - (n * k_ln2_lo_f64);
    return {g, pow2_f64(n)};
}

[[nodiscard]] std::pair<float, float> reduce_f32(float x) noexcept {
    const auto ln2_hi = std::bit_cast<float>(0x3f317180U);
    const auto ln2_lo = std::bit_cast<float>(0x3717f7d1U);
    const auto log2_e = std::bit_cast<float>(0x3fb8aa3bU);
    const float n = std::round(x * log2_e);
    const float g = (x - (n * ln2_hi)) - (n * ln2_lo);
    return {g, pow2_f32(n)};
}

} // namespace

template <KernelScalarConcept T> T eval_kernel_exp(T x, KernelExp mode) noexcept {
    if (mode == KernelExp::Accurate || !std::isfinite(x)) {
        return std::exp(x);
    }
    if constexpr (std::same_as<T, double>) {
        const auto [g, scale] = reduce_f64(x);
        return horner(g, k_taylor_f64.data(), k_taylor_f64.size()) * scale;
    } else {
        const auto [g, scale] = reduce_f32(x);
        return horner(g, k_taylor_f32.data(), k_taylor_f32.size()) * scale;
    }
}

template <KernelScalarConcept T> ExpJet<T> eval_kernel_exp_jet(T x, KernelExp mode) noexcept {
    if (mode == KernelExp::Accurate || !std::isfinite(x)) {
        const T e = std::exp(x);
        return {.v = e, .d1 = e, .d2 = e};
    }
    if constexpr (std::same_as<T, double>) {
        const auto [g, scale] = reduce_f64(x);
        return {
            .v = horner(g, k_taylor_f64.data(), k_taylor_f64.size()) * scale,
            .d1 = horner(g, k_taylor_f64.data(), k_taylor_f64.size() - 1U) * scale,
            .d2 = horner(g, k_taylor_f64.data(), k_taylor_f64.size() - 2U) * scale,
        };
    } else {
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

} // namespace cxxgpr
