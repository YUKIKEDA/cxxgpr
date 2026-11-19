// SPDX-License-Identifier: Apache-2.0
/// @file exp.hpp
/// @brief Exponential evaluation backend selection and derivative-jet structures for kernel evaluation.
/// @details Declares KernelExp for selecting between standard-library std::exp and inline range-reduced
/// degree-7 FastApprox polynomial evaluation (using IEEE-754 magic-bias bit-manipulation range reduction
/// and Cody-Waite subtraction), together with ExpJet and inline evaluation entry points.
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

    /// @brief Selects the exponential evaluation backend used inside built-in kernel leaves.
    /// @details Applies to kernel matrix, gradient, and Hessian exp(-x) evaluations; parameter transforms
    /// such as exp(log_theta) and user-defined CustomKernel leaves always use standard std::exp.
    enum class KernelExp : std::uint8_t
    {
        /// @brief Exact standard library std::exp in the storage scalar.
        Accurate,
        /// @brief Range-reduced degree-7 minimax-style Taylor polynomial approximation in the storage scalar.
        FastApprox,
    };

    /// @brief Value and first two derivatives of exp(x) with respect to the exponent x.
    /// @details Used by FastApprox gradient and Hessian formulas to preserve exact polynomial-derivative
    /// consistency with the approximated exponential value.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> struct ExpJet
    {
        /// @brief Approximated function value exp(x).
        T v = T{0};
        /// @brief First derivative d/dx exp(x).
        T d1 = T{0};
        /// @brief Second derivative d^2/dx^2 exp(x).
        T d2 = T{0};
    };

    /// @brief Internal implementation helpers for inline fast exponential approximation.
    namespace detail
    {

        /// @brief Polynomial degree used by the range-reduced FastApprox exponential approximation.
        inline constexpr std::size_t k_fast_exp_degree = 7U;

        /// @brief Builds the degree-7 Taylor coefficient table `[1/0!, 1/1!, ..., 1/7!]` at compile time.
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @return Ascending-degree Taylor coefficient array of length 8.
        template <typename T> consteval std::array<T, k_fast_exp_degree + 1U> make_exp_taylor_coeffs() noexcept
        {
            std::array<T, k_fast_exp_degree + 1U> c{};
            c[0] = T{1};
            T fact = T{1};
            // Compute 1 / k! iteratively so polynomial evaluation avoids runtime division.
            for (std::size_t k = 1U; k <= k_fast_exp_degree; ++k)
            {
                fact *= static_cast<T>(k);
                c[k] = T{1} / fact;
            }
            return c;
        }

        /// @brief Compile-time Taylor coefficient table `[1/0!, ..., 1/7!]` in binary64 precision.
        inline constexpr std::array<double, k_fast_exp_degree + 1U> k_taylor_f64 = make_exp_taylor_coeffs<double>();

        /// @brief Compile-time Taylor coefficient table `[1/0!, ..., 1/7!]` in binary32 precision.
        inline constexpr std::array<float, k_fast_exp_degree + 1U> k_taylor_f32 = make_exp_taylor_coeffs<float>();

        /// @brief Evaluates `sum_{i=0}^{len-1} c[i] * g^i` using Horner's method.
        /// @tparam T Floating-point scalar type.
        /// @param[in] g Reduced exponent argument in `[-ln(2)/2, ln(2)/2]`.
        /// @param[in] c Pointer to ascending-degree polynomial coefficients.
        /// @param[in] len Number of coefficients (`>= 1`).
        /// @return Polynomial value evaluated at `g`.
        template <typename T> [[nodiscard]] constexpr T exp_horner(T g, const T* c, std::size_t len) noexcept
        {
            T acc = c[len - 1U];
            for (std::size_t i = len - 1U; i > 0U; --i)
            {
                acc = (acc * g) + c[i - 1U];
            }
            return acc;
        }

        /// @brief Fast IEEE-754 magic-bias range reduction and degree-7 polynomial exp(x) in double precision.
        /// @details On the normal finite range `[-708.0, 709.0]`, adds `1.5 * 2^52` to `x / ln(2)` for single-op
        /// integer rounding and reads biased exponent bits directly to form `2^k`; falls back cleanly on extremes.
        /// @param[in] x Scalar exponent in double precision.
        /// @return Approximated exp(x) preserving 0, subnormal/underflow, +inf, and NaN semantics.
        [[nodiscard]] inline double fast_exp_f64(double x) noexcept
        {
            // Fast normal range avoids subnormal bit shifts and infinity/NaN branches.
            if (x >= -708.0 && x <= 709.0)
            {
                // 1.5 * 2^52 aligns integer bit 0 in bits[50:0]; low 12 bits of (d) mod 4096 equal k mod 4096.
                constexpr double k_magic = 6755399441055744.0;
                // NOLINTNEXTLINE(modernize-use-std-numbers)
                constexpr double k_ln2_hi = 6.93147180369123816490e-1;
                constexpr double k_ln2_lo = 1.90821492927058770002e-10;
                const double d = (x * std::numbers::log2e) + k_magic;
                const double n = d - k_magic;
                const double g = (x - (n * k_ln2_hi)) - (n * k_ln2_lo);
                // Because k_magic has 0x000 in its low 12 bits, (bits(d) + 1023) << 52 yields exact 2^n.
                const std::uint64_t exp_bits = (std::bit_cast<std::uint64_t>(d) + 1023ULL) << 52U;
                const double scale = std::bit_cast<double>(exp_bits);
                return exp_horner(g, k_taylor_f64.data(), k_taylor_f64.size()) * scale;
            }
            // Handle out-of-normal underflow, overflow, subnormals, and NaN cleanly.
            if (x < -746.0)
            {
                return 0.0;
            }
            if (x > 709.0)
            {
                return std::numeric_limits<double>::infinity();
            }
            return std::exp(x);
        }

        /// @brief Fast IEEE-754 magic-bias range reduction and degree-7 polynomial exp(x) in float precision.
        /// @details On the normal finite range `[-87.0F, 88.0F]`, adds `1.5 * 2^23` to `x / ln(2)` for single-op
        /// integer rounding and constructs `2^k` via bit shift; handles extremes cleanly outside that range.
        /// @param[in] x Scalar exponent in float precision.
        /// @return Approximated exp(x) preserving 0, subnormal/underflow, +inf, and NaN semantics.
        [[nodiscard]] inline float fast_exp_f32(float x) noexcept
        {
            if (x >= -87.0F && x <= 88.0F)
            {
                // 1.5 * 2^23 has 0x000 in its low 9 bits so (bits(d) + 127) << 23 encodes 2^n directly.
                constexpr float k_magic = 12582912.0F;
                const auto ln2_hi = std::bit_cast<float>(0x3f317180U);
                const auto ln2_lo = std::bit_cast<float>(0x3717f7d1U);
                const auto log2_e = std::bit_cast<float>(0x3fb8aa3bU);
                const float d = (x * log2_e) + k_magic;
                const float n = d - k_magic;
                const float g = (x - (n * ln2_hi)) - (n * ln2_lo);
                const std::uint32_t exp_bits = (std::bit_cast<std::uint32_t>(d) + 127U) << 23U;
                const float scale = std::bit_cast<float>(exp_bits);
                return exp_horner(g, k_taylor_f32.data(), k_taylor_f32.size()) * scale;
            }
            if (x < -104.0F)
            {
                return 0.0F;
            }
            if (x > 88.0F)
            {
                return std::numeric_limits<float>::infinity();
            }
            return std::exp(x);
        }

        /// @brief Fast IEEE-754 range-reduced degree-7 polynomial value and first two derivatives in double.
        /// @param[in] x Scalar exponent in double precision.
        /// @return `ExpJet<double>` holding approximated exp(x) alongside matching polynomial derivatives.
        [[nodiscard]] inline ExpJet<double> fast_exp_jet_f64(double x) noexcept
        {
            if (x >= -708.0 && x <= 709.0)
            {
                constexpr double k_magic = 6755399441055744.0;
                // NOLINTNEXTLINE(modernize-use-std-numbers)
                constexpr double k_ln2_hi = 6.93147180369123816490e-1;
                constexpr double k_ln2_lo = 1.90821492927058770002e-10;
                const double d = (x * std::numbers::log2e) + k_magic;
                const double n = d - k_magic;
                const double g = (x - (n * k_ln2_hi)) - (n * k_ln2_lo);
                const std::uint64_t exp_bits = (std::bit_cast<std::uint64_t>(d) + 1023ULL) << 52U;
                const double scale = std::bit_cast<double>(exp_bits);
                return {
                    .v = exp_horner(g, k_taylor_f64.data(), k_taylor_f64.size()) * scale,
                    .d1 = exp_horner(g, k_taylor_f64.data(), k_taylor_f64.size() - 1U) * scale,
                    .d2 = exp_horner(g, k_taylor_f64.data(), k_taylor_f64.size() - 2U) * scale,
                };
            }
            const double e = fast_exp_f64(x);
            return {.v = e, .d1 = e, .d2 = e};
        }

        /// @brief Fast IEEE-754 range-reduced degree-7 polynomial value and first two derivatives in float.
        /// @param[in] x Scalar exponent in float precision.
        /// @return `ExpJet<float>` holding approximated exp(x) alongside matching polynomial derivatives.
        [[nodiscard]] inline ExpJet<float> fast_exp_jet_f32(float x) noexcept
        {
            if (x >= -87.0F && x <= 88.0F)
            {
                constexpr float k_magic = 12582912.0F;
                const auto ln2_hi = std::bit_cast<float>(0x3f317180U);
                const auto ln2_lo = std::bit_cast<float>(0x3717f7d1U);
                const auto log2_e = std::bit_cast<float>(0x3fb8aa3bU);
                const float d = (x * log2_e) + k_magic;
                const float n = d - k_magic;
                const float g = (x - (n * ln2_hi)) - (n * ln2_lo);
                const std::uint32_t exp_bits = (std::bit_cast<std::uint32_t>(d) + 127U) << 23U;
                const float scale = std::bit_cast<float>(exp_bits);
                return {
                    .v = exp_horner(g, k_taylor_f32.data(), k_taylor_f32.size()) * scale,
                    .d1 = exp_horner(g, k_taylor_f32.data(), k_taylor_f32.size() - 1U) * scale,
                    .d2 = exp_horner(g, k_taylor_f32.data(), k_taylor_f32.size() - 2U) * scale,
                };
            }
            const float e = fast_exp_f32(x);
            return {.v = e, .d1 = e, .d2 = e};
        }

    }  // namespace detail

    /// @brief Evaluates exp(x) under the selected KernelExp backend (inline for zero cross-TU call overhead).
    /// @details Uses standard library `std::exp` under `KernelExp::Accurate` and range-reduced degree-7 Horner
    /// approximation with magic-bias integer reduction under `KernelExp::FastApprox`.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] x Exponent value in the storage scalar.
    /// @param[in] mode Exponential evaluation backend selector.
    /// @return Approximated or exact value of exp(x).
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

    /// @brief Evaluates exp(x) and its first two derivatives under the selected KernelExp backend.
    /// @details Under KernelExp::FastApprox, returns derivatives matching the truncated degree-7 polynomial
    /// approximation rather than std::exp so finite-difference checks remain self-consistent.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] x Exponent value in the storage scalar.
    /// @param[in] mode Exponential evaluation backend selector.
    /// @return Value and first two derivatives with respect to x.
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
