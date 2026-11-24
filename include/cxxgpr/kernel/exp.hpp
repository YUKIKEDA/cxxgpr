// SPDX-License-Identifier: Apache-2.0
/// @file exp.hpp
/// @brief Kernel exponential evaluation selector, derivative jet, and entry points backed by herumi/fmath.
/// @details Defines KernelExp, ExpJet<T>, and inline eval_kernel_exp() / eval_kernel_exp_jet() entry points
/// dispatching KernelExp::Accurate to std::exp and KernelExp::FastApprox to internal herumi/fmath routines.

#pragma once

#include "cxxgpr/precision/policy.hpp"

#include <cmath>
#include <cstdint>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Selects exact vs fast-approximate exponential evaluation inside built-in kernels.
    /// @details Accurate evaluates std::exp in the active scalar type; FastApprox delegates to
    /// herumi/fmath approximate exponential evaluation while preserving parameter transforms and Custom leaves.
    enum class KernelExp : std::uint8_t
    {
        /// @brief Standard library std::exp in the storage scalar (default).
        Accurate,
        /// @brief Vectorized fast exponential approximation backed by herumi/fmath.
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

    /// @brief Internal fast-exponential implementation entry points backed by herumi/fmath.
    namespace detail
    {

        /// @brief Evaluates scalar double-precision fast approximate exp(x) via herumi/fmath.
        /// @param[in] x Scalar exponent argument.
        /// @return Fast approximate exp(x) with IEEE-754 boundary handling.
        [[nodiscard]] double fast_exp_f64(double x) noexcept;

        /// @brief Evaluates scalar single-precision fast approximate exp(x) via herumi/fmath.
        /// @param[in] x Scalar exponent argument.
        /// @return Fast approximate exp(x) with IEEE-754 boundary handling.
        [[nodiscard]] float fast_exp_f32(float x) noexcept;

        /// @brief Evaluates double-precision fast approximate exp(x) value and derivatives via herumi/fmath.
        /// @param[in] x Scalar exponent argument.
        /// @return ExpJet<double> containing value, first derivative, and second derivative.
        [[nodiscard]] ExpJet<double> fast_exp_jet_f64(double x) noexcept;

        /// @brief Evaluates single-precision fast approximate exp(x) value and derivatives via herumi/fmath.
        /// @param[in] x Scalar exponent argument.
        /// @return ExpJet<float> containing value, first derivative, and second derivative.
        [[nodiscard]] ExpJet<float> fast_exp_jet_f32(float x) noexcept;

    }  // namespace detail

    /// @brief Evaluates exp(x) under the active KernelExp mode in scalar T.
    /// @details Uses standard library `std::exp` under `KernelExp::Accurate` and `herumi/fmath`
    /// approximation under `KernelExp::FastApprox`.
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
    /// @details Under KernelExp::FastApprox, delegates to the herumi/fmath derivative jet. Under
    /// KernelExp::Accurate, sets `v = d1 = d2 = std::exp(x)`.
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
