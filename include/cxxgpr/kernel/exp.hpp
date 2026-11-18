// SPDX-License-Identifier: Apache-2.0
/// @file exp.hpp
/// @brief Exponential evaluation backend selection and derivative-jet structures for kernel evaluation.
/// @details Declares KernelExp for selecting between standard-library std::exp and range-reduced
/// degree-7 FastApprox polynomial evaluation, together with ExpJet and free evaluation entry points.
#pragma once

#include "cxxgpr/precision/policy.hpp"

#include <cstdint>

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

    /// @brief Evaluates exp(x) under the selected KernelExp backend.
    /// @details Falls back to std::exp for non-finite inputs or when mode is KernelExp::Accurate.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param x Exponent value in the storage scalar.
    /// @param mode Exponential evaluation backend selector.
    /// @return Approximated or exact value of exp(x).
    template <KernelScalarConcept T> [[nodiscard]] T eval_kernel_exp(T x, KernelExp mode) noexcept;

    /// @brief Evaluates exp(x) and its first two derivatives under the selected KernelExp backend.
    /// @details Under KernelExp::FastApprox, returns derivatives matching the truncated degree-7 polynomial
    /// approximation rather than std::exp so finite-difference checks remain self-consistent.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param x Exponent value in the storage scalar.
    /// @param mode Exponential evaluation backend selector.
    /// @return Value and first two derivatives with respect to x.
    template <KernelScalarConcept T> [[nodiscard]] ExpJet<T> eval_kernel_exp_jet(T x, KernelExp mode) noexcept;

    extern template double eval_kernel_exp<double>(double x, KernelExp mode) noexcept;
    extern template float eval_kernel_exp<float>(float x, KernelExp mode) noexcept;
    extern template ExpJet<double> eval_kernel_exp_jet<double>(double x, KernelExp mode) noexcept;
    extern template ExpJet<float> eval_kernel_exp_jet<float>(float x, KernelExp mode) noexcept;

}  // namespace cxxgpr
