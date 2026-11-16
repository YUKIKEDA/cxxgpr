// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/precision/policy.hpp"

#include <cstdint>

namespace cxxgpr {

/// Selects the exponential evaluation backend used inside built-in kernel leaves.
enum class KernelExp : std::uint8_t {
    /// Exact standard library std::exp in the storage scalar.
    Accurate,
    /// Range-reduced degree-7 Taylor minimax polynomial approximation in the storage scalar.
    FastApprox,
};

/// Value and first two derivatives of exp(x) with respect to the exponent x.
template <KernelScalarConcept T> struct ExpJet {
    /// Function value approximation exp(x).
    T v = T{0};
    /// First derivative d/dx exp(x).
    T d1 = T{0};
    /// Second derivative d^2/dx^2 exp(x).
    T d2 = T{0};
};

/// Evaluates exp(x) under the selected KernelExp backend.
template <KernelScalarConcept T> [[nodiscard]] T eval_kernel_exp(T x, KernelExp mode) noexcept;

/// Evaluates exp(x) and its first two derivatives under the selected KernelExp backend.
template <KernelScalarConcept T>
[[nodiscard]] ExpJet<T> eval_kernel_exp_jet(T x, KernelExp mode) noexcept;

extern template double eval_kernel_exp<double>(double x, KernelExp mode) noexcept;
extern template float eval_kernel_exp<float>(float x, KernelExp mode) noexcept;
extern template ExpJet<double> eval_kernel_exp_jet<double>(double x, KernelExp mode) noexcept;
extern template ExpJet<float> eval_kernel_exp_jet<float>(float x, KernelExp mode) noexcept;

} // namespace cxxgpr
