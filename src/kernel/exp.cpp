// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for batch FastApprox exponential evaluation.
/// @details Inline scalar `eval_kernel_exp` and `eval_kernel_exp_jet` definitions live in `cxxgpr/kernel/exp.hpp`
/// to avoid cross-TU call overhead, while this unit compiles and validates SSE2 batch exponential kernels.
#include "cxxgpr/kernel/exp.hpp"

#include "kernel/exp_simd.hpp"

/// @brief Internal SIMD batch exponential instantiation anchor keeping `src/kernel/exp.cpp` non-empty.
namespace cxxgpr::detail
{
    template void exp_scaled_nonpos_in_place<double>(double*, std::size_t, double, KernelExp) noexcept;
    template void exp_scaled_nonpos_in_place<float>(float*, std::size_t, float, KernelExp) noexcept;
}  // namespace cxxgpr::detail
