// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for SIMD finiteness verification helpers.
/// @details Inline scalar `eval_kernel_exp` and `eval_kernel_exp_jet` definitions live in `cxxgpr/kernel/exp.hpp`
/// to avoid cross-TU call overhead, while this unit exports `all_finite_slice` backed by `core/finite_simd.hpp`.
#include "cxxgpr/kernel/exp.hpp"

#include "core/finite_simd.hpp"
#include "kernel/exp_simd.hpp"

/// @brief Internal SIMD batch exponential and finiteness verification kernels.
namespace cxxgpr::detail
{
    /// @brief Check whether every binary64 element in a contiguous slice is finite.
    /// @details Delegates to `all_finite_simd<double>` (AVX-512 / AVX2 / SSE2 + scalar `std::isfinite` tail).
    /// @param[in] ptr Contiguous input buffer (`nullptr` allowed only when `len == 0`).
    /// @param[in] len Number of elements in the slice.
    /// @return `true` when all elements are finite; `false` on any `NaN` or `+/-Inf`.
    bool all_finite_slice(const double* ptr, std::size_t len) noexcept
    {
        return all_finite_simd(ptr, len);
    }

    /// @brief Check whether every binary32 element in a contiguous slice is finite.
    /// @details Delegates to `all_finite_simd<float>` (AVX-512 / AVX2 / SSE2 + scalar `std::isfinite` tail).
    /// @param[in] ptr Contiguous input buffer (`nullptr` allowed only when `len == 0`).
    /// @param[in] len Number of elements in the slice.
    /// @return `true` when all elements are finite; `false` on any `NaN` or `+/-Inf`.
    bool all_finite_slice(const float* ptr, std::size_t len) noexcept
    {
        return all_finite_simd(ptr, len);
    }
}  // namespace cxxgpr::detail
