// SPDX-License-Identifier: Apache-2.0
/// @file exp.cpp
/// @brief Translation-unit support for SIMD finiteness validation across contiguous kernel slices.

#include <cstddef>

#include "core/finite_simd.hpp"
#include "kernel/exp_simd.hpp"

namespace cxxgpr::detail
{

    /// @brief Checks whether all `len` double elements starting at `ptr` are finite.
    /// @param[in] ptr Contiguous input buffer pointer.
    /// @param[in] len Number of double elements to check.
    /// @return True if every element is finite, false otherwise.
    bool all_finite_slice(const double* ptr, std::size_t len) noexcept
    {
        return all_finite_simd(ptr, len);
    }

    /// @brief Checks whether all `len` float elements starting at `ptr` are finite.
    /// @param[in] ptr Contiguous input buffer pointer.
    /// @param[in] len Number of float elements to check.
    /// @return True if every element is finite, false otherwise.
    bool all_finite_slice(const float* ptr, std::size_t len) noexcept
    {
        return all_finite_simd(ptr, len);
    }

}  // namespace cxxgpr::detail
