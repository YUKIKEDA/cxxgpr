// SPDX-License-Identifier: Apache-2.0
/// @file refine_constants.hpp
/// @brief Internal acceptance rule shared by the coordinate and the supplied-distance mixed-precision engines.
#pragma once

#include <algorithm>
#include <limits>

/// @brief Internal helpers of the model engines.
namespace cxxgpr::detail
{

    /// @brief Multiple of the float epsilon accepted as storage-rounding backward error under PromoteStorage.
    /// @details The refined iterate solves the float-stored system, so against the exact double-precision kernel its
    /// relative residual is bounded by the storage rounding, a few float epsilons independent of n.
    inline constexpr double promote_storage_accept_factor = 10.0;

    /// @brief Returns the relative backward-error threshold that accepts a converged PromoteStorage iterate.
    /// @details The iterate solves the float-stored matrix to double accuracy, so against the exact kernel its
    /// backward error is the storage error. The threshold is the larger of the double-precision refinement
    /// tolerance and promote_storage_accept_factor float epsilons; a float evaluation that misses the exact kernel
    /// by more (for example coordinates far from the origin) still exceeds it and falls back.
    /// @param[in] refine_tol Double-precision relative refinement tolerance (10 n eps64).
    /// @return max(refine_tol, promote_storage_accept_factor * eps32).
    [[nodiscard]] inline double promote_storage_accept_tolerance(double refine_tol) noexcept
    {
        return std::max(
            refine_tol, promote_storage_accept_factor * static_cast<double>(std::numeric_limits<float>::epsilon())
        );
    }

}  // namespace cxxgpr::detail
