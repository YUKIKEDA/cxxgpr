// SPDX-License-Identifier: Apache-2.0
/// @file refine_constants.hpp
/// @brief Internal constants shared by the coordinate and the supplied-distance mixed-precision engines.
#pragma once

/// @brief Internal helpers of the model engines.
namespace cxxgpr::detail
{

    /// @brief Multiple of the float epsilon accepted as storage-rounding backward error under PromoteStorage.
    /// @details The refined iterate solves the float-stored system, so against the exact double-precision kernel its
    /// relative residual is bounded by the storage rounding, a few float epsilons independent of n.
    inline constexpr double promote_storage_accept_factor = 10.0;

}  // namespace cxxgpr::detail
