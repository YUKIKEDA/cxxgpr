// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <expected>
#include <span>

namespace cxxgpr::optimizer {

/// Marker optimizer performing no optimization iterations (used by Gpr<Fixed>::factor()).
class Fixed {
public:
    /// Fixed does not use single-coordinate incremental updates.
    static constexpr bool uses_change_indices = false;

    /// Evaluates the objective once at initial_params without taking any optimization steps.
    [[nodiscard]] std::expected<OptimizerResult, GprError>
    optimize(Objective& obj, std::span<const double> initial_params) const;

    /// Checks value equality between two Fixed instances.
    [[nodiscard]] constexpr bool operator==(const Fixed& other) const noexcept = default;
};

} // namespace cxxgpr::optimizer
