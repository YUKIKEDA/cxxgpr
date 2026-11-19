// SPDX-License-Identifier: Apache-2.0
/// @file fixed.hpp
/// @brief Fixed-hyperparameter marker optimizer performing single-point objective evaluation.
/// @details Declares Fixed, which validates bounds on initial_params and evaluates the objective once
/// without updating hyperparameters.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <expected>
#include <span>

/// @brief Hyperparameter optimization interfaces, coordinate transforms, and solver implementations.
namespace cxxgpr::optimizer
{

    /// @brief Marker optimizer performing no optimization iterations (used when hyperparameters are fixed).
    /// @details Validates open bounds and evaluates obj.value(initial_params) once, returning
    /// iterations = 0 and converged = true.
    class Fixed
    {
    public:
        /// @brief Fixed does not use single-coordinate incremental updates.
        static constexpr bool uses_change_indices = false;

        /// @brief Evaluates the objective once at initial_params without taking any optimization steps.
        /// @details Validates objective intervals and initial_params containment via LogBounds before
        /// evaluating obj.value(initial_params) and confirming finiteness.
        /// @param[in,out] obj Objective instance to evaluate.
        /// @param[in] initial_params Fixed log-space hyperparameter vector of length obj.dim().
        /// @return OptimizerResult with initial_params, evaluated value, iterations = 0, and converged = true,
        /// or GprError if bounds, initial parameters, or objective evaluation fail.
        /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
        /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
        /// @retval GprErrorCode::NonFiniteInput When initial_params or the evaluated value is non-finite.
        /// @retval GprErrorCode::InvalidInterval When objective intervals are invalid or params lie outside bounds.
        [[nodiscard]] std::expected<OptimizerResult, GprError> optimize(
            Objective& obj, std::span<const double> initial_params
        ) const;

        /// @brief Checks value equality between two Fixed instances.
        /// @details Stateless marker instances always compare equal.
        /// @param[in] other Fixed instance to compare against.
        /// @return Always true.
        [[nodiscard]] constexpr bool operator==(const Fixed& other) const noexcept = default;
    };

}  // namespace cxxgpr::optimizer
