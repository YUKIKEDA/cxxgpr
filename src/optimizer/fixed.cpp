// SPDX-License-Identifier: Apache-2.0
/// @file fixed.cpp
/// @brief Implementation of the Fixed single-point objective evaluation marker optimizer.
/// @details Validates parameter bounds on initial_params and evaluates the objective once without
/// taking optimization steps.

#include "cxxgpr/optimizer/fixed.hpp"

#include "cxxgpr/optimizer/bounds.hpp"

#include <cmath>
#include <vector>

namespace cxxgpr::optimizer
{

    /// @brief Evaluates the objective once at initial_params without taking any optimization steps.
    /// @details Constructs LogBounds from obj, checks initial_params strictly inside open bounds,
    /// calls obj.value(initial_params), rejects non-finite outputs, and returns iterations = 0.
    /// @param[in,out] obj Objective instance to evaluate.
    /// @param[in] initial_params Fixed log-space hyperparameter vector of length obj.dim().
    /// @return OptimizerResult with initial_params, evaluated value, iterations = 0, and converged = true,
    /// or GprError if bounds, initial parameters, or objective evaluation fail.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
    /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
    /// @retval GprErrorCode::NonFiniteInput When initial_params or the evaluated value is non-finite.
    /// @retval GprErrorCode::InvalidInterval When objective intervals are invalid or params lie outside bounds.
    std::expected<OptimizerResult, GprError> Fixed::optimize(
        Objective& obj, std::span<const double> initial_params
    ) const
    {
        // Validate objective bounds and parameter membership before evaluating obj.value().
        auto bounds = LogBounds::from_objective(obj);
        if (!bounds.has_value())
        {
            return std::unexpected(std::move(bounds).error());
        }
        if (auto valid = bounds->validate_log_params(initial_params); !valid.has_value())
        {
            return std::unexpected(std::move(valid).error());
        }
        // Evaluate once and verify a finite scalar objective before marking converged at 0 iterations.
        auto val = obj.value(initial_params);
        if (!val.has_value())
        {
            return std::unexpected(std::move(val).error());
        }
        if (!std::isfinite(*val))
        {
            return std::unexpected(GprError::non_finite_input());
        }
        return OptimizerResult{
            .params = std::vector<double>(initial_params.begin(), initial_params.end()),
            .value = *val,
            .iterations = 0U,
            .converged = true,
        };
    }

}  // namespace cxxgpr::optimizer
