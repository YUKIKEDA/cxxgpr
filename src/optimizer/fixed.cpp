// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/optimizer/fixed.hpp"

#include "cxxgpr/optimizer/bounds.hpp"

#include <cmath>
#include <vector>

namespace cxxgpr::optimizer
{

    std::expected<OptimizerResult, GprError> Fixed::optimize(
        Objective& obj, std::span<const double> initial_params
    ) const
    {
        auto bounds = LogBounds::from_objective(obj);
        if (!bounds.has_value())
        {
            return std::unexpected(std::move(bounds).error());
        }
        if (auto valid = bounds->validate_log_params(initial_params); !valid.has_value())
        {
            return std::unexpected(std::move(valid).error());
        }
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
