// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/interval.hpp"

#include <cmath>
#include <expected>

namespace cxxgpr
{

    std::expected<Interval, GprError> Interval::create(double lo, double hi)
    {
        if (std::isfinite(lo) && std::isfinite(hi) && lo < hi)
        {
            return Interval{lo, hi};
        }
        return std::unexpected(GprError::invalid_interval_bounds(lo, hi));
    }

    std::expected<Interval, GprError> Interval::positive(double lo, double hi)
    {
        if (std::isfinite(lo) && std::isfinite(hi) && lo > 0.0 && lo < hi)
        {
            return Interval{lo, hi};
        }
        return std::unexpected(GprError::invalid_interval_bounds(lo, hi));
    }

    bool Interval::contains(double value) const noexcept
    {
        return std::isfinite(value) && value > lo_ && value < hi_;
    }

    std::expected<BoundedParam, GprError> BoundedParam::create(double value, Interval interval)
    {
        if (interval.contains(value))
        {
            return BoundedParam{value, interval};
        }
        return std::unexpected(GprError::invalid_interval_out_of_range(value, interval.lo(), interval.hi()));
    }

    std::expected<BoundedParam, GprError> BoundedParam::default_positive(double value)
    {
        return create(value, Interval::default_positive());
    }

    double BoundedParam::ln() const noexcept
    {
        return std::log(value_);
    }

    std::expected<BoundedParam, GprError> BoundedParam::with_value(double value) const
    {
        return create(value, interval_);
    }

    std::expected<BoundedParam, GprError> BoundedParam::with_interval(Interval interval) const
    {
        return create(value_, interval);
    }

}  // namespace cxxgpr
