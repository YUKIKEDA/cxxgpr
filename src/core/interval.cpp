// SPDX-License-Identifier: Apache-2.0
/// @file interval.cpp
/// @brief Implementation of open Interval and BoundedParam validation helpers.
/// @details Validates finiteness, endpoint ordering, open interior containment, and log transforms.

#include "cxxgpr/core/interval.hpp"

#include <cmath>
#include <expected>

namespace cxxgpr
{

    /// @brief Builds a finite open interval requiring lo < hi and both endpoints finite.
    /// @details Returns InvalidInterval with InvalidBounds when either endpoint is non-finite or lo >= hi.
    /// @param lo Exclusive lower bound.
    /// @param hi Exclusive upper bound.
    /// @return Validated Interval, or GprError on invalid bounds.
    /// @retval GprErrorCode::InvalidInterval When lo or hi is non-finite or lo >= hi.
    std::expected<Interval, GprError> Interval::create(double lo, double hi)
    {
        // Reject NaN, +/-Inf, empty intervals (lo == hi), and inverted bounds (lo > hi).
        if (std::isfinite(lo) && std::isfinite(hi) && lo < hi)
        {
            return Interval{lo, hi};
        }
        return std::unexpected(GprError::invalid_interval_bounds(lo, hi));
    }

    /// @brief Builds a strictly positive finite open interval requiring 0 < lo < hi.
    /// @details Enforces lo > 0.0 in addition to finiteness and strict endpoint ordering.
    /// @param lo Exclusive positive lower bound.
    /// @param hi Exclusive upper bound.
    /// @return Validated strictly positive Interval, or GprError on invalid bounds.
    /// @retval GprErrorCode::InvalidInterval When lo or hi is non-finite, lo <= 0, or lo >= hi.
    std::expected<Interval, GprError> Interval::positive(double lo, double hi)
    {
        // Positive intervals require strictly positive lower bounds so log(x) is finite inside (lo, hi).
        if (std::isfinite(lo) && std::isfinite(hi) && lo > 0.0 && lo < hi)
        {
            return Interval{lo, hi};
        }
        return std::unexpected(GprError::invalid_interval_bounds(lo, hi));
    }

    /// @brief Returns true if value is finite and strictly inside (lo, hi).
    /// @details Tests open-interval bounds strictly (lo_ < value < hi_).
    /// @param value Scalar value to test.
    /// @return True iff value is finite and strictly between lo_ and hi_.
    bool Interval::contains(double value) const noexcept
    {
        // Open interval excludes endpoints lo_ and hi_ as well as NaN/Inf.
        return std::isfinite(value) && value > lo_ && value < hi_;
    }

    /// @brief Builds a parameter strictly inside interval.
    /// @details Rejects values outside the open interval with InvalidInterval (OutOfRange).
    /// @param value Candidate parameter value in user units.
    /// @param interval Bounding open interval.
    /// @return Validated BoundedParam, or GprError when value is out of range.
    /// @retval GprErrorCode::InvalidInterval When interval.contains(value) is false.
    std::expected<BoundedParam, GprError> BoundedParam::create(double value, Interval interval)
    {
        // Validate strict interior membership before wrapping in BoundedParam.
        if (interval.contains(value))
        {
            return BoundedParam{value, interval};
        }
        return std::unexpected(GprError::invalid_interval_out_of_range(value, interval.lo(), interval.hi()));
    }

    /// @brief Builds a parameter strictly inside Interval::default_positive().
    /// @details Delegates to create() using Interval::default_positive().
    /// @param value Candidate positive parameter value in user units.
    /// @return Validated BoundedParam, or GprError when outside default bounds.
    /// @retval GprErrorCode::InvalidInterval When value is outside (1e-5, 1e5).
    std::expected<BoundedParam, GprError> BoundedParam::default_positive(double value)
    {
        return create(value, Interval::default_positive());
    }

    /// @brief Returns log(value) for the optimizer's unconstrained log parameterization.
    /// @details Computes std::log(value_) without re-checking bounds.
    /// @return Natural logarithm of value_.
    double BoundedParam::ln() const noexcept
    {
        return std::log(value_);
    }

    /// @brief Rebuilds this parameter with a new value against the existing interval.
    /// @details Validates value against interval_ without mutating *this.
    /// @param value Candidate replacement scalar value.
    /// @return New BoundedParam with updated value, or GprError on range violation.
    /// @retval GprErrorCode::InvalidInterval When value is outside interval_.
    std::expected<BoundedParam, GprError> BoundedParam::with_value(double value) const
    {
        return create(value, interval_);
    }

    /// @brief Rebuilds this parameter with a new interval checking the current value remains inside.
    /// @details Validates value_ against the replacement interval without mutating *this.
    /// @param interval Replacement bounding interval.
    /// @return New BoundedParam with updated interval, or GprError if value_ falls outside.
    /// @retval GprErrorCode::InvalidInterval When value_ is outside interval.
    std::expected<BoundedParam, GprError> BoundedParam::with_interval(Interval interval) const
    {
        return create(value_, interval);
    }

}  // namespace cxxgpr
