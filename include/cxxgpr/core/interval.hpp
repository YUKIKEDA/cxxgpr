// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <expected>

namespace cxxgpr
{

    /// Represents a finite open interval (lo, hi) with lo < hi in user parameter units.
    class Interval
    {
    public:
        /// Returns the default open interval (1e-5, 1e5) for positive hyperparameters and noise
        /// variance.
        [[nodiscard]] static constexpr Interval default_positive() noexcept { return {1e-5, 1e5}; }

        /// Builds a finite open interval requiring lo < hi and both endpoints finite.
        [[nodiscard]] static std::expected<Interval, GprError> create(double lo, double hi);

        /// Builds a strictly positive finite open interval requiring 0 < lo < hi.
        [[nodiscard]] static std::expected<Interval, GprError> positive(double lo, double hi);

        /// Returns the exclusive lower endpoint.
        [[nodiscard]] constexpr double lo() const noexcept { return lo_; }

        /// Returns the exclusive upper endpoint.
        [[nodiscard]] constexpr double hi() const noexcept { return hi_; }

        /// Returns the interval width hi - lo.
        [[nodiscard]] constexpr double width() const noexcept { return hi_ - lo_; }

        /// Returns true if value is finite and strictly inside (lo, hi).
        [[nodiscard]] bool contains(double value) const noexcept;

        /// Checks value equality between two open intervals.
        [[nodiscard]] constexpr bool operator==(const Interval& other) const noexcept = default;

    private:
        constexpr Interval(double lo, double hi) noexcept : lo_(lo), hi_(hi) {}

        double lo_;
        double hi_;
    };

    /// Parameter in user units guaranteed to lie strictly inside its open Interval.
    class BoundedParam
    {
    public:
        /// Builds a parameter strictly inside interval.
        [[nodiscard]] static std::expected<BoundedParam, GprError> create(double value, Interval interval);

        /// Builds a parameter strictly inside Interval::default_positive().
        [[nodiscard]] static std::expected<BoundedParam, GprError> default_positive(double value);

        /// Returns the parameter value in user units.
        [[nodiscard]] constexpr double value() const noexcept { return value_; }

        /// Returns the open interval bounding this parameter.
        [[nodiscard]] constexpr Interval interval() const noexcept { return interval_; }

        /// Returns log(value) for the optimizer's unconstrained log parameterization.
        [[nodiscard]] double ln() const noexcept;

        /// Rebuilds this parameter with a new value against the existing interval.
        [[nodiscard]] std::expected<BoundedParam, GprError> with_value(double value) const;

        /// Rebuilds this parameter with a new interval checking the current value remains inside.
        [[nodiscard]] std::expected<BoundedParam, GprError> with_interval(Interval interval) const;

        /// Checks value equality between two bounded parameters.
        [[nodiscard]] constexpr bool operator==(const BoundedParam& other) const noexcept = default;

    private:
        constexpr BoundedParam(double value, Interval interval) noexcept : value_(value), interval_(interval) {}

        double value_;
        Interval interval_;
    };

}  // namespace cxxgpr
