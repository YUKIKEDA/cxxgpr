// SPDX-License-Identifier: Apache-2.0
/// @file interval.hpp
/// @brief Validated open intervals and bounded parameter types in user units.
/// @details Enforces finiteness, strict ordering lo < hi, and strict interior membership at
/// construction so hyperparameters and noise variances remain strictly valid.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <expected>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Represents a finite open interval (lo, hi) with lo < hi in user parameter units.
    /// @details Prevents invalid or degenerate intervals from being constructed directly by forcing
    /// callers through validated factory functions.
    class Interval
    {
    public:
        /// @brief Returns the default open interval (1e-5, 1e5) for positive hyperparameters and noise variance.
        /// @details Provides the canonical wide positive search range used by model defaults.
        /// @return Open Interval(1e-5, 1e5).
        [[nodiscard]] static constexpr Interval default_positive() noexcept { return {1e-5, 1e5}; }

        /// @brief Builds a finite open interval requiring lo < hi and both endpoints finite.
        /// @details Allows negative or zero lower endpoints as long as both bounds are finite and lo < hi.
        /// @param lo Exclusive lower bound.
        /// @param hi Exclusive upper bound.
        /// @return Validated Interval, or GprError if bounds are invalid.
        /// @retval GprErrorCode::InvalidInterval When lo or hi is non-finite or lo >= hi.
        [[nodiscard]] static std::expected<Interval, GprError> create(double lo, double hi);

        /// @brief Builds a strictly positive finite open interval requiring 0 < lo < hi.
        /// @details Rejects non-positive lower bounds in addition to non-finite or reversed endpoints.
        /// @param lo Exclusive lower bound (must be > 0).
        /// @param hi Exclusive upper bound (must be > lo).
        /// @return Validated strictly positive Interval, or GprError if bounds are invalid.
        /// @retval GprErrorCode::InvalidInterval When lo or hi is non-finite, lo <= 0, or lo >= hi.
        [[nodiscard]] static std::expected<Interval, GprError> positive(double lo, double hi);

        /// @brief Returns the exclusive lower endpoint.
        /// @details Guaranteed finite and strictly less than hi().
        /// @return Lower bound lo.
        [[nodiscard]] constexpr double lo() const noexcept { return lo_; }

        /// @brief Returns the exclusive upper endpoint.
        /// @details Guaranteed finite and strictly greater than lo().
        /// @return Upper bound hi.
        [[nodiscard]] constexpr double hi() const noexcept { return hi_; }

        /// @brief Returns the interval width hi - lo.
        /// @details Always strictly positive for any valid Interval.
        /// @return Difference hi_ - lo_.
        [[nodiscard]] constexpr double width() const noexcept { return hi_ - lo_; }

        /// @brief Returns true if value is finite and strictly inside (lo, hi).
        /// @details Endpoints lo and hi themselves are excluded because the interval is open.
        /// @param value Scalar value to test.
        /// @return True iff std::isfinite(value) && value > lo_ && value < hi_.
        [[nodiscard]] bool contains(double value) const noexcept;

        /// @brief Checks value equality between two open intervals.
        /// @details Compares lo_ and hi_ endpoints directly.
        /// @param other Interval instance to compare against.
        /// @return True if both endpoints match.
        [[nodiscard]] constexpr bool operator==(const Interval& other) const noexcept = default;

    private:
        /// @brief Constructs an Interval from pre-validated endpoints.
        /// @details Private so all public instances satisfy finiteness and lo < hi invariants.
        /// @param lo Validated lower bound.
        /// @param hi Validated upper bound.
        constexpr Interval(double lo, double hi) noexcept : lo_(lo), hi_(hi) {}

        /// @brief Exclusive lower bound of the open interval.
        double lo_;
        /// @brief Exclusive upper bound of the open interval.
        double hi_;
    };

    /// @brief Parameter in user units guaranteed to lie strictly inside its open Interval.
    /// @details Pairs a validated scalar value with its bounding Interval for log-space conversion
    /// and bound-checked updates.
    class BoundedParam
    {
    public:
        /// @brief Builds a parameter strictly inside interval.
        /// @details Validates value via interval.contains(value).
        /// @param value Candidate parameter value in user units.
        /// @param interval Bounding open interval.
        /// @return Validated BoundedParam, or GprError if value is out of range.
        /// @retval GprErrorCode::InvalidInterval When value is non-finite or not strictly inside interval.
        [[nodiscard]] static std::expected<BoundedParam, GprError> create(double value, Interval interval);

        /// @brief Builds a parameter strictly inside Interval::default_positive().
        /// @details Equivalent to create(value, Interval::default_positive()).
        /// @param value Candidate positive parameter value in user units.
        /// @return Validated BoundedParam, or GprError if value is outside (1e-5, 1e5).
        /// @retval GprErrorCode::InvalidInterval When value is non-finite or outside Interval::default_positive().
        [[nodiscard]] static std::expected<BoundedParam, GprError> default_positive(double value);

        /// @brief Returns the parameter value in user units.
        /// @details Guaranteed finite and strictly inside interval().
        /// @return Stored parameter scalar.
        [[nodiscard]] constexpr double value() const noexcept { return value_; }

        /// @brief Returns the open interval bounding this parameter.
        /// @details Contains value() strictly in its interior.
        /// @return Bounding Interval.
        [[nodiscard]] constexpr Interval interval() const noexcept { return interval_; }

        /// @brief Returns log(value) for the optimizer's unconstrained log parameterization.
        /// @details Assumes value_ > 0 for positive hyperparameter/likelihood intervals.
        /// @return Natural logarithm std::log(value_).
        [[nodiscard]] double ln() const noexcept;

        /// @brief Rebuilds this parameter with a new value against the existing interval.
        /// @details Leaves *this unchanged; returns a new validated BoundedParam or an error.
        /// @param value Candidate replacement value in user units.
        /// @return Updated BoundedParam, or GprError if value is outside interval().
        /// @retval GprErrorCode::InvalidInterval When value is not strictly inside interval().
        [[nodiscard]] std::expected<BoundedParam, GprError> with_value(double value) const;

        /// @brief Rebuilds this parameter with a new interval checking the current value remains inside.
        /// @details Leaves *this unchanged; returns a new validated BoundedParam or an error.
        /// @param interval Replacement bounding open interval.
        /// @return Updated BoundedParam, or GprError if value() is outside interval.
        /// @retval GprErrorCode::InvalidInterval When value() is not strictly inside interval.
        [[nodiscard]] std::expected<BoundedParam, GprError> with_interval(Interval interval) const;

        /// @brief Checks value equality between two bounded parameters.
        /// @details Compares both stored value and bounding interval.
        /// @param other BoundedParam instance to compare against.
        /// @return True if value and interval both match.
        [[nodiscard]] constexpr bool operator==(const BoundedParam& other) const noexcept = default;

    private:
        /// @brief Constructs a BoundedParam from pre-validated value and interval.
        /// @details Private so every public BoundedParam preserves value in (lo, hi).
        /// @param value Validated scalar value.
        /// @param interval Bounding open interval.
        constexpr BoundedParam(double value, Interval interval) noexcept : value_(value), interval_(interval) {}

        /// @brief Parameter scalar value in user units.
        double value_;
        /// @brief Open interval strictly containing value_.
        Interval interval_;
    };

}  // namespace cxxgpr
