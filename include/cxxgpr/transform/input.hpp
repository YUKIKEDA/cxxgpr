// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cxxgpr::transform {

/// Base interface for a fitted input feature transform.
class FittedInputTransform {
public:
    virtual ~FittedInputTransform() = default;

    /// Applies the fitted transform to a column-major feature matrix in place.
    [[nodiscard]] virtual std::expected<void, GprError>
    apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const = 0;

    /// Applies the fitted transform out of place into a preallocated output buffer.
    [[nodiscard]] virtual std::expected<void, GprError> apply_into(std::span<const double> x,
                                                                   std::size_t n_rows,
                                                                   std::size_t n_cols,
                                                                   std::span<double> out) const;

    /// Maps transformed features back to original coordinates in place.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const = 0;

    /// Maps transformed features back to original coordinates into a preallocated buffer.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_apply_into(std::span<const double> x, std::size_t n_rows, std::size_t n_cols,
                       std::span<double> out) const;

    /// Returns a polymorphic deep copy of this fitted transform.
    [[nodiscard]] virtual std::unique_ptr<FittedInputTransform> clone() const = 0;

    /// Returns the registry identifier for a caller-defined transform, or empty for built-ins.
    [[nodiscard]] virtual std::string_view persist_id() const {
        return "";
    }

    /// Returns the JSON state string paired with persist_id.
    [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
};

/// Base interface for an unfitted input feature transform.
class UnfittedInputTransform {
public:
    virtual ~UnfittedInputTransform() = default;

    /// Estimates transform parameters from column-major training features.
    [[nodiscard]] virtual std::expected<std::unique_ptr<FittedInputTransform>, GprError>
    fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const = 0;

    /// Returns a polymorphic deep copy of this unfitted transform.
    [[nodiscard]] virtual std::unique_ptr<UnfittedInputTransform> clone() const = 0;

    /// Returns the registry identifier for a caller-defined transform, or empty for built-ins.
    [[nodiscard]] virtual std::string_view persist_id() const {
        return "";
    }

    /// Returns the JSON state string paired with persist_id.
    [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
};

/// Fitted identity input transform that leaves features unchanged.
class FittedIdentityInput final : public FittedInputTransform {
public:
    /// Validates packing and finiteness without modifying x.
    [[nodiscard]] std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows,
                                                      std::size_t n_cols) const override;

    /// Validates packing and finiteness and copies x into out.
    [[nodiscard]] std::expected<void, GprError> apply_into(std::span<const double> x,
                                                           std::size_t n_rows, std::size_t n_cols,
                                                           std::span<double> out) const override;

    /// Validates packing and finiteness without modifying x.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Validates packing and finiteness and copies x into out.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply_into(std::span<const double> x, std::size_t n_rows, std::size_t n_cols,
                       std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted identity transform.
    [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

    /// Compares two FittedIdentityInput instances for equality.
    [[nodiscard]] bool operator==(const FittedIdentityInput& other) const noexcept = default;
};

/// Unfitted identity input transform (default).
class IdentityInput final : public UnfittedInputTransform {
public:
    /// Validates packing and finiteness and returns a typed fitted identity transform.
    [[nodiscard]] std::expected<FittedIdentityInput, GprError>
    fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const;

    /// Validates packing and finiteness and returns a boxed fitted identity transform.
    [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError>
    fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Returns a polymorphic deep copy of this unfitted identity transform.
    [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

    /// Compares two IdentityInput instances for equality.
    [[nodiscard]] bool operator==(const IdentityInput& other) const noexcept = default;
};

/// Fitted per-column standardization transform storing training means and population stds.
class FittedStandardizeInput final : public FittedInputTransform {
public:
    /// Constructs a FittedStandardizeInput from validated per-column means and positive stds.
    [[nodiscard]] static std::expected<FittedStandardizeInput, GprError>
    from_parts(std::vector<double> mean, std::vector<double> std);

    /// Returns per-column training means.
    [[nodiscard]] std::span<const double> mean() const noexcept {
        return mean_;
    }

    /// Returns per-column training scales (population std, or 1.0 for constant columns).
    [[nodiscard]] std::span<const double> std() const noexcept {
        return std_;
    }

    /// Standardizes features in place: (x - mean) / std.
    [[nodiscard]] std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows,
                                                      std::size_t n_cols) const override;

    /// Standardizes features into out: (x - mean) / std.
    [[nodiscard]] std::expected<void, GprError> apply_into(std::span<const double> x,
                                                           std::size_t n_rows, std::size_t n_cols,
                                                           std::span<double> out) const override;

    /// Restores original feature coordinates in place: x * std + mean.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Restores original feature coordinates into out: x * std + mean.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply_into(std::span<const double> x, std::size_t n_rows, std::size_t n_cols,
                       std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted standardize transform.
    [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

    /// Compares two FittedStandardizeInput instances for value equality.
    [[nodiscard]] bool operator==(const FittedStandardizeInput& other) const noexcept = default;

private:
    FittedStandardizeInput(std::vector<double> mean, std::vector<double> std) noexcept
        : mean_(std::move(mean)), std_(std::move(std)) {}

    std::vector<double> mean_;
    std::vector<double> std_;
};

/// Unfitted per-column center-and-scale transform.
class StandardizeInput final : public UnfittedInputTransform {
public:
    /// Estimates per-column means and population stds, using 1.0 for constant columns.
    [[nodiscard]] std::expected<FittedStandardizeInput, GprError>
    fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const;

    /// Estimates per-column means and population stds and returns a boxed fitted transform.
    [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError>
    fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Returns a polymorphic deep copy of this unfitted standardize transform.
    [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

    /// Compares two StandardizeInput instances for equality.
    [[nodiscard]] bool operator==(const StandardizeInput& other) const noexcept = default;
};

/// Fitted per-column min-max transform onto a closed target interval.
class FittedMinMaxInput final : public FittedInputTransform {
public:
    /// Constructs a FittedMinMaxInput from validated extrema and finite range [range_lo, range_hi].
    [[nodiscard]] static std::expected<FittedMinMaxInput, GprError>
    from_parts(std::vector<double> data_min, std::vector<double> data_max, double range_lo,
               double range_hi);

    /// Returns the output interval [lo, hi].
    [[nodiscard]] std::pair<double, double> feature_range() const noexcept {
        return {range_lo_, range_hi_};
    }

    /// Returns per-column training minima.
    [[nodiscard]] std::span<const double> min() const noexcept {
        return data_min_;
    }

    /// Returns per-column training maxima.
    [[nodiscard]] std::span<const double> max() const noexcept {
        return data_max_;
    }

    /// Scales features in place onto [range_lo, range_hi].
    [[nodiscard]] std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows,
                                                      std::size_t n_cols) const override;

    /// Scales features into out onto [range_lo, range_hi].
    [[nodiscard]] std::expected<void, GprError> apply_into(std::span<const double> x,
                                                           std::size_t n_rows, std::size_t n_cols,
                                                           std::span<double> out) const override;

    /// Restores original feature coordinates in place.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Restores original feature coordinates into out.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply_into(std::span<const double> x, std::size_t n_rows, std::size_t n_cols,
                       std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted min-max transform.
    [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

    /// Compares two FittedMinMaxInput instances for value equality.
    [[nodiscard]] bool operator==(const FittedMinMaxInput& other) const noexcept = default;

private:
    FittedMinMaxInput(std::vector<double> data_min, std::vector<double> data_max, double range_lo,
                      double range_hi) noexcept
        : data_min_(std::move(data_min)), data_max_(std::move(data_max)), range_lo_(range_lo),
          range_hi_(range_hi) {}

    std::vector<double> data_min_;
    std::vector<double> data_max_;
    double range_lo_ = 0.0;
    double range_hi_ = 1.0;
};

/// Unfitted per-column min-max transform onto [0, 1] or a validated [lo, hi] interval.
class MinMaxInput final : public UnfittedInputTransform {
public:
    /// Constructs a MinMaxInput mapping onto the default interval [0, 1].
    MinMaxInput() noexcept = default;

    /// Creates a MinMaxInput mapping onto [lo, hi], rejecting non-finite bounds or hi <= lo.
    [[nodiscard]] static std::expected<MinMaxInput, GprError> create(double lo, double hi);

    /// Returns the output interval [lo, hi].
    [[nodiscard]] std::pair<double, double> feature_range() const noexcept {
        return {range_lo_, range_hi_};
    }

    /// Estimates per-column minima and maxima and returns a typed fitted transform.
    [[nodiscard]] std::expected<FittedMinMaxInput, GprError>
    fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const;

    /// Estimates per-column minima and maxima and returns a boxed fitted transform.
    [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError>
    fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Returns a polymorphic deep copy of this unfitted min-max transform.
    [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

    /// Compares two MinMaxInput instances for value equality.
    [[nodiscard]] bool operator==(const MinMaxInput& other) const noexcept = default;

private:
    MinMaxInput(double lo, double hi) noexcept : range_lo_(lo), range_hi_(hi) {}

    double range_lo_ = 0.0;
    double range_hi_ = 1.0;
};

} // namespace cxxgpr::transform
