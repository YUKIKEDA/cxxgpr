// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace cxxgpr::transform {

/// Base interface for a fitted target transform.
class FittedTargetTransform {
public:
    virtual ~FittedTargetTransform() = default;

    /// Applies the forward transform to target values in place.
    [[nodiscard]] virtual std::expected<void, GprError> transform(std::span<double> y) const = 0;

    /// Applies the forward transform into a preallocated output buffer.
    [[nodiscard]] virtual std::expected<void, GprError> transform_into(std::span<const double> y,
                                                                       std::span<double> out) const;

    /// Maps predictive means from transformed space back to the original target scale in place.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_transform_mean(std::span<double> mean) const = 0;

    /// Maps predictive means back to the original target scale into out.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_transform_mean_into(std::span<const double> mean, std::span<double> out) const;

    /// Maps predictive variances from transformed space back to the original target scale in place.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_transform_variance(std::span<double> var) const = 0;

    /// Maps predictive variances back to the original target scale into out.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_transform_variance_into(std::span<const double> var, std::span<double> out) const;

    /// Maps a query-query covariance matrix from transformed space back to the original scale.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_transform_covariance(std::span<double> cov) const;

    /// Maps a query-query covariance matrix back to the original scale into out.
    [[nodiscard]] virtual std::expected<void, GprError>
    inverse_transform_covariance_into(std::span<const double> cov, std::span<double> out) const;

    /// Returns a polymorphic deep copy of this fitted target transform.
    [[nodiscard]] virtual std::unique_ptr<FittedTargetTransform> clone() const = 0;

    /// Returns the registry identifier for a caller-defined transform, or empty for built-ins.
    [[nodiscard]] virtual std::string_view persist_id() const {
        return "";
    }

    /// Returns the JSON state string paired with persist_id.
    [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
};

/// Base interface for an unfitted target transform.
class UnfittedTargetTransform {
public:
    virtual ~UnfittedTargetTransform() = default;

    /// Estimates transform parameters from training targets.
    [[nodiscard]] virtual std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
    fit(std::span<const double> y) const = 0;

    /// Returns a polymorphic deep copy of this unfitted target transform.
    [[nodiscard]] virtual std::unique_ptr<UnfittedTargetTransform> clone() const = 0;

    /// Returns the registry identifier for a caller-defined transform, or empty for built-ins.
    [[nodiscard]] virtual std::string_view persist_id() const {
        return "";
    }

    /// Returns the JSON state string paired with persist_id.
    [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
};

/// Fitted identity target transform that leaves targets and predictions unchanged.
class FittedIdentityTarget final : public FittedTargetTransform {
public:
    /// Validates non-emptiness and finiteness without modifying y.
    [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

    /// Validates non-emptiness and finiteness and copies y into out.
    [[nodiscard]] std::expected<void, GprError>
    transform_into(std::span<const double> y, std::span<double> out) const override;

    /// Validates non-emptiness and finiteness without modifying mean.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean(std::span<double> mean) const override;

    /// Validates non-emptiness and finiteness and copies mean into out.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean_into(std::span<const double> mean, std::span<double> out) const override;

    /// Validates non-emptiness and finiteness without modifying var.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance(std::span<double> var) const override;

    /// Validates non-emptiness and finiteness and copies var into out.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance_into(std::span<const double> var,
                                    std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted identity target transform.
    [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

    /// Compares two FittedIdentityTarget instances for equality.
    [[nodiscard]] bool operator==(const FittedIdentityTarget& other) const noexcept = default;
};

/// Unfitted identity target transform (default).
class IdentityTarget final : public UnfittedTargetTransform {
public:
    /// Validates non-emptiness and finiteness and returns a typed fitted identity transform.
    [[nodiscard]] std::expected<FittedIdentityTarget, GprError>
    fit_typed(std::span<const double> y) const;

    /// Validates non-emptiness and finiteness and returns a boxed fitted identity transform.
    [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
    fit(std::span<const double> y) const override;

    /// Returns a polymorphic deep copy of this unfitted identity target transform.
    [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

    /// Compares two IdentityTarget instances for equality.
    [[nodiscard]] bool operator==(const IdentityTarget& other) const noexcept = default;
};

/// Fitted target standardization transform storing training mean and population std.
class FittedStandardizeTarget final : public FittedTargetTransform {
public:
    /// Constructs a FittedStandardizeTarget from finite mean and positive finite std.
    [[nodiscard]] static std::expected<FittedStandardizeTarget, GprError> from_parts(double mean,
                                                                                     double std);

    /// Returns the training mean.
    [[nodiscard]] double mean() const noexcept {
        return mean_;
    }

    /// Returns the training scale (population std, or 1.0 for constant targets).
    [[nodiscard]] double std() const noexcept {
        return std_;
    }

    /// Standardizes targets in place: (y - mean) / std.
    [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

    /// Standardizes targets into out: (y - mean) / std.
    [[nodiscard]] std::expected<void, GprError>
    transform_into(std::span<const double> y, std::span<double> out) const override;

    /// Restores predictive means in place: mean + std * z.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean(std::span<double> mean) const override;

    /// Restores predictive means into out: mean + std * z.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean_into(std::span<const double> mean, std::span<double> out) const override;

    /// Scales predictive variances in place by std^2.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance(std::span<double> var) const override;

    /// Scales predictive variances into out by std^2.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance_into(std::span<const double> var,
                                    std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted standardize target transform.
    [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

    /// Compares two FittedStandardizeTarget instances for value equality.
    [[nodiscard]] bool operator==(const FittedStandardizeTarget& other) const noexcept = default;

private:
    FittedStandardizeTarget(double mean, double std) noexcept : mean_(mean), std_(std) {}

    double mean_ = 0.0;
    double std_ = 1.0;
};

/// Unfitted target center-and-scale transform.
class StandardizeTarget final : public UnfittedTargetTransform {
public:
    /// Estimates training mean and population std, using 1.0 for constant targets.
    [[nodiscard]] std::expected<FittedStandardizeTarget, GprError>
    fit_typed(std::span<const double> y) const;

    /// Estimates training mean and population std and returns a boxed fitted transform.
    [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
    fit(std::span<const double> y) const override;

    /// Returns a polymorphic deep copy of this unfitted standardize target transform.
    [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

    /// Compares two StandardizeTarget instances for equality.
    [[nodiscard]] bool operator==(const StandardizeTarget& other) const noexcept = default;
};

/// Fitted target min-max transform onto a closed target interval.
class FittedMinMaxTarget final : public FittedTargetTransform {
public:
    /// Constructs a FittedMinMaxTarget from validated extrema and finite range [range_lo,
    /// range_hi].
    [[nodiscard]] static std::expected<FittedMinMaxTarget, GprError>
    from_parts(double data_min, double data_max, double range_lo, double range_hi);

    /// Returns the output interval [lo, hi].
    [[nodiscard]] std::pair<double, double> feature_range() const noexcept {
        return {range_lo_, range_hi_};
    }

    /// Returns the training minimum.
    [[nodiscard]] double min() const noexcept {
        return data_min_;
    }

    /// Returns the training maximum.
    [[nodiscard]] double max() const noexcept {
        return data_max_;
    }

    /// Scales targets in place onto [range_lo, range_hi].
    [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

    /// Scales targets into out onto [range_lo, range_hi].
    [[nodiscard]] std::expected<void, GprError>
    transform_into(std::span<const double> y, std::span<double> out) const override;

    /// Restores predictive means in place to the original target scale.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean(std::span<double> mean) const override;

    /// Restores predictive means into out on the original target scale.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean_into(std::span<const double> mean, std::span<double> out) const override;

    /// Scales predictive variances in place by ((data_max - data_min) / (range_hi - range_lo))^2.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance(std::span<double> var) const override;

    /// Scales predictive variances into out on the original target scale.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance_into(std::span<const double> var,
                                    std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted min-max target transform.
    [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

    /// Compares two FittedMinMaxTarget instances for value equality.
    [[nodiscard]] bool operator==(const FittedMinMaxTarget& other) const noexcept = default;

private:
    FittedMinMaxTarget(double data_min, double data_max, double range_lo, double range_hi) noexcept
        : data_min_(data_min), data_max_(data_max), range_lo_(range_lo), range_hi_(range_hi) {}

    [[nodiscard]] double data_span() const noexcept;
    [[nodiscard]] double out_span() const noexcept;

    double data_min_ = 0.0;
    double data_max_ = 0.0;
    double range_lo_ = 0.0;
    double range_hi_ = 1.0;
};

/// Unfitted target min-max transform onto [0, 1] or a validated [lo, hi] interval.
class MinMaxTarget final : public UnfittedTargetTransform {
public:
    /// Constructs a MinMaxTarget mapping onto the default interval [0, 1].
    MinMaxTarget() noexcept = default;

    /// Creates a MinMaxTarget mapping onto [lo, hi], rejecting non-finite bounds or hi <= lo.
    [[nodiscard]] static std::expected<MinMaxTarget, GprError> create(double lo, double hi);

    /// Returns the output interval [lo, hi].
    [[nodiscard]] std::pair<double, double> feature_range() const noexcept {
        return {range_lo_, range_hi_};
    }

    /// Estimates training minimum and maximum and returns a typed fitted transform.
    [[nodiscard]] std::expected<FittedMinMaxTarget, GprError>
    fit_typed(std::span<const double> y) const;

    /// Estimates training minimum and maximum and returns a boxed fitted transform.
    [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
    fit(std::span<const double> y) const override;

    /// Returns a polymorphic deep copy of this unfitted min-max target transform.
    [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

    /// Compares two MinMaxTarget instances for value equality.
    [[nodiscard]] bool operator==(const MinMaxTarget& other) const noexcept = default;

private:
    MinMaxTarget(double lo, double hi) noexcept : range_lo_(lo), range_hi_(hi) {}

    double range_lo_ = 0.0;
    double range_hi_ = 1.0;
};

} // namespace cxxgpr::transform
