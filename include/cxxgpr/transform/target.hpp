// SPDX-License-Identifier: Apache-2.0
/// @file target.hpp
/// @brief Target variable transform interfaces and built-in preprocessing maps.
/// @details Defines FittedTargetTransform and UnfittedTargetTransform base classes alongside identity, standardization,
/// and min-max target transforms that map predictive means, variances, and query-query covariances back to original
/// target units.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

/// @brief Feature and target preprocessing transforms and composable pipelines.
namespace cxxgpr::transform
{

    /// @brief Base interface for a fitted target transform.
    /// @details Transforms training targets into model space and maps posterior predictive means, variances, and
    /// covariances back to user units via analytical Jacobian scaling.
    class FittedTargetTransform
    {
    public:
        /// @brief Virtual destructor enabling polymorphic deletion through base pointers.
        /// @details Ensures derived fitted target transforms release owned state cleanly.
        virtual ~FittedTargetTransform() = default;

        /// @brief Applies the forward transform to target values in place.
        /// @details Mutates y directly in place after validating non-emptiness and finiteness.
        /// @param y Mutable span of target values to transform in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> transform(std::span<double> y) const = 0;

        /// @brief Applies the forward transform into a preallocated output buffer.
        /// @details Default implementation validates y and out lengths, copies y when non-aliasing, and delegates to
        /// transform(out).
        /// @param y Read-only span of target values.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> transform_into(
            std::span<const double> y, std::span<double> out
        ) const;

        /// @brief Maps predictive means from transformed space back to the original target scale in place.
        /// @details Evaluates the inverse target map on posterior mean predictions.
        /// @param mean Mutable span of transformed predictive means updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_transform_mean(std::span<double> mean) const = 0;

        /// @brief Maps predictive means back to the original target scale into out.
        /// @details Default implementation validates lengths, copies mean when non-aliasing, and delegates to
        /// inverse_transform_mean(out).
        /// @param mean Read-only span of transformed predictive means.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_transform_mean_into(
            std::span<const double> mean, std::span<double> out
        ) const;

        /// @brief Maps predictive variances from transformed space back to the original target scale in place.
        /// @details Scales variances by the squared inverse-map derivative (dy/dz)^2.
        /// @param var Mutable span of transformed predictive variances updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_transform_variance(std::span<double> var) const = 0;

        /// @brief Maps predictive variances back to the original target scale into out.
        /// @details Default implementation validates lengths, copies var when non-aliasing, and delegates to
        /// inverse_transform_variance(out).
        /// @param var Read-only span of transformed predictive variances.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_transform_variance_into(
            std::span<const double> var, std::span<double> out
        ) const;

        /// @brief Maps a query-query covariance matrix from transformed space back to the original scale.
        /// @details Default implementation delegates to inverse_transform_variance(cov), which is exact for all affine
        /// target transforms because each entry scales by (dy/dz)^2.
        /// @param cov Mutable flattened m x m covariance buffer updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When cov is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_transform_covariance(std::span<double> cov) const;

        /// @brief Maps a query-query covariance matrix back to the original scale into out.
        /// @details Default implementation delegates to inverse_transform_variance_into(cov, out), exact for affine
        /// target transforms.
        /// @param cov Read-only flattened m x m transformed covariance buffer.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When cov is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != cov.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_transform_covariance_into(
            std::span<const double> cov, std::span<double> out
        ) const;

        /// @brief Returns a polymorphic deep copy of this fitted target transform.
        /// @details Used when copying target pipelines or fitted models.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] virtual std::unique_ptr<FittedTargetTransform> clone() const = 0;

        /// @brief Returns the registry identifier for a caller-defined transform, or empty for built-ins.
        /// @details Built-in transforms serialize through structured configuration rather than custom registry IDs.
        /// @return Registry identifier view, or empty string view for built-in transforms.
        [[nodiscard]] virtual std::string_view persist_id() const { return ""; }

        /// @brief Returns the JSON state string paired with persist_id.
        /// @details Custom persistable transforms override this method; the default returns
        /// PersistFailed(NotPersistable).
        /// @return Serialized JSON state string on success, or GprError if persistence is unsupported.
        /// @retval GprErrorCode::PersistFailed With PersistErrorKind::NotPersistable when not overridden.
        [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
    };

    /// @brief Base interface for an unfitted target transform.
    /// @details Estimates target normalization parameters from training targets without mutating the unfitted config.
    class UnfittedTargetTransform
    {
    public:
        /// @brief Virtual destructor enabling polymorphic deletion through base pointers.
        /// @details Ensures derived unfitted target transforms release owned configuration cleanly.
        virtual ~UnfittedTargetTransform() = default;

        /// @brief Estimates transform parameters from training targets.
        /// @details Validates target non-emptiness and finiteness before returning a boxed FittedTargetTransform.
        /// @param y Read-only span of training target values.
        /// @return Boxed fitted target transform on success, or GprError on invalid training targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] virtual std::expected<std::unique_ptr<FittedTargetTransform>, GprError> fit(
            std::span<const double> y
        ) const = 0;

        /// @brief Returns a polymorphic deep copy of this unfitted target transform.
        /// @details Used when copying model specifications or target pipelines.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] virtual std::unique_ptr<UnfittedTargetTransform> clone() const = 0;

        /// @brief Returns the registry identifier for a caller-defined transform, or empty for built-ins.
        /// @details Built-in transforms serialize through structured configuration rather than custom registry IDs.
        /// @return Registry identifier view, or empty string view for built-in transforms.
        [[nodiscard]] virtual std::string_view persist_id() const { return ""; }

        /// @brief Returns the JSON state string paired with persist_id.
        /// @details Custom persistable transforms override this method; the default returns
        /// PersistFailed(NotPersistable).
        /// @return Serialized JSON state string on success, or GprError if persistence is unsupported.
        /// @retval GprErrorCode::PersistFailed With PersistErrorKind::NotPersistable when not overridden.
        [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
    };

    /// @brief Fitted identity target transform that leaves targets and predictions unchanged.
    /// @details Validates non-emptiness and finiteness on all forward and inverse calls without modifying values.
    class FittedIdentityTarget final : public FittedTargetTransform
    {
    public:
        /// @brief Validates non-emptiness and finiteness without modifying y.
        /// @details Performs canonical target vector validation in place.
        /// @param y Mutable span of target values to validate.
        /// @return Empty expected on success, or GprError on empty or non-finite values.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

        /// @brief Validates non-emptiness and finiteness and copies y into out.
        /// @details Skips copying when out aliases y.
        /// @param y Read-only span of target values.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> transform_into(
            std::span<const double> y, std::span<double> out
        ) const override;

        /// @brief Validates non-emptiness and finiteness without modifying mean.
        /// @details Performs canonical target vector validation in place.
        /// @param mean Mutable span of predictive means to validate.
        /// @return Empty expected on success, or GprError on empty or non-finite values.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean(std::span<double> mean) const override;

        /// @brief Validates non-emptiness and finiteness and copies mean into out.
        /// @details Delegates directly to transform_into(mean, out).
        /// @param mean Read-only span of predictive means.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean_into(
            std::span<const double> mean, std::span<double> out
        ) const override;

        /// @brief Validates non-emptiness and finiteness without modifying var.
        /// @details Performs canonical target vector validation in place.
        /// @param var Mutable span of predictive variances to validate.
        /// @return Empty expected on success, or GprError on empty or non-finite values.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance(std::span<double> var) const override;

        /// @brief Validates non-emptiness and finiteness and copies var into out.
        /// @details Delegates directly to transform_into(var, out).
        /// @param var Read-only span of predictive variances.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance_into(
            std::span<const double> var, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted identity target transform.
        /// @details Allocates a new FittedIdentityTarget instance.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

        /// @brief Compares two FittedIdentityTarget instances for equality.
        /// @details Always returns true because FittedIdentityTarget carries no state.
        /// @param other Another FittedIdentityTarget instance to compare against.
        /// @return True for any two FittedIdentityTarget instances.
        [[nodiscard]] bool operator==(const FittedIdentityTarget& other) const noexcept = default;
    };

    /// @brief Unfitted identity target transform (default).
    /// @details Produces FittedIdentityTarget after validating training targets.
    class IdentityTarget final : public UnfittedTargetTransform
    {
    public:
        /// @brief Validates non-emptiness and finiteness and returns a typed fitted identity transform.
        /// @details Does not compute summary statistics on y.
        /// @param y Read-only span of training target values.
        /// @return FittedIdentityTarget on success, or GprError on invalid training targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<FittedIdentityTarget, GprError> fit_typed(std::span<const double> y) const;

        /// @brief Validates non-emptiness and finiteness and returns a boxed fitted identity transform.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedTargetTransform>.
        /// @param y Read-only span of training target values.
        /// @return Boxed FittedIdentityTarget on success, or GprError on invalid training targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError> fit(
            std::span<const double> y
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted identity target transform.
        /// @details Allocates a new IdentityTarget instance.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

        /// @brief Compares two IdentityTarget instances for equality.
        /// @details Always returns true because IdentityTarget carries no state.
        /// @param other Another IdentityTarget instance to compare against.
        /// @return True for any two IdentityTarget instances.
        [[nodiscard]] bool operator==(const IdentityTarget& other) const noexcept = default;
    };

    /// @brief Fitted target standardization transform storing training mean and population std.
    /// @details Transforms y -> (y - mean) / std and maps predictive means by mean + std * z and variances/covariances
    /// by std^2 * v.
    class FittedStandardizeTarget final : public FittedTargetTransform
    {
    public:
        /// @brief Constructs a FittedStandardizeTarget from finite mean and positive finite std.
        /// @details Enforces finiteness of mean and std before checking std > 0.
        /// @param mean Finite target mean offset.
        /// @param std Strictly positive finite target scale divisor.
        /// @return Validated FittedStandardizeTarget on success, or GprError on invalid parameters.
        /// @retval GprErrorCode::NonFiniteInput When mean or std is NaN or infinite.
        /// @retval GprErrorCode::InvalidHyperparameter When std <= 0.
        [[nodiscard]] static std::expected<FittedStandardizeTarget, GprError> from_parts(double mean, double std);

        /// @brief Returns the training mean.
        /// @details Subtracted in forward transforms and added back in inverse mean transforms.
        /// @return Finite target mean offset.
        [[nodiscard]] double mean() const noexcept { return mean_; }

        /// @brief Returns the training scale (population std, or 1.0 for constant targets).
        /// @details Always finite and strictly positive.
        /// @return Positive target standard deviation scale.
        [[nodiscard]] double std() const noexcept { return std_; }

        /// @brief Standardizes targets in place: (y - mean) / std.
        /// @details Validates y before applying the affine shift and scale in place.
        /// @param y Mutable span of target values updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

        /// @brief Standardizes targets into out: (y - mean) / std.
        /// @details Writes standardized values into out without heap allocation.
        /// @param y Read-only span of target values.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> transform_into(
            std::span<const double> y, std::span<double> out
        ) const override;

        /// @brief Restores predictive means in place: mean + std * z.
        /// @details Inverts standardization on transformed predictive means.
        /// @param mean Mutable span of standardized predictive means updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean(std::span<double> mean) const override;

        /// @brief Restores predictive means into out: mean + std * z.
        /// @details Writes unscaled predictive means directly into out without heap allocation.
        /// @param mean Read-only span of standardized predictive means.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean_into(
            std::span<const double> mean, std::span<double> out
        ) const override;

        /// @brief Scales predictive variances in place by std^2.
        /// @details Applies the squared Jacobian factor (dy/dz)^2 = std^2 to transformed variances.
        /// @param var Mutable span of standardized predictive variances updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance(std::span<double> var) const override;

        /// @brief Scales predictive variances into out by std^2.
        /// @details Writes original-scale predictive variances into out without heap allocation.
        /// @param var Read-only span of standardized predictive variances.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance_into(
            std::span<const double> var, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted standardize target transform.
        /// @details Allocates a copy preserving stored mean and standard deviation.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

        /// @brief Compares two FittedStandardizeTarget instances for value equality.
        /// @details Compares mean_ and std_ fields for equality.
        /// @param other Another FittedStandardizeTarget instance to compare against.
        /// @return True if both instances store equal mean and std values.
        [[nodiscard]] bool operator==(const FittedStandardizeTarget& other) const noexcept = default;

    private:
        /// @brief Private constructor storing validated target mean and positive scale.
        /// @details Called only by FittedStandardizeTarget::from_parts after validation succeeds.
        /// @param mean Validated finite target mean.
        /// @param std Validated positive finite target scale.
        FittedStandardizeTarget(double mean, double std) noexcept : mean_(mean), std_(std) {}

        /// @brief Training target mean subtracted in forward transforms.
        double mean_ = 0.0;
        /// @brief Positive target scale factor (population std or 1.0 for constant targets).
        double std_ = 1.0;
    };

    /// @brief Unfitted target center-and-scale transform.
    /// @details Computes empirical target mean and population standard deviation (using 1.0 when constant).
    class StandardizeTarget final : public UnfittedTargetTransform
    {
    public:
        /// @brief Estimates training mean and population std, using 1.0 for constant targets.
        /// @details Uses population normalization (dividing squared deviations by y.size()).
        /// @param y Read-only span of training target values.
        /// @return FittedStandardizeTarget on success, or GprError on invalid targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<FittedStandardizeTarget, GprError> fit_typed(std::span<const double> y) const;

        /// @brief Estimates training mean and population std and returns a boxed fitted transform.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedTargetTransform>.
        /// @param y Read-only span of training target values.
        /// @return Boxed FittedStandardizeTarget on success, or GprError on invalid targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError> fit(
            std::span<const double> y
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted standardize target transform.
        /// @details Allocates a new StandardizeTarget instance.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

        /// @brief Compares two StandardizeTarget instances for equality.
        /// @details Always returns true because StandardizeTarget carries no configuration state.
        /// @param other Another StandardizeTarget instance to compare against.
        /// @return True for any two StandardizeTarget instances.
        [[nodiscard]] bool operator==(const StandardizeTarget& other) const noexcept = default;
    };

    /// @brief Fitted target min-max transform onto a closed target interval.
    /// @details Maps [data_min, data_max] onto [range_lo, range_hi] with constant-target fallback span 1.0, and scales
    /// predictive variances by (data_span / out_span)^2.
    class FittedMinMaxTarget final : public FittedTargetTransform
    {
    public:
        /// @brief Constructs a FittedMinMaxTarget from validated extrema and finite range [range_lo, range_hi].
        /// @details Enforces valid finite interval range_lo < range_hi, finite extrema, and data_min <= data_max.
        /// @param data_min Minimum training target value.
        /// @param data_max Maximum training target value (must be >= data_min).
        /// @param range_lo Finite lower endpoint of the target interval.
        /// @param range_hi Finite upper endpoint of the target interval (must exceed range_lo).
        /// @return Validated FittedMinMaxTarget on success, or GprError on invalid range or extrema.
        /// @retval GprErrorCode::InvalidConfig When range_lo or range_hi is non-finite or range_hi <= range_lo.
        /// @retval GprErrorCode::NonFiniteInput When data_min or data_max is NaN or infinite.
        /// @retval GprErrorCode::InvalidHyperparameter When data_min > data_max.
        [[nodiscard]] static std::expected<FittedMinMaxTarget, GprError> from_parts(
            double data_min, double data_max, double range_lo, double range_hi
        );

        /// @brief Returns the output interval [lo, hi].
        /// @details Configured at construction via MinMaxTarget::create or default [0.0, 1.0].
        /// @return Pair {range_lo, range_hi}.
        [[nodiscard]] std::pair<double, double> feature_range() const noexcept { return {range_lo_, range_hi_}; }

        /// @brief Returns the training minimum.
        /// @details Lower bound observed during target fitting.
        /// @return Finite training target minimum.
        [[nodiscard]] double min() const noexcept { return data_min_; }

        /// @brief Returns the training maximum.
        /// @details Upper bound observed during target fitting.
        /// @return Finite training target maximum.
        [[nodiscard]] double max() const noexcept { return data_max_; }

        /// @brief Scales targets in place onto [range_lo, range_hi].
        /// @details Applies range_lo + out_span * (y - data_min) / data_span in place.
        /// @param y Mutable span of target values updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

        /// @brief Scales targets into out onto [range_lo, range_hi].
        /// @details Writes scaled targets into out without heap allocation.
        /// @param y Read-only span of target values.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> transform_into(
            std::span<const double> y, std::span<double> out
        ) const override;

        /// @brief Restores predictive means in place to the original target scale.
        /// @details Applies data_min + data_span * (mean - range_lo) / out_span in place.
        /// @param mean Mutable span of transformed predictive means updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean(std::span<double> mean) const override;

        /// @brief Restores predictive means into out on the original target scale.
        /// @details Writes original-scale predictive means into out without heap allocation.
        /// @param mean Read-only span of transformed predictive means.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean_into(
            std::span<const double> mean, std::span<double> out
        ) const override;

        /// @brief Scales predictive variances in place by (data_span / (range_hi - range_lo))^2.
        /// @details Applies the squared Jacobian factor (dy/dz)^2 using safe constant-target span fallback.
        /// @param var Mutable span of transformed predictive variances updated in place.
        /// @return Empty expected on success, or GprError on empty or non-finite input.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance(std::span<double> var) const override;

        /// @brief Scales predictive variances into out on the original target scale.
        /// @details Writes original-scale predictive variances into out without heap allocation.
        /// @param var Read-only span of transformed predictive variances.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance_into(
            std::span<const double> var, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted min-max target transform.
        /// @details Allocates a copy preserving stored extrema and target interval bounds.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

        /// @brief Compares two FittedMinMaxTarget instances for value equality.
        /// @details Compares training extrema and output interval bounds.
        /// @param other Another FittedMinMaxTarget instance to compare against.
        /// @return True if both instances store identical extrema and intervals.
        [[nodiscard]] bool operator==(const FittedMinMaxTarget& other) const noexcept = default;

    private:
        /// @brief Private constructor storing pre-validated extrema and target interval bounds.
        /// @details Called only by FittedMinMaxTarget::from_parts after validation succeeds.
        /// @param data_min Validated training target minimum.
        /// @param data_max Validated training target maximum.
        /// @param range_lo Validated lower bound of the target output interval.
        /// @param range_hi Validated upper bound of the target output interval.
        FittedMinMaxTarget(double data_min, double data_max, double range_lo, double range_hi) noexcept :
            data_min_(data_min),
            data_max_(data_max),
            range_lo_(range_lo),
            range_hi_(range_hi)
        {
        }

        /// @brief Returns the safe training data span (data_max_ - data_min_, or 1.0 when constant).
        /// @details Prevents division by zero when all training targets share the same value.
        /// @return Positive finite data span.
        [[nodiscard]] double data_span() const noexcept;

        /// @brief Returns the configured output interval width range_hi_ - range_lo_.
        /// @details Guaranteed strictly positive because range_hi_ > range_lo_ is enforced on creation.
        /// @return Positive interval width.
        [[nodiscard]] double out_span() const noexcept;

        /// @brief Minimum training target observed during fitting.
        double data_min_ = 0.0;
        /// @brief Maximum training target observed during fitting.
        double data_max_ = 0.0;
        /// @brief Lower endpoint of the target output interval.
        double range_lo_ = 0.0;
        /// @brief Upper endpoint of the target output interval.
        double range_hi_ = 1.0;
    };

    /// @brief Unfitted target min-max transform onto [0, 1] or a validated [lo, hi] interval.
    /// @details Records empirical target extrema during fit() and stores the configured output range.
    class MinMaxTarget final : public UnfittedTargetTransform
    {
    public:
        /// @brief Constructs a MinMaxTarget mapping onto the default interval [0, 1].
        /// @details Sets range_lo_ = 0.0 and range_hi_ = 1.0.
        MinMaxTarget() noexcept = default;

        /// @brief Creates a MinMaxTarget mapping onto [lo, hi], rejecting non-finite bounds or hi <= lo.
        /// @details Validates target interval bounds before constructing the transform configuration.
        /// @param lo Finite lower endpoint of the target output interval.
        /// @param hi Finite upper endpoint of the target output interval (must exceed lo).
        /// @return Validated MinMaxTarget on success, or GprErrorCode::InvalidConfig on invalid bounds.
        /// @retval GprErrorCode::InvalidConfig When lo or hi is non-finite or hi <= lo.
        [[nodiscard]] static std::expected<MinMaxTarget, GprError> create(double lo, double hi);

        /// @brief Returns the output interval [lo, hi].
        /// @details Defaults to {0.0, 1.0} when default-constructed.
        /// @return Pair {range_lo, range_hi}.
        [[nodiscard]] std::pair<double, double> feature_range() const noexcept { return {range_lo_, range_hi_}; }

        /// @brief Estimates training minimum and maximum and returns a typed fitted transform.
        /// @details Validates y before scanning extrema across all training targets.
        /// @param y Read-only span of training target values.
        /// @return FittedMinMaxTarget on success, or GprError on invalid training targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<FittedMinMaxTarget, GprError> fit_typed(std::span<const double> y) const;

        /// @brief Estimates training minimum and maximum and returns a boxed fitted transform.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedTargetTransform>.
        /// @param y Read-only span of training target values.
        /// @return Boxed FittedMinMaxTarget on success, or GprError on invalid training targets.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError> fit(
            std::span<const double> y
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted min-max target transform.
        /// @details Allocates a new MinMaxTarget preserving feature_range().
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

        /// @brief Compares two MinMaxTarget instances for value equality.
        /// @details Compares configured range_lo and range_hi endpoints.
        /// @param other Another MinMaxTarget instance to compare against.
        /// @return True if both instances share the same feature_range.
        [[nodiscard]] bool operator==(const MinMaxTarget& other) const noexcept = default;

    private:
        /// @brief Private constructor storing validated interval endpoints [lo, hi].
        /// @details Called only by MinMaxTarget::create after validate_feature_range succeeds.
        /// @param lo Validated lower bound.
        /// @param hi Validated upper bound.
        MinMaxTarget(double lo, double hi) noexcept : range_lo_(lo), range_hi_(hi) {}

        /// @brief Lower endpoint of the target output interval.
        double range_lo_ = 0.0;
        /// @brief Upper endpoint of the target output interval.
        double range_hi_ = 1.0;
    };

}  // namespace cxxgpr::transform
