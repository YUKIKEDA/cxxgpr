// SPDX-License-Identifier: Apache-2.0
/// @file target.cpp
/// @brief Implementations of built-in target variable transforms and Jacobian scaling rules.
/// @details Implements IdentityTarget, StandardizeTarget, MinMaxTarget, their fitted counterparts, and base class
/// out-of-place and persistence helpers.

#include "cxxgpr/transform/target.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "transform/detail.hpp"

namespace cxxgpr::transform
{

    namespace
    {

        /// @brief Validates a target or prediction span and applies fn element-wise in place.
        /// @details Enforces canonical vector validation before updating each element v = fn(v).
        /// @tparam Fn Callable mapping double -> double.
        /// @param values Mutable span updated in place.
        /// @param fn Element-wise scalar transformation callable.
        /// @return Empty expected on success, or GprError on empty or non-finite values.
        /// @retval GprErrorCode::EmptyInput When values is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of values is NaN or infinite.
        template <typename Fn> std::expected<void, GprError> apply_vec_in_place(std::span<double> values, Fn&& fn)
        {
            if (auto check = detail::validate_target_vector(values); !check)
            {
                return check;
            }
            for (double& v : values)
            {
                v = fn(v);
            }
            return {};
        }

        /// @brief Validates target and output spans and writes fn(values[i]) into out[i].
        /// @details Enforces canonical vector validation with output length parity before element-wise evaluation.
        /// @tparam Fn Callable mapping double -> double.
        /// @param values Read-only input span.
        /// @param out Preallocated mutable output span of matching length.
        /// @param fn Element-wise scalar transformation callable.
        /// @return Empty expected on success, or GprError on invalid length or non-finite values.
        /// @retval GprErrorCode::EmptyInput When values is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != values.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of values is NaN or infinite.
        template <typename Fn>
        std::expected<void, GprError> apply_vec_into(std::span<const double> values, std::span<double> out, Fn&& fn)
        {
            if (auto check = detail::validate_target_vector(values, out.size()); !check)
            {
                return check;
            }
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = fn(values[i]);
            }
            return {};
        }

    }  // namespace

    /// @brief Applies the forward transform into a preallocated output buffer.
    /// @details Validates input and output sizes, copies y when non-aliasing, and delegates to transform(out).
    /// @param y Read-only span of target values.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedTargetTransform::transform_into(
        std::span<const double> y, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_target_vector(y, out.size()); !check)
        {
            return check;
        }
        // Avoid self-copy when out already aliases y.
        if (out.data() != y.data())
        {
            std::ranges::copy(y, out.begin());
        }
        return transform(out);
    }

    /// @brief Maps predictive means back to the original target scale into out.
    /// @details Validates input and output sizes, copies mean when non-aliasing, and delegates to
    /// inverse_transform_mean(out).
    /// @param mean Read-only span of transformed predictive means.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedTargetTransform::inverse_transform_mean_into(
        std::span<const double> mean, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_target_vector(mean, out.size()); !check)
        {
            return check;
        }
        if (out.data() != mean.data())
        {
            std::ranges::copy(mean, out.begin());
        }
        return inverse_transform_mean(out);
    }

    /// @brief Maps predictive variances back to the original target scale into out.
    /// @details Validates input and output sizes, copies var when non-aliasing, and delegates to
    /// inverse_transform_variance(out).
    /// @param var Read-only span of transformed predictive variances.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedTargetTransform::inverse_transform_variance_into(
        std::span<const double> var, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_target_vector(var, out.size()); !check)
        {
            return check;
        }
        if (out.data() != var.data())
        {
            std::ranges::copy(var, out.begin());
        }
        return inverse_transform_variance(out);
    }

    /// @brief Maps a query-query covariance matrix from transformed space back to the original scale.
    /// @details For affine target transforms y = a * z + b, every covariance entry scales by the constant squared
    /// Jacobian (dy/dz)^2 = a^2, matching inverse_transform_variance.
    /// @param cov Mutable flattened m x m covariance buffer updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When cov is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
    std::expected<void, GprError> FittedTargetTransform::inverse_transform_covariance(std::span<double> cov) const
    {
        // Affine maps y = a*z + b scale Cov(y_i, y_j) = a^2 * Cov(z_i, z_j), identical to entrywise variance scaling.
        return inverse_transform_variance(cov);
    }

    /// @brief Maps a query-query covariance matrix back to the original scale into out.
    /// @details Delegates to inverse_transform_variance_into(cov, out), exact for affine target transforms.
    /// @param cov Read-only flattened m x m transformed covariance buffer.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When cov is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != cov.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
    std::expected<void, GprError> FittedTargetTransform::inverse_transform_covariance_into(
        std::span<const double> cov, std::span<double> out
    ) const
    {
        return inverse_transform_variance_into(cov, out);
    }

    /// @brief Returns the JSON state string paired with persist_id.
    /// @details Base fallback reporting NotPersistable for target transforms without custom JSON serialization.
    /// @return Unexpected GprErrorCode::PersistFailed with PersistErrorKind::NotPersistable.
    /// @retval GprErrorCode::PersistFailed Always returned unless overridden by a persistable subclass.
    std::expected<std::string, GprError> FittedTargetTransform::persist_state_json() const
    {
        return std::unexpected(
            GprError::persist_failed(
                PersistErrorKind::NotPersistable, "this target transform does not implement persist_state"
            )
        );
    }

    /// @brief Returns the JSON state string paired with persist_id.
    /// @details Base fallback reporting NotPersistable for target transforms without custom JSON serialization.
    /// @return Unexpected GprErrorCode::PersistFailed with PersistErrorKind::NotPersistable.
    /// @retval GprErrorCode::PersistFailed Always returned unless overridden by a persistable subclass.
    std::expected<std::string, GprError> UnfittedTargetTransform::persist_state_json() const
    {
        return std::unexpected(
            GprError::persist_failed(
                PersistErrorKind::NotPersistable, "this target transform does not implement persist_state"
            )
        );
    }

    // --- IdentityTarget / FittedIdentityTarget ---

    /// @brief Validates non-emptiness and finiteness without modifying y.
    /// @details Delegates to detail::validate_target_vector.
    /// @param y Mutable span of target values validated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite values.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedIdentityTarget::transform(std::span<double> y) const
    {
        return detail::validate_target_vector(y);
    }

    /// @brief Validates non-emptiness and finiteness and copies y into out.
    /// @details Guards aliasing via out.data() != y.data() before copying.
    /// @param y Read-only span of target values.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedIdentityTarget::transform_into(
        std::span<const double> y, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_target_vector(y, out.size()); !check)
        {
            return check;
        }
        if (out.data() != y.data())
        {
            std::ranges::copy(y, out.begin());
        }
        return {};
    }

    /// @brief Validates non-emptiness and finiteness without modifying mean.
    /// @details Delegates to detail::validate_target_vector.
    /// @param mean Mutable span of predictive means validated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite values.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedIdentityTarget::inverse_transform_mean(std::span<double> mean) const
    {
        return detail::validate_target_vector(mean);
    }

    /// @brief Validates non-emptiness and finiteness and copies mean into out.
    /// @details Delegates directly to transform_into(mean, out).
    /// @param mean Read-only span of predictive means.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedIdentityTarget::inverse_transform_mean_into(
        std::span<const double> mean, std::span<double> out
    ) const
    {
        return transform_into(mean, out);
    }

    /// @brief Validates non-emptiness and finiteness without modifying var.
    /// @details Delegates to detail::validate_target_vector.
    /// @param var Mutable span of predictive variances validated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite values.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedIdentityTarget::inverse_transform_variance(std::span<double> var) const
    {
        return detail::validate_target_vector(var);
    }

    /// @brief Validates non-emptiness and finiteness and copies var into out.
    /// @details Delegates directly to transform_into(var, out).
    /// @param var Read-only span of predictive variances.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedIdentityTarget::inverse_transform_variance_into(
        std::span<const double> var, std::span<double> out
    ) const
    {
        return transform_into(var, out);
    }

    /// @brief Returns a polymorphic deep copy of this fitted identity target transform.
    /// @details Allocates a new FittedIdentityTarget copy.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<FittedTargetTransform> FittedIdentityTarget::clone() const
    {
        return std::make_unique<FittedIdentityTarget>(*this);
    }

    /// @brief Validates non-emptiness and finiteness and returns a typed fitted identity transform.
    /// @details Validates training targets before returning a stateless FittedIdentityTarget.
    /// @param y Read-only span of training target values.
    /// @return FittedIdentityTarget on success, or GprError on invalid training targets.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<FittedIdentityTarget, GprError> IdentityTarget::fit_typed(std::span<const double> y) const
    {
        if (auto check = detail::validate_target_vector(y); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        return FittedIdentityTarget{};
    }

    /// @brief Validates non-emptiness and finiteness and returns a boxed fitted identity transform.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param y Read-only span of training target values.
    /// @return Boxed FittedIdentityTarget on success, or GprError on invalid training targets.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<std::unique_ptr<FittedTargetTransform>, GprError> IdentityTarget::fit(std::span<const double> y) const
    {
        auto typed = fit_typed(y);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedIdentityTarget>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted identity target transform.
    /// @details Allocates a new IdentityTarget copy.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<UnfittedTargetTransform> IdentityTarget::clone() const
    {
        return std::make_unique<IdentityTarget>(*this);
    }

    // --- StandardizeTarget / FittedStandardizeTarget ---

    /// @brief Constructs a FittedStandardizeTarget from finite mean and positive finite std.
    /// @details Enforces finiteness of both parameters before checking std > 0.
    /// @param mean Finite target mean offset.
    /// @param std Strictly positive finite target scale divisor.
    /// @return Validated FittedStandardizeTarget on success, or GprError on invalid parameters.
    /// @retval GprErrorCode::NonFiniteInput When mean or std is NaN or infinite.
    /// @retval GprErrorCode::InvalidHyperparameter When std <= 0.
    std::expected<FittedStandardizeTarget, GprError> FittedStandardizeTarget::from_parts(double mean, double std)
    {
        // Reject non-finite parameters before checking strict positivity of scale.
        if (!std::isfinite(mean) || !std::isfinite(std))
        {
            return std::unexpected(GprError::non_finite_input());
        }
        if (std <= 0.0)
        {
            return std::unexpected(
                GprError::invalid_hyperparameter(
                    std::format("fitted standardize target scale must be positive, got std={}", std)
                )
            );
        }
        return FittedStandardizeTarget(mean, std);
    }

    /// @brief Standardizes targets in place: (y - mean) / std.
    /// @details Applies affine centering and scaling element-wise in place.
    /// @param y Mutable span of target values updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeTarget::transform(std::span<double> y) const
    {
        const double m = mean_;
        const double s = std_;
        return apply_vec_in_place(y, [m, s](double v) noexcept { return (v - m) / s; });
    }

    /// @brief Standardizes targets into out: (y - mean) / std.
    /// @details Writes standardized targets directly into out without allocation.
    /// @param y Read-only span of target values.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeTarget::transform_into(
        std::span<const double> y, std::span<double> out
    ) const
    {
        const double m = mean_;
        const double s = std_;
        return apply_vec_into(y, out, [m, s](double v) noexcept { return (v - m) / s; });
    }

    /// @brief Restores predictive means in place: mean + std * z.
    /// @details Inverts standardization on transformed predictive means in place.
    /// @param mean Mutable span of standardized predictive means updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeTarget::inverse_transform_mean(std::span<double> mean) const
    {
        const double m = mean_;
        const double s = std_;
        return apply_vec_in_place(mean, [m, s](double v) noexcept { return m + (s * v); });
    }

    /// @brief Restores predictive means into out: mean + std * z.
    /// @details Writes unscaled predictive means directly into out without allocation.
    /// @param mean Read-only span of standardized predictive means.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeTarget::inverse_transform_mean_into(
        std::span<const double> mean, std::span<double> out
    ) const
    {
        const double m = mean_;
        const double s = std_;
        return apply_vec_into(mean, out, [m, s](double v) noexcept { return m + (s * v); });
    }

    /// @brief Scales predictive variances in place by std^2.
    /// @details For y = mean + std * z, Var(y) = (dy/dz)^2 * Var(z) = std^2 * Var(z).
    /// @param var Mutable span of standardized predictive variances updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeTarget::inverse_transform_variance(std::span<double> var) const
    {
        // Inverse Jacobian dy/dz = std_, so predictive variances scale by (dy/dz)^2 = std_^2.
        const double scale2 = std_ * std_;
        return apply_vec_in_place(var, [scale2](double v) noexcept { return v * scale2; });
    }

    /// @brief Scales predictive variances into out by std^2.
    /// @details Writes original-scale predictive variances directly into out without allocation.
    /// @param var Read-only span of standardized predictive variances.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeTarget::inverse_transform_variance_into(
        std::span<const double> var, std::span<double> out
    ) const
    {
        // Inverse Jacobian dy/dz = std_, so predictive variances scale by (dy/dz)^2 = std_^2.
        const double scale2 = std_ * std_;
        return apply_vec_into(var, out, [scale2](double v) noexcept { return v * scale2; });
    }

    /// @brief Returns a polymorphic deep copy of this fitted standardize target transform.
    /// @details Allocates a copy preserving stored mean and standard deviation.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<FittedTargetTransform> FittedStandardizeTarget::clone() const
    {
        return std::make_unique<FittedStandardizeTarget>(*this);
    }

    /// @brief Estimates training mean and population std, using 1.0 for constant targets.
    /// @details Validates y before computing two-pass population statistics.
    /// @param y Read-only span of training target values.
    /// @return FittedStandardizeTarget on success, or GprError on invalid targets.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<FittedStandardizeTarget, GprError> StandardizeTarget::fit_typed(std::span<const double> y) const
    {
        if (auto check = detail::validate_target_vector(y); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        const auto [mean, std_dev] = detail::compute_mean_and_population_std(y);
        return FittedStandardizeTarget::from_parts(mean, std_dev);
    }

    /// @brief Estimates training mean and population std and returns a boxed fitted transform.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param y Read-only span of training target values.
    /// @return Boxed FittedStandardizeTarget on success, or GprError on invalid targets.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<std::unique_ptr<FittedTargetTransform>, GprError> StandardizeTarget::fit(
        std::span<const double> y
    ) const
    {
        auto typed = fit_typed(y);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedStandardizeTarget>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted standardize target transform.
    /// @details Allocates a new StandardizeTarget copy.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<UnfittedTargetTransform> StandardizeTarget::clone() const
    {
        return std::make_unique<StandardizeTarget>(*this);
    }

    // --- MinMaxTarget / FittedMinMaxTarget ---

    /// @brief Constructs a FittedMinMaxTarget from validated extrema and finite range [range_lo, range_hi].
    /// @details Validates output interval bounds first, then extremum finiteness and ordering data_min <= data_max.
    /// @param data_min Minimum training target value.
    /// @param data_max Maximum training target value (must be >= data_min).
    /// @param range_lo Finite lower endpoint of the target interval.
    /// @param range_hi Finite upper endpoint of the target interval (must exceed range_lo).
    /// @return Validated FittedMinMaxTarget on success, or GprError on invalid range or extrema.
    /// @retval GprErrorCode::InvalidConfig When range_lo or range_hi is non-finite or range_hi <= range_lo.
    /// @retval GprErrorCode::NonFiniteInput When data_min or data_max is NaN or infinite.
    /// @retval GprErrorCode::InvalidHyperparameter When data_min > data_max.
    std::expected<FittedMinMaxTarget, GprError> FittedMinMaxTarget::from_parts(
        double data_min, double data_max, double range_lo, double range_hi
    )
    {
        // Enforce validation order: target interval range -> extremum finiteness -> data_min <= data_max.
        if (auto range_res = detail::validate_feature_range(range_lo, range_hi); !range_res)
        {
            return std::unexpected(std::move(range_res).error());
        }
        if (!std::isfinite(data_min) || !std::isfinite(data_max))
        {
            return std::unexpected(GprError::non_finite_input());
        }
        if (data_min > data_max)
        {
            return std::unexpected(
                GprError::invalid_hyperparameter(
                    std::format("fitted min-max target needs data_min <= data_max, got {} > {}", data_min, data_max)
                )
            );
        }
        return FittedMinMaxTarget(data_min, data_max, range_lo, range_hi);
    }

    /// @brief Returns the safe training data span (data_max_ - data_min_, or 1.0 when constant).
    /// @details Delegates to detail::safe_span so zero-span targets do not divide by zero.
    /// @return Positive finite data span.
    double FittedMinMaxTarget::data_span() const noexcept
    {
        return detail::safe_span(data_max_, data_min_);
    }

    /// @brief Returns the configured output interval width range_hi_ - range_lo_.
    /// @details Strictly positive because range_hi_ > range_lo_ is validated on construction.
    /// @return Positive interval width.
    double FittedMinMaxTarget::out_span() const noexcept
    {
        return range_hi_ - range_lo_;
    }

    /// @brief Scales targets in place onto [range_lo, range_hi].
    /// @details Applies range_lo + out_span * (y - data_min) / data_span element-wise in place.
    /// @param y Mutable span of target values updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxTarget::transform(std::span<double> y) const
    {
        const double min_v = data_min_;
        const double span = data_span();
        const double lo = range_lo_;
        const double out_s = out_span();
        return apply_vec_in_place(y, [min_v, span, lo, out_s](double v) noexcept {
            return lo + (out_s * (v - min_v) / span);
        });
    }

    /// @brief Scales targets into out onto [range_lo, range_hi].
    /// @details Writes scaled targets directly into out without allocation.
    /// @param y Read-only span of target values.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxTarget::transform_into(
        std::span<const double> y, std::span<double> out
    ) const
    {
        const double min_v = data_min_;
        const double span = data_span();
        const double lo = range_lo_;
        const double out_s = out_span();
        return apply_vec_into(y, out, [min_v, span, lo, out_s](double v) noexcept {
            return lo + (out_s * (v - min_v) / span);
        });
    }

    /// @brief Restores predictive means in place to the original target scale.
    /// @details Applies data_min + data_span * (z - range_lo) / out_span element-wise in place.
    /// @param mean Mutable span of transformed predictive means updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxTarget::inverse_transform_mean(std::span<double> mean) const
    {
        const double min_v = data_min_;
        const double span = data_span();
        const double lo = range_lo_;
        const double out_s = out_span();
        return apply_vec_in_place(mean, [min_v, span, lo, out_s](double v) noexcept {
            return min_v + (span * (v - lo) / out_s);
        });
    }

    /// @brief Restores predictive means into out on the original target scale.
    /// @details Writes original-scale predictive means directly into out without allocation.
    /// @param mean Read-only span of transformed predictive means.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxTarget::inverse_transform_mean_into(
        std::span<const double> mean, std::span<double> out
    ) const
    {
        const double min_v = data_min_;
        const double span = data_span();
        const double lo = range_lo_;
        const double out_s = out_span();
        return apply_vec_into(mean, out, [min_v, span, lo, out_s](double v) noexcept {
            return min_v + (span * (v - lo) / out_s);
        });
    }

    /// @brief Scales predictive variances in place by (data_span / (range_hi - range_lo))^2.
    /// @details For y = data_min + (data_span / out_span) * (z - range_lo), the squared Jacobian factor is (dy/dz)^2 =
    /// (data_span / out_span)^2.
    /// @param var Mutable span of transformed predictive variances updated in place.
    /// @return Empty expected on success, or GprError on empty or non-finite input.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxTarget::inverse_transform_variance(std::span<double> var) const
    {
        // Inverse Jacobian dy/dz = data_span() / out_span(), so variances scale by (dy/dz)^2.
        const double scale = data_span() / out_span();
        const double scale2 = scale * scale;
        return apply_vec_in_place(var, [scale2](double v) noexcept { return v * scale2; });
    }

    /// @brief Scales predictive variances into out on the original target scale.
    /// @details Writes original-scale predictive variances directly into out without allocation using (dy/dz)^2.
    /// @param var Read-only span of transformed predictive variances.
    /// @param out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on empty input, length mismatch, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxTarget::inverse_transform_variance_into(
        std::span<const double> var, std::span<double> out
    ) const
    {
        // Inverse Jacobian dy/dz = data_span() / out_span(), so variances scale by (dy/dz)^2.
        const double scale = data_span() / out_span();
        const double scale2 = scale * scale;
        return apply_vec_into(var, out, [scale2](double v) noexcept { return v * scale2; });
    }

    /// @brief Returns a polymorphic deep copy of this fitted min-max target transform.
    /// @details Allocates a copy preserving stored extrema and target interval bounds.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<FittedTargetTransform> FittedMinMaxTarget::clone() const
    {
        return std::make_unique<FittedMinMaxTarget>(*this);
    }

    /// @brief Creates a MinMaxTarget mapping onto [lo, hi], rejecting non-finite bounds or hi <= lo.
    /// @details Validates target interval bounds via detail::validate_feature_range.
    /// @param lo Finite lower endpoint of the target output interval.
    /// @param hi Finite upper endpoint of the target output interval (must exceed lo).
    /// @return Validated MinMaxTarget on success, or GprErrorCode::InvalidConfig on invalid bounds.
    /// @retval GprErrorCode::InvalidConfig When lo or hi is non-finite or hi <= lo.
    std::expected<MinMaxTarget, GprError> MinMaxTarget::create(double lo, double hi)
    {
        if (auto range_res = detail::validate_feature_range(lo, hi); !range_res)
        {
            return std::unexpected(std::move(range_res).error());
        }
        return MinMaxTarget(lo, hi);
    }

    /// @brief Estimates training minimum and maximum and returns a typed fitted transform.
    /// @details Validates y before finding target extrema with std::ranges::minmax_element.
    /// @param y Read-only span of training target values.
    /// @return FittedMinMaxTarget on success, or GprError on invalid training targets.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<FittedMinMaxTarget, GprError> MinMaxTarget::fit_typed(std::span<const double> y) const
    {
        if (auto check = detail::validate_target_vector(y); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        // Record target min and max extrema across the validated training span.
        const auto [min_it, max_it] = std::ranges::minmax_element(y);
        return FittedMinMaxTarget::from_parts(*min_it, *max_it, range_lo_, range_hi_);
    }

    /// @brief Estimates training minimum and maximum and returns a boxed fitted transform.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param y Read-only span of training target values.
    /// @return Boxed FittedMinMaxTarget on success, or GprError on invalid training targets.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    std::expected<std::unique_ptr<FittedTargetTransform>, GprError> MinMaxTarget::fit(std::span<const double> y) const
    {
        auto typed = fit_typed(y);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedMinMaxTarget>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted min-max target transform.
    /// @details Allocates a new MinMaxTarget copy preserving feature_range().
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<UnfittedTargetTransform> MinMaxTarget::clone() const
    {
        return std::make_unique<MinMaxTarget>(*this);
    }

}  // namespace cxxgpr::transform
