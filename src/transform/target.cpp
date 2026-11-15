// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/target.hpp"

#include "transform/detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace cxxgpr::transform {

namespace {

template <typename Fn>
std::expected<void, GprError> apply_vec_in_place(std::span<double> values, Fn&& fn) {
    if (auto check = detail::validate_target_vector(values); !check) {
        return check;
    }
    for (double& v : values) {
        v = fn(v);
    }
    return {};
}

template <typename Fn>
std::expected<void, GprError> apply_vec_into(std::span<const double> values, std::span<double> out,
                                             Fn&& fn) {
    if (auto check = detail::validate_target_vector(values, out.size()); !check) {
        return check;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = fn(values[i]);
    }
    return {};
}

} // namespace

std::expected<void, GprError> FittedTargetTransform::transform_into(std::span<const double> y,
                                                                    std::span<double> out) const {
    if (auto check = detail::validate_target_vector(y, out.size()); !check) {
        return check;
    }
    if (out.data() != y.data()) {
        std::ranges::copy(y, out.begin());
    }
    return transform(out);
}

std::expected<void, GprError>
FittedTargetTransform::inverse_transform_mean_into(std::span<const double> mean,
                                                   std::span<double> out) const {
    if (auto check = detail::validate_target_vector(mean, out.size()); !check) {
        return check;
    }
    if (out.data() != mean.data()) {
        std::ranges::copy(mean, out.begin());
    }
    return inverse_transform_mean(out);
}

std::expected<void, GprError>
FittedTargetTransform::inverse_transform_variance_into(std::span<const double> var,
                                                       std::span<double> out) const {
    if (auto check = detail::validate_target_vector(var, out.size()); !check) {
        return check;
    }
    if (out.data() != var.data()) {
        std::ranges::copy(var, out.begin());
    }
    return inverse_transform_variance(out);
}

std::expected<void, GprError>
FittedTargetTransform::inverse_transform_covariance(std::span<double> cov) const {
    return inverse_transform_variance(cov);
}

std::expected<void, GprError>
FittedTargetTransform::inverse_transform_covariance_into(std::span<const double> cov,
                                                         std::span<double> out) const {
    return inverse_transform_variance_into(cov, out);
}

std::expected<std::string, GprError> FittedTargetTransform::persist_state_json() const {
    return std::unexpected(
        GprError::persist_failed(PersistErrorKind::NotPersistable,
                                 "this target transform does not implement persist_state"));
}

std::expected<std::string, GprError> UnfittedTargetTransform::persist_state_json() const {
    return std::unexpected(
        GprError::persist_failed(PersistErrorKind::NotPersistable,
                                 "this target transform does not implement persist_state"));
}

// --- IdentityTarget / FittedIdentityTarget ---

std::expected<void, GprError> FittedIdentityTarget::transform(std::span<double> y) const {
    return detail::validate_target_vector(y);
}

std::expected<void, GprError> FittedIdentityTarget::transform_into(std::span<const double> y,
                                                                   std::span<double> out) const {
    if (auto check = detail::validate_target_vector(y, out.size()); !check) {
        return check;
    }
    if (out.data() != y.data()) {
        std::ranges::copy(y, out.begin());
    }
    return {};
}

std::expected<void, GprError>
FittedIdentityTarget::inverse_transform_mean(std::span<double> mean) const {
    return detail::validate_target_vector(mean);
}

std::expected<void, GprError>
FittedIdentityTarget::inverse_transform_mean_into(std::span<const double> mean,
                                                  std::span<double> out) const {
    return transform_into(mean, out);
}

std::expected<void, GprError>
FittedIdentityTarget::inverse_transform_variance(std::span<double> var) const {
    return detail::validate_target_vector(var);
}

std::expected<void, GprError>
FittedIdentityTarget::inverse_transform_variance_into(std::span<const double> var,
                                                      std::span<double> out) const {
    return transform_into(var, out);
}

std::unique_ptr<FittedTargetTransform> FittedIdentityTarget::clone() const {
    return std::make_unique<FittedIdentityTarget>(*this);
}

std::expected<FittedIdentityTarget, GprError>
IdentityTarget::fit_typed(std::span<const double> y) const {
    if (auto check = detail::validate_target_vector(y); !check) {
        return std::unexpected(std::move(check).error());
    }
    return FittedIdentityTarget{};
}

std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
IdentityTarget::fit(std::span<const double> y) const {
    auto typed = fit_typed(y);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedIdentityTarget>(std::move(*typed));
}

std::unique_ptr<UnfittedTargetTransform> IdentityTarget::clone() const {
    return std::make_unique<IdentityTarget>(*this);
}

// --- StandardizeTarget / FittedStandardizeTarget ---

std::expected<FittedStandardizeTarget, GprError> FittedStandardizeTarget::from_parts(double mean,
                                                                                     double std) {
    if (!std::isfinite(mean) || !std::isfinite(std)) {
        return std::unexpected(GprError::non_finite_input());
    }
    if (std <= 0.0) {
        return std::unexpected(GprError::invalid_hyperparameter(
            std::format("fitted standardize target scale must be positive, got std={}", std)));
    }
    return FittedStandardizeTarget(mean, std);
}

std::expected<void, GprError> FittedStandardizeTarget::transform(std::span<double> y) const {
    const double m = mean_;
    const double s = std_;
    return apply_vec_in_place(y, [m, s](double v) noexcept { return (v - m) / s; });
}

std::expected<void, GprError> FittedStandardizeTarget::transform_into(std::span<const double> y,
                                                                      std::span<double> out) const {
    const double m = mean_;
    const double s = std_;
    return apply_vec_into(y, out, [m, s](double v) noexcept { return (v - m) / s; });
}

std::expected<void, GprError>
FittedStandardizeTarget::inverse_transform_mean(std::span<double> mean) const {
    const double m = mean_;
    const double s = std_;
    return apply_vec_in_place(mean, [m, s](double v) noexcept { return m + (s * v); });
}

std::expected<void, GprError>
FittedStandardizeTarget::inverse_transform_mean_into(std::span<const double> mean,
                                                     std::span<double> out) const {
    const double m = mean_;
    const double s = std_;
    return apply_vec_into(mean, out, [m, s](double v) noexcept { return m + (s * v); });
}

std::expected<void, GprError>
FittedStandardizeTarget::inverse_transform_variance(std::span<double> var) const {
    const double scale2 = std_ * std_;
    return apply_vec_in_place(var, [scale2](double v) noexcept { return v * scale2; });
}

std::expected<void, GprError>
FittedStandardizeTarget::inverse_transform_variance_into(std::span<const double> var,
                                                         std::span<double> out) const {
    const double scale2 = std_ * std_;
    return apply_vec_into(var, out, [scale2](double v) noexcept { return v * scale2; });
}

std::unique_ptr<FittedTargetTransform> FittedStandardizeTarget::clone() const {
    return std::make_unique<FittedStandardizeTarget>(*this);
}

std::expected<FittedStandardizeTarget, GprError>
StandardizeTarget::fit_typed(std::span<const double> y) const {
    if (auto check = detail::validate_target_vector(y); !check) {
        return std::unexpected(std::move(check).error());
    }
    const auto [mean, std_dev] = detail::compute_mean_and_population_std(y);
    return FittedStandardizeTarget::from_parts(mean, std_dev);
}

std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
StandardizeTarget::fit(std::span<const double> y) const {
    auto typed = fit_typed(y);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedStandardizeTarget>(std::move(*typed));
}

std::unique_ptr<UnfittedTargetTransform> StandardizeTarget::clone() const {
    return std::make_unique<StandardizeTarget>(*this);
}

// --- MinMaxTarget / FittedMinMaxTarget ---

std::expected<FittedMinMaxTarget, GprError>
FittedMinMaxTarget::from_parts(double data_min, double data_max, double range_lo, double range_hi) {
    if (auto range_res = detail::validate_feature_range(range_lo, range_hi); !range_res) {
        return std::unexpected(std::move(range_res).error());
    }
    if (!std::isfinite(data_min) || !std::isfinite(data_max)) {
        return std::unexpected(GprError::non_finite_input());
    }
    if (data_min > data_max) {
        return std::unexpected(GprError::invalid_hyperparameter(std::format(
            "fitted min-max target needs data_min <= data_max, got {} > {}", data_min, data_max)));
    }
    return FittedMinMaxTarget(data_min, data_max, range_lo, range_hi);
}

double FittedMinMaxTarget::data_span() const noexcept {
    return detail::safe_span(data_max_, data_min_);
}

double FittedMinMaxTarget::out_span() const noexcept {
    return range_hi_ - range_lo_;
}

std::expected<void, GprError> FittedMinMaxTarget::transform(std::span<double> y) const {
    const double min_v = data_min_;
    const double span = data_span();
    const double lo = range_lo_;
    const double out_s = out_span();
    return apply_vec_in_place(y, [min_v, span, lo, out_s](double v) noexcept {
        return lo + (out_s * (v - min_v) / span);
    });
}

std::expected<void, GprError> FittedMinMaxTarget::transform_into(std::span<const double> y,
                                                                 std::span<double> out) const {
    const double min_v = data_min_;
    const double span = data_span();
    const double lo = range_lo_;
    const double out_s = out_span();
    return apply_vec_into(y, out, [min_v, span, lo, out_s](double v) noexcept {
        return lo + (out_s * (v - min_v) / span);
    });
}

std::expected<void, GprError>
FittedMinMaxTarget::inverse_transform_mean(std::span<double> mean) const {
    const double min_v = data_min_;
    const double span = data_span();
    const double lo = range_lo_;
    const double out_s = out_span();
    return apply_vec_in_place(mean, [min_v, span, lo, out_s](double v) noexcept {
        return min_v + (span * (v - lo) / out_s);
    });
}

std::expected<void, GprError>
FittedMinMaxTarget::inverse_transform_mean_into(std::span<const double> mean,
                                                std::span<double> out) const {
    const double min_v = data_min_;
    const double span = data_span();
    const double lo = range_lo_;
    const double out_s = out_span();
    return apply_vec_into(mean, out, [min_v, span, lo, out_s](double v) noexcept {
        return min_v + (span * (v - lo) / out_s);
    });
}

std::expected<void, GprError>
FittedMinMaxTarget::inverse_transform_variance(std::span<double> var) const {
    const double scale = data_span() / out_span();
    const double scale2 = scale * scale;
    return apply_vec_in_place(var, [scale2](double v) noexcept { return v * scale2; });
}

std::expected<void, GprError>
FittedMinMaxTarget::inverse_transform_variance_into(std::span<const double> var,
                                                    std::span<double> out) const {
    const double scale = data_span() / out_span();
    const double scale2 = scale * scale;
    return apply_vec_into(var, out, [scale2](double v) noexcept { return v * scale2; });
}

std::unique_ptr<FittedTargetTransform> FittedMinMaxTarget::clone() const {
    return std::make_unique<FittedMinMaxTarget>(*this);
}

std::expected<MinMaxTarget, GprError> MinMaxTarget::create(double lo, double hi) {
    if (auto range_res = detail::validate_feature_range(lo, hi); !range_res) {
        return std::unexpected(std::move(range_res).error());
    }
    return MinMaxTarget(lo, hi);
}

std::expected<FittedMinMaxTarget, GprError>
MinMaxTarget::fit_typed(std::span<const double> y) const {
    if (auto check = detail::validate_target_vector(y); !check) {
        return std::unexpected(std::move(check).error());
    }
    const auto [min_it, max_it] = std::ranges::minmax_element(y);
    return FittedMinMaxTarget::from_parts(*min_it, *max_it, range_lo_, range_hi_);
}

std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
MinMaxTarget::fit(std::span<const double> y) const {
    auto typed = fit_typed(y);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedMinMaxTarget>(std::move(*typed));
}

std::unique_ptr<UnfittedTargetTransform> MinMaxTarget::clone() const {
    return std::make_unique<MinMaxTarget>(*this);
}

} // namespace cxxgpr::transform
