// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/pipeline.hpp"

#include "transform/detail.hpp"

#include <algorithm>
#include <cstddef>
#include <expected>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace cxxgpr::transform {

namespace {

template <typename Ptr> std::vector<Ptr> clone_ptr_vec(const std::vector<Ptr>& src) {
    std::vector<Ptr> out;
    out.reserve(src.size());
    for (const auto& item : src) {
        if (item != nullptr) {
            out.push_back(item->clone());
        }
    }
    return out;
}

} // namespace

// --- FittedPipeline ---

FittedPipeline::FittedPipeline(std::vector<std::unique_ptr<FittedInputTransform>> steps) noexcept
    : steps_(std::move(steps)) {}

FittedPipeline::FittedPipeline(const FittedPipeline& other) : steps_(clone_ptr_vec(other.steps_)) {}

FittedPipeline& FittedPipeline::operator=(const FittedPipeline& other) {
    if (this != &other) {
        steps_ = clone_ptr_vec(other.steps_);
    }
    return *this;
}

std::expected<void, GprError> FittedPipeline::apply(std::span<double> x, std::size_t n_rows,
                                                    std::size_t n_cols) const {
    if (steps_.empty()) {
        return FittedIdentityInput{}.apply(x, n_rows, n_cols);
    }
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check) {
        return check;
    }
    for (const auto& step : steps_) {
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        if (auto res = step->apply(x, n_rows, n_cols); !res) {
            return res;
        }
    }
    return {};
}

std::expected<void, GprError> FittedPipeline::apply_into(std::span<const double> x,
                                                         std::size_t n_rows, std::size_t n_cols,
                                                         std::span<double> out) const {
    if (steps_.empty()) {
        return FittedIdentityInput{}.apply_into(x, n_rows, n_cols, out);
    }
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size());
        !check) {
        return check;
    }
    if (steps_[0] == nullptr) {
        return std::unexpected(GprError::invalid_config("pipeline contains null step"));
    }
    if (auto res = steps_[0]->apply_into(x, n_rows, n_cols, out); !res) {
        return res;
    }
    for (std::size_t i = 1; i < steps_.size(); ++i) {
        if (steps_[i] == nullptr) {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        if (auto res = steps_[i]->apply(out, n_rows, n_cols); !res) {
            return res;
        }
    }
    return {};
}

std::expected<void, GprError> FittedPipeline::inverse_apply(std::span<double> x, std::size_t n_rows,
                                                            std::size_t n_cols) const {
    if (steps_.empty()) {
        return FittedIdentityInput{}.inverse_apply(x, n_rows, n_cols);
    }
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check) {
        return check;
    }
    for (const auto& step : std::ranges::reverse_view(steps_)) {
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        if (auto res = step->inverse_apply(x, n_rows, n_cols); !res) {
            return res;
        }
    }
    return {};
}

std::expected<void, GprError> FittedPipeline::inverse_apply_into(std::span<const double> x,
                                                                 std::size_t n_rows,
                                                                 std::size_t n_cols,
                                                                 std::span<double> out) const {
    if (steps_.empty()) {
        return FittedIdentityInput{}.inverse_apply_into(x, n_rows, n_cols, out);
    }
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size());
        !check) {
        return check;
    }
    const std::size_t last = steps_.size() - 1U;
    if (steps_[last] == nullptr) {
        return std::unexpected(GprError::invalid_config("pipeline contains null step"));
    }
    if (auto res = steps_[last]->inverse_apply_into(x, n_rows, n_cols, out); !res) {
        return res;
    }
    for (std::size_t idx = last; idx > 0; --idx) {
        const auto& step = steps_[idx - 1U];
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        if (auto res = step->inverse_apply(out, n_rows, n_cols); !res) {
            return res;
        }
    }
    return {};
}

std::unique_ptr<FittedInputTransform> FittedPipeline::clone() const {
    return std::make_unique<FittedPipeline>(*this);
}

// --- Pipeline ---

Pipeline::Pipeline(std::vector<std::unique_ptr<UnfittedInputTransform>> steps) noexcept
    : steps_(std::move(steps)) {}

Pipeline::Pipeline(const Pipeline& other) : steps_(clone_ptr_vec(other.steps_)) {}

Pipeline& Pipeline::operator=(const Pipeline& other) {
    if (this != &other) {
        steps_ = clone_ptr_vec(other.steps_);
    }
    return *this;
}

Pipeline& Pipeline::then(const UnfittedInputTransform& step) & {
    steps_.push_back(step.clone());
    return *this;
}

Pipeline&& Pipeline::then(const UnfittedInputTransform& step) && {
    steps_.push_back(step.clone());
    return std::move(*this);
}

Pipeline& Pipeline::then(std::unique_ptr<UnfittedInputTransform> step) & {
    if (step != nullptr) {
        steps_.push_back(std::move(step));
    }
    return *this;
}

Pipeline&& Pipeline::then(std::unique_ptr<UnfittedInputTransform> step) && {
    if (step != nullptr) {
        steps_.push_back(std::move(step));
    }
    return std::move(*this);
}

std::expected<FittedPipeline, GprError>
Pipeline::fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check) {
        return std::unexpected(std::move(check).error());
    }
    if (steps_.empty()) {
        return FittedPipeline{};
    }
    std::vector<double> buf(x.begin(), x.end());
    std::vector<std::unique_ptr<FittedInputTransform>> fitted;
    fitted.reserve(steps_.size());
    for (const auto& step : steps_) {
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        auto fitted_step = step->fit(buf, n_rows, n_cols);
        if (!fitted_step) {
            return std::unexpected(std::move(fitted_step).error());
        }
        if (auto res = (*fitted_step)->apply(buf, n_rows, n_cols); !res) {
            return std::unexpected(std::move(res).error());
        }
        fitted.push_back(std::move(*fitted_step));
    }
    return FittedPipeline(std::move(fitted));
}

std::expected<std::unique_ptr<FittedInputTransform>, GprError>
Pipeline::fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    auto typed = fit_typed(x, n_rows, n_cols);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedPipeline>(std::move(*typed));
}

std::unique_ptr<UnfittedInputTransform> Pipeline::clone() const {
    return std::make_unique<Pipeline>(*this);
}

// --- FittedTargetPipeline ---

FittedTargetPipeline::FittedTargetPipeline(
    std::vector<std::unique_ptr<FittedTargetTransform>> steps) noexcept
    : steps_(std::move(steps)) {}

FittedTargetPipeline::FittedTargetPipeline(const FittedTargetPipeline& other)
    : steps_(clone_ptr_vec(other.steps_)) {}

FittedTargetPipeline& FittedTargetPipeline::operator=(const FittedTargetPipeline& other) {
    if (this != &other) {
        steps_ = clone_ptr_vec(other.steps_);
    }
    return *this;
}

std::expected<void, GprError> FittedTargetPipeline::transform(std::span<double> y) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.transform(y);
    }
    if (auto check = detail::validate_target_vector(y); !check) {
        return check;
    }
    for (const auto& step : steps_) {
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
        }
        if (auto res = step->transform(y); !res) {
            return res;
        }
    }
    return {};
}

std::expected<void, GprError> FittedTargetPipeline::transform_into(std::span<const double> y,
                                                                   std::span<double> out) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.transform_into(y, out);
    }
    if (auto check = detail::validate_target_vector(y, out.size()); !check) {
        return check;
    }
    if (steps_[0] == nullptr) {
        return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
    }
    if (auto res = steps_[0]->transform_into(y, out); !res) {
        return res;
    }
    for (std::size_t i = 1; i < steps_.size(); ++i) {
        if (steps_[i] == nullptr) {
            return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
        }
        if (auto res = steps_[i]->transform(out); !res) {
            return res;
        }
    }
    return {};
}

namespace {

template <typename InPlaceFn>
std::expected<void, GprError>
reverse_target_in_place(const std::vector<std::unique_ptr<FittedTargetTransform>>& steps,
                        std::span<double> values, InPlaceFn&& fn) {
    if (auto check = detail::validate_target_vector(values); !check) {
        return check;
    }
    for (const auto& step : std::ranges::reverse_view(steps)) {
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
        }
        if (auto res = fn(*step, values); !res) {
            return res;
        }
    }
    return {};
}

template <typename IntoFn, typename InPlaceFn>
std::expected<void, GprError>
reverse_target_into(const std::vector<std::unique_ptr<FittedTargetTransform>>& steps,
                    std::span<const double> values, std::span<double> out, IntoFn&& into_fn,
                    InPlaceFn&& inplace_fn) {
    if (auto check = detail::validate_target_vector(values, out.size()); !check) {
        return check;
    }
    const std::size_t last = steps.size() - 1U;
    if (steps[last] == nullptr) {
        return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
    }
    if (auto res = into_fn(*steps[last], values, out); !res) {
        return res;
    }
    for (std::size_t idx = last; idx > 0; --idx) {
        const auto& step = steps[idx - 1U];
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
        }
        if (auto res = inplace_fn(*step, out); !res) {
            return res;
        }
    }
    return {};
}

} // namespace

std::expected<void, GprError>
FittedTargetPipeline::inverse_transform_mean(std::span<double> mean) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.inverse_transform_mean(mean);
    }
    return reverse_target_in_place(steps_, mean,
                                   [](const FittedTargetTransform& step, std::span<double> buf) {
                                       return step.inverse_transform_mean(buf);
                                   });
}

std::expected<void, GprError>
FittedTargetPipeline::inverse_transform_mean_into(std::span<const double> mean,
                                                  std::span<double> out) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.inverse_transform_mean_into(mean, out);
    }
    return reverse_target_into(
        steps_, mean, out,
        [](const FittedTargetTransform& step, std::span<const double> src, std::span<double> dst) {
            return step.inverse_transform_mean_into(src, dst);
        },
        [](const FittedTargetTransform& step, std::span<double> buf) {
            return step.inverse_transform_mean(buf);
        });
}

std::expected<void, GprError>
FittedTargetPipeline::inverse_transform_variance(std::span<double> var) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.inverse_transform_variance(var);
    }
    return reverse_target_in_place(steps_, var,
                                   [](const FittedTargetTransform& step, std::span<double> buf) {
                                       return step.inverse_transform_variance(buf);
                                   });
}

std::expected<void, GprError>
FittedTargetPipeline::inverse_transform_variance_into(std::span<const double> var,
                                                      std::span<double> out) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.inverse_transform_variance_into(var, out);
    }
    return reverse_target_into(
        steps_, var, out,
        [](const FittedTargetTransform& step, std::span<const double> src, std::span<double> dst) {
            return step.inverse_transform_variance_into(src, dst);
        },
        [](const FittedTargetTransform& step, std::span<double> buf) {
            return step.inverse_transform_variance(buf);
        });
}

std::expected<void, GprError>
FittedTargetPipeline::inverse_transform_covariance(std::span<double> cov) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.inverse_transform_covariance(cov);
    }
    return reverse_target_in_place(steps_, cov,
                                   [](const FittedTargetTransform& step, std::span<double> buf) {
                                       return step.inverse_transform_covariance(buf);
                                   });
}

std::expected<void, GprError>
FittedTargetPipeline::inverse_transform_covariance_into(std::span<const double> cov,
                                                        std::span<double> out) const {
    if (steps_.empty()) {
        return FittedIdentityTarget{}.inverse_transform_covariance_into(cov, out);
    }
    return reverse_target_into(
        steps_, cov, out,
        [](const FittedTargetTransform& step, std::span<const double> src, std::span<double> dst) {
            return step.inverse_transform_covariance_into(src, dst);
        },
        [](const FittedTargetTransform& step, std::span<double> buf) {
            return step.inverse_transform_covariance(buf);
        });
}

std::unique_ptr<FittedTargetTransform> FittedTargetPipeline::clone() const {
    return std::make_unique<FittedTargetPipeline>(*this);
}

// --- TargetPipeline ---

TargetPipeline::TargetPipeline(std::vector<std::unique_ptr<UnfittedTargetTransform>> steps) noexcept
    : steps_(std::move(steps)) {}

TargetPipeline::TargetPipeline(const TargetPipeline& other) : steps_(clone_ptr_vec(other.steps_)) {}

TargetPipeline& TargetPipeline::operator=(const TargetPipeline& other) {
    if (this != &other) {
        steps_ = clone_ptr_vec(other.steps_);
    }
    return *this;
}

TargetPipeline& TargetPipeline::then(const UnfittedTargetTransform& step) & {
    steps_.push_back(step.clone());
    return *this;
}

TargetPipeline&& TargetPipeline::then(const UnfittedTargetTransform& step) && {
    steps_.push_back(step.clone());
    return std::move(*this);
}

TargetPipeline& TargetPipeline::then(std::unique_ptr<UnfittedTargetTransform> step) & {
    if (step != nullptr) {
        steps_.push_back(std::move(step));
    }
    return *this;
}

TargetPipeline&& TargetPipeline::then(std::unique_ptr<UnfittedTargetTransform> step) && {
    if (step != nullptr) {
        steps_.push_back(std::move(step));
    }
    return std::move(*this);
}

std::expected<FittedTargetPipeline, GprError>
TargetPipeline::fit_typed(std::span<const double> y) const {
    if (auto check = detail::validate_target_vector(y); !check) {
        return std::unexpected(std::move(check).error());
    }
    if (steps_.empty()) {
        return FittedTargetPipeline{};
    }
    std::vector<double> buf(y.begin(), y.end());
    std::vector<std::unique_ptr<FittedTargetTransform>> fitted;
    fitted.reserve(steps_.size());
    for (const auto& step : steps_) {
        if (step == nullptr) {
            return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
        }
        auto fitted_step = step->fit(buf);
        if (!fitted_step) {
            return std::unexpected(std::move(fitted_step).error());
        }
        if (auto res = (*fitted_step)->transform(buf); !res) {
            return std::unexpected(std::move(res).error());
        }
        fitted.push_back(std::move(*fitted_step));
    }
    return FittedTargetPipeline(std::move(fitted));
}

std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
TargetPipeline::fit(std::span<const double> y) const {
    auto typed = fit_typed(y);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedTargetPipeline>(std::move(*typed));
}

std::unique_ptr<UnfittedTargetTransform> TargetPipeline::clone() const {
    return std::make_unique<TargetPipeline>(*this);
}

} // namespace cxxgpr::transform
