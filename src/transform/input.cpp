// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/input.hpp"

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
#include <vector>

namespace cxxgpr::transform {

namespace {

template <typename Fn>
std::expected<void, GprError> apply_columns_in_place(std::span<double> x, std::size_t n_rows,
                                                     std::size_t n_cols, std::size_t expected_cols,
                                                     Fn&& col_fn) {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, expected_cols); !check) {
        return check;
    }
    for (std::size_t col = 0; col < n_cols; ++col) {
        auto col_span = x.subspan(col * n_rows, n_rows);
        col_fn(col, col_span);
    }
    return {};
}

template <typename Fn>
std::expected<void, GprError> apply_columns_into(std::span<const double> x, std::size_t n_rows,
                                                 std::size_t n_cols, std::size_t expected_cols,
                                                 std::span<double> out, Fn&& col_fn) {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, expected_cols, out.size());
        !check) {
        return check;
    }
    for (std::size_t col = 0; col < n_cols; ++col) {
        auto src_col = x.subspan(col * n_rows, n_rows);
        auto dst_col = out.subspan(col * n_rows, n_rows);
        col_fn(col, src_col, dst_col);
    }
    return {};
}

} // namespace

std::expected<void, GprError> FittedInputTransform::apply_into(std::span<const double> x,
                                                               std::size_t n_rows,
                                                               std::size_t n_cols,
                                                               std::span<double> out) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size());
        !check) {
        return check;
    }
    if (out.data() != x.data()) {
        std::ranges::copy(x, out.begin());
    }
    return apply(out, n_rows, n_cols);
}

std::expected<void, GprError>
FittedInputTransform::inverse_apply_into(std::span<const double> x, std::size_t n_rows,
                                         std::size_t n_cols, std::span<double> out) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size());
        !check) {
        return check;
    }
    if (out.data() != x.data()) {
        std::ranges::copy(x, out.begin());
    }
    return inverse_apply(out, n_rows, n_cols);
}

std::expected<std::string, GprError> FittedInputTransform::persist_state_json() const {
    return std::unexpected(GprError::persist_failed(
        PersistErrorKind::NotPersistable, "this input transform does not implement persist_state"));
}

std::expected<std::string, GprError> UnfittedInputTransform::persist_state_json() const {
    return std::unexpected(GprError::persist_failed(
        PersistErrorKind::NotPersistable, "this input transform does not implement persist_state"));
}

// --- IdentityInput / FittedIdentityInput ---

std::expected<void, GprError> FittedIdentityInput::apply(std::span<double> x, std::size_t n_rows,
                                                         std::size_t n_cols) const {
    return detail::validate_input_matrix(x, n_rows, n_cols);
}

std::expected<void, GprError> FittedIdentityInput::apply_into(std::span<const double> x,
                                                              std::size_t n_rows,
                                                              std::size_t n_cols,
                                                              std::span<double> out) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size());
        !check) {
        return check;
    }
    if (out.data() != x.data()) {
        std::ranges::copy(x, out.begin());
    }
    return {};
}

std::expected<void, GprError> FittedIdentityInput::inverse_apply(std::span<double> x,
                                                                 std::size_t n_rows,
                                                                 std::size_t n_cols) const {
    return apply(x, n_rows, n_cols);
}

std::expected<void, GprError> FittedIdentityInput::inverse_apply_into(std::span<const double> x,
                                                                      std::size_t n_rows,
                                                                      std::size_t n_cols,
                                                                      std::span<double> out) const {
    return apply_into(x, n_rows, n_cols, out);
}

std::unique_ptr<FittedInputTransform> FittedIdentityInput::clone() const {
    return std::make_unique<FittedIdentityInput>(*this);
}

std::expected<FittedIdentityInput, GprError>
IdentityInput::fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check) {
        return std::unexpected(std::move(check).error());
    }
    return FittedIdentityInput{};
}

std::expected<std::unique_ptr<FittedInputTransform>, GprError>
IdentityInput::fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    auto typed = fit_typed(x, n_rows, n_cols);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedIdentityInput>(std::move(*typed));
}

std::unique_ptr<UnfittedInputTransform> IdentityInput::clone() const {
    return std::make_unique<IdentityInput>(*this);
}

// --- StandardizeInput / FittedStandardizeInput ---

std::expected<FittedStandardizeInput, GprError>
FittedStandardizeInput::from_parts(std::vector<double> mean, std::vector<double> std) {
    if (mean.empty() || std.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (auto len_res = require_count(std.size(), mean.size(), "std"); !len_res) {
        return std::unexpected(std::move(len_res).error());
    }
    if (auto fin_mean = require_finite(mean); !fin_mean) {
        return std::unexpected(std::move(fin_mean).error());
    }
    if (auto fin_std = require_finite(std); !fin_std) {
        return std::unexpected(std::move(fin_std).error());
    }
    for (const double s : std) {
        if (s <= 0.0) {
            return std::unexpected(
                GprError::invalid_hyperparameter("fitted standardize scale must be positive"));
        }
    }
    return FittedStandardizeInput(std::move(mean), std::move(std));
}

std::expected<void, GprError> FittedStandardizeInput::apply(std::span<double> x, std::size_t n_rows,
                                                            std::size_t n_cols) const {
    return apply_columns_in_place(x, n_rows, n_cols, mean_.size(),
                                  [this](std::size_t col, std::span<double> col_span) noexcept {
                                      const double m = mean_[col];
                                      const double s = std_[col];
                                      for (double& v : col_span) {
                                          v = (v - m) / s;
                                      }
                                  });
}

std::expected<void, GprError> FittedStandardizeInput::apply_into(std::span<const double> x,
                                                                 std::size_t n_rows,
                                                                 std::size_t n_cols,
                                                                 std::span<double> out) const {
    return apply_columns_into(
        x, n_rows, n_cols, mean_.size(), out,
        [this](std::size_t col, std::span<const double> src, std::span<double> dst) noexcept {
            const double m = mean_[col];
            const double s = std_[col];
            for (std::size_t i = 0; i < src.size(); ++i) {
                dst[i] = (src[i] - m) / s;
            }
        });
}

std::expected<void, GprError> FittedStandardizeInput::inverse_apply(std::span<double> x,
                                                                    std::size_t n_rows,
                                                                    std::size_t n_cols) const {
    return apply_columns_in_place(x, n_rows, n_cols, mean_.size(),
                                  [this](std::size_t col, std::span<double> col_span) noexcept {
                                      const double m = mean_[col];
                                      const double s = std_[col];
                                      for (double& v : col_span) {
                                          v = (v * s) + m;
                                      }
                                  });
}

std::expected<void, GprError>
FittedStandardizeInput::inverse_apply_into(std::span<const double> x, std::size_t n_rows,
                                           std::size_t n_cols, std::span<double> out) const {
    return apply_columns_into(
        x, n_rows, n_cols, mean_.size(), out,
        [this](std::size_t col, std::span<const double> src, std::span<double> dst) noexcept {
            const double m = mean_[col];
            const double s = std_[col];
            for (std::size_t i = 0; i < src.size(); ++i) {
                dst[i] = (src[i] * s) + m;
            }
        });
}

std::unique_ptr<FittedInputTransform> FittedStandardizeInput::clone() const {
    return std::make_unique<FittedStandardizeInput>(*this);
}

std::expected<FittedStandardizeInput, GprError>
StandardizeInput::fit_typed(std::span<const double> x, std::size_t n_rows,
                            std::size_t n_cols) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check) {
        return std::unexpected(std::move(check).error());
    }
    std::vector<double> mean;
    std::vector<double> std_dev;
    mean.reserve(n_cols);
    std_dev.reserve(n_cols);
    for (std::size_t col = 0; col < n_cols; ++col) {
        const auto [m, s] =
            detail::compute_mean_and_population_std(x.subspan(col * n_rows, n_rows));
        mean.push_back(m);
        std_dev.push_back(s);
    }
    return FittedStandardizeInput::from_parts(std::move(mean), std::move(std_dev));
}

std::expected<std::unique_ptr<FittedInputTransform>, GprError>
StandardizeInput::fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    auto typed = fit_typed(x, n_rows, n_cols);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedStandardizeInput>(std::move(*typed));
}

std::unique_ptr<UnfittedInputTransform> StandardizeInput::clone() const {
    return std::make_unique<StandardizeInput>(*this);
}

// --- MinMaxInput / FittedMinMaxInput ---

std::expected<FittedMinMaxInput, GprError>
FittedMinMaxInput::from_parts(std::vector<double> data_min, std::vector<double> data_max,
                              double range_lo, double range_hi) {
    if (data_min.empty() || data_max.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (auto len_res = require_count(data_max.size(), data_min.size(), "data_max"); !len_res) {
        return std::unexpected(std::move(len_res).error());
    }
    if (auto fin_min = require_finite(data_min); !fin_min) {
        return std::unexpected(std::move(fin_min).error());
    }
    if (auto fin_max = require_finite(data_max); !fin_max) {
        return std::unexpected(std::move(fin_max).error());
    }
    if (auto range_res = detail::validate_feature_range(range_lo, range_hi); !range_res) {
        return std::unexpected(std::move(range_res).error());
    }
    for (std::size_t col = 0; col < data_min.size(); ++col) {
        if (data_min[col] > data_max[col]) {
            return std::unexpected(GprError::invalid_hyperparameter(std::format(
                "fitted min-max input needs data_min <= data_max, column {} has {} > {}", col,
                data_min[col], data_max[col])));
        }
    }
    return FittedMinMaxInput(std::move(data_min), std::move(data_max), range_lo, range_hi);
}

std::expected<void, GprError> FittedMinMaxInput::apply(std::span<double> x, std::size_t n_rows,
                                                       std::size_t n_cols) const {
    const double out_span = range_hi_ - range_lo_;
    return apply_columns_in_place(
        x, n_rows, n_cols, data_min_.size(),
        [this, out_span](std::size_t col, std::span<double> col_span) noexcept {
            const double min_v = data_min_[col];
            const double span = detail::safe_span(data_max_[col], min_v);
            for (double& v : col_span) {
                v = range_lo_ + (out_span * (v - min_v) / span);
            }
        });
}

std::expected<void, GprError> FittedMinMaxInput::apply_into(std::span<const double> x,
                                                            std::size_t n_rows, std::size_t n_cols,
                                                            std::span<double> out) const {
    const double out_span = range_hi_ - range_lo_;
    return apply_columns_into(x, n_rows, n_cols, data_min_.size(), out,
                              [this, out_span](std::size_t col, std::span<const double> src,
                                               std::span<double> dst) noexcept {
                                  const double min_v = data_min_[col];
                                  const double span = detail::safe_span(data_max_[col], min_v);
                                  for (std::size_t i = 0; i < src.size(); ++i) {
                                      dst[i] = range_lo_ + (out_span * (src[i] - min_v) / span);
                                  }
                              });
}

std::expected<void, GprError> FittedMinMaxInput::inverse_apply(std::span<double> x,
                                                               std::size_t n_rows,
                                                               std::size_t n_cols) const {
    const double out_span = range_hi_ - range_lo_;
    return apply_columns_in_place(
        x, n_rows, n_cols, data_min_.size(),
        [this, out_span](std::size_t col, std::span<double> col_span) noexcept {
            const double min_v = data_min_[col];
            const double span = detail::safe_span(data_max_[col], min_v);
            for (double& v : col_span) {
                v = min_v + (span * (v - range_lo_) / out_span);
            }
        });
}

std::expected<void, GprError> FittedMinMaxInput::inverse_apply_into(std::span<const double> x,
                                                                    std::size_t n_rows,
                                                                    std::size_t n_cols,
                                                                    std::span<double> out) const {
    const double out_span = range_hi_ - range_lo_;
    return apply_columns_into(x, n_rows, n_cols, data_min_.size(), out,
                              [this, out_span](std::size_t col, std::span<const double> src,
                                               std::span<double> dst) noexcept {
                                  const double min_v = data_min_[col];
                                  const double span = detail::safe_span(data_max_[col], min_v);
                                  for (std::size_t i = 0; i < src.size(); ++i) {
                                      dst[i] = min_v + (span * (src[i] - range_lo_) / out_span);
                                  }
                              });
}

std::unique_ptr<FittedInputTransform> FittedMinMaxInput::clone() const {
    return std::make_unique<FittedMinMaxInput>(*this);
}

std::expected<MinMaxInput, GprError> MinMaxInput::create(double lo, double hi) {
    if (auto range_res = detail::validate_feature_range(lo, hi); !range_res) {
        return std::unexpected(std::move(range_res).error());
    }
    return MinMaxInput(lo, hi);
}

std::expected<FittedMinMaxInput, GprError>
MinMaxInput::fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check) {
        return std::unexpected(std::move(check).error());
    }
    std::vector<double> data_min;
    std::vector<double> data_max;
    data_min.reserve(n_cols);
    data_max.reserve(n_cols);
    for (std::size_t col = 0; col < n_cols; ++col) {
        const auto col_span = x.subspan(col * n_rows, n_rows);
        const auto [min_it, max_it] = std::ranges::minmax_element(col_span);
        data_min.push_back(*min_it);
        data_max.push_back(*max_it);
    }
    return FittedMinMaxInput::from_parts(std::move(data_min), std::move(data_max), range_lo_,
                                         range_hi_);
}

std::expected<std::unique_ptr<FittedInputTransform>, GprError>
MinMaxInput::fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const {
    auto typed = fit_typed(x, n_rows, n_cols);
    if (!typed) {
        return std::unexpected(std::move(typed).error());
    }
    return std::make_unique<FittedMinMaxInput>(std::move(*typed));
}

std::unique_ptr<UnfittedInputTransform> MinMaxInput::clone() const {
    return std::make_unique<MinMaxInput>(*this);
}

} // namespace cxxgpr::transform
