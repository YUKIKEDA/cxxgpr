// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/leaves.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <utility>

namespace cxxgpr {

namespace {

[[nodiscard]] std::expected<Interval, GprError> require_positive_interval(Interval interval) {
    return Interval::positive(interval.lo(), interval.hi());
}

[[nodiscard]] std::expected<BoundedParam, GprError> make_positive_param(double value,
                                                                        Interval interval) {
    auto pos_iv = require_positive_interval(interval);
    if (!pos_iv.has_value()) {
        return std::unexpected(std::move(pos_iv).error());
    }
    return BoundedParam::create(value, *pos_iv);
}

[[nodiscard]] std::expected<std::vector<BoundedParam>, GprError>
make_positive_params(std::span<const double> values, std::span<const Interval> intervals) {
    if (auto ne = require_nonempty(values.size()); !ne.has_value()) {
        return std::unexpected(std::move(ne).error());
    }
    if (auto cnt = require_count(intervals.size(), values.size(), "intervals"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::vector<BoundedParam> out;
    out.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto param = make_positive_param(values[i], intervals[i]);
        if (!param.has_value()) {
            return std::unexpected(std::move(param).error());
        }
        out.push_back(*param);
    }
    return out;
}

[[nodiscard]] std::expected<std::vector<BoundedParam>, GprError>
make_default_positive_params(std::span<const double> values) {
    if (auto ne = require_nonempty(values.size()); !ne.has_value()) {
        return std::unexpected(std::move(ne).error());
    }
    std::vector<BoundedParam> out;
    out.reserve(values.size());
    for (double val : values) {
        auto param = BoundedParam::default_positive(val);
        if (!param.has_value()) {
            return std::unexpected(std::move(param).error());
        }
        out.push_back(*param);
    }
    return out;
}

[[nodiscard]] std::expected<BoundedParam, GprError>
update_param_from_log(const BoundedParam& current, double log_value) {
    if (!std::isfinite(log_value)) {
        return std::unexpected(GprError::non_finite_input());
    }
    const double val = std::exp(log_value);
    if (!std::isfinite(val)) {
        return std::unexpected(GprError::invalid_interval_out_of_range(val, current.interval().lo(),
                                                                       current.interval().hi()));
    }
    return current.with_value(val);
}

[[nodiscard]] std::expected<void, GprError> get_single_param(const BoundedParam& param,
                                                             std::span<double> out) {
    if (auto cnt = require_count(out.size(), 1U, "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    out[0] = param.ln();
    return {};
}

[[nodiscard]] std::expected<void, GprError> set_single_param(BoundedParam& param,
                                                             std::span<const double> params) {
    if (auto cnt = require_count(params.size(), 1U, "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    auto next = update_param_from_log(param, params[0]);
    if (!next.has_value()) {
        return std::unexpected(std::move(next).error());
    }
    param = *next;
    return {};
}

[[nodiscard]] std::expected<void, GprError> get_single_bound(const BoundedParam& param,
                                                             std::span<Interval> out) {
    if (auto cnt = require_count(out.size(), 1U, "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    out[0] = param.interval();
    return {};
}

[[nodiscard]] std::vector<double> extract_values(std::span<const BoundedParam> params) {
    std::vector<double> values;
    values.reserve(params.size());
    for (const auto& param : params) {
        values.push_back(param.value());
    }
    return values;
}

[[nodiscard]] std::vector<Interval> extract_intervals(std::span<const BoundedParam> params) {
    std::vector<Interval> intervals;
    intervals.reserve(params.size());
    for (const auto& param : params) {
        intervals.push_back(param.interval());
    }
    return intervals;
}

} // namespace

// --- RbfKernel ---

std::expected<RbfKernel, GprError> RbfKernel::create(double lengthscale) {
    return create(lengthscale, Interval::default_positive());
}

std::expected<RbfKernel, GprError> RbfKernel::create(double lengthscale, Interval interval) {
    auto param = make_positive_param(lengthscale, interval);
    if (!param.has_value()) {
        return std::unexpected(std::move(param).error());
    }
    return RbfKernel(*param);
}

std::expected<void, GprError> RbfKernel::get_params(std::span<double> out) const {
    return get_single_param(lengthscale_, out);
}

std::expected<void, GprError> RbfKernel::set_params(std::span<const double> params) {
    return set_single_param(lengthscale_, params);
}

std::expected<void, GprError> RbfKernel::bounds_into(std::span<Interval> out) const {
    return get_single_bound(lengthscale_, out);
}

// --- RbfArdKernel ---

std::expected<RbfArdKernel, GprError> RbfArdKernel::create(std::span<const double> lengthscales) {
    auto params = make_default_positive_params(lengthscales);
    if (!params.has_value()) {
        return std::unexpected(std::move(params).error());
    }
    return RbfArdKernel(std::move(*params));
}

std::expected<RbfArdKernel, GprError> RbfArdKernel::create(std::span<const double> lengthscales,
                                                           std::span<const Interval> intervals) {
    auto params = make_positive_params(lengthscales, intervals);
    if (!params.has_value()) {
        return std::unexpected(std::move(params).error());
    }
    return RbfArdKernel(std::move(*params));
}

std::vector<double> RbfArdKernel::lengthscales() const {
    return extract_values(lengthscales_);
}

std::vector<Interval> RbfArdKernel::intervals() const {
    return extract_intervals(lengthscales_);
}

std::expected<void, GprError> RbfArdKernel::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        out[i] = lengthscales_[i].ln();
    }
    return {};
}

std::expected<void, GprError> RbfArdKernel::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), lengthscales_.size(), "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::vector<BoundedParam> updated;
    updated.reserve(lengthscales_.size());
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        auto next = update_param_from_log(lengthscales_[i], params[i]);
        if (!next.has_value()) {
            return std::unexpected(std::move(next).error());
        }
        updated.push_back(*next);
    }
    lengthscales_ = std::move(updated);
    return {};
}

std::expected<void, GprError> RbfArdKernel::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        out[i] = lengthscales_[i].interval();
    }
    return {};
}

// --- MaternKernel ---

std::expected<MaternKernel, GprError> MaternKernel::create(double lengthscale, MaternNu nu) {
    return create(lengthscale, nu, Interval::default_positive());
}

std::expected<MaternKernel, GprError> MaternKernel::create(double lengthscale, MaternNu nu,
                                                           Interval interval) {
    auto param = make_positive_param(lengthscale, interval);
    if (!param.has_value()) {
        return std::unexpected(std::move(param).error());
    }
    return MaternKernel(*param, nu);
}

std::expected<void, GprError> MaternKernel::get_params(std::span<double> out) const {
    return get_single_param(lengthscale_, out);
}

std::expected<void, GprError> MaternKernel::set_params(std::span<const double> params) {
    return set_single_param(lengthscale_, params);
}

std::expected<void, GprError> MaternKernel::bounds_into(std::span<Interval> out) const {
    return get_single_bound(lengthscale_, out);
}

// --- MaternArdKernel ---

std::expected<MaternArdKernel, GprError>
MaternArdKernel::create(std::span<const double> lengthscales, MaternNu nu) {
    auto params = make_default_positive_params(lengthscales);
    if (!params.has_value()) {
        return std::unexpected(std::move(params).error());
    }
    return MaternArdKernel(std::move(*params), nu);
}

std::expected<MaternArdKernel, GprError>
MaternArdKernel::create(std::span<const double> lengthscales, MaternNu nu,
                        std::span<const Interval> intervals) {
    auto params = make_positive_params(lengthscales, intervals);
    if (!params.has_value()) {
        return std::unexpected(std::move(params).error());
    }
    return MaternArdKernel(std::move(*params), nu);
}

std::vector<double> MaternArdKernel::lengthscales() const {
    return extract_values(lengthscales_);
}

std::vector<Interval> MaternArdKernel::intervals() const {
    return extract_intervals(lengthscales_);
}

std::expected<void, GprError> MaternArdKernel::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        out[i] = lengthscales_[i].ln();
    }
    return {};
}

std::expected<void, GprError> MaternArdKernel::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), lengthscales_.size(), "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::vector<BoundedParam> updated;
    updated.reserve(lengthscales_.size());
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        auto next = update_param_from_log(lengthscales_[i], params[i]);
        if (!next.has_value()) {
            return std::unexpected(std::move(next).error());
        }
        updated.push_back(*next);
    }
    lengthscales_ = std::move(updated);
    return {};
}

std::expected<void, GprError> MaternArdKernel::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        out[i] = lengthscales_[i].interval();
    }
    return {};
}

// --- PeriodicKernel ---

std::expected<PeriodicKernel, GprError> PeriodicKernel::create(double lengthscale, double period) {
    return create(lengthscale, period, Interval::default_positive(), Interval::default_positive());
}

std::expected<PeriodicKernel, GprError> PeriodicKernel::create(double lengthscale, double period,
                                                               Interval lengthscale_interval,
                                                               Interval period_interval) {
    auto ls = make_positive_param(lengthscale, lengthscale_interval);
    if (!ls.has_value()) {
        return std::unexpected(std::move(ls).error());
    }
    auto p = make_positive_param(period, period_interval);
    if (!p.has_value()) {
        return std::unexpected(std::move(p).error());
    }
    return PeriodicKernel(*ls, *p);
}

std::expected<void, GprError> PeriodicKernel::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    out[0] = lengthscale_.ln();
    out[1] = period_.ln();
    return {};
}

std::expected<void, GprError> PeriodicKernel::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), 2U, "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    auto next_ls = update_param_from_log(lengthscale_, params[0]);
    if (!next_ls.has_value()) {
        return std::unexpected(std::move(next_ls).error());
    }
    auto next_p = update_param_from_log(period_, params[1]);
    if (!next_p.has_value()) {
        return std::unexpected(std::move(next_p).error());
    }
    lengthscale_ = *next_ls;
    period_ = *next_p;
    return {};
}

std::expected<void, GprError> PeriodicKernel::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    out[0] = lengthscale_.interval();
    out[1] = period_.interval();
    return {};
}

// --- RationalQuadraticKernel ---

std::expected<RationalQuadraticKernel, GprError> RationalQuadraticKernel::create(double lengthscale,
                                                                                 double alpha) {
    return create(lengthscale, alpha, Interval::default_positive(), Interval::default_positive());
}

std::expected<RationalQuadraticKernel, GprError>
RationalQuadraticKernel::create(double lengthscale, double alpha, Interval lengthscale_interval,
                                Interval alpha_interval) {
    auto ls = make_positive_param(lengthscale, lengthscale_interval);
    if (!ls.has_value()) {
        return std::unexpected(std::move(ls).error());
    }
    auto a = make_positive_param(alpha, alpha_interval);
    if (!a.has_value()) {
        return std::unexpected(std::move(a).error());
    }
    return RationalQuadraticKernel(*ls, *a);
}

std::expected<void, GprError> RationalQuadraticKernel::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    out[0] = lengthscale_.ln();
    out[1] = alpha_.ln();
    return {};
}

std::expected<void, GprError> RationalQuadraticKernel::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), 2U, "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    auto next_ls = update_param_from_log(lengthscale_, params[0]);
    if (!next_ls.has_value()) {
        return std::unexpected(std::move(next_ls).error());
    }
    auto next_a = update_param_from_log(alpha_, params[1]);
    if (!next_a.has_value()) {
        return std::unexpected(std::move(next_a).error());
    }
    lengthscale_ = *next_ls;
    alpha_ = *next_a;
    return {};
}

std::expected<void, GprError> RationalQuadraticKernel::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    out[0] = lengthscale_.interval();
    out[1] = alpha_.interval();
    return {};
}

// --- RationalQuadraticArdKernel ---

std::expected<RationalQuadraticArdKernel, GprError>
RationalQuadraticArdKernel::create(std::span<const double> lengthscales, double alpha) {
    auto ls = make_default_positive_params(lengthscales);
    if (!ls.has_value()) {
        return std::unexpected(std::move(ls).error());
    }
    auto a = BoundedParam::default_positive(alpha);
    if (!a.has_value()) {
        return std::unexpected(std::move(a).error());
    }
    return RationalQuadraticArdKernel(std::move(*ls), *a);
}

std::expected<RationalQuadraticArdKernel, GprError>
RationalQuadraticArdKernel::create(std::span<const double> lengthscales, double alpha,
                                   std::span<const Interval> lengthscale_intervals,
                                   Interval alpha_interval) {
    auto ls = make_positive_params(lengthscales, lengthscale_intervals);
    if (!ls.has_value()) {
        return std::unexpected(std::move(ls).error());
    }
    auto a = make_positive_param(alpha, alpha_interval);
    if (!a.has_value()) {
        return std::unexpected(std::move(a).error());
    }
    return RationalQuadraticArdKernel(std::move(*ls), *a);
}

std::vector<double> RationalQuadraticArdKernel::lengthscales() const {
    return extract_values(lengthscales_);
}

std::vector<Interval> RationalQuadraticArdKernel::lengthscale_intervals() const {
    return extract_intervals(lengthscales_);
}

std::expected<void, GprError> RationalQuadraticArdKernel::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        out[i] = lengthscales_[i].ln();
    }
    out[lengthscales_.size()] = alpha_.ln();
    return {};
}

std::expected<void, GprError>
RationalQuadraticArdKernel::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::vector<BoundedParam> updated;
    updated.reserve(lengthscales_.size());
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        auto next = update_param_from_log(lengthscales_[i], params[i]);
        if (!next.has_value()) {
            return std::unexpected(std::move(next).error());
        }
        updated.push_back(*next);
    }
    auto next_a = update_param_from_log(alpha_, params[lengthscales_.size()]);
    if (!next_a.has_value()) {
        return std::unexpected(std::move(next_a).error());
    }
    lengthscales_ = std::move(updated);
    alpha_ = *next_a;
    return {};
}

std::expected<void, GprError>
RationalQuadraticArdKernel::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    for (std::size_t i = 0; i < lengthscales_.size(); ++i) {
        out[i] = lengthscales_[i].interval();
    }
    out[lengthscales_.size()] = alpha_.interval();
    return {};
}

// --- ConstantKernel ---

std::expected<ConstantKernel, GprError> ConstantKernel::create(double constant) {
    return create(constant, Interval::default_positive());
}

std::expected<ConstantKernel, GprError> ConstantKernel::create(double constant, Interval interval) {
    auto param = make_positive_param(constant, interval);
    if (!param.has_value()) {
        return std::unexpected(std::move(param).error());
    }
    return ConstantKernel(*param);
}

std::expected<void, GprError> ConstantKernel::get_params(std::span<double> out) const {
    return get_single_param(constant_, out);
}

std::expected<void, GprError> ConstantKernel::set_params(std::span<const double> params) {
    return set_single_param(constant_, params);
}

std::expected<void, GprError> ConstantKernel::bounds_into(std::span<Interval> out) const {
    return get_single_bound(constant_, out);
}

// --- LinearKernel ---

std::expected<LinearKernel, GprError> LinearKernel::create(double variance) {
    return create(variance, Interval::default_positive());
}

std::expected<LinearKernel, GprError> LinearKernel::create(double variance, Interval interval) {
    auto param = make_positive_param(variance, interval);
    if (!param.has_value()) {
        return std::unexpected(std::move(param).error());
    }
    return LinearKernel(*param);
}

std::expected<void, GprError> LinearKernel::get_params(std::span<double> out) const {
    return get_single_param(variance_, out);
}

std::expected<void, GprError> LinearKernel::set_params(std::span<const double> params) {
    return set_single_param(variance_, params);
}

std::expected<void, GprError> LinearKernel::bounds_into(std::span<Interval> out) const {
    return get_single_bound(variance_, out);
}

// --- WhiteKernel ---

std::expected<WhiteKernel, GprError> WhiteKernel::create(double variance) {
    return create(variance, Interval::default_positive());
}

std::expected<WhiteKernel, GprError> WhiteKernel::create(double variance, Interval interval) {
    auto param = make_positive_param(variance, interval);
    if (!param.has_value()) {
        return std::unexpected(std::move(param).error());
    }
    return WhiteKernel(*param);
}

std::expected<void, GprError> WhiteKernel::get_params(std::span<double> out) const {
    return get_single_param(variance_, out);
}

std::expected<void, GprError> WhiteKernel::set_params(std::span<const double> params) {
    return set_single_param(variance_, params);
}

std::expected<void, GprError> WhiteKernel::bounds_into(std::span<Interval> out) const {
    return get_single_bound(variance_, out);
}

} // namespace cxxgpr
