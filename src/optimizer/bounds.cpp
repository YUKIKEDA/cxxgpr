// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/optimizer/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cxxgpr::optimizer {
namespace {

constexpr double k_logit_clamp = 36.0;

[[nodiscard]] double stable_sigmoid(double s) noexcept {
    const double clamped = std::clamp(s, -k_logit_clamp, k_logit_clamp);
    if (clamped >= 0.0) {
        const double e = std::exp(-clamped);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(clamped);
    return e / (1.0 + e);
}

[[nodiscard]] double clamp_strict_interior(double val, double lo, double hi) noexcept {
    const double min_val = std::nextafter(lo, hi);
    const double max_val = std::nextafter(hi, lo);
    return std::clamp(val, min_val, max_val);
}

} // namespace

std::expected<LogBounds, GprError> LogBounds::from_objective(const Objective& obj) {
    const std::size_t d = obj.dim();
    if (d == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    std::vector<Interval> intervals(d, Interval::default_positive());
    if (auto res = obj.fill_intervals(intervals); !res.has_value()) {
        return std::unexpected(std::move(res).error());
    }
    return from_intervals(intervals);
}

std::expected<LogBounds, GprError> LogBounds::from_intervals(std::span<const Interval> intervals) {
    if (intervals.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    std::vector<double> lo_log;
    std::vector<double> hi_log;
    lo_log.reserve(intervals.size());
    hi_log.reserve(intervals.size());
    for (const auto& iv : intervals) {
        if (!std::isfinite(iv.lo()) || !std::isfinite(iv.hi()) || iv.lo() <= 0.0 ||
            iv.lo() >= iv.hi()) {
            return std::unexpected(GprError::invalid_interval_bounds(iv.lo(), iv.hi()));
        }
        const double l = std::log(iv.lo());
        const double u = std::log(iv.hi());
        if (!std::isfinite(l) || !std::isfinite(u) || l >= u) {
            return std::unexpected(GprError::invalid_interval_bounds(iv.lo(), iv.hi()));
        }
        lo_log.push_back(l);
        hi_log.push_back(u);
    }
    return LogBounds(std::move(lo_log), std::move(hi_log));
}

std::expected<void, GprError> LogBounds::validate_log_params(std::span<const double> theta) const {
    const std::size_t d = dim();
    if (theta.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (theta.size() != d) {
        return std::unexpected(
            GprError::length_mismatch("initial parameter length must match objective dimension"));
    }
    for (std::size_t i = 0; i < d; ++i) {
        if (!std::isfinite(theta[i])) {
            return std::unexpected(GprError::non_finite_input());
        }
        if (theta[i] <= lo_log_[i] || theta[i] >= hi_log_[i]) {
            return std::unexpected(
                GprError::invalid_interval_out_of_range(theta[i], lo_log_[i], hi_log_[i]));
        }
    }
    return {};
}

std::expected<void, GprError> LogBounds::to_unconstrained(std::span<const double> theta,
                                                          std::span<double> out_z) const {
    if (auto valid = validate_log_params(theta); !valid.has_value()) {
        return valid;
    }
    const std::size_t d = dim();
    if (out_z.size() != d) {
        return std::unexpected(
            GprError::length_mismatch("unconstrained buffer length must match dimension"));
    }
    for (std::size_t i = 0; i < d; ++i) {
        const double width = hi_log_[i] - lo_log_[i];
        const double p = (theta[i] - lo_log_[i]) / width;
        out_z[i] = 0.25 * width * std::log(p / (1.0 - p));
    }
    return {};
}

void LogBounds::to_constrained(std::span<const double> z,
                               std::span<double> out_theta) const noexcept {
    const std::size_t d = dim();
    for (std::size_t i = 0; i < d; ++i) {
        const double width = hi_log_[i] - lo_log_[i];
        const double sig = stable_sigmoid((4.0 * z[i]) / width);
        out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * sig), lo_log_[i], hi_log_[i]);
    }
}

void LogBounds::to_constrained_with_jacobian(std::span<const double> z, std::span<double> out_theta,
                                             std::span<double> out_dtheta_dz) const noexcept {
    const std::size_t d = dim();
    for (std::size_t i = 0; i < d; ++i) {
        const double width = hi_log_[i] - lo_log_[i];
        const double sig = stable_sigmoid((4.0 * z[i]) / width);
        out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * sig), lo_log_[i], hi_log_[i]);
        out_dtheta_dz[i] = 4.0 * sig * (1.0 - sig);
    }
}

void LogBounds::to_constrained_with_derivatives(std::span<const double> z,
                                                std::span<double> out_theta,
                                                std::span<double> out_d1,
                                                std::span<double> out_d2) const noexcept {
    const std::size_t d = dim();
    for (std::size_t i = 0; i < d; ++i) {
        const double width = hi_log_[i] - lo_log_[i];
        const double sig = stable_sigmoid((4.0 * z[i]) / width);
        out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * sig), lo_log_[i], hi_log_[i]);
        const double sig_om = sig * (1.0 - sig);
        out_d1[i] = 4.0 * sig_om;
        out_d2[i] = (16.0 / width) * sig_om * (1.0 - (2.0 * sig));
    }
}

void LogBounds::chain_gradient(std::span<const double> g_theta, std::span<const double> d1,
                               std::span<double> out_g_z) noexcept {
    const std::size_t d = g_theta.size();
    for (std::size_t i = 0; i < d; ++i) {
        out_g_z[i] = g_theta[i] * d1[i];
    }
}

void LogBounds::chain_hessian(std::span<const double> h_theta, std::span<const double> g_theta,
                              std::span<const double> d1, std::span<const double> d2,
                              std::span<double> out_h_z) noexcept {
    const std::size_t d = g_theta.size();
    for (std::size_t col = 0; col < d; ++col) {
        for (std::size_t row = 0; row < d; ++row) {
            double val = h_theta[(col * d) + row] * d1[row] * d1[col];
            if (row == col) {
                val += g_theta[row] * d2[row];
            }
            out_h_z[(col * d) + row] = val;
        }
    }
}

void LogBounds::sample_log_uniform(Xoshiro256PlusPlus& rng,
                                   std::span<double> out_theta) const noexcept {
    const std::size_t d = dim();
    for (std::size_t i = 0; i < d; ++i) {
        const double u = rng.open_unit();
        const double width = hi_log_[i] - lo_log_[i];
        out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * u), lo_log_[i], hi_log_[i]);
    }
}

std::expected<double, GprError>
EvalCache::eval_unconstrained(Differentiable& obj, const LogBounds& bounds,
                              std::span<const double> z, std::span<double> theta_buf,
                              std::span<double> d1_buf, std::span<double> grad_theta_buf,
                              std::span<double> out_grad_z) {
    const std::size_t d = bounds.dim();
    if (valid_ && last_z_.size() == d && std::equal(z.begin(), z.end(), last_z_.begin())) {
        std::ranges::copy(last_grad_z_, out_grad_z.begin());
        return last_val_;
    }

    for (const double val : z) {
        if (!std::isfinite(val)) {
            return std::unexpected(GprError::non_finite_input());
        }
    }
    bounds.to_constrained_with_jacobian(z, theta_buf, d1_buf);
    auto val = obj.value_and_gradient_into(theta_buf, grad_theta_buf);
    if (!val.has_value()) {
        return val;
    }
    if (!std::isfinite(*val)) {
        return std::unexpected(GprError::non_finite_input());
    }
    for (const double g : grad_theta_buf) {
        if (!std::isfinite(g)) {
            return std::unexpected(GprError::non_finite_input());
        }
    }
    LogBounds::chain_gradient(grad_theta_buf, d1_buf, out_grad_z);

    last_z_.assign(z.begin(), z.end());
    last_grad_z_.assign(out_grad_z.begin(), out_grad_z.end());
    last_val_ = *val;
    valid_ = true;
    return *val;
}

} // namespace cxxgpr::optimizer
