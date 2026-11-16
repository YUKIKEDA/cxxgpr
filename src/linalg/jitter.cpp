// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/linalg/jitter.hpp"

#include <cmath>
#include <format>

namespace cxxgpr {

std::optional<double> JitterSchedule::next() noexcept {
    if (remaining_ == 0 || !next_.has_value()) {
        return std::nullopt;
    }
    const double current = *next_;
    if (current > max_jitter_) {
        remaining_ = 0;
        next_ = std::nullopt;
        return std::nullopt;
    }
    --remaining_;
    const double grown = current * multiplier_;
    if (std::isfinite(grown)) {
        next_ = grown;
    } else {
        next_ = std::nullopt;
    }
    return current;
}

std::expected<JitterPolicy, GprError> JitterPolicy::fixed(double jitter) {
    if (!std::isfinite(jitter) || jitter < 0.0) {
        return std::unexpected(GprError::invalid_config(
            std::format("jitter must be finite and non-negative, got {}", jitter)));
    }
    return JitterPolicy(FixedJitter(jitter));
}

std::expected<JitterPolicy, GprError> JitterPolicy::adaptive(double initial, double multiplier,
                                                             std::size_t max_retries,
                                                             double max_jitter) {
    if (!std::isfinite(initial) || initial <= 0.0) {
        return std::unexpected(GprError::invalid_config(
            std::format("adaptive initial jitter must be finite and positive, got {}", initial)));
    }
    if (!std::isfinite(multiplier) || multiplier <= 1.0) {
        return std::unexpected(GprError::invalid_config(std::format(
            "adaptive jitter multiplier must be finite and greater than 1, got {}", multiplier)));
    }
    if (max_retries < 1) {
        return std::unexpected(
            GprError::invalid_config("adaptive jitter max_retries must be at least 1"));
    }
    if (!std::isfinite(max_jitter) || max_jitter < initial) {
        return std::unexpected(GprError::invalid_config(
            std::format("adaptive max_jitter must be finite and at least initial ({}), got {}",
                        initial, max_jitter)));
    }
    return JitterPolicy(AdaptiveJitter(initial, multiplier, max_retries, max_jitter));
}

JitterSchedule JitterPolicy::retry_schedule() const noexcept {
    if (kind_ == Kind::Fixed) {
        if (fixed_.jitter() > 0.0) {
            return {fixed_.jitter(), 1.0, 1, fixed_.jitter()};
        }
        return {std::nullopt, 1.0, 0, 0.0};
    }
    return {adaptive_.initial(), adaptive_.multiplier(), adaptive_.max_retries(),
            adaptive_.max_jitter()};
}

} // namespace cxxgpr
