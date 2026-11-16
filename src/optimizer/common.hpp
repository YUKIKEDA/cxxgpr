// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/optimizer/bounds.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cmath>
#include <cstddef>
#include <expected>
#include <span>
#include <vector>

namespace cxxgpr::optimizer::detail {

[[nodiscard]] inline double dot_vec(std::span<const double> a, std::span<const double> b) noexcept {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

[[nodiscard]] inline double norm2_vec(std::span<const double> a) noexcept {
    return std::sqrt(dot_vec(a, a));
}

template <typename RunSingleFn>
[[nodiscard]] std::expected<OptimizerResult, GprError>
run_with_restarts(const Objective& obj, std::span<const double> initial_params,
                  RestartConfig restarts, RunSingleFn&& run_single) {
    auto bounds = LogBounds::from_objective(obj);
    if (!bounds.has_value()) {
        return std::unexpected(std::move(bounds).error());
    }
    if (auto valid = bounds->validate_log_params(initial_params); !valid.has_value()) {
        return std::unexpected(std::move(valid).error());
    }

    std::expected<OptimizerResult, GprError> best = run_single(*bounds, initial_params);
    if (restarts.count == 0U) {
        return best;
    }

    Xoshiro256PlusPlus rng(restarts.seed);
    std::vector<double> start(bounds->dim());
    for (std::size_t r = 0; r < restarts.count; ++r) {
        bounds->sample_log_uniform(rng, start);
        auto candidate = run_single(*bounds, start);
        if (!candidate.has_value()) {
            if (!best.has_value()) {
                best = std::move(candidate);
            }
            continue;
        }
        if (!best.has_value() || candidate->value < best->value) {
            best = std::move(candidate);
        }
    }
    return best;
}

} // namespace cxxgpr::optimizer::detail
