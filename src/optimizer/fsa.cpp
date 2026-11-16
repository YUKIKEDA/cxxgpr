// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/optimizer/fsa.hpp"

#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/optimizer/bounds.hpp"
#include "common.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace cxxgpr::optimizer {
namespace {

[[nodiscard]] double apply_boundary(double val, double lo, double hi,
                                    BoundaryPolicy policy) noexcept {
    const double min_val = std::nextafter(lo, hi);
    const double max_val = std::nextafter(hi, lo);
    if (policy == BoundaryPolicy::Clamp) {
        return std::clamp(val, min_val, max_val);
    }
    const double width = hi - lo;
    if (val > lo && val < hi) {
        return val;
    }
    const double period = 2.0 * width;
    double offset = std::fmod(val - lo, period);
    if (offset < 0.0) {
        offset += period;
    }
    const double reflected = (offset <= width) ? (lo + offset) : (hi - (offset - width));
    return std::clamp(reflected, min_val, max_val);
}

[[nodiscard]] std::expected<OptimizerResult, GprError>
run_fsa_single(Objective& obj, const LogBounds& bounds, std::span<const double> initial_theta,
               std::size_t max_iterations, double initial_temperature, double cooling_rate,
               std::uint64_t seed, BoundaryPolicy boundary_policy) {
    const std::size_t d = bounds.dim();
    std::vector<double> current(initial_theta.begin(), initial_theta.end());
    auto init_val = obj.value(current);
    if (!init_val.has_value()) {
        return std::unexpected(std::move(init_val).error());
    }
    if (!std::isfinite(*init_val)) {
        return std::unexpected(GprError::non_finite_input());
    }

    double current_val = *init_val;
    std::vector<double> best_theta = current;
    double best_val = current_val;

    std::vector<double> last_eval = current;
    std::vector<double> candidate(d);
    std::vector<std::size_t> changed_indices;
    changed_indices.reserve(2U);

    const auto lo_log = bounds.lo_log();
    const auto hi_log = bounds.hi_log();
    Xoshiro256PlusPlus rng(seed);
    const double inv_d = 1.0 / static_cast<double>(d);

    const double inv_max_iter = 1.0 / static_cast<double>(max_iterations);
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        const double progress = static_cast<double>(iter + 1U) * inv_max_iter;
        const double temp =
            initial_temperature * std::exp(-cooling_rate * std::pow(progress, inv_d));
        const std::size_t coord = iter % d;
        const double u = rng.open_unit();
        const double step = temp * std::tan(std::numbers::pi * (u - 0.5));

        std::ranges::copy(current, candidate.begin());
        candidate[coord] =
            apply_boundary(current[coord] + step, lo_log[coord], hi_log[coord], boundary_policy);

        changed_indices.clear();
        for (std::size_t i = 0; i < d; ++i) {
            if (std::bit_cast<std::uint64_t>(candidate[i]) !=
                std::bit_cast<std::uint64_t>(last_eval[i])) {
                changed_indices.push_back(i);
            }
        }

        auto cand_res = changed_indices.empty() ? std::expected<double, GprError>(current_val)
                                                : obj.value_at_changes(candidate, changed_indices);
        if (!changed_indices.empty()) {
            last_eval = candidate;
        }
        if (!cand_res.has_value() || !std::isfinite(*cand_res)) {
            continue;
        }

        const double cand_val = *cand_res;
        const double delta = cand_val - current_val;
        const bool accept = (delta <= 0.0) || (temp > 0.0 && rng.unit() < std::exp(-delta / temp));
        if (accept) {
            current = candidate;
            current_val = cand_val;
            if (current_val < best_val) {
                best_val = current_val;
                best_theta = current;
            }
        }
    }

    return OptimizerResult{
        .params = std::move(best_theta),
        .value = best_val,
        .iterations = max_iterations,
        .converged = true,
    };
}

} // namespace

std::expected<OptimizerResult, GprError>
FastSimulatedAnnealing::optimize(Objective& obj, std::span<const double> initial_params) const {
    if (!std::isfinite(initial_temperature_) || initial_temperature_ <= 0.0) {
        return std::unexpected(
            GprError::invalid_config("initial_temperature must be finite and positive"));
    }
    if (!std::isfinite(cooling_rate_) || cooling_rate_ < 0.0) {
        return std::unexpected(
            GprError::invalid_config("cooling_rate must be finite and non-negative"));
    }
    std::size_t run_idx = 0U;
    return detail::run_with_restarts(
        obj, initial_params, restarts_,
        [&](const LogBounds& bounds, std::span<const double> start_theta) {
            const std::uint64_t run_seed = seed_ + run_idx;
            ++run_idx;
            return run_fsa_single(obj, bounds, start_theta, max_iterations_, initial_temperature_,
                                  cooling_rate_, run_seed, boundary_policy_);
        });
}

} // namespace cxxgpr::optimizer
