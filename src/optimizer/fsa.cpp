// SPDX-License-Identifier: Apache-2.0
/// @file fsa.cpp
/// @brief Implementation of Fast Simulated Annealing with Cauchy proposals and incremental updates.
/// @details Implements Clamp and Reflect boundary projections, Ingber exponential cooling, Metropolis
/// acceptance, bit-exact changed-index tracking for Objective::value_at_changes(), and multi-start restarts.

#include "cxxgpr/optimizer/fsa.hpp"

#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/optimizer/bounds.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#include "common.hpp"

namespace cxxgpr::optimizer
{
    namespace
    {

        /// @brief Projects a proposed coordinate back into the strict open interval (lo, hi).
        /// @details Under BoundaryPolicy::Clamp, clamps to [nextafter(lo, hi), nextafter(hi, lo)]; under
        /// BoundaryPolicy::Reflect, folds excess distance via a period-2*width triangle wave across lo and hi.
        /// @param[in] val Proposed scalar value in log-theta space.
        /// @param[in] lo Exclusive lower bound in log-theta space.
        /// @param[in] hi Exclusive upper bound in log-theta space.
        /// @param[in] policy Boundary projection strategy (Clamp or Reflect).
        /// @return Projected coordinate strictly inside (lo, hi).
        [[nodiscard]] double apply_boundary(double val, double lo, double hi, BoundaryPolicy policy) noexcept
        {
            const double lo_bound = std::nextafter(lo, hi);
            const double hi_bound = std::nextafter(hi, lo);
            if (policy == BoundaryPolicy::Clamp)
            {
                return std::clamp(val, lo_bound, hi_bound);
            }
            // Reflect values outside (lo, hi) across boundaries using a periodic triangle wave of period 2 * width.
            const double width = hi - lo;
            if (val > lo && val < hi)
            {
                return val;
            }
            const double period = 2.0 * width;
            double offset = std::fmod(val - lo, period);
            if (offset < 0.0)
            {
                offset += period;
            }
            val = (offset <= width) ? (lo + offset) : (hi - (offset - width));
            return std::clamp(val, lo_bound, hi_bound);
        }

        /// @brief Runs one single-start Fast Simulated Annealing trajectory from initial_theta.
        /// @details Evaluates initial_theta via obj.value(), cycles through coordinates iter % d proposing
        /// Cauchy steps temp * tan(pi * (u - 0.5)) under Ingber cooling, detects bit-exact changed indices
        /// relative to last_eval for Objective::value_at_changes(), and accepts via Metropolis criterion.
        /// @param[in,out] obj Objective evaluated in log-theta space (supports value_at_changes()).
        /// @param[in] bounds Log-space open bounds for boundary projection.
        /// @param[in] initial_theta Starting log-space parameter vector strictly inside bounds.
        /// @param[in] max_iterations Maximum annealing steps to perform.
        /// @param[in] initial_temperature Positive starting annealing temperature T_0.
        /// @param[in] cooling_rate Non-negative Ingber cooling factor c.
        /// @param[in] seed Deterministic 64-bit PRNG seed for this trajectory.
        /// @param[in] boundary_policy Out-of-bounds projection policy (Clamp or Reflect).
        /// @return Best parameters encountered along the trajectory, or GprError if initial eval fails.
        /// @retval GprErrorCode::NonFiniteInput When initial evaluation yields NaN or +/-Inf.
        [[nodiscard]] std::expected<OptimizerResult, GprError> run_fsa_single(
            Objective& obj,
            const LogBounds& bounds,
            std::span<const double> initial_theta,
            std::size_t max_iterations,
            double initial_temperature,
            double cooling_rate,
            std::uint64_t seed,
            BoundaryPolicy boundary_policy
        )
        {
            const std::size_t d = bounds.dim();
            std::vector<double> current(initial_theta.begin(), initial_theta.end());
            auto init_val = obj.value(current);
            if (!init_val.has_value())
            {
                return std::unexpected(std::move(init_val).error());
            }
            if (!std::isfinite(*init_val))
            {
                return std::unexpected(GprError::non_finite_input());
            }

            double current_val = *init_val;
            std::vector<double> best_theta = current;
            double best_val = current_val;

            // Track the last parameter vector actually passed to obj so incremental diffs stay exact even on rejected
            // steps.
            std::vector<double> last_eval = current;
            std::vector<double> candidate(d);
            std::vector<std::size_t> changed_indices;
            changed_indices.reserve(2U);

            const auto lo_log = bounds.lo_log();
            const auto hi_log = bounds.hi_log();
            Xoshiro256PlusPlus rng(seed);
            const double inv_d = 1.0 / static_cast<double>(d);

            const double inv_max_iter = 1.0 / static_cast<double>(max_iterations);
            for (std::size_t iter = 0; iter < max_iterations; ++iter)
            {
                // Ingber exponential cooling schedule T(k) = T_0 * exp(-c * (k / K)^(1 / d)) with Cauchy proposal.
                const double progress = static_cast<double>(iter + 1U) * inv_max_iter;
                const double temp = initial_temperature * std::exp(-cooling_rate * std::pow(progress, inv_d));
                const std::size_t coord = iter % d;
                const double u = rng.open_unit();
                const double step = temp * std::tan(std::numbers::pi * (u - 0.5));

                std::ranges::copy(current, candidate.begin());
                candidate[coord] = apply_boundary(current[coord] + step, lo_log[coord], hi_log[coord], boundary_policy);

                // Collect bit-exact differences against last_eval to satisfy LeafCache's change-index contract.
                changed_indices.clear();
                for (std::size_t i = 0; i < d; ++i)
                {
                    if (std::bit_cast<std::uint64_t>(candidate[i]) != std::bit_cast<std::uint64_t>(last_eval[i]))
                    {
                        changed_indices.push_back(i);
                    }
                }

                // Skip objective call if boundary clamping snapped candidate back identically to last_eval.
                auto cand_res = changed_indices.empty() ? std::expected<double, GprError>(current_val)
                                                        : obj.value_at_changes(candidate, changed_indices);
                if (!changed_indices.empty())
                {
                    last_eval = candidate;
                }
                if (!cand_res.has_value() || !std::isfinite(*cand_res))
                {
                    continue;
                }

                // Metropolis acceptance criterion: accept all downhill moves and uphill moves with exp(-delta / temp).
                const double cand_val = *cand_res;
                const double delta = cand_val - current_val;
                const bool accept = (delta <= 0.0) || (temp > 0.0 && rng.unit() < std::exp(-delta / temp));
                if (accept)
                {
                    current = candidate;
                    current_val = cand_val;
                    if (current_val < best_val)
                    {
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

    }  // namespace

    /// @brief Minimizes obj starting from log-space initial_params.
    /// @details Validates initial_temperature_ > 0 and cooling_rate_ >= 0, then delegates to
    /// detail::run_with_restarts() incrementing run_seed = seed_ + run_idx per trajectory.
    /// @param[in,out] obj Objective evaluated in log-theta space (supports value_at_changes()).
    /// @param[in] initial_params Starting log-space parameter vector strictly inside objective bounds.
    /// @return Best OptimizerResult across the initial run and restarts, or GprError on invalid config/input.
    /// @retval GprErrorCode::InvalidConfig When initial_temperature <= 0 or cooling_rate < 0 or non-finite.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
    /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
    /// @retval GprErrorCode::NonFiniteInput When initial_params or initial objective value is non-finite.
    /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
    std::expected<OptimizerResult, GprError> FastSimulatedAnnealing::optimize(
        Objective& obj, std::span<const double> initial_params
    ) const
    {
        if (!std::isfinite(initial_temperature_) || initial_temperature_ <= 0.0)
        {
            return std::unexpected(GprError::invalid_config("initial_temperature must be finite and positive"));
        }
        if (!std::isfinite(cooling_rate_) || cooling_rate_ < 0.0)
        {
            return std::unexpected(GprError::invalid_config("cooling_rate must be finite and non-negative"));
        }
        // Offset per-start PRNG seed deterministically so multi-start trajectories explore distinct Cauchy sequences.
        std::size_t run_idx = 0U;
        return detail::run_with_restarts(
            obj, initial_params, restarts_, [&](const LogBounds& bounds, std::span<const double> start_theta)
        {
            const std::uint64_t run_seed = seed_ + run_idx;
            ++run_idx;
            return run_fsa_single(
                obj, bounds, start_theta, max_iterations_, initial_temperature_, cooling_rate_, run_seed,
                boundary_policy_
            );
        }
        );
    }

}  // namespace cxxgpr::optimizer
