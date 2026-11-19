// SPDX-License-Identifier: Apache-2.0
/// @file fsa.hpp
/// @brief Fast Simulated Annealing optimizer using Cauchy proposals and incremental leaf caching.
/// @details Declares BoundaryPolicy and FastSimulatedAnnealing for derivative-free global
/// optimization directly in constrained log-theta space using single-coordinate proposals.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

/// @brief Hyperparameter optimization interfaces, coordinate transforms, and solver implementations.
namespace cxxgpr::optimizer
{

    /// @brief Selects how FastSimulatedAnnealing handles proposals outside open log-theta bounds.
    /// @details Applied coordinate-wise after adding a Cauchy step so candidates remain strictly inside
    /// (lo_log, hi_log).
    enum class BoundaryPolicy : std::uint8_t
    {
        /// @brief Clamps out-of-bounds proposals strictly inside the open interval.
        Clamp,
        /// @brief Reflects out-of-bounds proposals back across interval boundaries.
        Reflect,
    };

    /// @brief Fast Simulated Annealing optimizer operating directly in log-theta space.
    /// @details Uses coordinate-cycling Cauchy proposals, Metropolis acceptance, Ingber exponential
    /// cooling T(k) = T_0 * exp(-c * (k / K)^(1/d)), and incremental single-coordinate updates via
    /// Objective::value_at_changes(), with optional deterministic multi-start restarts.
    class FastSimulatedAnnealing
    {
    public:
        /// @brief FSA uses single-coordinate incremental updates via value_at_changes().
        static constexpr bool uses_change_indices = true;

        /// @brief Constructs FastSimulatedAnnealing with default settings.
        /// @details Defaults to max_iterations = 100, initial_temperature = 1.0, cooling_rate = 3.0,
        /// seed = 0, boundary_policy = BoundaryPolicy::Clamp, and no extra restarts.
        FastSimulatedAnnealing() noexcept = default;

        /// @brief Returns a copy with updated maximum annealing iterations.
        /// @details Controls the cooling schedule horizon and number of single-coordinate proposals per run.
        /// @param[in] max_iterations Maximum number of annealing iterations per start.
        /// @return Updated FastSimulatedAnnealing configuration copy.
        [[nodiscard]] FastSimulatedAnnealing with_max_iterations(std::size_t max_iterations) const noexcept
        {
            FastSimulatedAnnealing copy = *this;
            copy.max_iterations_ = max_iterations;
            return copy;
        }

        /// @brief Returns a copy with updated initial annealing temperature (> 0).
        /// @details Validated at optimize() time to be finite and strictly positive.
        /// @param[in] initial_temperature Initial scale T_0 > 0 for Cauchy steps and Metropolis acceptance.
        /// @return Updated FastSimulatedAnnealing configuration copy.
        [[nodiscard]] FastSimulatedAnnealing with_initial_temperature(double initial_temperature) const noexcept
        {
            FastSimulatedAnnealing copy = *this;
            copy.initial_temperature_ = initial_temperature;
            return copy;
        }

        /// @brief Returns a copy with updated Ingber exponential cooling rate (>= 0).
        /// @details Validated at optimize() time to be finite and non-negative.
        /// @param[in] cooling_rate Exponential cooling factor c >= 0 in T_0 * exp(-c * progress^(1/d)).
        /// @return Updated FastSimulatedAnnealing configuration copy.
        [[nodiscard]] FastSimulatedAnnealing with_cooling_rate(double cooling_rate) const noexcept
        {
            FastSimulatedAnnealing copy = *this;
            copy.cooling_rate_ = cooling_rate;
            return copy;
        }

        /// @brief Returns a copy with updated Xoshiro256++ seed for proposals and acceptance.
        /// @details Each multi-start restart adds its zero-based run index to seed to stay independent and
        /// deterministic.
        /// @param[in] seed Deterministic 64-bit PRNG seed.
        /// @return Updated FastSimulatedAnnealing configuration copy.
        [[nodiscard]] FastSimulatedAnnealing with_seed(std::uint64_t seed) const noexcept
        {
            FastSimulatedAnnealing copy = *this;
            copy.seed_ = seed;
            return copy;
        }

        /// @brief Returns a copy with updated boundary handling policy (Clamp or Reflect).
        /// @details Governs how out-of-bounds single-coordinate proposals are mapped back inside (lo_log, hi_log).
        /// @param[in] boundary_policy Boundary projection strategy.
        /// @return Updated FastSimulatedAnnealing configuration copy.
        [[nodiscard]] FastSimulatedAnnealing with_boundary_policy(BoundaryPolicy boundary_policy) const noexcept
        {
            FastSimulatedAnnealing copy = *this;
            copy.boundary_policy_ = boundary_policy;
            return copy;
        }

        /// @brief Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
        /// @details Runs count additional restarts from log-uniform samples after evaluating initial_params.
        /// @param[in] count Number of extra random restarts.
        /// @param[in] seed Seed used to generate restart starting points.
        /// @return Updated FastSimulatedAnnealing configuration copy.
        [[nodiscard]] FastSimulatedAnnealing with_restarts(std::size_t count, std::uint64_t seed = 0U) const noexcept
        {
            FastSimulatedAnnealing copy = *this;
            copy.restarts_ = RestartConfig{count, seed};
            return copy;
        }

        /// @brief Returns the configured maximum iterations.
        /// @details Defaults to 100.
        /// @return Maximum annealing iteration count per start.
        [[nodiscard]] std::size_t max_iterations() const noexcept { return max_iterations_; }

        /// @brief Returns the configured initial temperature.
        /// @details Defaults to 1.0.
        /// @return Initial annealing temperature T_0.
        [[nodiscard]] double initial_temperature() const noexcept { return initial_temperature_; }

        /// @brief Returns the configured cooling rate.
        /// @details Defaults to 3.0.
        /// @return Ingber exponential cooling rate c.
        [[nodiscard]] double cooling_rate() const noexcept { return cooling_rate_; }

        /// @brief Returns the configured PRNG seed.
        /// @details Defaults to 0.
        /// @return Base 64-bit PRNG seed.
        [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

        /// @brief Returns the configured boundary policy.
        /// @details Defaults to BoundaryPolicy::Clamp.
        /// @return Active BoundaryPolicy.
        [[nodiscard]] BoundaryPolicy boundary_policy() const noexcept { return boundary_policy_; }

        /// @brief Returns the configured restart policy.
        /// @details Defaults to zero extra restarts.
        /// @return Active RestartConfig.
        [[nodiscard]] RestartConfig restarts() const noexcept { return restarts_; }

        /// @brief Minimizes obj starting from log-space initial_params.
        /// @details Validates initial_temperature_ > 0 and cooling_rate_ >= 0, validates bounds and
        /// initial_params, runs single or multi-start Fast Simulated Annealing, and returns the lowest
        /// feasible objective parameters observed.
        /// @param[in,out] obj Objective evaluated in log-theta space (supports value_at_changes()).
        /// @param[in] initial_params Starting log-space parameter vector strictly inside objective bounds.
        /// @return Best OptimizerResult across the initial run and restarts, or GprError on invalid config/input.
        /// @retval GprErrorCode::InvalidConfig When initial_temperature <= 0 or cooling_rate < 0 or non-finite.
        /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
        /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
        /// @retval GprErrorCode::NonFiniteInput When initial_params or initial objective value is non-finite.
        /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
        [[nodiscard]] std::expected<OptimizerResult, GprError> optimize(
            Objective& obj, std::span<const double> initial_params
        ) const;

        /// @brief Checks value equality between two FastSimulatedAnnealing configurations.
        /// @details Compares all configuration parameters and restart settings.
        /// @param[in] other FastSimulatedAnnealing instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const FastSimulatedAnnealing& other) const noexcept = default;

    private:
        /// @brief Maximum annealing iterations per start.
        std::size_t max_iterations_{100U};
        /// @brief Initial annealing temperature T_0 (> 0).
        double initial_temperature_{1.0};
        /// @brief Ingber exponential cooling rate (>= 0).
        double cooling_rate_{3.0};
        /// @brief Base 64-bit PRNG seed for Cauchy proposals and Metropolis draws.
        std::uint64_t seed_{0U};
        /// @brief Boundary projection policy applied to out-of-bounds proposals.
        BoundaryPolicy boundary_policy_{BoundaryPolicy::Clamp};
        /// @brief Deterministic multi-start restart configuration.
        RestartConfig restarts_{};
    };

}  // namespace cxxgpr::optimizer
