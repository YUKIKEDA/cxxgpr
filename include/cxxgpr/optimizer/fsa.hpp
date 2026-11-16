// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace cxxgpr::optimizer {

/// Selects how FastSimulatedAnnealing handles proposals outside open log-theta bounds.
enum class BoundaryPolicy : std::uint8_t {
    /// Clamps out-of-bounds proposals strictly inside the open interval.
    Clamp,
    /// Reflects out-of-bounds proposals back across interval boundaries.
    Reflect,
};

/// Fast Simulated Annealing optimizer operating directly in log-theta space using Cauchy
/// proposals, Metropolis acceptance, Ingber exponential cooling, and incremental single-coordinate
/// updates via Objective::value_at_changes().
class FastSimulatedAnnealing {
public:
    /// FSA uses single-coordinate incremental updates via value_at_changes().
    static constexpr bool uses_change_indices = true;

    /// Constructs FastSimulatedAnnealing with default settings (max_iterations=100,
    /// initial_temperature=1.0, cooling_rate=3.0, seed=0, boundary_policy=Clamp).
    FastSimulatedAnnealing() noexcept = default;

    /// Sets the maximum number of annealing iterations.
    [[nodiscard]] FastSimulatedAnnealing
    with_max_iterations(std::size_t max_iterations) const noexcept {
        FastSimulatedAnnealing copy = *this;
        copy.max_iterations_ = max_iterations;
        return copy;
    }

    /// Sets the initial annealing temperature (> 0).
    [[nodiscard]] FastSimulatedAnnealing
    with_initial_temperature(double initial_temperature) const noexcept {
        FastSimulatedAnnealing copy = *this;
        copy.initial_temperature_ = initial_temperature;
        return copy;
    }

    /// Sets the Ingber exponential cooling rate (>= 0).
    [[nodiscard]] FastSimulatedAnnealing with_cooling_rate(double cooling_rate) const noexcept {
        FastSimulatedAnnealing copy = *this;
        copy.cooling_rate_ = cooling_rate;
        return copy;
    }

    /// Sets the Xoshiro256++ seed for Cauchy proposals and Metropolis acceptance.
    [[nodiscard]] FastSimulatedAnnealing with_seed(std::uint64_t seed) const noexcept {
        FastSimulatedAnnealing copy = *this;
        copy.seed_ = seed;
        return copy;
    }

    /// Sets the boundary handling policy (Clamp or Reflect).
    [[nodiscard]] FastSimulatedAnnealing
    with_boundary_policy(BoundaryPolicy boundary_policy) const noexcept {
        FastSimulatedAnnealing copy = *this;
        copy.boundary_policy_ = boundary_policy;
        return copy;
    }

    /// Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
    [[nodiscard]] FastSimulatedAnnealing with_restarts(std::size_t count,
                                                       std::uint64_t seed = 0U) const noexcept {
        FastSimulatedAnnealing copy = *this;
        copy.restarts_ = RestartConfig{count, seed};
        return copy;
    }

    /// Returns the configured maximum iterations.
    [[nodiscard]] std::size_t max_iterations() const noexcept {
        return max_iterations_;
    }

    /// Returns the configured initial temperature.
    [[nodiscard]] double initial_temperature() const noexcept {
        return initial_temperature_;
    }

    /// Returns the configured cooling rate.
    [[nodiscard]] double cooling_rate() const noexcept {
        return cooling_rate_;
    }

    /// Returns the configured PRNG seed.
    [[nodiscard]] std::uint64_t seed() const noexcept {
        return seed_;
    }

    /// Returns the configured boundary policy.
    [[nodiscard]] BoundaryPolicy boundary_policy() const noexcept {
        return boundary_policy_;
    }

    /// Returns the configured restart policy.
    [[nodiscard]] RestartConfig restarts() const noexcept {
        return restarts_;
    }

    /// Minimizes obj starting from log-space initial_params.
    [[nodiscard]] std::expected<OptimizerResult, GprError>
    optimize(Objective& obj, std::span<const double> initial_params) const;

    /// Checks value equality between two FastSimulatedAnnealing configurations.
    [[nodiscard]] bool operator==(const FastSimulatedAnnealing& other) const noexcept = default;

private:
    std::size_t max_iterations_{100U};
    double initial_temperature_{1.0};
    double cooling_rate_{3.0};
    std::uint64_t seed_{0U};
    BoundaryPolicy boundary_policy_{BoundaryPolicy::Clamp};
    RestartConfig restarts_{};
};

} // namespace cxxgpr::optimizer
