// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

namespace cxxgpr::optimizer
{

    /// Derivative-free Nelder-Mead simplex optimizer operating in midpoint unit-Jacobian logit
    /// coordinates using only Objective::value().
    class NelderMead
    {
    public:
        /// NelderMead does not use single-coordinate incremental updates.
        static constexpr bool uses_change_indices = false;

        /// Constructs NelderMead with default settings (max_iterations=100, tolerance=sqrt(eps)).
        NelderMead() noexcept = default;

        /// Sets the maximum number of simplex iterations.
        [[nodiscard]] NelderMead with_max_iterations(std::size_t max_iterations) const noexcept
        {
            NelderMead copy = *this;
            copy.max_iterations_ = max_iterations;
            return copy;
        }

        /// Sets the simplex value and diameter convergence tolerance.
        [[nodiscard]] NelderMead with_tolerance(double tolerance) const noexcept
        {
            NelderMead copy = *this;
            copy.tolerance_ = tolerance;
            return copy;
        }

        /// Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
        [[nodiscard]] NelderMead with_restarts(std::size_t count, std::uint64_t seed = 0U) const noexcept
        {
            NelderMead copy = *this;
            copy.restarts_ = RestartConfig{count, seed};
            return copy;
        }

        /// Returns the configured maximum iterations.
        [[nodiscard]] std::size_t max_iterations() const noexcept { return max_iterations_; }

        /// Returns the configured convergence tolerance.
        [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

        /// Returns the configured restart policy.
        [[nodiscard]] RestartConfig restarts() const noexcept { return restarts_; }

        /// Minimizes obj starting from log-space initial_params.
        [[nodiscard]] std::expected<OptimizerResult, GprError> optimize(
            Objective& obj, std::span<const double> initial_params
        ) const;

        /// Checks value equality between two NelderMead configurations.
        [[nodiscard]] bool operator==(const NelderMead& other) const noexcept = default;

    private:
        std::size_t max_iterations_{100U};
        double tolerance_{std::sqrt(std::numeric_limits<double>::epsilon())};
        RestartConfig restarts_{};
    };

}  // namespace cxxgpr::optimizer
