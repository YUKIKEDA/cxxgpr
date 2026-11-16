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

namespace cxxgpr::optimizer {

/// Trust-region Newton optimizer using analytical Hessians (with second-order logit chain rule)
/// and Steihaug-Toint truncated conjugate gradient subproblem solves.
class TrustRegion {
public:
    /// TrustRegion does not use single-coordinate incremental updates.
    static constexpr bool uses_change_indices = false;

    /// Constructs TrustRegion with default settings (max_iterations=100, tolerance=sqrt(eps),
    /// initial_radius=1.0, max_radius=100.0).
    TrustRegion() noexcept = default;

    /// Sets the maximum number of trust-region outer iterations.
    [[nodiscard]] TrustRegion with_max_iterations(std::size_t max_iterations) const noexcept {
        TrustRegion copy = *this;
        copy.max_iterations_ = max_iterations;
        return copy;
    }

    /// Sets the gradient norm convergence tolerance.
    [[nodiscard]] TrustRegion with_tolerance(double tolerance) const noexcept {
        TrustRegion copy = *this;
        copy.tolerance_ = tolerance;
        return copy;
    }

    /// Sets the initial trust-region radius in unconstrained logit coordinates.
    [[nodiscard]] TrustRegion with_initial_radius(double initial_radius) const noexcept {
        TrustRegion copy = *this;
        copy.initial_radius_ = initial_radius;
        return copy;
    }

    /// Sets the maximum allowed trust-region radius.
    [[nodiscard]] TrustRegion with_max_radius(double max_radius) const noexcept {
        TrustRegion copy = *this;
        copy.max_radius_ = max_radius;
        return copy;
    }

    /// Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
    [[nodiscard]] TrustRegion with_restarts(std::size_t count,
                                            std::uint64_t seed = 0U) const noexcept {
        TrustRegion copy = *this;
        copy.restarts_ = RestartConfig{count, seed};
        return copy;
    }

    /// Returns the configured maximum iterations.
    [[nodiscard]] std::size_t max_iterations() const noexcept {
        return max_iterations_;
    }

    /// Returns the configured convergence tolerance.
    [[nodiscard]] double tolerance() const noexcept {
        return tolerance_;
    }

    /// Returns the configured initial trust-region radius.
    [[nodiscard]] double initial_radius() const noexcept {
        return initial_radius_;
    }

    /// Returns the configured maximum trust-region radius.
    [[nodiscard]] double max_radius() const noexcept {
        return max_radius_;
    }

    /// Returns the configured restart policy.
    [[nodiscard]] RestartConfig restarts() const noexcept {
        return restarts_;
    }

    /// Minimizes obj starting from log-space initial_params.
    [[nodiscard]] std::expected<OptimizerResult, GprError>
    optimize(TwiceDifferentiable& obj, std::span<const double> initial_params) const;

    /// Checks value equality between two TrustRegion configurations.
    [[nodiscard]] bool operator==(const TrustRegion& other) const noexcept = default;

private:
    std::size_t max_iterations_{100U};
    double tolerance_{std::sqrt(std::numeric_limits<double>::epsilon())};
    double initial_radius_{1.0};
    double max_radius_{100.0};
    RestartConfig restarts_{};
};

} // namespace cxxgpr::optimizer
