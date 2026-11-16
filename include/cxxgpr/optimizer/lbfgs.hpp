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

/// Limited-memory BFGS optimizer using Nocedal 2-loop recursion and Moré-Thuente line search
/// over midpoint unit-Jacobian logit-transformed parameter bounds.
class Lbfgs {
public:
    /// L-BFGS does not use single-coordinate incremental updates.
    static constexpr bool uses_change_indices = false;

    /// Constructs Lbfgs with default settings (max_iterations=100, tolerance=sqrt(eps),
    /// history_size=10).
    Lbfgs() noexcept = default;

    /// Sets the maximum number of L-BFGS iterations.
    [[nodiscard]] Lbfgs with_max_iterations(std::size_t max_iterations) const noexcept {
        Lbfgs copy = *this;
        copy.max_iterations_ = max_iterations;
        return copy;
    }

    /// Sets the relative gradient / function convergence tolerance.
    [[nodiscard]] Lbfgs with_tolerance(double tolerance) const noexcept {
        Lbfgs copy = *this;
        copy.tolerance_ = tolerance;
        return copy;
    }

    /// Sets the number of (s, y) correction pairs retained for 2-loop recursion.
    [[nodiscard]] Lbfgs with_history_size(std::size_t history_size) const noexcept {
        Lbfgs copy = *this;
        copy.history_size_ = history_size;
        return copy;
    }

    /// Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
    [[nodiscard]] Lbfgs with_restarts(std::size_t count, std::uint64_t seed = 0U) const noexcept {
        Lbfgs copy = *this;
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

    /// Returns the configured L-BFGS history size.
    [[nodiscard]] std::size_t history_size() const noexcept {
        return history_size_;
    }

    /// Returns the configured restart policy.
    [[nodiscard]] RestartConfig restarts() const noexcept {
        return restarts_;
    }

    /// Minimizes obj starting from log-space initial_params.
    [[nodiscard]] std::expected<OptimizerResult, GprError>
    optimize(Differentiable& obj, std::span<const double> initial_params) const;

    /// Checks value equality between two Lbfgs configurations.
    [[nodiscard]] bool operator==(const Lbfgs& other) const noexcept = default;

private:
    std::size_t max_iterations_{100U};
    double tolerance_{std::sqrt(std::numeric_limits<double>::epsilon())};
    std::size_t history_size_{10U};
    RestartConfig restarts_{};
};

} // namespace cxxgpr::optimizer
