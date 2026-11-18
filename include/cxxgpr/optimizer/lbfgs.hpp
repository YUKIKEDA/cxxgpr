// SPDX-License-Identifier: Apache-2.0
/// @file lbfgs.hpp
/// @brief Limited-memory BFGS quasi-Newton optimizer over logit-transformed parameter bounds.
/// @details Declares Lbfgs using Nocedal two-loop inverse Hessian approximation and Moré–Thuente
/// strong-Wolfe line search over midpoint unit-Jacobian logit coordinates.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

/// @brief Hyperparameter optimization interfaces, coordinate transforms, and solver implementations.
namespace cxxgpr::optimizer
{

    /// @brief Limited-memory BFGS optimizer using Nocedal two-loop recursion and Moré–Thuente line search.
    /// @details Solves unconstrained optimization in midpoint unit-Jacobian logit space z, maintaining
    /// up to history_size_ curvature pairs (s, y) and supporting deterministic multi-start restarts.
    class Lbfgs
    {
    public:
        /// @brief L-BFGS does not use single-coordinate incremental updates.
        static constexpr bool uses_change_indices = false;

        /// @brief Constructs Lbfgs with default settings.
        /// @details Defaults to max_iterations = 100, tolerance = sqrt(eps), history_size = 10,
        /// and zero extra restarts.
        Lbfgs() noexcept = default;

        /// @brief Returns a copy with updated maximum L-BFGS iterations.
        /// @details Bounds the number of outer quasi-Newton steps per start.
        /// @param max_iterations Maximum number of L-BFGS iterations per start.
        /// @return Updated Lbfgs configuration copy.
        [[nodiscard]] Lbfgs with_max_iterations(std::size_t max_iterations) const noexcept
        {
            Lbfgs copy = *this;
            copy.max_iterations_ = max_iterations;
            return copy;
        }

        /// @brief Returns a copy with updated relative gradient and function convergence tolerance.
        /// @details Validated at optimize() time to be finite and non-negative.
        /// @param tolerance Convergence tolerance threshold (>= 0).
        /// @return Updated Lbfgs configuration copy.
        [[nodiscard]] Lbfgs with_tolerance(double tolerance) const noexcept
        {
            Lbfgs copy = *this;
            copy.tolerance_ = tolerance;
            return copy;
        }

        /// @brief Returns a copy with updated number of (s, y) correction pairs retained.
        /// @details Validated at optimize() time to be at least 1.
        /// @param history_size Number of recent secant pairs used in two-loop recursion (>= 1).
        /// @return Updated Lbfgs configuration copy.
        [[nodiscard]] Lbfgs with_history_size(std::size_t history_size) const noexcept
        {
            Lbfgs copy = *this;
            copy.history_size_ = history_size;
            return copy;
        }

        /// @brief Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
        /// @details Runs count additional starts from log-uniform samples after evaluating initial_params.
        /// @param count Number of extra random restarts.
        /// @param seed Seed used to generate restart starting points.
        /// @return Updated Lbfgs configuration copy.
        [[nodiscard]] Lbfgs with_restarts(std::size_t count, std::uint64_t seed = 0U) const noexcept
        {
            Lbfgs copy = *this;
            copy.restarts_ = RestartConfig{count, seed};
            return copy;
        }

        /// @brief Returns the configured maximum iterations.
        /// @details Defaults to 100.
        /// @return Maximum L-BFGS iteration count per start.
        [[nodiscard]] std::size_t max_iterations() const noexcept { return max_iterations_; }

        /// @brief Returns the configured convergence tolerance.
        /// @details Defaults to sqrt(std::numeric_limits<double>::epsilon()).
        /// @return Convergence tolerance threshold.
        [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

        /// @brief Returns the configured L-BFGS history size.
        /// @details Defaults to 10.
        /// @return Maximum number of stored (s, y) curvature pairs.
        [[nodiscard]] std::size_t history_size() const noexcept { return history_size_; }

        /// @brief Returns the configured restart policy.
        /// @details Defaults to zero extra restarts.
        /// @return Active RestartConfig.
        [[nodiscard]] RestartConfig restarts() const noexcept { return restarts_; }

        /// @brief Minimizes obj starting from log-space initial_params.
        /// @details Validates tolerance_ >= 0 and history_size_ >= 1, validates bounds and initial_params,
        /// and executes single or multi-start L-BFGS optimization in unconstrained logit coordinates.
        /// @param obj Differentiable objective evaluated in log-theta space.
        /// @param initial_params Starting log-space parameter vector strictly inside objective bounds.
        /// @return Best OptimizerResult across initial run and restarts, or GprError on invalid config/input.
        /// @retval GprErrorCode::InvalidConfig When tolerance is invalid/negative or history_size == 0.
        /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
        /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
        /// @retval GprErrorCode::NonFiniteInput When initial_params or initial objective/gradient is non-finite.
        /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
        [[nodiscard]] std::expected<OptimizerResult, GprError> optimize(
            Differentiable& obj, std::span<const double> initial_params
        ) const;

        /// @brief Checks value equality between two Lbfgs configurations.
        /// @details Compares max_iterations_, tolerance_, history_size_, and restarts_.
        /// @param other Lbfgs instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const Lbfgs& other) const noexcept = default;

    private:
        /// @brief Maximum number of outer L-BFGS iterations per start.
        std::size_t max_iterations_{100U};
        /// @brief Relative gradient and function-change convergence tolerance.
        double tolerance_{std::sqrt(std::numeric_limits<double>::epsilon())};
        /// @brief Maximum number of (s, y) secant pairs kept for two-loop recursion.
        std::size_t history_size_{10U};
        /// @brief Deterministic multi-start restart configuration.
        RestartConfig restarts_{};
    };

}  // namespace cxxgpr::optimizer
