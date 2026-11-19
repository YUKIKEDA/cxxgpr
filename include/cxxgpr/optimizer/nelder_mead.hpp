// SPDX-License-Identifier: Apache-2.0
/// @file nelder_mead.hpp
/// @brief Derivative-free Nelder-Mead simplex optimizer over logit-transformed parameter bounds.
/// @details Declares NelderMead using standard simplex reflection, expansion, contraction, and shrink
/// steps in unconstrained midpoint unit-Jacobian logit space via Objective::value().
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

    /// @brief Derivative-free Nelder-Mead simplex optimizer operating in logit coordinates.
    /// @details Constructs an axis-aligned initial simplex around the logit-transformed starting point,
    /// performs reflection, expansion, outside/inside contraction, and shrink updates using only
    /// Objective::value(), and supports deterministic multi-start restarts.
    class NelderMead
    {
    public:
        /// @brief NelderMead does not use single-coordinate incremental updates.
        static constexpr bool uses_change_indices = false;

        /// @brief Constructs NelderMead with default settings.
        /// @details Defaults to max_iterations = 100, tolerance = sqrt(eps), and zero extra restarts.
        NelderMead() noexcept = default;

        /// @brief Returns a copy with updated maximum simplex iterations.
        /// @details Bounds the number of simplex reflection/contraction/shrink iterations per start.
        /// @param[in] max_iterations Maximum number of simplex iterations per start.
        /// @return Updated NelderMead configuration copy.
        [[nodiscard]] NelderMead with_max_iterations(std::size_t max_iterations) const noexcept
        {
            NelderMead copy = *this;
            copy.max_iterations_ = max_iterations;
            return copy;
        }

        /// @brief Returns a copy with updated simplex value and diameter convergence tolerance.
        /// @details Validated at optimize() time to be finite and non-negative.
        /// @param[in] tolerance Convergence threshold for both function spread and simplex infinity-norm radius (>= 0).
        /// @return Updated NelderMead configuration copy.
        [[nodiscard]] NelderMead with_tolerance(double tolerance) const noexcept
        {
            NelderMead copy = *this;
            copy.tolerance_ = tolerance;
            return copy;
        }

        /// @brief Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
        /// @details Runs count additional starts from log-uniform samples after evaluating initial_params.
        /// @param[in] count Number of extra random restarts.
        /// @param[in] seed Seed used to generate restart starting points.
        /// @return Updated NelderMead configuration copy.
        [[nodiscard]] NelderMead with_restarts(std::size_t count, std::uint64_t seed = 0U) const noexcept
        {
            NelderMead copy = *this;
            copy.restarts_ = RestartConfig{count, seed};
            return copy;
        }

        /// @brief Returns the configured maximum iterations.
        /// @details Defaults to 100.
        /// @return Maximum simplex iteration count per start.
        [[nodiscard]] std::size_t max_iterations() const noexcept { return max_iterations_; }

        /// @brief Returns the configured convergence tolerance.
        /// @details Defaults to sqrt(std::numeric_limits<double>::epsilon()).
        /// @return Convergence tolerance threshold.
        [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

        /// @brief Returns the configured restart policy.
        /// @details Defaults to zero extra restarts.
        /// @return Active RestartConfig.
        [[nodiscard]] RestartConfig restarts() const noexcept { return restarts_; }

        /// @brief Minimizes obj starting from log-space initial_params.
        /// @details Validates tolerance_ >= 0, validates bounds and initial_params, and runs single or
        /// multi-start Nelder-Mead optimization in unconstrained logit coordinates.
        /// @param[in,out] obj Objective evaluated in log-theta space.
        /// @param[in] initial_params Starting log-space parameter vector strictly inside objective bounds.
        /// @return Best OptimizerResult across initial run and restarts, or GprError on invalid config/input.
        /// @retval GprErrorCode::InvalidConfig When tolerance is non-finite or negative.
        /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
        /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
        /// @retval GprErrorCode::NonFiniteInput When initial_params or simplex evaluation yields non-finite values.
        /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
        [[nodiscard]] std::expected<OptimizerResult, GprError> optimize(
            Objective& obj, std::span<const double> initial_params
        ) const;

        /// @brief Checks value equality between two NelderMead configurations.
        /// @details Compares max_iterations_, tolerance_, and restarts_.
        /// @param[in] other NelderMead instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const NelderMead& other) const noexcept = default;

    private:
        /// @brief Maximum number of simplex iterations per start.
        std::size_t max_iterations_{100U};
        /// @brief Convergence tolerance on simplex function spread and vertex distance.
        double tolerance_{std::sqrt(std::numeric_limits<double>::epsilon())};
        /// @brief Deterministic multi-start restart configuration.
        RestartConfig restarts_{};
    };

}  // namespace cxxgpr::optimizer
