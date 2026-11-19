// SPDX-License-Identifier: Apache-2.0
/// @file trust_region.hpp
/// @brief Second-order Trust-Region Newton optimizer over logit-transformed parameter bounds.
/// @details Declares TrustRegion using analytical Hessians transformed via the second-order logit
/// chain rule and Steihaug-Toint truncated conjugate gradient subproblem solves.
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

    /// @brief Trust-region Newton optimizer using analytical Hessians and Steihaug-Toint truncated CG.
    /// @details Operates in unconstrained midpoint unit-Jacobian logit space z, propagating exact
    /// first- and second-order derivatives via LogBounds::chain_gradient() and LogBounds::chain_hessian()
    /// while handling indefinite Hessians and trust-region boundaries with Steihaug-Toint CG.
    class TrustRegion
    {
    public:
        /// @brief TrustRegion does not use single-coordinate incremental updates.
        static constexpr bool uses_change_indices = false;

        /// @brief Constructs TrustRegion with default settings.
        /// @details Defaults to max_iterations = 100, tolerance = sqrt(eps), initial_radius = 1.0,
        /// max_radius = 100.0, and zero extra restarts.
        TrustRegion() noexcept = default;

        /// @brief Returns a copy with updated maximum trust-region outer iterations.
        /// @details Bounds the number of trust-region quadratic subproblem trials per start.
        /// @param[in] max_iterations Maximum number of outer trust-region iterations per start.
        /// @return Updated TrustRegion configuration copy.
        [[nodiscard]] TrustRegion with_max_iterations(std::size_t max_iterations) const noexcept
        {
            TrustRegion copy = *this;
            copy.max_iterations_ = max_iterations;
            return copy;
        }

        /// @brief Returns a copy with updated gradient norm and step convergence tolerance.
        /// @details Validated at optimize() time to be finite and non-negative.
        /// @param[in] tolerance Convergence tolerance threshold (>= 0).
        /// @return Updated TrustRegion configuration copy.
        [[nodiscard]] TrustRegion with_tolerance(double tolerance) const noexcept
        {
            TrustRegion copy = *this;
            copy.tolerance_ = tolerance;
            return copy;
        }

        /// @brief Returns a copy with updated initial trust-region radius in unconstrained logit coordinates.
        /// @details Validated at optimize() time to be finite, positive, and <= max_radius_.
        /// @param[in] initial_radius Starting trust-region radius (> 0).
        /// @return Updated TrustRegion configuration copy.
        [[nodiscard]] TrustRegion with_initial_radius(double initial_radius) const noexcept
        {
            TrustRegion copy = *this;
            copy.initial_radius_ = initial_radius;
            return copy;
        }

        /// @brief Returns a copy with updated maximum allowed trust-region radius.
        /// @details Validated at optimize() time to be finite and >= initial_radius_.
        /// @param[in] max_radius Upper bound on adaptive trust-region radius (>= initial_radius_).
        /// @return Updated TrustRegion configuration copy.
        [[nodiscard]] TrustRegion with_max_radius(double max_radius) const noexcept
        {
            TrustRegion copy = *this;
            copy.max_radius_ = max_radius;
            return copy;
        }

        /// @brief Configures deterministic multi-start restarts sampled log-uniformly via Xoshiro256++.
        /// @details Runs count additional starts from log-uniform samples after evaluating initial_params.
        /// @param[in] count Number of extra random restarts.
        /// @param[in] seed Seed used to generate restart starting points.
        /// @return Updated TrustRegion configuration copy.
        [[nodiscard]] TrustRegion with_restarts(std::size_t count, std::uint64_t seed = 0U) const noexcept
        {
            TrustRegion copy = *this;
            copy.restarts_ = RestartConfig{count, seed};
            return copy;
        }

        /// @brief Returns the configured maximum iterations.
        /// @details Defaults to 100.
        /// @return Maximum trust-region outer iteration count per start.
        [[nodiscard]] std::size_t max_iterations() const noexcept { return max_iterations_; }

        /// @brief Returns the configured convergence tolerance.
        /// @details Defaults to sqrt(std::numeric_limits<double>::epsilon()).
        /// @return Convergence tolerance threshold.
        [[nodiscard]] double tolerance() const noexcept { return tolerance_; }

        /// @brief Returns the configured initial trust-region radius.
        /// @details Defaults to 1.0.
        /// @return Initial trust-region radius.
        [[nodiscard]] double initial_radius() const noexcept { return initial_radius_; }

        /// @brief Returns the configured maximum trust-region radius.
        /// @details Defaults to 100.0.
        /// @return Maximum allowed trust-region radius.
        [[nodiscard]] double max_radius() const noexcept { return max_radius_; }

        /// @brief Returns the configured restart policy.
        /// @details Defaults to zero extra restarts.
        /// @return Active RestartConfig.
        [[nodiscard]] RestartConfig restarts() const noexcept { return restarts_; }

        /// @brief Minimizes obj starting from log-space initial_params.
        /// @details Validates tolerance_ >= 0, 0 < initial_radius_ <= max_radius_, validates bounds and
        /// initial_params, and runs single or multi-start trust-region Newton optimization.
        /// @param[in,out] obj TwiceDifferentiable objective providing value, analytical gradient, and Hessian.
        /// @param[in] initial_params Starting log-space parameter vector strictly inside objective bounds.
        /// @return Best OptimizerResult across initial run and restarts, or GprError on invalid config/input.
        /// @retval GprErrorCode::InvalidConfig When tolerance or radius configuration is invalid.
        /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
        /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
        /// @retval GprErrorCode::NonFiniteInput When initial_params or initial value/gradient/Hessian is non-finite.
        /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
        [[nodiscard]] std::expected<OptimizerResult, GprError> optimize(
            TwiceDifferentiable& obj, std::span<const double> initial_params
        ) const;

        /// @brief Checks value equality between two TrustRegion configurations.
        /// @details Compares max_iterations_, tolerance_, initial_radius_, max_radius_, and restarts_.
        /// @param[in] other TrustRegion instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const TrustRegion& other) const noexcept = default;

    private:
        /// @brief Maximum number of outer trust-region iterations per start.
        std::size_t max_iterations_{100U};
        /// @brief Relative gradient norm and reduction convergence tolerance.
        double tolerance_{std::sqrt(std::numeric_limits<double>::epsilon())};
        /// @brief Initial trust-region radius in unconstrained logit space.
        double initial_radius_{1.0};
        /// @brief Maximum trust-region radius ceiling.
        double max_radius_{100.0};
        /// @brief Deterministic multi-start restart configuration.
        RestartConfig restarts_{};
    };

}  // namespace cxxgpr::optimizer
