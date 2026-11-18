// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

namespace cxxgpr::optimizer
{

    /// Log-space open parameter bounds (log(lo), log(hi)) with midpoint unit-Jacobian logit
    /// mapping to and from unconstrained space z in R^d.
    class LogBounds
    {
    public:
        /// Builds log-space bounds from Objective::fill_intervals(), requiring strictly positive open
        /// intervals (0 < lo < hi).
        [[nodiscard]] static std::expected<LogBounds, GprError> from_objective(const Objective& obj);

        /// Builds log-space bounds from explicit user-space open intervals (0 < lo < hi).
        [[nodiscard]] static std::expected<LogBounds, GprError> from_intervals(std::span<const Interval> intervals);

        /// Returns the parameter dimension.
        [[nodiscard]] std::size_t dim() const noexcept { return lo_log_.size(); }

        /// Returns the exclusive lower bounds in log-theta space.
        [[nodiscard]] std::span<const double> lo_log() const noexcept { return lo_log_; }

        /// Returns the exclusive upper bounds in log-theta space.
        [[nodiscard]] std::span<const double> hi_log() const noexcept { return hi_log_; }

        /// Validates that theta is finite, has length dim(), and lies strictly inside (lo_log, hi_log).
        [[nodiscard]] std::expected<void, GprError> validate_log_params(std::span<const double> theta) const;

        /// Maps constrained log-space theta into unconstrained z with d(theta)/dz = 1 at the midpoint.
        [[nodiscard]] std::expected<void, GprError> to_unconstrained(
            std::span<const double> theta, std::span<double> out_z
        ) const;

        /// Maps unconstrained z back into strictly interior log-space theta.
        void to_constrained(std::span<const double> z, std::span<double> out_theta) const noexcept;

        /// Maps unconstrained z to theta and computes first derivatives d(theta_i)/dz_i.
        void to_constrained_with_jacobian(
            std::span<const double> z, std::span<double> out_theta, std::span<double> out_dtheta_dz
        ) const noexcept;

        /// Maps unconstrained z to theta and computes first and second derivatives d(theta_i)/dz_i
        /// and d^2(theta_i)/dz_i^2.
        void to_constrained_with_derivatives(
            std::span<const double> z, std::span<double> out_theta, std::span<double> out_d1, std::span<double> out_d2
        ) const noexcept;

        /// Applies the chain rule g_z[i] = g_theta[i] * d1[i].
        static void chain_gradient(
            std::span<const double> g_theta, std::span<const double> d1, std::span<double> out_g_z
        ) noexcept;

        /// Applies the second-order chain rule H_z = J^T H_theta J + diag(g_theta * d2).
        static void chain_hessian(
            std::span<const double> h_theta,
            std::span<const double> g_theta,
            std::span<const double> d1,
            std::span<const double> d2,
            std::span<double> out_h_z
        ) noexcept;

        /// Samples a deterministic log-uniform point strictly inside (lo_log, hi_log).
        void sample_log_uniform(Xoshiro256PlusPlus& rng, std::span<double> out_theta) const noexcept;

    private:
        LogBounds(std::vector<double> lo_log, std::vector<double> hi_log) noexcept :
            lo_log_(std::move(lo_log)),
            hi_log_(std::move(hi_log))
        {
        }

        std::vector<double> lo_log_;
        std::vector<double> hi_log_;
    };

    /// Single-entry evaluation cache avoiding redundant value/gradient evaluations at identical
    /// unconstrained points during line search and step acceptance.
    class EvalCache
    {
    public:
        /// Clears any cached evaluation.
        void clear() noexcept { valid_ = false; }

        /// Evaluates (or retrieves cached) objective value and gradient at unconstrained z.
        [[nodiscard]] std::expected<double, GprError> eval_unconstrained(
            Differentiable& obj,
            const LogBounds& bounds,
            std::span<const double> z,
            std::span<double> theta_buf,
            std::span<double> d1_buf,
            std::span<double> grad_theta_buf,
            std::span<double> out_grad_z
        );

    private:
        bool valid_{false};
        std::vector<double> last_z_;
        double last_val_{0.0};
        std::vector<double> last_grad_z_;
    };

}  // namespace cxxgpr::optimizer
