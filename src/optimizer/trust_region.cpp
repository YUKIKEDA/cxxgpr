// SPDX-License-Identifier: Apache-2.0
/// @file trust_region.cpp
/// @brief Implementation of Trust-Region Newton optimization using Steihaug-Toint truncated CG.
/// @details Implements dense matrix-vector products, Steihaug-Toint trust-region subproblem solves,
/// second-order logit chain-rule evaluation, adaptive radius updates, and multi-start restarts.

#include "cxxgpr/optimizer/trust_region.hpp"

#include "cxxgpr/optimizer/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.hpp"

namespace cxxgpr::optimizer
{
    namespace
    {

        /// @brief Computes out = H * v for a flattened d x d symmetric Hessian matrix h.
        /// @details Reads entry (row, col) at (col * d) + row across all columns of h.
        /// @param h Flattened d x d Hessian span.
        /// @param v Input vector span of length d.
        /// @param out Destination span of length d receiving H * v.
        void sym_matvec(std::span<const double> h, std::span<const double> v, std::span<double> out) noexcept
        {
            const std::size_t d = v.size();
            for (std::size_t row = 0; row < d; ++row)
            {
                double acc = 0.0;
                for (std::size_t col = 0; col < d; ++col)
                {
                    acc += h[(col * d) + row] * v[col];
                }
                out[row] = acc;
            }
        }

        /// @brief Computes the positive scalar tau >= 0 solving ||p + tau * d||_2^2 = radius^2.
        /// @details Evaluates the positive root of the quadratic dd * tau^2 + 2*(p^T d)*tau + (pp - r^2) = 0
        /// so the truncated CG step lands cleanly on the trust-region sphere boundary.
        /// @param p Current interior CG iterate vector.
        /// @param d Search direction along which to extend p.
        /// @param radius Current trust-region radius > ||p||_2.
        /// @return Non-negative boundary intersection distance tau along d (0.0 if ||d||_2 == 0).
        [[nodiscard]] double step_to_boundary(
            std::span<const double> p, std::span<const double> d, double radius
        ) noexcept
        {
            const double pp = detail::dot_vec(p, p);
            const double pd = detail::dot_vec(p, d);
            const double dd = detail::dot_vec(d, d);
            if (dd <= 0.0)
            {
                return 0.0;
            }
            // Solve ||p + tau * d||^2 = radius^2 for the positive step multiplier tau >= 0.
            const double disc = std::max(0.0, (pd * pd) + (dd * ((radius * radius) - pp)));
            return (-pd + std::sqrt(disc)) / dd;
        }

        /// @brief Approximately minimizes m(p) = g^T * p + 0.5 * p^T * H * p subject to ||p||_2 <= radius.
        /// @details Executes Steihaug-Toint truncated conjugate gradients: exits immediately if ||g|| == 0,
        /// jumps to the trust-region boundary upon encountering non-positive curvature (d^T H d <= 0) or
        /// when ||p_next||_2 >= radius, and stops early when residual norm satisfies Eisenstat-Walker forcing
        /// min(0.5, sqrt(||g||)) * ||g||.
        /// @param grad Unconstrained gradient g_z of length d.
        /// @param hess Flattened d x d unconstrained Hessian H_z.
        /// @param radius Current trust-region radius.
        /// @param out_step Destination span of length d receiving the computed trial step p.
        void steihaug_toint_cg(
            std::span<const double> grad, std::span<const double> hess, double radius, std::span<double> out_step
        )
        {
            const std::size_t d = grad.size();
            std::ranges::fill(out_step, 0.0);

            const double g_norm = detail::norm2_vec(grad);
            if (g_norm == 0.0)
            {
                return;
            }

            std::vector<double> r(grad.begin(), grad.end());
            std::vector<double> dir(d);
            for (std::size_t i = 0; i < d; ++i)
            {
                dir[i] = -r[i];
            }
            std::vector<double> h_dir(d);
            std::vector<double> p_next(d);

            // Eisenstat-Walker inexact-Newton residual stopping tolerance.
            double rr = detail::dot_vec(r, r);
            const double cg_tol = std::min(0.5, std::sqrt(g_norm)) * g_norm;
            const std::size_t max_cg_iters = std::max<std::size_t>(2U * d, 10U);

            for (std::size_t k = 0; k < max_cg_iters; ++k)
            {
                sym_matvec(hess, dir, h_dir);
                const double d_hd = detail::dot_vec(dir, h_dir);

                // Negative or non-finite curvature: move along dir to the trust-region boundary and stop.
                if (!std::isfinite(d_hd) || d_hd <= 0.0)
                {
                    const double tau = step_to_boundary(out_step, dir, radius);
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        out_step[i] += tau * dir[i];
                    }
                    return;
                }

                const double alpha = rr / d_hd;
                for (std::size_t i = 0; i < d; ++i)
                {
                    p_next[i] = out_step[i] + (alpha * dir[i]);
                }

                // Candidate leaves the trust region: truncate at the sphere boundary and return.
                if (detail::norm2_vec(p_next) >= radius)
                {
                    const double tau = step_to_boundary(out_step, dir, radius);
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        out_step[i] += tau * dir[i];
                    }
                    return;
                }

                // Advance CG iterate, update residual r += alpha * H * dir, and check forcing tolerance.
                std::ranges::copy(p_next, out_step.begin());
                for (std::size_t i = 0; i < d; ++i)
                {
                    r[i] += alpha * h_dir[i];
                }
                const double rr_next = detail::dot_vec(r, r);
                if (std::sqrt(rr_next) <= cg_tol)
                {
                    return;
                }

                const double beta = rr_next / rr;
                for (std::size_t i = 0; i < d; ++i)
                {
                    dir[i] = -r[i] + (beta * dir[i]);
                }
                rr = rr_next;
            }
        }

        /// @brief Evaluates objective value, gradient, and Hessian at unconstrained z using logit chain rules.
        /// @details Maps z -> (theta, d1, d2), evaluates obj.value_gradient_hessian_into(), checks finiteness
        /// of value, gradient, and Hessian, and transforms g_theta, H_theta into unconstrained g_z, H_z.
        /// @param obj TwiceDifferentiable objective evaluated in log-theta space.
        /// @param bounds Log-space bounds governing the z <-> theta transform.
        /// @param z Unconstrained logit parameter vector of length bounds.dim().
        /// @param theta_buf Caller-supplied scratch span of length d for constrained theta.
        /// @param d1_buf Caller-supplied scratch span of length d for first derivatives d(theta)/dz.
        /// @param d2_buf Caller-supplied scratch span of length d for second derivatives d^2(theta)/dz^2.
        /// @param g_theta_buf Caller-supplied scratch span of length d for constrained gradient.
        /// @param h_theta_buf Caller-supplied scratch span of length d * d for constrained Hessian.
        /// @param out_g_z Destination span of length d receiving unconstrained gradient g_z.
        /// @param out_h_z Destination span of length d * d receiving unconstrained Hessian H_z.
        /// @return Finite objective scalar value at z, or GprError if evaluation or finiteness checks fail.
        /// @retval GprErrorCode::NonFiniteInput When objective value, gradient, or Hessian contains non-finite values.
        [[nodiscard]] std::expected<double, GprError> eval_vgh_unconstrained(
            TwiceDifferentiable& obj,
            const LogBounds& bounds,
            std::span<const double> z,
            std::span<double> theta_buf,
            std::span<double> d1_buf,
            std::span<double> d2_buf,
            std::span<double> g_theta_buf,
            std::span<double> h_theta_buf,
            std::span<double> out_g_z,
            std::span<double> out_h_z
        )
        {
            // Transform z to interior theta alongside d1 and d2, then evaluate constrained value/grad/Hessian.
            bounds.to_constrained_with_derivatives(z, theta_buf, d1_buf, d2_buf);
            auto val = obj.value_gradient_hessian_into(theta_buf, g_theta_buf, h_theta_buf);
            if (!val.has_value())
            {
                return val;
            }
            if (!std::isfinite(*val))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            for (const double g : g_theta_buf)
            {
                if (!std::isfinite(g))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            for (const double h : h_theta_buf)
            {
                if (!std::isfinite(h))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            // Apply first- and second-order chain rules g_z = J * g_theta and H_z = J^T H_theta J + diag(g * d2).
            LogBounds::chain_gradient(g_theta_buf, d1_buf, out_g_z);
            LogBounds::chain_hessian(h_theta_buf, g_theta_buf, d1_buf, d2_buf, out_h_z);
            return *val;
        }

        /// @brief Runs a single-start Trust-Region Newton optimization trajectory from initial_theta.
        /// @details Maps initial_theta into unconstrained z, solves Steihaug-Toint trust-region subproblems,
        /// compares actual vs predicted quadratic model reduction rho, adapts trust-region radius within
        /// (0, max_radius], accepts steps with rho > 1e-4, and stops on gradient/reduction tolerance.
        /// @param obj TwiceDifferentiable objective evaluated in log-theta space.
        /// @param bounds Log-space bounds governing the z <-> theta transform.
        /// @param initial_theta Starting log-space parameter vector strictly inside bounds.
        /// @param max_iterations Maximum outer trust-region iterations.
        /// @param tolerance Gradient norm and relative reduction convergence tolerance.
        /// @param initial_radius Starting trust-region radius (> 0).
        /// @param max_radius Maximum allowed trust-region radius (>= initial_radius).
        /// @return OptimizerResult with best feasible log-space parameters found, or GprError if initial eval fails.
        /// @retval GprErrorCode::NonFiniteInput When initial objective, gradient, or Hessian is non-finite.
        [[nodiscard]] std::expected<OptimizerResult, GprError> run_trust_region_single(
            TwiceDifferentiable& obj,
            const LogBounds& bounds,
            std::span<const double> initial_theta,
            std::size_t max_iterations,
            double tolerance,
            double initial_radius,
            double max_radius
        )
        {
            const std::size_t d = bounds.dim();
            std::vector<double> z(d);
            if (auto u = bounds.to_unconstrained(initial_theta, z); !u.has_value())
            {
                return std::unexpected(std::move(u).error());
            }

            std::vector<double> theta_buf(d);
            std::vector<double> d1_buf(d);
            std::vector<double> d2_buf(d);
            std::vector<double> g_theta_buf(d);
            std::vector<double> h_theta_buf(d * d);
            std::vector<double> grad_z(d);
            std::vector<double> hess_z(d * d);

            auto f_res = eval_vgh_unconstrained(
                obj, bounds, z, theta_buf, d1_buf, d2_buf, g_theta_buf, h_theta_buf, grad_z, hess_z
            );
            if (!f_res.has_value())
            {
                return std::unexpected(std::move(f_res).error());
            }
            double f_cur = *f_res;
            std::vector<double> best_theta(initial_theta.begin(), initial_theta.end());
            double best_val = f_cur;

            std::vector<double> step(d);
            std::vector<double> hp(d);
            std::vector<double> z_trial(d);
            double radius = initial_radius;
            std::size_t iter = 0;
            bool converged = false;

            for (; iter < max_iterations; ++iter)
            {
                const double g_norm = detail::norm2_vec(grad_z);
                if (g_norm <= tolerance * std::max(1.0, detail::norm2_vec(z)))
                {
                    converged = true;
                    break;
                }

                // Solve the quadratic trust-region subproblem approximately via Steihaug-Toint CG.
                steihaug_toint_cg(grad_z, hess_z, radius, step);
                const double step_norm = detail::norm2_vec(step);
                if (step_norm == 0.0 || radius < 1e-15)
                {
                    break;
                }

                sym_matvec(hess_z, step, hp);
                const double pred_red = -(detail::dot_vec(grad_z, step) + (0.5 * detail::dot_vec(step, hp)));

                for (std::size_t i = 0; i < d; ++i)
                {
                    z_trial[i] = z[i] + step[i];
                }
                bounds.to_constrained(z_trial, theta_buf);
                auto trial_val = obj.value(theta_buf);

                // Shrink radius on failed/non-finite objective evaluations or non-positive model reduction.
                if (!trial_val.has_value() || !std::isfinite(*trial_val) || pred_red <= 0.0)
                {
                    radius *= 0.25;
                    continue;
                }

                // Adapt trust-region radius based on actual-to-predicted reduction ratio rho.
                const double actual_red = f_cur - *trial_val;
                const double rho = actual_red / pred_red;

                if (rho < 0.25)
                {
                    radius *= 0.25;
                }
                else if (rho > 0.75 && step_norm >= 0.99 * radius)
                {
                    radius = std::min(2.0 * radius, max_radius);
                }

                // Accept trial step if rho > 1e-4 and refresh unconstrained gradient/Hessian at z_trial.
                if (rho > 1e-4)
                {
                    auto next_res = eval_vgh_unconstrained(
                        obj, bounds, z_trial, theta_buf, d1_buf, d2_buf, g_theta_buf, h_theta_buf, grad_z, hess_z
                    );
                    if (!next_res.has_value())
                    {
                        radius *= 0.25;
                        continue;
                    }
                    z = z_trial;
                    f_cur = *next_res;
                    if (f_cur <= best_val)
                    {
                        best_val = f_cur;
                        best_theta.assign(theta_buf.begin(), theta_buf.end());
                    }
                    if (std::abs(actual_red) <= tolerance * tolerance * std::max(1.0, std::abs(f_cur))
                        && detail::norm2_vec(grad_z) <= tolerance * std::max(1.0, detail::norm2_vec(z)))
                    {
                        converged = true;
                        ++iter;
                        break;
                    }
                }
            }

            return OptimizerResult{
                .params = std::move(best_theta),
                .value = best_val,
                .iterations = iter,
                .converged = converged,
            };
        }

    }  // namespace

    /// @brief Minimizes obj starting from log-space initial_params.
    /// @details Validates tolerance_ >= 0 and 0 < initial_radius_ <= max_radius_, then delegates to
    /// detail::run_with_restarts() running run_trust_region_single().
    /// @param obj TwiceDifferentiable objective providing value, analytical gradient, and Hessian.
    /// @param initial_params Starting log-space parameter vector strictly inside objective bounds.
    /// @return Best OptimizerResult across initial run and restarts, or GprError on invalid config/input.
    /// @retval GprErrorCode::InvalidConfig When tolerance or radius configuration is invalid.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
    /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
    /// @retval GprErrorCode::NonFiniteInput When initial_params or initial value/gradient/Hessian is non-finite.
    /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
    std::expected<OptimizerResult, GprError> TrustRegion::optimize(
        TwiceDifferentiable& obj, std::span<const double> initial_params
    ) const
    {
        if (!std::isfinite(tolerance_) || tolerance_ < 0.0)
        {
            return std::unexpected(GprError::invalid_config("tolerance must be finite and non-negative"));
        }
        if (!std::isfinite(initial_radius_) || initial_radius_ <= 0.0)
        {
            return std::unexpected(GprError::invalid_config("initial_radius must be finite and positive"));
        }
        if (!std::isfinite(max_radius_) || max_radius_ < initial_radius_)
        {
            return std::unexpected(GprError::invalid_config("max_radius must be finite and >= initial_radius"));
        }
        return detail::run_with_restarts(
            obj, initial_params, restarts_, [&](const LogBounds& bounds, std::span<const double> start_theta) {
            return run_trust_region_single(
                obj, bounds, start_theta, max_iterations_, tolerance_, initial_radius_, max_radius_
            );
        }
        );
    }

}  // namespace cxxgpr::optimizer
