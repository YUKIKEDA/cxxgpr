// SPDX-License-Identifier: Apache-2.0
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

        // Positive root tau >= 0 of ||p + tau * d||_2^2 = radius^2.
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
            const double disc = std::max(0.0, (pd * pd) + (dd * ((radius * radius) - pp)));
            return (-pd + std::sqrt(disc)) / dd;
        }

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

            double rr = detail::dot_vec(r, r);
            const double cg_tol = std::min(0.5, std::sqrt(g_norm)) * g_norm;
            const std::size_t max_cg_iters = std::max<std::size_t>(2U * d, 10U);

            for (std::size_t k = 0; k < max_cg_iters; ++k)
            {
                sym_matvec(hess, dir, h_dir);
                const double d_hd = detail::dot_vec(dir, h_dir);

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

                if (detail::norm2_vec(p_next) >= radius)
                {
                    const double tau = step_to_boundary(out_step, dir, radius);
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        out_step[i] += tau * dir[i];
                    }
                    return;
                }

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
            LogBounds::chain_gradient(g_theta_buf, d1_buf, out_g_z);
            LogBounds::chain_hessian(h_theta_buf, g_theta_buf, d1_buf, d2_buf, out_h_z);
            return *val;
        }

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

                if (!trial_val.has_value() || !std::isfinite(*trial_val) || pred_red <= 0.0)
                {
                    radius *= 0.25;
                    continue;
                }

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
