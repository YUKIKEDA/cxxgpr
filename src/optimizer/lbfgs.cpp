// SPDX-License-Identifier: Apache-2.0
/// @file lbfgs.cpp
/// @brief Implementation of L-BFGS optimization with Nocedal two-loop recursion and Moré–Thuente line search.
/// @details Implements unconstrained L-BFGS over midpoint unit-Jacobian logit coordinates using a
/// safeguarded cubic/quadratic strong-Wolfe Moré–Thuente line search and deterministic multi-start restarts.

#include "cxxgpr/optimizer/lbfgs.hpp"

#include "cxxgpr/optimizer/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include "common.hpp"

namespace cxxgpr::optimizer
{
    namespace
    {

        /// @brief Mutable bracket state maintained across Moré–Thuente 1D line-search step updates.
        /// @details Tracks the current best step stx, opposite-endpoint bracket sty, their directional
        /// values/derivatives, and whether a minimum has been bracketed between stx and sty.
        struct LineSearchState
        {
            /// @brief Current best step length along the search direction.
            double stx{0.0};
            /// @brief Objective value at stx.
            double fx{0.0};
            /// @brief Directional derivative at stx.
            double dx{0.0};
            /// @brief Opposite bracket endpoint step length (valid when bracketed is true).
            double sty{0.0};
            /// @brief Objective value at sty.
            double fy{0.0};
            /// @brief Directional derivative at sty.
            double dy{0.0};
            /// @brief True once stx and sty bracket a local minimizer.
            bool bracketed{false};
        };

        /// @brief Computes a safeguarded cubic/quadratic Moré–Thuente (1994) trial step and updates bracket state.
        /// @details Handles all four Moré–Thuente interpolation cases (Case 1: fp > fx, Case 2: opposite-sign
        /// derivatives, Case 3: same-sign decreasing slope magnitude, Case 4: non-decreasing slope magnitude),
        /// updates stx/sty endpoints, clamps to [stpmin, stpmax], and bisects if bracket bounds are violated.
        /// @param[in,out] st LineSearchState updated in place with the new bracket endpoints.
        /// @param[in] stp Current trial step length.
        /// @param[in] fp Objective value (or modified Stage 1 value) at stp.
        /// @param[in] dp Directional derivative (or modified Stage 1 derivative) at stp.
        /// @param[in] stpmin Inclusive lower safeguard bound for the next trial step.
        /// @param[in] stpmax Inclusive upper safeguard bound for the next trial step.
        /// @return Safeguarded next trial step length along the search direction.
        double cstep(LineSearchState& st, double stp, double fp, double dp, double stpmin, double stpmax)
        {
            const double sgnd = dp * (st.dx / std::abs(st.dx));
            double stpf = stp;

            if (fp > st.fx)
            {
                // Case 1: Higher function value; minimum is bracketed between stx and stp.
                const double theta = (3.0 * (st.fx - fp) / (stp - st.stx)) + st.dx + dp;
                const double s = std::max({std::abs(theta), std::abs(st.dx), std::abs(dp)});
                const double gamma =
                    s * std::sqrt(std::max(0.0, ((theta / s) * (theta / s)) - ((st.dx / s) * (dp / s))));
                const double g_signed = (stp < st.stx) ? -gamma : gamma;
                const double p = (g_signed - st.dx) + theta;
                const double q = ((g_signed - st.dx) + g_signed) + dp;
                const double r = p / q;
                const double stpc = st.stx + (r * (stp - st.stx));
                const double stpq =
                    st.stx + (((st.dx / (((st.fx - fp) / (stp - st.stx)) + st.dx)) / 2.0) * (stp - st.stx));
                if (std::abs(stpc - st.stx) < std::abs(stpq - st.stx))
                {
                    stpf = stpc;
                }
                else
                {
                    stpf = stpc + ((stpq - stpc) / 2.0);
                }
                st.bracketed = true;
            }
            else if (sgnd < 0.0)
            {
                // Case 2: Lower function value and opposite-sign derivatives; minimum is bracketed.
                const double theta = (3.0 * (st.fx - fp) / (stp - st.stx)) + st.dx + dp;
                const double s = std::max({std::abs(theta), std::abs(st.dx), std::abs(dp)});
                const double gamma =
                    s * std::sqrt(std::max(0.0, ((theta / s) * (theta / s)) - ((st.dx / s) * (dp / s))));
                const double g_signed = (stp > st.stx) ? -gamma : gamma;
                const double p = (g_signed - dp) + theta;
                const double q = ((g_signed - dp) + g_signed) + st.dx;
                const double r = p / q;
                const double stpc = stp + (r * (st.stx - stp));
                const double stpq = stp + ((dp / (dp - st.dx)) * (st.stx - stp));
                if (std::abs(stpc - stp) > std::abs(stpq - stp))
                {
                    stpf = stpc;
                }
                else
                {
                    stpf = stpq;
                }
                st.bracketed = true;
            }
            else if (std::abs(dp) < std::abs(st.dx))
            {
                // Case 3: Lower function value, same-sign derivatives, magnitude decreases.
                const double theta = (3.0 * (st.fx - fp) / (stp - st.stx)) + st.dx + dp;
                const double s = std::max({std::abs(theta), std::abs(st.dx), std::abs(dp)});
                const double disc = std::max(0.0, ((theta / s) * (theta / s)) - ((st.dx / s) * (dp / s)));
                double gamma = s * std::sqrt(disc);
                if (stp > st.stx)
                {
                    gamma = -gamma;
                }
                const double p = (gamma - dp) + theta;
                const double q = (gamma + (st.dx - dp)) + gamma;
                const double r = p / q;
                double stpc = 0.0;
                if (r < 0.0 && gamma != 0.0)
                {
                    stpc = stp + (r * (st.stx - stp));
                }
                else if (stp > st.stx)
                {
                    stpc = stpmax;
                }
                else
                {
                    stpc = stpmin;
                }
                const double stpq = stp + ((dp / (dp - st.dx)) * (st.stx - stp));
                if (st.bracketed)
                {
                    stpf = (std::abs(stp - stpc) < std::abs(stp - stpq)) ? stpc : stpq;
                }
                else
                {
                    stpf = (std::abs(stp - stpc) > std::abs(stp - stpq)) ? stpc : stpq;
                }
            }
            else
            {
                // Case 4: Lower function value, same-sign derivatives, magnitude does not decrease.
                if (st.bracketed)
                {
                    const double theta = (3.0 * (fp - st.fy) / (st.sty - stp)) + st.dy + dp;
                    const double s = std::max({std::abs(theta), std::abs(st.dy), std::abs(dp)});
                    const double gamma =
                        s * std::sqrt(std::max(0.0, ((theta / s) * (theta / s)) - ((st.dy / s) * (dp / s))));
                    const double g_signed = (stp > st.sty) ? -gamma : gamma;
                    const double p = (g_signed - dp) + theta;
                    const double q = ((g_signed - dp) + g_signed) + st.dy;
                    const double r = p / q;
                    stpf = stp + (r * (st.sty - stp));
                }
                else if (stp > st.stx)
                {
                    stpf = stpmax;
                }
                else
                {
                    stpf = stpmin;
                }
            }

            // Update bracket endpoints stx (best step) and sty (opposite side of bracket).
            if (fp > st.fx)
            {
                st.sty = stp;
                st.fy = fp;
                st.dy = dp;
            }
            else
            {
                if (sgnd < 0.0)
                {
                    st.sty = st.stx;
                    st.fy = st.fx;
                    st.dy = st.dx;
                }
                st.stx = stp;
                st.fx = fp;
                st.dx = dp;
            }

            // Clamp step within [stpmin, stpmax] and bisect if an interpolated bracket step hits an endpoint.
            stpf = std::clamp(stpf, stpmin, stpmax);
            if (st.bracketed)
            {
                const double lo = std::min(st.stx, st.sty);
                const double hi = std::max(st.stx, st.sty);
                if (stpf <= lo || stpf >= hi)
                {
                    stpf = st.stx + (0.5 * (st.sty - st.stx));
                }
            }
            return stpf;
        }

        /// @brief Secant correction pair (s, y) and reciprocal curvature rho = 1 / (s^T * y) for L-BFGS.
        /// @details Stored in chronological order inside a bounded deque of size history_size_.
        struct LbfgsPair
        {
            /// @brief Step displacement vector s_k = z_{k+1} - z_k in unconstrained logit space.
            std::vector<double> s;
            /// @brief Gradient difference vector y_k = g_{k+1} - g_k in unconstrained logit space.
            std::vector<double> y;
            /// @brief Reciprocal curvature weight rho_k = 1 / (s_k^T * y_k).
            double rho{0.0};
        };

        /// @brief Computes the L-BFGS search direction d = -H_k * g using Nocedal two-loop recursion.
        /// @details Runs the backward loop from newest to oldest pair to compute alpha_i and update q,
        /// applies Oren-Luenberger diagonal scaling gamma = (s^T * y) / (y^T * y) from the newest pair,
        /// and runs the forward loop to reconstruct -H_k * g into out_dir.
        /// @param[in] grad Current unconstrained gradient g_z of length d.
        /// @param[in] history Bounded deque of recent (s, y, rho) curvature pairs (oldest to newest).
        /// @param[out] out_dir Destination span of length d receiving the descent search direction.
        void compute_two_loop_direction(
            std::span<const double> grad, const std::deque<LbfgsPair>& history, std::span<double> out_dir
        )
        {
            const std::size_t d = grad.size();
            for (std::size_t i = 0; i < d; ++i)
            {
                out_dir[i] = -grad[i];
            }
            if (history.empty())
            {
                return;
            }

            // Backward pass: accumulate multipliers alpha_i and subtract alpha_i * y_i from q.
            const std::size_t m = history.size();
            std::vector<double> alpha(m, 0.0);
            for (std::size_t idx = m; idx-- > 0;)
            {
                const double a = history[idx].rho * detail::dot_vec(history[idx].s, out_dir);
                alpha[idx] = a;
                for (std::size_t i = 0; i < d; ++i)
                {
                    out_dir[i] -= a * history[idx].y[i];
                }
            }

            // Apply initial Hessian approximation H_0 = gamma * I using the latest curvature pair.
            const auto& newest = history.back();
            const double yy = detail::dot_vec(newest.y, newest.y);
            const double sy = detail::dot_vec(newest.s, newest.y);
            const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
            for (std::size_t i = 0; i < d; ++i)
            {
                out_dir[i] *= gamma;
            }

            // Forward pass: add (alpha_i - beta_i) * s_i to form the final quasi-Newton direction.
            for (std::size_t idx = 0; idx < m; ++idx)
            {
                const double beta = history[idx].rho * detail::dot_vec(history[idx].y, out_dir);
                const double coeff = alpha[idx] - beta;
                for (std::size_t i = 0; i < d; ++i)
                {
                    out_dir[i] += coeff * history[idx].s[i];
                }
            }
        }

        /// @brief Runs a single-start L-BFGS trajectory from initial_theta in unconstrained logit space.
        /// @details Maps initial_theta into z, iteratively evaluates two-loop descent directions, performs
        /// Moré–Thuente strong-Wolfe line search with EvalCache reuse and Stage 1 modified function
        /// shifting, appends secant pairs satisfying curvature safety, and stops on tolerance or iteration cap. The
        /// first trial step without curvature history is scaled to at most one unit in z; a line-search failure counts
        /// as converged only at the function-precision gradient floor sqrt(tolerance).
        /// @param[in,out] obj Differentiable objective evaluated in log-theta space.
        /// @param[in] bounds Log-space bounds governing the z <-> theta transform.
        /// @param[in] initial_theta Starting log-space parameter vector strictly inside bounds.
        /// @param[in] max_iterations Maximum outer L-BFGS iterations.
        /// @param[in] tolerance Relative gradient and function-change convergence tolerance.
        /// @param[in] history_size Maximum number of (s, y) secant pairs retained.
        /// @return OptimizerResult with best feasible log-space parameters found, or GprError if initial eval fails.
        /// @retval GprErrorCode::NonFiniteInput When initial objective or gradient evaluation yields non-finite values.
        [[nodiscard]] std::expected<OptimizerResult, GprError> run_lbfgs_single(
            Differentiable& obj,
            const LogBounds& bounds,
            std::span<const double> initial_theta,
            std::size_t max_iterations,
            double tolerance,
            std::size_t history_size
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
            std::vector<double> grad_theta_buf(d);
            std::vector<double> grad_z(d);
            EvalCache cache;

            auto f_res = cache.eval_unconstrained(obj, bounds, z, theta_buf, d1_buf, grad_theta_buf, grad_z);
            if (!f_res.has_value())
            {
                return std::unexpected(std::move(f_res).error());
            }
            double f_cur = *f_res;

            std::vector<double> best_theta(initial_theta.begin(), initial_theta.end());
            double best_val = f_cur;

            std::deque<LbfgsPair> history;
            std::vector<double> dir(d);
            std::vector<double> z_trial(d);
            std::vector<double> grad_trial(d);
            std::vector<double> s_vec(d);
            std::vector<double> y_vec(d);

            // Moré–Thuente strong-Wolfe parameters (c1 = 1e-4 sufficient decrease, c2 = 0.9 curvature).
            constexpr double k_c1 = 1e-4;
            constexpr double k_c2 = 0.9;
            constexpr double k_xtol = 1e-15;
            constexpr double k_stp_min = 1e-20;
            constexpr double k_stp_max = 1e20;
            constexpr std::size_t k_max_ls_fev = 20U;

            std::size_t iter = 0;
            bool converged = false;

            for (; iter < max_iterations; ++iter)
            {
                const double g_norm = detail::norm2_vec(grad_z);
                const double z_scale = std::max(1.0, detail::norm2_vec(z));
                if (g_norm <= tolerance * z_scale)
                {
                    converged = true;
                    break;
                }

                // Compute quasi-Newton direction; reset L-BFGS memory to steepest descent if not a descent direction.
                compute_two_loop_direction(grad_z, history, dir);
                double dg_init = detail::dot_vec(grad_z, dir);
                if (!std::isfinite(dg_init) || dg_init >= 0.0)
                {
                    history.clear();
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        dir[i] = -grad_z[i];
                    }
                    dg_init = detail::dot_vec(grad_z, dir);
                    if (dg_init >= 0.0)
                    {
                        converged = true;
                        break;
                    }
                }

                const double f_init = f_cur;
                const double dg_test = k_c1 * dg_init;
                // Without curvature information the direction is the raw negative gradient, whose length is arbitrary.
                // Scale the first trial step so it moves at most one unit in z (Nocedal & Wright, eq. 7.20); a unit
                // step along a large gradient otherwise jumps to the saturated, flat region of the bounds.
                double stp = history.empty() ? std::min(1.0, 1.0 / detail::norm2_vec(dir)) : 1.0;
                double width = k_stp_max - k_stp_min;
                double width1 = 2.0 * width;
                bool stage1 = true;

                LineSearchState ls_state{
                    .stx = 0.0,
                    .fx = f_init,
                    .dx = dg_init,
                    .sty = 0.0,
                    .fy = f_init,
                    .dy = dg_init,
                    .bracketed = false,
                };

                bool step_accepted = false;
                double f_next = f_init;

                // Execute Moré–Thuente strong-Wolfe line search along dir starting from unit step stp = 1.
                for (std::size_t fev = 0; fev < k_max_ls_fev; ++fev)
                {
                    const double stpmin_cur = ls_state.bracketed ? std::min(ls_state.stx, ls_state.sty) : k_stp_min;
                    const double stpmax_cur = ls_state.bracketed ? std::max(ls_state.stx, ls_state.sty)
                                                                 : (stp + (4.0 * (stp - ls_state.stx)));
                    stp = std::clamp(stp, k_stp_min, k_stp_max);

                    for (std::size_t i = 0; i < d; ++i)
                    {
                        z_trial[i] = z[i] + (stp * dir[i]);
                    }

                    auto trial_res =
                        cache.eval_unconstrained(obj, bounds, z_trial, theta_buf, d1_buf, grad_theta_buf, grad_trial);
                    if (!trial_res.has_value())
                    {
                        stp *= 0.5;
                        if (stp < k_stp_min)
                        {
                            break;
                        }
                        continue;
                    }

                    f_next = *trial_res;
                    if (f_next < best_val)
                    {
                        best_val = f_next;
                        best_theta.assign(theta_buf.begin(), theta_buf.end());
                    }

                    const double dg = detail::dot_vec(grad_trial, dir);
                    const double f_test1 = f_init + (stp * dg_test);

                    // Accept immediately when both strong-Wolfe sufficient decrease and curvature conditions hold.
                    if (f_next <= f_test1 && std::abs(dg) <= -k_c2 * dg_init)
                    {
                        step_accepted = true;
                        break;
                    }

                    // Exit if bracket collapsed to xtol or hit step bounds while retaining decreasing trial values.
                    if (ls_state.bracketed
                        && (stp <= stpmin_cur || stp >= stpmax_cur || (stpmax_cur - stpmin_cur) <= k_xtol * stpmax_cur))
                    {
                        if (f_next <= f_init)
                        {
                            step_accepted = true;
                        }
                        break;
                    }

                    // Transition from Moré–Thuente Stage 1 (modified psi) to Stage 2 once sufficient decrease holds.
                    if (stage1 && f_next <= f_test1 && dg >= std::min(k_c1, k_c2) * dg_init)
                    {
                        stage1 = false;
                    }

                    if (stage1 && f_next <= ls_state.fx && f_next > f_test1)
                    {
                        // In Stage 1, pass shifted auxiliary function psi(alpha) = f(alpha) - alpha * dg_test to cstep.
                        const double fm = f_next - (stp * dg_test);
                        const double fxm = ls_state.fx - (ls_state.stx * dg_test);
                        const double fym = ls_state.fy - (ls_state.sty * dg_test);
                        const double dgm = dg - dg_test;
                        const double dxm = ls_state.dx - dg_test;
                        const double dym = ls_state.dy - dg_test;

                        LineSearchState mod_st{
                            .stx = ls_state.stx,
                            .fx = fxm,
                            .dx = dxm,
                            .sty = ls_state.sty,
                            .fy = fym,
                            .dy = dym,
                            .bracketed = ls_state.bracketed,
                        };
                        stp = cstep(mod_st, stp, fm, dgm, stpmin_cur, std::min(k_stp_max, stpmax_cur));
                        ls_state.stx = mod_st.stx;
                        ls_state.fx = mod_st.fx + (ls_state.stx * dg_test);
                        ls_state.dx = mod_st.dx + dg_test;
                        ls_state.sty = mod_st.sty;
                        ls_state.fy = mod_st.fy + (ls_state.sty * dg_test);
                        ls_state.dy = mod_st.dy + dg_test;
                        ls_state.bracketed = mod_st.bracketed;
                    }
                    else
                    {
                        stp = cstep(ls_state, stp, f_next, dg, stpmin_cur, std::min(k_stp_max, stpmax_cur));
                    }

                    // Force a bisection step whenever two iterations fail to shrink bracket width by at least 0.66.
                    if (ls_state.bracketed)
                    {
                        const double new_width = std::abs(ls_state.sty - ls_state.stx);
                        if (new_width >= 0.66 * width1)
                        {
                            stp = ls_state.stx + (0.5 * (ls_state.sty - ls_state.stx));
                        }
                        width1 = width;
                        width = new_width;
                    }
                }

                // Fall back to the best bracketed point stx if strong-Wolfe termination hit k_max_ls_fev.
                if (!step_accepted)
                {
                    if (ls_state.fx < f_cur)
                    {
                        for (std::size_t i = 0; i < d; ++i)
                        {
                            z_trial[i] = z[i] + (ls_state.stx * dir[i]);
                        }
                        auto fallback = cache.eval_unconstrained(
                            obj, bounds, z_trial, theta_buf, d1_buf, grad_theta_buf, grad_trial
                        );
                        if (fallback.has_value())
                        {
                            f_next = *fallback;
                            step_accepted = true;
                        }
                    }
                    if (!step_accepted)
                    {
                        // No acceptable step exists. Near a minimum the objective is only known to about
                        // sqrt(eps) relative accuracy, so a line search must fail once the gradient has been reduced
                        // to that floor; that is convergence, not a failure. A larger gradient means a real failure.
                        const double floor_tolerance = std::sqrt(tolerance);
                        converged = detail::norm2_vec(grad_z) <= floor_tolerance * std::max(1.0, detail::norm2_vec(z));
                        break;
                    }
                }

                for (std::size_t i = 0; i < d; ++i)
                {
                    s_vec[i] = z_trial[i] - z[i];
                    y_vec[i] = grad_trial[i] - grad_z[i];
                }
                const double sy = detail::dot_vec(s_vec, y_vec);
                const double s_norm = detail::norm2_vec(s_vec);
                const double y_norm = detail::norm2_vec(y_vec);

                z = z_trial;
                grad_z = grad_trial;
                const double f_prev = f_cur;
                f_cur = f_next;

                if (f_cur <= best_val)
                {
                    best_val = f_cur;
                    best_theta.assign(theta_buf.begin(), theta_buf.end());
                }

                // Retain secant pair only when curvature condition s^T y > 1e-12 * ||s|| * ||y|| holds.
                if (history_size > 0U && sy > 1e-12 * s_norm * y_norm)
                {
                    if (history.size() == history_size)
                    {
                        history.pop_front();
                    }
                    history.push_back(
                        LbfgsPair{
                            .s = s_vec,
                            .y = y_vec,
                            .rho = 1.0 / sy,
                        }
                    );
                }

                const double rel_f = std::abs(f_prev - f_cur) / std::max({1.0, std::abs(f_prev), std::abs(f_cur)});
                if (rel_f <= tolerance * tolerance
                    && detail::norm2_vec(grad_z) <= tolerance * std::max(1.0, detail::norm2_vec(z)))
                {
                    converged = true;
                    ++iter;
                    break;
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
    /// @details Validates tolerance_ >= 0 and history_size_ >= 1, then runs run_lbfgs_single() across
    /// initial_params and deterministic log-uniform restarts via detail::run_with_restarts().
    /// @param[in,out] obj Differentiable objective evaluated in log-theta space.
    /// @param[in] initial_params Starting log-space parameter vector strictly inside objective bounds.
    /// @return Best OptimizerResult across initial run and restarts, or GprError on invalid config/input.
    /// @retval GprErrorCode::InvalidConfig When tolerance is invalid/negative or history_size == 0.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
    /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
    /// @retval GprErrorCode::NonFiniteInput When initial_params or initial objective/gradient is non-finite.
    /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
    std::expected<OptimizerResult, GprError> Lbfgs::optimize(
        Differentiable& obj, std::span<const double> initial_params
    ) const
    {
        if (!std::isfinite(tolerance_) || tolerance_ < 0.0)
        {
            return std::unexpected(GprError::invalid_config("tolerance must be finite and non-negative"));
        }
        if (history_size_ == 0U)
        {
            return std::unexpected(GprError::invalid_config("history_size must be at least 1"));
        }
        return detail::run_with_restarts(
            obj, initial_params, restarts_, [&](const LogBounds& bounds, std::span<const double> start_theta)
        { return run_lbfgs_single(obj, bounds, start_theta, max_iterations_, tolerance_, history_size_); }
        );
    }

}  // namespace cxxgpr::optimizer
