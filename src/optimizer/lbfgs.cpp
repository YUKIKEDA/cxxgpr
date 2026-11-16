// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/optimizer/lbfgs.hpp"

#include "cxxgpr/optimizer/bounds.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace cxxgpr::optimizer {
namespace {

struct LineSearchState {
    double stx{0.0};
    double fx{0.0};
    double dx{0.0};
    double sty{0.0};
    double fy{0.0};
    double dy{0.0};
    bool bracketed{false};
};

// Moré–Thuente (1994) safeguarded cubic/quadratic step update.
double cstep(LineSearchState& st, double stp, double fp, double dp, double stpmin, double stpmax) {
    const double sgnd = dp * (st.dx / std::abs(st.dx));
    double stpf = stp;

    if (fp > st.fx) {
        // Case 1: Higher function value; minimum is bracketed.
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
        if (std::abs(stpc - st.stx) < std::abs(stpq - st.stx)) {
            stpf = stpc;
        } else {
            stpf = stpc + ((stpq - stpc) / 2.0);
        }
        st.bracketed = true;
    } else if (sgnd < 0.0) {
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
        if (std::abs(stpc - stp) > std::abs(stpq - stp)) {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        st.bracketed = true;
    } else if (std::abs(dp) < std::abs(st.dx)) {
        // Case 3: Lower function value, same-sign derivatives, magnitude decreases.
        const double theta = (3.0 * (st.fx - fp) / (stp - st.stx)) + st.dx + dp;
        const double s = std::max({std::abs(theta), std::abs(st.dx), std::abs(dp)});
        const double disc = std::max(0.0, ((theta / s) * (theta / s)) - ((st.dx / s) * (dp / s)));
        double gamma = s * std::sqrt(disc);
        if (stp > st.stx) {
            gamma = -gamma;
        }
        const double p = (gamma - dp) + theta;
        const double q = (gamma + (st.dx - dp)) + gamma;
        const double r = p / q;
        double stpc = 0.0;
        if (r < 0.0 && gamma != 0.0) {
            stpc = stp + (r * (st.stx - stp));
        } else if (stp > st.stx) {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        const double stpq = stp + ((dp / (dp - st.dx)) * (st.stx - stp));
        if (st.bracketed) {
            stpf = (std::abs(stp - stpc) < std::abs(stp - stpq)) ? stpc : stpq;
        } else {
            stpf = (std::abs(stp - stpc) > std::abs(stp - stpq)) ? stpc : stpq;
        }
    } else {
        // Case 4: Lower function value, same-sign derivatives, magnitude does not decrease.
        if (st.bracketed) {
            const double theta = (3.0 * (fp - st.fy) / (st.sty - stp)) + st.dy + dp;
            const double s = std::max({std::abs(theta), std::abs(st.dy), std::abs(dp)});
            const double gamma =
                s *
                std::sqrt(std::max(0.0, ((theta / s) * (theta / s)) - ((st.dy / s) * (dp / s))));
            const double g_signed = (stp > st.sty) ? -gamma : gamma;
            const double p = (g_signed - dp) + theta;
            const double q = ((g_signed - dp) + g_signed) + st.dy;
            const double r = p / q;
            stpf = stp + (r * (st.sty - stp));
        } else if (stp > st.stx) {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }

    if (fp > st.fx) {
        st.sty = stp;
        st.fy = fp;
        st.dy = dp;
    } else {
        if (sgnd < 0.0) {
            st.sty = st.stx;
            st.fy = st.fx;
            st.dy = st.dx;
        }
        st.stx = stp;
        st.fx = fp;
        st.dx = dp;
    }

    stpf = std::clamp(stpf, stpmin, stpmax);
    if (st.bracketed) {
        const double lo = std::min(st.stx, st.sty);
        const double hi = std::max(st.stx, st.sty);
        if (stpf <= lo || stpf >= hi) {
            stpf = st.stx + (0.5 * (st.sty - st.stx));
        }
    }
    return stpf;
}

struct LbfgsPair {
    std::vector<double> s;
    std::vector<double> y;
    double rho{0.0};
};

void compute_two_loop_direction(std::span<const double> grad, const std::deque<LbfgsPair>& history,
                                std::span<double> out_dir) {
    const std::size_t d = grad.size();
    for (std::size_t i = 0; i < d; ++i) {
        out_dir[i] = -grad[i];
    }
    if (history.empty()) {
        return;
    }

    const std::size_t m = history.size();
    std::vector<double> alpha(m, 0.0);
    for (std::size_t idx = m; idx-- > 0;) {
        const double a = history[idx].rho * detail::dot_vec(history[idx].s, out_dir);
        alpha[idx] = a;
        for (std::size_t i = 0; i < d; ++i) {
            out_dir[i] -= a * history[idx].y[i];
        }
    }

    const auto& newest = history.back();
    const double yy = detail::dot_vec(newest.y, newest.y);
    const double sy = detail::dot_vec(newest.s, newest.y);
    const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
    for (std::size_t i = 0; i < d; ++i) {
        out_dir[i] *= gamma;
    }

    for (std::size_t idx = 0; idx < m; ++idx) {
        const double beta = history[idx].rho * detail::dot_vec(history[idx].y, out_dir);
        const double coeff = alpha[idx] - beta;
        for (std::size_t i = 0; i < d; ++i) {
            out_dir[i] += coeff * history[idx].s[i];
        }
    }
}

[[nodiscard]] std::expected<OptimizerResult, GprError>
run_lbfgs_single(Differentiable& obj, const LogBounds& bounds,
                 std::span<const double> initial_theta, std::size_t max_iterations,
                 double tolerance, std::size_t history_size) {
    const std::size_t d = bounds.dim();
    std::vector<double> z(d);
    if (auto u = bounds.to_unconstrained(initial_theta, z); !u.has_value()) {
        return std::unexpected(std::move(u).error());
    }

    std::vector<double> theta_buf(d);
    std::vector<double> d1_buf(d);
    std::vector<double> grad_theta_buf(d);
    std::vector<double> grad_z(d);
    EvalCache cache;

    auto f_res =
        cache.eval_unconstrained(obj, bounds, z, theta_buf, d1_buf, grad_theta_buf, grad_z);
    if (!f_res.has_value()) {
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

    constexpr double k_c1 = 1e-4;
    constexpr double k_c2 = 0.9;
    constexpr double k_xtol = 1e-15;
    constexpr double k_stp_min = 1e-20;
    constexpr double k_stp_max = 1e20;
    constexpr std::size_t k_max_ls_fev = 20U;

    std::size_t iter = 0;
    bool converged = false;

    for (; iter < max_iterations; ++iter) {
        const double g_norm = detail::norm2_vec(grad_z);
        const double z_scale = std::max(1.0, detail::norm2_vec(z));
        if (g_norm <= tolerance * z_scale) {
            converged = true;
            break;
        }

        compute_two_loop_direction(grad_z, history, dir);
        double dg_init = detail::dot_vec(grad_z, dir);
        if (!std::isfinite(dg_init) || dg_init >= 0.0) {
            history.clear();
            for (std::size_t i = 0; i < d; ++i) {
                dir[i] = -grad_z[i];
            }
            dg_init = detail::dot_vec(grad_z, dir);
            if (dg_init >= 0.0) {
                converged = true;
                break;
            }
        }

        const double f_init = f_cur;
        const double dg_test = k_c1 * dg_init;
        double stp = 1.0;
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

        for (std::size_t fev = 0; fev < k_max_ls_fev; ++fev) {
            const double stpmin_cur =
                ls_state.bracketed ? std::min(ls_state.stx, ls_state.sty) : k_stp_min;
            const double stpmax_cur = ls_state.bracketed ? std::max(ls_state.stx, ls_state.sty)
                                                         : (stp + (4.0 * (stp - ls_state.stx)));
            stp = std::clamp(stp, k_stp_min, k_stp_max);

            for (std::size_t i = 0; i < d; ++i) {
                z_trial[i] = z[i] + (stp * dir[i]);
            }

            auto trial_res = cache.eval_unconstrained(obj, bounds, z_trial, theta_buf, d1_buf,
                                                      grad_theta_buf, grad_trial);
            if (!trial_res.has_value()) {
                stp *= 0.5;
                if (stp < k_stp_min) {
                    break;
                }
                continue;
            }

            f_next = *trial_res;
            if (f_next < best_val) {
                best_val = f_next;
                best_theta.assign(theta_buf.begin(), theta_buf.end());
            }

            const double dg = detail::dot_vec(grad_trial, dir);
            const double f_test1 = f_init + (stp * dg_test);

            if (f_next <= f_test1 && std::abs(dg) <= -k_c2 * dg_init) {
                step_accepted = true;
                break;
            }

            if (ls_state.bracketed && (stp <= stpmin_cur || stp >= stpmax_cur ||
                                       (stpmax_cur - stpmin_cur) <= k_xtol * stpmax_cur)) {
                if (f_next <= f_init) {
                    step_accepted = true;
                }
                break;
            }

            if (stage1 && f_next <= f_test1 && dg >= std::min(k_c1, k_c2) * dg_init) {
                stage1 = false;
            }

            if (stage1 && f_next <= ls_state.fx && f_next > f_test1) {
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
            } else {
                stp = cstep(ls_state, stp, f_next, dg, stpmin_cur, std::min(k_stp_max, stpmax_cur));
            }

            if (ls_state.bracketed) {
                const double new_width = std::abs(ls_state.sty - ls_state.stx);
                if (new_width >= 0.66 * width1) {
                    stp = ls_state.stx + (0.5 * (ls_state.sty - ls_state.stx));
                }
                width1 = width;
                width = new_width;
            }
        }

        if (!step_accepted) {
            if (ls_state.fx < f_cur) {
                for (std::size_t i = 0; i < d; ++i) {
                    z_trial[i] = z[i] + (ls_state.stx * dir[i]);
                }
                auto fallback = cache.eval_unconstrained(obj, bounds, z_trial, theta_buf, d1_buf,
                                                         grad_theta_buf, grad_trial);
                if (fallback.has_value()) {
                    f_next = *fallback;
                    step_accepted = true;
                }
            }
            if (!step_accepted) {
                break;
            }
        }

        for (std::size_t i = 0; i < d; ++i) {
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

        if (f_cur <= best_val) {
            best_val = f_cur;
            best_theta.assign(theta_buf.begin(), theta_buf.end());
        }

        if (history_size > 0U && sy > 1e-12 * s_norm * y_norm) {
            if (history.size() == history_size) {
                history.pop_front();
            }
            history.push_back(LbfgsPair{
                .s = s_vec,
                .y = y_vec,
                .rho = 1.0 / sy,
            });
        }

        const double rel_f =
            std::abs(f_prev - f_cur) / std::max({1.0, std::abs(f_prev), std::abs(f_cur)});
        if (rel_f <= tolerance * tolerance &&
            detail::norm2_vec(grad_z) <= tolerance * std::max(1.0, detail::norm2_vec(z))) {
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

} // namespace

std::expected<OptimizerResult, GprError>
Lbfgs::optimize(Differentiable& obj, std::span<const double> initial_params) const {
    if (!std::isfinite(tolerance_) || tolerance_ < 0.0) {
        return std::unexpected(
            GprError::invalid_config("tolerance must be finite and non-negative"));
    }
    if (history_size_ == 0U) {
        return std::unexpected(GprError::invalid_config("history_size must be at least 1"));
    }
    return detail::run_with_restarts(
        obj, initial_params, restarts_,
        [&](const LogBounds& bounds, std::span<const double> start_theta) {
            return run_lbfgs_single(obj, bounds, start_theta, max_iterations_, tolerance_,
                                    history_size_);
        });
}

} // namespace cxxgpr::optimizer
