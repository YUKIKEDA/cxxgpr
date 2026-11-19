// SPDX-License-Identifier: Apache-2.0
/// @file nelder_mead.cpp
/// @brief Implementation of Nelder-Mead simplex optimization over logit-transformed parameter bounds.
/// @details Implements derivative-free simplex initialization, reflection, expansion, outside/inside
/// contraction, and shrink steps in unconstrained logit coordinates with multi-start restarts.

#include "cxxgpr/optimizer/nelder_mead.hpp"

#include "cxxgpr/optimizer/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "common.hpp"

namespace cxxgpr::optimizer
{
    namespace
    {

        /// @brief Evaluates obj in constrained log-theta space from unconstrained logit coordinates z.
        /// @details Guards finiteness of z, maps z to theta_buf via bounds.to_constrained(), calls
        /// obj.value(theta_buf), and rejects non-finite scalar outputs.
        /// @param[in,out] obj Objective evaluated in log-theta space.
        /// @param[in] bounds Log-space bounds governing the z -> theta transform.
        /// @param[in] z Unconstrained logit coordinate vector of length bounds.dim().
        /// @param[in,out] theta_buf Caller-supplied scratch span of length bounds.dim().
        /// @return Finite objective value at z, or GprError if evaluation fails or returns non-finite value.
        /// @retval GprErrorCode::NonFiniteInput When z or the evaluated objective value is non-finite.
        [[nodiscard]] std::expected<double, GprError> eval_logit(
            Objective& obj, const LogBounds& bounds, std::span<const double> z, std::span<double> theta_buf
        )
        {
            // Reject non-finite unconstrained coordinates before transforming into open log-theta bounds.
            for (const double val : z)
            {
                if (!std::isfinite(val))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            bounds.to_constrained(z, theta_buf);
            auto res = obj.value(theta_buf);
            if (!res.has_value())
            {
                return res;
            }
            if (!std::isfinite(*res))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            return res;
        }

        /// @brief Runs one single-start Nelder-Mead simplex optimization from initial_theta.
        /// @details Converts initial_theta to logit z0, builds d + 1 simplex vertices via axis-aligned
        /// perturbations, and iteratively updates the worst vertex via reflection (alpha = 1), expansion
        /// (gamma = 2), outside/inside contraction (rho = 0.5), or full simplex shrink (sigma = 0.5)
        /// until both objective spread and simplex diameter satisfy tolerance or max_iterations is reached.
        /// @param[in,out] obj Objective evaluated in log-theta space.
        /// @param[in] bounds Log-space bounds governing the z <-> theta transform.
        /// @param[in] initial_theta Starting log-space parameter vector strictly inside bounds.
        /// @param[in] max_iterations Maximum number of simplex iterations.
        /// @param[in] tolerance Convergence tolerance on simplex value spread and vertex infinity-norm distance.
        /// @return OptimizerResult holding best vertex mapped back to log-theta space, or GprError on failure.
        /// @retval GprErrorCode::NonFiniteInput When any simplex evaluation produces non-finite values.
        [[nodiscard]] std::expected<OptimizerResult, GprError> run_nelder_mead_single(
            Objective& obj,
            const LogBounds& bounds,
            std::span<const double> initial_theta,
            std::size_t max_iterations,
            double tolerance
        )
        {
            const std::size_t d = bounds.dim();
            std::vector<double> z0(d);
            if (auto u = bounds.to_unconstrained(initial_theta, z0); !u.has_value())
            {
                return std::unexpected(std::move(u).error());
            }

            // Initialize d + 1 simplex vertices around z0 using relative 5% steps (or 0.25 near 0).
            const std::size_t n_verts = d + 1U;
            std::vector<std::vector<double>> verts(n_verts, z0);
            for (std::size_t k = 1; k < n_verts; ++k)
            {
                const std::size_t coord = k - 1U;
                const double step = (std::abs(z0[coord]) > 1e-6) ? (0.05 * z0[coord]) : 0.25;
                verts[k][coord] += step;
            }

            std::vector<double> theta_buf(d);
            std::vector<double> vals(n_verts, 0.0);
            for (std::size_t k = 0; k < n_verts; ++k)
            {
                auto v = eval_logit(obj, bounds, verts[k], theta_buf);
                if (!v.has_value())
                {
                    return std::unexpected(std::move(v).error());
                }
                vals[k] = *v;
            }

            std::vector<std::size_t> order(n_verts);
            std::ranges::iota(order, 0U);
            auto sort_simplex = [&]()
            {
                std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return vals[a] < vals[b]; });
            };

            // Standard Nelder-Mead simplex coefficients: reflection, expansion, contraction, shrink.
            constexpr double k_alpha = 1.0;
            constexpr double k_gamma = 2.0;
            constexpr double k_rho = 0.5;
            constexpr double k_sigma = 0.5;

            std::vector<double> centroid(d);
            std::vector<double> xr(d);
            std::vector<double> xe(d);
            std::vector<double> xc(d);

            std::size_t iter = 0;
            bool converged = false;

            for (; iter < max_iterations; ++iter)
            {
                sort_simplex();
                const std::size_t best_idx = order[0];
                const std::size_t second_worst_idx = order[n_verts - 2U];
                const std::size_t worst_idx = order[n_verts - 1U];

                // Terminate only when both objective spread and simplex diameter fall below tolerance.
                const double f_diff = vals[worst_idx] - vals[best_idx];
                double max_dist = 0.0;
                for (std::size_t k = 1; k < n_verts; ++k)
                {
                    const auto& vk = verts[order[k]];
                    const auto& v0 = verts[best_idx];
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        max_dist = std::max(max_dist, std::abs(vk[i] - v0[i]));
                    }
                }
                if (f_diff <= tolerance * std::max(1.0, std::abs(vals[best_idx])) && max_dist <= tolerance)
                {
                    converged = true;
                    break;
                }

                // Compute centroid of the best d vertices excluding the worst vertex.
                std::ranges::fill(centroid, 0.0);
                for (std::size_t k = 0; k < n_verts - 1U; ++k)
                {
                    const auto& vk = verts[order[k]];
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        centroid[i] += vk[i];
                    }
                }
                const double inv_d = 1.0 / static_cast<double>(d);
                for (std::size_t i = 0; i < d; ++i)
                {
                    centroid[i] *= inv_d;
                }

                // Reflection across the centroid away from the worst vertex.
                for (std::size_t i = 0; i < d; ++i)
                {
                    xr[i] = centroid[i] + (k_alpha * (centroid[i] - verts[worst_idx][i]));
                }
                auto fr = eval_logit(obj, bounds, xr, theta_buf);
                if (!fr.has_value())
                {
                    return std::unexpected(std::move(fr).error());
                }

                if (*fr >= vals[best_idx] && *fr < vals[second_worst_idx])
                {
                    verts[worst_idx] = xr;
                    vals[worst_idx] = *fr;
                    continue;
                }

                if (*fr < vals[best_idx])
                {
                    // Expansion when reflection achieves a strict new simplex minimum.
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        xe[i] = centroid[i] + (k_gamma * (xr[i] - centroid[i]));
                    }
                    auto fe = eval_logit(obj, bounds, xe, theta_buf);
                    if (fe.has_value() && *fe < *fr)
                    {
                        verts[worst_idx] = xe;
                        vals[worst_idx] = *fe;
                    }
                    else
                    {
                        verts[worst_idx] = xr;
                        vals[worst_idx] = *fr;
                    }
                    continue;
                }

                // Outside contraction if reflection improved on worst; otherwise inside contraction.
                if (*fr < vals[worst_idx])
                {
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        xc[i] = centroid[i] + (k_rho * (xr[i] - centroid[i]));
                    }
                    auto fc = eval_logit(obj, bounds, xc, theta_buf);
                    if (fc.has_value() && *fc <= *fr)
                    {
                        verts[worst_idx] = xc;
                        vals[worst_idx] = *fc;
                        continue;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        xc[i] = centroid[i] - (k_rho * (centroid[i] - verts[worst_idx][i]));
                    }
                    auto fc = eval_logit(obj, bounds, xc, theta_buf);
                    if (fc.has_value() && *fc < vals[worst_idx])
                    {
                        verts[worst_idx] = xc;
                        vals[worst_idx] = *fc;
                        continue;
                    }
                }

                // Shrink all non-best vertices halfway toward the best vertex if contraction failed.
                const auto& v_best = verts[best_idx];
                for (std::size_t k = 1; k < n_verts; ++k)
                {
                    const std::size_t idx = order[k];
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        verts[idx][i] = v_best[i] + (k_sigma * (verts[idx][i] - v_best[i]));
                    }
                    auto fs = eval_logit(obj, bounds, verts[idx], theta_buf);
                    if (!fs.has_value())
                    {
                        return std::unexpected(std::move(fs).error());
                    }
                    vals[idx] = *fs;
                }
            }

            sort_simplex();
            const std::size_t best_idx = order[0];
            bounds.to_constrained(verts[best_idx], theta_buf);
            return OptimizerResult{
                .params = std::move(theta_buf),
                .value = vals[best_idx],
                .iterations = iter,
                .converged = converged,
            };
        }

    }  // namespace

    /// @brief Minimizes obj starting from log-space initial_params.
    /// @details Validates tolerance_ >= 0 and delegates to detail::run_with_restarts() with
    /// run_nelder_mead_single().
    /// @param[in,out] obj Objective evaluated in log-theta space.
    /// @param[in] initial_params Starting log-space parameter vector strictly inside objective bounds.
    /// @return Best OptimizerResult across initial run and restarts, or GprError on invalid config/input.
    /// @retval GprErrorCode::InvalidConfig When tolerance is non-finite or negative.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
    /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
    /// @retval GprErrorCode::NonFiniteInput When initial_params or simplex evaluation yields non-finite values.
    /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params are out of bounds.
    std::expected<OptimizerResult, GprError> NelderMead::optimize(
        Objective& obj, std::span<const double> initial_params
    ) const
    {
        if (!std::isfinite(tolerance_) || tolerance_ < 0.0)
        {
            return std::unexpected(GprError::invalid_config("tolerance must be finite and non-negative"));
        }
        return detail::run_with_restarts(
            obj, initial_params, restarts_, [&](const LogBounds& bounds, std::span<const double> start_theta)
        { return run_nelder_mead_single(obj, bounds, start_theta, max_iterations_, tolerance_); }
        );
    }

}  // namespace cxxgpr::optimizer
