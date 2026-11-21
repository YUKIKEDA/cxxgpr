// SPDX-License-Identifier: Apache-2.0
/// @file common.hpp
/// @brief Internal vector math helpers and deterministic multi-start restart driver for optimizers.
/// @details Provides Euclidean dot product, L2 norm, and run_with_restarts() shared across
/// FastSimulatedAnnealing, Lbfgs, NelderMead, and TrustRegion implementations.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/optimizer/bounds.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <span>
#include <vector>

/// @brief Internal helper utilities shared among cxxgpr optimizer implementations.
namespace cxxgpr::optimizer::detail
{

    /// @brief Computes the Euclidean dot product of two equal-length vectors.
    /// @details Accumulates sum_i a[i] * b[i] in double precision.
    /// @param[in] a First input vector span.
    /// @param[in] b Second input vector span (must have length >= a.size()).
    /// @return Inner product a^T * b.
    [[nodiscard]] inline double dot_vec(std::span<const double> a, std::span<const double> b) noexcept
    {
        double acc = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            acc += a[i] * b[i];
        }
        return acc;
    }

    /// @brief Computes the Euclidean L2 norm ||a||_2.
    /// @details Evaluates sqrt(dot_vec(a, a)).
    /// @param[in] a Input vector span.
    /// @return Non-negative Euclidean norm of a.
    [[nodiscard]] inline double norm2_vec(std::span<const double> a) noexcept
    {
        return std::sqrt(dot_vec(a, a));
    }

    /// @brief Computes the Euclidean norm of the projected gradient of theta on the closed log-space box.
    /// @details Evaluates chi = || theta - clamp(theta - g_theta, lo_log, hi_log) ||_2, the first-order stationarity
    /// measure of the bound-constrained problem min f(theta) subject to lo_log <= theta <= hi_log. It is computed
    /// from the log-space gradient itself, never from the unconstrained gradient divided by the logit Jacobian, so it
    /// does not shrink when the logit transform saturates. Per component, an interior entry equals g_theta[i]; an entry
    /// whose descent direction points out of the box is cut to the distance to that bound, so a bound-constrained
    /// minimizer with an outward gradient has a vanishing measure while an inward gradient at a bound keeps its full
    /// size.
    /// @param[in] theta Interior log-space parameters of length bounds.dim().
    /// @param[in] grad_theta Gradient of the objective with respect to theta, of length bounds.dim().
    /// @param[in] bounds Log-space bounds whose closure is the feasible box.
    /// @return Non-negative projected-gradient norm.
    [[nodiscard]] inline double projected_gradient_norm(
        std::span<const double> theta, std::span<const double> grad_theta, const LogBounds& bounds
    ) noexcept
    {
        const std::span<const double> lo = bounds.lo_log();
        const std::span<const double> hi = bounds.hi_log();
        double acc = 0.0;
        for (std::size_t i = 0; i < theta.size(); ++i)
        {
            const double p = theta[i] - std::clamp(theta[i] - grad_theta[i], lo[i], hi[i]);
            acc += p * p;
        }
        return std::sqrt(acc);
    }

    /// @brief Returns the stationarity threshold tolerance * max(1, ||theta||_2) of the gradient stopping test.
    /// @details The scale is the norm of the log-space parameters, which the bounds keep finite; unlike the norm of
    /// the unconstrained coordinates it does not grow without limit when a parameter approaches a bound.
    /// @param[in] theta Log-space parameters.
    /// @param[in] tolerance Non-negative user tolerance.
    /// @return Threshold that the projected-gradient norm must not exceed.
    [[nodiscard]] inline double stationarity_threshold(std::span<const double> theta, double tolerance) noexcept
    {
        return tolerance * std::max(1.0, norm2_vec(theta));
    }

    /// @brief Returns the band within which two objective values are treated as indistinguishable.
    /// @details sqrt(eps) * (1 + |f|). The rounding level of a computed objective is eps * (1 + |f|), and a value
    /// obtained through a Cholesky factorization carries that error amplified by the conditioning of the matrix
    /// (about 7e2 * eps * |f| was measured for a Gaussian-process marginal likelihood); the square root of eps keeps a
    /// margin of several orders above both while still resolving eight significant digits of f. The band only lets a
    /// step search accept a step on derivative evidence when the decrease the step can produce is itself below the
    /// band; it never decides convergence.
    /// @param[in] f Objective value at the start of the step.
    /// @return Non-negative value band.
    [[nodiscard]] inline double value_resolution(double f) noexcept
    {
        return std::sqrt(std::numeric_limits<double>::epsilon()) * (1.0 + std::abs(f));
    }

    /// @brief Returns the projected-gradient level below which the objective value cannot resolve further decrease.
    /// @details sqrt(eps) * (1 + |f|), used only after a step search has failed to find any acceptable step. Along a
    /// direction with curvature h, a projected gradient chi can buy at most the decrease chi^2 / (2 h). With the
    /// rounding level eps * (1 + |f|) of the objective and a curvature of the order of the objective scale,
    /// h = 1 + |f|, that decrease stays below half the rounding level exactly when chi <= sqrt(eps) * (1 + |f|). The
    /// level depends only on machine epsilon and the scale of f, never on the user tolerance; it is conservative
    /// because the real evaluation error of f is usually larger than its rounding level.
    /// @param[in] f Objective value at the point being classified.
    /// @return Projected-gradient level at which a failed step search counts as convergence.
    [[nodiscard]] inline double precision_floor(double f) noexcept
    {
        return std::sqrt(std::numeric_limits<double>::epsilon()) * (1.0 + std::abs(f));
    }

    /// @brief Executes a single-start solver at initial_params plus deterministic log-uniform restarts.
    /// @details Builds LogBounds from obj, validates initial_params against open bounds, runs
    /// run_single(*bounds, initial_params), and if restarts.count > 0 samples restarts.count additional
    /// interior log-uniform points via Xoshiro256PlusPlus(restarts.seed) and keeps the feasible start
    /// with the lowest objective value.
    /// @tparam RunSingleFn Callable with signature std::expected<OptimizerResult, GprError>(const LogBounds&,
    /// std::span<const double>).
    /// @param[in] obj Optimization objective supplying dimension and open parameter bounds.
    /// @param[in] initial_params Caller-provided initial log-space parameter vector.
    /// @param[in] restarts Multi-start restart count and deterministic Xoshiro256++ seed.
    /// @param[in] run_single Per-start solver callback invoked for initial_params and each sampled restart.
    /// @return Best OptimizerResult across all runs, or GprError if initial validation or all runs fail.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0 or initial_params is empty.
    /// @retval GprErrorCode::LengthMismatch When initial_params.size() != obj.dim().
    /// @retval GprErrorCode::NonFiniteInput When initial_params contains NaN or +/-Inf.
    /// @retval GprErrorCode::InvalidInterval When objective intervals or initial_params violate open bounds.
    template <typename RunSingleFn>
    [[nodiscard]] std::expected<OptimizerResult, GprError> run_with_restarts(
        const Objective& obj, std::span<const double> initial_params, RestartConfig restarts, RunSingleFn&& run_single
    )
    {
        // Validate objective bounds and caller initial point before starting any optimization run.
        auto bounds = LogBounds::from_objective(obj);
        if (!bounds.has_value())
        {
            return std::unexpected(std::move(bounds).error());
        }
        if (auto valid = bounds->validate_log_params(initial_params); !valid.has_value())
        {
            return std::unexpected(std::move(valid).error());
        }

        // Run from the caller-supplied starting point first and return directly if no restarts requested.
        std::expected<OptimizerResult, GprError> best = run_single(*bounds, initial_params);
        if (restarts.count == 0U)
        {
            return best;
        }

        // Sample deterministic log-uniform restarts and retain the lowest finite objective found.
        Xoshiro256PlusPlus rng(restarts.seed);
        std::vector<double> start(bounds->dim());
        for (std::size_t r = 0; r < restarts.count; ++r)
        {
            bounds->sample_log_uniform(rng, start);
            auto candidate = run_single(*bounds, start);
            if (!candidate.has_value())
            {
                if (!best.has_value())
                {
                    best = std::move(candidate);
                }
                continue;
            }
            if (!best.has_value() || candidate->value < best->value)
            {
                best = std::move(candidate);
            }
        }
        return best;
    }

}  // namespace cxxgpr::optimizer::detail
