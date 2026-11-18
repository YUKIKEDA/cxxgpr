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

#include <cmath>
#include <cstddef>
#include <expected>
#include <span>
#include <vector>

/// @brief Internal helper utilities shared among cxxgpr optimizer implementations.
namespace cxxgpr::optimizer::detail
{

    /// @brief Computes the Euclidean dot product of two equal-length vectors.
    /// @details Accumulates sum_i a[i] * b[i] in double precision.
    /// @param a First input vector span.
    /// @param b Second input vector span (must have length >= a.size()).
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
    /// @param a Input vector span.
    /// @return Non-negative Euclidean norm of a.
    [[nodiscard]] inline double norm2_vec(std::span<const double> a) noexcept
    {
        return std::sqrt(dot_vec(a, a));
    }

    /// @brief Executes a single-start solver at initial_params plus deterministic log-uniform restarts.
    /// @details Builds LogBounds from obj, validates initial_params against open bounds, runs
    /// run_single(*bounds, initial_params), and if restarts.count > 0 samples restarts.count additional
    /// interior log-uniform points via Xoshiro256PlusPlus(restarts.seed) and keeps the feasible start
    /// with the lowest objective value.
    /// @tparam RunSingleFn Callable with signature std::expected<OptimizerResult, GprError>(const LogBounds&,
    /// std::span<const double>).
    /// @param obj Optimization objective supplying dimension and open parameter bounds.
    /// @param initial_params Caller-provided initial log-space parameter vector.
    /// @param restarts Multi-start restart count and deterministic Xoshiro256++ seed.
    /// @param run_single Per-start solver callback invoked for initial_params and each sampled restart.
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
