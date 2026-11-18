// SPDX-License-Identifier: Apache-2.0
/// @file refine.cpp
/// @brief Implementation of mixed-precision iterative refinement, norms, and promoted residuals.
/// @details Implements infinity-norm helpers, relative residual scaling checks, column-major
/// float-to-double residual computation, and the refine_in_place() loop with stagnation fallback.

#include "cxxgpr/precision/refine.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <utility>

namespace cxxgpr
{

    namespace
    {

        /// @brief Result of evaluating scaled relative residual convergence.
        /// @details Captures both the numeric relative residual and threshold comparison flag.
        struct ResidualEvaluation
        {
            /// @brief Computed relative residual ||r||_inf / (||B||_inf * ||w||_inf + ||b||_inf).
            double relative_residual = 0.0;
            /// @brief True when relative_residual is finite and strictly below the stopping tolerance.
            bool converged = false;
        };

        /// @brief Evaluates the scaled infinity-norm stopping criterion for a candidate iterate.
        /// @details Guards non-finite inputs, negative norms, non-positive tolerances, and zero
        /// scaling denominators before comparing ||r||_inf / (||B||_inf * ||w||_inf + ||b||_inf) < tol.
        /// @param b_inf Matrix infinity norm ||B||_inf.
        /// @param r_inf Residual infinity norm ||r||_inf.
        /// @param w Current iterate vector.
        /// @param b Right-hand side vector.
        /// @param tol Convergence tolerance on the relative residual.
        /// @return ResidualEvaluation holding relative_residual and convergence verdict.
        ResidualEvaluation evaluate_residual_scale(
            double b_inf, double r_inf, std::span<const double> w, std::span<const double> b, double tol
        ) noexcept
        {
            // Reject invalid operator norms or non-positive tolerances immediately.
            if (!std::isfinite(b_inf) || b_inf < 0.0 || !std::isfinite(tol) || tol <= 0.0)
            {
                return {
                    .relative_residual = (r_inf == 0.0) ? 0.0 : std::numeric_limits<double>::infinity(),
                    .converged = false
                };
            }
            // Compute backward-error scaling denominator ||B||_inf * ||w||_inf + ||b||_inf.
            const double denom = (b_inf * inf_norm(w)) + inf_norm(b);
            if (!std::isfinite(r_inf) || !std::isfinite(denom) || denom <= 0.0)
            {
                return {
                    .relative_residual = (r_inf == 0.0) ? 0.0 : std::numeric_limits<double>::infinity(),
                    .converged = false
                };
            }
            const double rel = r_inf / denom;
            return {.relative_residual = rel, .converged = rel < tol};
        }

    }  // namespace

    /// @brief Returns the infinity norm max_i |values[i]|, or 0.0 when empty.
    /// @details Propagates NaN immediately upon encountering any NaN entry.
    /// @param values Input scalar vector span.
    /// @return Maximum absolute element, 0.0 if empty, or NaN if any entry is NaN.
    double inf_norm(std::span<const double> values) noexcept
    {
        double max_abs = 0.0;
        for (const double value : values)
        {
            const double abs_value = std::abs(value);
            // Return NaN immediately so std::max never masks invalid residual entries.
            if (std::isnan(abs_value))
            {
                return abs_value;
            }
            max_abs = std::max(abs_value, max_abs);
        }
        return max_abs;
    }

    /// @brief Evaluates whether ||residual||_inf / (b_inf * ||w||_inf + ||b||_inf) < tol with denom > 0.
    /// @details Delegates norm computation and denominator checks to evaluate_residual_scale().
    /// @param b_inf Operator infinity norm ||B||_inf.
    /// @param w Current iterate vector.
    /// @param b Right-hand side vector.
    /// @param residual Residual vector r = b - B * w.
    /// @param tol Relative residual convergence threshold.
    /// @return True iff the scaled relative residual is finite and strictly below tol.
    bool meets_relative_residual(
        double b_inf, std::span<const double> w, std::span<const double> b, std::span<const double> residual, double tol
    ) noexcept
    {
        return evaluate_residual_scale(b_inf, inf_norm(residual), w, b, tol).converged;
    }

    /// @brief Computes residual = y - a_f32 * alpha and ||a_f32||_inf using caller-supplied scratch spans.
    /// @details Validates shapes and scratch capacity, zeroes caller scratch, traverses a_f32 in
    /// ascending column-major order for deterministic double accumulation, and writes y - sum.
    /// @param a_f32 Square n x n float system matrix in column-major order.
    /// @param alpha Current double iterate of length n.
    /// @param y Double right-hand side vector of length n.
    /// @param residual Output residual span of length n.
    /// @param row_abs_scratch Caller-owned scratch buffer of length >= n.
    /// @param sum_scratch Caller-owned scratch buffer of length >= n.
    /// @return Infinity norm ||a_f32||_inf, or GprError on invalid shapes or undersized scratch.
    /// @retval GprErrorCode::EmptyInput When any input span or matrix view is empty or null.
    /// @retval GprErrorCode::LengthMismatch When a_f32, alpha, or residual dimensions disagree with y.size().
    /// @retval GprErrorCode::WorkspaceTooSmall When row_abs_scratch or sum_scratch has fewer than n entries.
    std::expected<double, GprError> promoted_residual_into(
        MatrixRef<float> a_f32,
        std::span<const double> alpha,
        std::span<const double> y,
        std::span<double> residual,
        std::span<double> row_abs_scratch,
        std::span<double> sum_scratch
    )
    {
        // Enforce EmptyInput -> LengthMismatch -> WorkspaceTooSmall before touching scratch buffers.
        if (y.empty() || alpha.empty() || residual.empty() || a_f32.is_empty() || a_f32.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = y.size();
        if (a_f32.rows() != n || a_f32.cols() != n)
        {
            return std::unexpected(
                GprError::length_mismatch(
                    std::format("expected {}x{} a_f32 matrix, got {}x{}", n, n, a_f32.rows(), a_f32.cols())
                )
            );
        }
        if (auto ok = require_count(alpha.size(), n, "alpha"); !ok)
        {
            return std::unexpected(std::move(ok).error());
        }
        if (auto ok = require_count(residual.size(), n, "residual"); !ok)
        {
            return std::unexpected(std::move(ok).error());
        }
        if (row_abs_scratch.size() < n || sum_scratch.size() < n)
        {
            return std::unexpected(GprError::workspace_too_small());
        }

        // Zero only the active n-element prefix of caller-provided scratch arrays.
        std::fill_n(row_abs_scratch.data(), n, 0.0);
        std::fill_n(sum_scratch.data(), n, 0.0);

        // Column-major traversal accumulates each row sum and row absolute sum in ascending column
        // order.
        for (std::size_t j = 0; j < n; ++j)
        {
            const auto col = a_f32.col(j);
            const double alpha_j = alpha[j];
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto a_ij = static_cast<double>(col[i]);
                row_abs_scratch[i] += std::abs(a_ij);
                sum_scratch[i] += a_ij * alpha_j;
            }
        }

        // Subtract the promoted matrix-vector product from y in double precision to form r.
        for (std::size_t i = 0; i < n; ++i)
        {
            residual[i] = y[i] - sum_scratch[i];
        }
        return inf_norm(row_abs_scratch.first(n));
    }

    /// @brief Computes residual = y - a_f32 * alpha and ||a_f32||_inf with temporary internal scratch.
    /// @details Allocates two temporary n-element double buffers and delegates to promoted_residual_into().
    /// @param a_f32 Square n x n float system matrix in column-major order.
    /// @param alpha Current double iterate of length n.
    /// @param y Double right-hand side vector of length n.
    /// @param residual Output residual span of length n.
    /// @return Infinity norm ||a_f32||_inf, or GprError on invalid dimensions.
    /// @retval GprErrorCode::EmptyInput When any input span or matrix view is empty or null.
    /// @retval GprErrorCode::LengthMismatch When a_f32, alpha, or residual dimensions disagree with y.size().
    std::expected<double, GprError> promoted_residual(
        MatrixRef<float> a_f32, std::span<const double> alpha, std::span<const double> y, std::span<double> residual
    )
    {
        // Reject empty inputs before allocating temporary scratch vectors.
        if (y.empty() || alpha.empty() || residual.empty() || a_f32.is_empty() || a_f32.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = y.size();
        std::vector<double> row_abs(n, 0.0);
        std::vector<double> sum(n, 0.0);
        return promoted_residual_into(a_f32, alpha, y, residual, row_abs, sum);
    }

    /// @brief Default acceptance hook accepting any iterate that met the residual tolerance.
    /// @details Derived systems (such as kernel-reevaluated systems) may override this check to
    /// verify the residual against true double-precision matrix entries before accepting.
    /// @return Always returns true.
    std::expected<bool, GprError> RefineSystem::accept(std::span<const double> /*w*/, double /*tol*/)
    {
        return true;
    }

    /// @brief Refines an initial iterate w in-place using a caller-owned residual buffer with zero heap allocations.
    /// @details Executes up to max_refinement_corrections steps of residual evaluation, convergence
    /// gating via system.accept(), stagnation streak detection, and triangular solve correction,
    /// falling back to system.fallback() whenever convergence cannot be confirmed.
    /// @param system Polymorphic linear system providing residual, correction, acceptance, and fallback.
    /// @param w Initial iterate overwritten in-place with the refined or fallback solution.
    /// @param residual_scratch Caller-supplied buffer of length >= w.size().
    /// @return RefinementSummary detailing final status, step count, and relative residual, or GprError.
    /// @retval GprErrorCode::EmptyInput When w or system.rhs() is empty.
    /// @retval GprErrorCode::LengthMismatch When w.size() != system.rhs().size().
    /// @retval GprErrorCode::WorkspaceTooSmall When residual_scratch.size() < w.size().
    std::expected<RefinementSummary, GprError> refine_in_place(
        RefineSystem& system, std::span<double> w, std::span<double> residual_scratch
    )
    {
        // Validate iterate, right-hand side, and scratch length before entering the zero-alloc loop.
        const std::span<const double> b = system.rhs();
        if (w.empty() || b.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto ok = require_count(w.size(), b.size(), "refinement iterate w"); !ok)
        {
            return std::unexpected(std::move(ok).error());
        }
        const std::size_t dim = w.size();
        if (residual_scratch.size() < dim)
        {
            return std::unexpected(GprError::workspace_too_small());
        }

        // Stopping threshold is 10 * dim * eps_64 on the relative infinity-norm residual.
        const double tol = refinement_tolerance(dim);
        const std::span<double> resid = residual_scratch.first(dim);
        std::optional<double> prev_r_inf;
        std::size_t streak = 0;
        double last_rel_resid = 0.0;

        for (std::size_t step = 0; step < max_refinement_corrections; ++step)
        {
            // Evaluate r = b - B * w and ||B||_inf at the start of the step.
            auto b_inf = system.residual(w, resid);
            if (!b_inf)
            {
                return std::unexpected(std::move(b_inf).error());
            }

            const double r_inf = inf_norm(resid);
            const ResidualEvaluation eval = evaluate_residual_scale(*b_inf, r_inf, w, b, tol);
            last_rel_resid = eval.relative_residual;

            if (eval.converged)
            {
                // Run optional higher-precision acceptance verification before declaring convergence.
                auto accepted = system.accept(w, tol);
                if (!accepted)
                {
                    return std::unexpected(std::move(accepted).error());
                }
                if (!*accepted)
                {
                    if (auto fb = system.fallback(w); !fb)
                    {
                        return std::unexpected(std::move(fb).error());
                    }
                    return RefinementSummary{
                        .status = RefinementStatus::RejectedByAccept,
                        .iterations = step,
                        .final_relative_residual = last_rel_resid,
                    };
                }
                return RefinementSummary{
                    .status = RefinementStatus::Converged,
                    .iterations = step,
                    .final_relative_residual = last_rel_resid,
                };
            }

            // Track consecutive non-improving or non-finite residual norms to trigger early fallback.
            if (prev_r_inf.has_value())
            {
                const bool non_finite = !std::isfinite(r_inf) || !std::isfinite(*prev_r_inf);
                const double ratio = (!non_finite && (*prev_r_inf > 0.0)) ? (r_inf / *prev_r_inf) : 0.0;
                if (non_finite || !std::isfinite(ratio) || ratio > refinement_stagnation_ratio)
                {
                    ++streak;
                    if (streak >= refinement_stagnation_steps)
                    {
                        if (auto fb = system.fallback(w); !fb)
                        {
                            return std::unexpected(std::move(fb).error());
                        }
                        return RefinementSummary{
                            .status = RefinementStatus::Stagnated,
                            .iterations = step,
                            .final_relative_residual = last_rel_resid,
                        };
                    }
                }
                else
                {
                    streak = 0;
                }
            }

            // Apply the stored float Cholesky correction step w <- w + L^{-T} L^{-1} r.
            prev_r_inf = r_inf;
            if (auto corr = system.correct(resid, w); !corr)
            {
                return std::unexpected(std::move(corr).error());
            }
        }

        // Fall back to a full double solve if max_refinement_corrections corrections did not converge.
        if (auto fb = system.fallback(w); !fb)
        {
            return std::unexpected(std::move(fb).error());
        }
        return RefinementSummary{
            .status = RefinementStatus::MaxCorrectionsReached,
            .iterations = max_refinement_corrections,
            .final_relative_residual = last_rel_resid,
        };
    }

    /// @brief Convenience wrapper that allocates residual scratch and returns the refined iterate vector.
    /// @details Validates non-emptiness, allocates w.size() doubles for residuals, and calls refine_in_place().
    /// @param system Polymorphic linear system providing residual, correction, acceptance, and fallback.
    /// @param w Initial iterate vector moved in and updated in-place.
    /// @return Refined (or fallback) solution vector, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When w or system.rhs() is empty.
    /// @retval GprErrorCode::LengthMismatch When w.size() != system.rhs().size().
    std::expected<std::vector<double>, GprError> refine(RefineSystem& system, std::vector<double> w)
    {
        // Guard empty inputs before allocating scratch; refine_in_place() enforces exact length matching.
        if (w.empty() || system.rhs().empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        std::vector<double> residual_scratch(w.size(), 0.0);
        auto summary = refine_in_place(system, w, residual_scratch);
        if (!summary)
        {
            return std::unexpected(std::move(summary).error());
        }
        return w;
    }

}  // namespace cxxgpr
