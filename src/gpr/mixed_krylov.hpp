// SPDX-License-Identifier: Apache-2.0
/// @file mixed_krylov.hpp
/// @brief Preconditioner-shifted float Cholesky, scaled triangular solves, and restarted double FGMRES-IR.
/// @details Internal numerical helpers shared by the coordinate and supplied-distance engines to solve
/// ill-conditioned mixed-precision systems without perturbing the linear equation B * alpha = y unless the
/// true double-precision system is genuinely non-positive-definite.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/precision/refine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <span>
#include <utility>
#include <vector>

/// @brief Internal mixed-precision Krylov refinement helpers.
namespace cxxgpr::detail
{

    /// @brief Computes deterministic double-precision dot product a^T * b in ascending index order.
    /// @param[in] a First vector span of length n.
    /// @param[in] b Second vector span of length n.
    /// @return Inner product sum_{i=0}^{n-1} a[i] * b[i].
    [[nodiscard]] inline double dot_f64(std::span<const double> a, std::span<const double> b) noexcept
    {
        double acc = 0.0;
        const std::size_t n = a.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            acc += a[i] * b[i];
        }
        return acc;
    }

    /// @brief Computes deterministic double-precision Euclidean norm ||v||_2 in ascending index order.
    /// @param[in] v Input vector span.
    /// @return Euclidean norm sqrt(v^T * v).
    [[nodiscard]] inline double norm2_f64(std::span<const double> v) noexcept
    {
        const double max_abs = inf_norm(v);
        if (!std::isfinite(max_abs) || max_abs == 0.0)
        {
            return max_abs;
        }
        const double inv_scale = 1.0 / max_abs;
        double sum_sq = 0.0;
        for (const double x : v)
        {
            const double scaled = x * inv_scale;
            sum_sq += scaled * scaled;
        }
        return max_abs * std::sqrt(sum_sq);
    }

    /// @brief Computes scaled in-place vector update y <- y + alpha * x.
    /// @param[in] alpha Scalar multiplier.
    /// @param[in] x Direction vector of length n.
    /// @param[in,out] y Accumulator vector of length n updated in place.
    inline void axpy_f64(double alpha, std::span<const double> x, std::span<double> y) noexcept
    {
        const std::size_t n = y.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            y[i] += alpha * x[i];
        }
    }

    /// @brief Factors a float system matrix with diagonal scaling and a preconditioner-only shift when needed.
    /// @details Tries unshifted lower-triangular float Cholesky first using backup_mut as scratch. On failure,
    /// verifies positive finite diagonal entries, then attempts geometric relative shifts delta in
    /// {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0} applied to the diagonally scaled matrix
    /// D^{-1/2} B D^{-1/2} + delta I, rescaling rows back to L_eff = D^{1/2} L_hat so L_eff * L_eff^T approximates
    /// B + delta D. Falls back to unscaled relative shift delta * max_diag if rescaling overflows float.
    /// @param[in,out] k_mut Square n x n float system matrix lower triangle on entry, factor L on success.
    /// @param[out] backup_mut Preallocated n x n float backup buffer holding the lower triangle of B.
    /// @param[out] out_shift Adopted preconditioner-only diagonal shift (0.0 when unshifted succeeded).
    /// @return Empty expected on success, or CholeskyFailed / NonFiniteInput on unfactorizable diagonal or failure.
    [[nodiscard]] inline std::expected<void, GprError> factor_float_with_preconditioner_shift(
        MatrixMut<float> k_mut, MatrixMut<float> backup_mut, double& out_shift
    )
    {
        out_shift = 0.0;
        const std::size_t n = k_mut.rows();
        if (n == 0U || k_mut.data() == nullptr || backup_mut.data() == nullptr || !k_mut.is_square()
            || backup_mut.rows() != n || !backup_mut.is_square())
        {
            return std::unexpected(GprError::empty_input());
        }

        // Preserve lower triangle before attempting unshifted factorization.
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col; row < n; ++row)
            {
                backup_mut(row, col) = k_mut(row, col);
            }
        }

        if (auto direct = linalg::cholesky_lower_in_place(k_mut, CholeskyStage::Fit); direct.has_value())
        {
            out_shift = 0.0;
            return {};
        }

        std::vector<double> d_sqrt(n, 0.0);
        std::vector<double> inv_d_sqrt(n, 0.0);
        double max_diag = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const float raw_diag = backup_mut(i, i);
            if (!std::isfinite(raw_diag))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            if (!(raw_diag > 0.0F))
            {
                return std::unexpected(GprError::cholesky_failed(0.0, n, CholeskyStage::Fit));
            }
            const auto d_i = static_cast<double>(raw_diag);
            d_sqrt[i] = std::sqrt(d_i);
            inv_d_sqrt[i] = 1.0 / d_sqrt[i];
            max_diag = std::max(max_diag, d_i);
        }

        static constexpr std::array<double, 7> preconditioner_shifts{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
        for (const double delta : preconditioner_shifts)
        {
            // Attempt diagonally scaled shifted preconditioner D^{-1/2} B D^{-1/2} + delta I.
            bool scaled_finite = true;
            for (std::size_t col = 0; col < n && scaled_finite; ++col)
            {
                const double inv_col = inv_d_sqrt[col];
                k_mut(col, col) = static_cast<float>(1.0 + delta);
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const auto scaled_entry =
                        static_cast<float>(static_cast<double>(backup_mut(row, col)) * inv_d_sqrt[row] * inv_col);
                    if (!std::isfinite(scaled_entry))
                    {
                        scaled_finite = false;
                        break;
                    }
                    k_mut(row, col) = scaled_entry;
                }
            }

            if (scaled_finite)
            {
                if (auto res = linalg::cholesky_lower_in_place(k_mut, CholeskyStage::Fit); res.has_value())
                {
                    bool rescale_valid = true;
                    for (std::size_t col = 0; col < n && rescale_valid; ++col)
                    {
                        for (std::size_t row = col; row < n; ++row)
                        {
                            const auto rescaled =
                                static_cast<float>(static_cast<double>(k_mut(row, col)) * d_sqrt[row]);
                            if (!std::isfinite(rescaled) || (row == col && !(rescaled > 0.0F)))
                            {
                                rescale_valid = false;
                                break;
                            }
                            k_mut(row, col) = rescaled;
                        }
                    }
                    if (rescale_valid)
                    {
                        out_shift = delta;
                        return {};
                    }
                }
            }

            // Fallback within the same shift level: unscaled diagonal shift B + delta * max_diag * I.
            bool direct_finite = true;
            const double diag_bump = delta * max_diag;
            for (std::size_t col = 0; col < n && direct_finite; ++col)
            {
                const auto shifted_diag = static_cast<float>(static_cast<double>(backup_mut(col, col)) + diag_bump);
                if (!std::isfinite(shifted_diag) || !(shifted_diag > 0.0F))
                {
                    direct_finite = false;
                    break;
                }
                k_mut(col, col) = shifted_diag;
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    k_mut(row, col) = backup_mut(row, col);
                }
            }
            if (direct_finite)
            {
                if (auto res = linalg::cholesky_lower_in_place(k_mut, CholeskyStage::Fit); res.has_value())
                {
                    out_shift = delta;
                    return {};
                }
            }
        }

        return std::unexpected(GprError::cholesky_failed(0.0, n, CholeskyStage::Fit));
    }

    /// @brief Applies the float Cholesky preconditioner L * L^T with infinity-norm scaling to prevent float overflow.
    /// @details Scales input r by 1 / ||r||_inf prior to float triangular substitution and rescales out_z by ||r||_inf.
    /// @param[in] l_f32 Lower-triangular float Cholesky factor view of dimension n x n.
    /// @param[in] r Input double-precision residual or Arnoldi vector of length n.
    /// @param[out] tmp_f32 Preallocated float scratch span of length n.
    /// @param[out] out_z Output double-precision preconditioned vector of length n.
    /// @return Empty expected on success, or GprError on non-finite input/output or solve failure.
    [[nodiscard]] inline std::expected<void, GprError> apply_scaled_f32_preconditioner(
        MatrixRef<float> l_f32, std::span<const double> r, std::span<float> tmp_f32, std::span<double> out_z
    )
    {
        const std::size_t n = r.size();
        const double scale = inf_norm(r);
        if (!std::isfinite(scale))
        {
            return std::unexpected(GprError::non_finite_kernel_value());
        }
        if (scale == 0.0)
        {
            std::ranges::fill(out_z, 0.0);
            return {};
        }
        const double inv_scale = 1.0 / scale;
        for (std::size_t i = 0; i < n; ++i)
        {
            tmp_f32[i] = static_cast<float>(r[i] * inv_scale);
        }
        if (auto sv = linalg::solve_llt_in_place(l_f32, tmp_f32); !sv.has_value())
        {
            return sv;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const double val = static_cast<double>(tmp_f32[i]) * scale;
            if (!std::isfinite(val))
            {
                return std::unexpected(GprError::non_finite_kernel_value());
            }
            out_z[i] = val;
        }
        return {};
    }

    /// @brief Evaluates backward-error relative residual ||r||_inf / (b_inf * ||w||_inf + ||b||_inf).
    /// @param[in] b_inf Matrix infinity norm ||B||_inf.
    /// @param[in] r_inf Residual infinity norm ||r||_inf.
    /// @param[in] w Candidate solution vector.
    /// @param[in] b_norm_inf Right-hand side infinity norm ||b||_inf.
    /// @return Scaled relative residual (or +infinity if denominator is zero or norms are non-finite).
    [[nodiscard]] inline double compute_relative_residual_value(
        double b_inf, double r_inf, std::span<const double> w, double b_norm_inf
    ) noexcept
    {
        if (!std::isfinite(b_inf) || b_inf < 0.0 || !std::isfinite(r_inf) || !std::isfinite(b_norm_inf))
        {
            return (r_inf == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        }
        const double denom = (b_inf * inf_norm(w)) + b_norm_inf;
        if (!std::isfinite(denom) || denom <= 0.0)
        {
            return (r_inf == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        }
        return r_inf / denom;
    }

    /// @brief Refines alpha under MixedPrecision<ReevaluateKernel> using hybrid IR and restarted double FGMRES-IR.
    /// @details Runs stationary IR first when preconditioner_shift == 0.0, escalates to restarted double FGMRES-IR
    /// (checking positive definiteness z^T B z > 0 at each Krylov vector) on stagnation or nonzero shift, and
    /// delegates to ops.fallback(w) only if FGMRES-IR cannot reach tolerance within max_krylov_iterations.
    /// @tparam Ops Operations adapter providing rhs(), matvec_and_inf_norm(), apply_preconditioner(), fallback().
    /// @param[in,out] ops System adapter exposing exact blocked double matvecs, float preconditioning, and fallback.
    /// @param[in,out] w Initial iterate overwritten in place with the converged or fallback double weights.
    /// @param[in] preconditioner_shift Diagonal relative shift applied when building the float preconditioner.
    /// @return RefinementSummary describing convergence status, iteration counts, shift, and Krylov usage.
    template <typename Ops>
    [[nodiscard]] std::expected<RefinementSummary, GprError> refine_reevaluate_with_krylov(
        Ops& ops, std::span<double> w, double preconditioner_shift
    )
    {
        const std::span<const double> b = ops.rhs();
        const std::size_t n = b.size();
        if (w.empty() || b.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (w.size() != n)
        {
            return std::unexpected(GprError::length_mismatch("refinement iterate length must match rhs length"));
        }

        const double tol = refinement_tolerance(n);
        const double b_norm_inf = inf_norm(b);

        std::vector<double> bv(n, 0.0);
        std::vector<double> resid(n, 0.0);
        std::vector<double> corr(n, 0.0);

        auto initial_norm = ops.matvec_and_inf_norm(w, bv, true);
        if (!initial_norm.has_value())
        {
            return std::unexpected(std::move(initial_norm).error());
        }
        const double b_inf = *initial_norm;
        for (std::size_t i = 0; i < n; ++i)
        {
            resid[i] = b[i] - bv[i];
        }

        double r_inf = inf_norm(resid);
        double rel = compute_relative_residual_value(b_inf, r_inf, w, b_norm_inf);
        if (std::isfinite(rel) && rel < tol)
        {
            return RefinementSummary{
                .status = RefinementStatus::Converged,
                .iterations = 0U,
                .final_relative_residual = rel,
                .preconditioner_shift = preconditioner_shift,
                .used_krylov = false,
            };
        }

        std::vector<double> w_best(w.begin(), w.end());
        std::vector<double> r_best(resid.begin(), resid.end());
        double best_r_inf = std::isfinite(r_inf) ? r_inf : std::numeric_limits<double>::infinity();
        double best_rel = rel;
        std::size_t ir_iters = 0U;
        bool stagnated = false;

        // Phase 1: classic stationary iterative refinement when unshifted float Cholesky succeeded.
        if (preconditioner_shift == 0.0)
        {
            std::optional<double> prev_r_inf = r_inf;
            std::size_t streak = 0U;
            for (std::size_t step = 0; step < max_refinement_corrections; ++step)
            {
                if (auto pc = ops.apply_preconditioner(resid, corr); !pc.has_value())
                {
                    break;
                }
                for (std::size_t i = 0; i < n; ++i)
                {
                    w[i] += corr[i];
                }
                ++ir_iters;

                auto norm_res = ops.matvec_and_inf_norm(w, bv, false);
                if (!norm_res.has_value())
                {
                    return std::unexpected(std::move(norm_res).error());
                }
                for (std::size_t i = 0; i < n; ++i)
                {
                    resid[i] = b[i] - bv[i];
                }
                r_inf = inf_norm(resid);
                rel = compute_relative_residual_value(b_inf, r_inf, w, b_norm_inf);
                if (std::isfinite(r_inf) && r_inf < best_r_inf)
                {
                    best_r_inf = r_inf;
                    best_rel = rel;
                    std::ranges::copy(w, w_best.begin());
                    std::ranges::copy(resid, r_best.begin());
                }
                if (std::isfinite(rel) && rel < tol)
                {
                    return RefinementSummary{
                        .status = RefinementStatus::Converged,
                        .iterations = ir_iters,
                        .final_relative_residual = rel,
                        .preconditioner_shift = 0.0,
                        .used_krylov = false,
                    };
                }

                if (prev_r_inf.has_value())
                {
                    const bool non_finite = !std::isfinite(r_inf) || !std::isfinite(*prev_r_inf);
                    const double ratio = (!non_finite && (*prev_r_inf > 0.0)) ? (r_inf / *prev_r_inf) : 0.0;
                    if (non_finite || !std::isfinite(ratio) || ratio > refinement_stagnation_ratio)
                    {
                        ++streak;
                        if (streak >= refinement_stagnation_steps)
                        {
                            stagnated = true;
                            break;
                        }
                    }
                    else
                    {
                        streak = 0U;
                    }
                }
                prev_r_inf = r_inf;
            }
            // Start Krylov refinement from the best iterate seen during Phase 1.
            std::ranges::copy(w_best, w.begin());
            std::ranges::copy(r_best, resid.begin());
            r_inf = best_r_inf;
            rel = best_rel;
        }

        // Phase 2: restarted double-precision FGMRES-IR using one-column-per-vector storage (8*n bytes max alloc).
        const std::size_t restart_m = std::min(krylov_restart_dim, n);
        std::vector<std::vector<double>> basis_v(restart_m + 1U, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> basis_z(restart_m, std::vector<double>(n, 0.0));
        std::vector<double> hess_h((restart_m + 1U) * restart_m, 0.0);
        std::vector<double> cs(restart_m, 0.0);
        std::vector<double> sn(restart_m, 0.0);
        std::vector<double> g_vec(restart_m + 1U, 0.0);
        std::vector<double> eta(restart_m, 0.0);
        std::vector<double> u(n, 0.0);

        auto h_at = [&hess_h, stride = restart_m + 1U](std::size_t row, std::size_t col) -> double&
        {
            return hess_h[(col * stride) + row];
        };

        std::size_t krylov_iters = 0U;
        bool indefinite_detected = false;
        std::size_t cycle_stagnation = 0U;
        double prev_cycle_beta = norm2_f64(resid);

        while (krylov_iters < max_krylov_iterations)
        {
            const double beta = norm2_f64(resid);
            if (!std::isfinite(beta))
            {
                stagnated = true;
                break;
            }
            if (beta == 0.0)
            {
                return RefinementSummary{
                    .status = RefinementStatus::Converged,
                    .iterations = ir_iters + krylov_iters,
                    .final_relative_residual = 0.0,
                    .preconditioner_shift = preconditioner_shift,
                    .used_krylov = true,
                };
            }

            std::ranges::fill(hess_h, 0.0);
            std::ranges::fill(cs, 0.0);
            std::ranges::fill(sn, 0.0);
            std::ranges::fill(g_vec, 0.0);
            g_vec[0] = beta;

            const double inv_beta = 1.0 / beta;
            for (std::size_t i = 0; i < n; ++i)
            {
                basis_v[0][i] = resid[i] * inv_beta;
            }

            std::size_t k_sub = 0U;
            for (std::size_t j = 0; j < restart_m && krylov_iters < max_krylov_iterations; ++j)
            {
                if (auto pc = ops.apply_preconditioner(basis_v[j], basis_z[j]); !pc.has_value())
                {
                    stagnated = true;
                    break;
                }
                if (auto mv = ops.matvec_and_inf_norm(basis_z[j], u, false); !mv.has_value())
                {
                    return std::unexpected(std::move(mv).error());
                }

                // Check SPD Rayleigh quotient z^T B z > 0 before admitting this direction.
                const double zz = dot_f64(basis_z[j], basis_z[j]);
                const double zbz = dot_f64(basis_z[j], u);
                if (!std::isfinite(zbz) || (zz > 0.0 && zbz <= 0.0))
                {
                    indefinite_detected = true;
                    break;
                }

                // Two-pass Modified Gram-Schmidt for orthogonality across severe condition numbers.
                for (std::size_t i = 0; i <= j; ++i)
                {
                    const double hij = dot_f64(u, basis_v[i]);
                    h_at(i, j) += hij;
                    axpy_f64(-hij, basis_v[i], u);
                }
                for (std::size_t i = 0; i <= j; ++i)
                {
                    const double corr_hij = dot_f64(u, basis_v[i]);
                    h_at(i, j) += corr_hij;
                    axpy_f64(-corr_hij, basis_v[i], u);
                }

                const double h_next = norm2_f64(u);
                h_at(j + 1U, j) = h_next;
                if (h_next > 0.0 && std::isfinite(h_next))
                {
                    const double inv_h = 1.0 / h_next;
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        basis_v[j + 1U][row] = u[row] * inv_h;
                    }
                }

                for (std::size_t i = 0; i < j; ++i)
                {
                    const double h0 = h_at(i, j);
                    const double h1 = h_at(i + 1U, j);
                    h_at(i, j) = (cs[i] * h0) + (sn[i] * h1);
                    h_at(i + 1U, j) = (-sn[i] * h0) + (cs[i] * h1);
                }
                const double diag_val = h_at(j, j);
                const double sub_val = h_at(j + 1U, j);
                const double radius = std::hypot(diag_val, sub_val);
                if (!(radius > 0.0) || !std::isfinite(radius))
                {
                    break;
                }
                cs[j] = diag_val / radius;
                sn[j] = sub_val / radius;
                h_at(j, j) = radius;
                h_at(j + 1U, j) = 0.0;
                const double gj = g_vec[j];
                g_vec[j] = cs[j] * gj;
                g_vec[j + 1U] = -sn[j] * gj;

                k_sub = j + 1U;
                ++krylov_iters;

                const double denom_est = (b_inf * inf_norm(w)) + b_norm_inf;
                const double est_res = std::abs(g_vec[j + 1U]);
                if (h_next == 0.0 || !std::isfinite(h_next) || (denom_est > 0.0 && est_res <= 0.25 * tol * denom_est))
                {
                    break;
                }
            }

            if (k_sub == 0U)
            {
                break;
            }

            // Back-substitute upper triangular H(0..k_sub-1, 0..k_sub-1) * eta = g_vec(0..k_sub-1).
            bool bad_tri = false;
            for (std::size_t rev = k_sub; rev > 0U; --rev)
            {
                const std::size_t row = rev - 1U;
                double sum = g_vec[row];
                for (std::size_t col = row + 1U; col < k_sub; ++col)
                {
                    sum -= h_at(row, col) * eta[col];
                }
                const double diag = h_at(row, row);
                if (diag == 0.0 || !std::isfinite(diag))
                {
                    bad_tri = true;
                    break;
                }
                eta[row] = sum / diag;
                if (!std::isfinite(eta[row]))
                {
                    bad_tri = true;
                    break;
                }
            }
            if (bad_tri)
            {
                stagnated = true;
                break;
            }

            for (std::size_t col = 0; col < k_sub; ++col)
            {
                axpy_f64(eta[col], basis_z[col], w);
            }

            if (auto mv = ops.matvec_and_inf_norm(w, bv, false); !mv.has_value())
            {
                return std::unexpected(std::move(mv).error());
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                resid[i] = b[i] - bv[i];
            }
            r_inf = inf_norm(resid);
            rel = compute_relative_residual_value(b_inf, r_inf, w, b_norm_inf);
            if (std::isfinite(rel) && rel < tol)
            {
                return RefinementSummary{
                    .status = RefinementStatus::Converged,
                    .iterations = ir_iters + krylov_iters,
                    .final_relative_residual = rel,
                    .preconditioner_shift = preconditioner_shift,
                    .used_krylov = true,
                };
            }

            if (indefinite_detected)
            {
                break;
            }
            const double new_beta = norm2_f64(resid);
            if (!std::isfinite(new_beta) || (prev_cycle_beta > 0.0 && new_beta >= 0.99 * prev_cycle_beta))
            {
                ++cycle_stagnation;
                if (cycle_stagnation >= refinement_stagnation_steps)
                {
                    stagnated = true;
                    break;
                }
            }
            else
            {
                cycle_stagnation = 0U;
            }
            prev_cycle_beta = new_beta;
        }

        // Release Krylov bases before allocating double-precision fallback buffers.
        std::vector<std::vector<double>>{}.swap(basis_v);
        std::vector<std::vector<double>>{}.swap(basis_z);
        std::vector<double>{}.swap(hess_h);
        std::vector<double>{}.swap(u);
        std::vector<double>{}.swap(bv);
        std::vector<double>{}.swap(resid);
        std::vector<double>{}.swap(corr);

        if (auto fb = ops.fallback(w); !fb.has_value())
        {
            return std::unexpected(std::move(fb).error());
        }
        return RefinementSummary{
            .status = (stagnated || indefinite_detected) ? RefinementStatus::Stagnated
                                                         : RefinementStatus::MaxCorrectionsReached,
            .iterations = ir_iters + krylov_iters,
            .final_relative_residual = rel,
            .preconditioner_shift = preconditioner_shift,
            .used_krylov = krylov_iters > 0U,
        };
    }

}  // namespace cxxgpr::detail
