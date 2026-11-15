// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/precision/refine.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <utility>

namespace cxxgpr {

namespace {

struct ResidualEvaluation {
    double relative_residual = 0.0;
    bool converged = false;
};

ResidualEvaluation evaluate_residual_scale(double b_inf, double r_inf, std::span<const double> w,
                                           std::span<const double> b, double tol) noexcept {
    if (!std::isfinite(b_inf) || b_inf < 0.0 || !std::isfinite(tol) || tol <= 0.0) {
        return {.relative_residual = (r_inf == 0.0) ? 0.0 : std::numeric_limits<double>::infinity(),
                .converged = false};
    }
    const double denom = (b_inf * inf_norm(w)) + inf_norm(b);
    if (!std::isfinite(r_inf) || !std::isfinite(denom) || denom <= 0.0) {
        return {.relative_residual = (r_inf == 0.0) ? 0.0 : std::numeric_limits<double>::infinity(),
                .converged = false};
    }
    const double rel = r_inf / denom;
    return {.relative_residual = rel, .converged = rel < tol};
}

} // namespace

double inf_norm(std::span<const double> values) noexcept {
    double max_abs = 0.0;
    for (const double value : values) {
        const double abs_value = std::abs(value);
        if (std::isnan(abs_value)) {
            return abs_value;
        }
        max_abs = std::max(abs_value, max_abs);
    }
    return max_abs;
}

bool meets_relative_residual(double b_inf, std::span<const double> w, std::span<const double> b,
                             std::span<const double> residual, double tol) noexcept {
    return evaluate_residual_scale(b_inf, inf_norm(residual), w, b, tol).converged;
}

std::expected<double, GprError>
promoted_residual_into(MatrixRef<float> a_f32, std::span<const double> alpha,
                       std::span<const double> y, std::span<double> residual,
                       std::span<double> row_abs_scratch, std::span<double> sum_scratch) {
    if (y.empty() || alpha.empty() || residual.empty() || a_f32.is_empty() ||
        a_f32.data() == nullptr) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t n = y.size();
    if (a_f32.rows() != n || a_f32.cols() != n) {
        return std::unexpected(GprError::length_mismatch(std::format(
            "expected {}x{} a_f32 matrix, got {}x{}", n, n, a_f32.rows(), a_f32.cols())));
    }
    if (auto ok = require_count(alpha.size(), n, "alpha"); !ok) {
        return std::unexpected(std::move(ok).error());
    }
    if (auto ok = require_count(residual.size(), n, "residual"); !ok) {
        return std::unexpected(std::move(ok).error());
    }
    if (row_abs_scratch.size() < n || sum_scratch.size() < n) {
        return std::unexpected(GprError::workspace_too_small());
    }

    std::fill_n(row_abs_scratch.data(), n, 0.0);
    std::fill_n(sum_scratch.data(), n, 0.0);

    // Column-major traversal accumulates each row sum and row absolute sum in ascending column
    // order.
    for (std::size_t j = 0; j < n; ++j) {
        const auto col = a_f32.col(j);
        const double alpha_j = alpha[j];
        for (std::size_t i = 0; i < n; ++i) {
            const auto a_ij = static_cast<double>(col[i]);
            row_abs_scratch[i] += std::abs(a_ij);
            sum_scratch[i] += a_ij * alpha_j;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        residual[i] = y[i] - sum_scratch[i];
    }
    return inf_norm(row_abs_scratch.first(n));
}

std::expected<double, GprError> promoted_residual(MatrixRef<float> a_f32,
                                                  std::span<const double> alpha,
                                                  std::span<const double> y,
                                                  std::span<double> residual) {
    if (y.empty() || alpha.empty() || residual.empty() || a_f32.is_empty() ||
        a_f32.data() == nullptr) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t n = y.size();
    std::vector<double> row_abs(n, 0.0);
    std::vector<double> sum(n, 0.0);
    return promoted_residual_into(a_f32, alpha, y, residual, row_abs, sum);
}

std::expected<bool, GprError> RefineSystem::accept(std::span<const double> /*w*/, double /*tol*/) {
    return true;
}

std::expected<RefinementSummary, GprError>
refine_in_place(RefineSystem& system, std::span<double> w, std::span<double> residual_scratch) {
    const std::span<const double> b = system.rhs();
    if (w.empty() || b.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (auto ok = require_count(w.size(), b.size(), "refinement iterate w"); !ok) {
        return std::unexpected(std::move(ok).error());
    }
    const std::size_t dim = w.size();
    if (residual_scratch.size() < dim) {
        return std::unexpected(GprError::workspace_too_small());
    }

    const double tol = refinement_tolerance(dim);
    const std::span<double> resid = residual_scratch.first(dim);
    std::optional<double> prev_r_inf;
    std::size_t streak = 0;
    double last_rel_resid = 0.0;

    for (std::size_t step = 0; step < max_refinement_corrections; ++step) {
        auto b_inf = system.residual(w, resid);
        if (!b_inf) {
            return std::unexpected(std::move(b_inf).error());
        }

        const double r_inf = inf_norm(resid);
        const ResidualEvaluation eval = evaluate_residual_scale(*b_inf, r_inf, w, b, tol);
        last_rel_resid = eval.relative_residual;

        if (eval.converged) {
            auto accepted = system.accept(w, tol);
            if (!accepted) {
                return std::unexpected(std::move(accepted).error());
            }
            if (!*accepted) {
                if (auto fb = system.fallback(w); !fb) {
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

        if (prev_r_inf.has_value()) {
            const bool non_finite = !std::isfinite(r_inf) || !std::isfinite(*prev_r_inf);
            const double ratio = (!non_finite && (*prev_r_inf > 0.0)) ? (r_inf / *prev_r_inf) : 0.0;
            if (non_finite || !std::isfinite(ratio) || ratio > refinement_stagnation_ratio) {
                ++streak;
                if (streak >= refinement_stagnation_steps) {
                    if (auto fb = system.fallback(w); !fb) {
                        return std::unexpected(std::move(fb).error());
                    }
                    return RefinementSummary{
                        .status = RefinementStatus::Stagnated,
                        .iterations = step,
                        .final_relative_residual = last_rel_resid,
                    };
                }
            } else {
                streak = 0;
            }
        }

        prev_r_inf = r_inf;
        if (auto corr = system.correct(resid, w); !corr) {
            return std::unexpected(std::move(corr).error());
        }
    }

    if (auto fb = system.fallback(w); !fb) {
        return std::unexpected(std::move(fb).error());
    }
    return RefinementSummary{
        .status = RefinementStatus::MaxCorrectionsReached,
        .iterations = max_refinement_corrections,
        .final_relative_residual = last_rel_resid,
    };
}

std::expected<std::vector<double>, GprError> refine(RefineSystem& system, std::vector<double> w) {
    if (w.empty() || system.rhs().empty()) {
        return std::unexpected(GprError::empty_input());
    }
    std::vector<double> residual_scratch(w.size(), 0.0);
    auto summary = refine_in_place(system, w, residual_scratch);
    if (!summary) {
        return std::unexpected(std::move(summary).error());
    }
    return w;
}

} // namespace cxxgpr
