// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <vector>

namespace cxxgpr
{

    /// Maximum number of correction steps attempted before falling back to a full double solve.
    inline constexpr std::size_t max_refinement_corrections = 10;

    /// Residual norm ratio threshold above which a step counts toward refinement stagnation.
    inline constexpr double refinement_stagnation_ratio = 0.9;

    /// Number of consecutive stagnating steps that trigger an immediate fallback solve.
    inline constexpr std::size_t refinement_stagnation_steps = 2;

    /// Number of matrix columns processed per block when re-evaluating double-precision kernels.
    inline constexpr std::size_t refinement_column_block = 32;

    /// Returns the stopping tolerance 10 * dim * eps_64 for mixed-precision iterative refinement.
    [[nodiscard]] constexpr double refinement_tolerance(std::size_t dim) noexcept
    {
        return 10.0 * static_cast<double>(dim) * std::numeric_limits<double>::epsilon();
    }

    /// Returns the infinity norm max_i |values[i]|, or 0.0 when empty.
    [[nodiscard]] double inf_norm(std::span<const double> values) noexcept;

    /// Copies the strictly lower triangle of a square matrix view into its upper triangle in-place.
    template <KernelScalarConcept T> void symmetrize_lower(MatrixMut<T> matrix) noexcept
    {
        if (matrix.is_empty() || matrix.data() == nullptr || !matrix.is_square())
        {
            return;
        }
        const std::size_t n = matrix.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col + 1; row < n; ++row)
            {
                matrix(col, row) = matrix(row, col);
            }
        }
    }

    /// Evaluates whether ||residual||_inf / (b_inf * ||w||_inf + ||b||_inf) < tol with denom > 0.
    [[nodiscard]] bool meets_relative_residual(
        double b_inf, std::span<const double> w, std::span<const double> b, std::span<const double> residual, double tol
    ) noexcept;

    /// Computes residual = y - a_f32 * alpha and ||a_f32||_inf using caller-supplied scratch spans.
    [[nodiscard]] std::expected<double, GprError> promoted_residual_into(
        MatrixRef<float> a_f32,
        std::span<const double> alpha,
        std::span<const double> y,
        std::span<double> residual,
        std::span<double> row_abs_scratch,
        std::span<double> sum_scratch
    );

    /// Computes residual = y - a_f32 * alpha and ||a_f32||_inf with temporary internal scratch.
    [[nodiscard]] std::expected<double, GprError> promoted_residual(
        MatrixRef<float> a_f32, std::span<const double> alpha, std::span<const double> y, std::span<double> residual
    );

    /// Outcome of running the mixed-precision iterative refinement loop.
    enum class RefinementStatus : std::uint8_t
    {
        Converged,
        RejectedByAccept,
        Stagnated,
        MaxCorrectionsReached,
    };

    /// Diagnostic summary produced by iterative refinement.
    struct RefinementSummary
    {
        /// Final termination reason reported by the refinement loop.
        RefinementStatus status = RefinementStatus::MaxCorrectionsReached;
        /// Number of correction steps executed before termination.
        std::size_t iterations = 0;
        /// Last evaluated relative infinity-norm residual prior to termination or fallback.
        double final_relative_residual = 0.0;

        /// Returns true when refinement deferred to the double-precision fallback solve.
        [[nodiscard]] constexpr bool used_fallback() const noexcept { return status != RefinementStatus::Converged; }

        /// Checks value equality between refinement summaries.
        [[nodiscard]] friend constexpr bool operator==(
            const RefinementSummary&, const RefinementSummary&
        ) noexcept = default;
    };

    /// Abstract linear system interface refined against double-precision residuals.
    class RefineSystem
    {
    public:
        /// Destroys the linear system cleanly through a base pointer.
        virtual ~RefineSystem() = default;

        /// Non-copyable to avoid slicing derived system state.
        RefineSystem(const RefineSystem&) = delete;
        /// Non-copy-assignable to avoid slicing derived system state.
        RefineSystem& operator=(const RefineSystem&) = delete;
        /// Non-movable through the polymorphic base class.
        RefineSystem(RefineSystem&&) = delete;
        /// Non-move-assignable through the polymorphic base class.
        RefineSystem& operator=(RefineSystem&&) = delete;

        /// Returns the double-precision right-hand side vector b.
        [[nodiscard]] virtual std::span<const double> rhs() const noexcept = 0;

        /// Writes r = b - B * w and returns ||B||_inf.
        [[nodiscard]] virtual std::expected<double, GprError> residual(
            std::span<const double> w, std::span<double> r
        ) = 0;

        /// Adds the stored lower-precision factor's solve of r into w.
        [[nodiscard]] virtual std::expected<void, GprError> correct(std::span<const double> r, std::span<double> w) = 0;

        /// Validates whether a converged iterate satisfies true higher-precision acceptance checks.
        [[nodiscard]] virtual std::expected<bool, GprError> accept(std::span<const double> w, double tol);

        /// Replaces w with the full double-precision fallback solution when refinement stops early.
        [[nodiscard]] virtual std::expected<void, GprError> fallback(std::span<double> w) = 0;

    protected:
        /// Default constructs the polymorphic interface for derived linear systems.
        RefineSystem() = default;
    };

    /// Refines an initial iterate w in-place using a caller-owned residual buffer with zero heap
    /// allocations.
    [[nodiscard]] std::expected<RefinementSummary, GprError> refine_in_place(
        RefineSystem& system, std::span<double> w, std::span<double> residual_scratch
    );

    /// Convenience wrapper that allocates residual scratch and returns the refined iterate vector.
    [[nodiscard]] std::expected<std::vector<double>, GprError> refine(RefineSystem& system, std::vector<double> w);

}  // namespace cxxgpr
