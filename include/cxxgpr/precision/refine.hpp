// SPDX-License-Identifier: Apache-2.0
/// @file refine.hpp
/// @brief Mixed-precision iterative refinement loop, residual evaluation, and acceptance interface.
/// @details Defines stopping criteria, promoted float-to-double residual computation, matrix
/// symmetrization, and the RefineSystem abstraction driving mixed-precision solve refinement and
/// double-precision fallback.

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

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Maximum number of correction steps attempted before escalating or falling back.
    inline constexpr std::size_t max_refinement_corrections = 10;

    /// @brief Maximum matrix-vector products allowed in restarted flexible GMRES iterative refinement.
    inline constexpr std::size_t max_krylov_iterations = 100;

    /// @brief Maximum subspace dimension per restart cycle in flexible GMRES iterative refinement.
    inline constexpr std::size_t krylov_restart_dim = 25;

    /// @brief Residual norm ratio threshold above which a step counts toward refinement stagnation.
    inline constexpr double refinement_stagnation_ratio = 0.9;

    /// @brief Number of consecutive stagnating steps that trigger an immediate fallback solve.
    inline constexpr std::size_t refinement_stagnation_steps = 2;

    /// @brief Number of matrix columns processed per block when re-evaluating double-precision kernels.
    inline constexpr std::size_t refinement_column_block = 32;

    /// @brief Returns the stopping tolerance 10 * dim * eps_64 for mixed-precision iterative refinement.
    /// @details Scales machine epsilon linearly with the system dimension dim.
    /// @param[in] dim Linear system dimension n.
    /// @return Convergence tolerance on the relative infinity-norm residual.
    [[nodiscard]] constexpr double refinement_tolerance(std::size_t dim) noexcept
    {
        return 10.0 * static_cast<double>(dim) * std::numeric_limits<double>::epsilon();
    }

    /// @brief Returns the infinity norm max_i |values[i]|, or 0.0 when empty.
    /// @details Propagates NaN immediately if any entry is NaN.
    /// @param[in] values Input scalar vector span.
    /// @return Maximum absolute entry, 0.0 if empty, or NaN if any entry is NaN.
    [[nodiscard]] double inf_norm(std::span<const double> values) noexcept;

    /// @brief Copies the strictly lower triangle of a square matrix view into its upper triangle in-place.
    /// @details No-ops if matrix is empty, null, or non-square; writes matrix(col, row) = matrix(row, col)
    /// for all row > col.
    /// @tparam T Kernel storage scalar type (float or double).
    /// @param[in,out] matrix Mutable column-major square matrix view to symmetrize.
    template <KernelScalarConcept T> void symmetrize_lower(MatrixMut<T> matrix) noexcept
    {
        // Guard against empty, null, or non-square views before indexing across triangles.
        if (matrix.is_empty() || matrix.data() == nullptr || !matrix.is_square())
        {
            return;
        }
        const std::size_t n = matrix.rows();
        // Mirror each column's subdiagonal entries into the symmetric upper-triangular row.
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col + 1; row < n; ++row)
            {
                matrix(col, row) = matrix(row, col);
            }
        }
    }

    /// @brief Evaluates whether ||residual||_inf / (b_inf * ||w||_inf + ||b||_inf) < tol with denom > 0.
    /// @details Rejects non-finite operands, non-positive tolerances, and zero or non-finite denominators.
    /// @param[in] b_inf Operator infinity norm ||B||_inf of the system matrix.
    /// @param[in] w Current iterate vector.
    /// @param[in] b Right-hand side vector.
    /// @param[in] residual Residual vector r = b - B * w.
    /// @param[in] tol Relative residual convergence threshold.
    /// @return True iff the scaled relative residual is finite and strictly below tol.
    [[nodiscard]] bool meets_relative_residual(
        double b_inf, std::span<const double> w, std::span<const double> b, std::span<const double> residual, double tol
    ) noexcept;

    /// @brief Computes residual = y - a_f32 * alpha and ||a_f32||_inf using caller-supplied scratch spans.
    /// @details Accumulates row sums and row absolute sums in deterministic ascending column order
    /// with zero heap allocation.
    /// @param[in] a_f32 Square n x n float system matrix in column-major order.
    /// @param[in] alpha Current double iterate of length n.
    /// @param[in] y Double right-hand side vector of length n.
    /// @param[out] residual Output residual span of length n receiving y - a_f32 * alpha.
    /// @param[in,out] row_abs_scratch Caller-owned scratch buffer of length >= n for row absolute sums.
    /// @param[in,out] sum_scratch Caller-owned scratch buffer of length >= n for matrix-vector products.
    /// @return Matrix infinity norm ||a_f32||_inf on success, or GprError on invalid shapes/scratch.
    /// @retval GprErrorCode::EmptyInput When any matrix or vector span is empty or a_f32.data() is null.
    /// @retval GprErrorCode::LengthMismatch When matrix or vector sizes disagree with n = y.size().
    /// @retval GprErrorCode::WorkspaceTooSmall When row_abs_scratch or sum_scratch has size < n.
    [[nodiscard]] std::expected<double, GprError> promoted_residual_into(
        MatrixRef<float> a_f32,
        std::span<const double> alpha,
        std::span<const double> y,
        std::span<double> residual,
        std::span<double> row_abs_scratch,
        std::span<double> sum_scratch
    );

    /// @brief Computes residual = y - a_f32 * alpha and ||a_f32||_inf with temporary internal scratch.
    /// @details Allocates temporary row_abs and sum scratch vectors and delegates to promoted_residual_into().
    /// @param[in] a_f32 Square n x n float system matrix in column-major order.
    /// @param[in] alpha Current double iterate of length n.
    /// @param[in] y Double right-hand side vector of length n.
    /// @param[out] residual Output residual span of length n receiving y - a_f32 * alpha.
    /// @return Matrix infinity norm ||a_f32||_inf on success, or GprError on invalid shapes.
    /// @retval GprErrorCode::EmptyInput When any matrix or vector span is empty or a_f32.data() is null.
    /// @retval GprErrorCode::LengthMismatch When matrix or vector sizes disagree with n = y.size().
    [[nodiscard]] std::expected<double, GprError> promoted_residual(
        MatrixRef<float> a_f32, std::span<const double> alpha, std::span<const double> y, std::span<double> residual
    );

    /// @brief Outcome of running the mixed-precision iterative refinement loop.
    /// @details Distinguishes clean convergence from the three fallback triggers.
    enum class RefinementStatus : std::uint8_t
    {
        /// @brief Iterate met the relative residual tolerance and passed accept().
        Converged,
        /// @brief Iterate met residual tolerance, but accept() rejected it and triggered fallback.
        RejectedByAccept,
        /// @brief Residual failed to shrink sufficiently for consecutive steps, triggering fallback.
        Stagnated,
        /// @brief Exhausted max_refinement_corrections without converging, triggering fallback.
        MaxCorrectionsReached,
    };

    /// @brief Diagnostic summary produced by iterative refinement.
    /// @details Records termination status, completed correction count, last relative residual,
    /// preconditioner shift, and Krylov escalation usage.
    struct RefinementSummary
    {
        /// @brief Final termination reason reported by the refinement loop.
        RefinementStatus status = RefinementStatus::MaxCorrectionsReached;
        /// @brief Number of correction and Krylov steps executed before termination.
        std::size_t iterations = 0;
        /// @brief Last evaluated relative infinity-norm residual prior to termination or fallback.
        double final_relative_residual = 0.0;
        /// @brief Preconditioner-only diagonal shift applied to stabilize the float Cholesky factor (0.0 when
        /// unshifted).
        double preconditioner_shift = 0.0;
        /// @brief True when refinement escalated to restarted double-precision FGMRES-IR.
        bool used_krylov = false;

        /// @brief Returns true when refinement deferred to the double-precision fallback solve.
        /// @details Equivalent to status != RefinementStatus::Converged.
        /// @return True if fallback() was invoked.
        [[nodiscard]] constexpr bool used_fallback() const noexcept { return status != RefinementStatus::Converged; }

        /// @brief Checks value equality between refinement summaries.
        /// @details Compares status, iterations, final_relative_residual, preconditioner_shift, and used_krylov.
        /// @param[in] lhs Left-hand refinement summary.
        /// @param[in] rhs Right-hand refinement summary.
        /// @return True if all fields compare equal.
        [[nodiscard]] friend constexpr bool operator==(
            const RefinementSummary& lhs, const RefinementSummary& rhs
        ) noexcept = default;
    };

    /// @brief Abstract linear system interface refined against double-precision residuals.
    /// @details Decouples the generic iterative refinement loop from concrete storage-promoted or
    /// kernel-reevaluated linear systems and their double-precision fallback solvers.
    class RefineSystem
    {
    public:
        /// @brief Destroys the linear system cleanly through a base pointer.
        /// @details Virtual destructor enabling polymorphic ownership of derived linear systems.
        virtual ~RefineSystem() = default;

        /// @brief Non-copyable to avoid slicing derived system state.
        /// @details Copy construction is deleted on the polymorphic base interface.
        RefineSystem(const RefineSystem&) = delete;
        /// @brief Non-copy-assignable to avoid slicing derived system state.
        /// @details Copy assignment is deleted on the polymorphic base interface.
        /// @return Reference to *this (deleted).
        RefineSystem& operator=(const RefineSystem&) = delete;
        /// @brief Non-movable through the polymorphic base class.
        /// @details Move construction is deleted on the polymorphic base interface.
        RefineSystem(RefineSystem&&) = delete;
        /// @brief Non-move-assignable through the polymorphic base class.
        /// @details Move assignment is deleted on the polymorphic base interface.
        /// @return Reference to *this (deleted).
        RefineSystem& operator=(RefineSystem&&) = delete;

        /// @brief Returns the double-precision right-hand side vector b.
        /// @details Must remain valid for the lifetime of refinement calls on *this.
        /// @return Non-owning span over b of length dim.
        [[nodiscard]] virtual std::span<const double> rhs() const noexcept = 0;

        /// @brief Writes r = b - B * w and returns ||B||_inf.
        /// @details Evaluates the higher-precision residual for stopping and correction steps.
        /// @param[in] w Current double iterate of length dim.
        /// @param[out] r Output residual buffer of length dim.
        /// @return Operator infinity norm ||B||_inf, or GprError if evaluation fails.
        [[nodiscard]] virtual std::expected<double, GprError> residual(
            std::span<const double> w, std::span<double> r
        ) = 0;

        /// @brief Adds the stored lower-precision factor's solve of r into w.
        /// @details Updates w <- w + delta_w where delta_w approximately solves B * delta_w = r.
        /// @param[in] r Current residual vector b - B * w.
        /// @param[in,out] w Iterate vector updated in-place.
        /// @return Empty expected on success, or GprError if the triangular solve fails.
        /// @retval {} Correction succeeded and updated w in place.
        /// @retval GprError When derived input checks or triangular solve fail.
        [[nodiscard]] virtual std::expected<void, GprError> correct(std::span<const double> r, std::span<double> w) = 0;

        /// @brief Validates whether a converged iterate satisfies true higher-precision acceptance checks.
        /// @details Default implementation returns true unconditionally; derived systems may verify
        /// exact double-precision residuals before accepting.
        /// @param[in] w Candidate converged iterate of length dim.
        /// @param[in] tol Target relative residual tolerance.
        /// @return True if w is accepted, false to trigger fallback(), or GprError if verification fails.
        [[nodiscard]] virtual std::expected<bool, GprError> accept(std::span<const double> w, double tol);

        /// @brief Replaces w with the full double-precision fallback solution when refinement stops early.
        /// @details Invoked when accept() returns false, stagnation occurs, or max corrections are reached.
        /// @param[out] w Iterate vector overwritten in-place with the double-precision fallback solve.
        /// @return Empty expected on success, or GprError if fallback factorization or solve fails.
        /// @retval {} Fallback succeeded and overwrote w in place.
        /// @retval GprError When derived fallback factorization or solve fails.
        [[nodiscard]] virtual std::expected<void, GprError> fallback(std::span<double> w) = 0;

    protected:
        /// @brief Default constructs the polymorphic interface for derived linear systems.
        /// @details Protected to prevent direct instantiation of the abstract base class.
        RefineSystem() = default;
    };

    /// @brief Refines an initial iterate w in-place using a caller-owned residual buffer with zero heap allocations.
    /// @details Iterates up to max_refinement_corrections steps, checking convergence via
    /// meets_relative_residual(), validating with system.accept(), and falling back via
    /// system.fallback() on rejection, stagnation, or iteration exhaustion.
    /// @param[in,out] system Polymorphic linear system providing residual, correction, acceptance, and fallback.
    /// @param[in,out] w Initial iterate overwritten in-place with the refined or fallback solution.
    /// @param[in,out] residual_scratch Caller-supplied buffer of length >= w.size() for intermediate residuals.
    /// @return RefinementSummary describing convergence or fallback outcome, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When w or system.rhs() is empty.
    /// @retval GprErrorCode::LengthMismatch When w.size() != system.rhs().size().
    /// @retval GprErrorCode::WorkspaceTooSmall When residual_scratch.size() < w.size().
    [[nodiscard]] std::expected<RefinementSummary, GprError> refine_in_place(
        RefineSystem& system, std::span<double> w, std::span<double> residual_scratch
    );

    /// @brief Convenience wrapper that allocates residual scratch and returns the refined iterate vector.
    /// @details Delegates to refine_in_place() using a temporary residual buffer of size w.size().
    /// @param[in,out] system Polymorphic linear system providing residual, correction, acceptance, and fallback.
    /// @param[in] w Initial iterate vector moved in and refined in-place.
    /// @return Refined (or fallback) solution vector, or GprError if validation or solver steps fail.
    /// @retval GprErrorCode::EmptyInput When w or system.rhs() is empty.
    /// @retval GprErrorCode::LengthMismatch When w.size() != system.rhs().size().
    [[nodiscard]] std::expected<std::vector<double>, GprError> refine(RefineSystem& system, std::vector<double> w);

}  // namespace cxxgpr
