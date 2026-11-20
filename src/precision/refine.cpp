// SPDX-License-Identifier: Apache-2.0
/// @file refine.cpp
/// @brief Implementation of mixed-precision iterative refinement, norms, and promoted residuals.
/// @details Implements infinity-norm helpers, relative residual scaling checks, column-major
/// float-to-double residual computation, and the refine_in_place() loop with stagnation fallback.

#include "cxxgpr/precision/refine.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

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
        /// @param[in] b_inf Matrix infinity norm ||B||_inf.
        /// @param[in] r_inf Residual infinity norm ||r||_inf.
        /// @param[in] w Current iterate vector.
        /// @param[in] b Right-hand side vector.
        /// @param[in] tol Convergence tolerance on the relative residual.
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

        /// @brief Computes `a * b + c` in double precision matching active SIMD FMA contraction.
        /// @param[in] a First multiplicand.
        /// @param[in] b Second multiplicand.
        /// @param[in] c Addend.
        /// @return `a * b + c`.
        [[nodiscard]] inline double refine_scalar_fmadd(double a, double b, double c) noexcept
        {
#if ((defined(__AVX512F__) && defined(__AVX512DQ__)) || defined(__AVX2__)) && (defined(__FMA__) || defined(_MSC_VER))
            return std::fma(a, b, c);
#else
            return c + (a * b);
#endif
        }

        /// @brief Accumulates 4 consecutive float columns into promoted double `row_abs` and `sum` buffers.
        /// @details Converts float entries to binary64 via AVX-512, AVX2+FMA, or SSE2 while preserving strict
        /// ascending column order `j0 -> j1 -> j2 -> j3` per row and cutting scratch loads/stores by 4x.
        /// @param[in] col0 Pointer to float column `j` base.
        /// @param[in] col1 Pointer to float column `j + 1` base.
        /// @param[in] col2 Pointer to float column `j + 2` base.
        /// @param[in] col3 Pointer to float column `j + 3` base.
        /// @param[in] alpha0 Binary64 weight for column `j`.
        /// @param[in] alpha1 Binary64 weight for column `j + 1`.
        /// @param[in] alpha2 Binary64 weight for column `j + 2`.
        /// @param[in] alpha3 Binary64 weight for column `j + 3`.
        /// @param[in,out] row_abs Scratch buffer accumulating row l1 sums in binary64.
        /// @param[in,out] sum Scratch buffer accumulating matrix-vector sums in binary64.
        /// @param[in] n Number of rows in each column.
        inline void accumulate_promoted_4cols_simd(
            const float* col0,
            const float* col1,
            const float* col2,
            const float* col3,
            double alpha0,
            double alpha1,
            double alpha2,
            double alpha3,
            double* row_abs,
            double* sum,
            std::size_t n
        ) noexcept
        {
            std::size_t i = 0U;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512i abs_mask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL);
            const __m512d va0 = _mm512_set1_pd(alpha0);
            const __m512d va1 = _mm512_set1_pd(alpha1);
            const __m512d va2 = _mm512_set1_pd(alpha2);
            const __m512d va3 = _mm512_set1_pd(alpha3);
            for (; i + 8U <= n; i += 8U)
            {
                __m512d rabs = _mm512_loadu_pd(row_abs + i);
                __m512d rsum = _mm512_loadu_pd(sum + i);
                const __m512d x0 = _mm512_cvtps_pd(_mm256_loadu_ps(col0 + i));
                const __m512d x1 = _mm512_cvtps_pd(_mm256_loadu_ps(col1 + i));
                const __m512d x2 = _mm512_cvtps_pd(_mm256_loadu_ps(col2 + i));
                const __m512d x3 = _mm512_cvtps_pd(_mm256_loadu_ps(col3 + i));

                rabs = _mm512_add_pd(rabs, _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x0), abs_mask)));
                rabs = _mm512_add_pd(rabs, _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x1), abs_mask)));
                rabs = _mm512_add_pd(rabs, _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x2), abs_mask)));
                rabs = _mm512_add_pd(rabs, _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x3), abs_mask)));

#    if defined(__FMA__) || defined(_MSC_VER)
                rsum = _mm512_fmadd_pd(x0, va0, rsum);
                rsum = _mm512_fmadd_pd(x1, va1, rsum);
                rsum = _mm512_fmadd_pd(x2, va2, rsum);
                rsum = _mm512_fmadd_pd(x3, va3, rsum);
#    else
                rsum = _mm512_add_pd(rsum, _mm512_mul_pd(x0, va0));
                rsum = _mm512_add_pd(rsum, _mm512_mul_pd(x1, va1));
                rsum = _mm512_add_pd(rsum, _mm512_mul_pd(x2, va2));
                rsum = _mm512_add_pd(rsum, _mm512_mul_pd(x3, va3));
#    endif
                _mm512_storeu_pd(row_abs + i, rabs);
                _mm512_storeu_pd(sum + i, rsum);
            }
#elifdef __AVX2__
            const __m256d abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
            const __m256d va0 = _mm256_set1_pd(alpha0);
            const __m256d va1 = _mm256_set1_pd(alpha1);
            const __m256d va2 = _mm256_set1_pd(alpha2);
            const __m256d va3 = _mm256_set1_pd(alpha3);
            for (; i + 4U <= n; i += 4U)
            {
                __m256d rabs = _mm256_loadu_pd(row_abs + i);
                __m256d rsum = _mm256_loadu_pd(sum + i);
                const __m256d x0 = _mm256_cvtps_pd(_mm_loadu_ps(col0 + i));
                const __m256d x1 = _mm256_cvtps_pd(_mm_loadu_ps(col1 + i));
                const __m256d x2 = _mm256_cvtps_pd(_mm_loadu_ps(col2 + i));
                const __m256d x3 = _mm256_cvtps_pd(_mm_loadu_ps(col3 + i));

                rabs = _mm256_add_pd(rabs, _mm256_and_pd(x0, abs_mask));
                rabs = _mm256_add_pd(rabs, _mm256_and_pd(x1, abs_mask));
                rabs = _mm256_add_pd(rabs, _mm256_and_pd(x2, abs_mask));
                rabs = _mm256_add_pd(rabs, _mm256_and_pd(x3, abs_mask));

#    if defined(__FMA__) || defined(_MSC_VER)
                rsum = _mm256_fmadd_pd(x0, va0, rsum);
                rsum = _mm256_fmadd_pd(x1, va1, rsum);
                rsum = _mm256_fmadd_pd(x2, va2, rsum);
                rsum = _mm256_fmadd_pd(x3, va3, rsum);
#    else
                rsum = _mm256_add_pd(rsum, _mm256_mul_pd(x0, va0));
                rsum = _mm256_add_pd(rsum, _mm256_mul_pd(x1, va1));
                rsum = _mm256_add_pd(rsum, _mm256_mul_pd(x2, va2));
                rsum = _mm256_add_pd(rsum, _mm256_mul_pd(x3, va3));
#    endif
                _mm256_storeu_pd(row_abs + i, rabs);
                _mm256_storeu_pd(sum + i, rsum);
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
            const __m128d va0 = _mm_set1_pd(alpha0);
            const __m128d va1 = _mm_set1_pd(alpha1);
            const __m128d va2 = _mm_set1_pd(alpha2);
            const __m128d va3 = _mm_set1_pd(alpha3);
            for (; i + 2U <= n; i += 2U)
            {
                __m128d rabs = _mm_loadu_pd(row_abs + i);
                __m128d rsum = _mm_loadu_pd(sum + i);
                const __m128d x0 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, col0[i + 1U], col0[i]));
                const __m128d x1 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, col1[i + 1U], col1[i]));
                const __m128d x2 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, col2[i + 1U], col2[i]));
                const __m128d x3 = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, col3[i + 1U], col3[i]));

                rabs = _mm_add_pd(rabs, _mm_and_pd(x0, abs_mask));
                rabs = _mm_add_pd(rabs, _mm_and_pd(x1, abs_mask));
                rabs = _mm_add_pd(rabs, _mm_and_pd(x2, abs_mask));
                rabs = _mm_add_pd(rabs, _mm_and_pd(x3, abs_mask));

                rsum = _mm_add_pd(rsum, _mm_mul_pd(x0, va0));
                rsum = _mm_add_pd(rsum, _mm_mul_pd(x1, va1));
                rsum = _mm_add_pd(rsum, _mm_mul_pd(x2, va2));
                rsum = _mm_add_pd(rsum, _mm_mul_pd(x3, va3));

                _mm_storeu_pd(row_abs + i, rabs);
                _mm_storeu_pd(sum + i, rsum);
            }
#endif
            for (; i < n; ++i)
            {
                const auto x0 = static_cast<double>(col0[i]);
                const auto x1 = static_cast<double>(col1[i]);
                const auto x2 = static_cast<double>(col2[i]);
                const auto x3 = static_cast<double>(col3[i]);
                double rabs = row_abs[i];
                double rsum = sum[i];
                rabs = ((rabs + std::abs(x0)) + std::abs(x1)) + (std::abs(x2) + std::abs(x3));
                rsum = refine_scalar_fmadd(x0, alpha0, rsum);
                rsum = refine_scalar_fmadd(x1, alpha1, rsum);
                rsum = refine_scalar_fmadd(x2, alpha2, rsum);
                rsum = refine_scalar_fmadd(x3, alpha3, rsum);
                row_abs[i] = rabs;
                sum[i] = rsum;
            }
        }

        /// @brief Accumulates one float column into promoted double `row_abs` and `sum` buffers using SIMD.
        /// @param[in] col Pointer to contiguous float column base.
        /// @param[in] alpha_j Binary64 weight for column `j`.
        /// @param[in,out] row_abs Scratch buffer accumulating row l1 sums in binary64.
        /// @param[in,out] sum Scratch buffer accumulating matrix-vector sums in binary64.
        /// @param[in] n Number of rows in `col`.
        inline void accumulate_promoted_1col_simd(
            const float* col, double alpha_j, double* row_abs, double* sum, std::size_t n
        ) noexcept
        {
            std::size_t i = 0U;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512i abs_mask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL);
            const __m512d va = _mm512_set1_pd(alpha_j);
            for (; i + 8U <= n; i += 8U)
            {
                const __m512d x = _mm512_cvtps_pd(_mm256_loadu_ps(col + i));
                const __m512d rabs = _mm512_add_pd(
                    _mm512_loadu_pd(row_abs + i),
                    _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x), abs_mask))
                );
#    if defined(__FMA__) || defined(_MSC_VER)
                const __m512d rsum = _mm512_fmadd_pd(x, va, _mm512_loadu_pd(sum + i));
#    else
                const __m512d rsum = _mm512_add_pd(_mm512_loadu_pd(sum + i), _mm512_mul_pd(x, va));
#    endif
                _mm512_storeu_pd(row_abs + i, rabs);
                _mm512_storeu_pd(sum + i, rsum);
            }
#elifdef __AVX2__
            const __m256d abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
            const __m256d va = _mm256_set1_pd(alpha_j);
            for (; i + 4U <= n; i += 4U)
            {
                const __m256d x = _mm256_cvtps_pd(_mm_loadu_ps(col + i));
                const __m256d rabs = _mm256_add_pd(_mm256_loadu_pd(row_abs + i), _mm256_and_pd(x, abs_mask));
#    if defined(__FMA__) || defined(_MSC_VER)
                const __m256d rsum = _mm256_fmadd_pd(x, va, _mm256_loadu_pd(sum + i));
#    else
                const __m256d rsum = _mm256_add_pd(_mm256_loadu_pd(sum + i), _mm256_mul_pd(x, va));
#    endif
                _mm256_storeu_pd(row_abs + i, rabs);
                _mm256_storeu_pd(sum + i, rsum);
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
            const __m128d va = _mm_set1_pd(alpha_j);
            for (; i + 2U <= n; i += 2U)
            {
                const __m128d x = _mm_cvtps_pd(_mm_set_ps(0.0F, 0.0F, col[i + 1U], col[i]));
                const __m128d rabs = _mm_add_pd(_mm_loadu_pd(row_abs + i), _mm_and_pd(x, abs_mask));
                const __m128d rsum = _mm_add_pd(_mm_loadu_pd(sum + i), _mm_mul_pd(x, va));
                _mm_storeu_pd(row_abs + i, rabs);
                _mm_storeu_pd(sum + i, rsum);
            }
#endif
            for (; i < n; ++i)
            {
                const auto x = static_cast<double>(col[i]);
                row_abs[i] += std::abs(x);
                sum[i] = refine_scalar_fmadd(x, alpha_j, sum[i]);
            }
        }

    }  // namespace

    /// @brief Returns the infinity norm max_i |values[i]|, or 0.0 when empty.
    /// @details Propagates NaN immediately upon encountering any NaN entry.
    /// @param[in] values Input scalar vector span.
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
    /// @param[in] b_inf Operator infinity norm ||B||_inf.
    /// @param[in] w Current iterate vector.
    /// @param[in] b Right-hand side vector.
    /// @param[in] residual Residual vector r = b - B * w.
    /// @param[in] tol Relative residual convergence threshold.
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
    /// @param[in] a_f32 Square n x n float system matrix in column-major order.
    /// @param[in] alpha Current double iterate of length n.
    /// @param[in] y Double right-hand side vector of length n.
    /// @param[out] residual Output residual span of length n.
    /// @param[in,out] row_abs_scratch Caller-owned scratch buffer of length >= n.
    /// @param[in,out] sum_scratch Caller-owned scratch buffer of length >= n.
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

        // Column-major traversal accumulates 4 columns per scratch pass in ascending column order via SIMD.
        std::size_t j = 0U;
        for (; j + 4U <= n; j += 4U)
        {
            accumulate_promoted_4cols_simd(
                a_f32.col(j).data(), a_f32.col(j + 1U).data(), a_f32.col(j + 2U).data(), a_f32.col(j + 3U).data(),
                alpha[j], alpha[j + 1U], alpha[j + 2U], alpha[j + 3U], row_abs_scratch.data(), sum_scratch.data(), n
            );
        }
        for (; j < n; ++j)
        {
            accumulate_promoted_1col_simd(a_f32.col(j).data(), alpha[j], row_abs_scratch.data(), sum_scratch.data(), n);
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
    /// @param[in] a_f32 Square n x n float system matrix in column-major order.
    /// @param[in] alpha Current double iterate of length n.
    /// @param[in] y Double right-hand side vector of length n.
    /// @param[out] residual Output residual span of length n.
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
    /// @param[in,out] system Polymorphic linear system providing residual, correction, acceptance, and fallback.
    /// @param[in,out] w Initial iterate overwritten in-place with the refined or fallback solution.
    /// @param[in,out] residual_scratch Caller-supplied buffer of length >= w.size().
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
    /// @param[in,out] system Polymorphic linear system providing residual, correction, acceptance, and fallback.
    /// @param[in] w Initial iterate vector moved in and updated in-place.
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
