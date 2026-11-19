// SPDX-License-Identifier: Apache-2.0
/// @file mll.cpp
/// @brief Implementations and explicit instantiations for marginal likelihood, gradient, and Hessian evaluation.
/// @details Implements Cholesky-factored negative MLL, single-pass keep_plan gradient evaluation with
/// W = alpha * alpha^T - A^-1, CholeskyBuffer::Reuse factor restoration, and analytical Hessian computation.

#include "cxxgpr/workspace/mll.hpp"

#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace cxxgpr
{

    namespace
    {

        /// @brief Populates a square mutable matrix view with the n x n identity matrix I_n.
        /// @details Writes 1 along the main diagonal and 0 elsewhere prior to solving A * X = I_n.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[out] mat Square mutable n x n column-major matrix view.
        template <KernelScalarConcept T> void fill_identity(MatrixMut<T> mat) noexcept
        {
            const std::size_t n = mat.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = 0; row < n; ++row)
                {
                    mat(row, col) = (row == col) ? T{1} : T{0};
                }
            }
        }

        /// @brief Populates the lower triangle of a square matrix view with diag_val * I_n.
        /// @details Writes diag_val on the diagonal and 0 strictly below it, representing dA/d(log sigma_n^2)
        /// and d^2A/d(log sigma_n^2)^2 = sigma_n^2 * I_n.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[out] mat Square mutable n x n column-major matrix view.
        /// @param[in] diag_val Scalar multiplier applied to main diagonal entries.
        template <KernelScalarConcept T> void fill_diagonal_scaled(MatrixMut<T> mat, T diag_val) noexcept
        {
            const std::size_t n = mat.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    mat(row, col) = (row == col) ? diag_val : T{0};
                }
            }
        }

        /// @brief Forms the lower triangle of W = alpha * alpha^T - A^-1 into w_out.
        /// @details Reads alpha and the lower triangle of a_inv (which may alias w_out under
        /// CholeskyBuffer::Reuse) and writes column-major lower entries in parallel.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] alpha Pre-solved weight vector A^-1 * y (length n).
        /// @param[in] a_inv Immutable view of A^-1 (only lower triangle read).
        /// @param[out] w_out Mutable destination matrix view receiving lower triangle of W.
        template <KernelScalarConcept T>
        void build_w_lower_from_inv(std::span<const T> alpha, MatrixRef<T> a_inv, MatrixMut<T> w_out) noexcept
        {
            const std::size_t n = alpha.size();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const T a_col = alpha[col];
                for (std::size_t row = col; row < n; ++row)
                {
                    w_out(row, col) = (alpha[row] * a_col) - a_inv(row, col);
                }
            }
        }

        /// @brief Computes u = A_sym * alpha reading only the lower triangle of symmetric matrix a_lower.
        /// @details Traverses lower-triangular entries once to accumulate both row and symmetric column
        /// contributions into u without touching the strictly upper triangle of a_lower.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] a_lower Square read-only view whose lower triangle stores symmetric matrix A_i.
        /// @param[in] alpha Input vector of length n.
        /// @param[out] u Mutable output vector of length n receiving A_i * alpha.
        template <KernelScalarConcept T>
        void gemv_sym_lower(MatrixRef<T> a_lower, std::span<const T> alpha, std::span<T> u) noexcept
        {
            const std::size_t n = alpha.size();
            std::ranges::fill(u, T{0});
            for (std::size_t col = 0; col < n; ++col)
            {
                const T a_col = alpha[col];
                u[col] += a_lower(col, col) * a_col;
                T acc = T{0};
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const T v = a_lower(row, col);
                    // Apply lower entry (row, col) and mirrored upper entry (col, row) simultaneously.
                    u[row] += v * a_col;
                    acc += v * alpha[row];
                }
                u[col] += acc;
            }
        }

        /// @brief Transposes a square column-major matrix view in place.
        /// @details Swaps strictly lower and strictly upper entries for col < row.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in,out] mat Square mutable n x n column-major matrix view.
        template <KernelScalarConcept T> void transpose_square_in_place(MatrixMut<T> mat) noexcept
        {
            const std::size_t n = mat.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const std::size_t r1 = row;
                    const std::size_t r2 = col;
                    std::swap(mat(r1, r2), mat(r2, r1));
                }
            }
        }

        /// @brief Computes deterministic symmetric Frobenius inner product <a, b>_F from lower triangles.
        /// @details Evaluates column sums diag + 2 * strictly_lower in double precision in parallel into
        /// col_partials, then folds columns in fixed ascending order for bitwise thread-count invariance.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] a First symmetric n x n matrix view (lower triangle read).
        /// @param[in] b Second symmetric n x n matrix view (lower triangle read).
        /// @param[in,out] col_partials Pre-allocated length-n scratch span for deterministic per-column reduction.
        /// @return Trace(a * b) computed deterministically in double precision.
        template <KernelScalarConcept T>
        [[nodiscard]] double frobenius_lower_det(
            MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials
        ) noexcept
        {
            const std::size_t n = a.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const auto diag = static_cast<double>(a(col, col)) * static_cast<double>(b(col, col));
                double off = 0.0;
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    off += static_cast<double>(a(row, col)) * static_cast<double>(b(row, col));
                }
                col_partials[col] = diag + (2.0 * off);
            }
            // Sum per-column partials sequentially in ascending column order for deterministic OpenMP results.
            double total = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                total += col_partials[col];
            }
            return total;
        }

        /// @brief Computes deterministic full-matrix Frobenius inner product <a, b>_F = tr(a^T * b).
        /// @details Accumulates each column inner product in double precision into col_partials in parallel,
        /// then folds columns sequentially in ascending order for bitwise determinism.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param[in] a First n x n column-major matrix view.
        /// @param[in] b Second n x n column-major matrix view.
        /// @param[in,out] col_partials Pre-allocated length-n scratch span for deterministic per-column reduction.
        /// @return Full Frobenius inner product in double precision.
        template <KernelScalarConcept T>
        [[nodiscard]] double frobenius_full_det(MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials) noexcept
        {
            const std::size_t n = a.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                double sum = 0.0;
                for (std::size_t row = 0; row < n; ++row)
                {
                    sum += static_cast<double>(a(row, col)) * static_cast<double>(b(row, col));
                }
                col_partials[col] = sum;
            }
            // Fold column sums in ascending index order for bitwise thread-count invariance.
            double total = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                total += col_partials[col];
            }
            return total;
        }

        /// @brief Ensures buffers.dist holds any distance or packed ARD tables required by kernel for x.
        /// @details Returns nullptr immediately when buffers.dist is disabled (Uncached policy), skips
        /// recomputation when existing dimensions and cache flags already match x, and populates on demand.
        /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
        /// @param[in] kernel Compiled execution-layer kernel tree queried for distance/ARD needs.
        /// @param[in] x Column-major n x d training feature matrix.
        /// @param[in,out] buffers Training buffers owning optional DistanceCache<Storage>.
        /// @return Pointer to ready DistanceCache, nullptr if uncached, or GprError if ensure() fails.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::NonFiniteInput When x contains non-finite entries.
        /// @retval GprErrorCode::SizeOverflow When distance cache element count overflows std::size_t.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<const DistanceCache<typename P::Storage>*, GprError> prepare_fit_dist_cache(
            const CompiledKernel<typename P::Storage>& kernel, MatrixRef<typename P::Storage> x, FitBuffers<P>& buffers
        )
        {
            if (!buffers.dist.has_value())
            {
                return nullptr;
            }
            const bool need_dist = kernel.reads_distances();
            const bool need_ard = kernel.needs_ard_sq_diff();
            if (!need_dist && !need_ard)
            {
                return &(*buffers.dist);
            }
            // Reuse existing distance cache across optimizer steps whenever shape and required tables match.
            const bool matches = buffers.dist->num_points() == x.rows() && buffers.dist->num_dims() == x.cols()
                && (!need_dist || buffers.dist->has_dist()) && (!need_ard || buffers.dist->has_ard());
            if (!matches)
            {
                if (auto ok = buffers.dist->ensure(x, need_dist, need_ard, DistanceCachePolicy::Cached);
                    !ok.has_value())
                {
                    return std::unexpected(std::move(ok).error());
                }
            }
            return &(*buffers.dist);
        }

        /// @brief Computes the analytical log-space Hessian from an already factored L and solved alpha.
        /// @details Forms W = alpha * alpha^T - A^-1, evaluates second-order curvature terms
        /// -0.5 * <W, d^2A/(dtheta_i dtheta_j)>_F, restores L if CholeskyBuffer::Reuse overwrote k_matrix,
        /// solves v_i = L^-1 A_i alpha and S_i = L^-1 A_i L^-T for each parameter i, and adds
        /// -0.5 * tr(S_i * S_j) + v_i^T * v_j symmetrically into out_hess.
        /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
        /// @param[in] kernel Compiled execution-layer kernel tree.
        /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] y Training target vector (length n), used to restore L under CholeskyBuffer::Reuse.
        /// @param[in] jitter Diagonal jitter policy used if L must be refactored under Reuse.
        /// @param[in,out] buffers Factored FitBuffers owning L, alpha, W buffer, and lazy HessianScratch.
        /// @param[out] out_hess Destination span of length p * p receiving the symmetric analytical Hessian.
        /// @return Empty expected on success, or GprError on solve, kernel derivative, or allocation failure.
        /// @retval GprErrorCode::EmptyInput When dimensions are empty.
        /// @retval GprErrorCode::SizeOverflow When HessianScratch sizing overflows std::size_t.
        /// @retval GprErrorCode::CholeskyFailed When L restoration fails under CholeskyBuffer::Reuse.
        /// @retval GprErrorCode::NonPositiveDefiniteMatrix When L has a non-positive diagonal pivot.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> fill_hessian_from_factor(
            const CompiledKernel<typename P::Storage>& kernel,
            const GaussianLikelihood& likelihood,
            MatrixRef<typename P::Storage> x,
            std::span<const typename P::Storage> y,
            const JitterPolicy& jitter,
            FitBuffers<P>& buffers,
            std::span<double> out_hess
        )
        {
            using Storage = P::Storage;
            const std::size_t n = buffers.core.order();
            const std::size_t n_kernel = kernel.num_params();
            const std::size_t n_params = n_kernel + 1U;

            // Lazily allocate scratch matrices S_i (p x n x n), V (n x p), u, and scratch_c.
            if (auto ok = buffers.core.ensure_kernel_scratch(); !ok.has_value())
            {
                return ok;
            }
            if (auto ok = buffers.core.hessian.ensure(n, n_params); !ok.has_value())
            {
                return ok;
            }
            if (buffers.core.nested.depth() < kernel.nested_depth() || buffers.core.nested.rows() < n
                || buffers.core.nested.cols() < n)
            {
                buffers.core.nested.resize(kernel.nested_depth(), n, n);
            }

            // Compute W = alpha * alpha^T - A^-1; under Reuse this temporarily overwrites k_matrix.
            MatrixMut<Storage> inv_buf = buffers.overwrites_cholesky() ? buffers.core.exp_mut() : buffers.w_mut();
            fill_identity(inv_buf);
            if (auto inv_ok = linalg::solve_llt_in_place(buffers.core.k_ref(), inv_buf); !inv_ok.has_value())
            {
                return inv_ok;
            }
            build_w_lower_from_inv<Storage>(buffers.core.rhs, inv_buf.as_ref(), buffers.w_mut());

            const DistanceCache<Storage>* dist_ptr =
                (buffers.dist.has_value() && (buffers.dist->has_dist() || buffers.dist->has_ard())) ? &(*buffers.dist)
                                                                                                    : nullptr;

            std::ranges::fill(out_hess, 0.0);
            const auto noise = static_cast<Storage>(likelihood.noise_variance());

            // Second-order contribution: -0.5 * <W, d^2 A / (dtheta_i dtheta_j)>_F
            for (std::size_t i = 0; i < n_params; ++i)
            {
                for (std::size_t j = i; j < n_params; ++j)
                {
                    if (i < n_kernel && j < n_kernel)
                    {
                        std::expected<void, GprError> hres;
                        if (dist_ptr != nullptr)
                        {
                            hres = kernel.hess_cached(
                                x, *dist_ptr, buffers.core.exp_mut(), i, j, Triangle::Lower,
                                buffers.core.kernel_scratch_mut(), buffers.core.hessian.scratch_c_mut(),
                                buffers.core.hessian.s_mut(0U), buffers.core.nested
                            );
                        }
                        else
                        {
                            hres = kernel.hess_points(
                                x, buffers.core.exp_mut(), i, j, Triangle::Lower, buffers.core.kernel_scratch_mut(),
                                buffers.core.hessian.scratch_c_mut(), buffers.core.hessian.s_mut(0U),
                                buffers.core.nested
                            );
                        }
                        if (!hres.has_value())
                        {
                            return hres;
                        }
                    }
                    else if (i == n_kernel && j == n_kernel)
                    {
                        // d^2(sigma_n^2 I) / d(log sigma_n^2)^2 = sigma_n^2 * I; cross-terms with kernel vanish.
                        fill_diagonal_scaled(buffers.core.exp_mut(), noise);
                    }
                    else
                    {
                        continue;
                    }
                    const double inner = frobenius_lower_det(
                        buffers.w_ref(), buffers.core.exp_ref(), std::span<double>(buffers.core.col_partials)
                    );
                    const double hij = -0.5 * inner;
                    out_hess[(i * n_params) + j] = hij;
                    out_hess[(j * n_params) + i] = hij;
                }
            }

            // Under CholeskyBuffer::Reuse, W overwrote k_matrix; refactor L before solving S_i and v_i.
            if (buffers.overwrites_cholesky())
            {
                auto restore_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, 0U);
                if (!restore_res.has_value())
                {
                    return std::unexpected(std::move(restore_res).error());
                }
            }

            // Compute S_i = L^-1 * A_i * L^-T and v_i = L^-1 * A_i * alpha for each parameter i.
            const auto l_ref = buffers.core.k_ref();
            const std::span<const Storage> alpha(buffers.core.rhs);
            for (std::size_t i = 0; i < n_params; ++i)
            {
                if (i < n_kernel)
                {
                    std::expected<void, GprError> gres;
                    if (dist_ptr != nullptr)
                    {
                        gres = kernel.grad_cached(
                            x, *dist_ptr, buffers.core.exp_mut(), i, Triangle::Lower, buffers.core.kernel_scratch_mut(),
                            buffers.core.hessian.scratch_c_mut(), buffers.core.nested
                        );
                    }
                    else
                    {
                        gres = kernel.grad_points(
                            x, buffers.core.exp_mut(), i, Triangle::Lower, buffers.core.kernel_scratch_mut(),
                            buffers.core.hessian.scratch_c_mut(), buffers.core.nested
                        );
                    }
                    if (!gres.has_value())
                    {
                        return gres;
                    }
                }
                else
                {
                    fill_diagonal_scaled(buffers.core.exp_mut(), noise);
                }

                // Solve v_i = L^-1 * (A_i * alpha) using symmetric lower-triangle matvec.
                const auto d_a = buffers.core.exp_ref();
                gemv_sym_lower<Storage>(d_a, alpha, std::span<Storage>(buffers.core.hessian.u));
                auto v_i = buffers.core.hessian.v_col_mut(i);
                std::ranges::copy(buffers.core.hessian.u, v_i.begin());
                if (auto sol_v = linalg::solve_lower_in_place(l_ref, v_i); !sol_v.has_value())
                {
                    return sol_v;
                }

                // Symmetrize A_i into S_i, solve L * Z = A_i, transpose Z, and solve L * S_i = Z^T.
                auto s_i = buffers.core.hessian.s_mut(i);
                for (std::size_t col = 0; col < n; ++col)
                {
                    s_i(col, col) = d_a(col, col);
                    for (std::size_t row = col + 1U; row < n; ++row)
                    {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const Storage val = d_a(r1, r2);
                        s_i(r1, r2) = val;
                        s_i(r2, r1) = val;
                    }
                }
                if (auto s1 = linalg::solve_lower_in_place(l_ref, s_i); !s1.has_value())
                {
                    return s1;
                }
                transpose_square_in_place(s_i);
                if (auto s2 = linalg::solve_lower_in_place(l_ref, s_i); !s2.has_value())
                {
                    return s2;
                }
            }

            // Add -0.5 * tr(S_i * S_j) + v_i^T * v_j symmetrically across all parameter pairs (i, j).
            for (std::size_t i = 0; i < n_params; ++i)
            {
                for (std::size_t j = 0; j <= i; ++j)
                {
                    const double tr = frobenius_full_det(
                        buffers.core.hessian.s_ref(i), buffers.core.hessian.s_ref(j),
                        std::span<double>(buffers.core.col_partials)
                    );
                    auto vdot =
                        linalg::dot<Storage>(buffers.core.hessian.v_col_ref(i), buffers.core.hessian.v_col_ref(j));
                    if (!vdot.has_value())
                    {
                        return std::unexpected(std::move(vdot).error());
                    }
                    const double add = (-0.5 * tr) + *vdot;
                    out_hess[(i * n_params) + j] += add;
                    if (i != j)
                    {
                        out_hess[(j * n_params) + i] += add;
                    }
                }
            }

            return {};
        }

    }  // namespace

    /// @brief Assembles A = K + sigma_n^2 I, factors L, solves alpha, and returns the negative MLL.
    /// @details Evaluates K into k_matrix (keeping keep_products factor Grams in buffers.core.weighted),
    /// adds likelihood.noise_variance() to the diagonal, performs jittered lower Cholesky decomposition,
    /// solves L * L^T * alpha = y into buffers.core.rhs, and returns 0.5 * (y^T * alpha + log|A| + n * log(2*pi)).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers receiving L, alpha, factor_jitter, and kept factor Grams.
    /// @param[in] keep_products Number of top-level product terms whose factor Grams are kept for gradients.
    /// @return Negative marginal log-likelihood, or GprError on invalid input or factorization failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When x.rows() or y.size() differs from buffers.core.order().
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When distance cache or workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization fails across the full jitter ladder.
    template <PrecisionPolicyConcept P>
    std::expected<double, GprError> factor_and_mll(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::size_t keep_products
    )
    {
        using Storage = P::Storage;
        if (x.is_empty() || x.data() == nullptr || y.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = buffers.core.order();
        if (x.rows() != n || y.size() != n)
        {
            return std::unexpected(GprError::length_mismatch("training sample count mismatch"));
        }
        for (const Storage val : y)
        {
            if (!std::isfinite(val))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }

        // Prepare cached distances/ARD differences if policy is Cached, and size tree/kept buffers.
        auto dist_ptr = prepare_fit_dist_cache(kernel, x, buffers);
        if (!dist_ptr.has_value())
        {
            return std::unexpected(std::move(dist_ptr).error());
        }

        if (buffers.core.nested.depth() < kernel.nested_depth() || buffers.core.nested.rows() < n
            || buffers.core.nested.cols() < n)
        {
            buffers.core.nested.resize(kernel.nested_depth(), n, n);
        }

        const std::size_t kept_count = kernel.kept_buffers(keep_products);
        if (kept_count > 0U)
        {
            if (auto ok = buffers.core.ensure_weighted(kernel.weighted_buffers(keep_products)); !ok.has_value())
            {
                return std::unexpected(std::move(ok).error());
            }
        }

        // Assemble lower triangle of K while saving factor Grams for keep_products product terms.
        std::expected<void, GprError> eval_res;
        if (*dist_ptr != nullptr && ((*dist_ptr)->has_dist() || (*dist_ptr)->has_ard()))
        {
            eval_res = kernel.apply_cached_keeping(
                x, **dist_ptr, buffers.core.k_mut(), Triangle::Lower, buffers.core.exp_mut(), buffers.core.nested,
                buffers.core.weighted, keep_products
            );
        }
        else
        {
            eval_res = kernel.apply_keeping(
                x, buffers.core.k_mut(), Triangle::Lower, buffers.core.exp_mut(), buffers.core.nested,
                buffers.core.weighted, keep_products
            );
        }
        if (!eval_res.has_value())
        {
            return std::unexpected(std::move(eval_res).error());
        }

        // Form noisy Gram system A = K + sigma_n^2 * I and factor A = L * L^T with jitter retries.
        if (auto diag_ok = linalg::add_to_diag(buffers.core.k_mut(), static_cast<Storage>(likelihood.noise_variance()));
            !diag_ok.has_value())
        {
            return std::unexpected(std::move(diag_ok).error());
        }

        auto jit_res = linalg::cholesky_lower_with_jitter(
            buffers.core.k_mut(), buffers.core.exp_mut(), jitter, CholeskyStage::Fit
        );
        if (!jit_res.has_value())
        {
            return std::unexpected(std::move(jit_res).error());
        }
        buffers.core.factor_jitter = *jit_res;

        // Solve (L * L^T) * alpha = y in buffers.core.rhs and compute 0.5 * (y^T * alpha + log|A| + n * log(2*pi)).
        std::ranges::copy(y, buffers.core.rhs.begin());
        if (auto sol_ok = linalg::solve_llt_in_place(buffers.core.k_ref(), std::span<Storage>(buffers.core.rhs));
            !sol_ok.has_value())
        {
            return std::unexpected(std::move(sol_ok).error());
        }

        auto dot_res = linalg::dot<Storage>(y, std::span<const Storage>(buffers.core.rhs));
        if (!dot_res.has_value())
        {
            return std::unexpected(std::move(dot_res).error());
        }
        auto logdet_res = linalg::log_det_from_l(buffers.core.k_ref());
        if (!logdet_res.has_value())
        {
            return std::unexpected(std::move(logdet_res).error());
        }
        const auto n_dbl = static_cast<double>(n);
        return 0.5 * (*dot_res + *logdet_res + (n_dbl * std::log(2.0 * std::numbers::pi)));
    }

    /// @brief Computes negative MLL and analytical log-space gradients in a single pass via keep_plan.
    /// @details Factors A with factor-Gram retention enabled, constructs W = alpha * alpha^T - A^-1 in
    /// buffers.w_mut(), runs single-pass weighted_grads() scaled by -0.5, computes the log-noise gradient
    /// -0.5 * sigma_n^2 * tr(W), and restores L if CholeskyBuffer::Reuse overwrote k_matrix.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers receiving updated L, alpha, W, and scratch state.
    /// @param[out] out_grad Destination span of length kernel.num_params() + 1 receiving log-space gradients.
    /// @return Negative marginal log-likelihood, or GprError on invalid inputs, size mismatch, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_grad.size() != kernel.num_params() + 1 or n mismatches.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When workspace or cache sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization (or CholeskyBuffer::Reuse restoration) fails.
    template <PrecisionPolicyConcept P>
    std::expected<double, GprError> mll_and_gradient_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::span<double> out_grad
    )
    {
        using Storage = P::Storage;
        const std::size_t n_kernel = kernel.num_params();
        if (out_grad.size() != n_kernel + 1U)
        {
            return std::unexpected(GprError::length_mismatch("gradient output length mismatch"));
        }

        // Ensure keep_plan walk buffers and leaf gradient scratch are allocated before factoring.
        const std::size_t products = kernel.kept_products();
        if (auto ok = buffers.core.ensure_weighted(kernel.weighted_buffers(products)); !ok.has_value())
        {
            return std::unexpected(std::move(ok).error());
        }
        if (auto ok = buffers.core.ensure_kernel_scratch(); !ok.has_value())
        {
            return std::unexpected(std::move(ok).error());
        }

        auto mll_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, products);
        if (!mll_res.has_value())
        {
            return mll_res;
        }

        // Solve A^-1 and form W = alpha * alpha^T - A^-1 (aliasing k_matrix when CholeskyBuffer::Reuse).
        const std::size_t n = buffers.core.order();
        MatrixMut<Storage> inv_buf = buffers.overwrites_cholesky() ? buffers.core.exp_mut() : buffers.w_mut();
        fill_identity(inv_buf);
        if (auto inv_ok = linalg::solve_llt_in_place(buffers.core.k_ref(), inv_buf); !inv_ok.has_value())
        {
            return std::unexpected(std::move(inv_ok).error());
        }
        build_w_lower_from_inv<Storage>(buffers.core.rhs, inv_buf.as_ref(), buffers.w_mut());

        const DistanceCache<Storage>* dist_ptr =
            (buffers.dist.has_value() && (buffers.dist->has_dist() || buffers.dist->has_ard())) ? &(*buffers.dist)
                                                                                                : nullptr;

        // Contract <W, dK/dtheta_p>_F across all kernel parameters in a single tree walk.
        if (auto wg_ok = kernel.weighted_grads(
                x, dist_ptr, buffers.w_ref(), out_grad.subspan(0, n_kernel), buffers.core.kernel_scratch_mut(),
                buffers.core.nested, buffers.core.weighted, products, std::span<double>(buffers.core.col_partials)
            );
            !wg_ok.has_value())
        {
            return std::unexpected(std::move(wg_ok).error());
        }
        for (std::size_t p = 0; p < n_kernel; ++p)
        {
            out_grad[p] *= -0.5;
        }

        // Compute d(-MLL)/d(log sigma_n^2) = -0.5 * tr(W * sigma_n^2 * I) = -0.5 * sigma_n^2 * tr(W).
        const auto w_ref = buffers.w_ref();
        double trace_w = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            trace_w += static_cast<double>(w_ref(i, i));
        }
        out_grad[n_kernel] = -0.5 * likelihood.noise_variance() * trace_w;

        // Restore L in k_matrix when Reuse mode overwrote it with W.
        if (buffers.overwrites_cholesky())
        {
            auto restore_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, 0U);
            if (!restore_res.has_value())
            {
                return std::unexpected(std::move(restore_res).error());
            }
        }

        return *mll_res;
    }

    /// @brief Computes the analytical log-space Hessian matrix of the negative marginal log-likelihood.
    /// @details Verifies out_hess length p * p, runs factor_and_mll(), and delegates to fill_hessian_from_factor().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers (core.hessian is lazily allocated on first call).
    /// @param[out] out_hess Destination span of length p * p receiving the symmetric log-space Hessian.
    /// @return Empty expected on success, or GprError on invalid inputs, size mismatch, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_hess.size() != p * p or n mismatches.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When HessianScratch or distance cache sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization or restoration fails.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> hessian_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::span<double> out_hess
    )
    {
        const std::size_t n_params = kernel.num_params() + 1U;
        if (out_hess.size() != n_params * n_params)
        {
            return std::unexpected(GprError::length_mismatch("hessian output length mismatch"));
        }
        auto mll_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, 0U);
        if (!mll_res.has_value())
        {
            return std::unexpected(std::move(mll_res).error());
        }
        return fill_hessian_from_factor(kernel, likelihood, x, y, jitter, buffers, out_hess);
    }

    /// @brief Computes negative MLL, analytical gradient, and analytical Hessian together in one pass.
    /// @details Verifies out_hess length p * p, runs mll_and_gradient_into() (which leaves L intact even
    /// under CholeskyBuffer::Reuse), and evaluates out_hess via fill_hessian_from_factor().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x Column-major n x d training coordinate matrix in storage precision.
    /// @param[in] y Training target vector in storage precision (length n).
    /// @param[in] jitter Diagonal jitter escalation policy for Cholesky factorization.
    /// @param[in,out] buffers Pre-allocated FitBuffers updated with factor state and Hessian scratch.
    /// @param[out] out_grad Destination span of length p = kernel.num_params() + 1 receiving log-space gradients.
    /// @param[out] out_hess Destination span of length p * p receiving the symmetric log-space Hessian.
    /// @return Negative marginal log-likelihood, or GprError on invalid inputs, size mismatch, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x or y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_grad or out_hess length disagrees with p or n mismatches.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD kernel leaf.
    /// @retval GprErrorCode::SizeOverflow When HessianScratch or distance cache sizing overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When factorization or restoration fails.
    template <PrecisionPolicyConcept P>
    std::expected<double, GprError> value_gradient_hessian_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x,
        std::span<const typename P::Storage> y,
        const JitterPolicy& jitter,
        FitBuffers<P>& buffers,
        std::span<double> out_grad,
        std::span<double> out_hess
    )
    {
        const std::size_t n_params = kernel.num_params() + 1U;
        if (out_hess.size() != n_params * n_params)
        {
            return std::unexpected(GprError::length_mismatch("hessian output length mismatch"));
        }
        auto mll_res = mll_and_gradient_into(kernel, likelihood, x, y, jitter, buffers, out_grad);
        if (!mll_res.has_value())
        {
            return mll_res;
        }
        if (auto h_ok = fill_hessian_from_factor(kernel, likelihood, x, y, jitter, buffers, out_hess);
            !h_ok.has_value())
        {
            return std::unexpected(std::move(h_ok).error());
        }
        return *mll_res;
    }

    template std::expected<double, GprError> factor_and_mll<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::size_t
    );
    template std::expected<double, GprError> factor_and_mll<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::size_t
    );
    template std::expected<double, GprError> factor_and_mll<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::size_t
    );
    template std::expected<double, GprError> factor_and_mll<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::size_t
    );

    template std::expected<double, GprError> mll_and_gradient_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::span<double>
    );
    template std::expected<double, GprError> mll_and_gradient_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::span<double>
    );
    template std::expected<double, GprError> mll_and_gradient_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::span<double>
    );
    template std::expected<double, GprError> mll_and_gradient_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::span<double>
    );

    template std::expected<void, GprError> hessian_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::span<double>
    );
    template std::expected<void, GprError> hessian_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::span<double>
    );
    template std::expected<void, GprError> hessian_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::span<double>
    );
    template std::expected<void, GprError> hessian_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::span<double>
    );

    template std::expected<double, GprError> value_gradient_hessian_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        const JitterPolicy&,
        FitBuffers<DoublePrecision>&,
        std::span<double>,
        std::span<double>
    );
    template std::expected<double, GprError> value_gradient_hessian_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<SinglePrecision>&,
        std::span<double>,
        std::span<double>
    );
    template std::expected<double, GprError> value_gradient_hessian_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<PromoteStorage>>&,
        std::span<double>,
        std::span<double>
    );
    template std::expected<double, GprError> value_gradient_hessian_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        const JitterPolicy&,
        FitBuffers<MixedPrecision<ReevaluateKernel>>&,
        std::span<double>,
        std::span<double>
    );

}  // namespace cxxgpr
