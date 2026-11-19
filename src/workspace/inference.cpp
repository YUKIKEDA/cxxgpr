// SPDX-License-Identifier: Apache-2.0
/// @file inference.cpp
/// @brief Implementations and explicit instantiations for posterior GPR prediction, covariance, sampling, and LOO.
/// @details Computes posterior means, diagonal/full predictive covariances, deterministic Cholesky
/// posterior draws, and closed-form Leave-One-Out predictions over pre-factored Cholesky models.

#include "cxxgpr/workspace/inference.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/linalg/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace cxxgpr
{
    namespace
    {

        /// @brief Returns the default adaptive jitter policy used when decomposing posterior covariance for sampling.
        /// @details Configures base jitter 1e-12, scale factor 10.0, 10 retry steps, and cap 1e-2.
        /// @return Pre-validated JitterPolicy for sample().
        [[nodiscard]] JitterPolicy default_sample_jitter()
        {
            auto policy = JitterPolicy::adaptive(1e-12, 10.0, 10U, 1e-2);
            return *policy;
        }

        /// @brief Validates training features, Cholesky factor, weight vector alpha, and query features.
        /// @details Enforces non-emptiness, n x n factor shape, alpha length n, feature column parity,
        /// and finiteness of alpha before allocating or mutating workspace buffers.
        /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
        /// @param[in] x_train Column-major n x d training feature matrix view.
        /// @param[in] l_factor Square n x n lower Cholesky factor view.
        /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision.
        /// @param[in] x_query Column-major m x d query feature matrix view.
        /// @return Empty expected on success, or GprError if any check fails.
        /// @retval GprErrorCode::EmptyInput When any input view or alpha span is empty.
        /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
        /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols().
        /// @retval GprErrorCode::NonFiniteInput When alpha contains NaN or +/-Inf.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> validate_predict_inputs(
            MatrixRef<typename P::Storage> x_train,
            MatrixRef<typename P::Storage> l_factor,
            std::span<const typename P::Refine> alpha,
            MatrixRef<typename P::Storage> x_query
        )
        {
            // Enforce validation order: non-empty -> factor & alpha dimensions -> feature dim -> finiteness.
            if (x_train.is_empty() || x_query.is_empty() || l_factor.is_empty() || alpha.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            const std::size_t n = x_train.rows();
            const std::size_t d = x_train.cols();
            if (l_factor.rows() != n || l_factor.cols() != n)
            {
                return std::unexpected(
                    GprError::length_mismatch("Cholesky factor dimension must match training samples")
                );
            }
            if (alpha.size() != n)
            {
                return std::unexpected(GprError::length_mismatch("alpha length must match training sample count"));
            }
            if (x_query.cols() != d)
            {
                return std::unexpected(GprError::dimension_mismatch(x_query.cols(), d));
            }
            for (const auto val : alpha)
            {
                if (!std::isfinite(val))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            return {};
        }

        /// @brief Evaluates K_* = k(X, X_*), forms posterior mean mu_* = K_*^T * alpha, and solves V = L^-1 * K_*.
        /// @details Grows workspace monotonically if needed, computes cross-covariance in Storage precision,
        /// accumulates K_*^T * alpha in Refine precision into out_mean, and overwrites workspace.query_k_star
        /// in place via linalg::solve_lower_in_place().
        /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
        /// @param[in] kernel Compiled execution-layer kernel tree.
        /// @param[in] x_train Column-major n x d training coordinate matrix.
        /// @param[in] l_factor Square n x n lower Cholesky factor L.
        /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision.
        /// @param[in] x_query Column-major m x d query coordinate matrix.
        /// @param[in,out] workspace Reusable query-time workspace storing K_* and then V = L^-1 * K_*.
        /// @param[out] out_mean Output vector resized to m receiving posterior mean mu_*.
        /// @return Empty expected on success, or GprError on invalid inputs, kernel error, or solve failure.
        /// @retval GprErrorCode::EmptyInput When inputs are empty.
        /// @retval GprErrorCode::LengthMismatch When l_factor or alpha sizes disagree with n.
        /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD.
        /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
        /// @retval GprErrorCode::SizeOverflow When workspace dimensions overflow std::size_t.
        /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> compute_mean_and_v_into(
            const CompiledKernel<typename P::Storage>& kernel,
            MatrixRef<typename P::Storage> x_train,
            MatrixRef<typename P::Storage> l_factor,
            std::span<const typename P::Refine> alpha,
            MatrixRef<typename P::Storage> x_query,
            QueryWorkspace<P>& workspace,
            std::vector<typename P::Refine>& out_mean
        )
        {
            using Storage = P::Storage;
            using Refine = P::Refine;

            if (auto valid = validate_predict_inputs<P>(x_train, l_factor, alpha, x_query); !valid.has_value())
            {
                return valid;
            }

            const std::size_t n = x_train.rows();
            const std::size_t m = x_query.rows();
            const std::size_t d = x_train.cols();

            // Grow query workspace once if dimensions exceed current capacity.
            if (auto ens = workspace.ensure(n, m, d, kernel.nested_depth()); !ens.has_value())
            {
                return ens;
            }

            // Evaluate cross-covariance matrix K_* = k(X_train, X_query) into query_k_star.
            MatrixMut<Storage> k_star_mut(workspace.query_k_star.data(), n, m, n);
            MatrixMut<Storage> scratch_mut(workspace.query_scratch.data(), n, m, n);
            if (auto cross =
                    kernel.apply_cross_points(x_train, x_query, k_star_mut, scratch_mut, workspace.query_nested);
                !cross.has_value())
            {
                return cross;
            }

            // Compute posterior mean mu_* = K_*^T * alpha in Refine precision before overwriting K_*.
            out_mean.resize(m);
            const std::size_t n_local = n;
            const std::size_t m_local = m;
            const Storage* k_star_ptr = workspace.query_k_star.data();
            const Refine* alpha_ptr = alpha.data();
            Refine* mean_ptr = out_mean.data();

            const auto m_signed = static_cast<std::ptrdiff_t>(m_local);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_local * m_local) >= 65536U)
#endif
            for (std::ptrdiff_t col_idx = 0; col_idx < m_signed; ++col_idx)
            {
                const auto j = static_cast<std::size_t>(col_idx);
                const Storage* col_ptr = k_star_ptr + j * n_local;
                Refine acc{0};
                for (std::size_t i = 0; i < n_local; ++i)
                {
                    acc += static_cast<Refine>(col_ptr[i]) * alpha_ptr[i];
                }
                mean_ptr[j] = acc;
            }

            // Overwrite K_* in place with V = L^-1 * K_* for subsequent variance/covariance projection.
            if (auto solve = linalg::solve_lower_in_place(l_factor, k_star_mut); !solve.has_value())
            {
                return solve;
            }

            return {};
        }

    }  // namespace

    /// @brief Computes posterior mean and diagonal predictive variance into out using pre-allocated buffers.
    /// @details Calls compute_mean_and_v_into() to obtain mu_* and V = L^-1 * K_*, evaluates k(x*_j, x*_j)
    /// into workspace.query_kss without heap allocation, and forms max(0, k_** - ||v_j||_2^2) (+ sigma_n^2).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation variance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[out] out Destination Prediction structure populated with length-m mean and variance vectors.
    /// @return Empty expected on success, or GprError on invalid inputs, dimensions, or triangular solve failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> predict_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        Prediction<typename P::Refine>& out
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        if (auto mv = compute_mean_and_v_into<P>(kernel, x_train, l_factor, alpha, x_query, workspace, out.mean);
            !mv.has_value())
        {
            return mv;
        }

        // Evaluate prior self-variances k(x*_j, x*_j) using pre-allocated diagonal scratch.
        const std::size_t n = x_train.rows();
        const std::size_t m = x_query.rows();
        std::span<Storage> kss_span(workspace.query_kss.data(), m);
        if (auto diag = kernel.fill_diag(x_query, kss_span, workspace.query_diag_scratch); !diag.has_value())
        {
            return diag;
        }

        out.variance.resize(m);
        out.variance_kind = options.variance_kind;

        // Add observation noise variance only when predicting noisy observations rather than latent f*.
        const Refine noise_var = (options.variance_kind == VarianceKind::Observation)
            ? static_cast<Refine>(likelihood.noise_variance())
            : Refine{0};
        const Storage* v_ptr = workspace.query_k_star.data();
        const Storage* kss_ptr = workspace.query_kss.data();
        Refine* var_ptr = out.variance.data();

        // Form diagonal predictive variance max(0, k(x*_j, x*_j) - ||V_:,j||_2^2) + noise_var.
        const auto m_signed = static_cast<std::ptrdiff_t>(m);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n * m) >= 65536U)
#endif
        for (std::ptrdiff_t col_idx = 0; col_idx < m_signed; ++col_idx)
        {
            const auto j = static_cast<std::size_t>(col_idx);
            const Storage* v_col = v_ptr + j * n;
            Refine sq_norm{0};
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto val = static_cast<Refine>(v_col[i]);
                sq_norm += val * val;
            }
            const auto prior_var = static_cast<Refine>(kss_ptr[j]);
            const Refine latent_var = std::max(Refine{0}, prior_var - sq_norm);
            var_ptr[j] = latent_var + noise_var;
        }

        return {};
    }

    /// @brief Computes posterior mean and diagonal predictive variance for query points x_query.
    /// @details Allocates a Prediction<P::Refine> instance and delegates to predict_into().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation variance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @return Populated Prediction<P::Refine>, or GprError if predict_into() fails.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    std::expected<Prediction<typename P::Refine>, GprError> predict(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace
    )
    {
        Prediction<typename P::Refine> result;
        if (auto res =
                predict_into<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, options, workspace, result);
            !res.has_value())
        {
            return std::unexpected(std::move(res).error());
        }
        return result;
    }

    /// @brief Computes posterior mean and full m x m column-major posterior covariance matrix into out.
    /// @details Computes mu_* and V = L^-1 * K_*, evaluates the lower triangle of the prior query Gram K_**,
    /// and forms Sigma_* = K_** - V^T * V (+ sigma_n^2 * I) symmetrically into out.covariance.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation covariance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[out] out Destination PredictiveCovariance structure populated with mean and m x m covariance.
    /// @return Empty expected on success, or GprError on invalid inputs, dimensions, or solve failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When m * m or workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> predict_covariance_into(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        PredictiveCovariance<typename P::Refine>& out
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        if (auto mv = compute_mean_and_v_into<P>(kernel, x_train, l_factor, alpha, x_query, workspace, out.mean);
            !mv.has_value())
        {
            return mv;
        }

        const std::size_t n = x_train.rows();
        const std::size_t m = x_query.rows();
        auto mm = checked_mul(m, m);
        if (!mm.has_value())
        {
            return std::unexpected(std::move(mm).error());
        }

        // Evaluate lower triangle of the m x m prior query Gram matrix K_** = k(X*, X*).
        std::vector<Storage> kss(*mm);
        std::vector<Storage> kss_scratch(*mm);
        auto nested_kss = kernel.make_nested_buffers(m, m);
        MatrixMut<Storage> kss_mut(kss.data(), m, m, m);
        MatrixMut<Storage> kss_scratch_mut(kss_scratch.data(), m, m, m);
        if (auto ap = kernel.apply_points(x_query, kss_mut, Triangle::Lower, kss_scratch_mut, nested_kss);
            !ap.has_value())
        {
            return ap;
        }

        out.covariance.resize(*mm);
        out.n_points = m;
        out.variance_kind = options.variance_kind;

        const Refine noise_var = (options.variance_kind == VarianceKind::Observation)
            ? static_cast<Refine>(likelihood.noise_variance())
            : Refine{0};
        const Storage* v_ptr = workspace.query_k_star.data();
        Refine* cov_ptr = out.covariance.data();

        // Compute lower triangle of Sigma_* = K_** - V^T V (+ sigma_n^2 I) column by column.
        for (std::size_t col = 0; col < m; ++col)
        {
            const Storage* v_col = v_ptr + col * n;
            for (std::size_t row = col; row < m; ++row)
            {
                const Storage* v_row = v_ptr + row * n;
                Refine dot_val{0};
                for (std::size_t r = 0; r < n; ++r)
                {
                    dot_val += static_cast<Refine>(v_row[r]) * static_cast<Refine>(v_col[r]);
                }
                Refine entry = static_cast<Refine>(kss[(col * m) + row]) - dot_val;
                if (row == col)
                {
                    entry += noise_var;
                }
                cov_ptr[(col * m) + row] = entry;
            }
        }

        // Mirror strictly lower triangle into strictly upper triangle for symmetric output.
        for (std::size_t col = 0; col < m; ++col)
        {
            for (std::size_t row = 0; row < col; ++row)
            {
                cov_ptr[(col * m) + row] = cov_ptr[(row * m) + col];
            }
        }

        return {};
    }

    /// @brief Computes posterior mean and full m x m column-major posterior covariance matrix.
    /// @details Allocates a PredictiveCovariance<P::Refine> result and delegates to predict_covariance_into().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] options Prediction configuration selecting latent or observation covariance.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @return Populated PredictiveCovariance<P::Refine>, or GprError if predict_covariance_into() fails.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When m * m or workspace sizing overflows std::size_t.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor has a non-positive diagonal pivot.
    template <PrecisionPolicyConcept P>
    std::expected<PredictiveCovariance<typename P::Refine>, GprError> predict_covariance(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace
    )
    {
        PredictiveCovariance<typename P::Refine> result;
        if (auto res = predict_covariance_into<P>(
                kernel, likelihood, x_train, l_factor, alpha, x_query, options, workspace, result
            );
            !res.has_value())
        {
            return std::unexpected(std::move(res).error());
        }
        return result;
    }

    /// @brief Draws n_draws deterministic posterior samples from the predictive distribution.
    /// @details Obtains posterior mean and covariance, decomposes Sigma_* = L_* * L_*^T with jitter
    /// retries, and transforms Box-Muller standard normal draws z ~ N(0, I) via mu_* + L_* * z.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree in storage precision.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] x_train Column-major n x d training feature matrix in storage precision.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] x_query Column-major m x d query feature matrix in storage precision.
    /// @param[in] n_draws Number of posterior draws to generate.
    /// @param[in] seed Deterministic 64-bit seed passed to Xoshiro256PlusPlus.
    /// @param[in] options Prediction configuration selecting latent or observation distribution.
    /// @param[in,out] workspace Reusable query-time prediction workspace grown on demand.
    /// @param[in] jitter Optional Cholesky jitter policy for Sigma_*; defaults to default_sample_jitter().
    /// @return Flat vector of length n_draws * m packed as draw * m + i, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When x_train, x_query, l_factor, or alpha is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is not n x n or alpha.size() != n.
    /// @retval GprErrorCode::DimensionMismatch When x_query.cols() != x_train.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When alpha or coordinates contain NaN or +/-Inf.
    /// @retval GprErrorCode::SizeOverflow When n_draws * m or m * m overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When Sigma_* factorization fails across the jitter ladder.
    template <PrecisionPolicyConcept P>
    std::expected<std::vector<typename P::Refine>, GprError> sample(
        const CompiledKernel<typename P::Storage>& kernel,
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        MatrixRef<typename P::Storage> x_query,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        std::optional<JitterPolicy> jitter
    )
    {
        using Refine = P::Refine;

        // Validate inputs and compute posterior mean and covariance before checking n_draws == 0.
        auto cov_res = predict_covariance<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, options, workspace);
        if (!cov_res.has_value())
        {
            return std::unexpected(std::move(cov_res).error());
        }
        if (n_draws == 0U)
        {
            return std::vector<Refine>{};
        }

        const std::size_t m = cov_res->n_points;
        auto total_len = checked_mul(n_draws, m);
        if (!total_len.has_value())
        {
            return std::unexpected(std::move(total_len).error());
        }

        // Factor Sigma_* = L_* * L_*^T using adaptive jitter to handle near-singular query covariances.
        std::vector<Refine> backup(cov_res->covariance.size());
        MatrixMut<Refine> cov_mut(cov_res->covariance.data(), m, m, m);
        MatrixMut<Refine> backup_mut(backup.data(), m, m, m);
        const JitterPolicy eff_jitter = jitter.value_or(default_sample_jitter());
        if (auto chol = linalg::cholesky_lower_with_jitter(cov_mut, backup_mut, eff_jitter, CholeskyStage::Predict);
            !chol.has_value())
        {
            return std::unexpected(std::move(chol).error());
        }

        // Generate each posterior sample y^(s) = mu_* + L_* * z^(s) from deterministic Box-Muller normals.
        std::vector<Refine> samples(*total_len);
        std::vector<Refine> z(m);
        Xoshiro256PlusPlus rng(seed);
        for (std::size_t draw = 0; draw < n_draws; ++draw)
        {
            for (std::size_t i = 0; i < m; ++i)
            {
                z[i] = static_cast<Refine>(rng.unit_normal());
            }
            Refine* out_draw = samples.data() + draw * m;
            for (std::size_t row = 0; row < m; ++row)
            {
                Refine acc = cov_res->mean[row];
                for (std::size_t col = 0; col <= row; ++col)
                {
                    acc += cov_mut(row, col) * z[col];
                }
                out_draw[row] = acc;
            }
        }

        return samples;
    }

    /// @brief Computes closed-form Leave-One-Out predictive mean and variance (GPML §5.4.2) into out.
    /// @details Evaluates q_ii = [A^-1]_ii via linalg::inv_diag_from_chol_l(), then computes
    /// mu_LOO,i = y_i - alpha_i / q_ii and sigma^2_LOO,i = 1 / q_ii (subtracting sigma_n^2 and
    /// clamping at zero when options.variance_kind == VarianceKind::Latent).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] y_train Transformed training target vector y in refine precision (length n).
    /// @param[in] options Prediction configuration selecting latent or observation LOO variance.
    /// @param[in,out] scratch_n_by_n Caller-provided n x n mutable workspace for L^-1 computation.
    /// @param[in,out] scratch_diag Caller-provided length-n mutable span receiving [A^-1]_ii.
    /// @param[out] out Destination Prediction structure populated with LOO means and variances.
    /// @return Empty expected on success, or GprError on invalid inputs, buffers, or non-positive q_ii.
    /// @retval GprErrorCode::EmptyInput When l_factor, alpha, or y_train is empty.
    /// @retval GprErrorCode::LengthMismatch When l_factor is non-square or alpha/y_train/scratch sizes mismatch.
    /// @retval GprErrorCode::NonFiniteInput When alpha or y_train contains NaN or +/-Inf.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor is singular or yields q_ii <= 0.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> loo_predict_into(
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        std::span<const typename P::Refine> y_train,
        const PredictOptions& options,
        MatrixMut<typename P::Storage> scratch_n_by_n,
        std::span<typename P::Storage> scratch_diag,
        Prediction<typename P::Refine>& out
    )
    {
        using Refine = P::Refine;

        // Validate non-emptiness, square factor geometry, vector lengths, and finiteness upfront.
        if (l_factor.is_empty() || alpha.empty() || y_train.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (!l_factor.is_square())
        {
            return std::unexpected(GprError::length_mismatch("Cholesky factor must be square"));
        }
        const std::size_t n = l_factor.rows();
        if (alpha.size() != n)
        {
            return std::unexpected(GprError::length_mismatch("alpha length must match factor dimension"));
        }
        if (y_train.size() != n)
        {
            return std::unexpected(GprError::length_mismatch("y_train length must match factor dimension"));
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!std::isfinite(alpha[i]) || !std::isfinite(y_train[i]))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }

        // Compute diagonal of A^-1 = (L * L^T)^-1 without allocating heap buffers.
        if (auto inv_res = linalg::inv_diag_from_chol_l(l_factor, scratch_diag, scratch_n_by_n); !inv_res.has_value())
        {
            return inv_res;
        }

        out.mean.resize(n);
        out.variance.resize(n);
        out.variance_kind = options.variance_kind;

        // Evaluate GPML §5.4.2 closed-form LOO mean y_i - alpha_i / [A^-1]_ii and variance 1 / [A^-1]_ii.
        const auto noise_var = static_cast<Refine>(likelihood.noise_variance());
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto q_ii = static_cast<Refine>(scratch_diag[i]);
            if (!std::isfinite(q_ii) || q_ii <= Refine{0})
            {
                return std::unexpected(GprError::non_positive_definite_matrix());
            }
            const Refine inv_q = Refine{1} / q_ii;
            out.mean[i] = y_train[i] - (alpha[i] / q_ii);
            out.variance[i] =
                (options.variance_kind == VarianceKind::Observation) ? inv_q : std::max(Refine{0}, inv_q - noise_var);
        }

        return {};
    }

    /// @brief Computes closed-form Leave-One-Out predictive mean and variance (GPML §5.4.2).
    /// @details Allocates temporary n x n and length-n scratch buffers and delegates to loo_predict_into().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] likelihood Observation Gaussian noise likelihood providing sigma_n^2.
    /// @param[in] l_factor Square n x n lower Cholesky factor L of A = K + sigma_n^2 I.
    /// @param[in] alpha Pre-solved weight vector A^-1 * y in refine precision (length n).
    /// @param[in] y_train Transformed training target vector y in refine precision (length n).
    /// @param[in] options Prediction configuration selecting latent or observation LOO variance.
    /// @return Populated LOO Prediction<P::Refine>, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When l_factor, alpha, or y_train is empty.
    /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l_factor is non-square or alpha/y_train lengths mismatch.
    /// @retval GprErrorCode::NonFiniteInput When alpha or y_train contains NaN or +/-Inf.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When l_factor is singular or yields q_ii <= 0.
    template <PrecisionPolicyConcept P>
    std::expected<Prediction<typename P::Refine>, GprError> loo_predict(
        const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        std::span<const typename P::Refine> y_train,
        const PredictOptions& options
    )
    {
        using Storage = P::Storage;

        if (l_factor.is_empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = l_factor.rows();
        auto nn = checked_mul(n, n);
        if (!nn.has_value())
        {
            return std::unexpected(std::move(nn).error());
        }
        std::vector<Storage> scratch_mat(*nn);
        std::vector<Storage> scratch_diag(n);
        MatrixMut<Storage> scratch_mut(scratch_mat.data(), n, n, n);

        Prediction<typename P::Refine> result;
        if (auto res =
                loo_predict_into<P>(likelihood, l_factor, alpha, y_train, options, scratch_mut, scratch_diag, result);
            !res.has_value())
        {
            return std::unexpected(std::move(res).error());
        }
        return result;
    }

    template std::expected<void, GprError> predict_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        Prediction<double>&
    );
    template std::expected<void, GprError> predict_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        Prediction<float>&
    );
    template std::expected<void, GprError> predict_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        Prediction<double>&
    );
    template std::expected<void, GprError> predict_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        Prediction<double>&
    );

    template std::expected<Prediction<double>, GprError> predict<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&
    );
    template std::expected<Prediction<float>, GprError> predict<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&
    );
    template std::expected<Prediction<double>, GprError> predict<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&
    );
    template std::expected<Prediction<double>, GprError> predict<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
    );

    template std::expected<void, GprError> predict_covariance_into<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        PredictiveCovariance<double>&
    );
    template std::expected<void, GprError> predict_covariance_into<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        PredictiveCovariance<float>&
    );
    template std::expected<void, GprError> predict_covariance_into<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        PredictiveCovariance<double>&
    );
    template std::expected<void, GprError> predict_covariance_into<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        PredictiveCovariance<double>&
    );

    template std::expected<PredictiveCovariance<double>, GprError> predict_covariance<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&
    );
    template std::expected<PredictiveCovariance<float>, GprError> predict_covariance<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&
    );
    template std::expected<PredictiveCovariance<double>, GprError> predict_covariance<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&
    );
    template std::expected<PredictiveCovariance<double>, GprError> predict_covariance<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
    );

    template std::expected<std::vector<double>, GprError> sample<DoublePrecision>(
        const CompiledKernel<double>&,
        const GaussianLikelihood&,
        MatrixRef<double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<float>, GprError> sample<SinglePrecision>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<double>, GprError> sample<MixedPrecision<PromoteStorage>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<double>, GprError> sample<MixedPrecision<ReevaluateKernel>>(
        const CompiledKernel<float>&,
        const GaussianLikelihood&,
        MatrixRef<float>,
        MatrixRef<float>,
        std::span<const double>,
        MatrixRef<float>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        std::optional<JitterPolicy>
    );

    template std::expected<void, GprError> loo_predict_into<DoublePrecision>(
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&,
        MatrixMut<double>,
        std::span<double>,
        Prediction<double>&
    );
    template std::expected<void, GprError> loo_predict_into<SinglePrecision>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        std::span<const float>,
        const PredictOptions&,
        MatrixMut<float>,
        std::span<float>,
        Prediction<float>&
    );
    template std::expected<void, GprError> loo_predict_into<MixedPrecision<PromoteStorage>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&,
        MatrixMut<float>,
        std::span<float>,
        Prediction<double>&
    );
    template std::expected<void, GprError> loo_predict_into<MixedPrecision<ReevaluateKernel>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&,
        MatrixMut<float>,
        std::span<float>,
        Prediction<double>&
    );

    template std::expected<Prediction<double>, GprError> loo_predict<DoublePrecision>(
        const GaussianLikelihood&,
        MatrixRef<double>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&
    );
    template std::expected<Prediction<float>, GprError> loo_predict<SinglePrecision>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const float>,
        std::span<const float>,
        const PredictOptions&
    );
    template std::expected<Prediction<double>, GprError> loo_predict<MixedPrecision<PromoteStorage>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&
    );
    template std::expected<Prediction<double>, GprError> loo_predict<MixedPrecision<ReevaluateKernel>>(
        const GaussianLikelihood&,
        MatrixRef<float>,
        std::span<const double>,
        std::span<const double>,
        const PredictOptions&
    );

}  // namespace cxxgpr
