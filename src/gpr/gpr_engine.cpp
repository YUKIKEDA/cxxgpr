// SPDX-License-Identifier: Apache-2.0
/// @file gpr_engine.cpp
/// @brief Out-of-line training, optimization objective, mixed-precision refinement, and inference engine.
/// @details Implements GprObjective (with optional incremental LeafCache evaluation), column-blocked
/// mixed-precision kernel re-evaluation and refinement, train_or_factor_payload, and fitted inference
/// routines explicitly instantiated across all four supported precision policies.

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/workspace/inference.hpp"
#include "cxxgpr/workspace/mll.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "gpr/mixed_krylov.hpp"
#include "gpr/refine_constants.hpp"
#include "workspace/predict_diagonal.hpp"
#include "workspace/sample_draws.hpp"

namespace cxxgpr::detail
{
    namespace
    {

        /// @brief Twice-differentiable negative marginal log-likelihood objective for GPR hyperparameter optimization.
        /// @details Adapts CompiledKernel, GaussianLikelihood, and FitBuffers to the optimizer::TwiceDifferentiable
        /// interface, supporting optional incremental LeafCache evaluation on value_at_changes() calls.
        /// @tparam P Precision policy controlling storage scalar type and buffer allocation.
        template <PrecisionPolicyConcept P> class GprObjective final : public optimizer::TwiceDifferentiable
        {
        public:
            /// @brief Storage scalar type used for kernel matrices and Cholesky factors.
            using Storage = P::Storage;

            /// @brief Constructs an optimization objective bound to borrowed training buffers and data.
            /// @details Enables incremental LeafCache evaluation only when requested by the optimizer and when
            /// buffers retain Cholesky workspace across iterations.
            /// @param[in,out] kernel Compiled kernel instance whose hyperparameters are updated per evaluation.
            /// @param[in,out] likelihood Gaussian observation noise likelihood updated per evaluation.
            /// @param[in] x_train Transformed training feature matrix (n x d) in Storage precision.
            /// @param[in] y_train Transformed training targets of length n in Storage precision.
            /// @param[in] jitter Cholesky stabilization jitter policy.
            /// @param[in,out] buffers Preallocated training workspace buffers.
            /// @param[in] enable_leaf_cache True if the optimizer supplies changed parameter indices.
            GprObjective(
                CompiledKernel<Storage>& kernel,
                GaussianLikelihood& likelihood,
                MatrixRef<Storage> x_train,
                std::span<const Storage> y_train,
                const JitterPolicy& jitter,
                FitBuffers<P>& buffers,
                bool enable_leaf_cache
            ) :
                kernel_(kernel),
                likelihood_(likelihood),
                x_train_(x_train),
                y_train_(y_train),
                jitter_(jitter),
                buffers_(buffers),
                enable_leaf_cache_(enable_leaf_cache),
                last_factored_params_(kernel.num_params() + likelihood.num_params(), 0.0)
            {
                // Preallocate per-leaf kernel caches only when incremental updates can retain matrices across steps.
                if (enable_leaf_cache_)
                {
                    (void)leaf_cache_.resize(kernel_.num_leaves(), x_train_.rows(), kernel_.ard_dims_per_leaf());
                }
            }

            /// @brief Returns total unconstrained hyperparameter dimension (kernel params + log noise variance).
            /// @return Combined parameter vector length.
            [[nodiscard]] std::size_t dim() const noexcept override
            {
                return kernel_.num_params() + likelihood_.num_params();
            }

            /// @brief Returns true when buffers_, kernel_, and likelihood_ already hold the state for params.
            /// @details Compares candidate parameters bitwise via std::bit_cast<std::uint64_t> against the last
            /// successful objective evaluation so train_or_factor_payload() can skip a duplicate factorization.
            /// @param[in] params Hyperparameter vector of length dim().
            /// @return True if buffers_ hold valid L, alpha, factor_jitter, and MLL matching params bitwise.
            [[nodiscard]] bool matches_last_factor(std::span<const double> params) const noexcept
            {
                if (!has_valid_factor_ || params.size() != last_factored_params_.size())
                {
                    return false;
                }
                for (std::size_t i = 0; i < params.size(); ++i)
                {
                    if (std::bit_cast<std::uint64_t>(params[i])
                        != std::bit_cast<std::uint64_t>(last_factored_params_[i]))
                    {
                        return false;
                    }
                }
                return true;
            }

            /// @brief Returns the negative MLL cached alongside the last valid factorization.
            /// @pre matches_last_factor() returned true for the active hyperparameter vector.
            /// @return Cached negative marginal log-likelihood scalar.
            [[nodiscard]] double last_factored_mll() const noexcept { return last_factored_mll_; }

            /// @brief Evaluates the negative marginal log-likelihood at params, invalidating any cached leaves.
            /// @details Updates kernel and likelihood parameters, clears LeafCache if enabled, and runs full
            /// Cholesky factorization and MLL evaluation.
            /// @param[in] params Unconstrained hyperparameter vector of length dim().
            /// @return Negative marginal log-likelihood scalar, or GprError on invalid params or Cholesky failure.
            /// @retval GprErrorCode::LengthMismatch When params.size() != dim().
            /// @retval GprErrorCode::CholeskyFailed When matrix factorization fails after jitter retries.
            [[nodiscard]] std::expected<double, GprError> value(std::span<const double> params) override
            {
                has_valid_factor_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }
                if (!enable_leaf_cache_)
                {
                    auto res = factor_and_mll(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_, 0U);
                    if (res.has_value())
                    {
                        record_valid_factor(params, *res);
                    }
                    return res;
                }

                // Populate DistanceCache (when Cached) and prime LeafCache + ARD s^2 baseline in one pass.
                const std::size_t n = x_train_.rows();
                if (buffers_.core.nested.depth() < kernel_.nested_depth() || buffers_.core.nested.rows() < n
                    || buffers_.core.nested.cols() < n)
                {
                    buffers_.core.nested.resize(kernel_.nested_depth(), n, n);
                }
                if (buffers_.dist.has_value())
                {
                    const bool need_dist = kernel_.reads_distances();
                    const bool need_ard = kernel_.needs_ard_sq_diff();
                    if ((need_dist || need_ard)
                        && (buffers_.dist->num_points() != n || buffers_.dist->num_dims() != x_train_.cols()
                            || (need_dist && !buffers_.dist->has_dist()) || (need_ard && !buffers_.dist->has_ard())))
                    {
                        if (auto ok = buffers_.dist->ensure(x_train_, need_dist, need_ard, DistanceCachePolicy::Cached);
                            !ok.has_value())
                        {
                            leaf_cache_.invalidate();
                            return std::unexpected(std::move(ok).error());
                        }
                    }
                }
                const DistanceCache<Storage>* dist_ptr = buffers_.dist.has_value() ? &(*buffers_.dist) : nullptr;
                if (auto primed = leaf_cache_.prime(
                        kernel_, x_train_, dist_ptr, params, buffers_.core.k_mut(), Triangle::Lower,
                        buffers_.core.exp_mut(), buffers_.core.nested
                    );
                    !primed.has_value())
                {
                    return std::unexpected(std::move(primed).error());
                }
                return factor_from_assembled_k_and_mll(params);
            }

            /// @brief Evaluates negative MLL incrementally when only changed_indices moved since the last step.
            /// @details Evaluates only dirty kernel leaves via LeafCache in non-destructive trial mode, adds noise
            /// variance to the diagonal, factors L with jitter, and computes MLL directly.
            /// @param[in] params Full hyperparameter vector of length dim().
            /// @param[in] changed_indices Parameter indices modified relative to the last evaluated step.
            /// @return Negative marginal log-likelihood scalar, or GprError on invalid params or factorization failure.
            /// @retval GprErrorCode::LengthMismatch When params.size() != dim().
            /// @retval GprErrorCode::NonFiniteKernelValue When evaluated MLL is non-finite.
            /// @retval GprErrorCode::CholeskyFailed When factorization fails after exhausting jitter retries.
            [[nodiscard]] std::expected<double, GprError> value_at_changes(
                std::span<const double> params, std::span<const std::size_t> changed_indices
            ) override
            {
                // Fall back to full evaluation when LeafCache is disabled (e.g., CholeskyBuffer::Reuse).
                if (!enable_leaf_cache_)
                {
                    return value(params);
                }
                has_valid_factor_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }

                // Recompute only affected kernel leaves as a non-destructive trial into buffers_.core.k_mut().
                const DistanceCache<Storage>* dist_ptr = buffers_.dist.has_value() ? &(*buffers_.dist) : nullptr;
                auto inc = leaf_cache_.evaluate_incremental(
                    kernel_, x_train_, dist_ptr, params, changed_indices, dim(), buffers_.core.k_mut(), Triangle::Lower,
                    buffers_.core.exp_mut(), buffers_.core.nested, false
                );
                if (!inc.has_value())
                {
                    return std::unexpected(std::move(inc).error());
                }
                return factor_from_assembled_k_and_mll(params);
            }

            /// @brief Commits or discards the pending LeafCache trial based on optimizer acceptance.
            /// @param[in] accepted True if the immediately preceding proposal became the accepted state.
            void notify_candidate_accepted(bool accepted) noexcept override
            {
                if (!enable_leaf_cache_)
                {
                    return;
                }
                const DistanceCache<Storage>* dist_ptr = buffers_.dist.has_value() ? &(*buffers_.dist) : nullptr;
                if (auto res = leaf_cache_.commit_candidate(accepted, kernel_, x_train_, dist_ptr); !res.has_value())
                {
                    leaf_cache_.invalidate();
                }
            }

            /// @brief Writes open optimization bounds for kernel and likelihood hyperparameters into out.
            /// @param[out] out Destination span of length dim().
            /// @return Empty expected on success, or LengthMismatch when out size differs from dim().
            /// @retval GprErrorCode::LengthMismatch When out.size() != kernel_.num_params() + 1.
            [[nodiscard]] std::expected<void, GprError> fill_intervals(std::span<Interval> out) const override
            {
                const std::size_t nk = kernel_.num_params();
                if (out.size() != nk + 1U)
                {
                    return std::unexpected(
                        GprError::length_mismatch("interval buffer length must match parameter count")
                    );
                }
                // Place kernel parameter bounds first, followed by the single likelihood noise bound.
                if (auto kb = kernel_.bounds_into(out.subspan(0, nk)); !kb.has_value())
                {
                    return kb;
                }
                return likelihood_.bounds_into(out.subspan(nk, 1U));
            }

            /// @brief Evaluates the negative MLL gradient into out_grad at params.
            /// @param[in] params Hyperparameter vector of length dim().
            /// @param[out] out_grad Output gradient span of length dim().
            /// @return Empty expected on success, or GprError propagated from value_and_gradient_into().
            /// @retval GprErrorCode::LengthMismatch When parameter or gradient span lengths mismatch.
            [[nodiscard]] std::expected<void, GprError> gradient_into(
                std::span<const double> params, std::span<double> out_grad
            ) override
            {
                auto mll = value_and_gradient_into(params, out_grad);
                if (!mll.has_value())
                {
                    return std::unexpected(std::move(mll).error());
                }
                return {};
            }

            /// @brief Evaluates negative MLL and its exact gradient with respect to params into out_grad.
            /// @param[in] params Hyperparameter vector of length dim().
            /// @param[out] out_grad Output gradient span of length dim().
            /// @return Negative marginal log-likelihood scalar, or GprError on invalid params or Cholesky failure.
            /// @retval GprErrorCode::LengthMismatch When parameter or gradient span lengths mismatch.
            /// @retval GprErrorCode::CholeskyFailed When factorization fails after jitter retries.
            [[nodiscard]] std::expected<double, GprError> value_and_gradient_into(
                std::span<const double> params, std::span<double> out_grad
            ) override
            {
                has_valid_factor_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }
                if (enable_leaf_cache_)
                {
                    leaf_cache_.invalidate();
                }
                auto res = mll_and_gradient_into(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_, out_grad);
                if (res.has_value())
                {
                    record_valid_factor(params, *res);
                }
                return res;
            }

            /// @brief Evaluates the exact Hessian of the negative MLL into out_hess at params.
            /// @param[in] params Hyperparameter vector of length dim().
            /// @param[out] out_hess Output column-major Hessian span of length dim() * dim().
            /// @return Empty expected on success, or GprError on invalid params or Cholesky failure.
            /// @retval GprErrorCode::LengthMismatch When parameter or Hessian span lengths mismatch.
            /// @retval GprErrorCode::CholeskyFailed When factorization fails after jitter retries.
            [[nodiscard]] std::expected<void, GprError> hessian_into(
                std::span<const double> params, std::span<double> out_hess
            ) override
            {
                has_valid_factor_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return set;
                }
                if (enable_leaf_cache_)
                {
                    leaf_cache_.invalidate();
                }
                return cxxgpr::hessian_into(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_, out_hess);
            }

            /// @brief Jointly evaluates negative MLL, gradient, and Hessian at params.
            /// @param[in] params Hyperparameter vector of length dim().
            /// @param[out] out_grad Output gradient span of length dim().
            /// @param[out] out_hess Output column-major Hessian span of length dim() * dim().
            /// @return Negative marginal log-likelihood scalar, or GprError on invalid params or Cholesky failure.
            /// @retval GprErrorCode::LengthMismatch When parameter, gradient, or Hessian span lengths mismatch.
            /// @retval GprErrorCode::CholeskyFailed When factorization fails after jitter retries.
            [[nodiscard]] std::expected<double, GprError> value_gradient_hessian_into(
                std::span<const double> params, std::span<double> out_grad, std::span<double> out_hess
            ) override
            {
                has_valid_factor_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }
                if (enable_leaf_cache_)
                {
                    leaf_cache_.invalidate();
                }
                auto res = cxxgpr::value_gradient_hessian_into(
                    kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_, out_grad, out_hess
                );
                if (res.has_value())
                {
                    record_valid_factor(params, *res);
                }
                return res;
            }

        private:
            /// @brief Adds observation noise, factors L with jitter, solves alpha, and computes negative MLL from K.
            /// @param[in] params Hyperparameter vector corresponding to the lower triangle assembled in `k_mut()`.
            /// @return Negative marginal log-likelihood scalar, or GprError on factorization or solve failure.
            [[nodiscard]] std::expected<double, GprError> factor_from_assembled_k_and_mll(
                std::span<const double> params
            )
            {
                const std::size_t n = x_train_.rows();
                const auto noise_var = static_cast<Storage>(likelihood_.noise_variance());
                if (auto diag = linalg::add_to_diag(buffers_.core.k_mut(), noise_var); !diag.has_value())
                {
                    return std::unexpected(std::move(diag).error());
                }
                auto jit = linalg::cholesky_lower_with_jitter(
                    buffers_.core.k_mut(), buffers_.core.exp_mut(), jitter_, CholeskyStage::Fit
                );
                if (!jit.has_value())
                {
                    return std::unexpected(std::move(jit).error());
                }
                buffers_.core.factor_jitter = *jit;

                std::ranges::copy(y_train_, buffers_.core.rhs.begin());
                if (auto solve =
                        linalg::solve_llt_in_place(buffers_.core.k_ref(), std::span<Storage>(buffers_.core.rhs));
                    !solve.has_value())
                {
                    return std::unexpected(std::move(solve).error());
                }
                auto logdet = linalg::log_det_from_l(buffers_.core.k_ref());
                if (!logdet.has_value())
                {
                    return std::unexpected(std::move(logdet).error());
                }
                auto y_alpha = linalg::dot(y_train_, std::span<const Storage>(buffers_.core.rhs));
                if (!y_alpha.has_value())
                {
                    return std::unexpected(std::move(y_alpha).error());
                }
                const double mll =
                    0.5 * (*y_alpha + *logdet + (static_cast<double>(n) * std::log(2.0 * std::numbers::pi)));
                if (!std::isfinite(mll))
                {
                    return std::unexpected(GprError::non_finite_kernel_value());
                }
                record_valid_factor(params, mll);
                return mll;
            }

            /// @brief Records parameters and negative MLL paired with the valid live factor in buffers_.
            /// @param[in] params Evaluated hyperparameter vector of length dim().
            /// @param[in] mll Evaluated negative marginal log-likelihood matching params.
            void record_valid_factor(std::span<const double> params, double mll) noexcept
            {
                std::ranges::copy(params, last_factored_params_.begin());
                last_factored_mll_ = mll;
                has_valid_factor_ = true;
            }

            /// @brief Applies kernel and likelihood parameters after validating vector length.
            /// @param[in] params Combined parameter vector [kernel_params..., log_noise_var].
            /// @return Empty expected on success, or LengthMismatch / validation GprError.
            [[nodiscard]] std::expected<void, GprError> apply_params(std::span<const double> params)
            {
                const std::size_t nk = kernel_.num_params();
                if (params.size() != nk + 1U)
                {
                    return std::unexpected(
                        GprError::length_mismatch("parameter length must match kernel + noise params")
                    );
                }
                // Apply kernel parameters before updating the trailing observation noise parameter.
                if (auto kp = kernel_.set_params(params.subspan(0, nk)); !kp.has_value())
                {
                    return kp;
                }
                return likelihood_.set_params(params.subspan(nk, 1U));
            }

            /// @brief Compiled kernel evaluator mutated per optimizer step.
            CompiledKernel<Storage>& kernel_;
            /// @brief Observation noise likelihood mutated per optimizer step.
            GaussianLikelihood& likelihood_;
            /// @brief Transformed training feature matrix (n x d) in Storage precision.
            MatrixRef<Storage> x_train_;
            /// @brief Transformed training targets of length n in Storage precision.
            std::span<const Storage> y_train_;
            /// @brief Cholesky stabilization jitter policy.
            const JitterPolicy& jitter_;
            /// @brief Shared training workspace buffers.
            FitBuffers<P>& buffers_;
            /// @brief True when incremental LeafCache updates are enabled for value_at_changes().
            bool enable_leaf_cache_{false};
            /// @brief Per-leaf cached kernel blocks supporting incremental parameter evaluation.
            LeafCache<Storage> leaf_cache_;
            /// @brief Pre-allocated parameter snapshot matching the last successful objective factorization.
            std::vector<double> last_factored_params_;
            /// @brief Cached negative marginal log-likelihood corresponding to last_factored_params_.
            double last_factored_mll_{0.0};
            /// @brief True when buffers_.core.k_ref() and rhs hold a valid factor matching last_factored_params_.
            bool has_valid_factor_{false};
        };

        /// @brief Computes double-precision B_f64 * v (and optionally ||B_f64||_inf) in width-32 column blocks.
        /// @details Evaluates diagonal entries explicitly via fill_diag() once per solve and streams cross-kernel
        /// column slices of width at most refinement_column_block without materializing the full n x n matrix.
        /// @param[in] kernel_f64 Double-precision compiled kernel evaluator.
        /// @param[in] x_f64 Double-precision transformed training features (n x d).
        /// @param[in] diag_offset Combined observation noise variance and accepted jitter added to diagonal entries.
        /// @param[in] v Input vector of length n in double precision.
        /// @param[out] out_bv Output matrix-vector product B_f64 * v of length n.
        /// @param[in] compute_b_inf True to compute and return ||B_f64||_inf alongside B_f64 * v.
        /// @param[in,out] diag_k Cached kernel diagonal vector (populated on first call when empty).
        /// @param[in,out] diag_scratch Scratch buffer for fill_diag().
        /// @param[in,out] x_block Reusable block_cols x d coordinate slice buffer.
        /// @param[in,out] k_block Reusable n x block_cols cross-kernel block buffer.
        /// @param[in,out] scratch_block Reusable n x block_cols kernel scratch buffer.
        /// @param[in,out] nested Reusable double-precision nested evaluation buffers.
        /// @param[in,out] row_abs_sum Reusable n-element buffer accumulating row l1 sums when compute_b_inf is true.
        /// @return ||B_f64||_inf when compute_b_inf is true, or 0.0 otherwise.
        [[nodiscard]] std::expected<double, GprError> block_reeval_matvec(
            const CompiledKernel<double>& kernel_f64,
            MatrixRef<double> x_f64,
            double diag_offset,
            std::span<const double> v,
            std::span<double> out_bv,
            bool compute_b_inf,
            std::vector<double>& diag_k,
            std::vector<double>& diag_scratch,
            std::vector<double>& x_block,
            std::vector<double>& k_block,
            std::vector<double>& scratch_block,
            NestedBuffers<double>& nested,
            std::vector<double>& row_abs_sum
        )
        {
            const std::size_t n = x_f64.rows();
            const std::size_t d = x_f64.cols();
            const std::size_t block_cols = std::min(refinement_column_block, n);

            // Evaluate main kernel diagonal once so diagonal-only leaves such as WhiteKernel are included.
            if (diag_k.size() < n)
            {
                diag_k.resize(n);
                diag_scratch.resize((kernel_f64.nested_depth() + 1U) * n);
                if (auto dg = kernel_f64.fill_diag(x_f64, diag_k, diag_scratch); !dg.has_value())
                {
                    return std::unexpected(std::move(dg).error());
                }
            }

            if (x_block.size() < block_cols * d)
            {
                x_block.resize(block_cols * d);
            }
            if (k_block.size() < n * block_cols)
            {
                k_block.resize(n * block_cols);
            }
            if (scratch_block.size() < n * block_cols)
            {
                scratch_block.resize(n * block_cols);
            }
            if (nested.depth() < kernel_f64.nested_depth() || nested.rows() < n || nested.cols() < block_cols)
            {
                nested.resize(kernel_f64.nested_depth(), n, block_cols);
            }

            std::ranges::fill(out_bv, 0.0);
            if (compute_b_inf)
            {
                row_abs_sum.assign(n, 0.0);
            }

            for (std::size_t col_begin = 0; col_begin < n; col_begin += block_cols)
            {
                const std::size_t cols_cur = std::min(block_cols, n - col_begin);
                // Pack the current block of training points contiguously for column-major cross evaluation.
                for (std::size_t feat = 0; feat < d; ++feat)
                {
                    for (std::size_t c = 0; c < cols_cur; ++c)
                    {
                        x_block[(feat * cols_cur) + c] = x_f64(col_begin + c, feat);
                    }
                }
                MatrixRef<double> xb_ref(x_block.data(), cols_cur, d, cols_cur);
                MatrixMut<double> kb_mut(k_block.data(), n, cols_cur, n);
                MatrixMut<double> sb_mut(scratch_block.data(), n, cols_cur, n);
                if (auto res = kernel_f64.apply_cross_points(x_f64, xb_ref, kb_mut, sb_mut, nested); !res.has_value())
                {
                    return std::unexpected(std::move(res).error());
                }
                for (std::size_t c = 0; c < cols_cur; ++c)
                {
                    const std::size_t global_col = col_begin + c;
                    const double v_col = v[global_col];
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        const double entry = (row == global_col) ? (diag_k[row] + diag_offset) : k_block[(c * n) + row];
                        out_bv[row] += entry * v_col;
                        if (compute_b_inf)
                        {
                            row_abs_sum[row] += std::abs(entry);
                        }
                    }
                }
            }

            return compute_b_inf ? inf_norm(row_abs_sum) : 0.0;
        }

        /// @brief Computes double-precision residual r = y_f64 - B_f64 * w and ||B_f64||_inf using column
        /// blocks of width 32 without materializing the full n x n double matrix.
        /// @param[in] kernel_f64 Double-precision compiled kernel evaluator.
        /// @param[in] x_f64 Double-precision transformed training features (n x d).
        /// @param[in] y_f64 Double-precision transformed training targets of length n.
        /// @param[in] diag_offset Combined observation noise variance and accepted jitter added to diagonal entries.
        /// @param[in] w Current candidate solution vector of length n in double precision.
        /// @param[out] out_r Output double-precision residual vector of length n.
        /// @return Infinity norm ||B_f64||_inf for backward-error stopping checks, or GprError on evaluation failure.
        [[nodiscard]] std::expected<double, GprError> block_reeval_residual(
            const CompiledKernel<double>& kernel_f64,
            MatrixRef<double> x_f64,
            std::span<const double> y_f64,
            double diag_offset,
            std::span<const double> w,
            std::span<double> out_r
        )
        {
            const std::size_t n = x_f64.rows();
            std::vector<double> diag_k;
            std::vector<double> diag_scratch;
            std::vector<double> x_block;
            std::vector<double> k_block;
            std::vector<double> scratch_block;
            NestedBuffers<double> nested;
            std::vector<double> matvec(n, 0.0);
            std::vector<double> row_abs_sum;

            auto b_inf = block_reeval_matvec(
                kernel_f64, x_f64, diag_offset, w, matvec, true, diag_k, diag_scratch, x_block, k_block, scratch_block,
                nested, row_abs_sum
            );
            if (!b_inf.has_value())
            {
                return std::unexpected(std::move(b_inf).error());
            }
            for (std::size_t row = 0; row < n; ++row)
            {
                out_r[row] = y_f64[row] - matvec[row];
            }
            return *b_inf;
        }

        /// @brief Mixed-precision iterative refinement adapter implementing RefineSystem and Krylov operations.
        /// @details Bridges float Cholesky factor L_f32 (for fast preconditioning) with either promoted float
        /// matrix residuals (PromoteStorage) or column-blocked double kernel re-evaluation (ReevaluateKernel).
        /// @tparam P Mixed-precision policy specifying Residual formula (PromoteStorage or ReevaluateKernel).
        template <PrecisionPolicyConcept P> class GprRefineSystem final : public RefineSystem
        {
        public:
            /// @brief Constructs the refinement adapter from double data, mutable float Cholesky factor, and A_f32.
            /// @param[in] spec Fitted kernel specification compiled in double precision for re-evaluation/fallback.
            /// @param[in] kernel_exp Exponential backend mode matching the fitted model.
            /// @param[in] x_f64 Double-precision transformed training features (n x d).
            /// @param[in] y_f64 Double-precision transformed training targets of length n.
            /// @param[in,out] l_f32 Mutable single-precision lower-triangular Cholesky factor L (n x n).
            /// @param[in] a_f32_symm Symmetrized single-precision noisy system matrix A (used by PromoteStorage).
            /// @param[in] diag_offset Combined observation noise variance and accepted jitter added to diagonal.
            GprRefineSystem(
                const KernelSpec& spec,
                KernelExp kernel_exp,
                MatrixRef<double> x_f64,
                std::span<const double> y_f64,
                MatrixMut<float> l_f32,
                MatrixRef<float> a_f32_symm,
                double diag_offset
            ) :
                kernel_f64_(CompiledKernel<double>::from_spec(spec)),
                x_f64_(x_f64),
                y_f64_(y_f64),
                l_f32_(l_f32),
                a_f32_(a_f32_symm),
                diag_offset_(diag_offset),
                r_f32_buf_(x_f64.rows(), 0.0F),
                row_abs_buf_(x_f64.rows(), 0.0),
                sum_buf_(x_f64.rows(), 0.0)
            {
                kernel_f64_.set_kernel_exp(kernel_exp);
            }

            /// @brief Returns the double-precision transformed target vector y_f64 as refinement RHS.
            /// @return Non-owning span over y_f64_.
            [[nodiscard]] std::span<const double> rhs() const noexcept override { return y_f64_; }

            /// @brief Computes residual r = y_f64 - A * w according to P::Residual and returns matrix norm.
            /// @param[in] w Current double-precision iterate of length n.
            /// @param[out] r Output double-precision residual buffer of length n.
            /// @return Infinity norm of the system matrix used for relative residual scaling, or GprError.
            [[nodiscard]] std::expected<double, GprError> residual(
                std::span<const double> w, std::span<double> r
            ) override
            {
                if constexpr (std::same_as<typename P::Residual, PromoteStorage>)
                {
                    return promoted_residual_into(a_f32_, w, y_f64_, r, row_abs_buf_, sum_buf_);
                }
                else
                {
                    auto b_inf = matvec_and_inf_norm(w, sum_buf_, true);
                    if (!b_inf.has_value())
                    {
                        return b_inf;
                    }
                    for (std::size_t i = 0; i < y_f64_.size(); ++i)
                    {
                        r[i] = y_f64_[i] - sum_buf_[i];
                    }
                    return b_inf;
                }
            }

            /// @brief Computes exact blocked double-precision B_f64 * v and optionally ||B_f64||_inf.
            /// @param[in] v Input vector of length n in double precision.
            /// @param[out] out_bv Output B_f64 * v buffer of length n.
            /// @param[in] compute_b_inf True to compute and return ||B_f64||_inf.
            /// @return ||B_f64||_inf when requested, or 0.0 otherwise.
            [[nodiscard]] std::expected<double, GprError> matvec_and_inf_norm(
                std::span<const double> v, std::span<double> out_bv, bool compute_b_inf
            )
            {
                return block_reeval_matvec(
                    kernel_f64_, x_f64_, diag_offset_, v, out_bv, compute_b_inf, diag_k_, diag_scratch_, x_block_,
                    k_block_, scratch_block_, nested_, row_abs_buf_
                );
            }

            /// @brief Applies scaled float Cholesky preconditioning z = (L_f32 L_f32^T)^{-1} r.
            /// @param[in] r Input residual or Krylov vector of length n.
            /// @param[out] z Output preconditioned double vector of length n.
            /// @return Empty expected on success, or GprError on solve failure or non-finite values.
            [[nodiscard]] std::expected<void, GprError> apply_preconditioner(
                std::span<const double> r, std::span<double> z
            )
            {
                return apply_scaled_f32_preconditioner(l_f32_.as_ref(), r, std::span<float>(r_f32_buf_), z);
            }

            /// @brief Applies one iterative refinement update w += (L_f32 L_f32^T)^-1 r.
            /// @param[in] r Double-precision residual vector of length n.
            /// @param[in,out] w In-out double-precision solution vector updated in place.
            /// @return Empty expected on success, or GprError if triangular solve fails.
            [[nodiscard]] std::expected<void, GprError> correct(std::span<const double> r, std::span<double> w) override
            {
                const std::size_t n = r.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    r_f32_buf_[i] = static_cast<float>(r[i]);
                }
                if (auto solve = linalg::solve_llt_in_place(l_f32_.as_ref(), std::span<float>(r_f32_buf_));
                    !solve.has_value())
                {
                    return solve;
                }
                for (std::size_t i = 0; i < n; ++i)
                {
                    w[i] += static_cast<double>(r_f32_buf_[i]);
                }
                return {};
            }

            /// @brief Verifies whether iterate w satisfies relative residual tolerance against the true double system.
            /// @param[in] w Candidate double-precision iterate of length n.
            /// @param[in] tol Relative backward-error acceptance threshold.
            /// @return True if iterate is accepted; false triggers full double-precision fallback().
            [[nodiscard]] std::expected<bool, GprError> accept(std::span<const double> w, double tol) override
            {
                if constexpr (std::same_as<typename P::Residual, ReevaluateKernel>)
                {
                    return true;
                }
                else
                {
                    std::vector<double> r_true(y_f64_.size());
                    auto b_inf = block_reeval_residual(kernel_f64_, x_f64_, y_f64_, diag_offset_, w, r_true);
                    if (!b_inf.has_value())
                    {
                        return std::unexpected(std::move(b_inf).error());
                    }
                    return meets_relative_residual(
                        *b_inf, w, y_f64_, r_true, detail::promote_storage_accept_tolerance(tol)
                    );
                }
            }

            /// @brief Computes an exact double-precision Cholesky solution when iterative refinement diverges or
            /// stalls.
            /// @details Releases refinement block scratch before materializing the full double Gram matrix, then
            /// updates the lower triangle of l_f32_ from the true double Cholesky factor on success.
            /// @param[out] w Output double-precision weight vector overwritten with A_f64^-1 y_f64.
            /// @return Empty expected on success, or GprError if kernel evaluation or Cholesky fails.
            [[nodiscard]] std::expected<void, GprError> fallback(std::span<double> w) override
            {
                std::vector<double>{}.swap(diag_k_);
                std::vector<double>{}.swap(diag_scratch_);
                std::vector<double>{}.swap(x_block_);
                std::vector<double>{}.swap(k_block_);
                std::vector<double>{}.swap(scratch_block_);
                nested_ = NestedBuffers<double>{};
                std::vector<float>{}.swap(r_f32_buf_);
                std::vector<double>{}.swap(row_abs_buf_);
                std::vector<double>{}.swap(sum_buf_);

                const std::size_t n = x_f64_.rows();
                std::vector<double> a_f64(n * n);
                std::vector<double> scratch_f64(n * n);
                auto nested = kernel_f64_.make_nested_buffers(n, n);
                MatrixMut<double> a_mut(a_f64.data(), n, n, n);
                MatrixMut<double> sc_mut(scratch_f64.data(), n, n, n);

                if (auto ap = kernel_f64_.apply_points(x_f64_, a_mut, Triangle::Lower, sc_mut, nested); !ap.has_value())
                {
                    return ap;
                }
                if (auto dg = linalg::add_to_diag(a_mut, diag_offset_); !dg.has_value())
                {
                    return dg;
                }
                if (auto ch = linalg::cholesky_lower_in_place(a_mut, CholeskyStage::Fit); !ch.has_value())
                {
                    return ch;
                }
                std::ranges::copy(y_f64_, w.begin());
                if (auto sv = linalg::solve_llt_in_place(a_mut.as_ref(), w); !sv.has_value())
                {
                    return sv;
                }

                // Refresh l_f32_ lower triangle from the true double Cholesky factor so L and alpha remain consistent.
                for (std::size_t col = 0; col < n; ++col)
                {
                    const auto diag = static_cast<float>(a_mut(col, col));
                    l_f32_(col, col) = (std::isfinite(diag) && diag > 0.0F) ? diag : std::numeric_limits<float>::min();
                    for (std::size_t row = col + 1U; row < n; ++row)
                    {
                        const auto val = static_cast<float>(a_mut(row, col));
                        l_f32_(row, col) = std::isfinite(val) ? val : 0.0F;
                    }
                }
                used_f64_fallback_ = true;
                return {};
            }

            /// @brief Returns true when fallback() succeeded with a full double-precision Cholesky solve.
            /// @return True iff fallback() completed during this refinement session.
            [[nodiscard]] bool used_f64_fallback() const noexcept { return used_f64_fallback_; }

        private:
            /// @brief Double-precision compiled kernel for blocked residual evaluation and fallback solves.
            CompiledKernel<double> kernel_f64_;
            /// @brief Double-precision transformed training features (n x d).
            MatrixRef<double> x_f64_;
            /// @brief Double-precision transformed training targets of length n.
            std::span<const double> y_f64_;
            /// @brief Mutable single-precision Cholesky factor L_f32 retained from the final fit factorization.
            MatrixMut<float> l_f32_;
            /// @brief Symmetrized float noisy Gram matrix A_f32 used when P::Residual is PromoteStorage.
            MatrixRef<float> a_f32_;
            /// @brief Combined noise variance and accepted jitter added along the diagonal.
            double diag_offset_{0.0};
            /// @brief Reusable single-precision residual buffer for triangular L_f32 solves.
            std::vector<float> r_f32_buf_;
            /// @brief Reusable row l1-norm scratch buffer.
            std::vector<double> row_abs_buf_;
            /// @brief Reusable double-precision matrix-vector accumulator buffer.
            std::vector<double> sum_buf_;
            /// @brief Cached double-precision kernel diagonal vector of length n.
            std::vector<double> diag_k_;
            /// @brief Scratch vector for double-precision fill_diag().
            std::vector<double> diag_scratch_;
            /// @brief Reusable coordinate block buffer for blocked matvecs.
            std::vector<double> x_block_;
            /// @brief Reusable cross-kernel block buffer for blocked matvecs.
            std::vector<double> k_block_;
            /// @brief Reusable kernel evaluation scratch block for blocked matvecs.
            std::vector<double> scratch_block_;
            /// @brief Reusable double-precision nested evaluation buffers for blocked matvecs.
            NestedBuffers<double> nested_;
            /// @brief Set true once fallback() overwrote w and l_f32_ from double Cholesky.
            bool used_f64_fallback_{false};
        };

        /// @brief Computes negative log marginal likelihood from factor L and weights alpha in double targets.
        /// @param[in] l_ref Lower-triangular Cholesky factor view of dimension n x n.
        /// @param[in] y_f64 Transformed training targets in double precision.
        /// @param[in] alpha Solved weights vector of length n in double precision.
        /// @return Negative marginal log-likelihood scalar, or GprError if non-finite.
        [[nodiscard]] std::expected<double, GprError> compute_mll_from_factor_and_alpha(
            MatrixRef<float> l_ref, std::span<const double> y_f64, std::span<const double> alpha
        )
        {
            auto logdet = linalg::log_det_from_l(l_ref);
            if (!logdet.has_value())
            {
                return std::unexpected(std::move(logdet).error());
            }
            const double y_alpha = dot_f64(y_f64, alpha);
            const double mll =
                0.5 * (y_alpha + *logdet + (static_cast<double>(y_f64.size()) * std::log(2.0 * std::numbers::pi)));
            if (!std::isfinite(mll))
            {
                return std::unexpected(GprError::non_finite_kernel_value());
            }
            return mll;
        }

        /// @brief Executes PromoteStorage iterative refinement starting from the float Cholesky solution.
        /// @tparam P Mixed-precision policy (PromoteStorage).
        /// @param[in] spec Fitted kernel specification compiled in double precision for accept/fallback.
        /// @param[in] kernel_exp Exponential backend mode matching the fitted model.
        /// @param[in] kernel_f32 Single-precision compiled kernel evaluator.
        /// @param[in] x_f64 Double-precision transformed training features (n x d).
        /// @param[in] x_f32 Single-precision transformed training features (n x d).
        /// @param[in] y_f64 Double-precision transformed training targets of length n.
        /// @param[in,out] buffers Training workspace buffers released progressively before refinement.
        /// @param[in] diag_offset Observation noise variance plus accepted factor jitter.
        /// @param[out] out_alpha Output refined double-precision weight vector of length n.
        /// @param[out] out_summary Output optional populated with refinement diagnostics.
        /// @return Empty expected on success, or GprError if kernel evaluation or fallback fails.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> run_mixed_precision_refinement(
            const KernelSpec& spec,
            KernelExp kernel_exp,
            const CompiledKernel<float>& kernel_f32,
            MatrixRef<double> x_f64,
            MatrixRef<float> x_f32,
            std::span<const double> y_f64,
            FitBuffers<P>& buffers,
            double diag_offset,
            std::vector<double>& out_alpha,
            std::optional<RefinementSummary>& out_summary
        )
        {
            const std::size_t n = x_f64.rows();
            out_alpha.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                out_alpha[i] = static_cast<double>(buffers.core.rhs[i]);
            }

            std::vector<float>{}.swap(buffers.w_matrix);
            buffers.dist.reset();
            buffers.core.weighted = NestedBuffers<float>{};
            std::vector<float>{}.swap(buffers.core.ard_s2_storage);
            std::vector<std::uint8_t>{}.swap(buffers.core.ard_s2_ready);
            buffers.core.hessian = HessianScratch<float>{};
            std::vector<float>{}.swap(buffers.core.rhs);

            if (auto ks = buffers.core.ensure_kernel_scratch(); !ks.has_value())
            {
                return std::unexpected(std::move(ks).error());
            }
            MatrixMut<float> a_f32_mut = buffers.core.exp_mut();
            if (auto ap = kernel_f32.apply_points(
                    x_f32, a_f32_mut, Triangle::Lower, buffers.core.kernel_scratch_mut(), buffers.core.nested
                );
                !ap.has_value())
            {
                return ap;
            }
            if (auto dg = linalg::add_to_diag(a_f32_mut, static_cast<float>(diag_offset)); !dg.has_value())
            {
                return dg;
            }
            symmetrize_lower(a_f32_mut);
            const MatrixRef<float> a_f32_ref = a_f32_mut.as_ref();
            std::vector<float>{}.swap(buffers.core.kernel_scratch);
            buffers.core.nested = NestedBuffers<float>{};

            GprRefineSystem<P> sys(spec, kernel_exp, x_f64, y_f64, buffers.core.k_mut(), a_f32_ref, diag_offset);
            std::vector<double> res_scratch(n);
            auto summary = refine_in_place(sys, out_alpha, res_scratch);
            if (!summary.has_value())
            {
                return std::unexpected(std::move(summary).error());
            }
            out_summary = *summary;
            return {};
        }

        /// @brief Factors and solves a MixedPrecision<ReevaluateKernel> system using preconditioner shifts and
        /// restarted FGMRES-IR before resorting to double fallback or equation jitter.
        /// @tparam P Mixed-precision policy (ReevaluateKernel).
        /// @param[in] spec Fitted kernel specification compiled in double precision for refinement and fallback.
        /// @param[in] kernel_exp Exponential backend mode matching the fitted model.
        /// @param[in] kernel_f32 Single-precision compiled kernel evaluator.
        /// @param[in] likelihood Observation noise likelihood supplying base noise variance.
        /// @param[in] jitter_policy Diagonal jitter policy invoked only when true double systems fail Cholesky.
        /// @param[in] x_f64 Double-precision transformed training features (n x d).
        /// @param[in] x_f32 Single-precision transformed training features (n x d).
        /// @param[in] y_f64 Double-precision transformed training targets of length n.
        /// @param[in,out] buffers Training workspace buffers holding L_f32 and scratch matrices.
        /// @param[in] existing_unshifted_factor_ready True when buffers already hold valid j = 0 factor L and rhs.
        /// @param[out] out_alpha Output double-precision solved weights of length n.
        /// @param[out] out_summary Output optional populated with refinement and Krylov diagnostics.
        /// @param[in,out] in_out_mll Negative marginal log-likelihood updated if re-factored or fallen back.
        /// @return Empty expected on success, or GprError if all retries and fallback fail.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<void, GprError> solve_mixed_reevaluate_payload(
            const KernelSpec& spec,
            KernelExp kernel_exp,
            const CompiledKernel<float>& kernel_f32,
            const GaussianLikelihood& likelihood,
            const JitterPolicy& jitter_policy,
            MatrixRef<double> x_f64,
            MatrixRef<float> x_f32,
            std::span<const double> y_f64,
            FitBuffers<P>& buffers,
            bool existing_unshifted_factor_ready,
            std::vector<double>& out_alpha,
            std::optional<RefinementSummary>& out_summary,
            double& in_out_mll
        )
        {
            const std::size_t n = x_f64.rows();
            out_alpha.assign(n, 0.0);

            // Release optimization-only storage prior to refinement or preconditioner retries.
            std::vector<float>{}.swap(buffers.w_matrix);
            buffers.dist.reset();
            buffers.core.weighted = NestedBuffers<float>{};
            std::vector<float>{}.swap(buffers.core.ard_s2_storage);
            std::vector<std::uint8_t>{}.swap(buffers.core.ard_s2_ready);
            buffers.core.hessian = HessianScratch<float>{};
            std::vector<float>{}.swap(buffers.core.kernel_scratch);

            // Fast path: reuse an already factored j = 0 unshifted float Cholesky when available.
            if (existing_unshifted_factor_ready)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    out_alpha[i] = static_cast<double>(buffers.core.rhs[i]);
                }
                std::vector<float>{}.swap(buffers.core.rhs);
                std::vector<float>{}.swap(buffers.core.exp_buf);
                buffers.core.nested = NestedBuffers<float>{};

                const double diag_offset = likelihood.noise_variance();
                GprRefineSystem<P> sys(
                    spec, kernel_exp, x_f64, y_f64, buffers.core.k_mut(), MatrixRef<float>{}, diag_offset
                );
                auto summary = refine_reevaluate_with_krylov(sys, out_alpha, 0.0);
                if (summary.has_value())
                {
                    if (sys.used_f64_fallback())
                    {
                        auto mll = compute_mll_from_factor_and_alpha(buffers.core.k_ref(), y_f64, out_alpha);
                        if (!mll.has_value())
                        {
                            return std::unexpected(std::move(mll).error());
                        }
                        in_out_mll = *mll;
                    }
                    buffers.core.factor_jitter = 0.0;
                    out_summary = *summary;
                    return {};
                }
                if (summary.error().code() != GprErrorCode::CholeskyFailed)
                {
                    return std::unexpected(std::move(summary).error());
                }
            }
            else
            {
                std::vector<float>{}.swap(buffers.core.rhs);
            }

            std::vector<double> jitter_candidates{0.0};
            JitterSchedule retry_sched = jitter_policy.retry_schedule();
            while (auto next_j = retry_sched.next())
            {
                jitter_candidates.push_back(*next_j);
            }

            std::vector<float> init_tmp_f32(n, 0.0F);
            GprError last_chol_err = GprError::cholesky_failed(0.0, n, CholeskyStage::Fit);

            for (const double j : jitter_candidates)
            {
                if (buffers.core.exp_buf.size() < n * n)
                {
                    buffers.core.exp_buf.assign(n * n, 0.0F);
                }
                if (buffers.core.nested.depth() < kernel_f32.nested_depth() || buffers.core.nested.rows() < n
                    || buffers.core.nested.cols() < n)
                {
                    buffers.core.nested.resize(kernel_f32.nested_depth(), n, n);
                }

                if (auto ap = kernel_f32.apply_points(
                        x_f32, buffers.core.k_mut(), Triangle::Lower, buffers.core.exp_mut(), buffers.core.nested
                    );
                    !ap.has_value())
                {
                    return ap;
                }
                const double diag_offset = likelihood.noise_variance() + j;
                if (auto dg = linalg::add_to_diag(buffers.core.k_mut(), static_cast<float>(diag_offset));
                    !dg.has_value())
                {
                    return dg;
                }

                double precond_shift = 0.0;
                auto precond_res =
                    factor_float_with_preconditioner_shift(buffers.core.k_mut(), buffers.core.exp_mut(), precond_shift);

                // Release n x n float scratch prior to Krylov or full-double fallback solves.
                std::vector<float>{}.swap(buffers.core.exp_buf);
                buffers.core.nested = NestedBuffers<float>{};

                GprRefineSystem<P> sys(
                    spec, kernel_exp, x_f64, y_f64, buffers.core.k_mut(), MatrixRef<float>{}, diag_offset
                );

                if (precond_res.has_value())
                {
                    std::optional<double> float_solve_mll;
                    if (precond_shift == 0.0)
                    {
                        std::vector<float> y_f32(n);
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            y_f32[i] = static_cast<float>(y_f64[i]);
                            init_tmp_f32[i] = y_f32[i];
                        }
                        if (auto sv = linalg::solve_llt_in_place(buffers.core.k_ref(), std::span<float>(init_tmp_f32));
                            sv.has_value())
                        {
                            auto logdet_f32 = linalg::log_det_from_l(buffers.core.k_ref());
                            auto y_alpha_f32 =
                                linalg::dot(std::span<const float>(y_f32), std::span<const float>(init_tmp_f32));
                            if (logdet_f32.has_value() && y_alpha_f32.has_value())
                            {
                                const double val = 0.5
                                    * (*y_alpha_f32 + *logdet_f32
                                        + (static_cast<double>(n) * std::log(2.0 * std::numbers::pi)));
                                if (std::isfinite(val))
                                {
                                    float_solve_mll = val;
                                }
                            }
                        }
                    }
                    if (auto init_sv = apply_scaled_f32_preconditioner(
                            buffers.core.k_ref(), y_f64, std::span<float>(init_tmp_f32), out_alpha
                        );
                        !init_sv.has_value())
                    {
                        std::ranges::fill(out_alpha, 0.0);
                    }
                    auto summary = refine_reevaluate_with_krylov(sys, out_alpha, precond_shift);
                    if (summary.has_value())
                    {
                        if (!sys.used_f64_fallback() && float_solve_mll.has_value())
                        {
                            in_out_mll = *float_solve_mll;
                        }
                        else
                        {
                            auto mll = compute_mll_from_factor_and_alpha(buffers.core.k_ref(), y_f64, out_alpha);
                            if (!mll.has_value())
                            {
                                return std::unexpected(std::move(mll).error());
                            }
                            in_out_mll = *mll;
                        }
                        buffers.core.factor_jitter = j;
                        out_summary = *summary;
                        return {};
                    }
                    if (summary.error().code() != GprErrorCode::CholeskyFailed)
                    {
                        return std::unexpected(std::move(summary).error());
                    }
                    last_chol_err = GprError::cholesky_failed(j, n, CholeskyStage::Fit);
                }
                else
                {
                    if (precond_res.error().code() != GprErrorCode::CholeskyFailed)
                    {
                        return precond_res;
                    }
                    auto fb = sys.fallback(out_alpha);
                    if (fb.has_value())
                    {
                        auto mll = compute_mll_from_factor_and_alpha(buffers.core.k_ref(), y_f64, out_alpha);
                        if (!mll.has_value())
                        {
                            return std::unexpected(std::move(mll).error());
                        }
                        buffers.core.factor_jitter = j;
                        in_out_mll = *mll;
                        out_summary = RefinementSummary{
                            .status = RefinementStatus::MaxCorrectionsReached,
                            .iterations = 0U,
                            .final_relative_residual = 0.0,
                            .preconditioner_shift = 0.0,
                            .used_krylov = false,
                        };
                        return {};
                    }
                    if (fb.error().code() != GprErrorCode::CholeskyFailed)
                    {
                        return fb;
                    }
                    last_chol_err = GprError::cholesky_failed(j, n, CholeskyStage::Fit);
                }
            }

            return std::unexpected(std::move(last_chol_err));
        }

        /// @brief Copies a strided column-major double or float MatrixRef into a dense double vector while checking
        /// finiteness.
        /// @details Rejects empty or null inputs, guards rows * cols multiplication overflow, and checks every
        /// matrix element for NaN or Inf before widening into @p dst.
        /// @param[in] src_var Source double or float column-major matrix view.
        /// @param[out] dst Destination contiguous column-major double buffer resized to rows * cols.
        /// @return Empty expected on success, or EmptyInput, SizeOverflow, or NonFiniteInput on invalid data.
        [[nodiscard]] std::expected<void, GprError> copy_double_matrix_col_major(
            PointMatrixRef src_var, std::vector<double>& dst
        )
        {
            return std::visit(
                [&dst](auto src) -> std::expected<void, GprError>
            {
                if (src.is_empty() || src.data() == nullptr)
                {
                    return std::unexpected(GprError::empty_input());
                }
                auto nm = checked_mul(src.rows(), src.cols());
                if (!nm.has_value())
                {
                    return std::unexpected(std::move(nm).error());
                }
                dst.resize(*nm);
                const std::size_t rows = src.rows();
                const std::size_t cols = src.cols();
                for (std::size_t col = 0; col < cols; ++col)
                {
                    for (std::size_t row = 0; row < rows; ++row)
                    {
                        const auto val = static_cast<double>(src(row, col));
                        if (!std::isfinite(val))
                        {
                            return std::unexpected(GprError::non_finite_input());
                        }
                        dst[(col * rows) + row] = val;
                    }
                }
                return {};
            }, src_var
            );
        }

        /// @brief Validates and widens a double or float target span into a dense double vector.
        /// @param[in] y_var Source double or float target span.
        /// @param[out] dst Destination double vector.
        /// @return Empty expected on success, or NonFiniteInput if any element is non-finite.
        [[nodiscard]] std::expected<void, GprError> copy_double_targets(TargetSpanRef y_var, std::vector<double>& dst)
        {
            return std::visit(
                [&dst](auto y) -> std::expected<void, GprError>
            {
                dst.resize(y.size());
                for (std::size_t i = 0; i < y.size(); ++i)
                {
                    const auto val = static_cast<double>(y[i]);
                    if (!std::isfinite(val))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                    dst[i] = val;
                }
                return {};
            }, y_var
            );
        }

    }  // namespace

    /// @brief Validates training inputs, fits transforms, optimizes or factors hyperparameters, and assembles
    /// FittedPayload<P>.
    /// @tparam P Precision policy controlling Storage and Refine arithmetic.
    /// @param[in] cfg Borrowed trainer configuration and transform prototypes.
    /// @param[in] x Column-major training feature matrix (n x d) in double or float precision.
    /// @param[in] y Training target vector of length n in double or float precision.
    /// @param[in] optimize_cb Callback invoking the concrete optimizer when cfg.optimize is true.
    /// @param[in,out] optimize_ctx Opaque pointer forwarded to optimize_cb.
    /// @return Populated FittedPayload<P> on success, or GprError on any validation, optimization, or solve error.
    template <PrecisionPolicyConcept P>
    std::expected<FittedPayload<P>, GprError> train_or_factor_payload(
        const TrainConfig<P>& cfg, PointMatrixRef x, TargetSpanRef y, OptimizeCallback optimize_cb, void* optimize_ctx
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        std::size_t n = 0U;
        std::size_t d = 0U;
        const bool x_empty = std::visit(
            [&n, &d](auto mat) noexcept
        {
            n = mat.rows();
            d = mat.cols();
            return mat.is_empty() || mat.data() == nullptr;
        }, x
        );
        const std::size_t y_size = std::visit([](auto sp) noexcept { return sp.size(); }, y);
        if (x_empty || y_size == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (y_size != n)
        {
            return std::unexpected(GprError::length_mismatch("target length must match training row count"));
        }
        std::vector<double> x_f64;
        if (auto cp = copy_double_matrix_col_major(x, x_f64); !cp.has_value())
        {
            return std::unexpected(std::move(cp).error());
        }
        std::vector<double> x_raw = x_f64;
        std::vector<double> y_raw;
        if (auto cy = copy_double_targets(y, y_raw); !cy.has_value())
        {
            return std::unexpected(std::move(cy).error());
        }

        // Fit and apply input and target transforms in double precision while retaining raw snapshots.
        auto fitted_in = cfg.input_transform.fit(x_f64, n, d);
        if (!fitted_in.has_value())
        {
            return std::unexpected(std::move(fitted_in).error());
        }
        if (auto ap = (*fitted_in)->apply(x_f64, n, d); !ap.has_value())
        {
            return std::unexpected(std::move(ap).error());
        }

        auto fitted_tgt = cfg.target_transform.fit(y_raw);
        if (!fitted_tgt.has_value())
        {
            return std::unexpected(std::move(fitted_tgt).error());
        }
        std::vector<double> y_f64 = y_raw;
        if (auto tf = (*fitted_tgt)->transform(y_f64); !tf.has_value())
        {
            return std::unexpected(std::move(tf).error());
        }

        // Downcast transformed training tensors to Storage and Refine scalar types for training.
        std::vector<Storage> x_storage(x_f64.size());
        for (std::size_t i = 0; i < x_f64.size(); ++i)
        {
            x_storage[i] = static_cast<Storage>(x_f64[i]);
        }
        std::vector<Storage> y_storage(n);
        std::vector<Refine> y_refine(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            y_storage[i] = static_cast<Storage>(y_f64[i]);
            y_refine[i] = static_cast<Refine>(y_f64[i]);
        }

        auto compiled = std::make_shared<CompiledKernel<Storage>>(CompiledKernel<Storage>::from_spec(cfg.kernel_spec));
        compiled->set_kernel_exp(cfg.kernel_exp);

        const FitPolicy effective_fit_policy = cfg.optimize ? cfg.fit_policy : FitPolicy::prefer_memory();
        // Value-only optimizers never construct W. Keep the requested distance and incremental leaf caches.
        const bool use_leaf_cache =
            cfg.use_change_indices && effective_fit_policy.cholesky_buffer == CholeskyBuffer::Retain;
        const auto buffer_policy = cfg.value_only ? CholeskyBuffer::Reuse : effective_fit_policy.cholesky_buffer;
        auto buffers = FitBuffers<P>::create(n, effective_fit_policy.distance_cache, buffer_policy);
        if (!buffers.has_value())
        {
            return std::unexpected(std::move(buffers).error());
        }

        const MatrixRef<Storage> x_train_ref(x_storage.data(), n, d, n);
        GaussianLikelihood likelihood = cfg.likelihood;
        const std::size_t nk = compiled->num_params();
        optimizer::OptimizerResult opt_res;
        double final_mll = 0.0;
        bool has_unshifted_opt_factor = false;
        constexpr bool is_mixed_reeval = std::same_as<P, MixedPrecision<ReevaluateKernel>>;
        {
            GprObjective<P> obj(*compiled, likelihood, x_train_ref, y_storage, cfg.jitter, *buffers, use_leaf_cache);

            const std::size_t total_params = obj.dim();
            std::vector<double> init_params(total_params);
            if (auto gp = compiled->get_params(std::span<double>(init_params.data(), nk)); !gp.has_value())
            {
                return std::unexpected(std::move(gp).error());
            }
            if (auto lp = likelihood.get_params(std::span<double>(init_params.data() + nk, 1U)); !lp.has_value())
            {
                return std::unexpected(std::move(lp).error());
            }

            // Run hyperparameter optimization or perform a single objective evaluation (deferring ReevaluateKernel
            // factor solves to the shifted-preconditioner / FGMRES-IR path below).
            if (cfg.optimize)
            {
                auto r = optimize_cb(optimize_ctx, obj, init_params);
                if (!r.has_value())
                {
                    return std::unexpected(std::move(r).error());
                }
                opt_res = std::move(*r);

                if (obj.matches_last_factor(opt_res.params))
                {
                    final_mll = obj.last_factored_mll();
                    has_unshifted_opt_factor = (buffers->core.factor_jitter == 0.0);
                }
                else
                {
                    if (auto sp = compiled->set_params(std::span<const double>(opt_res.params.data(), nk));
                        !sp.has_value())
                    {
                        return std::unexpected(std::move(sp).error());
                    }
                    if (auto lp = likelihood.set_params(std::span<const double>(opt_res.params.data() + nk, 1U));
                        !lp.has_value())
                    {
                        return std::unexpected(std::move(lp).error());
                    }
                    if constexpr (!is_mixed_reeval)
                    {
                        auto mll_res =
                            factor_and_mll(*compiled, likelihood, x_train_ref, y_storage, cfg.jitter, *buffers, 0U);
                        if (!mll_res.has_value())
                        {
                            return std::unexpected(std::move(mll_res).error());
                        }
                        final_mll = *mll_res;
                    }
                }
            }
            else if constexpr (is_mixed_reeval)
            {
                opt_res = optimizer::OptimizerResult{
                    .params = init_params,
                    .value = 0.0,
                    .iterations = 0U,
                    .converged = true,
                };
            }
            else
            {
                auto v = obj.value(init_params);
                if (!v.has_value())
                {
                    return std::unexpected(std::move(v).error());
                }
                opt_res = optimizer::OptimizerResult{
                    .params = init_params,
                    .value = *v,
                    .iterations = 0U,
                    .converged = true,
                };
                final_mll = *v;
            }
        }

        KernelSpec updated_spec = cfg.kernel_spec;
        if (auto sp = updated_spec.set_params(std::span<const double>(opt_res.params.data(), nk)); !sp.has_value())
        {
            return std::unexpected(std::move(sp).error());
        }

        std::vector<Refine> alpha_out(n);
        std::optional<RefinementSummary> ref_summary;

        // Refine weights alpha in double precision under MixedPrecision policies, or cast directly otherwise.
        if constexpr (is_mixed_reeval)
        {
            const MatrixRef<double> x_f64_ref(x_f64.data(), n, d, n);
            if (auto ref_res = solve_mixed_reevaluate_payload<P>(
                    updated_spec, cfg.kernel_exp, *compiled, likelihood, cfg.jitter, x_f64_ref, x_train_ref, y_f64,
                    *buffers, cfg.optimize && has_unshifted_opt_factor, alpha_out, ref_summary, final_mll
                );
                !ref_res.has_value())
            {
                return std::unexpected(std::move(ref_res).error());
            }
            if (!cfg.optimize)
            {
                opt_res.value = final_mll;
            }
        }
        else if constexpr (P::refines_in_f64)
        {
            const double diag_offset = likelihood.noise_variance() + buffers->core.factor_jitter;
            const MatrixRef<double> x_f64_ref(x_f64.data(), n, d, n);
            if (auto ref_res = run_mixed_precision_refinement<P>(
                    updated_spec, cfg.kernel_exp, *compiled, x_f64_ref, x_train_ref, y_f64, *buffers, diag_offset,
                    alpha_out, ref_summary
                );
                !ref_res.has_value())
            {
                return std::unexpected(std::move(ref_res).error());
            }
        }
        else
        {
            if constexpr (std::same_as<Refine, Storage>)
            {
                std::copy_n(buffers->core.rhs.data(), n, alpha_out.data());
            }
            else
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    alpha_out[i] = static_cast<Refine>(buffers->core.rhs[i]);
                }
            }
        }

        std::shared_ptr<const CompiledKernel<double>> compiled64;
        std::vector<double> x_train_f64_out;
        if constexpr (is_mixed_reeval)
        {
            auto ck64 = std::make_shared<CompiledKernel<double>>(CompiledKernel<double>::from_spec(updated_spec));
            ck64->set_kernel_exp(cfg.kernel_exp);
            compiled64 = std::move(ck64);
            x_train_f64_out = std::move(x_f64);
        }

        return FittedPayload<P>{
            .kernel_spec = std::move(updated_spec),
            .likelihood = likelihood,
            .compiled_kernel = std::move(compiled),
            .compiled_kernel_f64 = std::move(compiled64),
            .input_transform = std::move(*fitted_in),
            .target_transform = std::move(*fitted_tgt),
            .x_raw = std::move(x_raw),
            .y_raw = std::move(y_raw),
            .x_train = std::move(x_storage),
            .x_train_f64 = std::move(x_train_f64_out),
            .y_transformed = std::move(y_refine),
            .l_factor = std::move(buffers->core.k_matrix),
            .alpha = std::move(alpha_out),
            .n_train = n,
            .n_features = d,
            .factor_jitter = buffers->core.factor_jitter,
            .neg_log_marginal_likelihood = final_mll,
            .refinement_summary = ref_summary,
            .optimizer_result = std::move(opt_res),
        };
    }

    namespace
    {

        /// @brief Validates, transforms, and casts query coordinates xs into workspace buffers.
        /// @details Ensures workspace capacity, verifies feature count and finiteness, applies the fitted
        /// input transform in double precision, and casts transformed features to Storage precision.
        /// @tparam P Precision policy defining Storage scalar type.
        /// @param[in] payload Fitted model payload supplying n_train, n_features, compiled_kernel, and input_transform.
        /// @param[in] xs_var Column-major query feature matrix (m x d), double or float.
        /// @param[in,out] workspace Mutable query workspace reused across calls.
        /// @return Column-major MatrixRef<Storage> backed by workspace.query_x, or GprError on invalid query.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<MatrixRef<typename P::Storage>, GprError> prepare_query_matrix(
            const FittedPayload<P>& payload, PointMatrixRef xs_var, QueryWorkspace<P>& workspace
        )
        {
            using Storage = P::Storage;

            return std::visit(
                [&](auto xs) -> std::expected<MatrixRef<Storage>, GprError>
            {
                if (xs.is_empty() || xs.data() == nullptr)
                {
                    return std::unexpected(GprError::empty_input());
                }
                const std::size_t m = xs.rows();
                const std::size_t d = xs.cols();
                if (d != payload.n_features)
                {
                    return std::unexpected(GprError::dimension_mismatch(d, payload.n_features));
                }
                // Grow reusable query workspace only when m, n, d, or nested depth exceeds current capacity.
                if (auto ens = workspace.ensure(payload.n_train, m, d, payload.compiled_kernel->nested_depth());
                    !ens.has_value())
                {
                    return std::unexpected(std::move(ens).error());
                }

                // Copy/widen query coordinates column-major into double staging buffer while checking finiteness.
                for (std::size_t col = 0; col < d; ++col)
                {
                    for (std::size_t row = 0; row < m; ++row)
                    {
                        const auto val = static_cast<double>(xs(row, col));
                        if (!std::isfinite(val))
                        {
                            return std::unexpected(GprError::non_finite_input());
                        }
                        workspace.query_xs[(col * m) + row] = val;
                    }
                }

                // Apply fitted input transform in double precision before casting down to Storage. The workspace
                // only grows, so restrict the span to exactly m * d values (a smaller batch may follow a larger one).
                const std::span<double> query_span(workspace.query_xs.data(), m * d);
                if (auto ap = payload.input_transform->apply(query_span, m, d); !ap.has_value())
                {
                    return std::unexpected(std::move(ap).error());
                }

                const std::size_t md = m * d;
                for (std::size_t i = 0; i < md; ++i)
                {
                    workspace.query_x[i] = static_cast<Storage>(workspace.query_xs[i]);
                }
                return MatrixRef<Storage>(workspace.query_x.data(), m, d, m);
            }, xs_var
            );
        }

        /// @brief Applies inverse target transforms to a Prediction<T> using a preallocated double scratch span.
        /// @details Converts float means/variances to double in buf, invokes target_tf inverse transforms,
        /// and casts the unscaled outputs back to T without heap allocations.
        /// @tparam T Prediction scalar type (typically float when using scratch conversion).
        /// @param[in] target_tf Fitted target transform applied in reverse.
        /// @param[in,out] pred In-out Prediction<T> scaled back to the original target units.
        /// @param[in,out] buf Preallocated double scratch span of length >= pred.mean.size().
        /// @return Empty expected on success, or GprError on inverse transform failure.
        template <typename T>
        [[nodiscard]] std::expected<void, GprError> apply_target_inverse_with_buf(
            const transform::FittedTargetTransform& target_tf, Prediction<T>& pred, std::span<double> buf
        )
        {
            const std::size_t m = pred.mean.size();
            for (std::size_t i = 0; i < m; ++i)
            {
                buf[i] = static_cast<double>(pred.mean[i]);
            }
            if (auto m_res = target_tf.inverse_transform_mean(buf); !m_res.has_value())
            {
                return m_res;
            }
            for (std::size_t i = 0; i < m; ++i)
            {
                pred.mean[i] = static_cast<T>(buf[i]);
                buf[i] = static_cast<double>(pred.variance[i]);
            }
            if (auto v_res = target_tf.inverse_transform_variance(buf); !v_res.has_value())
            {
                return v_res;
            }
            for (std::size_t i = 0; i < m; ++i)
            {
                pred.variance[i] = static_cast<T>(buf[i]);
            }
            return {};
        }

        /// @brief Applies inverse target transforms to Prediction<T> on the original target scale.
        /// @details Operates directly in-place for T = double, or via scratch_d (or a fallback vector if
        /// scratch_d is smaller than m) for single-precision predictions.
        /// @tparam T Prediction scalar type (double or float).
        /// @param[in] target_tf Fitted target transform applied in reverse.
        /// @param[in,out] pred In-out Prediction<T> scaled back to original units.
        /// @param[in,out] scratch_d Optional reusable double scratch buffer for zero-allocation float conversion.
        /// @return Empty expected on success, or GprError on inverse transform failure.
        template <typename T>
        [[nodiscard]] std::expected<void, GprError> apply_target_inverse_prediction(
            const transform::FittedTargetTransform& target_tf, Prediction<T>& pred, std::span<double> scratch_d = {}
        )
        {
            if constexpr (std::same_as<T, double>)
            {
                if (auto m_res = target_tf.inverse_transform_mean(pred.mean); !m_res.has_value())
                {
                    return m_res;
                }
                return target_tf.inverse_transform_variance(pred.variance);
            }
            else
            {
                // Reuse caller scratch buffer (workspace.query_xs) when large enough to avoid allocations.
                const std::size_t m = pred.mean.size();
                if (scratch_d.size() >= m)
                {
                    return apply_target_inverse_with_buf(target_tf, pred, scratch_d.subspan(0, m));
                }
                std::vector<double> owned(m);
                return apply_target_inverse_with_buf(target_tf, pred, owned);
            }
        }

    }  // namespace

    /// @brief Computes double-precision posterior mean contributions mu = K_*^T * alpha in width-32 blocks.
    /// @details Evaluates cross-covariance K(X_train_f64, X_query_f64) in double-precision column blocks of
    /// width at most refinement_column_block (32) using pre-sized workspace buffers and deterministic
    /// ascending-row dot products.
    /// @param[in] kernel_f64 Double-precision compiled kernel evaluator.
    /// @param[in] x_train_f64 Double-precision transformed training features (n x d, column-major).
    /// @param[in] x_query_f64 Double-precision transformed query features (m x d, column-major).
    /// @param[in] alpha Double-precision weights vector of length n.
    /// @param[in,out] ws Reusable query workspace holding double column-block scratch.
    /// @param[out] out_mean Destination transformed-scale posterior mean vector of length m.
    /// @return Empty expected on success, or GprError on empty, mismatched, or failing kernel evaluation.
    std::expected<void, GprError> compute_reeval_cross_mean_f64(
        const CompiledKernel<double>& kernel_f64,
        MatrixRef<double> x_train_f64,
        MatrixRef<double> x_query_f64,
        std::span<const double> alpha,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>& ws,
        std::span<double> out_mean
    )
    {
        const std::size_t n = x_train_f64.rows();
        const std::size_t d = x_train_f64.cols();
        const std::size_t m = x_query_f64.rows();
        if (n == 0U || m == 0U || d == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (x_query_f64.cols() != d)
        {
            return std::unexpected(GprError::dimension_mismatch(x_query_f64.cols(), d));
        }
        if (alpha.size() != n || out_mean.size() != m)
        {
            return std::unexpected(GprError::length_mismatch("alpha or mean size mismatch in f64 cross evaluation"));
        }

        const std::size_t block_cols = std::min(m, refinement_column_block);
        if (ws.query_x_block_f64.size() < block_cols * d)
        {
            ws.query_x_block_f64.resize(block_cols * d);
        }
        if (ws.query_k_block_f64.size() < n * block_cols)
        {
            ws.query_k_block_f64.resize(n * block_cols);
        }
        if (ws.query_scratch_block_f64.size() < n * block_cols)
        {
            ws.query_scratch_block_f64.resize(n * block_cols);
        }
        if (ws.query_nested_f64.depth() < kernel_f64.nested_depth() || ws.query_nested_f64.rows() < n
            || ws.query_nested_f64.cols() < block_cols)
        {
            ws.query_nested_f64.resize(
                std::max(ws.query_nested_f64.depth(), kernel_f64.nested_depth()),
                std::max(ws.query_nested_f64.rows(), n), std::max(ws.query_nested_f64.cols(), block_cols)
            );
        }

        for (std::size_t col_begin = 0U; col_begin < m; col_begin += block_cols)
        {
            const std::size_t cols_cur = std::min(block_cols, m - col_begin);
            for (std::size_t feat = 0U; feat < d; ++feat)
            {
                for (std::size_t c = 0U; c < cols_cur; ++c)
                {
                    ws.query_x_block_f64[(feat * cols_cur) + c] = x_query_f64(col_begin + c, feat);
                }
            }
            const MatrixRef<double> xqb_ref(ws.query_x_block_f64.data(), cols_cur, d, cols_cur);
            const MatrixMut<double> kb_mut(ws.query_k_block_f64.data(), n, cols_cur, n);
            const MatrixMut<double> sb_mut(ws.query_scratch_block_f64.data(), n, cols_cur, n);
            if (auto ap = kernel_f64.apply_cross_points(x_train_f64, xqb_ref, kb_mut, sb_mut, ws.query_nested_f64);
                !ap.has_value())
            {
                return ap;
            }
            const auto cols_cur_s = static_cast<std::ptrdiff_t>(cols_cur);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n * cols_cur) >= 65536U)
#endif
            for (std::ptrdiff_t c_s = 0; c_s < cols_cur_s; ++c_s)
            {
                const auto c = static_cast<std::size_t>(c_s);
                const double* col_ptr = ws.query_k_block_f64.data() + (c * n);
                out_mean[col_begin + c] = dot_f64(std::span<const double>(col_ptr, n), alpha);
            }
        }
        return {};
    }

    /// @brief Evaluates posterior prediction into out and applies inverse target transforms.
    /// @tparam P Precision policy matching payload.
    /// @param[in] payload Fitted model payload.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[in] options Prediction variance options.
    /// @param[in,out] workspace Mutable query workspace reused across calls.
    /// @param[out] out Destination prediction struct populated on the original target scale.
    /// @return Empty expected on success, or GprError on validation, prediction, or inverse scaling failure.
    /// @retval GprErrorCode::EmptyInput When xs is empty or null.
    /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
    /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
    /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> fitted_predict_into(
        const FittedPayload<P>& payload,
        PointMatrixRef xs,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        Prediction<typename P::Refine>& out
    )
    {
        using Storage = P::Storage;
        // Validate and transform xs into workspace.query_x, then run prediction and inverse scaling.
        auto xq_ref = prepare_query_matrix(payload, xs, workspace);
        if (!xq_ref.has_value())
        {
            return std::unexpected(std::move(xq_ref).error());
        }
        const MatrixRef<Storage> x_train_ref(
            payload.x_train.data(), payload.n_train, payload.n_features, payload.n_train
        );
        const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train, payload.n_train);
        if (auto res = predict_diagonal_into<P>(
                *payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref, payload.alpha, *xq_ref, options,
                workspace, out, linalg::detail::FactorValidation::AlreadyValidated
            );
            !res.has_value())
        {
            return res;
        }
        if constexpr (std::same_as<P, MixedPrecision<ReevaluateKernel>>)
        {
            const std::size_t m = xq_ref->rows();
            if (payload.compiled_kernel_f64 && payload.x_train_f64.size() == payload.n_train * payload.n_features)
            {
                const MatrixRef<double> x_train_f64_ref(
                    payload.x_train_f64.data(), payload.n_train, payload.n_features, payload.n_train
                );
                const MatrixRef<double> xq_f64_ref(workspace.query_xs.data(), m, payload.n_features, m);
                if (auto mr = compute_reeval_cross_mean_f64(
                        *payload.compiled_kernel_f64, x_train_f64_ref, xq_f64_ref, payload.alpha, workspace, out.mean
                    );
                    !mr.has_value())
                {
                    return mr;
                }
            }
        }
        return apply_target_inverse_prediction(*payload.target_transform, out, workspace.query_xs);
    }

    /// @brief Evaluates posterior mean and full m x m covariance matrix on the original target scale.
    /// @tparam P Precision policy matching payload.
    /// @param[in] payload Fitted model payload.
    /// @param[in] xs Column-major query feature matrix (m x d) in double or float precision.
    /// @param[in] options Prediction variance options.
    /// @param[in,out] workspace Mutable query workspace.
    /// @return PredictiveCovariance<Refine> on the original target scale, or GprError on failure.
    template <PrecisionPolicyConcept P>
    std::expected<PredictiveCovariance<typename P::Refine>, GprError> fitted_predict_covariance(
        const FittedPayload<P>& payload, PointMatrixRef xs, const PredictOptions& options, QueryWorkspace<P>& workspace
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        auto xq_ref = prepare_query_matrix(payload, xs, workspace);
        if (!xq_ref.has_value())
        {
            return std::unexpected(std::move(xq_ref).error());
        }
        const MatrixRef<Storage> x_train_ref(
            payload.x_train.data(), payload.n_train, payload.n_features, payload.n_train
        );
        const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train, payload.n_train);
        auto cov = predict_covariance<P>(
            *payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref, payload.alpha, *xq_ref, options, workspace
        );
        if (!cov.has_value())
        {
            return cov;
        }
        if constexpr (std::same_as<P, MixedPrecision<ReevaluateKernel>>)
        {
            const std::size_t m = xq_ref->rows();
            if (payload.compiled_kernel_f64 && payload.x_train_f64.size() == payload.n_train * payload.n_features)
            {
                const MatrixRef<double> x_train_f64_ref(
                    payload.x_train_f64.data(), payload.n_train, payload.n_features, payload.n_train
                );
                const MatrixRef<double> xq_f64_ref(workspace.query_xs.data(), m, payload.n_features, m);
                if (auto mr = compute_reeval_cross_mean_f64(
                        *payload.compiled_kernel_f64, x_train_f64_ref, xq_f64_ref, payload.alpha, workspace, cov->mean
                    );
                    !mr.has_value())
                {
                    return std::unexpected(std::move(mr).error());
                }
            }
        }

        // Apply inverse target scaling to mean and covariance in double precision before returning.
        if constexpr (std::same_as<Refine, double>)
        {
            if (auto mr = payload.target_transform->inverse_transform_mean(cov->mean); !mr.has_value())
            {
                return std::unexpected(std::move(mr).error());
            }
            if (auto cr = payload.target_transform->inverse_transform_covariance(cov->covariance); !cr.has_value())
            {
                return std::unexpected(std::move(cr).error());
            }
        }
        else
        {
            std::vector<double> mean_d(cov->mean.begin(), cov->mean.end());
            std::vector<double> cov_d(cov->covariance.begin(), cov->covariance.end());
            if (auto mr = payload.target_transform->inverse_transform_mean(mean_d); !mr.has_value())
            {
                return std::unexpected(std::move(mr).error());
            }
            if (auto cr = payload.target_transform->inverse_transform_covariance(cov_d); !cr.has_value())
            {
                return std::unexpected(std::move(cr).error());
            }
            for (std::size_t i = 0; i < mean_d.size(); ++i)
            {
                cov->mean[i] = static_cast<Refine>(mean_d[i]);
            }
            for (std::size_t i = 0; i < cov_d.size(); ++i)
            {
                cov->covariance[i] = static_cast<Refine>(cov_d[i]);
            }
        }
        return cov;
    }

    /// @brief Draws deterministic posterior function realizations at xs on the original target scale.
    /// @tparam P Precision policy matching payload.
    /// @param[in] payload Fitted model payload.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[in] n_draws Number of posterior draws to generate.
    /// @param[in] seed Deterministic PRNG seed.
    /// @param[in] options Prediction variance options.
    /// @param[in,out] workspace Mutable query workspace.
    /// @param[in] jitter Optional override jitter policy for predictive covariance factorization.
    /// @return Column-major sample buffer (m x n_draws) on the original scale, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When xs is empty or null.
    /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
    /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
    /// @retval GprErrorCode::CholeskyFailed When predictive covariance factorization fails.
    template <PrecisionPolicyConcept P>
    std::expected<std::vector<typename P::Refine>, GprError> fitted_sample(
        const FittedPayload<P>& payload,
        PointMatrixRef xs,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace,
        std::optional<JitterPolicy> jitter
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        auto xq_ref = prepare_query_matrix(payload, xs, workspace);
        if (!xq_ref.has_value())
        {
            return std::unexpected(std::move(xq_ref).error());
        }
        const MatrixRef<Storage> x_train_ref(
            payload.x_train.data(), payload.n_train, payload.n_features, payload.n_train
        );
        const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train, payload.n_train);
        std::expected<std::vector<Refine>, GprError> samples = std::unexpected(GprError::empty_input());
        if constexpr (std::same_as<P, MixedPrecision<ReevaluateKernel>>)
        {
            if (payload.compiled_kernel_f64 && payload.x_train_f64.size() == payload.n_train * payload.n_features)
            {
                auto cov = predict_covariance<P>(
                    *payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref, payload.alpha, *xq_ref, options,
                    workspace
                );
                if (!cov.has_value())
                {
                    return std::unexpected(std::move(cov).error());
                }
                const std::size_t m_pts = xq_ref->rows();
                const MatrixRef<double> x_train_f64_ref(
                    payload.x_train_f64.data(), payload.n_train, payload.n_features, payload.n_train
                );
                const MatrixRef<double> xq_f64_ref(workspace.query_xs.data(), m_pts, payload.n_features, m_pts);
                if (auto mr = compute_reeval_cross_mean_f64(
                        *payload.compiled_kernel_f64, x_train_f64_ref, xq_f64_ref, payload.alpha, workspace, cov->mean
                    );
                    !mr.has_value())
                {
                    return std::unexpected(std::move(mr).error());
                }
                const JitterPolicy jit = jitter.has_value() ? *jitter : default_sample_jitter();
                samples = draw_samples_from_covariance<double>(std::move(*cov), n_draws, seed, jit);
            }
            else
            {
                samples = sample<P>(
                    *payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref, payload.alpha, *xq_ref, n_draws,
                    seed, options, workspace, jitter
                );
            }
        }
        else
        {
            samples = sample<P>(
                *payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref, payload.alpha, *xq_ref, n_draws, seed,
                options, workspace, jitter
            );
        }
        if (!samples.has_value())
        {
            return samples;
        }
        // Apply inverse mean transform to each column draw independently.
        const std::size_t m = xq_ref->rows();
        if constexpr (std::same_as<Refine, double>)
        {
            for (std::size_t draw = 0; draw < n_draws; ++draw)
            {
                const std::span<double> slice(samples->data() + (draw * m), m);
                if (auto mr = payload.target_transform->inverse_transform_mean(slice); !mr.has_value())
                {
                    return std::unexpected(std::move(mr).error());
                }
            }
        }
        else
        {
            std::vector<double> buf(m);
            for (std::size_t draw = 0; draw < n_draws; ++draw)
            {
                Refine* ptr = samples->data() + (draw * m);
                for (std::size_t i = 0; i < m; ++i)
                {
                    buf[i] = static_cast<double>(ptr[i]);
                }
                if (auto mr = payload.target_transform->inverse_transform_mean(buf); !mr.has_value())
                {
                    return std::unexpected(std::move(mr).error());
                }
                for (std::size_t i = 0; i < m; ++i)
                {
                    ptr[i] = static_cast<Refine>(buf[i]);
                }
            }
        }
        return samples;
    }

    /// @brief Computes closed-form Leave-One-Out predictive moments on the original target scale.
    /// @tparam P Precision policy matching payload.
    /// @param[in] payload Fitted model payload.
    /// @param[in] options Prediction variance options.
    /// @return Prediction<Refine> on the original target scale, or GprError on failure.
    /// @retval GprErrorCode::NonFiniteKernelValue When inverse scaling produces non-finite moments.
    template <PrecisionPolicyConcept P>
    std::expected<Prediction<typename P::Refine>, GprError> fitted_loo_predict(
        const FittedPayload<P>& payload, const PredictOptions& options
    )
    {
        using Storage = P::Storage;
        const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train, payload.n_train);
        // Evaluate LOO in transformed space, then map mean and variance back to the original scale.
        auto loo = loo_predict<P>(payload.likelihood, l_ref, payload.alpha, payload.y_transformed, options);
        if (!loo.has_value())
        {
            return loo;
        }
        if (auto inv = apply_target_inverse_prediction(*payload.target_transform, *loo); !inv.has_value())
        {
            return std::unexpected(std::move(inv).error());
        }
        return loo;
    }

    template std::expected<FittedPayload<DoublePrecision>, GprError> train_or_factor_payload<DoublePrecision>(
        const TrainConfig<DoublePrecision>&, PointMatrixRef, TargetSpanRef, OptimizeCallback, void*
    );
    template std::expected<FittedPayload<SinglePrecision>, GprError> train_or_factor_payload<SinglePrecision>(
        const TrainConfig<SinglePrecision>&, PointMatrixRef, TargetSpanRef, OptimizeCallback, void*
    );
    template std::expected<FittedPayload<MixedPrecision<PromoteStorage>>, GprError> train_or_factor_payload<
        MixedPrecision<PromoteStorage>>(
        const TrainConfig<MixedPrecision<PromoteStorage>>&, PointMatrixRef, TargetSpanRef, OptimizeCallback, void*
    );
    template std::expected<FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError> train_or_factor_payload<
        MixedPrecision<ReevaluateKernel>>(
        const TrainConfig<MixedPrecision<ReevaluateKernel>>&, PointMatrixRef, TargetSpanRef, OptimizeCallback, void*
    );

    template std::expected<void, GprError> fitted_predict_into<DoublePrecision>(
        const FittedPayload<DoublePrecision>&,
        PointMatrixRef,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        Prediction<double>&
    );
    template std::expected<void, GprError> fitted_predict_into<SinglePrecision>(
        const FittedPayload<SinglePrecision>&,
        PointMatrixRef,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        Prediction<float>&
    );
    template std::expected<void, GprError> fitted_predict_into<MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&,
        PointMatrixRef,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        Prediction<double>&
    );
    template std::expected<void, GprError> fitted_predict_into<MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
        PointMatrixRef,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        Prediction<double>&
    );

    template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<DoublePrecision>(
        const FittedPayload<DoublePrecision>&, PointMatrixRef, const PredictOptions&, QueryWorkspace<DoublePrecision>&
    );
    template std::expected<PredictiveCovariance<float>, GprError> fitted_predict_covariance<SinglePrecision>(
        const FittedPayload<SinglePrecision>&, PointMatrixRef, const PredictOptions&, QueryWorkspace<SinglePrecision>&
    );
    template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
        MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&,
        PointMatrixRef,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&
    );
    template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
        MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
        PointMatrixRef,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
    );

    template std::expected<std::vector<double>, GprError> fitted_sample<DoublePrecision>(
        const FittedPayload<DoublePrecision>&,
        PointMatrixRef,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<float>, GprError> fitted_sample<SinglePrecision>(
        const FittedPayload<SinglePrecision>&,
        PointMatrixRef,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&,
        PointMatrixRef,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
        PointMatrixRef,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        std::optional<JitterPolicy>
    );

    template std::expected<Prediction<double>, GprError> fitted_loo_predict<DoublePrecision>(
        const FittedPayload<DoublePrecision>&, const PredictOptions&
    );
    template std::expected<Prediction<float>, GprError> fitted_loo_predict<SinglePrecision>(
        const FittedPayload<SinglePrecision>&, const PredictOptions&
    );
    template std::expected<Prediction<double>, GprError> fitted_loo_predict<MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&, const PredictOptions&
    );
    template std::expected<Prediction<double>, GprError> fitted_loo_predict<MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&, const PredictOptions&
    );

}  // namespace cxxgpr::detail
