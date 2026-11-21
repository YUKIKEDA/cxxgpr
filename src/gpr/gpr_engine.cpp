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
#include <vector>

#include "workspace/predict_diagonal.hpp"

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
                    (void)leaf_cache_.resize(kernel_.num_leaves(), x_train_.rows());
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
                // Full evaluation discards incremental leaf tracking so future change-based calls recompute cleanly.
                if (enable_leaf_cache_)
                {
                    leaf_cache_.invalidate();
                }
                auto res = factor_and_mll(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_, 0U);
                if (res.has_value())
                {
                    record_valid_factor(params, *res);
                }
                return res;
            }

            /// @brief Evaluates negative MLL incrementally when only changed_indices moved since the last step.
            /// @details Re-evaluates only kernel leaves touched by changed_indices via LeafCache, adds noise
            /// variance to the diagonal, factors L with jitter, and computes MLL directly.
            /// @param[in] params Full hyperparameter vector of length dim().
            /// @param[in] changed_indices Parameter indices modified relative to the cached evaluation.
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

                // Recompute only affected kernel leaves and reassemble lower triangle into buffers_.core.k_mut().
                const DistanceCache<Storage>* dist_ptr = buffers_.dist.has_value() ? &(*buffers_.dist) : nullptr;
                auto inc = leaf_cache_.evaluate_incremental(
                    kernel_, x_train_, dist_ptr, params, changed_indices, dim(), buffers_.core.k_mut(), Triangle::Lower,
                    buffers_.core.exp_mut(), buffers_.core.nested
                );
                if (!inc.has_value())
                {
                    return std::unexpected(std::move(inc).error());
                }

                // Add observation noise variance, apply jittered Cholesky, and solve L L^T alpha = y.
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
                // Assemble negative MLL using the identical formula and parenthesization as factor_and_mll().
                const double mll =
                    0.5 * (*y_alpha + *logdet + (static_cast<double>(n) * std::log(2.0 * std::numbers::pi)));
                if (!std::isfinite(mll))
                {
                    return std::unexpected(GprError::non_finite_kernel_value());
                }
                record_valid_factor(params, mll);
                return mll;
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

        /// @brief Computes double-precision residual r = y_f64 - B_f64 * w and ||B_f64||_inf using column
        /// blocks of width 32 without materializing the full n x n double matrix.
        /// @details Evaluates cross-kernel slices B_f64(:, col_begin:col_end) in blocks of at most
        /// refinement_column_block columns, adds diag_offset along the main diagonal, accumulates B_f64 * w
        /// and row absolute sums, and writes y_f64 - B_f64 * w into out_r.
        /// @param[in] kernel_f64 Double-precision compiled kernel evaluator.
        /// @param[in] x_f64 Double-precision transformed training features (n x d).
        /// @param[in] y_f64 Double-precision transformed training targets of length n.
        /// @param[in] diag_offset Combined observation noise variance and accepted jitter added to diagonal entries.
        /// @param[in] w Current candidate solution vector of length n in double precision.
        /// @param[out] out_r Output double-precision residual vector of length n.
        /// @return Infinity norm ||B_f64||_inf for backward-error stopping checks, or GprError on evaluation failure.
        /// @retval GprErrorCode::NonFiniteKernelValue When kernel evaluation produces non-finite entries.
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
            const std::size_t d = x_f64.cols();
            const std::size_t block_cols = std::min(refinement_column_block, n);

            // Allocate O(n * block_cols) scratch buffers instead of O(n^2) double storage.
            std::vector<double> x_block(block_cols * d);
            std::vector<double> k_block(n * block_cols);
            std::vector<double> scratch_block(n * block_cols);
            auto nested = kernel_f64.make_nested_buffers(n, block_cols);
            std::vector<double> matvec(n, 0.0);
            std::vector<double> row_abs_sum(n, 0.0);

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
                // Accumulate matrix-vector product B_f64 * w and row l1 sums across the block columns.
                for (std::size_t c = 0; c < cols_cur; ++c)
                {
                    const std::size_t global_col = col_begin + c;
                    const double w_col = w[global_col];
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        double entry = k_block[(c * n) + row];
                        if (row == global_col)
                        {
                            entry += diag_offset;
                        }
                        matvec[row] += entry * w_col;
                        row_abs_sum[row] += std::abs(entry);
                    }
                }
            }

            // Compute residual r = y_f64 - B_f64 * w and return ||B_f64||_inf.
            for (std::size_t row = 0; row < n; ++row)
            {
                out_r[row] = y_f64[row] - matvec[row];
            }
            return inf_norm(row_abs_sum);
        }

        /// @brief Mixed-precision iterative refinement adapter implementing RefineSystem for GPR solves.
        /// @details Bridges float Cholesky factor L_f32 (for fast error solves) with either promoted float
        /// matrix residuals (PromoteStorage) or column-blocked double kernel re-evaluation (ReevaluateKernel).
        /// @tparam P Mixed-precision policy specifying Residual formula (PromoteStorage or ReevaluateKernel).
        template <PrecisionPolicyConcept P> class GprRefineSystem final : public RefineSystem
        {
        public:
            /// @brief Constructs the refinement adapter from double data, float Cholesky factor, and matrix A_f32.
            /// @param[in] spec Fitted kernel specification compiled in double precision for re-evaluation/fallback.
            /// @param[in] kernel_exp Exponential backend mode matching the fitted model.
            /// @param[in] x_f64 Double-precision transformed training features (n x d).
            /// @param[in] y_f64 Double-precision transformed training targets of length n.
            /// @param[in] l_f32 Single-precision lower-triangular Cholesky factor L (n x n).
            /// @param[in] a_f32_symm Symmetrized single-precision noisy system matrix A (used by PromoteStorage).
            /// @param[in] diag_offset Combined observation noise variance and accepted jitter added to diagonal.
            GprRefineSystem(
                const KernelSpec& spec,
                KernelExp kernel_exp,
                MatrixRef<double> x_f64,
                std::span<const double> y_f64,
                MatrixRef<float> l_f32,
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
            /// @details Promotes stored float entries for PromoteStorage using preallocated scratch, or recomputes
            /// double kernel blocks of width 32 via block_reeval_residual() for ReevaluateKernel.
            /// @param[in] w Current double-precision iterate of length n.
            /// @param[out] r Output double-precision residual buffer of length n.
            /// @return Infinity norm of the system matrix used for relative residual scaling, or GprError.
            /// @retval GprErrorCode::NonFiniteKernelValue When kernel re-evaluation yields non-finite entries.
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
                    return block_reeval_residual(kernel_f64_, x_f64_, y_f64_, diag_offset_, w, r);
                }
            }

            /// @brief Applies one iterative refinement update w += (L_f32 L_f32^T)^-1 r.
            /// @details Casts the double residual r to float, solves via l_f32_, and accumulates the correction
            /// back into double-precision w.
            /// @param[in] r Double-precision residual vector of length n.
            /// @param[in,out] w In-out double-precision solution vector updated in place.
            /// @return Empty expected on success, or GprError if triangular solve fails.
            [[nodiscard]] std::expected<void, GprError> correct(std::span<const double> r, std::span<double> w) override
            {
                // Downcast residual to float for fast L_f32 triangular solves, then accumulate in double.
                const std::size_t n = r.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    r_f32_buf_[i] = static_cast<float>(r[i]);
                }
                if (auto solve = linalg::solve_llt_in_place(l_f32_, std::span<float>(r_f32_buf_)); !solve.has_value())
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
            /// @details ReevaluateKernel already checked the true double residual during residual(); PromoteStorage
            /// performs an additional block_reeval_residual() check against tol.
            /// @param[in] w Candidate double-precision iterate of length n.
            /// @param[in] tol Relative backward-error acceptance threshold.
            /// @return True if iterate is accepted; false triggers full double-precision fallback().
            /// @retval GprErrorCode::NonFiniteKernelValue When kernel re-evaluation yields non-finite entries.
            [[nodiscard]] std::expected<bool, GprError> accept(std::span<const double> w, double tol) override
            {
                if constexpr (std::same_as<typename P::Residual, ReevaluateKernel>)
                {
                    // ReevaluateKernel residuals are already computed in double against the exact kernel.
                    return true;
                }
                else
                {
                    // Verify promoted-storage iterate against the true double-precision kernel residual.
                    std::vector<double> r_true(y_f64_.size());
                    auto b_inf = block_reeval_residual(kernel_f64_, x_f64_, y_f64_, diag_offset_, w, r_true);
                    if (!b_inf.has_value())
                    {
                        return std::unexpected(std::move(b_inf).error());
                    }
                    return meets_relative_residual(*b_inf, w, y_f64_, r_true, tol);
                }
            }

            /// @brief Computes an exact double-precision Cholesky solution when iterative refinement diverges or
            /// stalls.
            /// @details Materializes the double-precision Gram matrix with diag_offset_, factors it without
            /// additional jitter retries, and solves directly into w.
            /// @param[out] w Output double-precision weight vector overwritten with A_f64^-1 y_f64.
            /// @return Empty expected on success, or GprError if kernel evaluation or Cholesky fails.
            /// @retval GprErrorCode::CholeskyFailed When double-precision Cholesky factorization fails.
            [[nodiscard]] std::expected<void, GprError> fallback(std::span<double> w) override
            {
                // Materialize full double Gram matrix + diag_offset_ as a last-resort exact solve.
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
                return linalg::solve_llt_in_place(a_mut.as_ref(), w);
            }

        private:
            /// @brief Double-precision compiled kernel for blocked residual evaluation and fallback solves.
            CompiledKernel<double> kernel_f64_;
            /// @brief Double-precision transformed training features (n x d).
            MatrixRef<double> x_f64_;
            /// @brief Double-precision transformed training targets of length n.
            std::span<const double> y_f64_;
            /// @brief Single-precision Cholesky factor L_f32 retained from the final fit factorization.
            MatrixRef<float> l_f32_;
            /// @brief Symmetrized float noisy Gram matrix A_f32 used when P::Residual is PromoteStorage.
            MatrixRef<float> a_f32_;
            /// @brief Combined noise variance and accepted jitter added along the diagonal.
            double diag_offset_{0.0};
            /// @brief Reusable single-precision residual buffer for triangular L_f32 solves.
            std::vector<float> r_f32_buf_;
            /// @brief Reusable row l1-norm scratch buffer for promoted_residual_into().
            std::vector<double> row_abs_buf_;
            /// @brief Reusable double-precision matrix-vector accumulator buffer for promoted_residual_into().
            std::vector<double> sum_buf_;
        };

        /// @brief Executes mixed-precision iterative refinement starting from the float Cholesky solution.
        /// @details Copies the initial float weights from buffers.core.rhs, builds symmetrized A_f32 in exp_buf
        /// when using PromoteStorage, invokes refine_in_place(), and records RefinementSummary.
        /// @tparam P Mixed-precision policy satisfying P::refines_in_f64.
        /// @param[in] spec Final fitted kernel specification.
        /// @param[in] kernel_exp Exponential backend mode.
        /// @param[in] kernel_f32 Final single-precision compiled kernel evaluator.
        /// @param[in] x_f64 Double-precision transformed training features (n x d).
        /// @param[in] x_f32 Single-precision transformed training features (n x d).
        /// @param[in] y_f64 Double-precision transformed training targets of length n.
        /// @param[in,out] buffers Training workspace containing L_f32, initial rhs weights, and scratch matrices.
        /// @param[in] diag_offset Combined noise variance plus accepted factor jitter.
        /// @param[out] out_alpha Output refined double-precision weight vector of length n.
        /// @param[out] out_summary Output optional populated with RefinementSummary diagnostics.
        /// @return Empty expected on success, or GprError if refinement and fallback both fail.
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
            // Seed double iterate from the single-precision Cholesky solution stored in buffers.core.rhs.
            const std::size_t n = x_f64.rows();
            out_alpha.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                out_alpha[i] = static_cast<double>(buffers.core.rhs[i]);
            }

            MatrixMut<float> a_f32_mut = buffers.core.exp_mut();
            if constexpr (std::same_as<typename P::Residual, PromoteStorage>)
            {
                // Reconstruct full symmetric A_f32 = K_f32 + (sigma_n^2 + j) I into exp_buf.
                if (auto ap = kernel_f32.apply_points(
                        x_f32, a_f32_mut, Triangle::Lower, buffers.core.kernel_scratch_mut(), buffers.core.nested
                    );
                    !ap.has_value())
                {
                    // Note: kernel_scratch might be empty if ensure_kernel_scratch was not called yet.
                    return ap;
                }
                if (auto dg = linalg::add_to_diag(a_f32_mut, static_cast<float>(diag_offset)); !dg.has_value())
                {
                    return dg;
                }
                symmetrize_lower(a_f32_mut);
            }

            GprRefineSystem<P> sys(
                spec, kernel_exp, x_f64, y_f64, buffers.core.k_ref(), a_f32_mut.as_ref(), diag_offset
            );
            std::vector<double> res_scratch(n);
            auto summary = refine_in_place(sys, out_alpha, res_scratch);
            if (!summary.has_value())
            {
                return std::unexpected(std::move(summary).error());
            }
            out_summary = *summary;
            return {};
        }

        /// @brief Copies a possibly strided column-major MatrixRef<double> into a dense contiguous vector while
        /// checking finiteness.
        /// @details Rejects empty or null inputs, guards rows * cols multiplication overflow, and checks every
        /// matrix element for NaN or Inf.
        /// @param[in] src Source column-major matrix view.
        /// @param[out] dst Destination contiguous column-major buffer resized to rows * cols.
        /// @return Empty expected on success, or EmptyInput, SizeOverflow, or NonFiniteInput on invalid data.
        /// @retval GprErrorCode::EmptyInput When src is empty or has a null data pointer.
        /// @retval GprErrorCode::SizeOverflow When src.rows() * src.cols() overflows std::size_t.
        /// @retval GprErrorCode::NonFiniteInput When any matrix entry is NaN or Inf.
        [[nodiscard]] std::expected<void, GprError> copy_double_matrix_col_major(
            MatrixRef<double> src, std::vector<double>& dst
        )
        {
            if (src.is_empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            if (src.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            // Guard dense allocation size against std::size_t multiplication overflow.
            auto nm = checked_mul(src.rows(), src.cols());
            if (!nm.has_value())
            {
                return std::unexpected(std::move(nm).error());
            }
            dst.resize(*nm);
            const std::size_t rows = src.rows();
            const std::size_t cols = src.cols();
            // Copy column-major order while rejecting NaN or +/-Inf training coordinates.
            for (std::size_t col = 0; col < cols; ++col)
            {
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const double val = src(row, col);
                    if (!std::isfinite(val))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                    dst[(col * rows) + row] = val;
                }
            }
            return {};
        }

    }  // namespace

    /// @brief Validates training inputs, fits transforms, optimizes or factors hyperparameters, and assembles
    /// FittedPayload<P>.
    /// @details Preserves raw (x, y) copies for persistence, fits input/target transforms in double precision,
    /// allocates FitBuffers<P>, optionally runs optimize_cb, re-factors at the final parameters, and runs
    /// mixed-precision iterative refinement when P::refines_in_f64 is true.
    /// @tparam P Precision policy controlling Storage and Refine arithmetic.
    /// @param[in] cfg Borrowed trainer configuration and transform prototypes.
    /// @param[in] x Column-major training feature matrix (n x d).
    /// @param[in] y Training target vector of length n.
    /// @param[in] optimize_cb Callback invoking the concrete optimizer when cfg.optimize is true.
    /// @param[in,out] optimize_ctx Opaque pointer forwarded to optimize_cb.
    /// @return Populated FittedPayload<P> on success, or GprError on any validation, optimization, or solve error.
    /// @retval GprErrorCode::EmptyInput When x or y is empty or null.
    /// @retval GprErrorCode::LengthMismatch When y.size() != x.rows().
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or Inf.
    /// @retval GprErrorCode::CholeskyFailed When Cholesky decomposition fails after jitter retries.
    template <PrecisionPolicyConcept P>
    std::expected<FittedPayload<P>, GprError> train_or_factor_payload(
        const TrainConfig<P>& cfg,
        MatrixRef<double> x,
        std::span<const double> y,
        OptimizeCallback optimize_cb,
        void* optimize_ctx
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        // Validate dimensions, column-major feature entries, and target finiteness before mutating workspace.
        if (x.is_empty() || y.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = x.rows();
        const std::size_t d = x.cols();
        if (y.size() != n)
        {
            return std::unexpected(GprError::length_mismatch("target length must match training row count"));
        }
        std::vector<double> x_f64;
        if (auto cp = copy_double_matrix_col_major(x, x_f64); !cp.has_value())
        {
            return std::unexpected(std::move(cp).error());
        }
        std::vector<double> x_raw = x_f64;
        for (const double val : y)
        {
            if (!std::isfinite(val))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        std::vector<double> y_raw(y.begin(), y.end());

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

        auto fitted_tgt = cfg.target_transform.fit(y);
        if (!fitted_tgt.has_value())
        {
            return std::unexpected(std::move(fitted_tgt).error());
        }
        std::vector<double> y_f64(y.begin(), y.end());
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
        if constexpr (P::refines_in_f64)
        {
            // PromoteStorage refinement needs kernel_scratch to assemble A_f32 after factorization.
            if constexpr (std::same_as<typename P::Residual, PromoteStorage>)
            {
                if (auto ks = buffers->core.ensure_kernel_scratch(); !ks.has_value())
                {
                    return std::unexpected(std::move(ks).error());
                }
            }
        }

        const MatrixRef<Storage> x_train_ref(x_storage.data(), n, d, n);
        GaussianLikelihood likelihood = cfg.likelihood;
        GprObjective<P> obj(*compiled, likelihood, x_train_ref, y_storage, cfg.jitter, *buffers, use_leaf_cache);

        const std::size_t total_params = obj.dim();
        std::vector<double> init_params(total_params);
        const std::size_t nk = compiled->num_params();
        if (auto gp = compiled->get_params(std::span<double>(init_params.data(), nk)); !gp.has_value())
        {
            return std::unexpected(std::move(gp).error());
        }
        if (auto lp = likelihood.get_params(std::span<double>(init_params.data() + nk, 1U)); !lp.has_value())
        {
            return std::unexpected(std::move(lp).error());
        }

        // Run hyperparameter optimization or perform a single objective evaluation when factor-only.
        optimizer::OptimizerResult opt_res;
        if (cfg.optimize)
        {
            auto r = optimize_cb(optimize_ctx, obj, init_params);
            if (!r.has_value())
            {
                return std::unexpected(std::move(r).error());
            }
            opt_res = std::move(*r);
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
        }

        // Reuse the live factor L, alpha, and MLL when the last objective call already evaluated opt_res.params.
        double final_mll = 0.0;
        if (obj.matches_last_factor(opt_res.params))
        {
            final_mll = obj.last_factored_mll();
        }
        else
        {
            // Ensure kernel, likelihood, L, alpha, and MLL match opt_res.params exactly.
            if (auto sp = compiled->set_params(std::span<const double>(opt_res.params.data(), nk)); !sp.has_value())
            {
                return std::unexpected(std::move(sp).error());
            }
            if (auto lp = likelihood.set_params(std::span<const double>(opt_res.params.data() + nk, 1U));
                !lp.has_value())
            {
                return std::unexpected(std::move(lp).error());
            }
            auto mll_res = factor_and_mll(*compiled, likelihood, x_train_ref, y_storage, cfg.jitter, *buffers, 0U);
            if (!mll_res.has_value())
            {
                return std::unexpected(std::move(mll_res).error());
            }
            final_mll = *mll_res;
        }

        KernelSpec updated_spec = cfg.kernel_spec;
        if (auto sp = updated_spec.set_params(std::span<const double>(opt_res.params.data(), nk)); !sp.has_value())
        {
            return std::unexpected(std::move(sp).error());
        }

        std::vector<Refine> alpha_out(n);
        std::optional<RefinementSummary> ref_summary;

        // Refine weights alpha in double precision under MixedPrecision policies, or cast directly otherwise.
        if constexpr (P::refines_in_f64)
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

        return FittedPayload<P>{
            .kernel_spec = std::move(updated_spec),
            .likelihood = likelihood,
            .compiled_kernel = std::move(compiled),
            .input_transform = std::move(*fitted_in),
            .target_transform = std::move(*fitted_tgt),
            .x_raw = std::move(x_raw),
            .y_raw = std::move(y_raw),
            .x_train = std::move(x_storage),
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
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in,out] workspace Mutable query workspace reused across calls.
        /// @return Column-major MatrixRef<Storage> backed by workspace.query_x, or GprError on invalid query.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<MatrixRef<typename P::Storage>, GprError> prepare_query_matrix(
            const FittedPayload<P>& payload, MatrixRef<double> xs, QueryWorkspace<P>& workspace
        )
        {
            using Storage = P::Storage;

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
            if (auto ens = workspace.ensure_prediction(payload.n_train, m, d, payload.compiled_kernel->nested_depth());
                !ens.has_value())
            {
                return std::unexpected(std::move(ens).error());
            }

            // Copy query coordinates column-major into double staging buffer while checking finiteness.
            for (std::size_t col = 0; col < d; ++col)
            {
                for (std::size_t row = 0; row < m; ++row)
                {
                    const double val = xs(row, col);
                    if (!std::isfinite(val))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                    workspace.query_xs[(col * m) + row] = val;
                }
            }

            // Apply fitted input transform in double precision before casting down to Storage.
            if (auto ap = payload.input_transform->apply(workspace.query_xs, m, d); !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }

            const std::size_t md = m * d;
            for (std::size_t i = 0; i < md; ++i)
            {
                workspace.query_x[i] = static_cast<Storage>(workspace.query_xs[i]);
            }
            return MatrixRef<Storage>(workspace.query_x.data(), m, d, m);
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
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> fitted_predict_into(
        const FittedPayload<P>& payload,
        MatrixRef<double> xs,
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
        return apply_target_inverse_prediction(*payload.target_transform, out, workspace.query_xs);
    }

    /// @brief Evaluates posterior mean and full m x m covariance matrix on the original target scale.
    /// @tparam P Precision policy matching payload.
    /// @param[in] payload Fitted model payload.
    /// @param[in] xs Column-major query feature matrix (m x d).
    /// @param[in] options Prediction variance options.
    /// @param[in,out] workspace Mutable query workspace.
    /// @return PredictiveCovariance<Refine> on the original target scale, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When xs is empty or null.
    /// @retval GprErrorCode::DimensionMismatch When xs.cols() != payload.n_features.
    /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
    template <PrecisionPolicyConcept P>
    std::expected<PredictiveCovariance<typename P::Refine>, GprError> fitted_predict_covariance(
        const FittedPayload<P>& payload,
        MatrixRef<double> xs,
        const PredictOptions& options,
        QueryWorkspace<P>& workspace
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
        MatrixRef<double> xs,
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
        auto samples = sample<P>(
            *payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref, payload.alpha, *xq_ref, n_draws, seed,
            options, workspace, jitter
        );
        if (!samples.has_value())
        {
            return samples;
        }
        // Apply inverse mean transform to each column draw independently.
        const std::size_t m = xs.rows();
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
        const TrainConfig<DoublePrecision>&, MatrixRef<double>, std::span<const double>, OptimizeCallback, void*
    );
    template std::expected<FittedPayload<SinglePrecision>, GprError> train_or_factor_payload<SinglePrecision>(
        const TrainConfig<SinglePrecision>&, MatrixRef<double>, std::span<const double>, OptimizeCallback, void*
    );
    template std::expected<FittedPayload<MixedPrecision<PromoteStorage>>, GprError> train_or_factor_payload<
        MixedPrecision<PromoteStorage>>(
        const TrainConfig<MixedPrecision<PromoteStorage>>&,
        MatrixRef<double>,
        std::span<const double>,
        OptimizeCallback,
        void*
    );
    template std::expected<FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError> train_or_factor_payload<
        MixedPrecision<ReevaluateKernel>>(
        const TrainConfig<MixedPrecision<ReevaluateKernel>>&,
        MatrixRef<double>,
        std::span<const double>,
        OptimizeCallback,
        void*
    );

    template std::expected<void, GprError> fitted_predict_into<DoublePrecision>(
        const FittedPayload<DoublePrecision>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        Prediction<double>&
    );
    template std::expected<void, GprError> fitted_predict_into<SinglePrecision>(
        const FittedPayload<SinglePrecision>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        Prediction<float>&
    );
    template std::expected<void, GprError> fitted_predict_into<MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        Prediction<double>&
    );
    template std::expected<void, GprError> fitted_predict_into<MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
        Prediction<double>&
    );

    template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<DoublePrecision>(
        const FittedPayload<DoublePrecision>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&
    );
    template std::expected<PredictiveCovariance<float>, GprError> fitted_predict_covariance<SinglePrecision>(
        const FittedPayload<SinglePrecision>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&
    );
    template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
        MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&
    );
    template std::expected<PredictiveCovariance<double>, GprError> fitted_predict_covariance<
        MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
        MatrixRef<double>,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<ReevaluateKernel>>&
    );

    template std::expected<std::vector<double>, GprError> fitted_sample<DoublePrecision>(
        const FittedPayload<DoublePrecision>&,
        MatrixRef<double>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<DoublePrecision>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<float>, GprError> fitted_sample<SinglePrecision>(
        const FittedPayload<SinglePrecision>&,
        MatrixRef<double>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<SinglePrecision>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<PromoteStorage>>(
        const FittedPayload<MixedPrecision<PromoteStorage>>&,
        MatrixRef<double>,
        std::size_t,
        std::uint64_t,
        const PredictOptions&,
        QueryWorkspace<MixedPrecision<PromoteStorage>>&,
        std::optional<JitterPolicy>
    );
    template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<ReevaluateKernel>>(
        const FittedPayload<MixedPrecision<ReevaluateKernel>>&,
        MatrixRef<double>,
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
