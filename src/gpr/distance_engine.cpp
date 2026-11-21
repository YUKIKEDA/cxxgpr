// SPDX-License-Identifier: Apache-2.0
/// @file distance_engine.cpp
/// @brief Training and prediction engine for Gaussian processes on caller-supplied squared distances.
/// @details Evaluates the negative log marginal likelihood, its analytic gradient and Hessian for a
/// CompiledDistanceKernel using the same dense kernels as the coordinate model, and predicts from rectangular
/// cross-distance tables.

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/gpr/distance_gpr.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/workspace/inference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

#include "gpr/refine_constants.hpp"
#include "linalg/factor_validation.hpp"
#include "workspace/mll_kernels.hpp"
#include "workspace/sample_draws.hpp"

namespace cxxgpr
{

    namespace
    {

        /// @brief Brings the shared dense fill_diagonal_scaled kernel into scope.
        using mll_kernels::fill_diagonal_scaled;
        /// @brief Brings the shared dense frobenius_full_det kernel into scope.
        using mll_kernels::frobenius_full_det;
        /// @brief Brings the shared dense frobenius_lower_det kernel into scope.
        using mll_kernels::frobenius_lower_det;
        /// @brief Brings the shared dense gemv_sym_lower kernel into scope.
        using mll_kernels::gemv_sym_lower;
        /// @brief Brings the shared dense transpose_square_in_place kernel into scope.
        using mll_kernels::transpose_square_in_place;

        /// @brief Copies a double matrix view into a column-major vector, rejecting non-finite entries.
        /// @param[in] x Source view.
        /// @param[out] out Destination vector resized to rows * cols.
        /// @return Empty expected on success, or GprError.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::NonFiniteInput When x contains NaN or Inf.
        [[nodiscard]] std::expected<void, GprError> copy_points(MatrixRef<double> x, std::vector<double>& out)
        {
            if (x.is_empty() || x.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            out.resize(x.rows() * x.cols());
            for (std::size_t col = 0; col < x.cols(); ++col)
            {
                for (std::size_t row = 0; row < x.rows(); ++row)
                {
                    const double v = x(row, col);
                    if (!std::isfinite(v))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                    out[(col * x.rows()) + row] = v;
                }
            }
            return {};
        }

        /// @brief Negative log marginal likelihood objective over supplied-distance tables.
        /// @details Holds the compiled kernel, the validated store, the training targets and the dense buffers, and
        /// implements value, gradient and Hessian in log-parameter space exactly like the coordinate objective.
        /// @tparam P Precision policy.
        template <PrecisionPolicyConcept P> class DistanceObjective final : public optimizer::TwiceDifferentiable
        {
        public:
            /// @brief Storage scalar type.
            using Storage = typename P::Storage;

            /// @brief Binds the borrowed training state; nothing is evaluated until a method is called.
            /// @param[in,out] kernel Compiled kernel whose parameters the objective sets.
            /// @param[in,out] likelihood Likelihood whose noise parameter the objective sets.
            /// @param[in] store Validated square training tables.
            /// @param[in] x Optional training coordinates for WithPoints kernels.
            /// @param[in] y Transformed training targets.
            /// @param[in] jitter Jitter policy.
            /// @param[in,out] buffers Dense buffers sized for n.
            DistanceObjective(
                CompiledDistanceKernel<Storage>& kernel,
                GaussianLikelihood& likelihood,
                const CheckedDistanceStore<Storage>& store,
                std::optional<MatrixRef<Storage>> x,
                std::span<const Storage> y,
                const JitterPolicy& jitter,
                FitBuffers<P>& buffers
            ) :
                kernel_(kernel),
                likelihood_(likelihood),
                store_(store),
                x_(x),
                y_(y),
                jitter_(jitter),
                buffers_(buffers),
                last_params_(kernel.num_params() + likelihood.num_params(), 0.0)
            {
            }

            /// @brief Returns the number of optimized parameters (kernel parameters plus log noise variance).
            /// @return Parameter count.
            [[nodiscard]] std::size_t dim() const noexcept override
            {
                return kernel_.num_params() + likelihood_.num_params();
            }

            /// @brief Returns true when the live factor was computed at exactly these parameters.
            /// @param[in] params Log-space parameters.
            /// @return True on a bitwise match.
            [[nodiscard]] bool matches_last_factor(std::span<const double> params) const noexcept
            {
                if (!valid_ || params.size() != last_params_.size())
                {
                    return false;
                }
                for (std::size_t i = 0; i < params.size(); ++i)
                {
                    if (std::bit_cast<std::uint64_t>(params[i]) != std::bit_cast<std::uint64_t>(last_params_[i]))
                    {
                        return false;
                    }
                }
                return true;
            }

            /// @brief Returns the objective value of the live factor.
            /// @return Negative log marginal likelihood.
            [[nodiscard]] double last_mll() const noexcept { return last_mll_; }

            /// @brief Evaluates the negative log marginal likelihood.
            /// @param[in] params Log-space parameters.
            /// @return Objective value, or GprError.
            [[nodiscard]] std::expected<double, GprError> value(std::span<const double> params) override
            {
                valid_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }
                auto mll = factor_and_mll();
                if (mll.has_value())
                {
                    record(params, *mll);
                }
                return mll;
            }

            /// @brief Evaluates the objective for changed coordinates; distance leaves are rebuilt in full.
            /// @param[in] params Log-space parameters.
            /// @param[in] changed_indices Ignored.
            /// @return Objective value, or GprError.
            [[nodiscard]] std::expected<double, GprError> value_at_changes(
                std::span<const double> params, std::span<const std::size_t> /*changed_indices*/
            ) override
            {
                return value(params);
            }

            /// @brief Writes the parameter intervals (kernel parameters then noise variance).
            /// @param[out] out Interval buffer of length dim().
            /// @return Empty expected on success, or GprError.
            [[nodiscard]] std::expected<void, GprError> fill_intervals(std::span<Interval> out) const override
            {
                const std::size_t nk = kernel_.num_params();
                if (out.size() != nk + 1U)
                {
                    return std::unexpected(
                        GprError::length_mismatch("interval buffer length must match parameter count")
                    );
                }
                if (auto kb = kernel_.bounds_into(out.subspan(0, nk)); !kb.has_value())
                {
                    return kb;
                }
                return likelihood_.bounds_into(out.subspan(nk, 1U));
            }

            /// @brief Evaluates the gradient only.
            /// @param[in] params Log-space parameters.
            /// @param[out] out_grad Gradient of length dim().
            /// @return Empty expected on success, or GprError.
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

            /// @brief Evaluates the objective and its gradient.
            /// @param[in] params Log-space parameters.
            /// @param[out] out_grad Gradient of length dim().
            /// @return Objective value, or GprError.
            [[nodiscard]] std::expected<double, GprError> value_and_gradient_into(
                std::span<const double> params, std::span<double> out_grad
            ) override
            {
                valid_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }
                auto mll = factor_and_mll();
                if (!mll.has_value())
                {
                    return mll;
                }
                if (auto w = build_w(); !w.has_value())
                {
                    return std::unexpected(std::move(w).error());
                }
                if (auto g = fill_gradient(out_grad); !g.has_value())
                {
                    return std::unexpected(std::move(g).error());
                }
                record(params, *mll);
                return mll;
            }

            /// @brief Evaluates the Hessian only.
            /// @param[in] params Log-space parameters.
            /// @param[out] out_hess Row-major dim() x dim() Hessian.
            /// @return Empty expected on success, or GprError.
            [[nodiscard]] std::expected<void, GprError> hessian_into(
                std::span<const double> params, std::span<double> out_hess
            ) override
            {
                valid_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return set;
                }
                auto mll = factor_and_mll();
                if (!mll.has_value())
                {
                    return std::unexpected(std::move(mll).error());
                }
                if (auto w = build_w(); !w.has_value())
                {
                    return w;
                }
                return fill_hessian(out_hess);
            }

            /// @brief Evaluates objective, gradient and Hessian sharing one factorization and one W.
            /// @param[in] params Log-space parameters.
            /// @param[out] out_grad Gradient of length dim().
            /// @param[out] out_hess Row-major dim() x dim() Hessian.
            /// @return Objective value, or GprError.
            [[nodiscard]] std::expected<double, GprError> value_gradient_hessian_into(
                std::span<const double> params, std::span<double> out_grad, std::span<double> out_hess
            ) override
            {
                valid_ = false;
                if (auto set = apply_params(params); !set.has_value())
                {
                    return std::unexpected(std::move(set).error());
                }
                auto mll = factor_and_mll();
                if (!mll.has_value())
                {
                    return mll;
                }
                if (auto w = build_w(); !w.has_value())
                {
                    return std::unexpected(std::move(w).error());
                }
                if (auto g = fill_gradient(out_grad); !g.has_value())
                {
                    return std::unexpected(std::move(g).error());
                }
                if (auto h = fill_hessian(out_hess); !h.has_value())
                {
                    return std::unexpected(std::move(h).error());
                }
                record(params, *mll);
                return mll;
            }

        private:
            /// @brief Remembers the parameters and objective of the latest successful evaluation.
            /// @param[in] params Parameter vector that was evaluated.
            /// @param[in] mll Objective value at those parameters.
            void record(std::span<const double> params, double mll) noexcept
            {
                std::ranges::copy(params, last_params_.begin());
                last_mll_ = mll;
                valid_ = true;
            }

            /// @brief Writes the optimizer parameter vector into the kernel and the likelihood.
            /// @param[in] params Kernel parameters followed by the log noise variance.
            /// @return Success, or GprError when the length or a value is invalid.
            [[nodiscard]] std::expected<void, GprError> apply_params(std::span<const double> params)
            {
                const std::size_t nk = kernel_.num_params();
                if (params.size() != nk + 1U)
                {
                    return std::unexpected(
                        GprError::length_mismatch("parameter length must match kernel + noise params")
                    );
                }
                if (auto kp = kernel_.set_params(params.subspan(0, nk)); !kp.has_value())
                {
                    return kp;
                }
                return likelihood_.set_params(params.subspan(nk, 1U));
            }

            /// @brief Grows the shared scratch (kernel scratch, nested pool, second scratch matrix).
            /// @return Success, or GprError when preparing the kernel scratch fails.
            [[nodiscard]] std::expected<void, GprError> ensure_scratch()
            {
                auto& core = buffers_.core;
                const std::size_t n = core.order();
                if (auto ok = core.ensure_kernel_scratch(); !ok.has_value())
                {
                    return ok;
                }
                if (core.nested.depth() < kernel_.nested_depth() || core.nested.rows() < n || core.nested.cols() < n)
                {
                    core.nested.resize(kernel_.nested_depth(), n, n);
                }
                if (scratch_b_.size() < n * n)
                {
                    scratch_b_.assign(n * n, Storage{0});
                }
                return {};
            }

            /// @brief Returns the second scratch matrix as a square mutable view.
            /// @return Mutable view over the second scratch matrix.
            [[nodiscard]] MatrixMut<Storage> scratch_b_mut()
            {
                const std::size_t n = buffers_.core.order();
                return MatrixMut<Storage>(scratch_b_.data(), n, n, n);
            }

            /// @brief Assembles K + noise I, factors it with jitter, solves for alpha, and returns the objective.
            /// @return Negative log marginal likelihood, or GprError when the factorization or objective fails.
            [[nodiscard]] std::expected<double, GprError> factor_and_mll()
            {
                auto& core = buffers_.core;
                const std::size_t n = core.order();
                if (auto ok = ensure_scratch(); !ok.has_value())
                {
                    return std::unexpected(std::move(ok).error());
                }
                if (auto ap = kernel_.apply(
                        store_, x_, core.k_mut(), Triangle::Lower, core.kernel_scratch_mut(), core.nested
                    );
                    !ap.has_value())
                {
                    return std::unexpected(std::move(ap).error());
                }
                const auto noise = static_cast<Storage>(likelihood_.noise_variance());
                if (auto dg = linalg::add_to_diag(core.k_mut(), noise); !dg.has_value())
                {
                    return std::unexpected(std::move(dg).error());
                }
                auto jit =
                    linalg::cholesky_lower_with_jitter(core.k_mut(), core.exp_mut(), jitter_, CholeskyStage::Fit);
                if (!jit.has_value())
                {
                    return std::unexpected(std::move(jit).error());
                }
                core.factor_jitter = *jit;
                std::ranges::copy(y_, core.rhs.begin());
                if (auto solve = linalg::solve_llt_in_place(core.k_ref(), std::span<Storage>(core.rhs));
                    !solve.has_value())
                {
                    return std::unexpected(std::move(solve).error());
                }
                auto logdet = linalg::log_det_from_l(core.k_ref());
                if (!logdet.has_value())
                {
                    return std::unexpected(std::move(logdet).error());
                }
                auto y_alpha = linalg::dot(y_, std::span<const Storage>(core.rhs));
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
                return mll;
            }

            /// @brief Builds the lower triangle of W = alpha alpha^T - A^-1 from the live factor.
            /// @return Success, or GprError when the computation fails.
            [[nodiscard]] std::expected<void, GprError> build_w()
            {
                auto& core = buffers_.core;
                return mll_kernels::compute_w_lower_from_cholesky<Storage>(
                    core.k_ref(), std::span<const Storage>(core.rhs), core.exp_mut(), buffers_.w_mut()
                );
            }

            /// @brief Fills -0.5 <W, dA/dtheta_p>_F for every parameter, including the log noise variance.
            /// @param[out] out_grad Gradient with respect to the kernel parameters and the log noise variance.
            /// @return Success, or GprError when the output length mismatches or a kernel gradient fails.
            [[nodiscard]] std::expected<void, GprError> fill_gradient(std::span<double> out_grad)
            {
                auto& core = buffers_.core;
                const std::size_t n = core.order();
                const std::size_t nk = kernel_.num_params();
                if (out_grad.size() != nk + 1U)
                {
                    return std::unexpected(GprError::length_mismatch("gradient output length mismatch"));
                }
                for (std::size_t p = 0; p < nk; ++p)
                {
                    if (auto gr = kernel_.grad(
                            store_, x_, core.exp_mut(), p, Triangle::Lower, core.kernel_scratch_mut(), scratch_b_mut(),
                            core.nested
                        );
                        !gr.has_value())
                    {
                        return gr;
                    }
                    const double inner =
                        frobenius_lower_det(buffers_.w_ref(), core.exp_ref(), std::span<double>(core.col_partials));
                    out_grad[p] = -0.5 * inner;
                }
                const auto w_ref = buffers_.w_ref();
                double trace_w = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    trace_w += static_cast<double>(w_ref(i, i));
                }
                out_grad[nk] = -0.5 * likelihood_.noise_variance() * trace_w;
                return {};
            }

            /// @brief Fills the analytic log-space Hessian from the live factor and W.
            /// @param[out] out_hess Square Hessian whose side is the kernel parameter count plus one.
            /// @return Success, or GprError when the output length mismatches or a kernel derivative fails.
            [[nodiscard]] std::expected<void, GprError> fill_hessian(std::span<double> out_hess)
            {
                auto& core = buffers_.core;
                const std::size_t n = core.order();
                const std::size_t nk = kernel_.num_params();
                const std::size_t np = nk + 1U;
                if (out_hess.size() != np * np)
                {
                    return std::unexpected(GprError::length_mismatch("hessian output length mismatch"));
                }
                if (auto ok = ensure_scratch(); !ok.has_value())
                {
                    return ok;
                }
                if (auto ok = core.hessian.ensure(n, np); !ok.has_value())
                {
                    return ok;
                }
                const auto noise = static_cast<Storage>(likelihood_.noise_variance());
                std::ranges::fill(out_hess, 0.0);

                // Second-order term -0.5 <W, d^2 A / (dtheta_i dtheta_j)>_F.
                for (std::size_t i = 0; i < np; ++i)
                {
                    for (std::size_t j = i; j < np; ++j)
                    {
                        if (i < nk && j < nk)
                        {
                            if (auto hr = kernel_.hess(
                                    store_, x_, core.exp_mut(), i, j, Triangle::Lower, core.kernel_scratch_mut(),
                                    scratch_b_mut(), core.hessian.scratch_c_mut(), core.nested
                                );
                                !hr.has_value())
                            {
                                return hr;
                            }
                        }
                        else if (i == nk && j == nk)
                        {
                            fill_diagonal_scaled(core.exp_mut(), noise);
                        }
                        else
                        {
                            continue;
                        }
                        const double inner =
                            frobenius_lower_det(buffers_.w_ref(), core.exp_ref(), std::span<double>(core.col_partials));
                        out_hess[(i * np) + j] = -0.5 * inner;
                        out_hess[(j * np) + i] = -0.5 * inner;
                    }
                }

                // S_i = L^-1 A_i L^-T and v_i = L^-1 A_i alpha for every parameter.
                const auto l_ref = core.k_ref();
                constexpr auto validated = linalg::detail::FactorValidation::AlreadyValidated;
                const std::span<const Storage> alpha(core.rhs);
                for (std::size_t i = 0; i < np; ++i)
                {
                    if (i < nk)
                    {
                        if (auto gr = kernel_.grad(
                                store_, x_, core.exp_mut(), i, Triangle::Lower, core.kernel_scratch_mut(),
                                core.hessian.scratch_c_mut(), core.nested
                            );
                            !gr.has_value())
                        {
                            return gr;
                        }
                    }
                    else
                    {
                        fill_diagonal_scaled(core.exp_mut(), noise);
                    }
                    const auto d_a = core.exp_ref();
                    gemv_sym_lower<Storage>(d_a, alpha, std::span<Storage>(core.hessian.u));
                    auto v_i = core.hessian.v_col_mut(i);
                    std::ranges::copy(core.hessian.u, v_i.begin());
                    if (auto sv = linalg::detail::solve_lower(l_ref, MatrixMut<Storage>(v_i.data(), n, 1U), validated);
                        !sv.has_value())
                    {
                        return sv;
                    }
                    auto s_i = core.hessian.s_mut(i);
                    for (std::size_t col = 0; col < n; ++col)
                    {
                        s_i(col, col) = d_a(col, col);
                        for (std::size_t row = col + 1U; row < n; ++row)
                        {
                            const Storage val = d_a(row, col);
                            s_i(row, col) = val;
                            // NOLINTNEXTLINE(readability-suspicious-call-argument)
                            s_i(col, row) = val;
                        }
                    }
                    if (auto s1 = linalg::detail::solve_lower(l_ref, s_i, validated); !s1.has_value())
                    {
                        return s1;
                    }
                    transpose_square_in_place(s_i);
                    if (auto s2 = linalg::detail::solve_lower(l_ref, s_i, validated); !s2.has_value())
                    {
                        return s2;
                    }
                }

                // Add -0.5 tr(S_i S_j) + v_i^T v_j symmetrically.
                for (std::size_t i = 0; i < np; ++i)
                {
                    for (std::size_t j = 0; j <= i; ++j)
                    {
                        const double tr = frobenius_full_det(
                            core.hessian.s_ref(i), core.hessian.s_ref(j), std::span<double>(core.col_partials)
                        );
                        auto vdot = linalg::dot<Storage>(core.hessian.v_col_ref(i), core.hessian.v_col_ref(j));
                        if (!vdot.has_value())
                        {
                            return std::unexpected(std::move(vdot).error());
                        }
                        const double add = (-0.5 * tr) + *vdot;
                        out_hess[(i * np) + j] += add;
                        if (i != j)
                        {
                            out_hess[(j * np) + i] += add;
                        }
                    }
                }
                return {};
            }

            /// @brief Kernel being optimized.
            CompiledDistanceKernel<Storage>& kernel_;
            /// @brief Likelihood holding the noise variance being optimized.
            GaussianLikelihood& likelihood_;
            /// @brief Checked supplied distance tables.
            const CheckedDistanceStore<Storage>& store_;
            /// @brief Optional training points (with-points mode).
            std::optional<MatrixRef<Storage>> x_;
            /// @brief Standardised training targets.
            std::span<const Storage> y_;
            /// @brief Jitter policy used when factoring.
            const JitterPolicy& jitter_;
            /// @brief Shared fit buffers.
            FitBuffers<P>& buffers_;
            /// @brief Second scratch matrix for kernel gradients and Hessian terms.
            std::vector<Storage> scratch_b_;
            /// @brief Parameters of the latest successful evaluation.
            std::vector<double> last_params_;
            /// @brief Objective value of the latest successful evaluation.
            double last_mll_{0.0};
            /// @brief Whether the cached evaluation is valid.
            bool valid_{false};
        };

        /// @brief Iterative-refinement linear system of a supplied-distance model under a mixed-precision policy.
        /// @details The float factor corrects double-precision residuals. PromoteStorage evaluates the residual from
        /// the stored float matrix; ReevaluateKernel from the exact double Gram matrix built from the double tables.
        /// The acceptance check and the fallback always use the exact double system.
        /// @tparam P Mixed-precision policy.
        template <PrecisionPolicyConcept P> class DistanceRefineSystem final : public RefineSystem
        {
        public:
            /// @brief Binds the exact double-precision system and the float factor.
            /// @param[in] kernel64 Kernel compiled in double precision.
            /// @param[in] store64 Validated training tables in double precision.
            /// @param[in] x64 Optional transformed training coordinates in double precision.
            /// @param[in] y64 Transformed targets in double precision.
            /// @param[in] l_f32 Float lower Cholesky factor of the stored system.
            /// @param[in] a_f32 Float noisy system matrix, symmetrized (PromoteStorage only, otherwise empty).
            /// @param[in] diag_offset Noise variance plus accepted jitter.
            DistanceRefineSystem(
                const CompiledDistanceKernel<double>& kernel64,
                const CheckedDistanceStore<double>& store64,
                std::optional<MatrixRef<double>> x64,
                std::span<const double> y64,
                MatrixRef<float> l_f32,
                MatrixRef<float> a_f32,
                double diag_offset
            ) :
                kernel64_(kernel64),
                store64_(store64),
                x64_(x64),
                y64_(y64),
                l_f32_(l_f32),
                a_f32_(a_f32),
                diag_offset_(diag_offset),
                r_f32_(y64.size(), 0.0F),
                row_abs_(y64.size(), 0.0),
                sum_(y64.size(), 0.0)
            {
            }

            /// @brief Returns the double-precision targets as the refinement right-hand side.
            /// @return Target span.
            [[nodiscard]] std::span<const double> rhs() const noexcept override { return y64_; }

            /// @brief Computes r = y - B w and returns the infinity norm of B.
            /// @param[in] w Current iterate.
            /// @param[out] r Residual.
            /// @return Infinity norm of the system matrix used, or GprError.
            [[nodiscard]] std::expected<double, GprError> residual(
                std::span<const double> w, std::span<double> r
            ) override
            {
                if constexpr (std::same_as<typename P::Residual, PromoteStorage>)
                {
                    return promoted_residual_into(a_f32_, w, y64_, r, row_abs_, sum_);
                }
                else
                {
                    return exact_residual(w, r);
                }
            }

            /// @brief Adds the float-solved correction L L^T d = r to the iterate.
            /// @param[in] r Double residual.
            /// @param[in,out] w Iterate updated in place.
            /// @return Empty expected on success, or GprError.
            [[nodiscard]] std::expected<void, GprError> correct(std::span<const double> r, std::span<double> w) override
            {
                const std::size_t n = r.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    r_f32_[i] = static_cast<float>(r[i]);
                }
                if (auto solve = linalg::solve_llt_in_place(l_f32_, std::span<float>(r_f32_)); !solve.has_value())
                {
                    return solve;
                }
                for (std::size_t i = 0; i < n; ++i)
                {
                    w[i] += static_cast<double>(r_f32_[i]);
                }
                return {};
            }

            /// @brief Verifies a converged PromoteStorage iterate against the exact double-precision system.
            /// @param[in] w Candidate iterate.
            /// @param[in] tol Double-precision relative backward-error tolerance.
            /// @return True when accepted; false triggers the fallback.
            [[nodiscard]] std::expected<bool, GprError> accept(std::span<const double> w, double tol) override
            {
                if constexpr (std::same_as<typename P::Residual, ReevaluateKernel>)
                {
                    return true;
                }
                else
                {
                    std::vector<double> r_true(y64_.size());
                    auto b_inf = exact_residual(w, r_true);
                    if (!b_inf.has_value())
                    {
                        return std::unexpected(std::move(b_inf).error());
                    }
                    const double storage_tol = detail::promote_storage_accept_factor
                        * static_cast<double>(std::numeric_limits<float>::epsilon());
                    return meets_relative_residual(*b_inf, w, y64_, r_true, std::max(tol, storage_tol));
                }
            }

            /// @brief Solves the exact double-precision system with a fresh Cholesky factorization.
            /// @param[out] w Solution overwritten with the double-precision weights.
            /// @return Empty expected on success, or GprError.
            [[nodiscard]] std::expected<void, GprError> fallback(std::span<double> w) override
            {
                const std::size_t n = y64_.size();
                std::vector<double> gram(n * n);
                std::vector<double> scratch(n * n);
                auto nested = kernel64_.make_nested_buffers(n, n);
                MatrixMut<double> gram_mut(gram.data(), n, n, n);
                MatrixMut<double> scratch_mut(scratch.data(), n, n, n);
                if (auto ap = kernel64_.apply(store64_, x64_, gram_mut, Triangle::Lower, scratch_mut, nested);
                    !ap.has_value())
                {
                    return ap;
                }
                if (auto dg = linalg::add_to_diag(gram_mut, diag_offset_); !dg.has_value())
                {
                    return dg;
                }
                if (auto ch = linalg::cholesky_lower_in_place(gram_mut); !ch.has_value())
                {
                    return ch;
                }
                std::ranges::copy(y64_, w.begin());
                return linalg::solve_llt_in_place(gram_mut.as_ref(), w);
            }

        private:
            /// @brief Residual against the exact double Gram matrix plus the diagonal offset.
            /// @param[in] w Current solution vector.
            /// @param[out] r Residual of the exact system at w.
            /// @return Largest absolute row sum of the exact matrix, or GprError when the Gram matrix cannot be built.
            [[nodiscard]] std::expected<double, GprError> exact_residual(std::span<const double> w, std::span<double> r)
            {
                const std::size_t n = y64_.size();
                if (gram64_.size() < n * n)
                {
                    gram64_.resize(n * n);
                    gram_scratch_.resize(n * n);
                }
                auto nested = kernel64_.make_nested_buffers(n, n);
                MatrixMut<double> gram_mut(gram64_.data(), n, n, n);
                MatrixMut<double> scratch_mut(gram_scratch_.data(), n, n, n);
                if (auto ap = kernel64_.apply(store64_, x64_, gram_mut, Triangle::Full, scratch_mut, nested);
                    !ap.has_value())
                {
                    return std::unexpected(std::move(ap).error());
                }
                std::ranges::fill(row_abs_, 0.0);
                std::ranges::fill(sum_, 0.0);
                for (std::size_t col = 0; col < n; ++col)
                {
                    const double* g_col = gram64_.data() + (col * n);
                    const double w_col = w[col];
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        const double entry = g_col[row] + ((row == col) ? diag_offset_ : 0.0);
                        sum_[row] += entry * w_col;
                        row_abs_[row] += std::abs(entry);
                    }
                }
                for (std::size_t row = 0; row < n; ++row)
                {
                    r[row] = y64_[row] - sum_[row];
                }
                return inf_norm(row_abs_);
            }

            /// @brief Double-precision kernel.
            const CompiledDistanceKernel<double>& kernel64_;
            /// @brief Double-precision distance tables.
            const CheckedDistanceStore<double>& store64_;
            /// @brief Optional double-precision training points (with-points mode).
            std::optional<MatrixRef<double>> x64_;
            /// @brief Double-precision targets.
            std::span<const double> y64_;
            /// @brief Single-precision Cholesky factor.
            MatrixRef<float> l_f32_;
            /// @brief Single-precision system matrix.
            MatrixRef<float> a_f32_;
            /// @brief Diagonal offset (noise variance plus jitter).
            double diag_offset_;
            /// @brief Single-precision residual buffer.
            std::vector<float> r_f32_;
            /// @brief Row absolute sums of the exact matrix.
            std::vector<double> row_abs_;
            /// @brief Accumulator of the exact matrix-vector product.
            std::vector<double> sum_;
            /// @brief Exact double Gram matrix storage.
            std::vector<double> gram64_;
            /// @brief Scratch matrix for building the Gram matrix.
            std::vector<double> gram_scratch_;
        };

    }  // namespace

    /// @brief Trains a supplied-distance model and returns its immutable fitted payload.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] cfg Training configuration (kernel, likelihood, jitter, transforms, optimization switch).
    /// @param[in] train Supplied squared distances of the training set.
    /// @param[in] x Optional training points (with-points mode).
    /// @param[in] y Training targets.
    /// @param[in] optimize_cb Type-erased optimizer entry point.
    /// @param[in] optimize_ctx Opaque optimizer context passed to the entry point.
    /// @return Fitted payload, or GprError.
    template <PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<detail::DistanceFittedPayload<P, C>, GprError> detail::train_distance_payload(
        const DistanceTrainConfig<C>& cfg,
        const DistanceSource& train,
        std::optional<MatrixRef<double>> x,
        std::span<const double> y,
        std::expected<optimizer::OptimizerResult, GprError> (*optimize_cb)(
            void*, optimizer::TwiceDifferentiable&, std::span<const double>
        ),
        void* optimize_ctx
    )
    {
        using Storage = typename P::Storage;
        using Refine = typename P::Refine;
        constexpr bool with_points = std::same_as<C, WithPoints>;

        // Validation order: empty inputs, target length against the tables, finiteness, tables, transforms.
        const std::size_t n = y.size();
        if (n == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        std::vector<double> x_f64;
        std::size_t d = 0;
        if constexpr (with_points)
        {
            if (!x.has_value())
            {
                return std::unexpected(GprError::empty_input());
            }
            if (x->rows() != n)
            {
                return std::unexpected(GprError::length_mismatch("target length must match training row count"));
            }
            if (auto cp = copy_points(*x, x_f64); !cp.has_value())
            {
                return std::unexpected(std::move(cp).error());
            }
            d = x->cols();
        }
        for (const double v : y)
        {
            if (!std::isfinite(v))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }

        auto compiled = std::make_shared<CompiledDistanceKernel<Storage>>(cfg.kernel.template compile<Storage>());
        compiled->set_kernel_exp(cfg.kernel_exp);
        auto store = compiled->prepare_square(train);
        if (!store.has_value())
        {
            return std::unexpected(std::move(store).error());
        }
        if (store->rows() != n)
        {
            return std::unexpected(GprError::length_mismatch("target length must match training table order"));
        }

        DistanceFittedPayload<P, C> payload{
            .kernel = cfg.kernel,
            .likelihood = cfg.likelihood,
            .compiled = nullptr,
            .store = nullptr,
            .input_transform = nullptr,
            .target_transform = nullptr,
            .x_raw = {},
            .y_raw = std::vector<double>(y.begin(), y.end()),
            .x_train = {},
            .y_transformed = {},
            .l_factor = {},
            .alpha = {},
            .n_train = n,
            .n_features = d,
            .factor_jitter = 0.0,
            .neg_log_marginal_likelihood = 0.0,
            .optimizer_result = {},
            .refinement_summary = std::nullopt,
        };

        if constexpr (with_points)
        {
            payload.x_raw = x_f64;
            auto fitted_in = cfg.input_transform.fit(x_f64, n, d);
            if (!fitted_in.has_value())
            {
                return std::unexpected(std::move(fitted_in).error());
            }
            if (auto ap = (*fitted_in)->apply(x_f64, n, d); !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }
            payload.input_transform = std::shared_ptr<const transform::FittedInputTransform>(std::move(*fitted_in));
            payload.x_train.resize(x_f64.size());
            for (std::size_t i = 0; i < x_f64.size(); ++i)
            {
                payload.x_train[i] = static_cast<Storage>(x_f64[i]);
            }
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
        payload.target_transform = std::shared_ptr<const transform::FittedTargetTransform>(std::move(*fitted_tgt));
        std::vector<Storage> y_storage(n);
        payload.y_transformed.resize(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            y_storage[i] = static_cast<Storage>(y_f64[i]);
            payload.y_transformed[i] = static_cast<Refine>(y_f64[i]);
        }

        auto buffers = FitBuffers<P>::create(n, DistanceCachePolicy::Uncached, CholeskyBuffer::Retain);
        if (!buffers.has_value())
        {
            return std::unexpected(std::move(buffers).error());
        }
        std::optional<MatrixRef<Storage>> x_ref;
        if constexpr (with_points)
        {
            x_ref = MatrixRef<Storage>(payload.x_train.data(), n, d, n);
        }
        GaussianLikelihood likelihood = cfg.likelihood;
        DistanceObjective<P> obj(*compiled, likelihood, *store, x_ref, y_storage, cfg.jitter, *buffers);

        const std::size_t nk = compiled->num_params();
        std::vector<double> init_params(obj.dim());
        if (auto gp = compiled->get_params(std::span<double>(init_params.data(), nk)); !gp.has_value())
        {
            return std::unexpected(std::move(gp).error());
        }
        if (auto lp = likelihood.get_params(std::span<double>(init_params.data() + nk, 1U)); !lp.has_value())
        {
            return std::unexpected(std::move(lp).error());
        }

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
            opt_res =
                optimizer::OptimizerResult{.params = init_params, .value = *v, .iterations = 0U, .converged = true};
        }

        // Reuse the live factor when the last evaluation was at the returned parameters; otherwise factor there.
        double final_mll = 0.0;
        if (obj.matches_last_factor(opt_res.params))
        {
            final_mll = obj.last_mll();
        }
        else
        {
            auto v = obj.value(opt_res.params);
            if (!v.has_value())
            {
                return std::unexpected(std::move(v).error());
            }
            final_mll = *v;
        }
        if (auto sp = payload.kernel.set_params(std::span<const double>(opt_res.params.data(), nk)); !sp.has_value())
        {
            return std::unexpected(std::move(sp).error());
        }
        if (auto lp = payload.likelihood.set_params(std::span<const double>(opt_res.params.data() + nk, 1U));
            !lp.has_value())
        {
            return std::unexpected(std::move(lp).error());
        }

        payload.alpha.resize(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            payload.alpha[i] = static_cast<Refine>(buffers->core.rhs[i]);
        }
        if constexpr (P::refines_in_f64)
        {
            // Refine the weights against the exact double-precision system built from the double tables.
            auto compiled64 = cfg.kernel.template compile<double>();
            compiled64.set_kernel_exp(cfg.kernel_exp);
            if (auto sp = compiled64.set_params(std::span<const double>(opt_res.params.data(), nk)); !sp.has_value())
            {
                return std::unexpected(std::move(sp).error());
            }
            auto store64 = compiled64.prepare_square(train);
            if (!store64.has_value())
            {
                return std::unexpected(std::move(store64).error());
            }
            std::optional<MatrixRef<double>> x64;
            if constexpr (with_points)
            {
                x64 = MatrixRef<double>(x_f64.data(), n, d, n);
            }
            const double diag_offset = payload.likelihood.noise_variance() + buffers->core.factor_jitter;
            MatrixMut<float> a_f32_mut = buffers->core.exp_mut();
            if constexpr (std::same_as<typename P::Residual, PromoteStorage>)
            {
                // Reassemble A = K + (noise + jitter) I in float for the promoted residual.
                if (auto ap = compiled->apply(
                        *store, x_ref, a_f32_mut, Triangle::Lower, buffers->core.kernel_scratch_mut(),
                        buffers->core.nested
                    );
                    !ap.has_value())
                {
                    return std::unexpected(std::move(ap).error());
                }
                if (auto dg = linalg::add_to_diag(a_f32_mut, static_cast<float>(diag_offset)); !dg.has_value())
                {
                    return std::unexpected(std::move(dg).error());
                }
                if (auto sy = linalg::symmetrize_lower(a_f32_mut); !sy.has_value())
                {
                    return std::unexpected(std::move(sy).error());
                }
            }
            std::vector<double> alpha64(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                alpha64[i] = static_cast<double>(buffers->core.rhs[i]);
            }
            DistanceRefineSystem<P> sys(
                compiled64, *store64, x64, std::span<const double>(y_f64), buffers->core.k_ref(), a_f32_mut.as_ref(),
                diag_offset
            );
            std::vector<double> residual_scratch(n);
            auto summary = refine_in_place(sys, alpha64, residual_scratch);
            if (!summary.has_value())
            {
                return std::unexpected(std::move(summary).error());
            }
            payload.refinement_summary = *summary;
            for (std::size_t i = 0; i < n; ++i)
            {
                payload.alpha[i] = alpha64[i];
            }
        }
        payload.factor_jitter = buffers->core.factor_jitter;
        payload.neg_log_marginal_likelihood = final_mll;
        payload.optimizer_result = std::move(opt_res);
        payload.l_factor = std::move(buffers->core.k_matrix);
        payload.compiled = std::move(compiled);
        payload.store = std::make_shared<const CheckedDistanceStore<Storage>>(std::move(*store));
        return payload;
    }

    namespace
    {

        /// @brief Validates the query coordinates of a WithPoints model and stages them in double and storage scalar.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in] xq Query coordinates (required for WithPoints).
        /// @param[in,out] scratch Query buffers receiving the staged coordinates.
        /// @param[in] m Number of query points defined by the cross tables.
        /// @return Staged coordinates, or GprError.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<std::optional<MatrixRef<typename P::Storage>>, GprError> stage_queries(
            const detail::DistanceFittedPayload<P, C>& payload,
            detail::DistanceQueryScratch<typename P::Storage>& scratch,
            std::optional<MatrixRef<double>> xq,
            std::size_t m
        )
        {
            using Storage = typename P::Storage;
            if constexpr (std::same_as<C, WithPoints>)
            {
                if (!xq.has_value() || xq->is_empty() || xq->data() == nullptr)
                {
                    return std::unexpected(GprError::empty_input());
                }
                if (xq->cols() != payload.n_features)
                {
                    return std::unexpected(GprError::dimension_mismatch(xq->cols(), payload.n_features));
                }
                if (xq->rows() != m)
                {
                    return std::unexpected(GprError::length_mismatch("query coordinate rows must match the tables"));
                }
                if (auto cp = copy_points(*xq, scratch.xq_double); !cp.has_value())
                {
                    return std::unexpected(std::move(cp).error());
                }
                if (auto ap = payload.input_transform->apply(scratch.xq_double, m, payload.n_features); !ap.has_value())
                {
                    return std::unexpected(std::move(ap).error());
                }
                scratch.xq.resize(scratch.xq_double.size());
                for (std::size_t i = 0; i < scratch.xq_double.size(); ++i)
                {
                    scratch.xq[i] = static_cast<Storage>(scratch.xq_double[i]);
                }
                return std::optional<MatrixRef<Storage>>(
                    MatrixRef<Storage>(scratch.xq.data(), m, payload.n_features, m)
                );
            }
            else
            {
                (void)payload;
                (void)scratch;
                (void)xq;
                (void)m;
                return std::optional<MatrixRef<Storage>>{};
            }
        }

        /// @brief Computes K(X, X*), the posterior mean weights and V = L^-1 K(X, X*) in the scratch buffers.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in,out] scratch Query buffers; k_star holds V on success.
        /// @param[in] cross Rectangular n x m tables.
        /// @param[in] xq Optional query coordinates (WithPoints).
        /// @param[out] mean_t Posterior mean on the transformed scale, length m.
        /// @param[out] xq_staged Staged query coordinates for the prior variance.
        /// @return Number of query points, or GprError.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<std::size_t, GprError> cross_and_solve(
            const detail::DistanceFittedPayload<P, C>& payload,
            detail::DistanceQueryScratch<typename P::Storage>& scratch,
            const DistanceSource& cross,
            std::optional<MatrixRef<double>> xq,
            std::vector<typename P::Refine>& mean_t,
            std::optional<MatrixRef<typename P::Storage>>& xq_staged
        )
        {
            using Storage = typename P::Storage;
            using Refine = typename P::Refine;
            const std::size_t n = payload.n_train;

            // Validate the query supply before any workspace growth.
            auto cross_store = payload.compiled->prepare_cross(cross);
            if (!cross_store.has_value())
            {
                return std::unexpected(std::move(cross_store).error());
            }
            if (cross_store->rows() != n)
            {
                return std::unexpected(GprError::length_mismatch("cross tables must have one row per training point"));
            }
            const std::size_t m = cross_store->cols();
            auto nm = checked_mul(n, m);
            if (!nm.has_value())
            {
                return std::unexpected(std::move(nm).error());
            }
            auto staged = stage_queries<P, C>(payload, scratch, xq, m);
            if (!staged.has_value())
            {
                return std::unexpected(std::move(staged).error());
            }
            xq_staged = *staged;

            if (scratch.k_star.size() < *nm)
            {
                scratch.k_star.resize(*nm);
            }
            if (scratch.scratch.size() < *nm)
            {
                scratch.scratch.resize(*nm);
            }
            if (scratch.nested.depth() < payload.compiled->nested_depth() || scratch.nested.rows() < n
                || scratch.nested.cols() < m)
            {
                scratch.nested.resize(payload.compiled->nested_depth(), n, m);
            }
            std::optional<MatrixRef<Storage>> x_train;
            if constexpr (std::same_as<C, WithPoints>)
            {
                x_train = MatrixRef<Storage>(payload.x_train.data(), n, payload.n_features, n);
            }
            MatrixMut<Storage> k_star(scratch.k_star.data(), n, m, n);
            MatrixMut<Storage> k_scratch(scratch.scratch.data(), n, m, n);
            if (auto ap =
                    payload.compiled->apply_cross(*cross_store, x_train, xq_staged, k_star, k_scratch, scratch.nested);
                !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }

            // Mean on the transformed scale: K*^T alpha with double accumulation in column order.
            mean_t.resize(m);
            const auto m_signed = static_cast<std::ptrdiff_t>(m);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n * m) >= 65536U)
#endif
            for (std::ptrdiff_t col = 0; col < m_signed; ++col)
            {
                const Storage* k_col = scratch.k_star.data() + (static_cast<std::size_t>(col) * n);
                double acc = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    acc += static_cast<double>(k_col[i]) * static_cast<double>(payload.alpha[i]);
                }
                mean_t[static_cast<std::size_t>(col)] = static_cast<Refine>(acc);
            }

            const MatrixRef<Storage> l_ref(payload.l_factor.data(), n, n, n);
            if (auto sv =
                    linalg::detail::solve_lower(l_ref, k_star, linalg::detail::FactorValidation::AlreadyValidated);
                !sv.has_value())
            {
                return std::unexpected(std::move(sv).error());
            }
            return m;
        }

        /// @brief Applies the inverse target transform to a mean vector through the double staging buffer.
        /// @tparam Refine Output scalar type.
        /// @param[in] tf Fitted target transform.
        /// @param[in,out] values Values transformed in place.
        /// @param[in,out] tmp Double staging buffer.
        /// @return Empty expected on success, or GprError.
        template <typename Refine>
        [[nodiscard]] std::expected<void, GprError> inverse_mean(
            const transform::FittedTargetTransform& tf, std::span<Refine> values, std::vector<double>& tmp
        )
        {
            tmp.assign(values.begin(), values.end());
            if (auto r = tf.inverse_transform_mean(tmp); !r.has_value())
            {
                return r;
            }
            std::ranges::transform(tmp, values.begin(), [](double v) { return static_cast<Refine>(v); });
            return {};
        }

        /// @brief Applies the inverse target transform to variances through the double staging buffer.
        /// @tparam Refine Output scalar type.
        /// @param[in] tf Fitted target transform.
        /// @param[in,out] values Variances transformed in place.
        /// @param[in,out] tmp Double staging buffer.
        /// @return Empty expected on success, or GprError.
        template <typename Refine>
        [[nodiscard]] std::expected<void, GprError> inverse_variance(
            const transform::FittedTargetTransform& tf, std::span<Refine> values, std::vector<double>& tmp
        )
        {
            tmp.assign(values.begin(), values.end());
            if (auto r = tf.inverse_transform_variance(tmp); !r.has_value())
            {
                return r;
            }
            std::ranges::transform(tmp, values.begin(), [](double v) { return static_cast<Refine>(v); });
            return {};
        }

        /// @brief Computes the joint mean and covariance on the transformed scale (no inverse transform).
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] payload Fitted state.
        /// @param[in,out] scratch Query buffers.
        /// @param[in] cross Rectangular n x m tables.
        /// @param[in] query Square m x m tables between the query points.
        /// @param[in] xq Optional query coordinates.
        /// @param[in] options Variance selection.
        /// @return Mean and covariance, or GprError.
        template <PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<PredictiveCovariance<typename P::Refine>, GprError> covariance_transformed(
            const detail::DistanceFittedPayload<P, C>& payload,
            detail::DistanceQueryScratch<typename P::Storage>& scratch,
            const DistanceSource& cross,
            const DistanceSource& query,
            std::optional<MatrixRef<double>> xq,
            const PredictOptions& options
        )
        {
            using Storage = typename P::Storage;
            using Refine = typename P::Refine;
            const std::size_t n = payload.n_train;

            std::vector<Refine> mean_t;
            std::optional<MatrixRef<Storage>> xq_staged;
            auto m_res = cross_and_solve<P, C>(payload, scratch, cross, xq, mean_t, xq_staged);
            if (!m_res.has_value())
            {
                return std::unexpected(std::move(m_res).error());
            }
            const std::size_t m = *m_res;

            auto query_store = payload.compiled->prepare_square(query);
            if (!query_store.has_value())
            {
                return std::unexpected(std::move(query_store).error());
            }
            if (query_store->rows() != m)
            {
                return std::unexpected(GprError::length_mismatch("query tables must match the cross tables"));
            }
            auto mm = checked_mul(m, m);
            if (!mm.has_value())
            {
                return std::unexpected(std::move(mm).error());
            }
            std::vector<Storage> k_qq(*mm);
            std::vector<Storage> k_qq_scratch(*mm);
            auto nested = payload.compiled->make_nested_buffers(m, m);
            MatrixMut<Storage> kqq_mut(k_qq.data(), m, m, m);
            MatrixMut<Storage> kqq_scratch_mut(k_qq_scratch.data(), m, m, m);
            if (auto ap =
                    payload.compiled->apply(*query_store, xq_staged, kqq_mut, Triangle::Lower, kqq_scratch_mut, nested);
                !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }

            PredictiveCovariance<Refine> out;
            out.mean = std::move(mean_t);
            out.n_points = m;
            out.variance_kind = options.variance_kind;
            out.covariance.assign(*mm, Refine{0});
            const Refine noise = (options.variance_kind == VarianceKind::Observation)
                ? static_cast<Refine>(payload.likelihood.noise_variance())
                : Refine{0};
            const auto m_signed = static_cast<std::ptrdiff_t>(m);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n * m * m) >= 4194304U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < m_signed; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const Storage* v_col = scratch.k_star.data() + (col * n);
                for (std::size_t row = col; row < m; ++row)
                {
                    const Storage* v_row = scratch.k_star.data() + (row * n);
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        acc += static_cast<double>(v_row[i]) * static_cast<double>(v_col[i]);
                    }
                    auto value = static_cast<Refine>(static_cast<double>(k_qq[(col * m) + row]) - acc);
                    if (row == col)
                    {
                        value += noise;
                    }
                    out.covariance[(col * m) + row] = value;
                    out.covariance[(row * m) + col] = value;
                }
            }
            return out;
        }

    }  // namespace

    /// @brief Predicts the mean and variance at query points from supplied cross distances.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] payload Fitted state.
    /// @param[in,out] scratch Reusable query buffers.
    /// @param[in] cross Cross squared distances between training and query points.
    /// @param[in] xq Optional query points (with-points mode).
    /// @param[in] options Prediction options.
    /// @param[out] out Prediction written in place.
    /// @return Success, or GprError.
    template <PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<void, GprError> detail::predict_distance_into(
        const DistanceFittedPayload<P, C>& payload,
        DistanceQueryScratch<typename P::Storage>& scratch,
        const DistanceSource& cross,
        std::optional<MatrixRef<double>> xq,
        const PredictOptions& options,
        Prediction<typename P::Refine>& out
    )
    {
        using Storage = typename P::Storage;
        using Refine = typename P::Refine;
        const std::size_t n = payload.n_train;

        std::optional<MatrixRef<Storage>> xq_staged;
        auto m_res = cross_and_solve<P, C>(payload, scratch, cross, xq, out.mean, xq_staged);
        if (!m_res.has_value())
        {
            return std::unexpected(std::move(m_res).error());
        }
        const std::size_t m = *m_res;

        // Prior variances k(x*, x*) at zero distance.
        if (scratch.k_ss.size() < m)
        {
            scratch.k_ss.resize(m);
        }
        if (auto diag = payload.compiled->fill_diag(m, xq_staged, std::span<Storage>(scratch.k_ss.data(), m));
            !diag.has_value())
        {
            return diag;
        }
        out.variance.resize(m);
        out.variance_kind = options.variance_kind;
        const Refine noise = (options.variance_kind == VarianceKind::Observation)
            ? static_cast<Refine>(payload.likelihood.noise_variance())
            : Refine{0};
        const auto m_signed = static_cast<std::ptrdiff_t>(m);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n * m) >= 65536U)
#endif
        for (std::ptrdiff_t col = 0; col < m_signed; ++col)
        {
            const auto j = static_cast<std::size_t>(col);
            const Storage* v_col = scratch.k_star.data() + (j * n);
            double sq = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                sq += static_cast<double>(v_col[i]) * static_cast<double>(v_col[i]);
            }
            const double latent = std::max(0.0, static_cast<double>(scratch.k_ss[j]) - sq);
            out.variance[j] = static_cast<Refine>(latent) + noise;
        }

        if (auto r = inverse_mean<Refine>(*payload.target_transform, std::span<Refine>(out.mean), scratch.mean_tmp);
            !r.has_value())
        {
            return r;
        }
        return inverse_variance<Refine>(*payload.target_transform, std::span<Refine>(out.variance), scratch.var_tmp);
    }

    /// @brief Predicts the full posterior covariance at query points from supplied distances.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] payload Fitted state.
    /// @param[in,out] scratch Reusable query buffers.
    /// @param[in] cross Cross squared distances between training and query points.
    /// @param[in] query Squared distances among the query points.
    /// @param[in] xq Optional query points (with-points mode).
    /// @param[in] options Prediction options.
    /// @return Predictive covariance, or GprError.
    template <PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<PredictiveCovariance<typename P::Refine>, GprError> detail::predict_distance_covariance(
        const DistanceFittedPayload<P, C>& payload,
        DistanceQueryScratch<typename P::Storage>& scratch,
        const DistanceSource& cross,
        const DistanceSource& query,
        std::optional<MatrixRef<double>> xq,
        const PredictOptions& options
    )
    {
        using Refine = typename P::Refine;
        auto cov = covariance_transformed<P, C>(payload, scratch, cross, query, xq, options);
        if (!cov.has_value())
        {
            return cov;
        }
        if (auto r = inverse_mean<Refine>(*payload.target_transform, std::span<Refine>(cov->mean), scratch.mean_tmp);
            !r.has_value())
        {
            return std::unexpected(std::move(r).error());
        }
        scratch.var_tmp.assign(cov->covariance.begin(), cov->covariance.end());
        if (auto r = payload.target_transform->inverse_transform_covariance(scratch.var_tmp); !r.has_value())
        {
            return std::unexpected(std::move(r).error());
        }
        std::ranges::transform(scratch.var_tmp, cov->covariance.begin(), [](double v) {
            return static_cast<Refine>(v);
        });
        return cov;
    }

    /// @brief Draws joint posterior samples at query points from supplied distances.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] payload Fitted state.
    /// @param[in,out] scratch Reusable query buffers.
    /// @param[in] cross Cross squared distances between training and query points.
    /// @param[in] query Squared distances among the query points.
    /// @param[in] xq Optional query points (with-points mode).
    /// @param[in] n_draws Number of samples to draw.
    /// @param[in] seed Seed of the deterministic generator.
    /// @param[in] options Prediction options.
    /// @return Samples, or GprError.
    template <PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<std::vector<typename P::Refine>, GprError> detail::sample_distance(
        const DistanceFittedPayload<P, C>& payload,
        DistanceQueryScratch<typename P::Storage>& scratch,
        const DistanceSource& cross,
        const DistanceSource& query,
        std::optional<MatrixRef<double>> xq,
        std::size_t n_draws,
        std::uint64_t seed,
        const PredictOptions& options
    )
    {
        using Refine = typename P::Refine;
        auto cov = covariance_transformed<P, C>(payload, scratch, cross, query, xq, options);
        if (!cov.has_value())
        {
            return std::unexpected(std::move(cov).error());
        }
        if (n_draws == 0U)
        {
            return std::vector<Refine>{};
        }
        const std::size_t m = cov->n_points;
        auto samples = draw_samples_from_covariance<Refine>(std::move(*cov), n_draws, seed, default_sample_jitter());
        if (!samples.has_value())
        {
            return samples;
        }
        for (std::size_t draw = 0; draw < n_draws; ++draw)
        {
            const std::span<Refine> slice(samples->data() + (draw * m), m);
            if (auto r = inverse_mean<Refine>(*payload.target_transform, slice, scratch.mean_tmp); !r.has_value())
            {
                return std::unexpected(std::move(r).error());
            }
        }
        return samples;
    }

    /// @brief Computes leave-one-out predictions of the training set.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] payload Fitted state.
    /// @param[in] options Prediction options.
    /// @return Leave-one-out prediction, or GprError.
    template <PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<Prediction<typename P::Refine>, GprError> detail::loo_distance(
        const DistanceFittedPayload<P, C>& payload, const PredictOptions& options
    )
    {
        using Storage = typename P::Storage;
        using Refine = typename P::Refine;
        const std::size_t n = payload.n_train;
        std::vector<Storage> scratch_matrix(n * n);
        std::vector<Storage> scratch_diag(n);
        Prediction<Refine> out;
        const MatrixRef<Storage> l_ref(payload.l_factor.data(), n, n, n);
        if (auto r = loo_predict_into<P>(
                payload.likelihood, l_ref, std::span<const Refine>(payload.alpha),
                std::span<const Refine>(payload.y_transformed), options,
                MatrixMut<Storage>(scratch_matrix.data(), n, n, n), std::span<Storage>(scratch_diag), out
            );
            !r.has_value())
        {
            return std::unexpected(std::move(r).error());
        }
        const auto& tf = *payload.target_transform;
        std::vector<double> tmp;
        if (auto r = inverse_mean<Refine>(tf, std::span<Refine>(out.mean), tmp); !r.has_value())
        {
            return std::unexpected(std::move(r).error());
        }
        if (auto r = inverse_variance<Refine>(tf, std::span<Refine>(out.variance), tmp); !r.has_value())
        {
            return std::unexpected(std::move(r).error());
        }
        return out;
    }

    /// @brief Assembles the fitted payload of a supplied-distance model from stored parts, without optimizing.
    /// @details Checks the stored factor and weights, validates the distance tables once, applies the fitted
    /// transforms, and derives the marginal likelihood and the trivial optimizer trajectory from the stored parts.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] kernel Kernel carrying the fitted hyperparameters.
    /// @param[in] likelihood Likelihood carrying the fitted noise variance.
    /// @param[in] kernel_exp Exponential mode of the kernel evaluation.
    /// @param[in] train Square training distance tables bound to the kernel slots.
    /// @param[in] x_raw Raw training coordinates, column-major n x n_features (WithPoints only).
    /// @param[in] n_features Coordinate dimension (0 for DistanceOnly).
    /// @param[in] y_raw Raw training targets.
    /// @param[in] input_transform Fitted coordinate transform (WithPoints only).
    /// @param[in] target_transform Fitted target transform.
    /// @param[in] l_factor Lower Cholesky factor, column-major n x n.
    /// @param[in] alpha Weights in the transformed target space.
    /// @param[in] factor_jitter Accepted diagonal jitter.
    /// @return Fitted payload, or GprError describing the first failing check.
    /// @retval GprErrorCode::PersistFailed When the factor or the weights are not usable.
    /// @retval GprErrorCode::InvalidDistance When the tables are invalid.
    template <PrecisionPolicyConcept P, DistanceModeConcept C>
    std::expected<detail::DistanceFittedPayload<P, C>, GprError> detail::assemble_distance_payload(
        const DistanceKernel<C>& kernel,
        const GaussianLikelihood& likelihood,
        KernelExp kernel_exp,
        const DistanceSource& train,
        std::span<const double> x_raw,
        std::size_t n_features,
        std::span<const double> y_raw,
        const std::shared_ptr<const transform::FittedInputTransform>& input_transform,
        const std::shared_ptr<const transform::FittedTargetTransform>& target_transform,
        std::span<const typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha,
        double factor_jitter
    )
    {
        using Storage = typename P::Storage;
        using Refine = typename P::Refine;
        constexpr bool with_points = std::same_as<C, WithPoints>;
        const std::size_t n = y_raw.size();
        if (n == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (l_factor.size() != n * n || alpha.size() != n)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "factor tensor has the wrong size")
            );
        }
        auto tensor_error = [](std::string_view message)
        {
            return GprError::persist_failed(PersistErrorKind::Tensor, std::string(message));
        };
        for (std::size_t j = 0; j < n; ++j)
        {
            const Storage diag = l_factor[(j * n) + j];
            if (!std::isfinite(diag) || diag <= Storage{0})
            {
                return std::unexpected(tensor_error("factor diagonal must be finite and positive"));
            }
            for (std::size_t i = j + 1U; i < n; ++i)
            {
                if (!std::isfinite(l_factor[(j * n) + i]))
                {
                    return std::unexpected(tensor_error("factor contains a non-finite value"));
                }
            }
        }
        for (const auto a : alpha)
        {
            if (!std::isfinite(a))
            {
                return std::unexpected(tensor_error("weights contain a non-finite value"));
            }
        }

        auto compiled = std::make_shared<CompiledDistanceKernel<Storage>>(kernel.template compile<Storage>());
        compiled->set_kernel_exp(kernel_exp);
        auto store = compiled->prepare_square(train);
        if (!store.has_value())
        {
            return std::unexpected(std::move(store).error());
        }
        if (store->rows() != n)
        {
            return std::unexpected(GprError::length_mismatch("target length must match training table order"));
        }

        DistanceFittedPayload<P, C> payload{
            .kernel = kernel,
            .likelihood = likelihood,
            .compiled = nullptr,
            .store = nullptr,
            .input_transform = nullptr,
            .target_transform = target_transform,
            .x_raw = {},
            .y_raw = std::vector<double>(y_raw.begin(), y_raw.end()),
            .x_train = {},
            .y_transformed = {},
            .l_factor = std::vector<Storage>(l_factor.begin(), l_factor.end()),
            .alpha = std::vector<Refine>(alpha.begin(), alpha.end()),
            .n_train = n,
            .n_features = n_features,
            .factor_jitter = factor_jitter,
            .neg_log_marginal_likelihood = 0.0,
            .optimizer_result = {},
            .refinement_summary = std::nullopt,
        };
        if constexpr (with_points)
        {
            if (!input_transform || x_raw.size() != n * n_features)
            {
                return std::unexpected(GprError::empty_input());
            }
            payload.x_raw.assign(x_raw.begin(), x_raw.end());
            std::vector<double> x_t(x_raw.begin(), x_raw.end());
            if (auto ap = input_transform->apply(x_t, n, n_features); !ap.has_value())
            {
                return std::unexpected(std::move(ap).error());
            }
            payload.x_train.resize(x_t.size());
            for (std::size_t i = 0; i < x_t.size(); ++i)
            {
                if (!std::isfinite(x_t[i]))
                {
                    return std::unexpected(tensor_error("non-finite value after applying the input transform"));
                }
                payload.x_train[i] = static_cast<Storage>(x_t[i]);
            }
            payload.input_transform = input_transform;
        }
        std::vector<double> y_t(y_raw.begin(), y_raw.end());
        if (auto tf = payload.target_transform->transform(y_t); !tf.has_value())
        {
            return std::unexpected(std::move(tf).error());
        }
        payload.y_transformed.resize(n);
        double quad = 0.0;
        double log_det_half = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!std::isfinite(y_t[i]))
            {
                return std::unexpected(tensor_error("non-finite value after applying the target transform"));
            }
            payload.y_transformed[i] = static_cast<Refine>(y_t[i]);
            quad += y_t[i] * static_cast<double>(alpha[i]);
            log_det_half += std::log(static_cast<double>(l_factor[(i * n) + i]));
        }
        constexpr double log_two_pi = 1.8378770664093454835606594728112;
        payload.neg_log_marginal_likelihood = (0.5 * quad) + log_det_half + (0.5 * static_cast<double>(n) * log_two_pi);

        const std::size_t nk = kernel.num_params();
        std::vector<double> params(nk + 1U);
        if (auto gp = kernel.get_params(std::span<double>(params.data(), nk)); !gp.has_value())
        {
            return std::unexpected(std::move(gp).error());
        }
        if (auto lp = likelihood.get_params(std::span<double>(params.data() + nk, 1U)); !lp.has_value())
        {
            return std::unexpected(std::move(lp).error());
        }
        payload.optimizer_result = optimizer::OptimizerResult{
            .params = std::move(params),
            .value = payload.neg_log_marginal_likelihood,
            .iterations = 0U,
            .converged = true,
        };
        payload.compiled = std::move(compiled);
        payload.store = std::make_shared<const CheckedDistanceStore<Storage>>(std::move(*store));
        return payload;
    }

    // Explicit instantiations of the supplied-distance engine for every precision policy and kernel mode.

    template std::expected<detail::DistanceFittedPayload<DoublePrecision, DistanceOnly>, GprError> detail::
        train_distance_payload<DoublePrecision, DistanceOnly>(
            const DistanceTrainConfig<DistanceOnly>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<DoublePrecision, DistanceOnly>(
        const DistanceFittedPayload<DoublePrecision, DistanceOnly>&,
        DistanceQueryScratch<DoublePrecision::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<DoublePrecision::Refine>&
    );
    template std::expected<PredictiveCovariance<DoublePrecision::Refine>, GprError> detail::
        predict_distance_covariance<DoublePrecision, DistanceOnly>(
            const DistanceFittedPayload<DoublePrecision, DistanceOnly>&,
            DistanceQueryScratch<DoublePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<DoublePrecision::Refine>, GprError> detail::
        sample_distance<DoublePrecision, DistanceOnly>(
            const DistanceFittedPayload<DoublePrecision, DistanceOnly>&,
            DistanceQueryScratch<DoublePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<DoublePrecision::Refine>, GprError> detail::
        loo_distance<DoublePrecision, DistanceOnly>(
            const DistanceFittedPayload<DoublePrecision, DistanceOnly>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<DoublePrecision, WithPoints>, GprError> detail::
        train_distance_payload<DoublePrecision, WithPoints>(
            const DistanceTrainConfig<WithPoints>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<DoublePrecision, WithPoints>(
        const DistanceFittedPayload<DoublePrecision, WithPoints>&,
        DistanceQueryScratch<DoublePrecision::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<DoublePrecision::Refine>&
    );
    template std::expected<PredictiveCovariance<DoublePrecision::Refine>, GprError> detail::
        predict_distance_covariance<DoublePrecision, WithPoints>(
            const DistanceFittedPayload<DoublePrecision, WithPoints>&,
            DistanceQueryScratch<DoublePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<DoublePrecision::Refine>, GprError> detail::
        sample_distance<DoublePrecision, WithPoints>(
            const DistanceFittedPayload<DoublePrecision, WithPoints>&,
            DistanceQueryScratch<DoublePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<DoublePrecision::Refine>, GprError> detail::
        loo_distance<DoublePrecision, WithPoints>(
            const DistanceFittedPayload<DoublePrecision, WithPoints>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<SinglePrecision, DistanceOnly>, GprError> detail::
        train_distance_payload<SinglePrecision, DistanceOnly>(
            const DistanceTrainConfig<DistanceOnly>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<SinglePrecision, DistanceOnly>(
        const DistanceFittedPayload<SinglePrecision, DistanceOnly>&,
        DistanceQueryScratch<SinglePrecision::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<SinglePrecision::Refine>&
    );
    template std::expected<PredictiveCovariance<SinglePrecision::Refine>, GprError> detail::
        predict_distance_covariance<SinglePrecision, DistanceOnly>(
            const DistanceFittedPayload<SinglePrecision, DistanceOnly>&,
            DistanceQueryScratch<SinglePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<SinglePrecision::Refine>, GprError> detail::
        sample_distance<SinglePrecision, DistanceOnly>(
            const DistanceFittedPayload<SinglePrecision, DistanceOnly>&,
            DistanceQueryScratch<SinglePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<SinglePrecision::Refine>, GprError> detail::
        loo_distance<SinglePrecision, DistanceOnly>(
            const DistanceFittedPayload<SinglePrecision, DistanceOnly>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<SinglePrecision, WithPoints>, GprError> detail::
        train_distance_payload<SinglePrecision, WithPoints>(
            const DistanceTrainConfig<WithPoints>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<SinglePrecision, WithPoints>(
        const DistanceFittedPayload<SinglePrecision, WithPoints>&,
        DistanceQueryScratch<SinglePrecision::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<SinglePrecision::Refine>&
    );
    template std::expected<PredictiveCovariance<SinglePrecision::Refine>, GprError> detail::
        predict_distance_covariance<SinglePrecision, WithPoints>(
            const DistanceFittedPayload<SinglePrecision, WithPoints>&,
            DistanceQueryScratch<SinglePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<SinglePrecision::Refine>, GprError> detail::
        sample_distance<SinglePrecision, WithPoints>(
            const DistanceFittedPayload<SinglePrecision, WithPoints>&,
            DistanceQueryScratch<SinglePrecision::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<SinglePrecision::Refine>, GprError> detail::
        loo_distance<SinglePrecision, WithPoints>(
            const DistanceFittedPayload<SinglePrecision, WithPoints>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>, GprError>
        detail::train_distance_payload<MixedPrecision<PromoteStorage>, DistanceOnly>(
            const DistanceTrainConfig<DistanceOnly>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<MixedPrecision<PromoteStorage>, DistanceOnly>(
        const DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>&,
        DistanceQueryScratch<MixedPrecision<PromoteStorage>::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<MixedPrecision<PromoteStorage>::Refine>&
    );
    template std::expected<PredictiveCovariance<MixedPrecision<PromoteStorage>::Refine>, GprError> detail::
        predict_distance_covariance<MixedPrecision<PromoteStorage>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>&,
            DistanceQueryScratch<MixedPrecision<PromoteStorage>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<MixedPrecision<PromoteStorage>::Refine>, GprError> detail::
        sample_distance<MixedPrecision<PromoteStorage>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>&,
            DistanceQueryScratch<MixedPrecision<PromoteStorage>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<MixedPrecision<PromoteStorage>::Refine>, GprError> detail::
        loo_distance<MixedPrecision<PromoteStorage>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>, GprError> detail::
        train_distance_payload<MixedPrecision<PromoteStorage>, WithPoints>(
            const DistanceTrainConfig<WithPoints>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<MixedPrecision<PromoteStorage>, WithPoints>(
        const DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>&,
        DistanceQueryScratch<MixedPrecision<PromoteStorage>::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<MixedPrecision<PromoteStorage>::Refine>&
    );
    template std::expected<PredictiveCovariance<MixedPrecision<PromoteStorage>::Refine>, GprError> detail::
        predict_distance_covariance<MixedPrecision<PromoteStorage>, WithPoints>(
            const DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>&,
            DistanceQueryScratch<MixedPrecision<PromoteStorage>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<MixedPrecision<PromoteStorage>::Refine>, GprError> detail::
        sample_distance<MixedPrecision<PromoteStorage>, WithPoints>(
            const DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>&,
            DistanceQueryScratch<MixedPrecision<PromoteStorage>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<MixedPrecision<PromoteStorage>::Refine>, GprError> detail::
        loo_distance<MixedPrecision<PromoteStorage>, WithPoints>(
            const DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>, GprError>
        detail::train_distance_payload<MixedPrecision<ReevaluateKernel>, DistanceOnly>(
            const DistanceTrainConfig<DistanceOnly>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::
        predict_distance_into<MixedPrecision<ReevaluateKernel>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>&,
            DistanceQueryScratch<MixedPrecision<ReevaluateKernel>::Storage>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&,
            Prediction<MixedPrecision<ReevaluateKernel>::Refine>&
        );
    template std::expected<PredictiveCovariance<MixedPrecision<ReevaluateKernel>::Refine>, GprError> detail::
        predict_distance_covariance<MixedPrecision<ReevaluateKernel>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>&,
            DistanceQueryScratch<MixedPrecision<ReevaluateKernel>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<MixedPrecision<ReevaluateKernel>::Refine>, GprError> detail::
        sample_distance<MixedPrecision<ReevaluateKernel>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>&,
            DistanceQueryScratch<MixedPrecision<ReevaluateKernel>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<MixedPrecision<ReevaluateKernel>::Refine>, GprError> detail::
        loo_distance<MixedPrecision<ReevaluateKernel>, DistanceOnly>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>, GprError>
        detail::train_distance_payload<MixedPrecision<ReevaluateKernel>, WithPoints>(
            const DistanceTrainConfig<WithPoints>&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::span<const double>,
            std::expected<optimizer::OptimizerResult, GprError> (*)(
                void*, optimizer::TwiceDifferentiable&, std::span<const double>
            ),
            void*
        );
    template std::expected<void, GprError> detail::predict_distance_into<MixedPrecision<ReevaluateKernel>, WithPoints>(
        const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>&,
        DistanceQueryScratch<MixedPrecision<ReevaluateKernel>::Storage>&,
        const DistanceSource&,
        std::optional<MatrixRef<double>>,
        const PredictOptions&,
        Prediction<MixedPrecision<ReevaluateKernel>::Refine>&
    );
    template std::expected<PredictiveCovariance<MixedPrecision<ReevaluateKernel>::Refine>, GprError> detail::
        predict_distance_covariance<MixedPrecision<ReevaluateKernel>, WithPoints>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>&,
            DistanceQueryScratch<MixedPrecision<ReevaluateKernel>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            const PredictOptions&
        );
    template std::expected<std::vector<MixedPrecision<ReevaluateKernel>::Refine>, GprError> detail::
        sample_distance<MixedPrecision<ReevaluateKernel>, WithPoints>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>&,
            DistanceQueryScratch<MixedPrecision<ReevaluateKernel>::Storage>&,
            const DistanceSource&,
            const DistanceSource&,
            std::optional<MatrixRef<double>>,
            std::size_t,
            std::uint64_t,
            const PredictOptions&
        );
    template std::expected<Prediction<MixedPrecision<ReevaluateKernel>::Refine>, GprError> detail::
        loo_distance<MixedPrecision<ReevaluateKernel>, WithPoints>(
            const DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>&, const PredictOptions&
        );

    template std::expected<detail::DistanceFittedPayload<DoublePrecision, DistanceOnly>, GprError> detail::
        assemble_distance_payload<DoublePrecision, DistanceOnly>(
            const DistanceKernel<DistanceOnly>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const DoublePrecision::Storage>,
            std::span<const DoublePrecision::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<DoublePrecision, WithPoints>, GprError> detail::
        assemble_distance_payload<DoublePrecision, WithPoints>(
            const DistanceKernel<WithPoints>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const DoublePrecision::Storage>,
            std::span<const DoublePrecision::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<SinglePrecision, DistanceOnly>, GprError> detail::
        assemble_distance_payload<SinglePrecision, DistanceOnly>(
            const DistanceKernel<DistanceOnly>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const SinglePrecision::Storage>,
            std::span<const SinglePrecision::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<SinglePrecision, WithPoints>, GprError> detail::
        assemble_distance_payload<SinglePrecision, WithPoints>(
            const DistanceKernel<WithPoints>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const SinglePrecision::Storage>,
            std::span<const SinglePrecision::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<PromoteStorage>, DistanceOnly>, GprError>
        detail::assemble_distance_payload<MixedPrecision<PromoteStorage>, DistanceOnly>(
            const DistanceKernel<DistanceOnly>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const MixedPrecision<PromoteStorage>::Storage>,
            std::span<const MixedPrecision<PromoteStorage>::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<PromoteStorage>, WithPoints>, GprError> detail::
        assemble_distance_payload<MixedPrecision<PromoteStorage>, WithPoints>(
            const DistanceKernel<WithPoints>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const MixedPrecision<PromoteStorage>::Storage>,
            std::span<const MixedPrecision<PromoteStorage>::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, DistanceOnly>, GprError>
        detail::assemble_distance_payload<MixedPrecision<ReevaluateKernel>, DistanceOnly>(
            const DistanceKernel<DistanceOnly>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const MixedPrecision<ReevaluateKernel>::Storage>,
            std::span<const MixedPrecision<ReevaluateKernel>::Refine>,
            double
        );

    template std::expected<detail::DistanceFittedPayload<MixedPrecision<ReevaluateKernel>, WithPoints>, GprError>
        detail::assemble_distance_payload<MixedPrecision<ReevaluateKernel>, WithPoints>(
            const DistanceKernel<WithPoints>&,
            const GaussianLikelihood&,
            KernelExp,
            const DistanceSource&,
            std::span<const double>,
            std::size_t,
            std::span<const double>,
            const std::shared_ptr<const transform::FittedInputTransform>&,
            const std::shared_ptr<const transform::FittedTargetTransform>&,
            std::span<const MixedPrecision<ReevaluateKernel>::Storage>,
            std::span<const MixedPrecision<ReevaluateKernel>::Refine>,
            double
        );

}  // namespace cxxgpr
