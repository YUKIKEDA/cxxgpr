// SPDX-License-Identifier: Apache-2.0
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

        template <KernelScalarConcept T>
        void build_w_lower_from_inv(std::span<const T> alpha, MatrixRef<T> a_inv, MatrixMut<T> w_out) noexcept
        {
            const std::size_t n = alpha.size();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
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
                    u[row] += v * a_col;
                    acc += v * alpha[row];
                }
                u[col] += acc;
            }
        }

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

        template <KernelScalarConcept T>
        [[nodiscard]] double frobenius_lower_det(
            MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials
        ) noexcept
        {
            const std::size_t n = a.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
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
            double total = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                total += col_partials[col];
            }
            return total;
        }

        template <KernelScalarConcept T>
        [[nodiscard]] double frobenius_full_det(MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials) noexcept
        {
            const std::size_t n = a.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
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
            double total = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                total += col_partials[col];
            }
            return total;
        }

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

            if (buffers.overwrites_cholesky())
            {
                auto restore_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, 0U);
                if (!restore_res.has_value())
                {
                    return std::unexpected(std::move(restore_res).error());
                }
            }

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

                const auto d_a = buffers.core.exp_ref();
                gemv_sym_lower<Storage>(d_a, alpha, std::span<Storage>(buffers.core.hessian.u));
                auto v_i = buffers.core.hessian.v_col_mut(i);
                std::ranges::copy(buffers.core.hessian.u, v_i.begin());
                if (auto sol_v = linalg::solve_lower_in_place(l_ref, v_i); !sol_v.has_value())
                {
                    return sol_v;
                }

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

        const auto w_ref = buffers.w_ref();
        double trace_w = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            trace_w += static_cast<double>(w_ref(i, i));
        }
        out_grad[n_kernel] = -0.5 * likelihood.noise_variance() * trace_w;

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
