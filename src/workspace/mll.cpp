// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/workspace/mll.hpp"

#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cxxgpr {

namespace {

template <KernelScalarConcept T> void fill_identity(MatrixMut<T> mat) noexcept {
    const std::size_t n = mat.rows();
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            mat(row, col) = (row == col) ? T{1} : T{0};
        }
    }
}

template <KernelScalarConcept T>
void build_w_lower_from_inv(std::span<const T> alpha, MatrixRef<T> a_inv,
                            MatrixMut<T> w_out) noexcept {
    const std::size_t n = alpha.size();
    const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 64U)
#endif
    for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s) {
        const auto col = static_cast<std::size_t>(col_s);
        const T a_col = alpha[col];
        for (std::size_t row = col; row < n; ++row) {
            w_out(row, col) = (alpha[row] * a_col) - a_inv(row, col);
        }
    }
}

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<const DistanceCache<typename P::Storage>*, GprError>
prepare_fit_dist_cache(const CompiledKernel<typename P::Storage>& kernel,
                       MatrixRef<typename P::Storage> x, FitBuffers<P>& buffers) {
    if (!buffers.dist.has_value()) {
        return nullptr;
    }
    const bool need_dist = kernel.reads_distances();
    const bool need_ard = kernel.needs_ard_sq_diff();
    if (!need_dist && !need_ard) {
        return &(*buffers.dist);
    }
    const bool matches =
        buffers.dist->num_points() == x.rows() && buffers.dist->num_dims() == x.cols() &&
        (!need_dist || buffers.dist->has_dist()) && (!need_ard || buffers.dist->has_ard());
    if (!matches) {
        if (auto ok = buffers.dist->ensure(x, need_dist, need_ard, DistanceCachePolicy::Cached);
            !ok.has_value()) {
            return std::unexpected(std::move(ok).error());
        }
    }
    return &(*buffers.dist);
}

} // namespace

template <PrecisionPolicyConcept P>
std::expected<double, GprError>
factor_and_mll(const CompiledKernel<typename P::Storage>& kernel,
               const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> x,
               std::span<const typename P::Storage> y, const JitterPolicy& jitter,
               FitBuffers<P>& buffers, std::size_t keep_products) {
    using Storage = P::Storage;
    if (x.is_empty() || x.data() == nullptr || y.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t n = buffers.core.order();
    if (x.rows() != n || y.size() != n) {
        return std::unexpected(GprError::length_mismatch("training sample count mismatch"));
    }
    for (const Storage val : y) {
        if (!std::isfinite(val)) {
            return std::unexpected(GprError::non_finite_input());
        }
    }

    auto dist_ptr = prepare_fit_dist_cache(kernel, x, buffers);
    if (!dist_ptr.has_value()) {
        return std::unexpected(std::move(dist_ptr).error());
    }

    if (buffers.core.nested.depth() < kernel.nested_depth() || buffers.core.nested.rows() < n ||
        buffers.core.nested.cols() < n) {
        buffers.core.nested.resize(kernel.nested_depth(), n, n);
    }

    const std::size_t kept_count = kernel.kept_buffers(keep_products);
    if (kept_count > 0U) {
        if (auto ok = buffers.core.ensure_weighted(kernel.weighted_buffers(keep_products));
            !ok.has_value()) {
            return std::unexpected(std::move(ok).error());
        }
    }

    std::expected<void, GprError> eval_res;
    if (*dist_ptr != nullptr && ((*dist_ptr)->has_dist() || (*dist_ptr)->has_ard())) {
        eval_res = kernel.apply_cached_keeping(x, **dist_ptr, buffers.core.k_mut(), Triangle::Lower,
                                               buffers.core.exp_mut(), buffers.core.nested,
                                               buffers.core.weighted, keep_products);
    } else {
        eval_res =
            kernel.apply_keeping(x, buffers.core.k_mut(), Triangle::Lower, buffers.core.exp_mut(),
                                 buffers.core.nested, buffers.core.weighted, keep_products);
    }
    if (!eval_res.has_value()) {
        return std::unexpected(std::move(eval_res).error());
    }

    if (auto diag_ok = linalg::add_to_diag(buffers.core.k_mut(),
                                           static_cast<Storage>(likelihood.noise_variance()));
        !diag_ok.has_value()) {
        return std::unexpected(std::move(diag_ok).error());
    }

    auto jit_res = linalg::cholesky_lower_with_jitter(buffers.core.k_mut(), buffers.core.exp_mut(),
                                                      jitter, CholeskyStage::Fit);
    if (!jit_res.has_value()) {
        return std::unexpected(std::move(jit_res).error());
    }
    buffers.core.factor_jitter = *jit_res;

    std::ranges::copy(y, buffers.core.rhs.begin());
    if (auto sol_ok =
            linalg::solve_llt_in_place(buffers.core.k_ref(), std::span<Storage>(buffers.core.rhs));
        !sol_ok.has_value()) {
        return std::unexpected(std::move(sol_ok).error());
    }

    auto dot_res = linalg::dot<Storage>(y, std::span<const Storage>(buffers.core.rhs));
    if (!dot_res.has_value()) {
        return std::unexpected(std::move(dot_res).error());
    }
    auto logdet_res = linalg::log_det_from_l(buffers.core.k_ref());
    if (!logdet_res.has_value()) {
        return std::unexpected(std::move(logdet_res).error());
    }
    const auto n_dbl = static_cast<double>(n);
    return 0.5 * (*dot_res + *logdet_res + (n_dbl * std::log(2.0 * std::numbers::pi)));
}

template <PrecisionPolicyConcept P>
std::expected<double, GprError>
mll_and_gradient_into(const CompiledKernel<typename P::Storage>& kernel,
                      const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> x,
                      std::span<const typename P::Storage> y, const JitterPolicy& jitter,
                      FitBuffers<P>& buffers, std::span<double> out_grad) {
    using Storage = P::Storage;
    const std::size_t n_kernel = kernel.num_params();
    if (out_grad.size() != n_kernel + 1U) {
        return std::unexpected(GprError::length_mismatch("gradient output length mismatch"));
    }

    const std::size_t products = kernel.kept_products();
    if (auto ok = buffers.core.ensure_weighted(kernel.weighted_buffers(products));
        !ok.has_value()) {
        return std::unexpected(std::move(ok).error());
    }
    if (auto ok = buffers.core.ensure_kernel_scratch(); !ok.has_value()) {
        return std::unexpected(std::move(ok).error());
    }

    auto mll_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, products);
    if (!mll_res.has_value()) {
        return mll_res;
    }

    const std::size_t n = buffers.core.order();
    MatrixMut<Storage> inv_buf =
        buffers.overwrites_cholesky() ? buffers.core.exp_mut() : buffers.w_mut();
    fill_identity(inv_buf);
    if (auto inv_ok = linalg::solve_llt_in_place(buffers.core.k_ref(), inv_buf);
        !inv_ok.has_value()) {
        return std::unexpected(std::move(inv_ok).error());
    }
    build_w_lower_from_inv<Storage>(buffers.core.rhs, inv_buf.as_ref(), buffers.w_mut());

    const DistanceCache<Storage>* dist_ptr =
        (buffers.dist.has_value() && (buffers.dist->has_dist() || buffers.dist->has_ard()))
            ? &(*buffers.dist)
            : nullptr;

    if (auto wg_ok = kernel.weighted_grads(
            x, dist_ptr, buffers.w_ref(), out_grad.subspan(0, n_kernel),
            buffers.core.kernel_scratch_mut(), buffers.core.nested, buffers.core.weighted, products,
            std::span<double>(buffers.core.col_partials));
        !wg_ok.has_value()) {
        return std::unexpected(std::move(wg_ok).error());
    }
    for (std::size_t p = 0; p < n_kernel; ++p) {
        out_grad[p] *= -0.5;
    }

    const auto w_ref = buffers.w_ref();
    double trace_w = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        trace_w += static_cast<double>(w_ref(i, i));
    }
    out_grad[n_kernel] = -0.5 * likelihood.noise_variance() * trace_w;

    if (buffers.overwrites_cholesky()) {
        // Restore Cholesky factor L overwritten by W in Reuse mode.
        auto restore_res = factor_and_mll(kernel, likelihood, x, y, jitter, buffers, 0U);
        if (!restore_res.has_value()) {
            return std::unexpected(std::move(restore_res).error());
        }
    }

    return *mll_res;
}

template std::expected<double, GprError>
factor_and_mll<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                                MatrixRef<double>, std::span<const double>, const JitterPolicy&,
                                FitBuffers<DoublePrecision>&, std::size_t);
template std::expected<double, GprError>
factor_and_mll<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                MatrixRef<float>, std::span<const float>, const JitterPolicy&,
                                FitBuffers<SinglePrecision>&, std::size_t);
template std::expected<double, GprError> factor_and_mll<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<PromoteStorage>>&,
    std::size_t);
template std::expected<double, GprError> factor_and_mll<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<ReevaluateKernel>>&,
    std::size_t);

template std::expected<double, GprError> mll_and_gradient_into<DoublePrecision>(
    const CompiledKernel<double>&, const GaussianLikelihood&, MatrixRef<double>,
    std::span<const double>, const JitterPolicy&, FitBuffers<DoublePrecision>&, std::span<double>);
template std::expected<double, GprError> mll_and_gradient_into<SinglePrecision>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<SinglePrecision>&, std::span<double>);
template std::expected<double, GprError> mll_and_gradient_into<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<PromoteStorage>>&,
    std::span<double>);
template std::expected<double, GprError> mll_and_gradient_into<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<ReevaluateKernel>>&,
    std::span<double>);

} // namespace cxxgpr
