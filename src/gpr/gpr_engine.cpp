// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/gpr/gpr.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "cxxgpr/workspace/inference.hpp"
#include "cxxgpr/workspace/mll.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace cxxgpr::detail {
namespace {

template <PrecisionPolicyConcept P>
class GprObjective final : public optimizer::TwiceDifferentiable {
public:
    using Storage = P::Storage;

    GprObjective(CompiledKernel<Storage>& kernel, GaussianLikelihood& likelihood,
                 MatrixRef<Storage> x_train, std::span<const Storage> y_train,
                 const JitterPolicy& jitter, FitBuffers<P>& buffers, bool enable_leaf_cache)
        : kernel_(kernel), likelihood_(likelihood), x_train_(x_train), y_train_(y_train),
          jitter_(jitter), buffers_(buffers),
          enable_leaf_cache_(enable_leaf_cache && buffers.buffer_policy == CholeskyBuffer::Retain) {
        if (enable_leaf_cache_) {
            (void)leaf_cache_.resize(kernel_.num_leaves(), x_train_.rows());
        }
    }

    [[nodiscard]] std::size_t dim() const noexcept override {
        return kernel_.num_params() + likelihood_.num_params();
    }

    [[nodiscard]] std::expected<double, GprError> value(std::span<const double> params) override {
        if (auto set = apply_params(params); !set.has_value()) {
            return std::unexpected(std::move(set).error());
        }
        if (enable_leaf_cache_) {
            leaf_cache_.invalidate();
        }
        return factor_and_mll(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_, 0U);
    }

    [[nodiscard]] std::expected<double, GprError>
    value_at_changes(std::span<const double> params,
                     std::span<const std::size_t> changed_indices) override {
        if (!enable_leaf_cache_) {
            return value(params);
        }
        if (auto set = apply_params(params); !set.has_value()) {
            return std::unexpected(std::move(set).error());
        }

        const DistanceCache<Storage>* dist_ptr =
            buffers_.dist.has_value() ? &(*buffers_.dist) : nullptr;
        auto inc = leaf_cache_.evaluate_incremental(
            kernel_, x_train_, dist_ptr, params, changed_indices, dim(), buffers_.core.k_mut(),
            Triangle::Lower, buffers_.core.exp_mut(), buffers_.core.nested);
        if (!inc.has_value()) {
            return std::unexpected(std::move(inc).error());
        }

        const std::size_t n = x_train_.rows();
        const auto noise_var = static_cast<Storage>(likelihood_.noise_variance());
        if (auto diag = linalg::add_to_diag(buffers_.core.k_mut(), noise_var); !diag.has_value()) {
            return std::unexpected(std::move(diag).error());
        }
        auto jit = linalg::cholesky_lower_with_jitter(
            buffers_.core.k_mut(), buffers_.core.exp_mut(), jitter_, CholeskyStage::Fit);
        if (!jit.has_value()) {
            return std::unexpected(std::move(jit).error());
        }
        buffers_.core.factor_jitter = *jit;

        std::ranges::copy(y_train_, buffers_.core.rhs.begin());
        if (auto solve = linalg::solve_llt_in_place(buffers_.core.k_ref(),
                                                    std::span<Storage>(buffers_.core.rhs));
            !solve.has_value()) {
            return std::unexpected(std::move(solve).error());
        }
        auto half_ld = linalg::half_log_det_from_l(buffers_.core.k_ref());
        if (!half_ld.has_value()) {
            return std::unexpected(std::move(half_ld).error());
        }
        auto y_alpha = linalg::dot(y_train_, std::span<const Storage>(buffers_.core.rhs));
        if (!y_alpha.has_value()) {
            return std::unexpected(std::move(y_alpha).error());
        }
        const double log_2pi = std::log(2.0 * std::numbers::pi);
        const double mll =
            (0.5 * (*y_alpha)) + (*half_ld) + (0.5 * static_cast<double>(n) * log_2pi);
        if (!std::isfinite(mll)) {
            return std::unexpected(GprError::non_finite_kernel_value());
        }
        return mll;
    }

    [[nodiscard]] std::expected<void, GprError>
    fill_intervals(std::span<Interval> out) const override {
        const std::size_t nk = kernel_.num_params();
        if (out.size() != nk + 1U) {
            return std::unexpected(
                GprError::length_mismatch("interval buffer length must match parameter count"));
        }
        if (auto kb = kernel_.bounds_into(out.subspan(0, nk)); !kb.has_value()) {
            return kb;
        }
        return likelihood_.bounds_into(out.subspan(nk, 1U));
    }

    [[nodiscard]] std::expected<void, GprError> gradient_into(std::span<const double> params,
                                                              std::span<double> out_grad) override {
        auto mll = value_and_gradient_into(params, out_grad);
        if (!mll.has_value()) {
            return std::unexpected(std::move(mll).error());
        }
        return {};
    }

    [[nodiscard]] std::expected<double, GprError>
    value_and_gradient_into(std::span<const double> params, std::span<double> out_grad) override {
        if (auto set = apply_params(params); !set.has_value()) {
            return std::unexpected(std::move(set).error());
        }
        if (enable_leaf_cache_) {
            leaf_cache_.invalidate();
        }
        return mll_and_gradient_into(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_,
                                     out_grad);
    }

    [[nodiscard]] std::expected<void, GprError> hessian_into(std::span<const double> params,
                                                             std::span<double> out_hess) override {
        if (auto set = apply_params(params); !set.has_value()) {
            return set;
        }
        if (enable_leaf_cache_) {
            leaf_cache_.invalidate();
        }
        return cxxgpr::hessian_into(kernel_, likelihood_, x_train_, y_train_, jitter_, buffers_,
                                    out_hess);
    }

    [[nodiscard]] std::expected<double, GprError>
    value_gradient_hessian_into(std::span<const double> params, std::span<double> out_grad,
                                std::span<double> out_hess) override {
        if (auto set = apply_params(params); !set.has_value()) {
            return std::unexpected(std::move(set).error());
        }
        if (enable_leaf_cache_) {
            leaf_cache_.invalidate();
        }
        return cxxgpr::value_gradient_hessian_into(kernel_, likelihood_, x_train_, y_train_,
                                                   jitter_, buffers_, out_grad, out_hess);
    }

private:
    [[nodiscard]] std::expected<void, GprError> apply_params(std::span<const double> params) {
        const std::size_t nk = kernel_.num_params();
        if (params.size() != nk + 1U) {
            return std::unexpected(
                GprError::length_mismatch("parameter length must match kernel + noise params"));
        }
        if (auto kp = kernel_.set_params(params.subspan(0, nk)); !kp.has_value()) {
            return kp;
        }
        return likelihood_.set_params(params.subspan(nk, 1U));
    }

    CompiledKernel<Storage>& kernel_;
    GaussianLikelihood& likelihood_;
    MatrixRef<Storage> x_train_;
    std::span<const Storage> y_train_;
    const JitterPolicy& jitter_;
    FitBuffers<P>& buffers_;
    bool enable_leaf_cache_{false};
    LeafCache<Storage> leaf_cache_;
};

// Computes r = y_f64 - B_f64 * w and ||B_f64||_inf using column blocks of width 32 without
// materializing the full n x n double matrix.
[[nodiscard]] std::expected<double, GprError>
block_reeval_residual(const CompiledKernel<double>& kernel_f64, MatrixRef<double> x_f64,
                      std::span<const double> y_f64, double diag_offset, std::span<const double> w,
                      std::span<double> out_r) {
    const std::size_t n = x_f64.rows();
    const std::size_t d = x_f64.cols();
    const std::size_t block_cols = std::min(refinement_column_block, n);

    std::vector<double> x_block(block_cols * d);
    std::vector<double> k_block(n * block_cols);
    std::vector<double> scratch_block(n * block_cols);
    auto nested = kernel_f64.make_nested_buffers(n, block_cols);
    std::vector<double> matvec(n, 0.0);
    std::vector<double> row_abs_sum(n, 0.0);

    for (std::size_t col_begin = 0; col_begin < n; col_begin += block_cols) {
        const std::size_t cols_cur = std::min(block_cols, n - col_begin);
        for (std::size_t feat = 0; feat < d; ++feat) {
            for (std::size_t c = 0; c < cols_cur; ++c) {
                x_block[(feat * cols_cur) + c] = x_f64(col_begin + c, feat);
            }
        }
        MatrixRef<double> xb_ref(x_block.data(), cols_cur, d, cols_cur);
        MatrixMut<double> kb_mut(k_block.data(), n, cols_cur, n);
        MatrixMut<double> sb_mut(scratch_block.data(), n, cols_cur, n);
        if (auto res = kernel_f64.apply_cross_points(x_f64, xb_ref, kb_mut, sb_mut, nested);
            !res.has_value()) {
            return std::unexpected(std::move(res).error());
        }
        for (std::size_t c = 0; c < cols_cur; ++c) {
            const std::size_t global_col = col_begin + c;
            const double w_col = w[global_col];
            for (std::size_t row = 0; row < n; ++row) {
                double entry = k_block[(c * n) + row];
                if (row == global_col) {
                    entry += diag_offset;
                }
                matvec[row] += entry * w_col;
                row_abs_sum[row] += std::abs(entry);
            }
        }
    }

    for (std::size_t row = 0; row < n; ++row) {
        out_r[row] = y_f64[row] - matvec[row];
    }
    return inf_norm(row_abs_sum);
}

template <PrecisionPolicyConcept P> class GprRefineSystem final : public RefineSystem {
public:
    GprRefineSystem(const KernelSpec& spec, KernelExp kernel_exp, MatrixRef<double> x_f64,
                    std::span<const double> y_f64, MatrixRef<float> l_f32,
                    MatrixRef<float> a_f32_symm, double diag_offset)
        : kernel_f64_(CompiledKernel<double>::from_spec(spec)), x_f64_(x_f64), y_f64_(y_f64),
          l_f32_(l_f32), a_f32_(a_f32_symm), diag_offset_(diag_offset),
          r_f32_buf_(x_f64.rows(), 0.0F) {
        kernel_f64_.set_kernel_exp(kernel_exp);
    }

    [[nodiscard]] std::span<const double> rhs() const noexcept override {
        return y_f64_;
    }

    [[nodiscard]] std::expected<double, GprError> residual(std::span<const double> w,
                                                           std::span<double> r) override {
        if constexpr (std::same_as<typename P::Residual, PromoteStorage>) {
            return promoted_residual(a_f32_, w, y_f64_, r);
        } else {
            return block_reeval_residual(kernel_f64_, x_f64_, y_f64_, diag_offset_, w, r);
        }
    }

    [[nodiscard]] std::expected<void, GprError> correct(std::span<const double> r,
                                                        std::span<double> w) override {
        const std::size_t n = r.size();
        for (std::size_t i = 0; i < n; ++i) {
            r_f32_buf_[i] = static_cast<float>(r[i]);
        }
        if (auto solve = linalg::solve_llt_in_place(l_f32_, std::span<float>(r_f32_buf_));
            !solve.has_value()) {
            return solve;
        }
        for (std::size_t i = 0; i < n; ++i) {
            w[i] += static_cast<double>(r_f32_buf_[i]);
        }
        return {};
    }

    [[nodiscard]] std::expected<bool, GprError> accept(std::span<const double> w,
                                                       double tol) override {
        if constexpr (std::same_as<typename P::Residual, ReevaluateKernel>) {
            return true;
        } else {
            std::vector<double> r_true(y_f64_.size());
            auto b_inf =
                block_reeval_residual(kernel_f64_, x_f64_, y_f64_, diag_offset_, w, r_true);
            if (!b_inf.has_value()) {
                return std::unexpected(std::move(b_inf).error());
            }
            return meets_relative_residual(*b_inf, w, y_f64_, r_true, tol);
        }
    }

    [[nodiscard]] std::expected<void, GprError> fallback(std::span<double> w) override {
        const std::size_t n = x_f64_.rows();
        std::vector<double> a_f64(n * n);
        std::vector<double> scratch_f64(n * n);
        auto nested = kernel_f64_.make_nested_buffers(n, n);
        MatrixMut<double> a_mut(a_f64.data(), n, n, n);
        MatrixMut<double> sc_mut(scratch_f64.data(), n, n, n);

        if (auto ap = kernel_f64_.apply_points(x_f64_, a_mut, Triangle::Lower, sc_mut, nested);
            !ap.has_value()) {
            return ap;
        }
        if (auto dg = linalg::add_to_diag(a_mut, diag_offset_); !dg.has_value()) {
            return dg;
        }
        if (auto ch = linalg::cholesky_lower_in_place(a_mut, CholeskyStage::Fit); !ch.has_value()) {
            return ch;
        }
        std::ranges::copy(y_f64_, w.begin());
        return linalg::solve_llt_in_place(a_mut.as_ref(), w);
    }

private:
    CompiledKernel<double> kernel_f64_;
    MatrixRef<double> x_f64_;
    std::span<const double> y_f64_;
    MatrixRef<float> l_f32_;
    MatrixRef<float> a_f32_;
    double diag_offset_{0.0};
    std::vector<float> r_f32_buf_;
};

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError> run_mixed_precision_refinement(
    const KernelSpec& spec, KernelExp kernel_exp, const CompiledKernel<float>& kernel_f32,
    MatrixRef<double> x_f64, MatrixRef<float> x_f32, std::span<const double> y_f64,
    FitBuffers<P>& buffers, double diag_offset, std::vector<double>& out_alpha,
    std::optional<RefinementSummary>& out_summary) {
    const std::size_t n = x_f64.rows();
    out_alpha.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out_alpha[i] = static_cast<double>(buffers.core.rhs[i]);
    }

    MatrixMut<float> a_f32_mut = buffers.core.exp_mut();
    if constexpr (std::same_as<typename P::Residual, PromoteStorage>) {
        // Reconstruct full symmetric A_f32 = K_f32 + (sigma_n^2 + j) I into exp_buf.
        if (auto ap =
                kernel_f32.apply_points(x_f32, a_f32_mut, Triangle::Lower,
                                        buffers.core.kernel_scratch_mut(), buffers.core.nested);
            !ap.has_value()) {
            // Note: kernel_scratch might be empty if ensure_kernel_scratch was not called yet.
            return ap;
        }
        if (auto dg = linalg::add_to_diag(a_f32_mut, static_cast<float>(diag_offset));
            !dg.has_value()) {
            return dg;
        }
        symmetrize_lower(a_f32_mut);
    }

    GprRefineSystem<P> sys(spec, kernel_exp, x_f64, y_f64, buffers.core.k_ref(), a_f32_mut.as_ref(),
                           diag_offset);
    std::vector<double> res_scratch(n);
    auto summary = refine_in_place(sys, out_alpha, res_scratch);
    if (!summary.has_value()) {
        return std::unexpected(std::move(summary).error());
    }
    out_summary = *summary;
    return {};
}

[[nodiscard]] std::expected<void, GprError> copy_double_matrix_col_major(MatrixRef<double> src,
                                                                         std::vector<double>& dst) {
    if (src.is_empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (src.data() == nullptr) {
        return std::unexpected(GprError::empty_input());
    }
    auto nm = checked_mul(src.rows(), src.cols());
    if (!nm.has_value()) {
        return std::unexpected(std::move(nm).error());
    }
    dst.resize(*nm);
    const std::size_t rows = src.rows();
    const std::size_t cols = src.cols();
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = 0; row < rows; ++row) {
            const double val = src(row, col);
            if (!std::isfinite(val)) {
                return std::unexpected(GprError::non_finite_input());
            }
            dst[(col * rows) + row] = val;
        }
    }
    return {};
}

} // namespace

template <PrecisionPolicyConcept P>
std::expected<FittedPayload<P>, GprError>
train_or_factor_payload(const TrainConfig<P>& cfg, MatrixRef<double> x, std::span<const double> y,
                        OptimizeCallback optimize_cb, void* optimize_ctx) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    if (x.is_empty() || y.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t n = x.rows();
    const std::size_t d = x.cols();
    if (y.size() != n) {
        return std::unexpected(
            GprError::length_mismatch("target length must match training row count"));
    }
    std::vector<double> x_f64;
    if (auto cp = copy_double_matrix_col_major(x, x_f64); !cp.has_value()) {
        return std::unexpected(std::move(cp).error());
    }
    for (const double val : y) {
        if (!std::isfinite(val)) {
            return std::unexpected(GprError::non_finite_input());
        }
    }

    auto fitted_in = cfg.input_transform.fit(x_f64, n, d);
    if (!fitted_in.has_value()) {
        return std::unexpected(std::move(fitted_in).error());
    }
    if (auto ap = (*fitted_in)->apply(x_f64, n, d); !ap.has_value()) {
        return std::unexpected(std::move(ap).error());
    }

    auto fitted_tgt = cfg.target_transform.fit(y);
    if (!fitted_tgt.has_value()) {
        return std::unexpected(std::move(fitted_tgt).error());
    }
    std::vector<double> y_f64(y.begin(), y.end());
    if (auto tf = (*fitted_tgt)->transform(y_f64); !tf.has_value()) {
        return std::unexpected(std::move(tf).error());
    }

    std::vector<Storage> x_storage(x_f64.size());
    for (std::size_t i = 0; i < x_f64.size(); ++i) {
        x_storage[i] = static_cast<Storage>(x_f64[i]);
    }
    std::vector<Storage> y_storage(n);
    std::vector<Refine> y_refine(n);
    for (std::size_t i = 0; i < n; ++i) {
        y_storage[i] = static_cast<Storage>(y_f64[i]);
        y_refine[i] = static_cast<Refine>(y_f64[i]);
    }

    auto compiled = std::make_shared<CompiledKernel<Storage>>(
        CompiledKernel<Storage>::from_spec(cfg.kernel_spec));
    compiled->set_kernel_exp(cfg.kernel_exp);

    auto buffers = FitBuffers<P>::create(n, cfg.fit_policy);
    if (!buffers.has_value()) {
        return std::unexpected(std::move(buffers).error());
    }
    if constexpr (P::refines_in_f64) {
        if constexpr (std::same_as<typename P::Residual, PromoteStorage>) {
            if (auto ks = buffers->core.ensure_kernel_scratch(); !ks.has_value()) {
                return std::unexpected(std::move(ks).error());
            }
        }
    }

    const MatrixRef<Storage> x_train_ref(x_storage.data(), n, d, n);
    GaussianLikelihood likelihood = cfg.likelihood;
    GprObjective<P> obj(*compiled, likelihood, x_train_ref, y_storage, cfg.jitter, *buffers,
                        cfg.use_change_indices);

    const std::size_t total_params = obj.dim();
    std::vector<double> init_params(total_params);
    const std::size_t nk = compiled->num_params();
    if (auto gp = compiled->get_params(std::span<double>(init_params.data(), nk));
        !gp.has_value()) {
        return std::unexpected(std::move(gp).error());
    }
    if (auto lp = likelihood.get_params(std::span<double>(init_params.data() + nk, 1U));
        !lp.has_value()) {
        return std::unexpected(std::move(lp).error());
    }

    optimizer::OptimizerResult opt_res;
    if (cfg.optimize) {
        auto r = optimize_cb(optimize_ctx, obj, init_params);
        if (!r.has_value()) {
            return std::unexpected(std::move(r).error());
        }
        opt_res = std::move(*r);
    } else {
        auto v = obj.value(init_params);
        if (!v.has_value()) {
            return std::unexpected(std::move(v).error());
        }
        opt_res = optimizer::OptimizerResult{
            .params = init_params,
            .value = *v,
            .iterations = 0U,
            .converged = true,
        };
    }

    // Ensure kernel, likelihood, L, alpha, and MLL match opt_res.params exactly.
    if (auto sp = compiled->set_params(std::span<const double>(opt_res.params.data(), nk));
        !sp.has_value()) {
        return std::unexpected(std::move(sp).error());
    }
    if (auto lp = likelihood.set_params(std::span<const double>(opt_res.params.data() + nk, 1U));
        !lp.has_value()) {
        return std::unexpected(std::move(lp).error());
    }
    auto final_mll =
        factor_and_mll(*compiled, likelihood, x_train_ref, y_storage, cfg.jitter, *buffers, 0U);
    if (!final_mll.has_value()) {
        return std::unexpected(std::move(final_mll).error());
    }

    KernelSpec updated_spec = cfg.kernel_spec;
    if (auto sp = updated_spec.set_params(std::span<const double>(opt_res.params.data(), nk));
        !sp.has_value()) {
        return std::unexpected(std::move(sp).error());
    }

    std::vector<Storage> l_copy(n * n);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            l_copy[(col * n) + row] = buffers->core.k_ref()(row, col);
        }
    }

    std::vector<Refine> alpha_out(n);
    std::optional<RefinementSummary> ref_summary;

    if constexpr (P::refines_in_f64) {
        const double diag_offset = likelihood.noise_variance() + buffers->core.factor_jitter;
        const MatrixRef<double> x_f64_ref(x_f64.data(), n, d, n);
        if (auto ref_res = run_mixed_precision_refinement<P>(
                updated_spec, cfg.kernel_exp, *compiled, x_f64_ref, x_train_ref, y_f64, *buffers,
                diag_offset, alpha_out, ref_summary);
            !ref_res.has_value()) {
            return std::unexpected(std::move(ref_res).error());
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            alpha_out[i] = static_cast<Refine>(buffers->core.rhs[i]);
        }
    }

    return FittedPayload<P>{
        .kernel_spec = std::move(updated_spec),
        .likelihood = likelihood,
        .compiled_kernel = std::move(compiled),
        .input_transform = std::move(*fitted_in),
        .target_transform = std::move(*fitted_tgt),
        .x_train = std::move(x_storage),
        .y_transformed = std::move(y_refine),
        .l_factor = std::move(l_copy),
        .alpha = std::move(alpha_out),
        .n_train = n,
        .n_features = d,
        .factor_jitter = buffers->core.factor_jitter,
        .log_marginal_likelihood = *final_mll,
        .refinement_summary = ref_summary,
        .optimizer_result = std::move(opt_res),
    };
}

namespace {

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<MatrixRef<typename P::Storage>, GprError>
prepare_query_matrix(const FittedPayload<P>& payload, MatrixRef<double> xs,
                     QueryWorkspace<P>& workspace) {
    using Storage = P::Storage;

    if (xs.is_empty() || xs.data() == nullptr) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t m = xs.rows();
    const std::size_t d = xs.cols();
    if (d != payload.n_features) {
        return std::unexpected(GprError::dimension_mismatch(d, payload.n_features));
    }
    if (auto ens = workspace.ensure(payload.n_train, m, d, payload.compiled_kernel->nested_depth());
        !ens.has_value()) {
        return std::unexpected(std::move(ens).error());
    }

    for (std::size_t col = 0; col < d; ++col) {
        for (std::size_t row = 0; row < m; ++row) {
            const double val = xs(row, col);
            if (!std::isfinite(val)) {
                return std::unexpected(GprError::non_finite_input());
            }
            workspace.query_xs[(col * m) + row] = val;
        }
    }

    if (auto ap = payload.input_transform->apply(workspace.query_xs, m, d); !ap.has_value()) {
        return std::unexpected(std::move(ap).error());
    }

    const std::size_t md = m * d;
    for (std::size_t i = 0; i < md; ++i) {
        workspace.query_x[i] = static_cast<Storage>(workspace.query_xs[i]);
    }
    return MatrixRef<Storage>(workspace.query_x.data(), m, d, m);
}

template <typename T>
[[nodiscard]] std::expected<void, GprError>
apply_target_inverse_prediction(const transform::FittedTargetTransform& target_tf,
                                Prediction<T>& pred) {
    if constexpr (std::same_as<T, double>) {
        if (auto m_res = target_tf.inverse_transform_mean(pred.mean); !m_res.has_value()) {
            return m_res;
        }
        return target_tf.inverse_transform_variance(pred.variance);
    } else {
        std::vector<double> mean_d(pred.mean.begin(), pred.mean.end());
        std::vector<double> var_d(pred.variance.begin(), pred.variance.end());
        if (auto m_res = target_tf.inverse_transform_mean(mean_d); !m_res.has_value()) {
            return m_res;
        }
        if (auto v_res = target_tf.inverse_transform_variance(var_d); !v_res.has_value()) {
            return v_res;
        }
        for (std::size_t i = 0; i < pred.mean.size(); ++i) {
            pred.mean[i] = static_cast<T>(mean_d[i]);
            pred.variance[i] = static_cast<T>(var_d[i]);
        }
        return {};
    }
}

} // namespace

template <PrecisionPolicyConcept P>
std::expected<void, GprError>
fitted_predict_into(const FittedPayload<P>& payload, MatrixRef<double> xs,
                    const PredictOptions& options, QueryWorkspace<P>& workspace,
                    Prediction<typename P::Refine>& out) {
    using Storage = P::Storage;
    auto xq_ref = prepare_query_matrix(payload, xs, workspace);
    if (!xq_ref.has_value()) {
        return std::unexpected(std::move(xq_ref).error());
    }
    const MatrixRef<Storage> x_train_ref(payload.x_train.data(), payload.n_train,
                                         payload.n_features, payload.n_train);
    const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train,
                                   payload.n_train);
    if (auto res = predict_into<P>(*payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref,
                                   payload.alpha, *xq_ref, options, workspace, out);
        !res.has_value()) {
        return res;
    }
    return apply_target_inverse_prediction(*payload.target_transform, out);
}

template <PrecisionPolicyConcept P>
std::expected<PredictiveCovariance<typename P::Refine>, GprError>
fitted_predict_covariance(const FittedPayload<P>& payload, MatrixRef<double> xs,
                          const PredictOptions& options, QueryWorkspace<P>& workspace) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    auto xq_ref = prepare_query_matrix(payload, xs, workspace);
    if (!xq_ref.has_value()) {
        return std::unexpected(std::move(xq_ref).error());
    }
    const MatrixRef<Storage> x_train_ref(payload.x_train.data(), payload.n_train,
                                         payload.n_features, payload.n_train);
    const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train,
                                   payload.n_train);
    auto cov = predict_covariance<P>(*payload.compiled_kernel, payload.likelihood, x_train_ref,
                                     l_ref, payload.alpha, *xq_ref, options, workspace);
    if (!cov.has_value()) {
        return cov;
    }

    if constexpr (std::same_as<Refine, double>) {
        if (auto mr = payload.target_transform->inverse_transform_mean(cov->mean);
            !mr.has_value()) {
            return std::unexpected(std::move(mr).error());
        }
        if (auto cr = payload.target_transform->inverse_transform_covariance(cov->covariance);
            !cr.has_value()) {
            return std::unexpected(std::move(cr).error());
        }
    } else {
        std::vector<double> mean_d(cov->mean.begin(), cov->mean.end());
        std::vector<double> cov_d(cov->covariance.begin(), cov->covariance.end());
        if (auto mr = payload.target_transform->inverse_transform_mean(mean_d); !mr.has_value()) {
            return std::unexpected(std::move(mr).error());
        }
        if (auto cr = payload.target_transform->inverse_transform_covariance(cov_d);
            !cr.has_value()) {
            return std::unexpected(std::move(cr).error());
        }
        for (std::size_t i = 0; i < mean_d.size(); ++i) {
            cov->mean[i] = static_cast<Refine>(mean_d[i]);
        }
        for (std::size_t i = 0; i < cov_d.size(); ++i) {
            cov->covariance[i] = static_cast<Refine>(cov_d[i]);
        }
    }
    return cov;
}

template <PrecisionPolicyConcept P>
std::expected<std::vector<typename P::Refine>, GprError>
fitted_sample(const FittedPayload<P>& payload, MatrixRef<double> xs, std::size_t n_draws,
              std::uint64_t seed, const PredictOptions& options, QueryWorkspace<P>& workspace,
              std::optional<JitterPolicy> jitter) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    auto xq_ref = prepare_query_matrix(payload, xs, workspace);
    if (!xq_ref.has_value()) {
        return std::unexpected(std::move(xq_ref).error());
    }
    const MatrixRef<Storage> x_train_ref(payload.x_train.data(), payload.n_train,
                                         payload.n_features, payload.n_train);
    const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train,
                                   payload.n_train);
    auto samples = sample<P>(*payload.compiled_kernel, payload.likelihood, x_train_ref, l_ref,
                             payload.alpha, *xq_ref, n_draws, seed, options, workspace, jitter);
    if (!samples.has_value()) {
        return samples;
    }
    const std::size_t m = xs.rows();
    if constexpr (std::same_as<Refine, double>) {
        for (std::size_t draw = 0; draw < n_draws; ++draw) {
            const std::span<double> slice(samples->data() + (draw * m), m);
            if (auto mr = payload.target_transform->inverse_transform_mean(slice);
                !mr.has_value()) {
                return std::unexpected(std::move(mr).error());
            }
        }
    } else {
        std::vector<double> buf(m);
        for (std::size_t draw = 0; draw < n_draws; ++draw) {
            Refine* ptr = samples->data() + (draw * m);
            for (std::size_t i = 0; i < m; ++i) {
                buf[i] = static_cast<double>(ptr[i]);
            }
            if (auto mr = payload.target_transform->inverse_transform_mean(buf); !mr.has_value()) {
                return std::unexpected(std::move(mr).error());
            }
            for (std::size_t i = 0; i < m; ++i) {
                ptr[i] = static_cast<Refine>(buf[i]);
            }
        }
    }
    return samples;
}

template <PrecisionPolicyConcept P>
std::expected<Prediction<typename P::Refine>, GprError>
fitted_loo_predict(const FittedPayload<P>& payload, const PredictOptions& options) {
    using Storage = P::Storage;
    const MatrixRef<Storage> l_ref(payload.l_factor.data(), payload.n_train, payload.n_train,
                                   payload.n_train);
    auto loo =
        loo_predict<P>(payload.likelihood, l_ref, payload.alpha, payload.y_transformed, options);
    if (!loo.has_value()) {
        return loo;
    }
    if (auto inv = apply_target_inverse_prediction(*payload.target_transform, *loo);
        !inv.has_value()) {
        return std::unexpected(std::move(inv).error());
    }
    return loo;
}

template std::expected<FittedPayload<DoublePrecision>, GprError>
train_or_factor_payload<DoublePrecision>(const TrainConfig<DoublePrecision>&, MatrixRef<double>,
                                         std::span<const double>, OptimizeCallback, void*);
template std::expected<FittedPayload<SinglePrecision>, GprError>
train_or_factor_payload<SinglePrecision>(const TrainConfig<SinglePrecision>&, MatrixRef<double>,
                                         std::span<const double>, OptimizeCallback, void*);
template std::expected<FittedPayload<MixedPrecision<PromoteStorage>>, GprError>
train_or_factor_payload<MixedPrecision<PromoteStorage>>(
    const TrainConfig<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, std::span<const double>,
    OptimizeCallback, void*);
template std::expected<FittedPayload<MixedPrecision<ReevaluateKernel>>, GprError>
train_or_factor_payload<MixedPrecision<ReevaluateKernel>>(
    const TrainConfig<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>,
    std::span<const double>, OptimizeCallback, void*);

template std::expected<void, GprError>
fitted_predict_into<DoublePrecision>(const FittedPayload<DoublePrecision>&, MatrixRef<double>,
                                     const PredictOptions&, QueryWorkspace<DoublePrecision>&,
                                     Prediction<double>&);
template std::expected<void, GprError>
fitted_predict_into<SinglePrecision>(const FittedPayload<SinglePrecision>&, MatrixRef<double>,
                                     const PredictOptions&, QueryWorkspace<SinglePrecision>&,
                                     Prediction<float>&);
template std::expected<void, GprError> fitted_predict_into<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, Prediction<double>&);
template std::expected<void, GprError> fitted_predict_into<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>,
    const PredictOptions&, QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, Prediction<double>&);

template std::expected<PredictiveCovariance<double>, GprError>
fitted_predict_covariance<DoublePrecision>(const FittedPayload<DoublePrecision>&, MatrixRef<double>,
                                           const PredictOptions&, QueryWorkspace<DoublePrecision>&);
template std::expected<PredictiveCovariance<float>, GprError>
fitted_predict_covariance<SinglePrecision>(const FittedPayload<SinglePrecision>&, MatrixRef<double>,
                                           const PredictOptions&, QueryWorkspace<SinglePrecision>&);
template std::expected<PredictiveCovariance<double>, GprError>
fitted_predict_covariance<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&);
template std::expected<PredictiveCovariance<double>, GprError>
fitted_predict_covariance<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>,
    const PredictOptions&, QueryWorkspace<MixedPrecision<ReevaluateKernel>>&);

template std::expected<std::vector<double>, GprError>
fitted_sample<DoublePrecision>(const FittedPayload<DoublePrecision>&, MatrixRef<double>,
                               std::size_t, std::uint64_t, const PredictOptions&,
                               QueryWorkspace<DoublePrecision>&, std::optional<JitterPolicy>);
template std::expected<std::vector<float>, GprError>
fitted_sample<SinglePrecision>(const FittedPayload<SinglePrecision>&, MatrixRef<double>,
                               std::size_t, std::uint64_t, const PredictOptions&,
                               QueryWorkspace<SinglePrecision>&, std::optional<JitterPolicy>);
template std::expected<std::vector<double>, GprError> fitted_sample<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, MatrixRef<double>, std::size_t,
    std::uint64_t, const PredictOptions&, QueryWorkspace<MixedPrecision<PromoteStorage>>&,
    std::optional<JitterPolicy>);
template std::expected<std::vector<double>, GprError>
fitted_sample<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, MatrixRef<double>, std::size_t,
    std::uint64_t, const PredictOptions&, QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
    std::optional<JitterPolicy>);

template std::expected<Prediction<double>, GprError>
fitted_loo_predict<DoublePrecision>(const FittedPayload<DoublePrecision>&, const PredictOptions&);
template std::expected<Prediction<float>, GprError>
fitted_loo_predict<SinglePrecision>(const FittedPayload<SinglePrecision>&, const PredictOptions&);
template std::expected<Prediction<double>, GprError>
fitted_loo_predict<MixedPrecision<PromoteStorage>>(
    const FittedPayload<MixedPrecision<PromoteStorage>>&, const PredictOptions&);
template std::expected<Prediction<double>, GprError>
fitted_loo_predict<MixedPrecision<ReevaluateKernel>>(
    const FittedPayload<MixedPrecision<ReevaluateKernel>>&, const PredictOptions&);

} // namespace cxxgpr::detail
