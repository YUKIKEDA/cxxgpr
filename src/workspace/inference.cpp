// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/workspace/inference.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/linalg/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cxxgpr {
namespace {

[[nodiscard]] JitterPolicy default_sample_jitter() {
    auto policy = JitterPolicy::adaptive(1e-12, 10.0, 10U, 1e-2);
    return *policy;
}

template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError> validate_predict_inputs(
    MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
    std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query) {
    if (x_train.is_empty() || x_query.is_empty() || l_factor.is_empty() || alpha.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t n = x_train.rows();
    const std::size_t d = x_train.cols();
    if (l_factor.rows() != n || l_factor.cols() != n) {
        return std::unexpected(
            GprError::length_mismatch("Cholesky factor dimension must match training samples"));
    }
    if (alpha.size() != n) {
        return std::unexpected(
            GprError::length_mismatch("alpha length must match training sample count"));
    }
    if (x_query.cols() != d) {
        return std::unexpected(GprError::dimension_mismatch(x_query.cols(), d));
    }
    for (const auto val : alpha) {
        if (!std::isfinite(val)) {
            return std::unexpected(GprError::non_finite_input());
        }
    }
    return {};
}

/// Evaluates K_* = k(X, X_*), computes posterior mean mu_* = K_*^T alpha into out_mean, and
/// overwrites workspace.query_k_star with V = L^-1 K_*.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError> compute_mean_and_v_into(
    const CompiledKernel<typename P::Storage>& kernel, MatrixRef<typename P::Storage> x_train,
    MatrixRef<typename P::Storage> l_factor, std::span<const typename P::Refine> alpha,
    MatrixRef<typename P::Storage> x_query, QueryWorkspace<P>& workspace,
    std::vector<typename P::Refine>& out_mean) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    if (auto valid = validate_predict_inputs<P>(x_train, l_factor, alpha, x_query);
        !valid.has_value()) {
        return valid;
    }

    const std::size_t n = x_train.rows();
    const std::size_t m = x_query.rows();
    const std::size_t d = x_train.cols();

    if (auto ens = workspace.ensure(n, m, d, kernel.nested_depth()); !ens.has_value()) {
        return ens;
    }

    MatrixMut<Storage> k_star_mut(workspace.query_k_star.data(), n, m, n);
    MatrixMut<Storage> scratch_mut(workspace.query_scratch.data(), n, m, n);
    if (auto cross = kernel.apply_cross_points(x_train, x_query, k_star_mut, scratch_mut,
                                               workspace.query_nested);
        !cross.has_value()) {
        return cross;
    }

    out_mean.resize(m);
    const std::size_t n_local = n;
    const std::size_t m_local = m;
    const Storage* k_star_ptr = workspace.query_k_star.data();
    const Refine* alpha_ptr = alpha.data();
    Refine* mean_ptr = out_mean.data();

    const auto m_signed = static_cast<std::ptrdiff_t>(m_local);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m_local >= 16U)
#endif
    for (std::ptrdiff_t col_idx = 0; col_idx < m_signed; ++col_idx) {
        const auto j = static_cast<std::size_t>(col_idx);
        const Storage* col_ptr = k_star_ptr + j * n_local;
        Refine acc{0};
        for (std::size_t i = 0; i < n_local; ++i) {
            acc += static_cast<Refine>(col_ptr[i]) * alpha_ptr[i];
        }
        mean_ptr[j] = acc;
    }

    if (auto solve = linalg::solve_lower_in_place(l_factor, k_star_mut); !solve.has_value()) {
        return solve;
    }

    return {};
}

} // namespace

template <PrecisionPolicyConcept P>
std::expected<void, GprError>
predict_into(const CompiledKernel<typename P::Storage>& kernel,
             const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> x_train,
             MatrixRef<typename P::Storage> l_factor, std::span<const typename P::Refine> alpha,
             MatrixRef<typename P::Storage> x_query, const PredictOptions& options,
             QueryWorkspace<P>& workspace, Prediction<typename P::Refine>& out) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    if (auto mv = compute_mean_and_v_into<P>(kernel, x_train, l_factor, alpha, x_query, workspace,
                                             out.mean);
        !mv.has_value()) {
        return mv;
    }

    const std::size_t n = x_train.rows();
    const std::size_t m = x_query.rows();
    std::span<Storage> kss_span(workspace.query_kss.data(), m);
    if (auto diag = kernel.fill_diag(x_query, kss_span, workspace.query_diag_scratch);
        !diag.has_value()) {
        return diag;
    }

    out.variance.resize(m);
    out.variance_kind = options.variance_kind;

    const Refine noise_var = (options.variance_kind == VarianceKind::Observation)
                                 ? static_cast<Refine>(likelihood.noise_variance())
                                 : Refine{0};
    const Storage* v_ptr = workspace.query_k_star.data();
    const Storage* kss_ptr = workspace.query_kss.data();
    Refine* var_ptr = out.variance.data();

    const auto m_signed = static_cast<std::ptrdiff_t>(m);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m >= 16U)
#endif
    for (std::ptrdiff_t col_idx = 0; col_idx < m_signed; ++col_idx) {
        const auto j = static_cast<std::size_t>(col_idx);
        const Storage* v_col = v_ptr + j * n;
        Refine sq_norm{0};
        for (std::size_t i = 0; i < n; ++i) {
            const auto val = static_cast<Refine>(v_col[i]);
            sq_norm += val * val;
        }
        const auto prior_var = static_cast<Refine>(kss_ptr[j]);
        const Refine latent_var = std::max(Refine{0}, prior_var - sq_norm);
        var_ptr[j] = latent_var + noise_var;
    }

    return {};
}

template <PrecisionPolicyConcept P>
std::expected<Prediction<typename P::Refine>, GprError>
predict(const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options, QueryWorkspace<P>& workspace) {
    Prediction<typename P::Refine> result;
    if (auto res = predict_into<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, options,
                                   workspace, result);
        !res.has_value()) {
        return std::unexpected(std::move(res).error());
    }
    return result;
}

template <PrecisionPolicyConcept P>
std::expected<void, GprError> predict_covariance_into(
    const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
    MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
    std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
    const PredictOptions& options, QueryWorkspace<P>& workspace,
    PredictiveCovariance<typename P::Refine>& out) {
    using Storage = P::Storage;
    using Refine = P::Refine;

    if (auto mv = compute_mean_and_v_into<P>(kernel, x_train, l_factor, alpha, x_query, workspace,
                                             out.mean);
        !mv.has_value()) {
        return mv;
    }

    const std::size_t n = x_train.rows();
    const std::size_t m = x_query.rows();
    auto mm = checked_mul(m, m);
    if (!mm.has_value()) {
        return std::unexpected(std::move(mm).error());
    }

    std::vector<Storage> kss(*mm);
    std::vector<Storage> kss_scratch(*mm);
    auto nested_kss = kernel.make_nested_buffers(m, m);
    MatrixMut<Storage> kss_mut(kss.data(), m, m, m);
    MatrixMut<Storage> kss_scratch_mut(kss_scratch.data(), m, m, m);
    if (auto ap =
            kernel.apply_points(x_query, kss_mut, Triangle::Lower, kss_scratch_mut, nested_kss);
        !ap.has_value()) {
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
    for (std::size_t col = 0; col < m; ++col) {
        const Storage* v_col = v_ptr + col * n;
        for (std::size_t row = col; row < m; ++row) {
            const Storage* v_row = v_ptr + row * n;
            Refine dot_val{0};
            for (std::size_t r = 0; r < n; ++r) {
                dot_val += static_cast<Refine>(v_row[r]) * static_cast<Refine>(v_col[r]);
            }
            Refine entry = static_cast<Refine>(kss[(col * m) + row]) - dot_val;
            if (row == col) {
                entry += noise_var;
            }
            cov_ptr[(col * m) + row] = entry;
        }
    }

    // Mirror strictly lower triangle into strictly upper triangle for symmetric output.
    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t row = 0; row < col; ++row) {
            cov_ptr[(col * m) + row] = cov_ptr[(row * m) + col];
        }
    }

    return {};
}

template <PrecisionPolicyConcept P>
std::expected<PredictiveCovariance<typename P::Refine>, GprError> predict_covariance(
    const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
    MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
    std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
    const PredictOptions& options, QueryWorkspace<P>& workspace) {
    PredictiveCovariance<typename P::Refine> result;
    if (auto res = predict_covariance_into<P>(kernel, likelihood, x_train, l_factor, alpha, x_query,
                                              options, workspace, result);
        !res.has_value()) {
        return std::unexpected(std::move(res).error());
    }
    return result;
}

template <PrecisionPolicyConcept P>
std::expected<std::vector<typename P::Refine>, GprError>
sample(const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
       MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
       std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
       std::size_t n_draws, std::uint64_t seed, const PredictOptions& options,
       QueryWorkspace<P>& workspace, std::optional<JitterPolicy> jitter) {
    using Refine = P::Refine;

    auto cov_res = predict_covariance<P>(kernel, likelihood, x_train, l_factor, alpha, x_query,
                                         options, workspace);
    if (!cov_res.has_value()) {
        return std::unexpected(std::move(cov_res).error());
    }
    if (n_draws == 0U) {
        return std::vector<Refine>{};
    }

    const std::size_t m = cov_res->n_points;
    auto total_len = checked_mul(n_draws, m);
    if (!total_len.has_value()) {
        return std::unexpected(std::move(total_len).error());
    }

    std::vector<Refine> backup(cov_res->covariance.size());
    MatrixMut<Refine> cov_mut(cov_res->covariance.data(), m, m, m);
    MatrixMut<Refine> backup_mut(backup.data(), m, m, m);
    const JitterPolicy eff_jitter = jitter.value_or(default_sample_jitter());
    if (auto chol = linalg::cholesky_lower_with_jitter(cov_mut, backup_mut, eff_jitter,
                                                       CholeskyStage::Predict);
        !chol.has_value()) {
        return std::unexpected(std::move(chol).error());
    }

    std::vector<Refine> samples(*total_len);
    std::vector<Refine> z(m);
    Xoshiro256PlusPlus rng(seed);
    for (std::size_t draw = 0; draw < n_draws; ++draw) {
        for (std::size_t i = 0; i < m; ++i) {
            z[i] = static_cast<Refine>(rng.unit_normal());
        }
        Refine* out_draw = samples.data() + draw * m;
        for (std::size_t row = 0; row < m; ++row) {
            Refine acc = cov_res->mean[row];
            for (std::size_t col = 0; col <= row; ++col) {
                acc += cov_mut(row, col) * z[col];
            }
            out_draw[row] = acc;
        }
    }

    return samples;
}

template <PrecisionPolicyConcept P>
std::expected<void, GprError>
loo_predict_into(const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> l_factor,
                 std::span<const typename P::Refine> alpha,
                 std::span<const typename P::Refine> y_train, const PredictOptions& options,
                 MatrixMut<typename P::Storage> scratch_n_by_n,
                 std::span<typename P::Storage> scratch_diag, Prediction<typename P::Refine>& out) {
    using Refine = P::Refine;

    if (l_factor.is_empty() || alpha.empty() || y_train.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (!l_factor.is_square()) {
        return std::unexpected(GprError::length_mismatch("Cholesky factor must be square"));
    }
    const std::size_t n = l_factor.rows();
    if (alpha.size() != n) {
        return std::unexpected(
            GprError::length_mismatch("alpha length must match factor dimension"));
    }
    if (y_train.size() != n) {
        return std::unexpected(
            GprError::length_mismatch("y_train length must match factor dimension"));
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(alpha[i]) || !std::isfinite(y_train[i])) {
            return std::unexpected(GprError::non_finite_input());
        }
    }

    if (auto inv_res = linalg::inv_diag_from_chol_l(l_factor, scratch_diag, scratch_n_by_n);
        !inv_res.has_value()) {
        return inv_res;
    }

    out.mean.resize(n);
    out.variance.resize(n);
    out.variance_kind = options.variance_kind;

    const auto noise_var = static_cast<Refine>(likelihood.noise_variance());
    for (std::size_t i = 0; i < n; ++i) {
        const auto q_ii = static_cast<Refine>(scratch_diag[i]);
        if (!std::isfinite(q_ii) || q_ii <= Refine{0}) {
            return std::unexpected(GprError::non_positive_definite_matrix());
        }
        const Refine inv_q = Refine{1} / q_ii;
        out.mean[i] = y_train[i] - (alpha[i] / q_ii);
        out.variance[i] = (options.variance_kind == VarianceKind::Observation)
                              ? inv_q
                              : std::max(Refine{0}, inv_q - noise_var);
    }

    return {};
}

template <PrecisionPolicyConcept P>
std::expected<Prediction<typename P::Refine>, GprError>
loo_predict(const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> l_factor,
            std::span<const typename P::Refine> alpha, std::span<const typename P::Refine> y_train,
            const PredictOptions& options) {
    using Storage = P::Storage;

    if (l_factor.is_empty()) {
        return std::unexpected(GprError::empty_input());
    }
    const std::size_t n = l_factor.rows();
    auto nn = checked_mul(n, n);
    if (!nn.has_value()) {
        return std::unexpected(std::move(nn).error());
    }
    std::vector<Storage> scratch_mat(*nn);
    std::vector<Storage> scratch_diag(n);
    MatrixMut<Storage> scratch_mut(scratch_mat.data(), n, n, n);

    Prediction<typename P::Refine> result;
    if (auto res = loo_predict_into<P>(likelihood, l_factor, alpha, y_train, options, scratch_mut,
                                       scratch_diag, result);
        !res.has_value()) {
        return std::unexpected(std::move(res).error());
    }
    return result;
}

template std::expected<void, GprError>
predict_into<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                              MatrixRef<double>, MatrixRef<double>, std::span<const double>,
                              MatrixRef<double>, const PredictOptions&,
                              QueryWorkspace<DoublePrecision>&, Prediction<double>&);
template std::expected<void, GprError>
predict_into<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&,
                              MatrixRef<float>, MatrixRef<float>, std::span<const float>,
                              MatrixRef<float>, const PredictOptions&,
                              QueryWorkspace<SinglePrecision>&, Prediction<float>&);
template std::expected<void, GprError> predict_into<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, Prediction<double>&);
template std::expected<void, GprError> predict_into<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, Prediction<double>&);

template std::expected<Prediction<double>, GprError>
predict<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                         MatrixRef<double>, MatrixRef<double>, std::span<const double>,
                         MatrixRef<double>, const PredictOptions&,
                         QueryWorkspace<DoublePrecision>&);
template std::expected<Prediction<float>, GprError>
predict<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
                         MatrixRef<float>, std::span<const float>, MatrixRef<float>,
                         const PredictOptions&, QueryWorkspace<SinglePrecision>&);
template std::expected<Prediction<double>, GprError>
predict<MixedPrecision<PromoteStorage>>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                        MatrixRef<float>, MatrixRef<float>, std::span<const double>,
                                        MatrixRef<float>, const PredictOptions&,
                                        QueryWorkspace<MixedPrecision<PromoteStorage>>&);
template std::expected<Prediction<double>, GprError> predict<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&);

template std::expected<void, GprError> predict_covariance_into<DoublePrecision>(
    const CompiledKernel<double>&, const GaussianLikelihood&, MatrixRef<double>, MatrixRef<double>,
    std::span<const double>, MatrixRef<double>, const PredictOptions&,
    QueryWorkspace<DoublePrecision>&, PredictiveCovariance<double>&);
template std::expected<void, GprError> predict_covariance_into<SinglePrecision>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const float>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<SinglePrecision>&, PredictiveCovariance<float>&);
template std::expected<void, GprError> predict_covariance_into<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, PredictiveCovariance<double>&);
template std::expected<void, GprError> predict_covariance_into<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, PredictiveCovariance<double>&);

template std::expected<PredictiveCovariance<double>, GprError>
predict_covariance<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                                    MatrixRef<double>, MatrixRef<double>, std::span<const double>,
                                    MatrixRef<double>, const PredictOptions&,
                                    QueryWorkspace<DoublePrecision>&);
template std::expected<PredictiveCovariance<float>, GprError>
predict_covariance<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                    MatrixRef<float>, MatrixRef<float>, std::span<const float>,
                                    MatrixRef<float>, const PredictOptions&,
                                    QueryWorkspace<SinglePrecision>&);
template std::expected<PredictiveCovariance<double>, GprError>
predict_covariance<MixedPrecision<PromoteStorage>>(const CompiledKernel<float>&,
                                                   const GaussianLikelihood&, MatrixRef<float>,
                                                   MatrixRef<float>, std::span<const double>,
                                                   MatrixRef<float>, const PredictOptions&,
                                                   QueryWorkspace<MixedPrecision<PromoteStorage>>&);
template std::expected<PredictiveCovariance<double>, GprError>
predict_covariance<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&);

template std::expected<std::vector<double>, GprError>
sample<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&, MatrixRef<double>,
                        MatrixRef<double>, std::span<const double>, MatrixRef<double>, std::size_t,
                        std::uint64_t, const PredictOptions&, QueryWorkspace<DoublePrecision>&,
                        std::optional<JitterPolicy>);
template std::expected<std::vector<float>, GprError>
sample<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
                        MatrixRef<float>, std::span<const float>, MatrixRef<float>, std::size_t,
                        std::uint64_t, const PredictOptions&, QueryWorkspace<SinglePrecision>&,
                        std::optional<JitterPolicy>);
template std::expected<std::vector<double>, GprError> sample<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, std::size_t, std::uint64_t, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, std::optional<JitterPolicy>);
template std::expected<std::vector<double>, GprError> sample<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, std::size_t, std::uint64_t, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, std::optional<JitterPolicy>);

template std::expected<void, GprError> loo_predict_into<DoublePrecision>(
    const GaussianLikelihood&, MatrixRef<double>, std::span<const double>, std::span<const double>,
    const PredictOptions&, MatrixMut<double>, std::span<double>, Prediction<double>&);
template std::expected<void, GprError> loo_predict_into<SinglePrecision>(
    const GaussianLikelihood&, MatrixRef<float>, std::span<const float>, std::span<const float>,
    const PredictOptions&, MatrixMut<float>, std::span<float>, Prediction<float>&);
template std::expected<void, GprError> loo_predict_into<MixedPrecision<PromoteStorage>>(
    const GaussianLikelihood&, MatrixRef<float>, std::span<const double>, std::span<const double>,
    const PredictOptions&, MatrixMut<float>, std::span<float>, Prediction<double>&);
template std::expected<void, GprError> loo_predict_into<MixedPrecision<ReevaluateKernel>>(
    const GaussianLikelihood&, MatrixRef<float>, std::span<const double>, std::span<const double>,
    const PredictOptions&, MatrixMut<float>, std::span<float>, Prediction<double>&);

template std::expected<Prediction<double>, GprError>
loo_predict<DoublePrecision>(const GaussianLikelihood&, MatrixRef<double>, std::span<const double>,
                             std::span<const double>, const PredictOptions&);
template std::expected<Prediction<float>, GprError>
loo_predict<SinglePrecision>(const GaussianLikelihood&, MatrixRef<float>, std::span<const float>,
                             std::span<const float>, const PredictOptions&);
template std::expected<Prediction<double>, GprError>
loo_predict<MixedPrecision<PromoteStorage>>(const GaussianLikelihood&, MatrixRef<float>,
                                            std::span<const double>, std::span<const double>,
                                            const PredictOptions&);
template std::expected<Prediction<double>, GprError>
loo_predict<MixedPrecision<ReevaluateKernel>>(const GaussianLikelihood&, MatrixRef<float>,
                                              std::span<const double>, std::span<const double>,
                                              const PredictOptions&);

} // namespace cxxgpr
