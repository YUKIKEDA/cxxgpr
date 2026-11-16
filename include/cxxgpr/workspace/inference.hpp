// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/workspace/buffers.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

namespace cxxgpr {

/// Computes posterior mean and diagonal predictive variance into out using pre-allocated
/// QueryWorkspace buffers (zero heap allocations after warmup on matching query dimensions).
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError>
predict_into(const CompiledKernel<typename P::Storage>& kernel,
             const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> x_train,
             MatrixRef<typename P::Storage> l_factor, std::span<const typename P::Refine> alpha,
             MatrixRef<typename P::Storage> x_query, const PredictOptions& options,
             QueryWorkspace<P>& workspace, Prediction<typename P::Refine>& out);

/// Computes posterior mean and diagonal predictive variance for query points x_query.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError>
predict(const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
        MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
        std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
        const PredictOptions& options, QueryWorkspace<P>& workspace);

/// Computes posterior mean and full m x m column-major posterior covariance matrix into out.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError> predict_covariance_into(
    const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
    MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
    std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
    const PredictOptions& options, QueryWorkspace<P>& workspace,
    PredictiveCovariance<typename P::Refine>& out);

/// Computes posterior mean and full m x m column-major posterior covariance matrix.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<PredictiveCovariance<typename P::Refine>, GprError> predict_covariance(
    const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
    MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
    std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
    const PredictOptions& options, QueryWorkspace<P>& workspace);

/// Draws n_draws deterministic posterior samples (packed as draw * m + i) from the predictive
/// distribution using Xoshiro256++ and Box-Muller standard normals.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<std::vector<typename P::Refine>, GprError>
sample(const CompiledKernel<typename P::Storage>& kernel, const GaussianLikelihood& likelihood,
       MatrixRef<typename P::Storage> x_train, MatrixRef<typename P::Storage> l_factor,
       std::span<const typename P::Refine> alpha, MatrixRef<typename P::Storage> x_query,
       std::size_t n_draws, std::uint64_t seed, const PredictOptions& options,
       QueryWorkspace<P>& workspace, std::optional<JitterPolicy> jitter = std::nullopt);

/// Computes closed-form Leave-One-Out predictive mean and variance (GPML §5.4.2) into out using
/// caller-provided scratch buffers (zero heap allocations when out has sufficient capacity).
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<void, GprError>
loo_predict_into(const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> l_factor,
                 std::span<const typename P::Refine> alpha,
                 std::span<const typename P::Refine> y_train, const PredictOptions& options,
                 MatrixMut<typename P::Storage> scratch_n_by_n,
                 std::span<typename P::Storage> scratch_diag, Prediction<typename P::Refine>& out);

/// Computes closed-form Leave-One-Out predictive mean and variance (GPML §5.4.2).
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<Prediction<typename P::Refine>, GprError>
loo_predict(const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> l_factor,
            std::span<const typename P::Refine> alpha, std::span<const typename P::Refine> y_train,
            const PredictOptions& options);

extern template std::expected<void, GprError>
predict_into<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                              MatrixRef<double>, MatrixRef<double>, std::span<const double>,
                              MatrixRef<double>, const PredictOptions&,
                              QueryWorkspace<DoublePrecision>&, Prediction<double>&);
extern template std::expected<void, GprError>
predict_into<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&,
                              MatrixRef<float>, MatrixRef<float>, std::span<const float>,
                              MatrixRef<float>, const PredictOptions&,
                              QueryWorkspace<SinglePrecision>&, Prediction<float>&);
extern template std::expected<void, GprError> predict_into<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, Prediction<double>&);
extern template std::expected<void, GprError> predict_into<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, Prediction<double>&);

extern template std::expected<Prediction<double>, GprError>
predict<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                         MatrixRef<double>, MatrixRef<double>, std::span<const double>,
                         MatrixRef<double>, const PredictOptions&,
                         QueryWorkspace<DoublePrecision>&);
extern template std::expected<Prediction<float>, GprError>
predict<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
                         MatrixRef<float>, std::span<const float>, MatrixRef<float>,
                         const PredictOptions&, QueryWorkspace<SinglePrecision>&);
extern template std::expected<Prediction<double>, GprError>
predict<MixedPrecision<PromoteStorage>>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                        MatrixRef<float>, MatrixRef<float>, std::span<const double>,
                                        MatrixRef<float>, const PredictOptions&,
                                        QueryWorkspace<MixedPrecision<PromoteStorage>>&);
extern template std::expected<Prediction<double>, GprError>
predict<MixedPrecision<ReevaluateKernel>>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                          MatrixRef<float>, MatrixRef<float>,
                                          std::span<const double>, MatrixRef<float>,
                                          const PredictOptions&,
                                          QueryWorkspace<MixedPrecision<ReevaluateKernel>>&);

extern template std::expected<void, GprError> predict_covariance_into<DoublePrecision>(
    const CompiledKernel<double>&, const GaussianLikelihood&, MatrixRef<double>, MatrixRef<double>,
    std::span<const double>, MatrixRef<double>, const PredictOptions&,
    QueryWorkspace<DoublePrecision>&, PredictiveCovariance<double>&);
extern template std::expected<void, GprError> predict_covariance_into<SinglePrecision>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const float>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<SinglePrecision>&, PredictiveCovariance<float>&);
extern template std::expected<void, GprError>
predict_covariance_into<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, PredictiveCovariance<double>&);
extern template std::expected<void, GprError>
predict_covariance_into<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&, PredictiveCovariance<double>&);

extern template std::expected<PredictiveCovariance<double>, GprError>
predict_covariance<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                                    MatrixRef<double>, MatrixRef<double>, std::span<const double>,
                                    MatrixRef<double>, const PredictOptions&,
                                    QueryWorkspace<DoublePrecision>&);
extern template std::expected<PredictiveCovariance<float>, GprError>
predict_covariance<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                    MatrixRef<float>, MatrixRef<float>, std::span<const float>,
                                    MatrixRef<float>, const PredictOptions&,
                                    QueryWorkspace<SinglePrecision>&);
extern template std::expected<PredictiveCovariance<double>, GprError>
predict_covariance<MixedPrecision<PromoteStorage>>(const CompiledKernel<float>&,
                                                   const GaussianLikelihood&, MatrixRef<float>,
                                                   MatrixRef<float>, std::span<const double>,
                                                   MatrixRef<float>, const PredictOptions&,
                                                   QueryWorkspace<MixedPrecision<PromoteStorage>>&);
extern template std::expected<PredictiveCovariance<double>, GprError>
predict_covariance<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, const PredictOptions&,
    QueryWorkspace<MixedPrecision<ReevaluateKernel>>&);

extern template std::expected<std::vector<double>, GprError>
sample<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&, MatrixRef<double>,
                        MatrixRef<double>, std::span<const double>, MatrixRef<double>, std::size_t,
                        std::uint64_t, const PredictOptions&, QueryWorkspace<DoublePrecision>&,
                        std::optional<JitterPolicy>);
extern template std::expected<std::vector<float>, GprError>
sample<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
                        MatrixRef<float>, std::span<const float>, MatrixRef<float>, std::size_t,
                        std::uint64_t, const PredictOptions&, QueryWorkspace<SinglePrecision>&,
                        std::optional<JitterPolicy>);
extern template std::expected<std::vector<double>, GprError> sample<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>, MatrixRef<float>,
    std::span<const double>, MatrixRef<float>, std::size_t, std::uint64_t, const PredictOptions&,
    QueryWorkspace<MixedPrecision<PromoteStorage>>&, std::optional<JitterPolicy>);
extern template std::expected<std::vector<double>, GprError>
sample<MixedPrecision<ReevaluateKernel>>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                         MatrixRef<float>, MatrixRef<float>,
                                         std::span<const double>, MatrixRef<float>, std::size_t,
                                         std::uint64_t, const PredictOptions&,
                                         QueryWorkspace<MixedPrecision<ReevaluateKernel>>&,
                                         std::optional<JitterPolicy>);

extern template std::expected<void, GprError> loo_predict_into<DoublePrecision>(
    const GaussianLikelihood&, MatrixRef<double>, std::span<const double>, std::span<const double>,
    const PredictOptions&, MatrixMut<double>, std::span<double>, Prediction<double>&);
extern template std::expected<void, GprError> loo_predict_into<SinglePrecision>(
    const GaussianLikelihood&, MatrixRef<float>, std::span<const float>, std::span<const float>,
    const PredictOptions&, MatrixMut<float>, std::span<float>, Prediction<float>&);
extern template std::expected<void, GprError> loo_predict_into<MixedPrecision<PromoteStorage>>(
    const GaussianLikelihood&, MatrixRef<float>, std::span<const double>, std::span<const double>,
    const PredictOptions&, MatrixMut<float>, std::span<float>, Prediction<double>&);
extern template std::expected<void, GprError> loo_predict_into<MixedPrecision<ReevaluateKernel>>(
    const GaussianLikelihood&, MatrixRef<float>, std::span<const double>, std::span<const double>,
    const PredictOptions&, MatrixMut<float>, std::span<float>, Prediction<double>&);

extern template std::expected<Prediction<double>, GprError>
loo_predict<DoublePrecision>(const GaussianLikelihood&, MatrixRef<double>, std::span<const double>,
                             std::span<const double>, const PredictOptions&);
extern template std::expected<Prediction<float>, GprError>
loo_predict<SinglePrecision>(const GaussianLikelihood&, MatrixRef<float>, std::span<const float>,
                             std::span<const float>, const PredictOptions&);
extern template std::expected<Prediction<double>, GprError>
loo_predict<MixedPrecision<PromoteStorage>>(const GaussianLikelihood&, MatrixRef<float>,
                                            std::span<const double>, std::span<const double>,
                                            const PredictOptions&);
extern template std::expected<Prediction<double>, GprError>
loo_predict<MixedPrecision<ReevaluateKernel>>(const GaussianLikelihood&, MatrixRef<float>,
                                              std::span<const double>, std::span<const double>,
                                              const PredictOptions&);

} // namespace cxxgpr
