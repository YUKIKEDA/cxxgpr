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
#include <expected>
#include <span>

namespace cxxgpr {

/// Assembles A = K + sigma_n^2 I (keeping keep_products factor Grams), factors L, solves alpha,
/// and returns the negative marginal log-likelihood.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<double, GprError>
factor_and_mll(const CompiledKernel<typename P::Storage>& kernel,
               const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> x,
               std::span<const typename P::Storage> y, const JitterPolicy& jitter,
               FitBuffers<P>& buffers, std::size_t keep_products = 0);

/// Computes the negative marginal log-likelihood and analytical log-space parameter gradients
/// (kernel parameters followed by log(sigma_n)) in a single pass using keep_plan.
template <PrecisionPolicyConcept P>
[[nodiscard]] std::expected<double, GprError>
mll_and_gradient_into(const CompiledKernel<typename P::Storage>& kernel,
                      const GaussianLikelihood& likelihood, MatrixRef<typename P::Storage> x,
                      std::span<const typename P::Storage> y, const JitterPolicy& jitter,
                      FitBuffers<P>& buffers, std::span<double> out_grad);

extern template std::expected<double, GprError>
factor_and_mll<DoublePrecision>(const CompiledKernel<double>&, const GaussianLikelihood&,
                                MatrixRef<double>, std::span<const double>, const JitterPolicy&,
                                FitBuffers<DoublePrecision>&, std::size_t);
extern template std::expected<double, GprError>
factor_and_mll<SinglePrecision>(const CompiledKernel<float>&, const GaussianLikelihood&,
                                MatrixRef<float>, std::span<const float>, const JitterPolicy&,
                                FitBuffers<SinglePrecision>&, std::size_t);
extern template std::expected<double, GprError> factor_and_mll<MixedPrecision<PromoteStorage>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<PromoteStorage>>&,
    std::size_t);
extern template std::expected<double, GprError> factor_and_mll<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<ReevaluateKernel>>&,
    std::size_t);

extern template std::expected<double, GprError> mll_and_gradient_into<DoublePrecision>(
    const CompiledKernel<double>&, const GaussianLikelihood&, MatrixRef<double>,
    std::span<const double>, const JitterPolicy&, FitBuffers<DoublePrecision>&, std::span<double>);
extern template std::expected<double, GprError> mll_and_gradient_into<SinglePrecision>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<SinglePrecision>&, std::span<double>);
extern template std::expected<double, GprError>
mll_and_gradient_into<MixedPrecision<PromoteStorage>>(const CompiledKernel<float>&,
                                                      const GaussianLikelihood&, MatrixRef<float>,
                                                      std::span<const float>, const JitterPolicy&,
                                                      FitBuffers<MixedPrecision<PromoteStorage>>&,
                                                      std::span<double>);
extern template std::expected<double, GprError>
mll_and_gradient_into<MixedPrecision<ReevaluateKernel>>(
    const CompiledKernel<float>&, const GaussianLikelihood&, MatrixRef<float>,
    std::span<const float>, const JitterPolicy&, FitBuffers<MixedPrecision<ReevaluateKernel>>&,
    std::span<double>);

} // namespace cxxgpr
