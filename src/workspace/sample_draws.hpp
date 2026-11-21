// SPDX-License-Identifier: Apache-2.0
/// @file sample_draws.hpp
/// @brief Internal entry point turning a predictive mean and covariance into deterministic posterior samples.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/linalg/jitter.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

/// @brief Internal helpers shared by the coordinate and the supplied-distance sampling paths.
namespace cxxgpr::detail
{

    /// @brief Returns the default adaptive jitter policy used when factoring a posterior covariance for sampling.
    /// @details Base jitter 1e-12, scale factor 10, 10 retry steps and cap 1e-2.
    /// @return Pre-validated adaptive JitterPolicy.
    [[nodiscard]] JitterPolicy default_sample_jitter();

    /// @brief Draws deterministic posterior samples mu + L z from a predictive mean and covariance.
    /// @details Factors the covariance with the given jitter policy, then generates standard normals in batches of
    /// eight draws from Xoshiro256++ seeded by seed. The random stream, the accumulation order of every output entry
    /// and the result do not depend on the number of threads.
    /// @tparam Refine Output scalar type (double or float).
    /// @param[in] cov Predictive mean and column-major m x m covariance; the covariance buffer is overwritten.
    /// @param[in] n_draws Number of samples; zero returns an empty vector.
    /// @param[in] seed Seed of the generator.
    /// @param[in] jitter Jitter policy for the covariance factorization.
    /// @return Column-major m x n_draws samples, or GprError when the covariance cannot be factored.
    /// @retval GprErrorCode::SizeOverflow When m * n_draws overflows std::size_t.
    /// @retval GprErrorCode::CholeskyFailed When the covariance is not positive definite after jitter.
    template <typename Refine>
    [[nodiscard]] std::expected<std::vector<Refine>, GprError> draw_samples_from_covariance(
        PredictiveCovariance<Refine>&& cov, std::size_t n_draws, std::uint64_t seed, const JitterPolicy& jitter
    );

}  // namespace cxxgpr::detail
