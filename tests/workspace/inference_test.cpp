// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-08, REQ-11, DES-16, DES-17, DES-20
/// @file inference_test.cpp
/// @brief Unit and property tests for GPR posterior prediction, covariance, sampling, and LOO.
/// @details Verifies workspace inference routines against direct dense linear-algebra oracles,
/// checks zero-allocation predict_into() and loo_predict_into() across Double and MixedPrecision,
/// validates posterior sampling determinism and Monte Carlo moments, OpenMP bitwise invariance,
/// and DES-02 negative input validation precedence.

#include "cxxgpr/linalg/cholesky.hpp"
#include "cxxgpr/workspace/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "allocation_audit.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace
{

    /// @brief Pre-fitted Cholesky factor, alpha vector, and training arrays for inference tests.
    /// @details Bundles precision-typed training tensors, lower factor L, and double/refine-typed
    /// dual coefficients alpha produced by factor_and_mll() for reuse across prediction tests.
    /// @tparam P Precision policy struct providing Storage and Refine types.
    template <typename P> struct FittedFixture
    {
        /// @brief Kernel and matrix storage scalar type (float or double).
        using Storage = P::Storage;
        /// @brief Refinement and prediction output scalar type (float or double).
        using Refine = P::Refine;

        /// @brief Fitted Gaussian observation noise likelihood.
        cxxgpr::GaussianLikelihood likelihood;
        /// @brief Column-major n x d training inputs in storage precision.
        std::vector<Storage> x_train;
        /// @brief Training targets y in storage precision.
        std::vector<Storage> y_storage;
        /// @brief Training targets y in refinement/prediction precision.
        std::vector<Refine> y_refine;
        /// @brief Column-major n x n lower Cholesky factor L of (K + sigma_n^2 I).
        std::vector<Storage> l_factor;
        /// @brief Solved dual coefficient vector alpha = (K + sigma_n^2 I)^(-1) y.
        std::vector<Refine> alpha;
        /// @brief Number of training observations.
        std::size_t n = 0;
        /// @brief Input feature dimensionality.
        std::size_t d = 0;

        /// @brief Returns a non-owning column-major MatrixRef view over training inputs x_train.
        /// @return Read-only n x d view of training coordinates.
        [[nodiscard]] cxxgpr::MatrixRef<Storage> x_ref() const noexcept
        {
            return {x_train.data(), n, d, d == 0 ? 1 : n};
        }

        /// @brief Returns a non-owning column-major MatrixRef view over lower Cholesky factor L.
        /// @return Read-only n x n view of lower factor L.
        [[nodiscard]] cxxgpr::MatrixRef<Storage> l_ref() const noexcept { return {l_factor.data(), n, n, n}; }
    };

    /// @brief Constructs and factors a FittedFixture from raw double-precision training arrays.
    /// @details Converts inputs to P::Storage, invokes factor_and_mll() with zero fixed jitter, and
    /// copies the resulting lower Cholesky factor L and dual vector alpha into the returned fixture.
    /// @tparam P Precision policy type governing storage and refinement scalar types.
    /// @param[in] compiled Precompiled kernel evaluated at its current parameters.
    /// @param[in] x_vals Flattened column-major n x d training coordinate array.
    /// @param[in] y_vals Training target vector of length n.
    /// @param[in] n Number of training observations.
    /// @param[in] d Number of input feature dimensions.
    /// @param[in] noise_sd Gaussian noise standard deviation passed to GaussianLikelihood::create().
    /// @return Populated FittedFixture containing L, alpha, and casted training data.
    template <typename P>
    FittedFixture<P> make_fitted_fixture(
        const cxxgpr::CompiledKernel<typename P::Storage>& compiled,
        const std::vector<double>& x_vals,
        const std::vector<double>& y_vals,
        std::size_t n,
        std::size_t d,
        double noise_sd
    )
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        // Configure likelihood, zero jitter, and speed-optimized FitBuffers before factoring.
        auto lik = cxxgpr::GaussianLikelihood::create(noise_sd).value();
        auto jitter = cxxgpr::JitterPolicy::fixed(0.0).value();
        auto bufs = cxxgpr::FitBuffers<P>::create(n, cxxgpr::FitPolicy::prefer_speed());

        std::vector<Storage> x_storage(x_vals.size());
        for (std::size_t i = 0; i < x_vals.size(); ++i)
        {
            x_storage[i] = static_cast<Storage>(x_vals[i]);
        }
        std::vector<Storage> y_storage(y_vals.size());
        std::vector<Refine> y_refine(y_vals.size());
        for (std::size_t i = 0; i < y_vals.size(); ++i)
        {
            y_storage[i] = static_cast<Storage>(y_vals[i]);
            y_refine[i] = static_cast<Refine>(y_vals[i]);
        }

        cxxgpr::MatrixRef<Storage> x_ref(x_storage.data(), n, d, n);
        auto mll = cxxgpr::factor_and_mll(compiled, lik, x_ref, y_storage, jitter, *bufs);
        EXPECT_TRUE(mll.has_value());

        std::vector<Storage> l_copy(n * n);
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                l_copy[(col * n) + row] = bufs->core.k_ref()(row, col);
            }
        }
        std::vector<Refine> alpha_copy(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            alpha_copy[i] = static_cast<Refine>(bufs->core.rhs[i]);
        }

        return FittedFixture<P>{
            .likelihood = lik,
            .x_train = std::move(x_storage),
            .y_storage = std::move(y_storage),
            .y_refine = std::move(y_refine),
            .l_factor = std::move(l_copy),
            .alpha = std::move(alpha_copy),
            .n = n,
            .d = d,
        };
    }

    /// @brief Verifies predict_into() matches direct dense Cholesky solves and allocates zero memory.
    /// @details Compares Observation and Latent predictive variances, checks zero allocations after
    /// warmup via NoAllocationScope, and cross-checks mean and variance against solve_llt_in_place().
    TEST(WorkspaceInferenceTest, PredictMatchesDirectReferenceAndZeroAllocatesAfterWarmup)
    {
        // Spec: REQ-08, REQ-11, DES-16, DES-17
        const auto spec = (cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.85).value())
                              * cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.4).value()))
            + cxxgpr::KernelSpec(
                cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.7, 1.3}, cxxgpr::MaternNu::FiveHalves).value()
            );
        const std::size_t n = 6;
        const std::size_t d = 2;
        const std::size_t m = 4;
        const std::vector<double> x_train_vals = {
            -1.2, -0.4, 0.1, 0.7, 1.3, 1.9,  // col 0
            0.5, -0.9, 0.3, 1.1, -0.2, 0.8   // col 1
        };
        const std::vector<double> y_train_vals = {1.1, -0.3, 0.7, 1.8, 0.2, -0.9};
        const std::vector<double> x_query_vals = {
            -0.8, 0.2, 0.9, 1.5,  // col 0
            0.1, -0.5, 0.7, -0.1  // col 1
        };

        const auto kernel = cxxgpr::CompiledKernel<double>::from_spec(spec);
        auto fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, x_train_vals, y_train_vals, n, d, 0.15);
        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws;
        cxxgpr::MatrixRef<double> xq_ref(x_query_vals.data(), m, d, m);
        cxxgpr::Prediction<double> pred_obs;

        // Warmup call allocates QueryWorkspace and output vectors.
        auto r1 = cxxgpr::predict_into(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}, qws, pred_obs
        );
        ASSERT_TRUE(r1.has_value());
        ASSERT_EQ(pred_obs.mean.size(), m);
        ASSERT_EQ(pred_obs.variance.size(), m);
        EXPECT_EQ(pred_obs.variance_kind, cxxgpr::VarianceKind::Observation);

        // Second call with identical query shape must allocate zero bytes.
        cxxgpr::Prediction<double> pred_lat;
        pred_lat.mean.reserve(m);
        pred_lat.variance.reserve(m);
        {
            const cxxgpr::test_support::NoAllocationScope no_alloc;
            auto r2 = cxxgpr::predict_into(
                kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref,
                cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, qws, pred_lat
            );
            ASSERT_TRUE(r2.has_value());
        }
        EXPECT_EQ(pred_lat.variance_kind, cxxgpr::VarianceKind::Latent);
        for (std::size_t j = 0; j < m; ++j)
        {
            EXPECT_DOUBLE_EQ(pred_lat.mean[j], pred_obs.mean[j]);
            EXPECT_NEAR(pred_obs.variance[j] - pred_lat.variance[j], fix.likelihood.noise_variance(), 1e-13);
        }

        // Independent reference using direct dense solves: K_*^T alpha and diag(K_**) - diag(K_*^T A^-1
        // K_*).
        std::vector<double> k_star(n * m, 0.0);
        std::vector<double> k_star_scratch(n * m, 0.0);
        cxxgpr::MatrixMut<double> k_star_mut(k_star.data(), n, m, n);
        cxxgpr::MatrixMut<double> k_star_sc(k_star_scratch.data(), n, m, n);
        ASSERT_TRUE(kernel.apply_cross_points(fix.x_ref(), xq_ref, k_star_mut, k_star_sc).has_value());

        std::vector<double> k_ss_diag(m, 0.0);
        ASSERT_TRUE(kernel.fill_diag(xq_ref, k_ss_diag).has_value());

        std::vector<double> a_inv_k_star = k_star;
        cxxgpr::MatrixMut<double> rhs_mut(a_inv_k_star.data(), n, m, n);
        ASSERT_TRUE(cxxgpr::linalg::solve_llt_in_place(fix.l_ref(), rhs_mut).has_value());

        for (std::size_t j = 0; j < m; ++j)
        {
            double ref_mean = 0.0;
            double quad = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                ref_mean += k_star[(j * n) + i] * fix.alpha[i];
                quad += k_star[(j * n) + i] * a_inv_k_star[(j * n) + i];
            }
            const double ref_obs_var = std::max(0.0, k_ss_diag[j] - quad) + fix.likelihood.noise_variance();
            EXPECT_NEAR(pred_obs.mean[j], ref_mean, 1e-12);
            EXPECT_NEAR(pred_obs.variance[j], ref_obs_var, 1e-12);
        }
    }

    /// @brief Verifies MixedPrecision predict_into() allocates zero heap memory after warmup.
    /// @details Warms up QueryWorkspace under MixedPrecision<PromoteStorage> and asserts zero heap
    /// allocations during subsequent predict_into() calls of identical query shape.
    TEST(WorkspaceInferenceTest, MixedPrecisionPredictZeroAllocatesAfterWarmup)
    {
        // Spec: REQ-11, DES-16
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const std::size_t n = 5;
        const std::size_t d = 1;
        const std::size_t m = 3;
        const std::vector<double> x_vals = {-1.0, -0.4, 0.1, 0.6, 1.2};
        const std::vector<double> y_vals = {0.5, -0.2, 0.8, 1.1, -0.4};
        const std::vector<double> xq_vals = {-0.7, 0.0, 0.9};

        const auto kernel = cxxgpr::CompiledKernel<float>::from_spec(spec);
        auto fix =
            make_fitted_fixture<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(kernel, x_vals, y_vals, n, d, 0.1);
        std::vector<float> xq_float(xq_vals.size());
        for (std::size_t i = 0; i < xq_vals.size(); ++i)
        {
            xq_float[i] = static_cast<float>(xq_vals[i]);
        }
        cxxgpr::MatrixRef<float> xq_ref(xq_float.data(), m, d, m);
        cxxgpr::QueryWorkspace<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>> qws;
        cxxgpr::Prediction<double> out;

        ASSERT_TRUE(
            cxxgpr::predict_into(
                kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, cxxgpr::PredictOptions{}, qws, out
            )
                .has_value()
        );

        const cxxgpr::test_support::NoAllocationScope no_alloc;
        ASSERT_TRUE(
            cxxgpr::predict_into(
                kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, cxxgpr::PredictOptions{}, qws, out
            )
                .has_value()
        );
    }

    /// @brief Verifies predict_covariance() symmetry/diagonal parity and posterior sample() properties.
    /// @details Confirms covariance symmetry and diagonal match against predict(), zero-draw empty
    /// return, deterministic seed reproducibility, and 4000-draw Monte Carlo mean/covariance parity.
    TEST(WorkspaceInferenceTest, PredictCovarianceAndSampleMatchProperties)
    {
        // Spec: DES-16, DES-20
        const auto spec = cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(0.9, cxxgpr::MaternNu::ThreeHalves).value())
            * cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value());
        const std::size_t n = 5;
        const std::size_t d = 1;
        const std::size_t m = 3;
        const std::vector<double> x_vals = {-1.5, -0.6, 0.0, 0.7, 1.4};
        const std::vector<double> y_vals = {1.2, 0.4, -0.5, 0.9, 1.5};
        const std::vector<double> xq_vals = {-1.0, 0.2, 1.1};

        const auto kernel = cxxgpr::CompiledKernel<double>::from_spec(spec);
        auto fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, x_vals, y_vals, n, d, 0.1);
        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws;
        cxxgpr::MatrixRef<double> xq_ref(xq_vals.data(), m, d, m);

        auto pred = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}, qws
        );
        ASSERT_TRUE(pred.has_value());

        auto cov = cxxgpr::predict_covariance(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}, qws
        );
        ASSERT_TRUE(cov.has_value());
        ASSERT_EQ(cov->n_points, m);
        ASSERT_EQ(cov->mean.size(), m);
        ASSERT_EQ(cov->covariance.size(), m * m);

        for (std::size_t j = 0; j < m; ++j)
        {
            EXPECT_NEAR(cov->mean[j], pred->mean[j], 1e-13);
            EXPECT_NEAR(cov->covariance[(j * m) + j], pred->variance[j], 1e-13);
            for (std::size_t i = 0; i < m; ++i)
            {
                EXPECT_DOUBLE_EQ(cov->covariance[(j * m) + i], cov->covariance[(i * m) + j]);
            }
        }

        // Determinism of sample() with identical seeds, and empty vector for n_draws == 0.
        auto empty_samples = cxxgpr::sample(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, 0U, 42ULL,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, qws
        );
        ASSERT_TRUE(empty_samples.has_value());
        EXPECT_TRUE(empty_samples->empty());

        auto s1 = cxxgpr::sample(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, 4U, 12345ULL,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, qws
        );
        auto s2 = cxxgpr::sample(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, 4U, 12345ULL,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, qws
        );
        auto s_other = cxxgpr::sample(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, 4U, 99999ULL,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, qws
        );
        ASSERT_TRUE(s1.has_value());
        ASSERT_TRUE(s2.has_value());
        ASSERT_TRUE(s_other.has_value());
        ASSERT_EQ(s1->size(), 4U * m);
        EXPECT_EQ(*s1, *s2);
        EXPECT_NE(*s1, *s_other);

        // Crossing a draw-batch boundary must preserve the seeded prefix, including the partial final batch.
        auto more = cxxgpr::sample(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, 9U, 12345ULL,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, qws
        );
        ASSERT_TRUE(more.has_value());
        ASSERT_EQ(more->size(), 9U * m);
        EXPECT_TRUE(std::equal(s1->begin(), s1->end(), more->begin()));

        // Monte Carlo sanity: empirical mean and covariance of 4000 draws approach analytical
        // posterior.
        const std::size_t n_draws = 4000;
        auto mc = cxxgpr::sample(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, n_draws, 777ULL,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}, qws
        );
        ASSERT_TRUE(mc.has_value());
        std::vector<double> emp_mean(m, 0.0);
        for (std::size_t draw = 0; draw < n_draws; ++draw)
        {
            for (std::size_t i = 0; i < m; ++i)
            {
                emp_mean[i] += (*mc)[(draw * m) + i];
            }
        }
        for (std::size_t i = 0; i < m; ++i)
        {
            emp_mean[i] /= static_cast<double>(n_draws);
            EXPECT_NEAR(emp_mean[i], cov->mean[i], 0.06);
        }
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < m; ++j)
            {
                double c_ij = 0.0;
                for (std::size_t draw = 0; draw < n_draws; ++draw)
                {
                    c_ij += ((*mc)[(draw * m) + i] - emp_mean[i]) * ((*mc)[(draw * m) + j] - emp_mean[j]);
                }
                c_ij /= static_cast<double>(n_draws - 1U);
                EXPECT_NEAR(c_ij, cov->covariance[(j * m) + i], 0.09);
            }
        }
    }

    /// @brief Verifies analytical LOO predictions match explicit held-out refits with zero allocations.
    /// @details Compares loo_predict() Observation and Latent mean/variance against n individual
    /// refits omitting each point, and audits loo_predict_into() under NoAllocationScope.
    TEST(WorkspaceInferenceTest, LooPredictMatchesExplicitLeaveOneOutReference)
    {
        // Spec: REQ-08, DES-16
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.9).value())
            * cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.5).value());
        const std::size_t n = 6;
        const std::size_t d = 2;
        const std::vector<double> x_vals = {
            -1.3, -0.5, 0.0, 0.6, 1.1, 1.7,  // col 0
            0.2, -0.8, 0.5, -0.1, 1.0, -0.4  // col 1
        };
        const std::vector<double> y_vals = {0.8, -0.4, 1.2, 0.3, 1.6, -0.7};

        const auto kernel = cxxgpr::CompiledKernel<double>::from_spec(spec);
        auto fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, x_vals, y_vals, n, d, 0.2);

        auto loo_obs = cxxgpr::loo_predict<cxxgpr::DoublePrecision>(
            fix.likelihood, fix.l_ref(), fix.alpha, fix.y_refine,
            cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}
        );
        auto loo_lat = cxxgpr::loo_predict<cxxgpr::DoublePrecision>(
            fix.likelihood, fix.l_ref(), fix.alpha, fix.y_refine, cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}
        );
        ASSERT_TRUE(loo_obs.has_value());
        ASSERT_TRUE(loo_lat.has_value());
        ASSERT_EQ(loo_obs->mean.size(), n);
        ASSERT_EQ(loo_obs->variance.size(), n);

        // Explicit leave-one-out refit & predict for each held-out point i.
        for (std::size_t held = 0; held < n; ++held)
        {
            std::vector<double> sub_x((n - 1U) * d);
            std::vector<double> sub_y(n - 1U);
            std::size_t dst = 0;
            for (std::size_t row = 0; row < n; ++row)
            {
                if (row == held)
                {
                    continue;
                }
                for (std::size_t col = 0; col < d; ++col)
                {
                    sub_x[(col * (n - 1U)) + dst] = x_vals[(col * n) + row];
                }
                sub_y[dst] = y_vals[row];
                ++dst;
            }
            auto sub_fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, sub_x, sub_y, n - 1U, d, 0.2);
            std::vector<double> qx(d);
            for (std::size_t col = 0; col < d; ++col)
            {
                qx[col] = x_vals[(col * n) + held];
            }
            cxxgpr::MatrixRef<double> qx_ref(qx.data(), 1U, d, 1U);
            cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> sub_qws;
            auto sub_obs = cxxgpr::predict(
                kernel, sub_fix.likelihood, sub_fix.x_ref(), sub_fix.l_ref(), sub_fix.alpha, qx_ref,
                cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}, sub_qws
            );
            auto sub_lat = cxxgpr::predict(
                kernel, sub_fix.likelihood, sub_fix.x_ref(), sub_fix.l_ref(), sub_fix.alpha, qx_ref,
                cxxgpr::PredictOptions{cxxgpr::VarianceKind::Latent}, sub_qws
            );
            ASSERT_TRUE(sub_obs.has_value());
            ASSERT_TRUE(sub_lat.has_value());

            EXPECT_NEAR(loo_obs->mean[held], sub_obs->mean[0], 1e-11);
            EXPECT_NEAR(loo_obs->variance[held], sub_obs->variance[0], 1e-11);
            EXPECT_NEAR(loo_lat->mean[held], sub_lat->mean[0], 1e-11);
            EXPECT_NEAR(loo_lat->variance[held], sub_lat->variance[0], 1e-11);
        }

        // Zero-allocation check on loo_predict_into with pre-sized buffers.
        std::vector<double> sc_mat(n * n);
        std::vector<double> sc_diag(n);
        cxxgpr::Prediction<double> loo_out;
        loo_out.mean.reserve(n);
        loo_out.variance.reserve(n);
        {
            const cxxgpr::test_support::NoAllocationScope no_alloc;
            auto res = cxxgpr::loo_predict_into<cxxgpr::DoublePrecision>(
                fix.likelihood, fix.l_ref(), fix.alpha, fix.y_refine,
                cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation},
                cxxgpr::MatrixMut<double>(sc_mat.data(), n, n, n), sc_diag, loo_out
            );
            ASSERT_TRUE(res.has_value());
        }
    }

    /// @brief Verifies predict() outputs are bit-identical between 1 and 4 OpenMP threads.
    /// @details Compares std::bit_cast<uint64_t> representations of every predicted mean and variance
    /// across thread counts on an m = 32 query grid.
    TEST(WorkspaceInferenceTest, OpenMpThreadCountBitwiseDeterminism)
    {
        // Spec: REQ-11, DES-20
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.8).value());
        const std::size_t n = 8;
        const std::size_t d = 2;
        const std::size_t m = 32;
        std::vector<double> x_vals(n * d);
        std::vector<double> y_vals(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x_vals[i] = -1.0 + (0.3 * static_cast<double>(i));
            x_vals[n + i] = 0.5 - (0.15 * static_cast<double>(i));
            y_vals[i] = std::sin(x_vals[i]) + (0.2 * x_vals[n + i]);
        }
        std::vector<double> xq_vals(m * d);
        for (std::size_t j = 0; j < m; ++j)
        {
            xq_vals[j] = -1.2 + (0.08 * static_cast<double>(j));
            xq_vals[m + j] = 0.4 - (0.03 * static_cast<double>(j));
        }

        const auto kernel = cxxgpr::CompiledKernel<double>::from_spec(spec);
        auto fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, x_vals, y_vals, n, d, 0.1);
        cxxgpr::MatrixRef<double> xq_ref(xq_vals.data(), m, d, m);
        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws1;
        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws4;

#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        auto p1 = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, cxxgpr::PredictOptions{}, qws1
        );
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        auto p4 = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, xq_ref, cxxgpr::PredictOptions{}, qws4
        );
        ASSERT_TRUE(p1.has_value());
        ASSERT_TRUE(p4.has_value());
        for (std::size_t j = 0; j < m; ++j)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(p1->mean[j]), std::bit_cast<std::uint64_t>(p4->mean[j]));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(p1->variance[j]), std::bit_cast<std::uint64_t>(p4->variance[j]));
        }
    }

    /// @brief Negative controls verifying predict(), loo_predict(), and fill_diag() error precedence.
    /// @details Confirms DimensionMismatch, NonFiniteInput, LengthMismatch, EmptyInput, and DES-02
    /// LengthMismatch-over-DimensionMismatch priority across prediction and LOO entry points.
    TEST(WorkspaceInferenceTest, NegativeControlsRejectInvalidInputs)
    {
        // Spec: DES-16
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const std::vector<double> x_vals = {0.0, 1.0, 0.5, -0.5};
        const std::vector<double> y_vals = {1.0, -1.0};
        const auto kernel = cxxgpr::CompiledKernel<double>::from_spec(spec);
        auto fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, x_vals, y_vals, 2U, 2U, 0.1);
        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws;

        // A borrowed factor can change between calls using the same workspace.
        auto borrowed_factor = fix.l_factor;
        const cxxgpr::MatrixRef<double> borrowed_view(borrowed_factor.data(), 2U, 2U);
        cxxgpr::Prediction<double> prediction;
        const auto predict_borrowed = [&]()
        {
            return cxxgpr::predict_into(
                kernel, fix.likelihood, fix.x_ref(), borrowed_view, fix.alpha, fix.x_ref(), cxxgpr::PredictOptions{},
                qws, prediction
            );
        };
        ASSERT_TRUE(predict_borrowed().has_value());
        borrowed_factor[1] = std::numeric_limits<double>::quiet_NaN();
        const auto bad_lower = predict_borrowed();
        ASSERT_FALSE(bad_lower.has_value());
        EXPECT_EQ(bad_lower.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

        // Reject query column-count mismatch with DimensionMismatch.
        const std::vector<double> bad_dim_q = {0.1, 0.2};
        auto err_dim = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha,
            cxxgpr::MatrixRef<double>(bad_dim_q.data(), 2U, 1U, 2U), cxxgpr::PredictOptions{}, qws
        );
        ASSERT_FALSE(err_dim.has_value());
        EXPECT_EQ(err_dim.error().code(), cxxgpr::GprErrorCode::DimensionMismatch);

        const std::vector<double> nan_q = {std::numeric_limits<double>::quiet_NaN(), 0.0};
        auto err_nan = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha,
            cxxgpr::MatrixRef<double>(nan_q.data(), 1U, 2U, 1U), cxxgpr::PredictOptions{}, qws
        );
        ASSERT_FALSE(err_nan.has_value());
        EXPECT_EQ(err_nan.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

        const std::vector<double> bad_alpha = {1.0};
        auto err_alpha = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), bad_alpha,
            cxxgpr::MatrixRef<double>(x_vals.data(), 2U, 2U, 2U), cxxgpr::PredictOptions{}, qws
        );
        ASSERT_FALSE(err_alpha.has_value());
        EXPECT_EQ(err_alpha.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        // DES-02 priority: LengthMismatch takes precedence over DimensionMismatch.
        auto err_prio = cxxgpr::predict(
            kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), bad_alpha,
            cxxgpr::MatrixRef<double>(bad_dim_q.data(), 2U, 1U, 2U), cxxgpr::PredictOptions{}, qws
        );
        ASSERT_FALSE(err_prio.has_value());
        EXPECT_EQ(err_prio.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        const std::vector<double> empty_vec;
        auto err_empty_alpha = cxxgpr::loo_predict<cxxgpr::DoublePrecision>(
            fix.likelihood, fix.l_ref(), empty_vec, fix.y_refine, cxxgpr::PredictOptions{}
        );
        ASSERT_FALSE(err_empty_alpha.has_value());
        EXPECT_EQ(err_empty_alpha.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        auto err_nonsq_l = cxxgpr::loo_predict<cxxgpr::DoublePrecision>(
            fix.likelihood, cxxgpr::MatrixRef<double>(bad_dim_q.data(), 2U, 1U, 2U), fix.alpha, fix.y_refine,
            cxxgpr::PredictOptions{}
        );
        ASSERT_FALSE(err_nonsq_l.has_value());
        EXPECT_EQ(err_nonsq_l.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        const std::vector<double> bad_y = {1.0};
        auto err_loo = cxxgpr::loo_predict<cxxgpr::DoublePrecision>(
            fix.likelihood, fix.l_ref(), fix.alpha, bad_y, cxxgpr::PredictOptions{}
        );
        ASSERT_FALSE(err_loo.has_value());
        EXPECT_EQ(err_loo.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        std::vector<double> diag_out(2U);
        std::vector<double> tiny_scratch(1U);
        auto err_diag = kernel.fill_diag(fix.x_ref(), diag_out, tiny_scratch);
        ASSERT_FALSE(err_diag.has_value());
        EXPECT_EQ(err_diag.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
    }

    /// @brief Verifies a non-finite latent variance is reported instead of being clamped to zero.
    /// @details A subnormal positive diagonal is a valid finite factor, but solving against it overflows
    /// V = L^-1 K_* to +Inf, so k_** - ||v||^2 is -Inf. Clamping that value at zero would return the bare noise
    /// variance as if the query were perfectly known; every precision policy must instead fail with
    /// NonFiniteKernelValue for both variance kinds, and the same workspace must then predict normally again.
    TEST(WorkspaceInferenceTest, NonFiniteLatentVarianceIsReportedNotClampedToZero)
    {
        // Spec: DES-02, DES-16
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto check_policy = [&spec]<typename P>()
        {
            using Storage = P::Storage;
            using Refine = P::Refine;
            const auto kernel = spec.template compile<Storage>();
            const auto likelihood = cxxgpr::GaussianLikelihood::create(0.1).value();
            const std::vector<Storage> x_train{Storage{0}};
            const std::vector<Storage> x_query{Storage{0}, Storage{2}};
            const std::vector<Refine> alpha{Refine{0}};
            const cxxgpr::MatrixRef<Storage> x_train_ref(x_train.data(), 1U, 1U, 1U);
            const cxxgpr::MatrixRef<Storage> x_query_ref(x_query.data(), 2U, 1U, 2U);
            cxxgpr::QueryWorkspace<P> workspace;
            cxxgpr::Prediction<Refine> out;

            // The overflowing factor: k(0, x*) / denorm_min is +Inf in storage precision for both queries.
            const std::vector<Storage> overflowing{std::numeric_limits<Storage>::denorm_min()};
            const cxxgpr::MatrixRef<Storage> overflowing_ref(overflowing.data(), 1U, 1U, 1U);
            for (const auto kind : {cxxgpr::VarianceKind::Observation, cxxgpr::VarianceKind::Latent})
            {
                const cxxgpr::PredictOptions options{.variance_kind = kind};
                const auto res = cxxgpr::predict_into<P>(
                    kernel, likelihood, x_train_ref, overflowing_ref, alpha, x_query_ref, options, workspace, out
                );
                ASSERT_FALSE(res.has_value());
                EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::NonFiniteKernelValue);
            }

            // A well-conditioned factor through the same workspace still yields the closed-form n = 1 variance
            // k_** - k_*^2 / (1 + sigma_n^2) with k(0, 0) = 1 and k(0, 2) = exp(-2).
            const std::vector<Storage> sane{static_cast<Storage>(std::sqrt(1.0 + 0.01))};
            const cxxgpr::MatrixRef<Storage> sane_ref(sane.data(), 1U, 1U, 1U);
            const cxxgpr::PredictOptions latent{.variance_kind = cxxgpr::VarianceKind::Latent};
            const auto ok = cxxgpr::predict_into<P>(
                kernel, likelihood, x_train_ref, sane_ref, alpha, x_query_ref, latent, workspace, out
            );
            ASSERT_TRUE(ok.has_value());
            const double k_far = std::exp(-2.0);
            // float storage rounds the factor and kernel to about 6e-8 relative; 1e-6 covers it with margin.
            constexpr double tol = 1e-6;
            EXPECT_NEAR(static_cast<double>(out.variance[0]), 1.0 - (1.0 / 1.01), tol);
            EXPECT_NEAR(static_cast<double>(out.variance[1]), 1.0 - ((k_far * k_far) / 1.01), tol);
        };
        check_policy.template operator()<cxxgpr::DoublePrecision>();
        check_policy.template operator()<cxxgpr::SinglePrecision>();
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>();
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>();
    }

    /// @brief Verifies parallel predict_covariance_into() stays bit-identical across 1/2/4 threads for odd m.
    /// @details Evaluates DoublePrecision, SinglePrecision, and MixedPrecision<PromoteStorage> full covariance at
    /// n = 128, m = 67 and m = 200 under 1, 2, and 4 OpenMP threads, verifies exact bitwise equality across all mean
    /// and matrix elements, and checks accuracy against an independent naive K_** - V^T V reference.
    TEST(WorkspaceInferenceTest, PredictCovarianceBitwiseDeterministicAcrossThreads)
    {
        // Spec: DES-16, DES-20
        auto check_cov_policy = []<typename Policy>(std::size_t m)
        {
            using Storage = Policy::Storage;
            using Refine = Policy::Refine;
            constexpr std::size_t n = 128U;
            constexpr std::size_t d = 2U;
            std::vector<double> x_vals(n * d);
            std::vector<double> y_vals(n);
            for (std::size_t i = 0U; i < n; ++i)
            {
                const double x0 = -1.5 + (3.0 * static_cast<double>(i) / static_cast<double>(n - 1U));
                const double x1 = 0.8 * std::sin(0.19 * static_cast<double>(i));
                x_vals[i] = x0;
                x_vals[n + i] = x1;
                y_vals[i] = std::cos(1.3 * x0) + (0.4 * x1);
            }
            std::vector<Storage> q_vals(m * d);
            for (std::size_t j = 0U; j < m; ++j)
            {
                q_vals[j] = static_cast<Storage>(-1.4 + (2.8 * static_cast<double>(j) / static_cast<double>(m - 1U)));
                q_vals[m + j] = static_cast<Storage>(0.6 * std::cos(0.23 * static_cast<double>(j)));
            }
            const cxxgpr::MatrixRef<Storage> q_ref(q_vals.data(), m, d, m);

            const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.75).value())
                * cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.3).value());
            const auto kernel = spec.compile<Storage>();
            auto fix = make_fitted_fixture<Policy>(kernel, x_vals, y_vals, n, d, 0.25);

            auto run_threads = [&](int threads)
            {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#else
                (void)threads;
#endif
                cxxgpr::QueryWorkspace<Policy> qws;
                cxxgpr::PredictiveCovariance<Refine> out;
                EXPECT_TRUE(
                    cxxgpr::predict_covariance_into<Policy>(
                        kernel, fix.likelihood, fix.x_ref(), fix.l_ref(), fix.alpha, q_ref,
                        cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation}, qws, out
                    )
                        .has_value()
                );
                return out;
            };

#ifdef _OPENMP
            const int saved_threads = omp_get_max_threads();
#endif
            const auto c1 = run_threads(1);
            const auto c2 = run_threads(2);
            const auto c4 = run_threads(4);
#ifdef _OPENMP
            omp_set_num_threads(saved_threads);
#endif

            ASSERT_EQ(c1.mean.size(), m);
            ASSERT_EQ(c1.covariance.size(), m * m);
            for (std::size_t i = 0U; i < m; ++i)
            {
                if constexpr (std::same_as<Refine, double>)
                {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(c1.mean[i]), std::bit_cast<std::uint64_t>(c2.mean[i]));
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(c1.mean[i]), std::bit_cast<std::uint64_t>(c4.mean[i]));
                }
                else
                {
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(c1.mean[i]), std::bit_cast<std::uint32_t>(c2.mean[i]));
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(c1.mean[i]), std::bit_cast<std::uint32_t>(c4.mean[i]));
                }
            }
            for (std::size_t idx = 0U; idx < m * m; ++idx)
            {
                if constexpr (std::same_as<Refine, double>)
                {
                    EXPECT_EQ(
                        std::bit_cast<std::uint64_t>(c1.covariance[idx]),
                        std::bit_cast<std::uint64_t>(c2.covariance[idx])
                    );
                    EXPECT_EQ(
                        std::bit_cast<std::uint64_t>(c1.covariance[idx]),
                        std::bit_cast<std::uint64_t>(c4.covariance[idx])
                    );
                }
                else
                {
                    EXPECT_EQ(
                        std::bit_cast<std::uint32_t>(c1.covariance[idx]),
                        std::bit_cast<std::uint32_t>(c2.covariance[idx])
                    );
                    EXPECT_EQ(
                        std::bit_cast<std::uint32_t>(c1.covariance[idx]),
                        std::bit_cast<std::uint32_t>(c4.covariance[idx])
                    );
                }
            }

            // Compare against an explicit naive K_** - V^T V (+ sigma_n^2 I) oracle.
            std::vector<Storage> k_star(n * m, Storage{0});
            std::vector<Storage> k_star_scratch(n * m, Storage{0});
            auto nested_cross = kernel.make_nested_buffers(n, m);
            ASSERT_TRUE(kernel
                    .apply_cross_points(
                        fix.x_ref(), q_ref, cxxgpr::MatrixMut<Storage>(k_star.data(), n, m),
                        cxxgpr::MatrixMut<Storage>(k_star_scratch.data(), n, m), nested_cross
                    )
                    .has_value());
            ASSERT_TRUE(
                cxxgpr::linalg::solve_lower_in_place(fix.l_ref(), cxxgpr::MatrixMut<Storage>(k_star.data(), n, m))
                    .has_value()
            );

            std::vector<Storage> kss(m * m, Storage{0});
            std::vector<Storage> kss_scratch(m * m, Storage{0});
            auto nested_kss = kernel.make_nested_buffers(m, m);
            ASSERT_TRUE(kernel
                    .apply_points(
                        q_ref, cxxgpr::MatrixMut<Storage>(kss.data(), m, m), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<Storage>(kss_scratch.data(), m, m), nested_kss
                    )
                    .has_value());

            const double noise_var = fix.likelihood.noise_variance();
            double oracle_tol = 1e-11;
            if constexpr (std::same_as<Refine, float>)
            {
                oracle_tol = 1e-4;
            }
            else if constexpr (std::same_as<Storage, double>)
            {
                oracle_tol = 1e-12;
            }
            for (std::size_t col = 0U; col < m; ++col)
            {
                for (std::size_t row = col; row < m; ++row)
                {
                    double vtv = 0.0;
                    for (std::size_t r = 0U; r < n; ++r)
                    {
                        vtv += static_cast<double>(k_star[(row * n) + r]) * static_cast<double>(k_star[(col * n) + r]);
                    }
                    double ref_entry = static_cast<double>(kss[(col * m) + row]) - vtv;
                    if (row == col)
                    {
                        ref_entry += noise_var;
                    }
                    EXPECT_NEAR(c1.covariance[(col * m) + row], ref_entry, oracle_tol);
                    EXPECT_NEAR(c1.covariance[(row * m) + col], ref_entry, oracle_tol);
                }
            }
        };

        check_cov_policy.template operator()<cxxgpr::DoublePrecision>(67U);
        check_cov_policy.template operator()<cxxgpr::SinglePrecision>(67U);
        check_cov_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(67U);
        check_cov_policy.template operator()<cxxgpr::DoublePrecision>(200U);
        check_cov_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(200U);
    }

    /// @brief Verifies loo_predict and loo_predict_into reject subnormal Q_ii when 1 / Q_ii overflows to +Inf.
    /// @details Uses a 1x1 Cholesky factor whose inverse square Q_00 is positive and finite (IEEE 754 subnormal)
    /// so inv_diag_from_chol_l succeeds, yet 1 / Q_00 overflows to +Inf. Checks all four precision policies and
    /// both VarianceKind values return NonFiniteKernelValue, then confirms the buffers recover on a normal factor.
    TEST(WorkspaceInferenceTest, SubnormalQiiOverflowsLooVarianceToNonFiniteKernelValue)
    {
        // Spec: DES-02, DES-16
        const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();

        auto check_policy = [&]<typename P>(const char* name)
        {
            using Storage = typename P::Storage;
            using Refine = typename P::Refine;
            SCOPED_TRACE(name);

            // L_00 = 1e160 (double) or 1e20f (float) makes q_00 = 1/L_00^2 a finite positive subnormal in Storage.
            // For DoublePrecision and SinglePrecision, 1/q_00 overflows Refine; for MixedPrecision (Storage=float,
            // Refine=double), 1/q_00 ~ 1e40 is finite in double, so a large finite alpha overflows alpha/q_00 instead.
            Storage huge_l{};
            Refine large_alpha{0};
            if constexpr (std::same_as<Storage, float>)
            {
                huge_l = 1e20F;
                if constexpr (std::same_as<Refine, double>)
                {
                    large_alpha = 1e280;
                }
            }
            else
            {
                huge_l = 1e160;
            }
            const std::array<Storage, 1> l_subnormal{huge_l};
            const std::array<Refine, 1> alpha{large_alpha};
            const std::array<Refine, 1> y_train{Refine{1.5}};
            const cxxgpr::MatrixRef<Storage> l_sub_ref(l_subnormal.data(), 1U, 1U, 1U);

            std::array<Storage, 1> sc_mat{};
            std::array<Storage, 1> sc_diag{};
            cxxgpr::Prediction<Refine> out;

            for (const auto kind : {cxxgpr::VarianceKind::Observation, cxxgpr::VarianceKind::Latent})
            {
                const cxxgpr::PredictOptions opts{kind};
                const auto res_alloc = cxxgpr::loo_predict<P>(lik, l_sub_ref, alpha, y_train, opts);
                ASSERT_FALSE(res_alloc.has_value());
                EXPECT_EQ(res_alloc.error().code(), cxxgpr::GprErrorCode::NonFiniteKernelValue);

                const auto res_into = cxxgpr::loo_predict_into<P>(
                    lik, l_sub_ref, alpha, y_train, opts, cxxgpr::MatrixMut<Storage>(sc_mat.data(), 1U, 1U, 1U),
                    sc_diag, out
                );
                ASSERT_FALSE(res_into.has_value());
                EXPECT_EQ(res_into.error().code(), cxxgpr::GprErrorCode::NonFiniteKernelValue);
            }

            // Reusing the same buffers on a well-scaled factor L_00 = 2 (q_00 = 0.25, 1/q_00 = 4) succeeds.
            const std::array<Storage, 1> l_ok{Storage{2}};
            const std::array<Refine, 1> alpha_ok{Refine{0.5}};
            const cxxgpr::MatrixRef<Storage> l_ok_ref(l_ok.data(), 1U, 1U, 1U);
            const auto ok_res = cxxgpr::loo_predict_into<P>(
                lik, l_ok_ref, alpha_ok, y_train, cxxgpr::PredictOptions{cxxgpr::VarianceKind::Observation},
                cxxgpr::MatrixMut<Storage>(sc_mat.data(), 1U, 1U, 1U), sc_diag, out
            );
            ASSERT_TRUE(ok_res.has_value());
            ASSERT_EQ(out.mean.size(), 1U);
            ASSERT_EQ(out.variance.size(), 1U);
            // Closed-form: mu_LOO = 1.5 - 0.5 / 0.25 = -0.5, var_LOO = 1 / 0.25 = 4.0.
            EXPECT_NEAR(out.mean[0], -0.5, 1e-5);
            EXPECT_NEAR(out.variance[0], 4.0, 1e-5);
        };

        check_policy.template operator()<cxxgpr::DoublePrecision>("DoublePrecision");
        check_policy.template operator()<cxxgpr::SinglePrecision>("SinglePrecision");
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>("MixedPromote");
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>("MixedReevaluate");
    }

}  // namespace
