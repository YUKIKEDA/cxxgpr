// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-08, REQ-11, DES-16, DES-17, DES-20

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

    template <typename P> struct FittedFixture
    {
        using Storage = P::Storage;
        using Refine = P::Refine;

        cxxgpr::GaussianLikelihood likelihood;
        std::vector<Storage> x_train;
        std::vector<Storage> y_storage;
        std::vector<Refine> y_refine;
        std::vector<Storage> l_factor;
        std::vector<Refine> alpha;
        std::size_t n = 0;
        std::size_t d = 0;

        [[nodiscard]] cxxgpr::MatrixRef<Storage> x_ref() const noexcept
        {
            return {x_train.data(), n, d, d == 0 ? 1 : n};
        }

        [[nodiscard]] cxxgpr::MatrixRef<Storage> l_ref() const noexcept { return {l_factor.data(), n, n, n}; }
    };

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

    TEST(WorkspaceInferenceTest, NegativeControlsRejectInvalidInputs)
    {
        // Spec: DES-16
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const std::vector<double> x_vals = {0.0, 1.0, 0.5, -0.5};
        const std::vector<double> y_vals = {1.0, -1.0};
        const auto kernel = cxxgpr::CompiledKernel<double>::from_spec(spec);
        auto fix = make_fitted_fixture<cxxgpr::DoublePrecision>(kernel, x_vals, y_vals, 2U, 2U, 0.1);
        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws;

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

}  // namespace
