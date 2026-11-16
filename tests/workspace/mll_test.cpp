// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, REQ-08, DES-13, DES-15, DES-17, DES-20

#include "cxxgpr/workspace/workspace.hpp"

#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

TEST(WorkspaceMllTest, BufferSizingAndQueryWorkspaceReuse) {
    EXPECT_EQ(cxxgpr::WorkspaceCore<cxxgpr::DoublePrecision>::create(0U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(0U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);

    auto retain = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
        4U, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain);
    ASSERT_TRUE(retain.has_value());
    EXPECT_FALSE(retain->overwrites_cholesky());
    EXPECT_TRUE(retain->dist.has_value());
    EXPECT_EQ(retain->w_matrix.size(), 16U);

    auto reuse = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
        4U, cxxgpr::DistanceCachePolicy::Uncached, cxxgpr::CholeskyBuffer::Reuse);
    ASSERT_TRUE(reuse.has_value());
    EXPECT_TRUE(reuse->overwrites_cholesky());
    EXPECT_FALSE(reuse->dist.has_value());
    EXPECT_TRUE(reuse->w_matrix.empty());

    cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws;
    ASSERT_TRUE(qws.ensure(4U, 3U, 2U, 1U).has_value());
    {
        const cxxgpr::test_support::NoAllocationScope audit;
        ASSERT_TRUE(qws.ensure(4U, 3U, 2U, 1U).has_value());
        ASSERT_TRUE(qws.ensure(4U, 2U, 2U, 1U).has_value());
    }
}

TEST(WorkspaceMllTest, KeepPlanMllAndAnalyticalGradientMatchFiniteDifferences) {
    const std::vector<double> x_data{0.0, 0.4, 0.9, 1.4, -0.3, 0.2, 0.8, 1.5, -0.6, 0.1};
    const std::vector<double> y{0.5, 1.1, 0.8, -0.2, 0.3};
    constexpr std::size_t n = 5U;
    const cxxgpr::MatrixRef<double> x(x_data.data(), n, 2U, n);

    const auto spec =
        (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.4).value()) *
         cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()) *
         cxxgpr::KernelSpec(cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.3},
                                                            cxxgpr::MaternNu::ThreeHalves)
                                .value())) +
        (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(0.7).value()) *
         cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.8, 2.2).value())) +
        (cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.3).value()) *
         cxxgpr::KernelSpec(
             cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.2, 0.8}, 1.7)
                 .value())) +
        cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

    const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

    const auto single_prod_spec =
        (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.4).value()) *
         cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()) *
         cxxgpr::KernelSpec(cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.3},
                                                            cxxgpr::MaternNu::ThreeHalves)
                                .value())) +
        cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
    EXPECT_GE(single_prod_spec.compile<double>().kept_products(), 1U);

    for (const auto cache_pol :
         {cxxgpr::DistanceCachePolicy::Cached, cxxgpr::DistanceCachePolicy::Uncached}) {
        for (const auto buf_pol : {cxxgpr::CholeskyBuffer::Retain, cxxgpr::CholeskyBuffer::Reuse}) {
            auto compiled = spec.compile<double>();
            compiled.set_kernel_exp(cxxgpr::KernelExp::Accurate);

            auto lik = cxxgpr::GaussianLikelihood::create(0.08).value();
            auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cache_pol, buf_pol);
            ASSERT_TRUE(bufs.has_value());
            if (!bufs.has_value()) {
                return;
            }

            const std::size_t n_kernel = compiled.num_params();
            const std::size_t total_params = n_kernel + 1U;
            std::vector<double> theta(total_params);
            ASSERT_TRUE(compiled.get_params(std::span<double>(theta.data(), n_kernel)).has_value());
            theta[n_kernel] = std::log(lik.noise_variance());

            std::vector<double> grad(total_params, 0.0);
            auto mll_res = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, grad);
            ASSERT_TRUE(mll_res.has_value());

            // Subsequent calls must allocate zero heap memory.
            {
                const cxxgpr::test_support::NoAllocationScope audit;
                auto mll_again =
                    cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, grad);
                ASSERT_TRUE(mll_again.has_value());
                EXPECT_DOUBLE_EQ(*mll_again, *mll_res);
            }

            // In Reuse mode, L on k_matrix must be restored after mll_and_gradient_into.
            const std::vector<double> l_diag_after{bufs->core.k_ref()(0, 0),
                                                   bufs->core.k_ref()(1, 1)};
            EXPECT_GT(l_diag_after[0], 0.0);
            EXPECT_GT(l_diag_after[1], 0.0);

            constexpr double h = 1e-6;
            for (std::size_t p = 0; p < total_params; ++p) {
                auto tp = theta;
                tp[p] += h;
                if (p < n_kernel) {
                    ASSERT_TRUE(compiled.set_params(std::span<const double>(tp.data(), n_kernel))
                                    .has_value());
                } else {
                    lik = cxxgpr::GaussianLikelihood::create(std::exp(tp[p])).value();
                }
                auto mll_plus = cxxgpr::factor_and_mll(compiled, lik, x, y, jitter, *bufs);
                ASSERT_TRUE(mll_plus.has_value());

                auto tm = theta;
                tm[p] -= h;
                if (p < n_kernel) {
                    ASSERT_TRUE(compiled.set_params(std::span<const double>(tm.data(), n_kernel))
                                    .has_value());
                } else {
                    lik = cxxgpr::GaussianLikelihood::create(std::exp(tm[p])).value();
                }
                auto mll_minus = cxxgpr::factor_and_mll(compiled, lik, x, y, jitter, *bufs);
                ASSERT_TRUE(mll_minus.has_value());

                // Restore original theta & likelihood.
                if (p < n_kernel) {
                    ASSERT_TRUE(compiled.set_params(std::span<const double>(theta.data(), n_kernel))
                                    .has_value());
                } else {
                    lik = cxxgpr::GaussianLikelihood::create(std::exp(theta[p])).value();
                }

                const double fd = (*mll_plus - *mll_minus) / (2.0 * h);
                EXPECT_NEAR(grad[p], fd, 1e-5 * std::max(1.0, std::abs(fd))) << "p=" << p;
            }
        }
    }
}

TEST(WorkspaceMllTest, SingleAndMixedPrecisionSupportMllAndGradient) {
    const std::vector<float> x_data{0.0F, 0.6F, 1.3F, -0.2F, 0.5F, 1.1F};
    const std::vector<float> y{0.4F, 1.0F, 0.2F};
    constexpr std::size_t n = 3U;
    const cxxgpr::MatrixRef<float> x(x_data.data(), n, 2U, n);

    const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value()) +
                      cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.1).value());
    const auto compiled = spec.compile<float>();
    const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
    const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

    auto bufs_single = cxxgpr::FitBuffers<cxxgpr::SinglePrecision>::create(n);
    auto bufs_mixed = cxxgpr::FitBuffers<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>::create(n);
    ASSERT_TRUE(bufs_single.has_value() && bufs_mixed.has_value());

    std::array<double, 3> g_single{};
    std::array<double, 3> g_mixed{};
    auto m_single =
        cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs_single, g_single);
    auto m_mixed = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs_mixed, g_mixed);
    ASSERT_TRUE(m_single.has_value() && m_mixed.has_value());
    EXPECT_DOUBLE_EQ(*m_single, *m_mixed);
    for (std::size_t i = 0; i < 3U; ++i) {
        EXPECT_DOUBLE_EQ(g_single[i], g_mixed[i]);
    }
}

TEST(WorkspaceMllTest, MllAndGradientAreBitwiseDeterministicAcrossOpenMpThreads) {
    constexpr std::size_t n = 72U;
    constexpr std::size_t d = 2U;
    std::vector<double> x_data(n * d);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = 0.08 * static_cast<double>(i);
        x_data[i] = t;
        x_data[n + i] = std::cos(t);
        y[i] = std::sin(t);
    }
    const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

    const auto spec =
        (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value()) *
         cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()) *
         cxxgpr::KernelSpec(cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.4},
                                                            cxxgpr::MaternNu::ThreeHalves)
                                .value())) +
        cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
    auto compiled = spec.compile<double>();
    compiled.set_kernel_exp(cxxgpr::KernelExp::FastApprox);
    const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
    const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

    auto run_threads = [&](int threads) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
            n, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain);
        EXPECT_TRUE(bufs.has_value());
        std::vector<double> grad(compiled.num_params() + 1U, 0.0);
        auto mll = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, grad);
        EXPECT_TRUE(mll.has_value());
        grad.push_back(*mll);
        return grad;
    };

#ifdef _OPENMP
    const int saved = omp_get_max_threads();
#endif
    const auto r1 = run_threads(1);
    const auto r4 = run_threads(4);
#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif

    ASSERT_EQ(r1.size(), r4.size());
    for (std::size_t i = 0; i < r1.size(); ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(r1[i]), std::bit_cast<std::uint64_t>(r4[i]))
            << "i=" << i;
    }
}

TEST(WorkspaceMllTest, AnalyticalHessianMatchesFiniteDiffAndZeroAllocForSpeedAndMemoryPolicies) {
    const std::vector<double> x_data{0.0, 0.5, 1.1, 1.7, -0.4, 0.2, 0.9, 1.4};
    const std::vector<double> y{0.4, 1.0, 0.7, -0.3};
    constexpr std::size_t n = 4U;
    const cxxgpr::MatrixRef<double> x(x_data.data(), n, 2U, n);

    const auto spec =
        (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.3).value()) *
         cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())) +
        cxxgpr::KernelSpec(cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.4},
                                                           cxxgpr::MaternNu::ThreeHalves)
                               .value()) +
        cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

    const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

    for (const auto policy :
         {cxxgpr::FitPolicy::prefer_speed(), cxxgpr::FitPolicy::prefer_memory()}) {
        auto compiled = spec.compile<double>();
        auto lik = cxxgpr::GaussianLikelihood::create(0.08).value();
        auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, policy);
        ASSERT_TRUE(bufs.has_value());
        if (!bufs.has_value()) {
            return;
        }
        EXPECT_FALSE(bufs->core.hessian.is_allocated());

        const std::size_t n_kernel = compiled.num_params();
        const std::size_t total_params = n_kernel + 1U;
        std::vector<double> theta(total_params);
        ASSERT_TRUE(compiled.get_params(std::span<double>(theta.data(), n_kernel)).has_value());
        theta[n_kernel] = std::log(lik.noise_variance());

        std::vector<double> grad(total_params, 0.0);
        std::vector<double> hess(total_params * total_params, 0.0);

        auto mll_res =
            cxxgpr::value_gradient_hessian_into(compiled, lik, x, y, jitter, *bufs, grad, hess);
        ASSERT_TRUE(mll_res.has_value());
        EXPECT_TRUE(bufs->core.hessian.is_allocated());

        // Subsequent hessian_into and value_gradient_hessian_into must allocate zero heap memory.
        std::vector<double> hess2(total_params * total_params, 0.0);
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(
                cxxgpr::hessian_into(compiled, lik, x, y, jitter, *bufs, hess2).has_value());
            auto mll2 = cxxgpr::value_gradient_hessian_into(compiled, lik, x, y, jitter, *bufs,
                                                            grad, hess2);
            ASSERT_TRUE(mll2.has_value());
            EXPECT_DOUBLE_EQ(*mll2, *mll_res);
        }

        for (std::size_t idx = 0; idx < hess.size(); ++idx) {
            EXPECT_DOUBLE_EQ(hess2[idx], hess[idx]);
        }

        // Verify symmetry and finite-difference match against gradient.
        constexpr double h = 1e-6;
        std::vector<double> g_plus(total_params);
        std::vector<double> g_minus(total_params);
        for (std::size_t j = 0; j < total_params; ++j) {
            auto tp = theta;
            tp[j] += h;
            if (j < n_kernel) {
                ASSERT_TRUE(
                    compiled.set_params(std::span<const double>(tp.data(), n_kernel)).has_value());
            } else {
                lik = cxxgpr::GaussianLikelihood::create(std::exp(tp[j])).value();
            }
            ASSERT_TRUE(cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, g_plus)
                            .has_value());

            auto tm = theta;
            tm[j] -= h;
            if (j < n_kernel) {
                ASSERT_TRUE(
                    compiled.set_params(std::span<const double>(tm.data(), n_kernel)).has_value());
            } else {
                lik = cxxgpr::GaussianLikelihood::create(std::exp(tm[j])).value();
            }
            ASSERT_TRUE(cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, g_minus)
                            .has_value());

            if (j < n_kernel) {
                ASSERT_TRUE(compiled.set_params(std::span<const double>(theta.data(), n_kernel))
                                .has_value());
            } else {
                lik = cxxgpr::GaussianLikelihood::create(std::exp(theta[j])).value();
            }

            for (std::size_t i = 0; i < total_params; ++i) {
                EXPECT_DOUBLE_EQ(hess[(i * total_params) + j], hess[(j * total_params) + i]);
                const double fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
                EXPECT_NEAR(hess[(i * total_params) + j], fd, 1e-5 * std::max(1.0, std::abs(fd)))
                    << "i=" << i << ", j=" << j;
            }
        }
    }
}

TEST(WorkspaceMllTest, AnalyticalHessianIsBitwiseDeterministicAcrossOpenMpThreads) {
    constexpr std::size_t n = 72U;
    constexpr std::size_t d = 2U;
    std::vector<double> x_data(n * d);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = 0.08 * static_cast<double>(i);
        x_data[i] = t;
        x_data[n + i] = std::cos(t);
        y[i] = std::sin(t);
    }
    const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

    const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()) +
                      cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
    const auto compiled = spec.compile<double>();
    const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
    const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

    auto run_threads = [&](int threads) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
            n, cxxgpr::FitPolicy::prefer_speed());
        EXPECT_TRUE(bufs.has_value());
        const std::size_t p = compiled.num_params() + 1U;
        std::vector<double> hess(p * p, 0.0);
        EXPECT_TRUE(cxxgpr::hessian_into(compiled, lik, x, y, jitter, *bufs, hess).has_value());
        return hess;
    };

#ifdef _OPENMP
    const int saved = omp_get_max_threads();
#endif
    const auto h1 = run_threads(1);
    const auto h4 = run_threads(4);
#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif

    ASSERT_EQ(h1.size(), h4.size());
    for (std::size_t i = 0; i < h1.size(); ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(h1[i]), std::bit_cast<std::uint64_t>(h4[i]))
            << "i=" << i;
    }
}

} // namespace
