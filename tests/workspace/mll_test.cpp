// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, REQ-08, DES-13, DES-15, DES-17, DES-20
/// @file mll_test.cpp
/// @brief Unit and property tests for MLL, analytical gradients, Hessians, and FitBuffers policies.
/// @details Verifies workspace buffer sizing and QueryWorkspace zero-alloc reuse, KeepPlan
/// log-marginal-likelihood and analytical gradient/Hessian parity against central finite
/// differences across cache and Cholesky buffer policies, Single/MixedPrecision parity, and
/// bitwise determinism across OpenMP thread counts.

#include "cxxgpr/core/build_info.hpp"
#include "cxxgpr/workspace/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "allocation_audit.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace
{

    /// @brief Returns true when the compiled library links an external BLAS backend (OpenBLAS or MKL).
    /// @return True if `build_info().dependencies` reports a non-Eigen `blas_backend`, false otherwise.
    [[nodiscard]] bool has_external_blas_backend() noexcept
    {
        const auto info = cxxgpr::build_info();
        for (const auto& dep : info.dependencies)
        {
            if (dep.name == "blas_backend")
            {
                return dep.version != "Eigen";
            }
        }
        return false;
    }

    /// @brief Verifies FitBuffers/WorkspaceCore sizing policies and QueryWorkspace zero-alloc reuse.
    /// @details Checks EmptyInput on zero size, Retain vs Reuse W-matrix and distance buffer flags,
    /// and zero heap allocations on QueryWorkspace::ensure() when capacity is already sufficient.
    TEST(WorkspaceMllTest, BufferSizingAndQueryWorkspaceReuse)
    {
        // Reject zero training sample count before allocating any workspace buffers.
        EXPECT_EQ(
            cxxgpr::WorkspaceCore<cxxgpr::DoublePrecision>::create(0U).error().code(), cxxgpr::GprErrorCode::EmptyInput
        );
        EXPECT_EQ(
            cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(0U).error().code(), cxxgpr::GprErrorCode::EmptyInput
        );

        auto retain = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
            4U, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain
        );
        ASSERT_TRUE(retain.has_value());
        EXPECT_FALSE(retain->overwrites_cholesky());
        EXPECT_TRUE(retain->dist.has_value());
        EXPECT_EQ(retain->w_matrix.size(), 16U);

        auto reuse = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
            4U, cxxgpr::DistanceCachePolicy::Uncached, cxxgpr::CholeskyBuffer::Reuse
        );
        ASSERT_TRUE(reuse.has_value());
        EXPECT_TRUE(reuse->overwrites_cholesky());
        EXPECT_FALSE(reuse->dist.has_value());
        EXPECT_TRUE(reuse->w_matrix.empty());

        cxxgpr::QueryWorkspace<cxxgpr::DoublePrecision> qws;
        ASSERT_TRUE(qws.ensure(4U, 3U, 2U, 1U).has_value());
        EXPECT_EQ(qws.query_k_star.size(), 12U);
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(qws.ensure(4U, 3U, 2U, 1U).has_value());
            ASSERT_TRUE(qws.ensure(4U, 2U, 2U, 1U).has_value());
            // Re-growing to an equal or smaller shape must not touch the heap.
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }
    }

    /// @brief Verifies KeepPlan MLL and analytical gradients match finite differences with zero alloc.
    /// @details Sweeps all four combinations of DistanceCachePolicy and CholeskyBuffer across a
    /// multi-product composite kernel, verifying factor restoration in Reuse mode, zero-alloc
    /// re-evaluation under NoAllocationScope, and gradient parity against central finite differences.
    TEST(WorkspaceMllTest, KeepPlanMllAndAnalyticalGradientMatchFiniteDifferences)
    {
        const std::vector<double> x_data{0.0, 0.4, 0.9, 1.4, -0.3, 0.2, 0.8, 1.5, -0.6, 0.1};
        const std::vector<double> y{0.5, 1.1, 0.8, -0.2, 0.3};
        constexpr std::size_t n = 5U;
        const cxxgpr::MatrixRef<double> x(x_data.data(), n, 2U, n);

        // Multi-product sum kernel triggering KeepPlan intermediate product caching alongside White noise.
        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.4).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.3}, cxxgpr::MaternNu::ThreeHalves)
                        .value()
                ))
            + (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(0.7).value())
                * cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.8, 2.2).value()))
            + (cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.3).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.2, 0.8}, 1.7).value()
                ))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

        const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

        const auto single_prod_spec =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.4).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.3}, cxxgpr::MaternNu::ThreeHalves)
                        .value()
                ))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
        EXPECT_GE(single_prod_spec.compile<double>().kept_products(), 1U);

        for (const auto cache_pol : {cxxgpr::DistanceCachePolicy::Cached, cxxgpr::DistanceCachePolicy::Uncached})
        {
            for (const auto buf_pol : {cxxgpr::CholeskyBuffer::Retain, cxxgpr::CholeskyBuffer::Reuse})
            {
                auto compiled = spec.compile<double>();
                compiled.set_kernel_exp(cxxgpr::KernelExp::Accurate);

                auto lik = cxxgpr::GaussianLikelihood::create(0.08).value();
                auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cache_pol, buf_pol);
                ASSERT_TRUE(bufs.has_value());
                if (!bufs.has_value())
                {
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
                    auto mll_again = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, grad);
                    ASSERT_TRUE(mll_again.has_value());
                    EXPECT_DOUBLE_EQ(*mll_again, *mll_res);
                }

                // In Reuse mode, L on k_matrix must be restored after mll_and_gradient_into.
                const std::vector<double> l_diag_after{bufs->core.k_ref()(0, 0), bufs->core.k_ref()(1, 1)};
                EXPECT_GT(l_diag_after[0], 0.0);
                EXPECT_GT(l_diag_after[1], 0.0);

                constexpr double h = 1e-6;
                for (std::size_t p = 0; p < total_params; ++p)
                {
                    auto tp = theta;
                    tp[p] += h;
                    if (p < n_kernel)
                    {
                        ASSERT_TRUE(compiled.set_params(std::span<const double>(tp.data(), n_kernel)).has_value());
                    }
                    else
                    {
                        lik = cxxgpr::GaussianLikelihood::create(std::exp(tp[p])).value();
                    }
                    auto mll_plus = cxxgpr::factor_and_mll(compiled, lik, x, y, jitter, *bufs);
                    ASSERT_TRUE(mll_plus.has_value());

                    auto tm = theta;
                    tm[p] -= h;
                    if (p < n_kernel)
                    {
                        ASSERT_TRUE(compiled.set_params(std::span<const double>(tm.data(), n_kernel)).has_value());
                    }
                    else
                    {
                        lik = cxxgpr::GaussianLikelihood::create(std::exp(tm[p])).value();
                    }
                    auto mll_minus = cxxgpr::factor_and_mll(compiled, lik, x, y, jitter, *bufs);
                    ASSERT_TRUE(mll_minus.has_value());

                    // Restore original theta & likelihood.
                    if (p < n_kernel)
                    {
                        ASSERT_TRUE(compiled.set_params(std::span<const double>(theta.data(), n_kernel)).has_value());
                    }
                    else
                    {
                        lik = cxxgpr::GaussianLikelihood::create(std::exp(theta[p])).value();
                    }

                    const double fd = (*mll_plus - *mll_minus) / (2.0 * h);
                    EXPECT_NEAR(grad[p], fd, 1e-5 * std::max(1.0, std::abs(fd))) << "p=" << p;
                }
            }
        }
    }

    /// @brief Verifies SinglePrecision and MixedPrecision<PromoteStorage> produce identical MLL and gradients.
    /// @details Confirms both policies execute mll_and_gradient_into() on identical float training
    /// data and agree bitwise on MLL and parameter gradient entries when float solves match.
    TEST(WorkspaceMllTest, SingleAndMixedPrecisionSupportMllAndGradient)
    {
        // Evaluate MLL and gradient in float storage under SinglePrecision and MixedPrecision<PromoteStorage>.
        const std::vector<float> x_data{0.0F, 0.6F, 1.3F, -0.2F, 0.5F, 1.1F};
        const std::vector<float> y{0.4F, 1.0F, 0.2F};
        constexpr std::size_t n = 3U;
        const cxxgpr::MatrixRef<float> x(x_data.data(), n, 2U, n);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.1).value());
        const auto compiled = spec.compile<float>();
        const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
        const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

        auto bufs_single = cxxgpr::FitBuffers<cxxgpr::SinglePrecision>::create(n);
        auto bufs_mixed = cxxgpr::FitBuffers<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>::create(n);
        ASSERT_TRUE(bufs_single.has_value() && bufs_mixed.has_value());

        std::array<double, 3> g_single{};
        std::array<double, 3> g_mixed{};
        auto m_single = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs_single, g_single);
        auto m_mixed = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs_mixed, g_mixed);
        ASSERT_TRUE(m_single.has_value() && m_mixed.has_value());
        EXPECT_DOUBLE_EQ(*m_single, *m_mixed);
        for (std::size_t i = 0; i < 3U; ++i)
        {
            EXPECT_DOUBLE_EQ(g_single[i], g_mixed[i]);
        }
    }

    /// @brief Verifies MLL and analytical gradients stay deterministic on same-thread repeats and agree across threads.
    /// @details Compares std::bit_cast<uint64_t> bit patterns for repeated 4-thread runs (and 1 vs 4 threads under
    /// internal Eigen linear algebra, or scaled rounding tolerance when an external vendor BLAS is linked).
    TEST(WorkspaceMllTest, MllAndGradientAreBitwiseDeterministicAcrossOpenMpThreads)
    {
        // Construct an n = 72 2D synthetic dataset exercising OpenMP column reductions.
        constexpr std::size_t n = 72U;
        constexpr std::size_t d = 2U;
        std::vector<double> x_data(n * d);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double t = 0.08 * static_cast<double>(i);
            x_data[i] = t;
            x_data[n + i] = std::cos(t);
            y[i] = std::sin(t);
        }
        const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.4}, cxxgpr::MaternNu::ThreeHalves)
                        .value()
                ))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
        auto compiled = spec.compile<double>();
        compiled.set_kernel_exp(cxxgpr::KernelExp::FastApprox);
        const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
        const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

        auto run_threads = [&](int threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
                n, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain
            );
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
        const auto r4_repeat = run_threads(4);
#ifdef _OPENMP
        omp_set_num_threads(saved);
#endif

        ASSERT_EQ(r1.size(), r4.size());
        ASSERT_EQ(r4.size(), r4_repeat.size());
        const bool ext_blas = has_external_blas_backend();
        for (std::size_t i = 0; i < r1.size(); ++i)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(r4[i]), std::bit_cast<std::uint64_t>(r4_repeat[i])) << "i=" << i;
            if (!ext_blas)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(r1[i]), std::bit_cast<std::uint64_t>(r4[i])) << "i=" << i;
            }
            else
            {
                EXPECT_NEAR(r1[i], r4[i], 1e-11 * std::max(1.0, std::abs(r1[i]))) << "i=" << i;
            }
        }
    }

    /// @brief Verifies analytical MLL Hessian matches finite differences with lazy zero-alloc buffers.
    /// @details Checks lazy HessianWorkspace allocation for both prefer_speed() and prefer_memory(),
    /// audits zero-allocation subsequent calls under NoAllocationScope, and confirms symmetry and
    /// finite-difference parity against mll_and_gradient_into().
    TEST(WorkspaceMllTest, AnalyticalHessianMatchesFiniteDiffAndZeroAllocForSpeedAndMemoryPolicies)
    {
        // Use a composite Constant * RBF + MaternARD + White system across speed and memory policies.
        const std::vector<double> x_data{0.0, 0.5, 1.1, 1.7, -0.4, 0.2, 0.9, 1.4};
        const std::vector<double> y{0.4, 1.0, 0.7, -0.3};
        constexpr std::size_t n = 4U;
        const cxxgpr::MatrixRef<double> x(x_data.data(), n, 2U, n);

        const auto spec = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.3).value())
                              * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()))
            + cxxgpr::KernelSpec(
                cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.4}, cxxgpr::MaternNu::ThreeHalves).value()
            )
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

        const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

        for (const auto policy : {cxxgpr::FitPolicy::prefer_speed(), cxxgpr::FitPolicy::prefer_memory()})
        {
            auto compiled = spec.compile<double>();
            auto lik = cxxgpr::GaussianLikelihood::create(0.08).value();
            auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, policy);
            ASSERT_TRUE(bufs.has_value());
            if (!bufs.has_value())
            {
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

            auto mll_res = cxxgpr::value_gradient_hessian_into(compiled, lik, x, y, jitter, *bufs, grad, hess);
            ASSERT_TRUE(mll_res.has_value());
            EXPECT_TRUE(bufs->core.hessian.is_allocated());

            // Subsequent hessian_into and value_gradient_hessian_into must allocate zero heap memory.
            std::vector<double> hess2(total_params * total_params, 0.0);
            {
                const cxxgpr::test_support::NoAllocationScope audit;
                ASSERT_TRUE(cxxgpr::hessian_into(compiled, lik, x, y, jitter, *bufs, hess2).has_value());
                EXPECT_EQ(hess2, hess);
                auto mll2 = cxxgpr::value_gradient_hessian_into(compiled, lik, x, y, jitter, *bufs, grad, hess2);
                ASSERT_TRUE(mll2.has_value());
                EXPECT_DOUBLE_EQ(*mll2, *mll_res);
            }

            for (std::size_t idx = 0; idx < hess.size(); ++idx)
            {
                EXPECT_DOUBLE_EQ(hess2[idx], hess[idx]);
            }

            // Verify symmetry and finite-difference match against gradient.
            constexpr double h = 1e-6;
            std::vector<double> g_plus(total_params);
            std::vector<double> g_minus(total_params);
            for (std::size_t j = 0; j < total_params; ++j)
            {
                auto tp = theta;
                tp[j] += h;
                if (j < n_kernel)
                {
                    ASSERT_TRUE(compiled.set_params(std::span<const double>(tp.data(), n_kernel)).has_value());
                }
                else
                {
                    lik = cxxgpr::GaussianLikelihood::create(std::exp(tp[j])).value();
                }
                ASSERT_TRUE(cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, g_plus).has_value());

                auto tm = theta;
                tm[j] -= h;
                if (j < n_kernel)
                {
                    ASSERT_TRUE(compiled.set_params(std::span<const double>(tm.data(), n_kernel)).has_value());
                }
                else
                {
                    lik = cxxgpr::GaussianLikelihood::create(std::exp(tm[j])).value();
                }
                ASSERT_TRUE(cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, g_minus).has_value());

                if (j < n_kernel)
                {
                    ASSERT_TRUE(compiled.set_params(std::span<const double>(theta.data(), n_kernel)).has_value());
                }
                else
                {
                    lik = cxxgpr::GaussianLikelihood::create(std::exp(theta[j])).value();
                }

                for (std::size_t i = 0; i < total_params; ++i)
                {
                    EXPECT_DOUBLE_EQ(hess[(i * total_params) + j], hess[(j * total_params) + i]);
                    const double fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
                    EXPECT_NEAR(hess[(i * total_params) + j], fd, 1e-5 * std::max(1.0, std::abs(fd)))
                        << "i=" << i << ", j=" << j;
                }
            }
        }
    }

    /// @brief Verifies analytical MLL Hessian entries stay deterministic on repeats and agree across threads.
    /// @details Compares std::bit_cast<uint64_t> bit patterns for repeated 4-thread runs (and 1 vs 4 threads under
    /// internal Eigen linear algebra, or scaled rounding tolerance when external vendor BLAS is linked) on n = 72.
    TEST(WorkspaceMllTest, AnalyticalHessianIsBitwiseDeterministicAcrossOpenMpThreads)
    {
        // Evaluate full Hessian on n = 72 points under 1 and 4 threads and compare raw bits.
        constexpr std::size_t n = 72U;
        constexpr std::size_t d = 2U;
        std::vector<double> x_data(n * d);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double t = 0.08 * static_cast<double>(i);
            x_data[i] = t;
            x_data[n + i] = std::cos(t);
            y[i] = std::sin(t);
        }
        const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
        const auto compiled = spec.compile<double>();
        const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
        const auto jitter = cxxgpr::JitterPolicy::adaptive(1e-8, 10.0, 6U, 1.0).value();

        auto run_threads = [&](int threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cxxgpr::FitPolicy::prefer_speed());
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
        const auto h4_repeat = run_threads(4);
#ifdef _OPENMP
        omp_set_num_threads(saved);
#endif

        ASSERT_EQ(h1.size(), h4.size());
        ASSERT_EQ(h4.size(), h4_repeat.size());
        const bool ext_blas = has_external_blas_backend();
        for (std::size_t i = 0; i < h1.size(); ++i)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(h4[i]), std::bit_cast<std::uint64_t>(h4_repeat[i])) << "i=" << i;
            if (!ext_blas)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(h1[i]), std::bit_cast<std::uint64_t>(h4[i])) << "i=" << i;
            }
            else
            {
                EXPECT_NEAR(h1[i], h4[i], 1e-10 * std::max(1.0, std::abs(h1[i]))) << "i=" << i;
            }
        }
    }

    /// @brief Verifies 1-pass isotropic/linear adjoint contraction (n >= 128) and DistanceKernel non-zero diag parity.
    /// @details Confirms analytical gradients match central finite differences, stay bitwise deterministic across
    /// repeated calls and serial-vs-parallel col_partials (plus 1 vs 4 threads under Eigen, or 1e-11 relative tolerance
    /// under external BLAS), allocate zero heap bytes in reuse calls, and handle positive DistanceKernel diagonals
    /// cleanly.
    TEST(WorkspaceMllTest, IsotropicOnePassWeightedGradsLargeNAndNonZeroDistDiag)
    {
        // Case 1: n = 132 (>= 128 parallel threshold) across all 5 leaf_iso_weighted_grads kernel families (1D Periodic
        // is PSD).
        {
            constexpr std::size_t n = 132U;
            constexpr std::size_t d = 1U;
            std::vector<double> x_data(n * d);
            std::vector<double> y(n);
            for (std::size_t i = 0U; i < n; ++i)
            {
                x_data[i] = 0.08 * static_cast<double>(i);
                y[i] = std::sin(0.15 * static_cast<double>(i)) + (0.05 * static_cast<double>(i % 7U));
            }
            const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

            const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
                + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.3, cxxgpr::MaternNu::ThreeHalves).value())
                + cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.9, 2.4).value())
                + cxxgpr::KernelSpec(cxxgpr::RationalQuadraticKernel::create(1.2, 1.7).value())
                + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.7).value())
                + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

            const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
            const auto jitter = cxxgpr::JitterPolicy::fixed(0.0).value();

            auto run_threads = [&](int threads)
            {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#else
                (void)threads;
#endif
                auto compiled = spec.compile<double>();
                auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cxxgpr::FitPolicy::prefer_speed());
                EXPECT_TRUE(bufs.has_value());
                const std::size_t total_params = compiled.num_params() + 1U;
                std::vector<double> grad(total_params, 0.0);
                std::vector<double> grad_again(total_params, 0.0);
                const auto res1 = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, grad);
                EXPECT_TRUE(res1.has_value()) << res1.error().message();
                {
                    const cxxgpr::test_support::NoAllocationScope audit;
                    const auto res2 = cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs, grad_again);
                    EXPECT_TRUE(res2.has_value()) << res2.error().message();
                    EXPECT_EQ(audit.delta_allocations(), 0U);
                }
                return grad;
            };

#ifdef _OPENMP
            const int saved = omp_get_max_threads();
#endif
            const auto g1 = run_threads(1);
            const auto g4 = run_threads(4);
            const auto g4_repeat = run_threads(4);
#ifdef _OPENMP
            omp_set_num_threads(saved);
#endif
            ASSERT_EQ(g1.size(), g4.size());
            ASSERT_EQ(g4.size(), g4_repeat.size());
            const bool ext_blas = has_external_blas_backend();
            for (std::size_t p = 0U; p < g1.size(); ++p)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(g4[p]), std::bit_cast<std::uint64_t>(g4_repeat[p])) << "p=" << p;
                if (!ext_blas)
                {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(g1[p]), std::bit_cast<std::uint64_t>(g4[p])) << "p=" << p;
                }
                else
                {
                    EXPECT_NEAR(g1[p], g4[p], 1e-11 * std::max(1.0, std::abs(g1[p]))) << "p=" << p;
                }
            }

            // Finite-difference check on representative parameters across each isotropic/linear family.
            auto compiled = spec.compile<double>();
            auto bufs = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cxxgpr::FitPolicy::prefer_speed());
            ASSERT_TRUE(bufs.has_value());
            const std::size_t n_kernel = compiled.num_params();
            std::vector<double> theta(n_kernel);
            ASSERT_TRUE(compiled.get_params(theta).has_value());
            constexpr double h = 1e-6;
            for (std::size_t p = 0U; p < n_kernel; ++p)
            {
                auto tp = theta;
                tp[p] += h;
                ASSERT_TRUE(compiled.set_params(tp).has_value());
                auto mp = cxxgpr::factor_and_mll(compiled, lik, x, y, jitter, *bufs);
                ASSERT_TRUE(mp.has_value());
                auto tm = theta;
                tm[p] -= h;
                ASSERT_TRUE(compiled.set_params(tm).has_value());
                auto mm = cxxgpr::factor_and_mll(compiled, lik, x, y, jitter, *bufs);
                ASSERT_TRUE(mm.has_value());
                const double fd = (*mp - *mm) / (2.0 * h);
                EXPECT_NEAR(g1[p], fd, 1e-5 * std::max(1.0, std::abs(g1[p]))) << "p=" << p;
            }
        }

        // Case 2: Uncached (FitPolicy::prefer_memory) leaf_iso_weighted_grads parity against Cached and serial
        // col_partials (< 3 * n) fallback.
        {
            constexpr std::size_t n = 132U;
            constexpr std::size_t d = 1U;
            std::vector<double> x_data(n * d);
            std::vector<double> y(n);
            for (std::size_t i = 0U; i < n; ++i)
            {
                x_data[i] = 0.08 * static_cast<double>(i);
                y[i] = std::sin(0.15 * static_cast<double>(i)) + (0.05 * static_cast<double>(i % 7U));
            }
            const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

            const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
                + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.3, cxxgpr::MaternNu::ThreeHalves).value())
                + cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.9, 2.4).value())
                + cxxgpr::KernelSpec(cxxgpr::RationalQuadraticKernel::create(1.2, 1.7).value())
                + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.7).value())
                + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());
            auto compiled = spec.compile<double>();
            const auto lik = cxxgpr::GaussianLikelihood::create(0.1).value();
            const auto jitter = cxxgpr::JitterPolicy::fixed(0.0).value();

            auto bufs_cached =
                cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cxxgpr::FitPolicy::prefer_speed());
            auto bufs_uncached = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(
                n, cxxgpr::DistanceCachePolicy::Uncached, cxxgpr::CholeskyBuffer::Retain
            );
            ASSERT_TRUE(bufs_cached.has_value());
            ASSERT_TRUE(bufs_uncached.has_value());

            const std::size_t p_tot = compiled.num_params() + 1U;
            std::vector<double> g_cached(p_tot, 0.0);
            std::vector<double> g_uncached(p_tot, 0.0);
            ASSERT_TRUE(cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs_cached, g_cached).has_value());
            ASSERT_TRUE(
                cxxgpr::mll_and_gradient_into(compiled, lik, x, y, jitter, *bufs_uncached, g_uncached).has_value()
            );
            for (std::size_t p = 0U; p < p_tot; ++p)
            {
                EXPECT_NEAR(g_uncached[p], g_cached[p], 1e-11) << "p=" << p;
            }

            // Direct CompiledKernel::weighted_grads parity: col_partials length n (serial fallback) vs 3 * n.
            const std::size_t n_kernel = compiled.num_params();
            std::vector<double> wg_serial(n_kernel, 0.0);
            std::vector<double> wg_parallel(n_kernel, 0.0);
            std::vector<double> k_scratch(n * n, 0.0);
            std::vector<double> partials_1n(n, 0.0);
            std::vector<double> partials_3n(3U * n, 0.0);
            auto nested = compiled.make_nested_buffers(n, n);
            ASSERT_TRUE(compiled
                    .weighted_grads(
                        x, nullptr, bufs_cached->w_ref(), wg_serial, cxxgpr::MatrixMut<double>(k_scratch.data(), n, n),
                        nested, bufs_cached->core.weighted, 0U, partials_1n
                    )
                    .has_value());
            ASSERT_TRUE(compiled
                    .weighted_grads(
                        x, nullptr, bufs_cached->w_ref(), wg_parallel,
                        cxxgpr::MatrixMut<double>(k_scratch.data(), n, n), nested, bufs_cached->core.weighted, 0U,
                        partials_3n
                    )
                    .has_value());
            for (std::size_t p = 0U; p < n_kernel; ++p)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(wg_serial[p]), std::bit_cast<std::uint64_t>(wg_parallel[p]))
                    << "p=" << p;
            }
        }
    }

    /// @brief Verifies parallel W construction and MLL+grad stay deterministic on repeats and agree across threads.
    /// @details Evaluates DoublePrecision and SinglePrecision mll_and_gradient_into at n = 99, 131, and 389 with
    /// zero heap allocations under NoAllocationScope(AuditMode::CppAndEigen), checks exact bitwise invariance on
    /// repeated 4-thread calls (and across 1/2/4 threads under Eigen, or scaled rounding tolerances under external
    /// vendor BLAS), and verifies parity with CholeskyBuffer::Reuse.
    TEST(WorkspaceMllTest, MllAndGradBitwiseDeterministicAcrossThreadsAndOddSizes)
    {
        // Spec: DES-13, DES-15, DES-20
        auto check_policy = []<typename Policy>(std::size_t n)
        {
            using Scalar = Policy::Storage;
            constexpr std::size_t d = 2U;
            std::vector<Scalar> x_vals(n * d);
            std::vector<Scalar> y_vals(n);
            for (std::size_t i = 0U; i < n; ++i)
            {
                const double x0 = -1.2 + (2.4 * static_cast<double>(i) / static_cast<double>(n - 1U));
                const double x1 = 0.7 * std::cos(0.11 * static_cast<double>(i));
                x_vals[i] = static_cast<Scalar>(x0);
                x_vals[n + i] = static_cast<Scalar>(x1);
                y_vals[i] = static_cast<Scalar>(std::sin(2.0 * x0) + (0.25 * x1));
            }
            const cxxgpr::MatrixRef<Scalar> x(x_vals.data(), n, d, n);
            const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.6).value());
            const auto compiled = spec.compile<Scalar>();
            const auto lik = cxxgpr::GaussianLikelihood::create(0.15).value();
            const auto jitter = cxxgpr::JitterPolicy::fixed(0.0).value();

            auto bufs = cxxgpr::FitBuffers<Policy>::create(
                n, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain
            );
            ASSERT_TRUE(bufs.has_value());
            const std::size_t p_tot = compiled.num_params() + 1U;
            std::vector<double> warmup_grad(p_tot, 0.0);
            ASSERT_TRUE(
                cxxgpr::mll_and_gradient_into(compiled, lik, x, y_vals, jitter, *bufs, warmup_grad).has_value()
            );

            struct ThreadSnapshot
            {
                double mll = 0.0;
                std::vector<double> grad;
                std::vector<Scalar> w_lower;
            };

            auto run_threads = [&](int threads)
            {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#else
                (void)threads;
#endif
                ThreadSnapshot snap;
                snap.grad.assign(p_tot, 0.0);
                {
                    const cxxgpr::test_support::NoAllocationScope audit(cxxgpr::test_support::AuditMode::CppAndEigen);
                    auto res = cxxgpr::mll_and_gradient_into(compiled, lik, x, y_vals, jitter, *bufs, snap.grad);
                    EXPECT_TRUE(res.has_value());
                    EXPECT_EQ(audit.delta_allocations(), 0U);
                    snap.mll = *res;
                }
                const auto w_ref = bufs->w_ref();
                snap.w_lower.reserve((n * (n + 1U)) / 2U);
                for (std::size_t col = 0U; col < n; ++col)
                {
                    for (std::size_t row = col; row < n; ++row)
                    {
                        snap.w_lower.push_back(w_ref(row, col));
                    }
                }
                return snap;
            };

#ifdef _OPENMP
            const int saved_threads = omp_get_max_threads();
#endif
            const auto s1 = run_threads(1);
            const auto s2 = run_threads(2);
            const auto s4 = run_threads(4);
            const auto s4_repeat = run_threads(4);
#ifdef _OPENMP
            omp_set_num_threads(saved_threads);
#endif

            const bool ext_blas = has_external_blas_backend();
            const double blas_tol = std::same_as<Scalar, double> ? 1e-11 : 1e-4;
            EXPECT_EQ(std::bit_cast<std::uint64_t>(s4.mll), std::bit_cast<std::uint64_t>(s4_repeat.mll));
            if (!ext_blas)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(s1.mll), std::bit_cast<std::uint64_t>(s2.mll));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(s1.mll), std::bit_cast<std::uint64_t>(s4.mll));
            }
            else
            {
                const double mll_tol = blas_tol * std::max(1.0, std::abs(s1.mll));
                EXPECT_NEAR(s1.mll, s2.mll, mll_tol);
                EXPECT_NEAR(s1.mll, s4.mll, mll_tol);
            }
            for (std::size_t p = 0U; p < p_tot; ++p)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(s4.grad[p]), std::bit_cast<std::uint64_t>(s4_repeat.grad[p]));
                if (!ext_blas)
                {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(s1.grad[p]), std::bit_cast<std::uint64_t>(s2.grad[p]));
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(s1.grad[p]), std::bit_cast<std::uint64_t>(s4.grad[p]));
                }
                else
                {
                    const double grad_tol = blas_tol * std::max(1.0, std::abs(s1.grad[p]));
                    EXPECT_NEAR(s1.grad[p], s2.grad[p], grad_tol);
                    EXPECT_NEAR(s1.grad[p], s4.grad[p], grad_tol);
                }
            }
            for (std::size_t idx = 0U; idx < s1.w_lower.size(); ++idx)
            {
                if constexpr (std::same_as<Scalar, double>)
                {
                    EXPECT_EQ(
                        std::bit_cast<std::uint64_t>(s4.w_lower[idx]),
                        std::bit_cast<std::uint64_t>(s4_repeat.w_lower[idx])
                    );
                    if (!ext_blas)
                    {
                        EXPECT_EQ(
                            std::bit_cast<std::uint64_t>(s1.w_lower[idx]), std::bit_cast<std::uint64_t>(s2.w_lower[idx])
                        );
                        EXPECT_EQ(
                            std::bit_cast<std::uint64_t>(s1.w_lower[idx]), std::bit_cast<std::uint64_t>(s4.w_lower[idx])
                        );
                    }
                    else
                    {
                        const double w_tol = blas_tol * std::max(1.0, std::abs(s1.w_lower[idx]));
                        EXPECT_NEAR(s1.w_lower[idx], s2.w_lower[idx], w_tol);
                        EXPECT_NEAR(s1.w_lower[idx], s4.w_lower[idx], w_tol);
                    }
                }
                else
                {
                    EXPECT_EQ(
                        std::bit_cast<std::uint32_t>(s4.w_lower[idx]),
                        std::bit_cast<std::uint32_t>(s4_repeat.w_lower[idx])
                    );
                    if (!ext_blas)
                    {
                        EXPECT_EQ(
                            std::bit_cast<std::uint32_t>(s1.w_lower[idx]), std::bit_cast<std::uint32_t>(s2.w_lower[idx])
                        );
                        EXPECT_EQ(
                            std::bit_cast<std::uint32_t>(s1.w_lower[idx]), std::bit_cast<std::uint32_t>(s4.w_lower[idx])
                        );
                    }
                    else
                    {
                        const auto ref = static_cast<double>(s1.w_lower[idx]);
                        const double w_tol = blas_tol * std::max(1.0, std::abs(ref));
                        EXPECT_NEAR(ref, static_cast<double>(s2.w_lower[idx]), w_tol);
                        EXPECT_NEAR(ref, static_cast<double>(s4.w_lower[idx]), w_tol);
                    }
                }
            }

            // Verify parity with CholeskyBuffer::Reuse under matching parameters.
            auto bufs_reuse = cxxgpr::FitBuffers<Policy>::create(
                n, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Reuse
            );
            ASSERT_TRUE(bufs_reuse.has_value());
            std::vector<double> grad_reuse(p_tot, 0.0);
            auto mll_reuse = cxxgpr::mll_and_gradient_into(compiled, lik, x, y_vals, jitter, *bufs_reuse, grad_reuse);
            ASSERT_TRUE(mll_reuse.has_value());
            const double parity_tol = std::same_as<Scalar, double> ? 1e-12 : 1e-4;
            EXPECT_NEAR(*mll_reuse, s1.mll, parity_tol);
            for (std::size_t p = 0U; p < p_tot; ++p)
            {
                EXPECT_NEAR(grad_reuse[p], s1.grad[p], parity_tol);
            }
        };

        check_policy.template operator()<cxxgpr::DoublePrecision>(99U);
        check_policy.template operator()<cxxgpr::DoublePrecision>(131U);
        check_policy.template operator()<cxxgpr::DoublePrecision>(389U);
        check_policy.template operator()<cxxgpr::SinglePrecision>(99U);
        check_policy.template operator()<cxxgpr::SinglePrecision>(131U);
        check_policy.template operator()<cxxgpr::SinglePrecision>(389U);
    }

    /// @brief Verifies set_params and mll_and_gradient_into perform zero heap allocations across scalar, ARD, and
    /// composite kernels and all four precision policies, and that failed set_params leaves kernels unchanged.
    TEST(WorkspaceMllTest, SetParamsAndMllGradZeroAllocAcrossAllKernelsAndPrecisions)
    {
        // Spec: REQ-02, REQ-11, DES-13
        constexpr std::size_t n = 12;
        constexpr std::size_t d = 2;
        std::vector<double> x_raw(n * d);
        std::vector<double> y_raw(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x_raw[i] = 0.1 * static_cast<double>(i);
            x_raw[n + i] = 0.15 * static_cast<double>(i);
            y_raw[i] = std::sin(x_raw[i]) + (0.2 * x_raw[n + i]);
        }
        const cxxgpr::MatrixRef<double> x(x_raw.data(), n, d, n);
        const auto lik = cxxgpr::GaussianLikelihood::create(0.05).value();
        const auto jitter = cxxgpr::JitterPolicy::fixed(1e-8).value();

        const auto scalar_spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.8).value());
        const auto ard_spec = cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array{0.7, 1.3}).value());
        const auto composite_spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::MaternArdKernel::create(std::array{0.6, 1.1}, cxxgpr::MaternNu::ThreeHalves).value()
                )
            + cxxgpr::KernelSpec(cxxgpr::RationalQuadraticArdKernel::create(std::array{0.9, 1.4}, 2.0).value());

        auto check = [&]<typename P>(const cxxgpr::KernelSpec& spec, const char* label)
        {
            using Scalar = P::Storage;
            SCOPED_TRACE(label);
            auto compiled = spec.compile<Scalar>();
            const std::size_t nk = compiled.num_params();
            std::vector<double> params(nk);
            ASSERT_TRUE(compiled.get_params(params).has_value());
            std::vector<double> shifted(nk);
            for (std::size_t i = 0; i < nk; ++i)
            {
                shifted[i] = params[i] + 0.15;
            }

            // Failed set_params (out-of-bounds) must leave params unchanged.
            std::vector<double> bad = shifted;
            bad[nk - 1U] = 100.0;
            EXPECT_FALSE(compiled.set_params(bad).has_value());
            std::vector<double> after_bad(nk);
            ASSERT_TRUE(compiled.get_params(after_bad).has_value());
            EXPECT_EQ(after_bad, params);

            auto bufs =
                cxxgpr::FitBuffers<P>::create(n, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain);
            ASSERT_TRUE(bufs.has_value());
            std::vector<Scalar> x_s(x_raw.size());
            std::vector<Scalar> y_s(y_raw.size());
            for (std::size_t i = 0; i < x_raw.size(); ++i)
            {
                x_s[i] = static_cast<Scalar>(x_raw[i]);
            }
            for (std::size_t i = 0; i < y_raw.size(); ++i)
            {
                y_s[i] = static_cast<Scalar>(y_raw[i]);
            }
            const cxxgpr::MatrixRef<Scalar> x_ref(x_s.data(), n, d, n);
            std::vector<double> grad(nk + 1U, 0.0);
            ASSERT_TRUE(cxxgpr::mll_and_gradient_into(compiled, lik, x_ref, y_s, jitter, *bufs, grad).has_value());

            {
                const cxxgpr::test_support::NoAllocationScope audit_set;
                ASSERT_TRUE(compiled.set_params(shifted).has_value());
                EXPECT_EQ(audit_set.delta_allocations(), 0U);
            }
            {
                const cxxgpr::test_support::NoAllocationScope audit_fac;
                ASSERT_TRUE(cxxgpr::factor_and_mll(compiled, lik, x_ref, y_s, jitter, *bufs).has_value());
                EXPECT_EQ(audit_fac.delta_allocations(), 0U);
            }
            double mll_after = 0.0;
            {
                const cxxgpr::test_support::NoAllocationScope audit_eval;
                const auto mll2 = cxxgpr::mll_and_gradient_into(compiled, lik, x_ref, y_s, jitter, *bufs, grad);
                ASSERT_TRUE(mll2.has_value());
                EXPECT_EQ(audit_eval.delta_allocations(), 0U);
                mll_after = *mll2;
            }

            // Bitwise equivalence with a freshly compiled kernel updated to the same shifted parameters.
            auto fresh = spec.compile<Scalar>();
            ASSERT_TRUE(fresh.set_params(shifted).has_value());
            auto fresh_bufs =
                cxxgpr::FitBuffers<P>::create(n, cxxgpr::DistanceCachePolicy::Cached, cxxgpr::CholeskyBuffer::Retain);
            ASSERT_TRUE(fresh_bufs.has_value());
            std::vector<double> fresh_grad(nk + 1U, 0.0);
            const auto fresh_mll =
                cxxgpr::mll_and_gradient_into(fresh, lik, x_ref, y_s, jitter, *fresh_bufs, fresh_grad);
            ASSERT_TRUE(fresh_mll.has_value());
            EXPECT_EQ(std::bit_cast<std::uint64_t>(mll_after), std::bit_cast<std::uint64_t>(*fresh_mll));
            for (std::size_t i = 0; i < grad.size(); ++i)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(grad[i]), std::bit_cast<std::uint64_t>(fresh_grad[i]));
            }
        };

        for (const auto& [name, spec] : std::array<std::pair<const char*, cxxgpr::KernelSpec>, 3>{
                 {{"scalar", scalar_spec}, {"ard", ard_spec}, {"composite", composite_spec}}
        })
        {
            SCOPED_TRACE(name);
            check.template operator()<cxxgpr::DoublePrecision>(spec, "DoublePrecision");
            check.template operator()<cxxgpr::SinglePrecision>(spec, "SinglePrecision");
            check.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(spec, "MixedPromote");
            check.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec, "MixedReevaluate");
        }
    }

    /// @brief Verifies forward-to-backward ARD s^2 reuse in mll_and_gradient_into() across Cached and Uncached modes.
    /// @details Confirms that forward apply_keeping() / apply_cached_keeping() captured packed s^2 buffers match direct
    /// weighted_grads() (without precomputed s^2) and finite differences, with zero heap allocations after warm-up.
    TEST(WorkspaceMllTest, ArdForwardS2ReuseMatchesDirectWeightedGradsWithZeroAlloc)
    {
        // Spec: REQ-02, REQ-08, DES-11, DES-20
        constexpr std::size_t n = 10U;
        constexpr std::size_t d = 3U;
        std::vector<double> x_raw(n * d);
        std::vector<double> y_raw(n);
        for (std::size_t row = 0U; row < n; ++row)
        {
            for (std::size_t col = 0U; col < d; ++col)
            {
                x_raw[(col * n) + row] = (0.19 * static_cast<double>(row)) - (0.27 * static_cast<double>(col));
            }
            y_raw[row] = std::sin(0.5 * static_cast<double>(row)) + (0.05 * static_cast<double>(row));
        }

        const auto lik = cxxgpr::GaussianLikelihood::create(0.08).value();
        const auto jitter = cxxgpr::JitterPolicy::fixed(1e-8).value();
        const auto composite_ard = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.3).value())
                * cxxgpr::KernelSpec(cxxgpr::RationalQuadraticArdKernel::create(std::array{0.7, 1.1, 1.6}, 2.2).value())
            + cxxgpr::KernelSpec(
                cxxgpr::MaternArdKernel::create(std::array{0.9, 1.4, 2.0}, cxxgpr::MaternNu::ThreeHalves).value()
            )
            + cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array{1.2, 0.8, 1.5}).value());

        for (const auto cache_policy : {cxxgpr::DistanceCachePolicy::Cached, cxxgpr::DistanceCachePolicy::Uncached})
        {
            auto compiled = composite_ard.compile<double>();
            const std::size_t nk = compiled.num_params();
            const std::size_t n_ard = compiled.ard_leaf_count();
            ASSERT_EQ(n_ard, 3U);

            auto bufs =
                cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cache_policy, cxxgpr::CholeskyBuffer::Retain);
            ASSERT_TRUE(bufs.has_value());
            const cxxgpr::MatrixRef<double> x_ref(x_raw.data(), n, d, n);

            std::vector<double> grad_warm(nk + 1U, 0.0);
            ASSERT_TRUE(
                cxxgpr::mll_and_gradient_into(compiled, lik, x_ref, y_raw, jitter, *bufs, grad_warm).has_value()
            );

            std::vector<double> params(nk);
            ASSERT_TRUE(compiled.get_params(params).has_value());
            for (double& p : params)
            {
                p += 0.12;
            }
            ASSERT_TRUE(compiled.set_params(params).has_value());

            std::vector<double> grad_reused(nk + 1U, 0.0);
            {
                const cxxgpr::test_support::NoAllocationScope audit(cxxgpr::test_support::AuditMode::CppAndEigen);
                ASSERT_TRUE(
                    cxxgpr::mll_and_gradient_into(compiled, lik, x_ref, y_raw, jitter, *bufs, grad_reused).has_value()
                );
                EXPECT_EQ(audit.delta_allocations(), 0U);
            }

            // Confirm forward capture populated every ARD slot and matches direct weighted_grads() without s^2 reuse.
            const std::size_t per_dim = (n * (n + 1U)) / 2U;
            for (std::size_t slot = 0U; slot < n_ard; ++slot)
            {
                EXPECT_EQ(bufs->core.ard_s2_ref(slot).size(), per_dim);
            }
            std::vector<double> direct_kgrads(nk, 0.0);
            const cxxgpr::DistanceCache<double>* dist_ptr =
                (bufs->dist.has_value() && (bufs->dist->has_dist() || bufs->dist->has_ard())) ? &(*bufs->dist)
                                                                                              : nullptr;
            ASSERT_TRUE(compiled
                    .weighted_grads(
                        x_ref, dist_ptr, bufs->w_ref(), direct_kgrads, bufs->core.kernel_scratch_mut(),
                        bufs->core.nested, bufs->core.weighted, compiled.kept_products(),
                        std::span<double>(bufs->core.col_partials)
                    )
                    .has_value());
            for (std::size_t p = 0U; p < nk; ++p)
            {
                EXPECT_NEAR(grad_reused[p], -0.5 * direct_kgrads[p], 1e-12) << "param " << p;
            }
        }
    }

}  // namespace
