// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, REQ-05, DES-09
/// @file distance_cache_test.cpp
/// @brief Unit and property tests for DistanceCache layout, requirement detection, and cache reuse.
/// @details Verifies compiled kernel distance/ARD capability flags, zero-allocation build when
/// uncached or unneeded, packed column-major lower-triangular ARD layout, parity against direct
/// point evaluation across KernelExp modes, cache-poisoning mutation proof, and error handling.

#include "cxxgpr/kernel/kernel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Verifies kernel distance/ARD requirement flags and zero-alloc uncached builds.
    /// @details Checks reads_distances() and needs_ard_sq_diff() across coordinate-only, scalar,
    /// ARD, and mixed kernels, then confirms DistanceCache::build allocates zero bytes when Uncached
    /// or when neither cache buffer is required.
    TEST(DistanceCacheTest, KernelRequirementsAndZeroAllocationWhenUnneeded)
    {
        // Constant + Linear + White requires neither scalar squared distances nor ARD differences.
        const auto c_spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(2.0).value())
            + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.1).value());
        const auto c_comp = c_spec.compile<double>();
        EXPECT_FALSE(c_comp.reads_distances());
        EXPECT_FALSE(c_comp.needs_ard_sq_diff());

        const auto rbf_comp = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value()).compile<double>();
        EXPECT_TRUE(rbf_comp.reads_distances());
        EXPECT_FALSE(rbf_comp.needs_ard_sq_diff());

        const auto ard_comp =
            cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 2.0}).value()).compile<double>();
        EXPECT_FALSE(ard_comp.reads_distances());
        EXPECT_TRUE(ard_comp.needs_ard_sq_diff());

        const auto mixed_comp =
            (cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::MaternArdKernel::create(std::array<double, 2>{1.0, 2.0}, cxxgpr::MaternNu::ThreeHalves)
                        .value()
                ))
                .compile<double>();
        EXPECT_TRUE(mixed_comp.reads_distances());
        EXPECT_TRUE(mixed_comp.needs_ard_sq_diff());

        const std::vector<double> x_data{0.0, 1.0, 2.0, -0.5, 0.3, 1.4};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 3, 2, 3);

        // Verify populating an uncached or unneeded cache via ensure() allocates zero heap memory.
        // Note: pre-construct DistanceCache outside NoAllocationScope so MSVC Debug STL iterator
        // proxy allocations during std::vector default construction are excluded from the audit.
        cxxgpr::DistanceCache<double> uncached;
        cxxgpr::DistanceCache<double> unneeded;
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(uncached.ensure(x, true, true, cxxgpr::DistanceCachePolicy::Uncached).has_value());
            EXPECT_FALSE(uncached.has_dist());
            EXPECT_FALSE(uncached.has_ard());
            EXPECT_EQ(uncached.policy(), cxxgpr::DistanceCachePolicy::Uncached);

            ASSERT_TRUE(unneeded
                    .ensure(
                        x, c_comp.reads_distances(), c_comp.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
                    )
                    .has_value());
            EXPECT_FALSE(unneeded.has_dist());
            EXPECT_FALSE(unneeded.has_ard());
        }

#if !defined(_MSC_VER) || !defined(_DEBUG)
        // Outside MSVC Debug (where default-constructing empty std::vector allocates an iterator proxy),
        // also audit DistanceCache::build() directly for zero heap allocations.
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            auto built_uncached =
                cxxgpr::DistanceCache<double>::build(x, true, true, cxxgpr::DistanceCachePolicy::Uncached);
            ASSERT_TRUE(built_uncached.has_value());
            EXPECT_FALSE(built_uncached->has_dist());
            EXPECT_FALSE(built_uncached->has_ard());

            auto built_unneeded = cxxgpr::DistanceCache<double>::build(
                x, c_comp.reads_distances(), c_comp.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
            );
            ASSERT_TRUE(built_unneeded.has_value());
            EXPECT_FALSE(built_unneeded->has_dist());
            EXPECT_FALSE(built_unneeded->has_ard());
        }
#endif
    }

    /// @brief Verifies packed lower-triangular ARD layout and input validation error codes.
    /// @details Confirms EmptyInput and NonFiniteInput rejections and checks every packed ARD entry
    /// and dense squared-distance entry against explicit pairwise differences.
    TEST(DistanceCacheTest, PackedArdLayoutMatchesSpecificationAndHandlesErrors)
    {
        const std::vector<double> x_data{1.0, 2.5, -1.0, 0.5, 0.0, 3.0, 1.5, -0.5};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 4, 2, 4);

        // Negative / error paths.
        const cxxgpr::MatrixRef<double> empty_x;
        EXPECT_EQ(
            cxxgpr::DistanceCache<double>::build(empty_x, true, true).error().code(), cxxgpr::GprErrorCode::EmptyInput
        );

        std::vector<double> nan_x_data = x_data;
        nan_x_data[3] = std::numeric_limits<double>::quiet_NaN();
        const cxxgpr::MatrixRef<double> nan_x(nan_x_data.data(), 4, 2, 4);
        EXPECT_EQ(
            cxxgpr::DistanceCache<double>::build(nan_x, false, true).error().code(),
            cxxgpr::GprErrorCode::NonFiniteInput
        );

        auto cache = cxxgpr::DistanceCache<double>::build(x, true, true);
        ASSERT_TRUE(cache.has_value());
        if (!cache.has_value())
        {
            return;
        }

        EXPECT_TRUE(cache->has_dist());
        EXPECT_TRUE(cache->has_ard());
        EXPECT_EQ(cache->num_points(), 4U);
        EXPECT_EQ(cache->num_dims(), 2U);

        constexpr std::size_t n = 4U;
        constexpr std::size_t d = 2U;
        constexpr std::size_t tri = (n * (n + 1U)) / 2U;
        const auto packed = cache->ard_packed();
        ASSERT_EQ(packed.size(), d * tri);

        // Verify packed column-major lower-triangular ARD storage layout for each dimension m.
        for (std::size_t m = 0; m < d; ++m)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                for (std::size_t i = j; i < n; ++i)
                {
                    const std::size_t p = ((j * n) - ((j * (j - 1U)) / 2U)) + (i - j);
                    const double diff = x(i, m) - x(j, m);
                    EXPECT_DOUBLE_EQ(packed[(m * tri) + p], diff * diff);
                }
            }
        }

        // Verify dense Euclidean squared-distance matrix against explicit coordinate sums.
        const auto dist = cache->dist_view();
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                double expected = 0.0;
                for (std::size_t m = 0; m < d; ++m)
                {
                    const double diff = x(i, m) - x(j, m);
                    expected += diff * diff;
                }
                EXPECT_DOUBLE_EQ(dist(i, j), expected);
            }
        }
    }

    /// @brief Verifies cached kernel evaluation, gradients, and Hessians match point evaluation.
    /// @details Exercises apply_cached(), grad_cached(), and hess_cached() on a composite kernel
    /// under both Accurate and FastApprox exp modes and audits zero heap allocations.
    TEST(DistanceCacheTest, CachedEvaluationMatchesPointsAndReusesCacheZeroAlloc)
    {
        const std::vector<double> x_data{0.1, 0.7, 1.4, -0.5, 0.3, -0.8, 1.1, 0.2};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 4, 2, 4);

        // Composite tree exercising scalar distance, ARD squared-differences, Periodic, Linear, and White.
        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.8, 1.5}).value()))
            + cxxgpr::KernelSpec(
                cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.3}, cxxgpr::MaternNu::FiveHalves).value()
            )
            + cxxgpr::KernelSpec(
                cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.2, 0.7}, 1.8).value()
            )
            + cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.9, 2.2).value())
            + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.4).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

        for (const auto exp_mode : {cxxgpr::KernelExp::Accurate, cxxgpr::KernelExp::FastApprox})
        {
            auto compiled = spec.compile<double>();
            compiled.set_kernel_exp(exp_mode);
            EXPECT_TRUE(compiled.reads_distances());
            EXPECT_TRUE(compiled.needs_ard_sq_diff());

            auto cache = cxxgpr::DistanceCache<double>::build(
                x, compiled.reads_distances(), compiled.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
            );
            ASSERT_TRUE(cache.has_value());
            if (!cache.has_value())
            {
                return;
            }

            constexpr std::size_t n = 4U;
            std::vector<double> k_pts(n * n);
            std::vector<double> k_cached(n * n);
            std::vector<double> g_pts(n * n);
            std::vector<double> g_cached(n * n);
            std::vector<double> h_pts(n * n);
            std::vector<double> h_cached(n * n);
            std::vector<double> sc_a(n * n);
            std::vector<double> sc_b(n * n);
            std::vector<double> sc_c(n * n);
            auto nested = compiled.make_nested_buffers(n, n);

            ASSERT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(k_pts.data(), n, n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), nested
                    )
                    .has_value());

            // Audit cached Gram evaluation against direct apply_points().
            {
                const cxxgpr::test_support::NoAllocationScope audit;
                ASSERT_TRUE(compiled
                        .apply_cached(
                            x, *cache, cxxgpr::MatrixMut<double>(k_cached.data(), n, n, n), cxxgpr::Triangle::Full,
                            cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), nested
                        )
                        .has_value());
            }

            for (std::size_t idx = 0; idx < k_pts.size(); ++idx)
            {
                EXPECT_NEAR(k_cached[idx], k_pts[idx], 1e-13);
            }

            const std::size_t np = compiled.num_params();
            for (std::size_t p = 0; p < np; ++p)
            {
                ASSERT_TRUE(compiled
                        .grad_points(
                            x, cxxgpr::MatrixMut<double>(g_pts.data(), n, n, n), p, cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                            cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), nested
                        )
                        .has_value());
                {
                    const cxxgpr::test_support::NoAllocationScope audit;
                    ASSERT_TRUE(compiled
                            .grad_cached(
                                x, *cache, cxxgpr::MatrixMut<double>(g_cached.data(), n, n, n), p,
                                cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), nested
                            )
                            .has_value());
                }
                for (std::size_t col = 0; col < n; ++col)
                {
                    for (std::size_t row = col; row < n; ++row)
                    {
                        EXPECT_NEAR(g_cached[(col * n) + row], g_pts[(col * n) + row], 1e-13);
                    }
                }
            }

            for (std::size_t i = 0; i < np; ++i)
            {
                for (std::size_t j = i; j < np; ++j)
                {
                    ASSERT_TRUE(compiled
                            .hess_points(
                                x, cxxgpr::MatrixMut<double>(h_pts.data(), n, n, n), i, j, cxxgpr::Triangle::Lower,
                                cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_c.data(), n, n, n), nested
                            )
                            .has_value());
                    {
                        const cxxgpr::test_support::NoAllocationScope audit;
                        ASSERT_TRUE(compiled
                                .hess_cached(
                                    x, *cache, cxxgpr::MatrixMut<double>(h_cached.data(), n, n, n), i, j,
                                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                                    cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n),
                                    cxxgpr::MatrixMut<double>(sc_c.data(), n, n, n), nested
                                )
                                .has_value());
                    }
                    for (std::size_t col = 0; col < n; ++col)
                    {
                        for (std::size_t row = col; row < n; ++row)
                        {
                            EXPECT_NEAR(h_cached[(col * n) + row], h_pts[(col * n) + row], 1e-13);
                        }
                    }
                }
            }
        }
    }

    /// @brief Verifies float DistanceCache reuse via cache-poisoning mutation and shape checks.
    /// @details Mutates cached dist and ARD entries to confirm apply_cached() reads precomputed
    /// buffers rather than raw x, and checks LengthMismatch rejection on mismatched point counts.
    TEST(DistanceCacheTest, AlwaysReusesPopulatedCacheAndSupportsFloat)
    {
        const std::vector<float> x_data{0.0F, 1.0F, 2.0F, -0.5F, 0.5F, 1.5F};
        const cxxgpr::MatrixRef<float> x(x_data.data(), 3, 2, 3);

        const auto compiled =
            (cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value())
                + cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 1.0}).value()))
                .compile<float>();

        auto cache = cxxgpr::DistanceCache<float>::build(
            x, compiled.reads_distances(), compiled.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
        );
        ASSERT_TRUE(cache.has_value());
        if (!cache.has_value())
        {
            return;
        }

        constexpr std::size_t n = 3U;
        std::vector<float> k_clean(n * n);
        std::vector<float> k_poisoned(n * n);
        std::vector<float> sc(n * n);
        auto nested = compiled.make_nested_buffers(n, n);

        ASSERT_TRUE(compiled
                .apply_cached(
                    x, *cache, cxxgpr::MatrixMut<float>(k_clean.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<float>(sc.data(), n, n, n), nested
                )
                .has_value());

        // Poison an off-diagonal entry in both dist and ard caches to prove cached paths read them.
        cache->dist_mut()(1, 0) = 100.0F;
        cache->dist_mut()(0, 1) = 100.0F;
        cache->ard_packed_mut()[1] = 100.0F;

        ASSERT_TRUE(compiled
                .apply_cached(
                    x, *cache, cxxgpr::MatrixMut<float>(k_poisoned.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<float>(sc.data(), n, n, n), nested
                )
                .has_value());
        EXPECT_LT(k_poisoned[1], k_clean[1] * 0.1F);

        // Shape mismatch detection.
        const std::vector<float> bad_x_data{0.0F, 1.0F, 0.0F, 1.0F};
        const cxxgpr::MatrixRef<float> bad_x(bad_x_data.data(), 2, 2, 2);
        EXPECT_EQ(
            compiled
                .apply_cached(
                    bad_x, *cache, cxxgpr::MatrixMut<float>(k_clean.data(), 2, 2, 2), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<float>(sc.data(), 2, 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );
    }

}  // namespace
