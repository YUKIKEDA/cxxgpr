// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, REQ-08, DES-11
/// @file leaf_cache_test.cpp
/// @brief Unit and property tests for LeafCache incremental evaluation and OpenMP determinism.
/// @details Verifies optimizer_uses_change_indices_v trait detection, strict bitwise parameter
/// change verification including signed-zero sensitivity, incremental leaf reuse and poisoning
/// proof under NoAllocationScope, and bitwise thread-count invariance under OpenMP.

#include "cxxgpr/workspace/workspace.hpp"

#include <gtest/gtest.h>

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

    /// @brief Dummy optimizer tag lacking uses_change_indices for trait verification.
    struct DummyNormalOptimizer
    {
    };

    /// @brief Dummy optimizer tag opting into incremental LeafCache updates via uses_change_indices.
    struct DummyFsaOptimizer
    {
        /// @brief Opt-in flag enabling change-index tracking in optimizer traits.
        static constexpr bool uses_change_indices = true;
    };

    static_assert(!cxxgpr::optimizer_uses_change_indices_v<DummyNormalOptimizer>);
    static_assert(cxxgpr::optimizer_uses_change_indices_v<DummyFsaOptimizer>);

    /// @brief Negative and positive controls for require_change_indices() and bitwise dirty tracking.
    /// @details Confirms empty/out-of-range/duplicate index rejection, undeclared parameter mutation
    /// detection (including +0.0 vs -0.0 bit flips), and clean-leaf preservation on likelihood updates.
    TEST(LeafCacheTest, RequireChangeIndicesAndBitwiseChecksRejectInvalidSteps)
    {
        // Reject empty, out-of-range, and duplicate parameter-change index spans.
        EXPECT_EQ(
            cxxgpr::require_change_indices(std::span<const std::size_t>{}, 3U).error().code(),
            cxxgpr::GprErrorCode::IndexOutOfRange
        );

        const std::array<std::size_t, 1> oob{3U};
        EXPECT_EQ(cxxgpr::require_change_indices(oob, 3U).error().code(), cxxgpr::GprErrorCode::IndexOutOfRange);

        const std::array<std::size_t, 2> dup{1U, 1U};
        EXPECT_EQ(cxxgpr::require_change_indices(dup, 3U).error().code(), cxxgpr::GprErrorCode::IndexOutOfRange);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.1).value());
        const auto compiled = spec.compile<double>();

        cxxgpr::LeafCache<double> cache;
        ASSERT_TRUE(cache.resize(compiled.num_leaves(), 3U).has_value());

        const std::array<std::size_t, 2> init_idx{0U, 1U};
        const std::array<double, 3> p0{0.0, -2.0, 0.5};
        ASSERT_TRUE(cache.prepare_step(p0, init_idx, compiled, 3U).has_value());
        ASSERT_TRUE(cache.record(p0).has_value());
        EXPECT_TRUE(cache.is_primed());

        // Undeclared change in index 1 (must be rejected).
        const std::array<std::size_t, 1> only_0{0U};
        const std::array<double, 3> p_bad{0.2, -2.0 + 1e-12, 0.5};
        EXPECT_EQ(
            cache.prepare_step(p_bad, only_0, compiled, 3U).error().code(), cxxgpr::GprErrorCode::IndexOutOfRange
        );

        // +0.0 -> -0.0 bit difference at undeclared index 0 must also be detected.
        const std::array<std::size_t, 1> only_1{1U};
        const std::array<double, 3> p_neg_zero{-0.0, -1.5, 0.5};
        EXPECT_EQ(
            cache.prepare_step(p_neg_zero, only_1, compiled, 3U).error().code(), cxxgpr::GprErrorCode::IndexOutOfRange
        );

        // Changing only likelihood index 2 leaves all kernel leaves clean.
        const std::array<std::size_t, 1> only_2{2U};
        const std::array<double, 3> p_lik{0.0, -2.0, 0.9};
        ASSERT_TRUE(cache.prepare_step(p_lik, only_2, compiled, 3U).has_value());
        EXPECT_FALSE(cache.is_leaf_dirty(0U));
        EXPECT_FALSE(cache.is_leaf_dirty(1U));

        // Changing only leaf 0 (+0.0 -> -0.0) marks only leaf 0 dirty.
        const std::array<double, 3> p_neg_zero_only0{-0.0, -2.0, 0.5};
        ASSERT_TRUE(cache.prepare_step(p_neg_zero_only0, only_0, compiled, 3U).has_value());
        EXPECT_TRUE(cache.is_leaf_dirty(0U));
        EXPECT_FALSE(cache.is_leaf_dirty(1U));
    }

    /// @brief Verifies incremental leaf evaluation matches full evaluation with zero allocations.
    /// @details Runs initial full evaluation, single-leaf update under NoAllocationScope matching
    /// apply_points(), and a leaf-poisoning mutation proving clean cached leaf Grams are reused.
    TEST(LeafCacheTest, IncrementalEvaluationMatchesFullAndSkipsCleanLeavesZeroAlloc)
    {
        const std::vector<double> x_data{0.0, 0.5, 1.3, -0.4, 0.7, 1.1};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 3, 2, 3);

        const auto spec = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.5).value())
                              * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()))
            + (cxxgpr::KernelSpec(
                   cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.4}, cxxgpr::MaternNu::ThreeHalves)
                       .value()
               ) * cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.8, 2.2).value()))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

        auto compiled = spec.compile<double>();
        const std::size_t n_kernel = compiled.num_params();
        const std::size_t total_params = n_kernel + 1U;  // +1 for likelihood param

        auto dcache = cxxgpr::DistanceCache<double>::build(
            x, compiled.reads_distances(), compiled.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
        );
        ASSERT_TRUE(dcache.has_value());
        if (!dcache.has_value())
        {
            return;
        }

        constexpr std::size_t n = 3U;
        std::vector<double> theta(total_params, 0.0);
        ASSERT_TRUE(compiled.get_params(std::span<double>(theta.data(), n_kernel)).has_value());
        theta[n_kernel] = -1.0;

        std::vector<std::size_t> all_idx(total_params);
        for (std::size_t k = 0; k < total_params; ++k)
        {
            all_idx[k] = k;
        }

        cxxgpr::LeafCache<double> lcache;
        std::vector<double> out_inc(n * n);
        std::vector<double> out_ref(n * n);
        std::vector<double> scratch(n * n);
        auto nested = compiled.make_nested_buffers(n, n);

        // Step 1: full initial evaluation.
        ASSERT_TRUE(lcache
                .evaluate_incremental(
                    compiled, x, &(*dcache), theta, all_idx, total_params,
                    cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                )
                .has_value());
        ASSERT_TRUE(compiled
                .apply_points(
                    x, cxxgpr::MatrixMut<double>(out_ref.data(), n, n, n), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                )
                .has_value());
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col; row < n; ++row)
            {
                EXPECT_NEAR(out_inc[(col * n) + row], out_ref[(col * n) + row], 1e-13);
            }
        }

        // Step 2: change only leaf 1 (RbfKernel param at index 1) under NoAllocationScope.
        theta[1] = std::log(1.8);
        ASSERT_TRUE(compiled.set_params(std::span<const double>(theta.data(), n_kernel)).has_value());
        const std::array<std::size_t, 1> step2_idx{1U};
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(lcache
                    .evaluate_incremental(
                        compiled, x, &(*dcache), theta, step2_idx, total_params,
                        cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                    )
                    .has_value());
        }
        ASSERT_TRUE(compiled
                .apply_points(
                    x, cxxgpr::MatrixMut<double>(out_ref.data(), n, n, n), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                )
                .has_value());
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col; row < n; ++row)
            {
                EXPECT_NEAR(out_inc[(col * n) + row], out_ref[(col * n) + row], 1e-13);
            }
        }

        // Step 3: prove clean leaves are truly reused by poisoning leaf 0 (Constant) and updating only
        // likelihood param.
        lcache.leaf_gram_mut(0U)(0, 0) = 99.0;
        lcache.leaf_gram_mut(0U)(1, 0) = 99.0;
        theta[n_kernel] = -0.5;
        const std::array<std::size_t, 1> lik_only{n_kernel};
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(lcache
                    .evaluate_incremental(
                        compiled, x, nullptr, theta, lik_only, total_params,
                        cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                    )
                    .has_value());
        }
        EXPECT_GT(out_inc[0], 50.0);
    }

    /// @brief Verifies OpenMP parallel distance and kernel evaluation is bitwise deterministic.
    /// @details Compares std::bit_cast<uint64_t> bit patterns across DistanceCache, apply_points(),
    /// apply_cached(), grad_cached(), and apply_cross_points() between 1 and 4 OpenMP threads.
    TEST(LeafCacheTest, OpenMpColumnParallelEvaluationIsBitwiseDeterministic)
    {
        // Generate a 72x3 non-trivial point grid to exercise multi-block OpenMP scheduling.
        constexpr std::size_t n = 72U;
        constexpr std::size_t d = 3U;
        std::vector<double> x_data(n * d);
        for (std::size_t col = 0; col < d; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                x_data[(col * n) + row] = (0.15 * static_cast<double>(row)) - (0.3 * static_cast<double>(col));
            }
        }
        const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);

        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.2).value())
                * cxxgpr::KernelSpec(
                    cxxgpr::MaternArdKernel::create(std::array<double, 3>{0.8, 1.3, 1.7}, cxxgpr::MaternNu::FiveHalves)
                        .value()
                ))
            + cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.9, 2.5).value());
        auto compiled = spec.compile<double>();
        compiled.set_kernel_exp(cxxgpr::KernelExp::FastApprox);

        auto run_all = [&](int threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            auto dcache = cxxgpr::DistanceCache<double>::build(
                x, compiled.reads_distances(), compiled.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
            );
            EXPECT_TRUE(dcache.has_value());

            std::vector<double> k_pts(n * n, 0.0);
            std::vector<double> k_cached(n * n, 0.0);
            std::vector<double> g_cached(n * n, 0.0);
            std::vector<double> k_cross(n * n, 0.0);
            std::vector<double> sc_a(n * n, 0.0);
            std::vector<double> sc_b(n * n, 0.0);
            auto nested = compiled.make_nested_buffers(n, n);

            EXPECT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(k_pts.data(), n, n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), nested
                    )
                    .has_value());
            EXPECT_TRUE(compiled
                    .apply_cached(
                        x, *dcache, cxxgpr::MatrixMut<double>(k_cached.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), nested
                    )
                    .has_value());
            EXPECT_TRUE(compiled
                    .grad_cached(
                        x, *dcache, cxxgpr::MatrixMut<double>(g_cached.data(), n, n, n), 1U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), nested
                    )
                    .has_value());
            EXPECT_TRUE(compiled
                    .apply_cross_points(
                        x, x, cxxgpr::MatrixMut<double>(k_cross.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), nested
                    )
                    .has_value());

            std::vector<double> combined;
            combined.insert(combined.end(), dcache->dist_view().data(), dcache->dist_view().data() + (n * n));
            combined.insert(combined.end(), dcache->ard_packed().begin(), dcache->ard_packed().end());
            combined.insert(combined.end(), k_pts.begin(), k_pts.end());
            combined.insert(combined.end(), k_cached.begin(), k_cached.end());
            combined.insert(combined.end(), g_cached.begin(), g_cached.end());
            combined.insert(combined.end(), k_cross.begin(), k_cross.end());
            return combined;
        };

#ifdef _OPENMP
        const int saved_threads = omp_get_max_threads();
#endif
        const auto r1 = run_all(1);
        const auto r4 = run_all(4);
#ifdef _OPENMP
        omp_set_num_threads(saved_threads);
#endif

        ASSERT_EQ(r1.size(), r4.size());
        for (std::size_t idx = 0; idx < r1.size(); ++idx)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(r1[idx]), std::bit_cast<std::uint64_t>(r4[idx])) << "idx=" << idx;
        }
    }

    /// @brief Verifies LeafCache incremental ARD guards (large jump, underflow weight NaN, max steps, d > 64).
    /// @details Confirms exact parity against apply_points() when forcing full rebuilds on large ratio jumps,
    /// underflowed 1/ell^2 weights (> 1e155), reaching the 64-step incremental drift limit, and exceeding d = 64.
    TEST(LeafCacheTest, IncrementalArdGuardsAndFallbacksMatchFull)
    {
        // Case 1: Large trial jump, underflowed 1/ell^2 weights, and 65-step drift limit on d = 2.
        {
            constexpr std::size_t n = 12U;
            constexpr std::size_t d = 2U;
            std::vector<double> x_data(n * d);
            for (std::size_t c = 0U; c < d; ++c)
            {
                for (std::size_t r = 0U; r < n; ++r)
                {
                    x_data[(c * n) + r] = (0.2 * static_cast<double>(r)) + (0.1 * static_cast<double>(c));
                }
            }
            const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);
            const std::array<cxxgpr::Interval, 2> wide_intervals{
                cxxgpr::Interval::positive(1e-5, 1e200).value(),
                cxxgpr::Interval::positive(1e-5, 1e200).value(),
            };
            const auto spec = cxxgpr::KernelSpec(
                cxxgpr::MaternArdKernel::create(
                    std::array<double, 2>{1.0, 1.2}, cxxgpr::MaternNu::ThreeHalves, wide_intervals
                )
                    .value()
            );
            auto compiled = spec.compile<double>();
            auto dcache = cxxgpr::DistanceCache<double>::build(
                x, compiled.reads_distances(), compiled.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
            );
            ASSERT_TRUE(dcache.has_value());

            const std::size_t total_params = compiled.num_params() + 1U;
            std::vector<double> theta(total_params, 0.0);
            ASSERT_TRUE(compiled.get_params(std::span<double>(theta.data(), compiled.num_params())).has_value());
            theta.back() = -1.0;

            cxxgpr::LeafCache<double> lcache;
            std::vector<double> out_inc(n * n);
            std::vector<double> out_ref(n * n);
            std::vector<double> scratch(n * n);
            auto nested = compiled.make_nested_buffers(n, n);
            const std::array<std::size_t, 3> all_idx{0U, 1U, 2U};

            ASSERT_TRUE(lcache
                    .evaluate_incremental(
                        compiled, x, &(*dcache), theta, all_idx, total_params,
                        cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                    )
                    .has_value());

            // Drive 66 consecutive small incremental steps past k_max_inc_steps (64) to exercise reset rebuild.
            const std::array<std::size_t, 1> only_dim0{0U};
            for (int step = 1; step <= 66; ++step)
            {
                theta[0] = 0.01 * static_cast<double>(step);
                ASSERT_TRUE(
                    compiled.set_params(std::span<const double>(theta.data(), compiled.num_params())).has_value()
                );
                ASSERT_TRUE(lcache
                        .evaluate_incremental(
                            compiled, x, &(*dcache), theta, only_dim0, total_params,
                            cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                        )
                        .has_value());
            }
            ASSERT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(out_ref.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                    )
                    .has_value());
            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    EXPECT_NEAR(out_inc[(col * n) + row], out_ref[(col * n) + row], 1e-13);
                }
            }

            // Trigger ratio cancellation guard (large jump) and weight underflow guard (ell > 1e155).
            for (const double log_ell : {12.0, 368.4, 369.0, 0.5})
            {
                theta[0] = log_ell;
                ASSERT_TRUE(
                    compiled.set_params(std::span<const double>(theta.data(), compiled.num_params())).has_value()
                );
                ASSERT_TRUE(lcache
                        .evaluate_incremental(
                            compiled, x, &(*dcache), theta, only_dim0, total_params,
                            cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                        )
                        .has_value());
                ASSERT_TRUE(compiled
                        .apply_points(
                            x, cxxgpr::MatrixMut<double>(out_ref.data(), n, n, n), cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                        )
                        .has_value());
                for (std::size_t col = 0U; col < n; ++col)
                {
                    for (std::size_t row = col; row < n; ++row)
                    {
                        EXPECT_NEAR(out_inc[(col * n) + row], out_ref[(col * n) + row], 1e-13);
                    }
                }
            }
        }

        // Case 2: High dimension d = 65 (> 64 incremental ARD cache cutoff) falls back cleanly with zero allocs.
        {
            constexpr std::size_t n = 6U;
            constexpr std::size_t d = 65U;
            std::vector<double> x_data(n * d);
            for (std::size_t idx = 0U; idx < x_data.size(); ++idx)
            {
                x_data[idx] = 0.05 * static_cast<double>(idx % 17U);
            }
            const cxxgpr::MatrixRef<double> x(x_data.data(), n, d, n);
            std::vector<double> ell(d, 1.5);
            const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(ell).value());
            auto compiled = spec.compile<double>();
            auto dcache = cxxgpr::DistanceCache<double>::build(
                x, compiled.reads_distances(), compiled.needs_ard_sq_diff(), cxxgpr::DistanceCachePolicy::Cached
            );
            ASSERT_TRUE(dcache.has_value());

            const std::size_t total_params = d + 1U;
            std::vector<double> theta(total_params, std::log(1.5));
            theta.back() = -1.0;
            std::vector<std::size_t> all_idx(total_params);
            for (std::size_t k = 0U; k < total_params; ++k)
            {
                all_idx[k] = k;
            }

            cxxgpr::LeafCache<double> lcache;
            std::vector<double> out_inc(n * n);
            std::vector<double> out_ref(n * n);
            std::vector<double> scratch(n * n);
            auto nested = compiled.make_nested_buffers(n, n);
            ASSERT_TRUE(lcache
                    .evaluate_incremental(
                        compiled, x, &(*dcache), theta, all_idx, total_params,
                        cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                    )
                    .has_value());

            // Single-dimension change at d = 65 must fall back cleanly with zero heap allocations.
            theta[0] = std::log(1.7);
            ASSERT_TRUE(compiled.set_params(std::span<const double>(theta.data(), d)).has_value());
            const std::array<std::size_t, 1> only_0{0U};
            {
                const cxxgpr::test_support::NoAllocationScope audit;
                ASSERT_TRUE(lcache
                        .evaluate_incremental(
                            compiled, x, &(*dcache), theta, only_0, total_params,
                            cxxgpr::MatrixMut<double>(out_inc.data(), n, n, n), cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                        )
                        .has_value());
            }
            ASSERT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(out_ref.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(scratch.data(), n, n, n), nested
                    )
                    .has_value());
            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    EXPECT_NEAR(out_inc[(col * n) + row], out_ref[(col * n) + row], 1e-13);
                }
            }
        }
    }

}  // namespace
