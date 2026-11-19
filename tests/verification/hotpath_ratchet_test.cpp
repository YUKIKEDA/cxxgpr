// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-05, REQ-11, DES-05, DES-10, DES-13, DES-14, DES-15, DES-16, DES-17
/// @file hotpath_ratchet_test.cpp
/// @brief Verification ratchet tests enforcing zero hot-path allocations and supplied-distance parity.
/// @details Audits zero heap allocations under NoAllocationScope for factor_and_mll(),
/// mll_and_gradient_into(), value_gradient_hessian_into(), FittedGpr::predict_into(),
/// LoadedGpr::predict_into(), and DistanceKernel apply()/grad(), and enforces a median latency
/// ratio ratchet comparing supplied-distance evaluation against coordinate evaluation.

#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/persist/persist.hpp"
#include "cxxgpr/workspace/mll.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <system_error>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Audits zero heap allocations across MLL, gradient, and Hessian evaluations after warmup.
    /// @details Warms up FitBuffers under both prefer_speed() and prefer_memory() policies for
    /// precision P and asserts delta_allocations() == 0 inside NoAllocationScope.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] spec Kernel AST specification under audit.
    /// @param[in] n Number of training samples.
    /// @param[in] d Feature dimensionality.
    /// @param[in] x_raw Flattened column-major n x d training inputs in double precision.
    /// @param[in] y_raw Training target vector of length n in double precision.
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_mll_grad_hess_zero_alloc(
        const cxxgpr::KernelSpec& spec,
        std::size_t n,
        std::size_t d,
        std::span<const double> x_raw,
        std::span<const double> y_raw
    )
    {
        using Storage = P::Storage;
        // Convert double test inputs into precision P's storage representation prior to auditing.
        std::vector<Storage> x_s(x_raw.size());
        for (std::size_t i = 0; i < x_raw.size(); ++i)
        {
            x_s[i] = static_cast<Storage>(x_raw[i]);
        }
        std::vector<Storage> y_s(y_raw.size());
        for (std::size_t i = 0; i < y_raw.size(); ++i)
        {
            y_s[i] = static_cast<Storage>(y_raw[i]);
        }
        const auto x_ref = cxxgpr::MatrixRef<Storage>(x_s.data(), n, d, n);

        const auto ck = cxxgpr::CompiledKernel<Storage>::from_spec(spec);
        const auto lik = cxxgpr::GaussianLikelihood::create(1e-2).value();
        const auto jitter = cxxgpr::JitterPolicy::fixed(1e-7).value();
        const std::size_t n_params = ck.num_params() + 1U;
        std::vector<double> grad(n_params, 0.0);
        std::vector<double> hess(n_params * n_params, 0.0);

        const std::array<cxxgpr::FitPolicy, 2> policies{
            cxxgpr::FitPolicy::prefer_speed(), cxxgpr::FitPolicy::prefer_memory()
        };
        for (const auto policy : policies)
        {
            auto buf = cxxgpr::FitBuffers<P>::create(n, policy);
            ASSERT_TRUE(buf.has_value());

            // Warmup once (builds DistanceCache if Cached).
            ASSERT_TRUE(cxxgpr::factor_and_mll(ck, lik, x_ref, y_s, jitter, *buf, 0U).has_value());
            ASSERT_TRUE(cxxgpr::mll_and_gradient_into(ck, lik, x_ref, y_s, jitter, *buf, grad).has_value());
            ASSERT_TRUE(cxxgpr::value_gradient_hessian_into(ck, lik, x_ref, y_s, jitter, *buf, grad, hess).has_value());

            // Audit warmed-up factor, gradient, and Hessian paths individually for zero allocations.
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                auto mll1 = cxxgpr::factor_and_mll(ck, lik, x_ref, y_s, jitter, *buf, 0U);
                const std::size_t d1 = no_alloc.delta_allocations();
                ASSERT_TRUE(mll1.has_value());
                EXPECT_EQ(d1, 0U) << "factor_and_mll";
            }
            // Audit warmed-up factor, gradient, and Hessian paths individually for zero allocations.
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                auto mll2 = cxxgpr::mll_and_gradient_into(ck, lik, x_ref, y_s, jitter, *buf, grad);
                const std::size_t d2 = no_alloc.delta_allocations();
                ASSERT_TRUE(mll2.has_value());
                EXPECT_EQ(d2, 0U) << "mll_and_gradient_into";
            }
            // Audit warmed-up factor, gradient, and Hessian paths individually for zero allocations.
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                auto mll3 = cxxgpr::value_gradient_hessian_into(ck, lik, x_ref, y_s, jitter, *buf, grad, hess);
                const std::size_t d3 = no_alloc.delta_allocations();
                ASSERT_TRUE(mll3.has_value());
                EXPECT_EQ(d3, 0U) << "value_gradient_hessian_into";
            }
        }
    }

    /// @brief Enforces zero heap allocations on MLL, gradient, and Hessian hot paths across precisions.
    /// @details Exercises DoublePrecision, SinglePrecision, and MixedPrecision<PromoteStorage>
    /// on a composite Constant * RbfArd + Matern-5/2 kernel.
    TEST(HotpathRatchetTest, MllGradAndHessHotPathZeroAllocations)
    {
        // Spec: REQ-11, DES-13, DES-14, DES-15, DES-17
        const std::size_t n = 16;
        const std::size_t d = 2;
        std::vector<double> x_raw(n * d);
        std::vector<double> y_raw(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double x0 = 0.1 * static_cast<double>(i);
            const double x1 = 0.2 * static_cast<double>(i % 5U);
            x_raw[i] = x0;
            x_raw[n + i] = x1;
            y_raw[i] = std::sin(x0) + (0.5 * std::cos(x1));
        }

        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.9, 1.3}).value()))
            + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.0, cxxgpr::MaternNu::FiveHalves).value());

        verify_mll_grad_hess_zero_alloc<cxxgpr::DoublePrecision>(spec, n, d, x_raw, y_raw);
        verify_mll_grad_hess_zero_alloc<cxxgpr::SinglePrecision>(spec, n, d, x_raw, y_raw);
        verify_mll_grad_hess_zero_alloc<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(spec, n, d, x_raw, y_raw);
    }

    /// @brief Audits zero heap allocations on FittedGpr::predict_into() after initial workspace warmup.
    /// @details Fits a Fixed-optimizer GPR under precision policy P, runs one warmup predict_into(),
    /// and asserts delta_allocations() == 0 on the second predict_into() inside NoAllocationScope.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] spec Kernel AST specification under audit.
    /// @param[in] x_ref Column-major n x d training input view.
    /// @param[in] y_train Training target span of length n.
    /// @param[in] xq_ref Column-major m x d query input view.
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_predict_into_zero_alloc(
        const cxxgpr::KernelSpec& spec,
        cxxgpr::MatrixRef<double> x_ref,
        std::span<const double> y_train,
        cxxgpr::MatrixRef<double> xq_ref
    )
    {
        using Refine = P::Refine;
        const auto fit =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec).with_noise_variance(1e-3).value().factor(x_ref, y_train);
        ASSERT_TRUE(fit.has_value());

        // Warm up QueryWorkspace and Prediction output capacity before entering NoAllocationScope.
        cxxgpr::Prediction<Refine> pred;
        ASSERT_TRUE(fit->predict_into(xq_ref, pred).has_value());
        {
            const cxxgpr::test_support::NoAllocationScope no_alloc;
            ASSERT_TRUE(fit->predict_into(xq_ref, pred).has_value());
            EXPECT_EQ(no_alloc.delta_allocations(), 0U);
        }
    }

    /// @brief Enforces zero-allocation predict_into() across all precisions and LoadedGpr mmap paths.
    /// @details Audits Double, Single, MixedPrecision<PromoteStorage>, MixedPrecision<ReevaluateKernel>,
    /// and persisted mmap-backed LoadedGpr::predict_into() after warmup.
    TEST(HotpathRatchetTest, PredictIntoAndLoadedGprZeroAllocationsAfterWarmup)
    {
        // Spec: REQ-11, DES-05, DES-16, DES-17
        const std::size_t n = 12;
        const std::size_t d = 2;
        std::vector<double> x_train(n * d);
        std::vector<double> y_train(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x_train[i] = 0.2 * static_cast<double>(i);
            x_train[n + i] = 0.1 * static_cast<double>((i * 3U) % 7U);
            y_train[i] = std::cos(x_train[i]) + (0.2 * x_train[n + i]);
        }
        const std::array<double, 6> x_query{0.1, 0.5, 1.2, 0.2, 0.4, 0.6};
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);
        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 3U, d, 3U);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 1.5}).value());

        verify_predict_into_zero_alloc<cxxgpr::DoublePrecision>(spec, x_ref, y_train, xq_ref);
        verify_predict_into_zero_alloc<cxxgpr::SinglePrecision>(spec, x_ref, y_train, xq_ref);
        verify_predict_into_zero_alloc<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(spec, x_ref, y_train, xq_ref);
        verify_predict_into_zero_alloc<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec, x_ref, y_train, xq_ref);

        // LoadedGpr (mmap F64 factor) also zero-allocates on predict_into after warmup.
        const auto fit =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value().factor(x_ref, y_train);
        ASSERT_TRUE(fit.has_value());
        const auto temp_dir = std::filesystem::temp_directory_path() / "cxxgpr_test_hotpath_loaded";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, temp_dir).has_value());
        {
            auto loaded = cxxgpr::persist::LoadedGpr::load(temp_dir);
            ASSERT_TRUE(loaded.has_value());
            cxxgpr::Prediction<double> pred;
            ASSERT_TRUE(loaded->predict_into(xq_ref, pred).has_value());
            // Audit LoadedGpr::predict_into() for zero heap allocations after warmup.
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                ASSERT_TRUE(loaded->predict_into(xq_ref, pred).has_value());
                EXPECT_EQ(no_alloc.delta_allocations(), 0U);
            }
        }
        std::filesystem::remove_all(temp_dir, ec);
    }

    /// @brief Enforces zero allocations and relative speed ratchet for supplied-distance evaluation.
    /// @details Compares an n = 96, d = 8 RbfArd + Matern-5/2 DistanceKernel against coordinate
    /// evaluation for zero-allocation apply/grad, numerical parity, and bounded median latency ratio.
    TEST(HotpathRatchetTest, DistanceSupplyZeroAllocAndCoordinateRatioRatchet)
    {
        // Spec: REQ-05, REQ-11, DES-10
        const std::size_t n = 96;
        const std::size_t d = 8;
        std::vector<double> x_train(n * d);
        for (std::size_t col = 0; col < d; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                x_train[(col * n) + row] =
                    std::sin(0.1 * static_cast<double>(row + (col * 3U))) + (0.01 * static_cast<double>(row));
            }
        }
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);

        std::vector<double> ls(d, 1.2);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(ls).value();
        const auto m52 = cxxgpr::MaternKernel::create(1.5, cxxgpr::MaternNu::FiveHalves).value();

        const auto coord_spec = cxxgpr::KernelSpec(rbf_ard) + cxxgpr::KernelSpec(m52);
        const auto coord_ck = coord_spec.compile<double>();

        cxxgpr::ScalarDistance s_slot;
        auto a_slot = cxxgpr::ArdDistance::create(d);
        ASSERT_TRUE(a_slot.has_value());
        auto ard_dist_k = a_slot->rbf_ard(rbf_ard);
        ASSERT_TRUE(ard_dist_k.has_value());
        const auto dist_spec = *ard_dist_k + s_slot.matern(m52);
        const auto dist_ck = dist_spec.compile<double>();

        // Precompute squared Euclidean distances once (outside hot path).
        std::vector<double> d_scalar(n * n, 0.0);
        std::vector<std::vector<double>> d_ard(d, std::vector<double>(n * n, 0.0));
        std::vector<cxxgpr::MatrixRef<double>> ard_refs;
        ard_refs.reserve(d);
        for (std::size_t dim = 0; dim < d; ++dim)
        {
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = 0; row < n; ++row)
                {
                    const double diff = x_train[(dim * n) + row] - x_train[(dim * n) + col];
                    const double sq = diff * diff;
                    d_ard[dim][(col * n) + row] = sq;
                    d_scalar[(col * n) + row] += sq;
                }
            }
            ard_refs.emplace_back(d_ard[dim].data(), n, n, n);
        }

        cxxgpr::DistanceSource src;
        src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(d_scalar.data(), n, n, n))
            .bind_ard(a_slot->slot(), ard_refs);
        const auto store = dist_ck.prepare_square(src);
        ASSERT_TRUE(store.has_value());

        std::vector<double> out_coord(n * n, 0.0);
        std::vector<double> out_dist(n * n, 0.0);
        std::vector<double> scratch_a(n * n, 0.0);
        std::vector<double> scratch_b(n * n, 0.0);
        const auto out_c_mut = cxxgpr::MatrixMut<double>(out_coord.data(), n, n, n);
        const auto out_d_mut = cxxgpr::MatrixMut<double>(out_dist.data(), n, n, n);
        const auto sa_mut = cxxgpr::MatrixMut<double>(scratch_a.data(), n, n, n);
        const auto sb_mut = cxxgpr::MatrixMut<double>(scratch_b.data(), n, n, n);
        auto nested_coord = coord_ck.make_nested_buffers(n, n);
        auto nested_dist = dist_ck.make_nested_buffers(n, n);

        // Zero-allocation audit on distance supply apply and grad.
        ASSERT_TRUE(
            dist_ck.apply(*store, std::nullopt, out_d_mut, cxxgpr::Triangle::Lower, sa_mut, nested_dist).has_value()
        );
        {
            const cxxgpr::test_support::NoAllocationScope no_alloc;
            ASSERT_TRUE(
                dist_ck.apply(*store, std::nullopt, out_d_mut, cxxgpr::Triangle::Lower, sa_mut, nested_dist).has_value()
            );
            ASSERT_TRUE(dist_ck
                    .grad(*store, std::nullopt, out_d_mut, 0U, cxxgpr::Triangle::Lower, sa_mut, sb_mut, nested_dist)
                    .has_value());
            EXPECT_EQ(no_alloc.delta_allocations(), 0U);
        }

        // Warmup coordinate path as well and verify numerical parity.
        ASSERT_TRUE(coord_ck.apply_points(x_ref, out_c_mut, cxxgpr::Triangle::Lower, sa_mut, nested_coord).has_value());
        ASSERT_TRUE(
            dist_ck.apply(*store, std::nullopt, out_d_mut, cxxgpr::Triangle::Lower, sa_mut, nested_dist).has_value()
        );
        EXPECT_NEAR(out_dist[0], out_coord[0], 1e-12);
        EXPECT_NEAR(out_dist[n - 1U], out_coord[n - 1U], 1e-12);

        // Performance ratchet: supplied-distance apply must stay within 2.5x of coordinate
        // apply_points.
        constexpr std::size_t k_trials = 25U;
        std::vector<double> coord_ns(k_trials);
        std::vector<double> dist_ns(k_trials);
        for (std::size_t t = 0; t < k_trials; ++t)
        {
            const auto t0 = std::chrono::steady_clock::now();
            ASSERT_TRUE(
                coord_ck.apply_points(x_ref, out_c_mut, cxxgpr::Triangle::Lower, sa_mut, nested_coord).has_value()
            );
            const auto t1 = std::chrono::steady_clock::now();
            coord_ns[t] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

            const auto t2 = std::chrono::steady_clock::now();
            ASSERT_TRUE(
                dist_ck.apply(*store, std::nullopt, out_d_mut, cxxgpr::Triangle::Lower, sa_mut, nested_dist).has_value()
            );
            const auto t3 = std::chrono::steady_clock::now();
            dist_ns[t] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
            cxxgpr::test_support::touch_for_test(out_coord.data());
            cxxgpr::test_support::touch_for_test(out_dist.data());
        }
        std::ranges::sort(coord_ns);
        std::ranges::sort(dist_ns);
        const double median_coord = coord_ns[k_trials / 2U];
        const double median_dist = dist_ns[k_trials / 2U];
        if (median_coord > 0.0)
        {
            const double ratio = median_dist / median_coord;
#ifdef NDEBUG
            constexpr double k_max_ratio = 2.0;
#else
            constexpr double k_max_ratio = 4.0;
#endif
            EXPECT_LE(ratio, k_max_ratio);
        }
    }

}  // namespace
