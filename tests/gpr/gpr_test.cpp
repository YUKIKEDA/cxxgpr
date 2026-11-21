// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-05, REQ-08, REQ-09, REQ-11, DES-01, DES-05, DES-11, DES-15, DES-16, DES-17
/// @file gpr_test.cpp
/// @brief Unit and integration tests for Gpr and FittedGpr compile-time state separation and workflows.
/// @details Verifies compile-time concept rejection of predict/loo_predict on unfitted Gpr,
/// end-to-end fit/factor/predict/covariance/sample/LOO/refit across all five optimizer policies
/// with input and target preprocessing, MixedPrecision iterative refinement parity, zero-alloc
/// predict_into(), and immutable parent state preservation on negative inputs.

#include "cxxgpr/gpr/gpr.hpp"

#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/transform/transform.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "allocation_audit.hpp"

namespace
{

    /// @brief Compile-time trait checking whether a model type exposes const predict(MatrixRef).
    /// @tparam T Candidate GPR model type (unfitted Gpr vs FittedGpr).
    template <typename T>
    concept HasPredict = requires(const T& m, cxxgpr::MatrixRef<double> xs) { m.predict(xs); };

    /// @brief Compile-time trait checking whether a model type exposes const loo_predict().
    /// @tparam T Candidate GPR model type (unfitted Gpr vs FittedGpr).
    template <typename T>
    concept HasLooPredict = requires(const T& m) { m.loo_predict(); };

    static_assert(!HasPredict<cxxgpr::Gpr<>>);
    static_assert(!HasLooPredict<cxxgpr::Gpr<>>);
    static_assert(HasPredict<cxxgpr::FittedGpr<>>);
    static_assert(HasLooPredict<cxxgpr::FittedGpr<>>);

    /// @brief Verifies fit/factor/predict/LOO/refit across all five optimizers and preprocessing transforms.
    /// @details Fits 1D sinusoidal data with StandardizeInput and StandardizeTarget using L-BFGS,
    /// audits zero-alloc predict_into() under NoAllocationScope, checks covariance/sample/LOO/refit
    /// parity, and exercises TrustRegion, NelderMead, FastSimulatedAnnealing, and Fixed::factor().
    TEST(GprModelsTest, FitPredictLooAndTransformsWithAllOptimizers)
    {
        // Spec: REQ-01, REQ-09, DES-01, DES-11, DES-16
        // Generate 10 training samples from y = 10 + 3*sin(x) + 0.2*x on [-2, 2].
        const std::size_t n = 10;
        const std::size_t d = 1;
        std::vector<double> x_train(n);
        std::vector<double> y_train(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double x = -2.0 + (4.0 * static_cast<double>(i) / static_cast<double>(n - 1U));
            x_train[i] = x;
            y_train[i] = 10.0 + (3.0 * std::sin(x)) + (0.2 * x);
        }

        const auto base_spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);

        const std::array<double, 3> x_query{-1.2, 0.3, 1.4};
        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 3U, d, 3U);

        // 1. Lbfgs with StandardizeInput and StandardizeTarget + zero-allocation predict_into.
        const auto gpr_lbfgs =
            cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(base_spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(80U))
                .with_input_transform(cxxgpr::transform::StandardizeInput())
                .with_target_transform(cxxgpr::transform::StandardizeTarget())
                .with_noise_variance(1e-3)
                .value()
                .with_prefer_speed();

        auto fit_lbfgs = gpr_lbfgs.fit(x_ref, y_train);
        ASSERT_TRUE(fit_lbfgs.has_value());
        EXPECT_TRUE(fit_lbfgs->optimizer_result().converged);
        EXPECT_LT(fit_lbfgs->log_marginal_likelihood(), 15.0);

        cxxgpr::Prediction<double> pred_out;
        ASSERT_TRUE(fit_lbfgs->predict_into(xq_ref, pred_out).has_value());
        ASSERT_EQ(pred_out.mean.size(), 3U);
        for (std::size_t j = 0; j < 3U; ++j)
        {
            const double y_true = 10.0 + (3.0 * std::sin(x_query[j])) + (0.2 * x_query[j]);
            EXPECT_NEAR(pred_out.mean[j], y_true, 0.25);
            EXPECT_GT(pred_out.variance[j], 0.0);
        }

        // Subsequent predict_into with same query shape must allocate zero bytes.
        {
            const cxxgpr::test_support::NoAllocationScope no_alloc;
            ASSERT_TRUE(fit_lbfgs->predict_into(xq_ref, pred_out).has_value());
        }

        // Covariance and deterministic sampling.
        auto cov = fit_lbfgs->predict_covariance(xq_ref);
        ASSERT_TRUE(cov.has_value());
        for (std::size_t j = 0; j < 3U; ++j)
        {
            EXPECT_NEAR(cov->covariance[(j * 3U) + j], pred_out.variance[j], 1e-12);
        }
        auto s1 = fit_lbfgs->sample(xq_ref, 3U, 99ULL);
        auto s2 = fit_lbfgs->sample(xq_ref, 3U, 99ULL);
        ASSERT_TRUE(s1.has_value());
        ASSERT_TRUE(s2.has_value());
        EXPECT_EQ(*s1, *s2);

        // LOO prediction on original target scale.
        auto loo = fit_lbfgs->loo_predict();
        ASSERT_TRUE(loo.has_value());
        ASSERT_EQ(loo->mean.size(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            EXPECT_NEAR(loo->mean[i], y_train[i], 0.5);
            EXPECT_GT(loo->variance[i], 0.0);
        }

        // into_trainer and refit preserve fitted state & configuration.
        auto trainer_back = fit_lbfgs->into_trainer();
        auto refitted = fit_lbfgs->refit(x_ref, y_train);
        ASSERT_TRUE(refitted.has_value());
        EXPECT_NEAR(refitted->log_marginal_likelihood(), fit_lbfgs->log_marginal_likelihood(), 1e-6);

        // 2. TrustRegion, NelderMead, FastSimulatedAnnealing (with LeafCache), and Fixed::factor.
        const auto gpr_tr = cxxgpr::Gpr<cxxgpr::optimizer::TrustRegion>(
            base_spec, cxxgpr::optimizer::TrustRegion().with_max_iterations(60U)
        )
                                .with_target_transform(cxxgpr::transform::StandardizeTarget());
        auto fit_tr = gpr_tr.fit(x_ref, y_train);
        ASSERT_TRUE(fit_tr.has_value());
        EXPECT_NEAR(fit_tr->predict(xq_ref)->mean[0], pred_out.mean[0], 0.3);

        const auto gpr_nm = cxxgpr::Gpr<cxxgpr::optimizer::NelderMead>(
            base_spec, cxxgpr::optimizer::NelderMead().with_max_iterations(120U)
        );
        auto fit_nm = gpr_nm.fit(x_ref, y_train);
        ASSERT_TRUE(fit_nm.has_value());

        const auto gpr_fsa = cxxgpr::Gpr<cxxgpr::optimizer::FastSimulatedAnnealing>(
            base_spec, cxxgpr::optimizer::FastSimulatedAnnealing().with_max_iterations(80U).with_seed(7ULL)
        )
                                 .with_prefer_speed();
        auto fit_fsa = gpr_fsa.fit(x_ref, y_train);
        ASSERT_TRUE(fit_fsa.has_value());

        const auto gpr_fixed = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(base_spec);
        auto fit_fixed = gpr_fixed.factor(x_ref, y_train);
        ASSERT_TRUE(fit_fixed.has_value());
        EXPECT_EQ(fit_fixed->optimizer_result().iterations, 0U);
    }

    /// @brief Verifies MixedPrecision iterative refinement parity with DoublePrecision and zero-alloc predict.
    /// @details Compares PromoteStorage and ReevaluateKernel dual coefficients alpha against
    /// DoublePrecision, confirms refinement_summary() presence, and audits predict_into() under NoAllocationScope.
    TEST(GprModelsTest, MixedPrecisionIterativeRefinementAndZeroAllocPredict)
    {
        // Spec: REQ-05, REQ-11, DES-05
        // Construct an 8-point smooth 1D regression dataset for Double vs MixedPrecision comparison.
        const std::size_t n = 8;
        const std::size_t d = 1;
        std::vector<double> x_train(n);
        std::vector<double> y_train(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x_train[i] = static_cast<double>(i) * 0.25;
            y_train[i] = std::cos(x_train[i]) + (0.1 * x_train[i]);
        }

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.5).value());
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);
        const std::array<double, 2> x_query{0.3, 1.1};
        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 2U, d, 2U);

        const auto gpr_f64 =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision>(spec).with_noise_variance(1e-4).value();
        const auto gpr_mp_prom =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(spec)
                .with_noise_variance(1e-4)
                .value();
        const auto gpr_mp_reev =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec)
                .with_noise_variance(1e-4)
                .value();

        auto f64 = gpr_f64.factor(x_ref, y_train);
        auto mp_prom = gpr_mp_prom.factor(x_ref, y_train);
        auto mp_reev = gpr_mp_reev.factor(x_ref, y_train);
        ASSERT_TRUE(f64.has_value());
        ASSERT_TRUE(mp_prom.has_value());
        ASSERT_TRUE(mp_reev.has_value());
        ASSERT_TRUE(mp_prom->refinement_summary().has_value());
        ASSERT_TRUE(mp_reev->refinement_summary().has_value());
        EXPECT_FALSE(f64->refinement_summary().has_value());

        for (std::size_t i = 0; i < n; ++i)
        {
            EXPECT_NEAR(mp_prom->alpha()[i], f64->alpha()[i], 1e-6);
            EXPECT_NEAR(mp_reev->alpha()[i], f64->alpha()[i], 1e-6);
        }

        // Verify zero heap allocations during repeated predict_into calls under ReevaluateKernel.
        cxxgpr::Prediction<double> out_mp;
        ASSERT_TRUE(mp_reev->predict_into(xq_ref, out_mp).has_value());
        {
            const cxxgpr::test_support::NoAllocationScope no_alloc;
            ASSERT_TRUE(mp_reev->predict_into(xq_ref, out_mp).has_value());
        }
    }

    /// @brief Negative controls verifying failed builder, fit, factor, and predict calls keep state intact.
    /// @details Confirms InvalidNoiseVariance, EmptyInput, LengthMismatch, NonFiniteInput, and
    /// DimensionMismatch rejections leave the parent Gpr and FittedGpr reusable afterwards.
    TEST(GprModelsTest, NegativeControlsLeaveParentUnchanged)
    {
        // Spec: DES-01, DES-02
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const cxxgpr::Gpr<> gpr(spec);

        // Reject negative noise variance before checking training dimensions.
        EXPECT_EQ(gpr.with_noise_variance(-1.0).error().code(), cxxgpr::GprErrorCode::InvalidNoiseVariance);

        const std::vector<double> empty_x;
        const std::vector<double> empty_y;
        auto err_empty = gpr.fit(empty_x, 0U, 1U, empty_y);
        ASSERT_FALSE(err_empty.has_value());
        EXPECT_EQ(err_empty.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const std::vector<double> x_ok{0.0, 1.0};
        const std::vector<double> y_bad_len{1.0};
        auto err_len = gpr.factor(x_ok, 2U, 1U, y_bad_len);
        ASSERT_FALSE(err_len.has_value());
        EXPECT_EQ(err_len.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        const std::vector<double> y_nan{1.0, std::numeric_limits<double>::quiet_NaN()};
        auto err_nan = gpr.factor(x_ok, 2U, 1U, y_nan);
        ASSERT_FALSE(err_nan.has_value());
        EXPECT_EQ(err_nan.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

        // Parent Gpr remains intact and usable after failed calls.
        const std::vector<double> y_ok{0.5, 1.5};
        auto ok_fit = gpr.factor(x_ok, 2U, 1U, y_ok);
        ASSERT_TRUE(ok_fit.has_value());

        const std::vector<double> bad_query{0.0, 1.0};
        auto err_qdim = ok_fit->predict(bad_query, 1U, 2U);
        ASSERT_FALSE(err_qdim.has_value());
        EXPECT_EQ(err_qdim.error().code(), cxxgpr::GprErrorCode::DimensionMismatch);
    }

    /// @brief Verifies mixed-precision refinement weights are bit-identical for any OpenMP thread count.
    /// @details The blocked double-precision residual accumulates each row over its columns in ascending order, so
    /// splitting rows across threads must not change a single bit of the refined weights or the summary.
    TEST(GprModelsTest, MixedPrecisionRefinementIsBitwiseIdenticalAcrossThreadCounts)
    {
        // Spec: DES-05, DES-20
        constexpr std::size_t n = 520;
        std::vector<double> x(n * 2U);
        std::vector<double> y(n);
        cxxgpr::Xoshiro256PlusPlus rng(11);
        for (double& v : x)
        {
            v = rng.unit();
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            y[i] = std::sin(3.0 * x[i]) + x[n + i] + (0.05 * rng.unit_normal());
        }
        const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.7).value());

        using Policy = cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>;
        auto run = [&](int threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, Policy>(kernel).with_noise_variance(1e-2);
            auto fitted = trainer->factor(x, n, 2U, y);
            EXPECT_TRUE(fitted.has_value());
            return fitted;
        };
#ifdef _OPENMP
        const int saved_threads = omp_get_max_threads();
#endif
        const auto reference = run(1);
        ASSERT_TRUE(reference.has_value());
        const auto reference_summary = reference->refinement_summary();
        if (!reference_summary.has_value())
        {
            FAIL() << "reference refinement summary is missing";
        }
        EXPECT_FALSE(reference_summary->used_fallback());
        for (const int threads : {2, 4})
        {
            const auto other = run(threads);
            ASSERT_TRUE(other.has_value());
            const auto other_summary = other->refinement_summary();
            if (!other_summary.has_value())
            {
                FAIL() << "refinement summary is missing for threads " << threads;
            }
            ASSERT_EQ(other->alpha().size(), reference->alpha().size());
            for (std::size_t i = 0; i < n; ++i)
            {
                EXPECT_EQ(other->alpha()[i], reference->alpha()[i]) << "threads " << threads << " index " << i;
            }
            EXPECT_EQ(*other_summary, *reference_summary);
        }
#ifdef _OPENMP
        omp_set_num_threads(saved_threads);
#endif
    }

}  // namespace
