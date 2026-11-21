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
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "allocation_audit.hpp"
#include "span_shape_contract.hpp"

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
        EXPECT_LT(fit_lbfgs->neg_log_marginal_likelihood(), 15.0);

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
        EXPECT_NEAR(refitted->neg_log_marginal_likelihood(), fit_lbfgs->neg_log_marginal_likelihood(), 1e-6);

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

    /// @brief Checks that query batch sizes may shrink and grow in any order on one fitted model.
    /// @details Builds a 2-feature model from trainer, grows its query workspace with an 8-point batch, and
    /// compares the following 3-point batch bitwise against a separately factored model that never saw the
    /// larger batch, for predict, predict_covariance, and sample. Every comparison uses its own reference
    /// model. Finally audits that predict_into() allocates nothing for a smaller batch after a larger one.
    /// @tparam P Precision policy of the trainer.
    /// @param[in] trainer Unfitted trainer (kernel, noise, and preprocessing) factored repeatedly.
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_query_batch_may_shrink(const cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>& trainer)
    {
        // Twenty training points in two features (column-major) with a smooth target.
        const std::size_t n = 20;
        const std::size_t d = 2;
        std::vector<double> x(n * d);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = 0.3 * static_cast<double>(i);
            x[n + i] = 0.25 * static_cast<double>((i * 5U) % 7U);
            y[i] = 4.0 + std::sin(x[i]) + (0.5 * x[n + i]);
        }
        // Column-major query batches: 8 x 2 and 3 x 2. The second feature column of the small batch is
        // staged where the first column of the big batch was, so stale workspace contents would show.
        const std::vector<double> big{0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 0.2, 1.4, 0.6, 1.0, 0.3, 1.2, 0.8, 0.4};
        const std::vector<double> small{0.2, 1.1, 2.3, 0.7, 0.1, 1.3};
        const auto big_ref = cxxgpr::MatrixRef<double>(big.data(), 8U, d, 8U);
        const auto small_ref = cxxgpr::MatrixRef<double>(small.data(), 3U, d, 3U);

        // factor() is deterministic, so each call yields an identical model with an empty query workspace.
        const auto make = [&]
        {
            return trainer.factor(x, n, d, y);
        };

        // predict: large -> small -> large on one model; each reference model sees only one batch.
        {
            auto used = make();
            auto never_big = make();
            auto never_small = make();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->predict(big_ref).has_value());
            auto shrunk = used->predict(small_ref);
            auto shrunk_expected = never_big->predict(small_ref);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(shrunk->mean, shrunk_expected->mean);
            EXPECT_EQ(shrunk->variance, shrunk_expected->variance);

            auto regrown = used->predict(big_ref);
            auto regrown_expected = never_small->predict(big_ref);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(regrown->mean, regrown_expected->mean);
            EXPECT_EQ(regrown->variance, regrown_expected->variance);
        }

        // predict_covariance takes the same preparation path.
        {
            auto used = make();
            auto never_big = make();
            auto never_small = make();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->predict_covariance(big_ref).has_value());
            auto shrunk = used->predict_covariance(small_ref);
            auto shrunk_expected = never_big->predict_covariance(small_ref);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(shrunk->mean, shrunk_expected->mean);
            EXPECT_EQ(shrunk->covariance, shrunk_expected->covariance);

            auto regrown = used->predict_covariance(big_ref);
            auto regrown_expected = never_small->predict_covariance(big_ref);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(regrown->mean, regrown_expected->mean);
            EXPECT_EQ(regrown->covariance, regrown_expected->covariance);
        }

        // sample with a fixed seed.
        {
            auto used = make();
            auto never_big = make();
            auto never_small = make();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->sample(big_ref, 2U, 7U).has_value());
            auto shrunk = used->sample(small_ref, 2U, 7U);
            auto shrunk_expected = never_big->sample(small_ref, 2U, 7U);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(*shrunk, *shrunk_expected);

            auto regrown = used->sample(big_ref, 2U, 7U);
            auto regrown_expected = never_small->sample(big_ref, 2U, 7U);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(*regrown, *regrown_expected);
        }

        // predict_into: once the larger batch has sized the workspace and the output, a smaller batch and
        // a return to the larger batch perform no heap allocation and still match the reference values.
        {
            auto used = make();
            auto never_big = make();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            cxxgpr::Prediction<typename P::Refine> out;
            ASSERT_TRUE(used->predict_into(big_ref, out).has_value());
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                const auto res = used->predict_into(small_ref, out);
                const std::size_t allocations = no_alloc.delta_allocations();
                ASSERT_TRUE(res.has_value());
                EXPECT_EQ(allocations, 0U);
            }
            auto shrunk_expected = never_big->predict(small_ref);
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(out.mean, shrunk_expected->mean);
            EXPECT_EQ(out.variance, shrunk_expected->variance);
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                const auto res = used->predict_into(big_ref, out);
                const std::size_t allocations = no_alloc.delta_allocations();
                ASSERT_TRUE(res.has_value());
                EXPECT_EQ(allocations, 0U);
            }
        }
    }

    /// @brief Runs verify_query_batch_may_shrink() without and with fitted input and target transforms.
    /// @tparam P Precision policy under test.
    template <cxxgpr::PrecisionPolicyConcept P> void verify_query_batch_may_shrink_for_policy()
    {
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 1.5}).value());
        const auto identity = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec).with_noise_variance(1e-3);
        ASSERT_TRUE(identity.has_value());
        {
            SCOPED_TRACE("identity transforms");
            verify_query_batch_may_shrink<P>(*identity);
        }
        {
            SCOPED_TRACE("StandardizeInput + StandardizeTarget");
            verify_query_batch_may_shrink<P>(identity->with_input_transform(cxxgpr::transform::StandardizeInput())
                    .with_target_transform(cxxgpr::transform::StandardizeTarget()));
        }
    }

    /// @brief Verifies prediction batch sizes may shrink and grow in any order on one fitted model.
    /// @details The query workspace only grows; a smaller batch after a larger one must give the same values as a
    /// model that never saw the larger batch (bitwise) for predict, predict_covariance, and sample, and
    /// predict_into must not allocate for it. Covers all four precision policies, each without and with
    /// fitted input and target transforms.
    TEST(GprModelsTest, QueryBatchSizeMayShrinkAfterLargerBatch)
    {
        // Spec: REQ-07, REQ-10, REQ-11, DES-05, DES-16, DES-17, DES-18
        {
            SCOPED_TRACE("DoublePrecision");
            verify_query_batch_may_shrink_for_policy<cxxgpr::DoublePrecision>();
        }
        {
            SCOPED_TRACE("SinglePrecision");
            verify_query_batch_may_shrink_for_policy<cxxgpr::SinglePrecision>();
        }
        {
            SCOPED_TRACE("MixedPrecision<PromoteStorage>");
            verify_query_batch_may_shrink_for_policy<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>();
        }
        {
            SCOPED_TRACE("MixedPrecision<ReevaluateKernel>");
            verify_query_batch_may_shrink_for_policy<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>();
        }
    }

    /// @brief Verifies every raw-span overload of Gpr and FittedGpr against the span shape contract.
    /// @details Runs fit, factor, refit, predict, predict_into, predict_covariance, and sample through the shared
    /// shape cases (short span, long span, overflowing dimensions, empty span with a null or non-null pointer, zero
    /// dimension) and checks that results of the parent objects are bitwise unchanged after the rejected calls.
    TEST(GprModelsTest, SpanOverloadsRejectLengthsThatDisagreeWithDimensions)
    {
        // Spec: REQ-02, REQ-03, DES-01, DES-02
        using cxxgpr::test_support::error_code_of;
        using cxxgpr::test_support::expect_span_shape_contract;
        using Span = std::span<const double>;

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const cxxgpr::Gpr<cxxgpr::optimizer::Fixed> gpr(spec);
        const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> y{0.0, 1.0, 0.5, 1.2};
        const std::vector<double> query{0.5, 1.5};

        auto model = gpr.factor(x, 4U, 1U, y);
        ASSERT_TRUE(model.has_value());
        const auto pred_before = model->predict(query, 2U, 1U);
        const auto cov_before = model->predict_covariance(query, 2U, 1U);
        const auto draws_before = model->sample(query, 2U, 1U, 3U, 5U);
        ASSERT_TRUE(pred_before.has_value());
        ASSERT_TRUE(cov_before.has_value());
        ASSERT_TRUE(draws_before.has_value());

        // Training overloads: the trainer and the fitted model are const and stay usable.
        {
            SCOPED_TRACE("Gpr::fit");
            expect_span_shape_contract([&](Span xs, std::size_t n_rows, std::size_t n_cols) {
                return error_code_of(gpr.fit(xs, n_rows, n_cols, y));
            }, x, 4U, 1U);
        }
        {
            SCOPED_TRACE("Gpr::factor");
            expect_span_shape_contract([&](Span xs, std::size_t n_rows, std::size_t n_cols) {
                return error_code_of(gpr.factor(xs, n_rows, n_cols, y));
            }, x, 4U, 1U);
        }
        {
            SCOPED_TRACE("FittedGpr::refit");
            expect_span_shape_contract([&](Span xs, std::size_t n_rows, std::size_t n_cols) {
                return error_code_of(model->refit(xs, n_rows, n_cols, y));
            }, x, 4U, 1U);
        }

        // Query overloads.
        {
            SCOPED_TRACE("FittedGpr::predict");
            expect_span_shape_contract([&](Span xs, std::size_t n_rows, std::size_t n_cols) {
                return error_code_of(model->predict(xs, n_rows, n_cols));
            }, query, 2U, 1U);
        }
        {
            SCOPED_TRACE("FittedGpr::predict_into");
            cxxgpr::Prediction<double> out;
            ASSERT_TRUE(model->predict_into(query, 2U, 1U, out).has_value());
            const cxxgpr::Prediction<double> out_before = out;
            expect_span_shape_contract(
                [&](Span xs, std::size_t n_rows, std::size_t n_cols)
            {
                const auto code = error_code_of(model->predict_into(xs, n_rows, n_cols, out));
                // A rejected call leaves the caller's output untouched.
                EXPECT_EQ(out, out_before);
                return code;
            }, query, 2U, 1U
            );
        }
        {
            SCOPED_TRACE("FittedGpr::predict_covariance");
            expect_span_shape_contract([&](Span xs, std::size_t n_rows, std::size_t n_cols) {
                return error_code_of(model->predict_covariance(xs, n_rows, n_cols));
            }, query, 2U, 1U);
        }
        {
            SCOPED_TRACE("FittedGpr::sample");
            expect_span_shape_contract([&](Span xs, std::size_t n_rows, std::size_t n_cols) {
                return error_code_of(model->sample(xs, n_rows, n_cols, 3U, 5U));
            }, query, 2U, 1U);
        }

        // The fitted model answers exactly as it did before the rejected calls.
        const auto pred_after = model->predict(query, 2U, 1U);
        const auto cov_after = model->predict_covariance(query, 2U, 1U);
        const auto draws_after = model->sample(query, 2U, 1U, 3U, 5U);
        ASSERT_TRUE(pred_after.has_value());
        ASSERT_TRUE(cov_after.has_value());
        ASSERT_TRUE(draws_after.has_value());
        EXPECT_EQ(*pred_after, *pred_before);
        EXPECT_EQ(cov_after->mean, cov_before->mean);
        EXPECT_EQ(cov_after->covariance, cov_before->covariance);
        EXPECT_EQ(*draws_after, *draws_before);
    }

    /// @brief Pins the validation order of the span overloads across arguments and against later checks.
    /// @details An empty target vector is reported as EmptyInput before any size error of the feature span, a
    /// feature span of the wrong length is reported before a target of the wrong length is inspected further, and
    /// a query span of the wrong length is reported as LengthMismatch before the feature count is compared.
    TEST(GprModelsTest, SpanOverloadsReportViolationsInValidationOrder)
    {
        // Spec: REQ-03, DES-02
        using cxxgpr::GprErrorCode;
        using cxxgpr::test_support::error_code_of;
        using Code = std::optional<GprErrorCode>;
        using Span = std::span<const double>;

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const cxxgpr::Gpr<cxxgpr::optimizer::Fixed> gpr(spec);
        const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> y{0.0, 1.0, 0.5, 1.2};
        const Span x_short = Span(x).first(3U);
        const Span y_short = Span(y).first(3U);
        const Span y_empty;
        constexpr std::size_t huge = std::numeric_limits<std::size_t>::max();

        auto model = gpr.factor(x, 4U, 1U, y);
        ASSERT_TRUE(model.has_value());

        /// @brief Checks the cross-argument order of one training overload.
        const auto check_training = [&](const char* name, const auto& train)
        {
            SCOPED_TRACE(name);
            EXPECT_EQ(train(x, 4U, 1U, y_empty), Code{GprErrorCode::EmptyInput}) << "empty y alone";
            EXPECT_EQ(train(x_short, 4U, 1U, y_empty), Code{GprErrorCode::EmptyInput}) << "empty y before short x";
            EXPECT_EQ(train(x, huge, 2U, y_empty), Code{GprErrorCode::EmptyInput}) << "empty y before overflow";
            EXPECT_EQ(train(x, huge, 2U, y_short), Code{GprErrorCode::SizeOverflow}) << "overflow before short y";
            EXPECT_EQ(train(x_short, 4U, 1U, y_short), Code{GprErrorCode::LengthMismatch}) << "short x and y";
            EXPECT_EQ(train(x, 4U, 1U, y_short), Code{GprErrorCode::LengthMismatch}) << "short y alone";
            EXPECT_EQ(train(x, 4U, 1U, y), Code{}) << "valid input after the rejected calls";
        };
        check_training("Gpr::fit", [&](Span xs, std::size_t n_rows, std::size_t n_cols, Span ys) {
            return error_code_of(gpr.fit(xs, n_rows, n_cols, ys));
        });
        check_training("Gpr::factor", [&](Span xs, std::size_t n_rows, std::size_t n_cols, Span ys) {
            return error_code_of(gpr.factor(xs, n_rows, n_cols, ys));
        });
        check_training("FittedGpr::refit", [&](Span xs, std::size_t n_rows, std::size_t n_cols, Span ys) {
            return error_code_of(model->refit(xs, n_rows, n_cols, ys));
        });

        // Three values declared as 2 x 2: both the length and the feature count are wrong; the length comes first.
        const std::vector<double> q3{0.5, 1.5, 2.5};
        cxxgpr::Prediction<double> out;
        EXPECT_EQ(error_code_of(model->predict(q3, 2U, 2U)), Code{GprErrorCode::LengthMismatch});
        EXPECT_EQ(error_code_of(model->predict_into(q3, 2U, 2U, out)), Code{GprErrorCode::LengthMismatch});
        EXPECT_EQ(error_code_of(model->predict_covariance(q3, 2U, 2U)), Code{GprErrorCode::LengthMismatch});
        EXPECT_EQ(error_code_of(model->sample(q3, 2U, 2U, 1U, 5U)), Code{GprErrorCode::LengthMismatch});
        // With a consistent length the feature count is the remaining violation.
        const std::vector<double> q4{0.5, 1.5, 2.5, 3.5};
        EXPECT_EQ(error_code_of(model->predict(q4, 2U, 2U)), Code{GprErrorCode::DimensionMismatch});
    }

    /// @brief Verifies L-BFGS reaches the independent reference optimum, and reports convergence, from every start.
    /// @details A first step along the raw gradient used to jump to the saturated bounds and stop on a flat plateau
    /// that was reported as converged (negative log marginal likelihood about 200 or 210 instead of about -23.9), and
    /// runs that reached the optimum were reported as not converged. Every start must now converge to the optimum.
    TEST(GprModelsTest, LbfgsConvergesToTheSameOptimumFromEveryStartingLengthScale)
    {
        // Spec: REQ-09, DES-19
        constexpr std::size_t n = 30;
        std::vector<double> x(n);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = static_cast<double>(i) / static_cast<double>(n - 1);
            // Deterministic noise-like term, reproducible outside the library.
            const double noise = (static_cast<double>((37U * i) % 29U) / 14.0) - 1.0;
            y[i] = std::sin(6.0 * x[i]) + (0.5 * x[i]) + (0.05 * noise);
        }

        // Independent reference: scikit-learn 1.9.1 GaussianProcessRegressor (ConstantKernel * RBF + WhiteKernel,
        // bounds (1e-5, 1e5), normalize_y) with 50 restarts, polished by SciPy L-BFGS-B at gtol 1e-12, generated by
        // tools/qa/lbfgs_start_reference.py. Parameter order: log constant, log length scale, log noise variance.
        constexpr double ref_nll = -23.915244185702573;
        constexpr std::array<double, 3> ref_theta{2.0174208885645295, -0.95025549306360768, -5.8552151580189555};
        // The reference gradient norm there is 3.5e-6 and the smallest Hessian eigenvalue 0.768, so the reference
        // parameters are uncertain by 3.5e-6 / 0.768 = 4.6e-6; a converged run adds at most 1.2e-7. Its value is
        // uncertain by (3.5e-6)^2 / (2 * 0.768) = 8e-12 plus the rounding noise of the two objectives (about 2e-12
        // each, measured as the spread of values along a line search); 1e-10 leaves a margin of about 8.
        constexpr double theta_tol = 5e-6;
        constexpr double nll_tol = 1e-10;

        for (const double start : {0.05, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0})
        {
            const std::array<double, 1> length_scale{start};
            const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scale).value());
            auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(1e-2);
            ASSERT_TRUE(trainer.has_value());
            const auto configured = trainer->with_target_transform(cxxgpr::transform::StandardizeTarget{});
            auto fitted = configured.fit(x, n, 1U, y);
            ASSERT_TRUE(fitted.has_value()) << fitted.error().message();
            const auto& result = fitted->optimizer_result();
            EXPECT_TRUE(result.converged) << "start " << start;
            EXPECT_NEAR(result.value, ref_nll, nll_tol) << "start " << start;
            ASSERT_EQ(result.params.size(), ref_theta.size());
            for (std::size_t k = 0; k < ref_theta.size(); ++k)
            {
                EXPECT_NEAR(result.params[k], ref_theta[k], theta_tol) << "start " << start << " parameter " << k;
            }
        }
    }

}  // namespace
