// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-05, REQ-08, REQ-09, REQ-11, DES-01, DES-02, DES-05, DES-11, DES-15, DES-16, DES-17, DES-19
/// @file gpr_test.cpp
/// @brief Unit and integration tests for Gpr and FittedGpr compile-time state separation and workflows.
/// @details Verifies compile-time concept rejection of predict/loo_predict on unfitted Gpr,
/// end-to-end fit/factor/predict/covariance/sample/LOO/refit across all five optimizer policies
/// with input and target preprocessing, MixedPrecision iterative refinement parity, zero-alloc
/// predict_into(), immutable parent state preservation on negative inputs, non-convergence reported as a
/// successful fit, and unchanged propagation of user-defined optimizer outcomes.

#include "cxxgpr/gpr/gpr.hpp"

#include "cxxgpr/core/build_info.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/transform/transform.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "allocation_audit.hpp"
#include "small_stack.hpp"
#include "span_shape_contract.hpp"

namespace
{

    /// @brief Returns true when the compiled library links an external BLAS backend (OpenBLAS or MKL).
    /// @return True if `build_info().dependencies` reports an external vendor `blas_backend`, false otherwise.
    [[nodiscard]] bool has_external_blas_backend() noexcept
    {
        const auto info = cxxgpr::build_info();
        for (const auto& dep : info.dependencies)
        {
            if (dep.name == "blas_backend")
            {
                return dep.version != "BuiltIn";
            }
        }
        return false;
    }

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

    /// @brief User-defined optimizer that evaluates the objective once and then gives up with a GprError.
    /// @details Models a custom optimizer reporting non-convergence through
    /// GprError::optimization_not_converged() instead of a successful OptimizerResult.
    struct GiveUpOptimizer
    {
        /// @brief Iteration count reported in the returned error payload.
        static constexpr std::size_t k_iterations = 7U;
        /// @brief Declares that the optimizer does not supply changed parameter indices.
        static constexpr bool uses_change_indices = false;

        /// @brief Evaluates obj at initial_params and returns OptimizationNotConverged.
        /// @param[in,out] obj Objective supplied by Gpr::fit().
        /// @param[in] initial_params Starting log-space parameter vector.
        /// @return The objective error if the evaluation fails, otherwise OptimizationNotConverged.
        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable& obj, std::span<const double> initial_params
        ) const
        {
            if (auto v = obj.value(initial_params); !v.has_value())
            {
                return std::unexpected(std::move(v).error());
            }
            return std::unexpected(cxxgpr::GprError::optimization_not_converged(k_iterations));
        }
    };

    /// @brief User-defined optimizer that stops without converging but returns a usable point.
    /// @details Returns initial_params with converged == false, as the built-in optimizers do at their limit.
    struct StopEarlyOptimizer
    {
        /// @brief Iteration count reported in the returned OptimizerResult.
        static constexpr std::size_t k_iterations = 4U;
        /// @brief Declares that the optimizer does not supply changed parameter indices.
        static constexpr bool uses_change_indices = false;

        /// @brief Evaluates obj at initial_params and returns it as a non-converged result.
        /// @param[in,out] obj Objective supplied by Gpr::fit().
        /// @param[in] initial_params Starting log-space parameter vector.
        /// @return OptimizerResult at initial_params with converged == false, or the objective error.
        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable& obj, std::span<const double> initial_params
        ) const
        {
            auto v = obj.value(initial_params);
            if (!v.has_value())
            {
                return std::unexpected(std::move(v).error());
            }
            return cxxgpr::optimizer::OptimizerResult{
                .params = std::vector<double>(initial_params.begin(), initial_params.end()),
                .value = *v,
                .iterations = k_iterations,
                .converged = false,
            };
        }
    };

    /// @brief Training data y = 10 + 3*sin(x) + 0.2*x sampled at 10 points on [-2, 2] (1 feature).
    struct SineData
    {
        /// @brief Column-major training features (n x 1).
        std::vector<double> x;
        /// @brief Training targets of length n.
        std::vector<double> y;
    };

    /// @brief Builds the 10-point sinusoidal training set shared by the optimizer-outcome tests.
    /// @return Training features and targets.
    SineData make_sine_data()
    {
        constexpr std::size_t n = 10U;
        SineData data{.x = std::vector<double>(n), .y = std::vector<double>(n)};
        for (std::size_t i = 0; i < n; ++i)
        {
            const double x = -2.0 + (4.0 * static_cast<double>(i) / static_cast<double>(n - 1U));
            data.x[i] = x;
            data.y[i] = 10.0 + (3.0 * std::sin(x)) + (0.2 * x);
        }
        return data;
    }

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

        // ReevaluateKernel reaches the double-precision weights. PromoteStorage solves the float-stored system, so
        // its weights keep the storage rounding (relative to the largest weight, amplified by the conditioning).
        double alpha_max = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            alpha_max = std::max(alpha_max, std::abs(f64->alpha()[i]));
        }
        const auto prom_summary = mp_prom->refinement_summary();
        if (!prom_summary.has_value())
        {
            FAIL() << "PromoteStorage refinement summary is missing";
        }
        EXPECT_FALSE(prom_summary->used_fallback());
        for (std::size_t i = 0; i < n; ++i)
        {
            EXPECT_NEAR(mp_prom->alpha()[i], f64->alpha()[i], 1e-3 * alpha_max);
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

    /// @brief Restores the OpenMP thread count of the calling thread when the scope ends.
    /// @details Fatal GoogleTest assertions return from the test body early, so a test that changes the thread
    /// count restores it from a destructor instead of from a statement at the end of the body.
    class OmpThreadCountRestorer
    {
    public:
        /// @brief Records the current OpenMP thread count.
        OmpThreadCountRestorer() noexcept = default;
        /// @brief Deleted copy constructor; a copy would restore the thread count twice.
        OmpThreadCountRestorer(const OmpThreadCountRestorer&) = delete;
        /// @brief Deleted copy assignment; a copy would restore the thread count twice.
        /// @return Reference to *this (deleted).
        OmpThreadCountRestorer& operator=(const OmpThreadCountRestorer&) = delete;
        /// @brief Deleted move constructor; the guard stays bound to the scope that created it.
        OmpThreadCountRestorer(OmpThreadCountRestorer&&) = delete;
        /// @brief Deleted move assignment; the guard stays bound to the scope that created it.
        /// @return Reference to *this (deleted).
        OmpThreadCountRestorer& operator=(OmpThreadCountRestorer&&) = delete;

        /// @brief Sets the OpenMP thread count back to the recorded value.
        ~OmpThreadCountRestorer()
        {
#ifdef _OPENMP
            omp_set_num_threads(saved_threads_);
#endif
        }

    private:
#ifdef _OPENMP
        /// @brief Thread count in effect when the guard was constructed.
        int saved_threads_{omp_get_max_threads()};
#endif
    };

    /// @brief Verifies MixedPrecision iterative refinement is deterministic and consistent across OpenMP threads.
    /// @details Confirms repeated 4-thread fits produce bit-identical `alpha` and summaries, while 1/2/4-thread fits
    /// match bitwise under BuiltIn SIMD linear algebra or within 1e-8 relative tolerance under external vendor BLAS.
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
        const OmpThreadCountRestorer restore_threads;
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
        const auto reference = run(1);
        ASSERT_TRUE(reference.has_value());
        const auto reference_summary = reference->refinement_summary();
        if (!reference_summary.has_value())
        {
            FAIL() << "reference refinement summary is missing";
        }
        EXPECT_FALSE(reference_summary->used_fallback());
        const auto rep4 = run(4);
        const auto rep4_repeat = run(4);
        ASSERT_TRUE(rep4.has_value());
        ASSERT_TRUE(rep4_repeat.has_value());
        ASSERT_TRUE(rep4->refinement_summary().has_value());
        ASSERT_TRUE(rep4_repeat->refinement_summary().has_value());
        ASSERT_EQ(rep4->alpha().size(), n);
        ASSERT_EQ(rep4_repeat->alpha().size(), n);
        for (std::size_t i = 0; i < n; ++i)
        {
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(rep4->alpha()[i]), std::bit_cast<std::uint64_t>(rep4_repeat->alpha()[i])
            ) << "repeat index "
              << i;
        }
        EXPECT_EQ(*rep4->refinement_summary(), *rep4_repeat->refinement_summary());

        const bool ext_blas = has_external_blas_backend();
        for (const int threads : {2, 4})
        {
            const auto other = (threads == 4) ? rep4 : run(threads);
            ASSERT_TRUE(other.has_value());
            const auto other_summary = other->refinement_summary();
            if (!other_summary.has_value())
            {
                FAIL() << "refinement summary is missing for threads " << threads;
            }
            ASSERT_EQ(other->alpha().size(), reference->alpha().size());
            if (!ext_blas)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    EXPECT_EQ(other->alpha()[i], reference->alpha()[i]) << "threads " << threads << " index " << i;
                }
                EXPECT_EQ(*other_summary, *reference_summary);
            }
            else
            {
                EXPECT_EQ(other_summary->status, reference_summary->status);
                EXPECT_FALSE(other_summary->used_fallback());
                for (std::size_t i = 0; i < n; ++i)
                {
                    EXPECT_NEAR(
                        other->alpha()[i], reference->alpha()[i], 1e-8 * std::max(1.0, std::abs(reference->alpha()[i]))
                    ) << "threads "
                      << threads << " index " << i;
                }
            }
        }
    }

    /// @brief Verifies PromoteStorage refinement is accepted against the float storage rounding, not the f64 tolerance.
    /// @details The iterate solves the float-stored system to double accuracy, so its residual against the exact
    /// kernel is bounded by a few float epsilons. The fit must end Converged without the double-precision fallback,
    /// and the weights must stay close to the double-precision solution (within the conditioning-amplified storage
    /// error) and closer than a plain single-precision solve.
    TEST(GprModelsTest, PromoteStorageRefinementIsAcceptedAtStorageRoundingAndImprovesOnSingle)
    {
        // Spec: DES-05
        constexpr std::size_t n = 400;
        constexpr std::size_t d = 3;
        std::vector<double> x(n * d);
        std::vector<double> y(n);
        cxxgpr::Xoshiro256PlusPlus rng(17);
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
        // Noise variance 1e-3 gives a condition number large enough for float storage to matter.
        auto double_fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(kernel).with_noise_variance(1e-3)->factor(x, n, d, y);
        auto single_fit =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::SinglePrecision>(kernel).with_noise_variance(1e-3)->factor(
                x, n, d, y
            );
        using Promote = cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>;
        auto promote_fit =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, Promote>(kernel).with_noise_variance(1e-3)->factor(x, n, d, y);
        ASSERT_TRUE(double_fit.has_value());
        ASSERT_TRUE(single_fit.has_value());
        ASSERT_TRUE(promote_fit.has_value());

        const auto promote_summary = promote_fit->refinement_summary();
        if (!promote_summary.has_value())
        {
            FAIL() << "PromoteStorage refinement summary is missing";
        }
        EXPECT_EQ(promote_summary->status, cxxgpr::RefinementStatus::Converged);
        EXPECT_FALSE(promote_summary->used_fallback());

        double alpha_max = 0.0;
        double promote_err = 0.0;
        double single_err = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = double_fit->alpha()[i];
            alpha_max = std::max(alpha_max, std::abs(ref));
            promote_err = std::max(promote_err, std::abs(ref - promote_fit->alpha()[i]));
            single_err = std::max(single_err, std::abs(ref - static_cast<double>(single_fit->alpha()[i])));
        }
        EXPECT_LT(promote_err / alpha_max, 1e-2);
        EXPECT_LT(promote_err, single_err);
    }

    /// @brief Verifies PromoteStorage still falls back to double precision when the stored matrix misses the kernel.
    /// @details The inputs sit near 1000, where rounding a coordinate to float moves it by up to about 3e-5, so the
    /// float-stored kernel matrix differs from the exact kernel by far more than its storage rounding. Refinement
    /// converges on the stored system, but the residual against the exact kernel exceeds the storage-rounding
    /// threshold. The fit must report RejectedByAccept and return the weights of the double-precision solve.
    TEST(GprModelsTest, PromoteStorageFallsBackWhenStoredMatrixMissesTheExactKernel)
    {
        // Spec: DES-05
        constexpr std::size_t n = 200;
        constexpr std::size_t d = 2;
        std::vector<double> x(n * d);
        std::vector<double> y(n);
        cxxgpr::Xoshiro256PlusPlus rng(23);
        for (double& v : x)
        {
            v = 1000.0 + rng.unit();
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            y[i] = std::sin(3.0 * (x[i] - 1000.0)) + (x[n + i] - 1000.0) + (0.05 * rng.unit_normal());
        }
        const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.05).value());
        auto double_fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(kernel).with_noise_variance(1e-2)->factor(x, n, d, y);
        using Promote = cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>;
        auto promote_fit =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, Promote>(kernel).with_noise_variance(1e-2)->factor(x, n, d, y);
        ASSERT_TRUE(double_fit.has_value());
        ASSERT_TRUE(promote_fit.has_value());

        const auto promote_summary = promote_fit->refinement_summary();
        if (!promote_summary.has_value())
        {
            FAIL() << "PromoteStorage refinement summary is missing";
        }
        EXPECT_EQ(promote_summary->status, cxxgpr::RefinementStatus::RejectedByAccept);
        EXPECT_TRUE(promote_summary->used_fallback());

        double alpha_max = 0.0;
        double fallback_err = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ref = double_fit->alpha()[i];
            alpha_max = std::max(alpha_max, std::abs(ref));
            fallback_err = std::max(fallback_err, std::abs(ref - promote_fit->alpha()[i]));
        }
        EXPECT_LE(fallback_err, 1e-10 * alpha_max);
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

    /// @brief Verifies that the built-in optimizers report non-convergence as a successful fit.
    /// @details Limits Lbfgs (also with restarts), TrustRegion, and NelderMead to one iteration so that their
    /// stopping tests cannot hold, and checks that fit() succeeds with converged == false and the completed
    /// iteration count. FastSimulatedAnnealing runs its whole schedule and Fixed performs no iterations; both
    /// report converged == true.
    TEST(GprModelsTest, BuiltInOptimizersReportNonConvergenceAsSuccessfulFit)
    {
        // Spec: DES-02, DES-19
        const SineData data = make_sine_data();
        const std::size_t n = data.y.size();
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x.data(), n, 1U, n);

        // One iteration from the (1, 1, 1e-2) start cannot meet the default sqrt(eps) tolerance.
        const auto lbfgs =
            cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(1U))
                .fit(x_ref, data.y);
        if (!lbfgs.has_value())
        {
            FAIL() << "Lbfgs fit failed: " << lbfgs.error().message();
        }
        EXPECT_FALSE(lbfgs->optimizer_result().converged);
        EXPECT_EQ(lbfgs->optimizer_result().iterations, 1U);

        const auto lbfgs_restarts = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(
            spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(1U).with_restarts(2U, 11ULL)
        )
                                        .fit(x_ref, data.y);
        if (!lbfgs_restarts.has_value())
        {
            FAIL() << "Lbfgs fit with restarts failed: " << lbfgs_restarts.error().message();
        }
        EXPECT_FALSE(lbfgs_restarts->optimizer_result().converged);
        EXPECT_EQ(lbfgs_restarts->optimizer_result().iterations, 1U);

        const auto tr =
            cxxgpr::Gpr<cxxgpr::optimizer::TrustRegion>(spec, cxxgpr::optimizer::TrustRegion().with_max_iterations(1U))
                .fit(x_ref, data.y);
        if (!tr.has_value())
        {
            FAIL() << "TrustRegion fit failed: " << tr.error().message();
        }
        EXPECT_FALSE(tr->optimizer_result().converged);
        EXPECT_EQ(tr->optimizer_result().iterations, 1U);

        const auto nm =
            cxxgpr::Gpr<cxxgpr::optimizer::NelderMead>(spec, cxxgpr::optimizer::NelderMead().with_max_iterations(1U))
                .fit(x_ref, data.y);
        if (!nm.has_value())
        {
            FAIL() << "NelderMead fit failed: " << nm.error().message();
        }
        EXPECT_FALSE(nm->optimizer_result().converged);
        EXPECT_EQ(nm->optimizer_result().iterations, 1U);

        // FastSimulatedAnnealing has no stopping test: it spends the whole schedule and reports converged.
        const auto fsa = cxxgpr::Gpr<cxxgpr::optimizer::FastSimulatedAnnealing>(
            spec, cxxgpr::optimizer::FastSimulatedAnnealing().with_max_iterations(5U).with_seed(3ULL)
        )
                             .fit(x_ref, data.y);
        if (!fsa.has_value())
        {
            FAIL() << "FastSimulatedAnnealing fit failed: " << fsa.error().message();
        }
        EXPECT_TRUE(fsa->optimizer_result().converged);
        EXPECT_EQ(fsa->optimizer_result().iterations, 5U);

        const auto fixed = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).fit(x_ref, data.y);
        if (!fixed.has_value())
        {
            FAIL() << "Fixed fit failed: " << fixed.error().message();
        }
        EXPECT_TRUE(fixed->optimizer_result().converged);
        EXPECT_EQ(fixed->optimizer_result().iterations, 0U);
    }

    /// @brief Verifies that fit() forwards a user-defined optimizer's outcome unchanged.
    /// @details A custom optimizer returning GprError::optimization_not_converged() makes fit() fail with an
    /// error equal in code, message, and payload, and leaves the trainer usable; a custom optimizer returning a
    /// non-converged OptimizerResult makes fit() succeed with that result.
    TEST(GprModelsTest, CustomOptimizerOutcomePropagatesUnchangedFromFit)
    {
        // Spec: REQ-02, DES-02, DES-19
        const SineData data = make_sine_data();
        const std::size_t n = data.y.size();
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x.data(), n, 1U, n);

        const cxxgpr::Gpr<GiveUpOptimizer> give_up(spec);
        const auto expected_error = cxxgpr::GprError::optimization_not_converged(GiveUpOptimizer::k_iterations);
        const auto failed = give_up.fit(x_ref, data.y);
        ASSERT_FALSE(failed.has_value());
        EXPECT_EQ(failed.error(), expected_error);
        EXPECT_EQ(failed.error().code(), cxxgpr::GprErrorCode::OptimizationNotConverged);
        EXPECT_EQ(failed.error().iterations(), std::optional<std::size_t>(GiveUpOptimizer::k_iterations));
        EXPECT_EQ(failed.error().message(), "optimizer did not converge after 7 iterations");

        // The trainer is unchanged: a second fit fails identically and factor() still succeeds.
        const auto failed_again = give_up.fit(x_ref, data.y);
        ASSERT_FALSE(failed_again.has_value());
        EXPECT_EQ(failed_again.error(), expected_error);
        EXPECT_TRUE(give_up.factor(x_ref, data.y).has_value());

        const auto stopped = cxxgpr::Gpr<StopEarlyOptimizer>(spec).fit(x_ref, data.y);
        if (!stopped.has_value())
        {
            FAIL() << "fit with a non-converged custom result failed: " << stopped.error().message();
        }
        EXPECT_FALSE(stopped->optimizer_result().converged);
        EXPECT_EQ(stopped->optimizer_result().iterations, StopEarlyOptimizer::k_iterations);
    }

    /// @brief Verifies depth-128 alternating sum/product Gpr fit (L-BFGS and TrustRegion), predict, covariance, and LOO
    /// fit within the small thread stack budget.
    TEST(GprModelsTest, DeepAlternatingTreeFitPredictCovarianceLooFitInSmallStack)
    {
        // Spec: REQ-02, REQ-05, REQ-11
        const bool ran = cxxgpr::test_support::run_with_stack_size(
            cxxgpr::test_support::deep_model_stack_bytes(), []()
        {
            cxxgpr::KernelSpec tree(cxxgpr::RbfKernel::create(1.1).value());
            for (int depth = 1; depth < 128; ++depth)
            {
                const auto leaf = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.9).value());
                tree = (depth % 2 == 0) ? (std::move(tree) + leaf) : (std::move(tree) * leaf);
            }
            constexpr std::size_t n = 4U;
            const std::array<double, 4> x_raw{0.0, 0.5, 1.0, 1.5};
            const std::array<double, 4> y_raw{0.2, 0.8, 0.6, 0.1};
            const cxxgpr::MatrixRef<double> x(x_raw.data(), n, 1U, n);

            const auto fit_lbfgs =
                cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(tree, cxxgpr::optimizer::Lbfgs().with_max_iterations(2U))
                    .fit(x, y_raw);
            ASSERT_TRUE(fit_lbfgs.has_value()) << fit_lbfgs.error().message();
            EXPECT_TRUE(fit_lbfgs->predict(x).has_value());
            EXPECT_TRUE(fit_lbfgs->predict_covariance(x).has_value());
            EXPECT_TRUE(fit_lbfgs->loo_predict().has_value());

            const auto fit_tr = cxxgpr::Gpr<cxxgpr::optimizer::TrustRegion>(
                tree, cxxgpr::optimizer::TrustRegion().with_max_iterations(1U)
            )
                                    .fit(x, y_raw);
            ASSERT_TRUE(fit_tr.has_value()) << fit_tr.error().message();
        }
        );
        ASSERT_TRUE(ran);
    }

    /// @brief Verifies float coordinate and target overloads on Gpr and FittedGpr across precision policies.
    /// @details Confirms exact parity against double-widened versions of identical float data, zero heap
    /// allocations during warmed predict_into() / predict_with_into() with MatrixRef<float>, and DES-02 error
    /// precedence on checked float span overloads.
    TEST(GprModelsTest, FloatInputsMatchWidenedDoubleAndPreservePrecedenceAndZeroAlloc)
    {
        // Spec: DES-02, DES-16, REQ-02, REQ-09
        const std::vector<float> x_f{0.0F, 0.25F, 0.5F, 0.75F, 1.0F, 1.25F, 0.1F, 0.4F, 0.2F, 0.9F, 0.6F, 1.1F};
        const std::vector<float> y_f{1.1F, 0.95F, 0.6F, 0.2F, -0.15F, 0.35F};
        const std::vector<float> xq_f{0.15F, 0.65F, 1.15F, 0.3F, 0.7F, 0.95F};
        constexpr std::size_t n = 6U;
        constexpr std::size_t m = 3U;
        constexpr std::size_t d = 2U;

        const std::vector<double> x_d(x_f.begin(), x_f.end());
        const std::vector<double> y_d(y_f.begin(), y_f.end());
        const std::vector<double> xq_d(xq_f.begin(), xq_f.end());

        const auto spec = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                              * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.7).value()))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.02).value());

        auto check_policy = [&]<cxxgpr::PrecisionPolicyConcept P>()
        {
            using Refine = typename P::Refine;
            const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec).with_noise_variance(0.05).value();

            const cxxgpr::MatrixRef<float> x_ref_f(x_f.data(), n, d);
            const cxxgpr::MatrixRef<double> x_ref_d(x_d.data(), n, d);
            const cxxgpr::MatrixRef<float> xq_ref_f(xq_f.data(), m, d);
            const cxxgpr::MatrixRef<double> xq_ref_d(xq_d.data(), m, d);

            const auto fit_d = trainer.factor(x_ref_d, y_d);
            const auto fit_f_mat = trainer.factor(x_ref_f, std::span<const float>(y_f));
            const auto fit_f_span = trainer.factor(std::span<const float>(x_f), n, d, std::span<const float>(y_f));
            const auto fit_f_opt = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs, P>(spec).with_noise_variance(0.05).value().fit(
                x_ref_f, std::span<const float>(y_f)
            );
            const auto fit_mixed_xy = trainer.factor(x_ref_f, y_d);
            const auto fit_mixed_yx = trainer.factor(x_ref_d, std::span<const float>(y_f));
            ASSERT_TRUE(fit_d.has_value());
            ASSERT_TRUE(fit_f_mat.has_value());
            ASSERT_TRUE(fit_f_span.has_value());
            ASSERT_TRUE(fit_f_opt.has_value());
            ASSERT_TRUE(fit_mixed_xy.has_value());
            ASSERT_TRUE(fit_mixed_yx.has_value());

            const auto fit_opt_d =
                cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs, P>(spec).with_noise_variance(0.05).value().fit(x_ref_d, y_d);
            ASSERT_TRUE(fit_opt_d.has_value());
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(fit_opt_d->neg_log_marginal_likelihood()),
                std::bit_cast<std::uint64_t>(fit_f_opt->neg_log_marginal_likelihood())
            );
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(fit_d->neg_log_marginal_likelihood()),
                std::bit_cast<std::uint64_t>(fit_f_mat->neg_log_marginal_likelihood())
            );
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(fit_d->neg_log_marginal_likelihood()),
                std::bit_cast<std::uint64_t>(fit_f_span->neg_log_marginal_likelihood())
            );
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(fit_d->neg_log_marginal_likelihood()),
                std::bit_cast<std::uint64_t>(fit_mixed_xy->neg_log_marginal_likelihood())
            );
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(fit_d->neg_log_marginal_likelihood()),
                std::bit_cast<std::uint64_t>(fit_mixed_yx->neg_log_marginal_likelihood())
            );

            // predict, predict_with, predict_into, predict_with_into, predict_covariance, predict_covariance_with,
            // sample, sample_with, refit parity.
            const auto pred_d = fit_d->predict(xq_ref_d);
            const auto pred_f_mat = fit_f_mat->predict(xq_ref_f);
            const auto pred_f_span = fit_f_span->predict(std::span<const float>(xq_f), m, d);
            const cxxgpr::PredictOptions latent_opts{cxxgpr::VarianceKind::Latent};
            const auto pred_with_d = fit_d->predict_with(xq_ref_d, latent_opts);
            const auto pred_with_f = fit_f_mat->predict_with(xq_ref_f, latent_opts);
            ASSERT_TRUE(pred_d.has_value());
            ASSERT_TRUE(pred_f_mat.has_value());
            ASSERT_TRUE(pred_f_span.has_value());
            ASSERT_TRUE(pred_with_d.has_value());
            ASSERT_TRUE(pred_with_f.has_value());
            EXPECT_EQ(*pred_d, *pred_f_mat);
            EXPECT_EQ(*pred_d, *pred_f_span);
            EXPECT_EQ(*pred_with_d, *pred_with_f);

            cxxgpr::Prediction<Refine> warm_out;
            ASSERT_TRUE(fit_f_mat->predict_into(xq_ref_f, warm_out).has_value());
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                ASSERT_TRUE(fit_f_mat->predict_into(xq_ref_f, warm_out).has_value());
                ASSERT_TRUE(fit_f_mat->predict_with_into(xq_ref_f, latent_opts, warm_out).has_value());
            }
            EXPECT_EQ(warm_out, *pred_with_d);

            const auto cov_d = fit_d->predict_covariance(xq_ref_d);
            const auto cov_f = fit_f_mat->predict_covariance(xq_ref_f);
            const auto cov_with_d = fit_d->predict_covariance_with(xq_ref_d, latent_opts);
            const auto cov_with_f = fit_f_mat->predict_covariance_with(xq_ref_f, latent_opts);
            const auto cov_f_span = fit_f_mat->predict_covariance(std::span<const float>(xq_f), m, d);
            ASSERT_TRUE(cov_d.has_value());
            ASSERT_TRUE(cov_f.has_value());
            ASSERT_TRUE(cov_with_d.has_value());
            ASSERT_TRUE(cov_with_f.has_value());
            ASSERT_TRUE(cov_f_span.has_value());
            EXPECT_EQ(*cov_d, *cov_f);
            EXPECT_EQ(*cov_d, *cov_f_span);
            EXPECT_EQ(*cov_with_d, *cov_with_f);

            const auto samp_d = fit_d->sample(xq_ref_d, 4U, 99U);
            const auto samp_f = fit_f_mat->sample(xq_ref_f, 4U, 99U);
            const auto samp_f_span = fit_f_mat->sample(std::span<const float>(xq_f), m, d, 4U, 99U);
            const auto samp_with_d = fit_d->sample_with(xq_ref_d, 4U, 99U, latent_opts);
            const auto samp_with_f = fit_f_mat->sample_with(xq_ref_f, 4U, 99U, latent_opts);
            ASSERT_TRUE(samp_d.has_value());
            ASSERT_TRUE(samp_f.has_value());
            ASSERT_TRUE(samp_f_span.has_value());
            ASSERT_TRUE(samp_with_d.has_value());
            ASSERT_TRUE(samp_with_f.has_value());
            EXPECT_EQ(*samp_d, *samp_f);
            EXPECT_EQ(*samp_d, *samp_f_span);
            EXPECT_EQ(*samp_with_d, *samp_with_f);

            const auto refit_d = fit_opt_d->refit(x_ref_d, y_d);
            const auto refit_f_mat = fit_f_opt->refit(x_ref_f, std::span<const float>(y_f));
            const auto refit_f_span = fit_f_opt->refit(std::span<const float>(x_f), n, d, std::span<const float>(y_f));
            const auto refit_mixed_xy = fit_f_opt->refit(x_ref_f, y_d);
            const auto refit_mixed_yx = fit_f_opt->refit(x_ref_d, std::span<const float>(y_f));
            ASSERT_TRUE(refit_d.has_value());
            ASSERT_TRUE(refit_f_mat.has_value());
            ASSERT_TRUE(refit_f_span.has_value());
            ASSERT_TRUE(refit_mixed_xy.has_value());
            ASSERT_TRUE(refit_mixed_yx.has_value());
            EXPECT_EQ(refit_d->predict(xq_ref_d).value(), refit_f_mat->predict(xq_ref_f).value());
            EXPECT_EQ(refit_d->predict(xq_ref_d).value(), refit_f_span->predict(xq_ref_f).value());
            EXPECT_EQ(refit_d->predict(xq_ref_d).value(), refit_mixed_xy->predict(xq_ref_f).value());
            EXPECT_EQ(refit_d->predict(xq_ref_d).value(), refit_mixed_yx->predict(xq_ref_f).value());
        };

        check_policy.template operator()<cxxgpr::DoublePrecision>();
        check_policy.template operator()<cxxgpr::SinglePrecision>();
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>();

        // Negative validation precedence on float span overloads: EmptyInput -> SizeOverflow -> LengthMismatch ->
        // DimensionMismatch -> NonFiniteInput.
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(0.05).value();
        EXPECT_EQ(
            trainer.factor(std::span<const float>{}, 0U, d, std::span<const float>(y_f)).error().code(),
            cxxgpr::GprErrorCode::EmptyInput
        );
        EXPECT_EQ(
            trainer
                .factor(
                    std::span<const float>(x_f), std::numeric_limits<std::size_t>::max(), 2U,
                    std::span<const float>(y_f)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::SizeOverflow
        );
        EXPECT_EQ(
            trainer.factor(std::span<const float>(x_f.data(), x_f.size() - 1U), n, d, std::span<const float>(y_f))
                .error()
                .code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );

        std::vector<float> bad_x_f = x_f;
        bad_x_f[2] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_EQ(
            trainer.factor(std::span<const float>(bad_x_f), n, d, std::span<const float>(y_f)).error().code(),
            cxxgpr::GprErrorCode::NonFiniteInput
        );
        std::vector<float> bad_y_f = y_f;
        bad_y_f[1] = std::numeric_limits<float>::infinity();
        EXPECT_EQ(
            trainer.factor(std::span<const float>(x_f), n, d, std::span<const float>(bad_y_f)).error().code(),
            cxxgpr::GprErrorCode::NonFiniteInput
        );

        const auto fitted = trainer.factor(std::span<const float>(x_f), n, d, std::span<const float>(y_f)).value();
        EXPECT_EQ(fitted.predict(std::span<const float>{}, 0U, d).error().code(), cxxgpr::GprErrorCode::EmptyInput);
        EXPECT_EQ(
            fitted.predict(std::span<const float>(xq_f), std::numeric_limits<std::size_t>::max(), 2U).error().code(),
            cxxgpr::GprErrorCode::SizeOverflow
        );
        EXPECT_EQ(
            fitted.predict(std::span<const float>(xq_f.data(), xq_f.size() - 1U), m, d).error().code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );
        const std::vector<float> wrong_dim_q(m * (d + 1U), 0.1F);
        EXPECT_EQ(
            fitted.predict(std::span<const float>(wrong_dim_q), m, d + 1U).error().code(),
            cxxgpr::GprErrorCode::DimensionMismatch
        );
        std::vector<float> nan_q = xq_f;
        nan_q[0] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_EQ(
            fitted.predict(std::span<const float>(nan_q), m, d).error().code(), cxxgpr::GprErrorCode::NonFiniteInput
        );
    }

    /// @brief Verifies converged MixedPrecision refinement allocates no n x n double matrix and frees dead buffers.
    TEST(GprModelsTest, MixedPrecisionConvergedRefinementAllocatesNoFullDoubleMatrix)
    {
        // Spec: DES-05, REQ-07, REQ-11
        constexpr std::size_t n = 160;
        constexpr std::size_t d = 3;
        constexpr std::size_t full_double_matrix_bytes = n * n * sizeof(double);

        std::vector<double> x(n * d);
        std::vector<double> y(n);
        cxxgpr::Xoshiro256PlusPlus rng(41);
        for (double& v : x)
        {
            v = rng.unit();
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            y[i] = std::sin(2.5 * x[i]) + (0.4 * x[n + i]) + (0.05 * rng.unit_normal());
        }

        const auto spec = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.1).value())
                              * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.7).value()))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.01).value());
        const auto single_leaf_spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.7).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.01).value());

        const auto ref_factor =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-2).value().factor(x, n, d, y);
        ASSERT_TRUE(ref_factor.has_value());

        const auto opt = cxxgpr::optimizer::Lbfgs{}.with_max_iterations(5U);
        std::size_t ref_lbfgs_max_alloc = 0U;
        std::size_t ref_lbfgs_total_bytes = 0U;
        {
            const cxxgpr::test_support::AllocationBytesScope ref_lbfgs_scope;
            const auto ref_fit =
                cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec, opt).with_noise_variance(1e-2).value().fit(x, n, d, y);
            ASSERT_TRUE(ref_fit.has_value());
            ref_lbfgs_max_alloc = ref_lbfgs_scope.max_single_allocation_bytes();
            ref_lbfgs_total_bytes = ref_lbfgs_scope.delta_bytes();
        }

        auto verify_policy = [&]<typename P>(bool use_lbfgs, bool check_alpha_parity)
        {
            if (use_lbfgs)
            {
                {
                    const cxxgpr::test_support::AllocationBytesScope scope;
                    auto res =
                        cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs, P>(spec, opt).with_noise_variance(1e-2).value().fit(
                            x, n, d, y
                        );
                    ASSERT_TRUE(res.has_value()) << res.error().message();
                    const auto summary = res->refinement_summary();
                    if (!summary.has_value())
                    {
                        FAIL() << "mixed-precision refinement summary is missing";
                    }
                    EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
                    EXPECT_FALSE(summary->used_fallback());
                    EXPECT_LT(scope.max_single_allocation_bytes(), ref_lbfgs_max_alloc);
                    EXPECT_LT(scope.delta_bytes(), ref_lbfgs_total_bytes);
                }

                // Without multi-factor ProductTerms (weighted_buffers == 0), no single allocation reaches 8 * n * n.
                const cxxgpr::test_support::AllocationBytesScope single_scope;
                auto single_res = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs, P>(single_leaf_spec, opt)
                                      .with_noise_variance(1e-2)
                                      .value()
                                      .fit(x, n, d, y);
                ASSERT_TRUE(single_res.has_value()) << single_res.error().message();
                const auto single_summary = single_res->refinement_summary();
                if (!single_summary.has_value())
                {
                    FAIL() << "mixed-precision refinement summary is missing";
                }
                EXPECT_EQ(single_summary->status, cxxgpr::RefinementStatus::Converged);
                EXPECT_FALSE(single_summary->used_fallback());
                EXPECT_LT(single_scope.max_single_allocation_bytes(), full_double_matrix_bytes);
                return;
            }

            const cxxgpr::test_support::AllocationBytesScope scope;
            auto res =
                cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec).with_noise_variance(1e-2).value().factor(x, n, d, y);
            ASSERT_TRUE(res.has_value()) << res.error().message();
            const auto summary = res->refinement_summary();
            if (!summary.has_value())
            {
                FAIL() << "mixed-precision refinement summary is missing";
            }
            EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
            EXPECT_FALSE(summary->used_fallback());
            EXPECT_LT(scope.max_single_allocation_bytes(), full_double_matrix_bytes);

            if (check_alpha_parity)
            {
                double max_ref = 1.0;
                double max_diff = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    max_ref = std::max(max_ref, std::abs(ref_factor->alpha()[i]));
                    max_diff = std::max(max_diff, std::abs(ref_factor->alpha()[i] - res->alpha()[i]));
                }
                EXPECT_LT(max_diff / max_ref, 1e-5);
            }
        };

        verify_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(false, false);
        verify_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(true, false);
        verify_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(false, true);
        verify_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(true, false);
    }

    /// @brief Indefinite custom kernel returning 1 on zero distance and 2 on positive distance for jitter negative
    /// controls.
    /// @tparam T Evaluation scalar type (double or float).
    template <typename T> class IndefiniteControlKernelTerm final : public cxxgpr::KernelTerm<T>
    {
    public:
        /// @brief Clones this custom kernel leaf.
        /// @return Independent copy of this indefinite kernel leaf.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<T>> clone() const override
        {
            return std::make_unique<IndefiniteControlKernelTerm<T>>();
        }

        /// @brief Returns zero kernel parameters.
        /// @return Always returns 0.
        [[nodiscard]] std::size_t num_params() const noexcept override { return 0U; }

        /// @brief Writes zero parameters when @p out is empty.
        /// @param[out] out Destination span (must be empty).
        /// @return Empty expected on success, or LengthMismatch.
        std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            if (!out.empty())
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("expected 0 params"));
            }
            return {};
        }

        /// @brief Accepts an empty parameter span.
        /// @param[in] params Input span (must be empty).
        /// @return Empty expected on success, or LengthMismatch.
        std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            if (!params.empty())
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("expected 0 params"));
            }
            return {};
        }

        /// @brief Writes zero bounds when @p out is empty.
        /// @param[out] out Destination span (must be empty).
        /// @return Empty expected on success, or LengthMismatch.
        std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            if (!out.empty())
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("expected 0 bounds"));
            }
            return {};
        }

        /// @brief Writes 1 on diagonal (dist == 0) and 2 off-diagonal (dist > 0) so the matrix is indefinite.
        /// @param[in] dist Square matrix of squared distances.
        /// @param[out] out Output kernel matrix populated on @p uplo.
        /// @param[in] uplo Triangle or full region to fill.
        /// @return Empty expected on success.
        std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo
        ) const override
        {
            const std::size_t n = dist.rows();
            for (std::size_t col = 0U; col < n; ++col)
            {
                const std::size_t r_begin = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
                const std::size_t r_end = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = r_begin; row < r_end; ++row)
                {
                    out(row, col) = (row == col) ? T{1} : (dist(row, col) > T{0} ? T{2} : T{1});
                }
            }
            return {};
        }

        /// @brief Evaluates rectangular cross entries matching apply().
        /// @param[in] dist Rectangular squared-distance matrix.
        /// @param[out] out Destination cross-kernel matrix.
        /// @return Empty expected on success.
        std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out
        ) const override
        {
            for (std::size_t col = 0U; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0U; row < dist.rows(); ++row)
                {
                    out(row, col) = (dist(row, col) > T{0}) ? T{2} : T{1};
                }
            }
            return {};
        }

        /// @brief Fills zero-distance self-variances with 1.
        /// @param[out] out Destination diagonal span.
        /// @return Empty expected on success.
        std::expected<void, cxxgpr::GprError> fill_diag(std::span<T> out) const override
        {
            std::ranges::fill(out, T{1});
            return {};
        }

        /// @brief Rejects gradient calls (no parameters).
        /// @param[in] dist Unused.
        /// @param[out] d_k Unused.
        /// @param[in] param_idx Parameter index.
        /// @param[in] uplo Unused.
        /// @return IndexOutOfRange error.
        std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d_k, std::size_t param_idx, cxxgpr::Triangle uplo
        ) const override
        {
            (void)dist;
            (void)d_k;
            (void)param_idx;
            (void)uplo;
            return std::unexpected(cxxgpr::GprError::index_out_of_range("no parameters"));
        }

        /// @brief Rejects Hessian calls (no parameters).
        /// @param[in] dist Unused.
        /// @param[out] d2_k Unused.
        /// @param[in] i First parameter index.
        /// @param[in] j Second parameter index.
        /// @param[in] uplo Unused.
        /// @return IndexOutOfRange error.
        std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d2_k, std::size_t i, std::size_t j, cxxgpr::Triangle uplo
        ) const override
        {
            (void)dist;
            (void)d2_k;
            (void)i;
            (void)j;
            (void)uplo;
            return std::unexpected(cxxgpr::GprError::index_out_of_range("no parameters"));
        }
    };

    /// @brief Scaled RBF custom kernel leaf for platform-independent IR-to-Krylov escalation tests.
    /// @details Evaluates scale * exp(-d^2 / 0.125) so float (scale = 0.5) and double (scale = 2.0) yield
    /// positive-definite matrices whose preconditioned operator forces stationary IR stagnation into FGMRES-IR.
    /// @tparam T Evaluation scalar type (double or float).
    template <typename T> class ScaledRbfControlKernelTerm final : public cxxgpr::KernelTerm<T>
    {
    public:
        /// @brief Constructs the scaled RBF leaf with a fixed output amplitude.
        /// @param[in] scale Output variance multiplier applied to exp(-d^2 / 0.125).
        explicit ScaledRbfControlKernelTerm(double scale) noexcept : scale_(static_cast<T>(scale)) {}

        /// @brief Clones this custom kernel leaf.
        /// @return Independent copy of this scaled RBF kernel leaf.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<T>> clone() const override
        {
            return std::make_unique<ScaledRbfControlKernelTerm<T>>(static_cast<double>(scale_));
        }

        /// @brief Returns zero kernel parameters.
        /// @return Always returns 0.
        [[nodiscard]] std::size_t num_params() const noexcept override { return 0U; }

        /// @brief Writes zero parameters when @p out is empty.
        /// @param[out] out Destination span (must be empty).
        /// @return Empty expected on success, or LengthMismatch.
        std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            if (!out.empty())
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("expected 0 params"));
            }
            return {};
        }

        /// @brief Accepts an empty parameter span.
        /// @param[in] params Input span (must be empty).
        /// @return Empty expected on success, or LengthMismatch.
        std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            if (!params.empty())
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("expected 0 params"));
            }
            return {};
        }

        /// @brief Writes zero bounds when @p out is empty.
        /// @param[out] out Destination span (must be empty).
        /// @return Empty expected on success, or LengthMismatch.
        std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            if (!out.empty())
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("expected 0 bounds"));
            }
            return {};
        }

        /// @brief Evaluates scale * exp(-dist / 0.125) on the requested triangle or full matrix.
        /// @param[in] dist Square matrix of squared distances.
        /// @param[out] out Destination matrix populated on @p uplo.
        /// @param[in] uplo Triangle or full region to populate.
        /// @return Empty expected on success.
        std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo
        ) const override
        {
            const std::size_t n = dist.rows();
            constexpr T inv_denom = static_cast<T>(8.0);
            for (std::size_t col = 0U; col < n; ++col)
            {
                const std::size_t r_begin = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
                const std::size_t r_end = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = r_begin; row < r_end; ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col) * inv_denom);
                }
            }
            return {};
        }

        /// @brief Evaluates scale * exp(-dist / 0.125) on a rectangular cross distance matrix.
        /// @param[in] dist Rectangular squared-distance matrix (n x m).
        /// @param[out] out Destination cross-kernel matrix (n x m).
        /// @return Empty expected on success.
        std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out
        ) const override
        {
            constexpr T inv_denom = static_cast<T>(8.0);
            for (std::size_t col = 0U; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0U; row < dist.rows(); ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col) * inv_denom);
                }
            }
            return {};
        }

        /// @brief Writes constant diagonal variance scale into @p out.
        /// @param[out] out Destination diagonal span.
        /// @return Empty expected on success.
        std::expected<void, cxxgpr::GprError> fill_diag(std::span<T> out) const override
        {
            std::ranges::fill(out, scale_);
            return {};
        }

        /// @brief Rejects gradient calls (no parameters).
        /// @param[in] dist Unused.
        /// @param[out] d_k Unused.
        /// @param[in] param_idx Parameter index.
        /// @param[in] uplo Unused.
        /// @return IndexOutOfRange error.
        std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d_k, std::size_t param_idx, cxxgpr::Triangle uplo
        ) const override
        {
            (void)dist;
            (void)d_k;
            (void)param_idx;
            (void)uplo;
            return std::unexpected(cxxgpr::GprError::index_out_of_range("no parameters"));
        }

        /// @brief Rejects Hessian calls (no parameters).
        /// @param[in] dist Unused.
        /// @param[out] d2_k Unused.
        /// @param[in] i First parameter index.
        /// @param[in] j Second parameter index.
        /// @param[in] uplo Unused.
        /// @return IndexOutOfRange error.
        std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d2_k, std::size_t i, std::size_t j, cxxgpr::Triangle uplo
        ) const override
        {
            (void)dist;
            (void)d2_k;
            (void)i;
            (void)j;
            (void)uplo;
            return std::unexpected(cxxgpr::GprError::index_out_of_range("no parameters"));
        }

    private:
        /// @brief Scalar amplitude applied to the RBF evaluation.
        T scale_{T{1}};
    };

    /// @brief Verifies shifted float preconditioning, Krylov refinement, jitter priority, and double mean accuracy.
    /// @details Confirms SinglePrecision fails at noise = 1e-10 while MixedPrecision<ReevaluateKernel> converges
    /// without equation jitter, stays under 8 * n * n bytes per allocation, matches DoublePrecision alpha and mean
    /// across multi-block queries with zero post-warmup allocations, escalates to Krylov when unshifted IR stalls,
    /// and delegates genuinely indefinite matrices to JitterPolicy.
    TEST(GprModelsTest, MixedPrecisionReevaluateSolvesSmallNoiseWithShiftedPreconditionerAndDoubleMean)
    {
        // Spec: REQ-07, DES-04, DES-05, DES-16
        const auto make_small_noise_likelihood = [](double noise)
        {
            auto lik = cxxgpr::GaussianLikelihood::create(1e-4)
                           .value()
                           .with_bounds(cxxgpr::Interval::create(1e-20, 1e5).value())
                           .value();
            EXPECT_TRUE(lik.set_params(std::array<double, 1>{std::log(noise)}).has_value());
            return lik;
        };

        constexpr std::size_t n = 128U;
        constexpr std::size_t d = 1U;
        constexpr std::size_t m = 40U;
        constexpr std::size_t m_small = 10U;
        const std::size_t full_double_matrix_bytes = n * n * sizeof(double);

        std::vector<double> x(n);
        std::vector<double> y(n);
        for (std::size_t i = 0U; i < n; ++i)
        {
            const double xi = static_cast<double>(i) / static_cast<double>(n - 1U);
            x[i] = xi;
            y[i] = std::sin(6.283185307179586 * xi) + (0.3 * std::cos(12.566370614359172 * xi));
        }
        std::vector<double> xq(m);
        for (std::size_t j = 0U; j < m; ++j)
        {
            xq[j] = (static_cast<double>(j) + 0.35) / static_cast<double>(m);
        }
        const cxxgpr::MatrixRef<double> x_ref(x.data(), n, d, n);
        const cxxgpr::MatrixRef<double> xq_ref(xq.data(), m, d, m);
        const cxxgpr::MatrixRef<double> xq_small_ref(xq.data(), m_small, d, m_small);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto small_lik = make_small_noise_likelihood(1e-10);
        const auto zero_jitter = cxxgpr::JitterPolicy::fixed(0.0).value();

        // Confirm SinglePrecision fails unshifted Cholesky on this small-noise system.
        const auto single_res = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::SinglePrecision>(spec)
                                    .with_likelihood(small_lik)
                                    .with_jitter(zero_jitter)
                                    .factor(x_ref, y);
        ASSERT_FALSE(single_res.has_value());
        EXPECT_EQ(single_res.error().code(), cxxgpr::GprErrorCode::CholeskyFailed);

        const auto ref_f64 = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision>(spec)
                                 .with_likelihood(small_lik)
                                 .with_jitter(zero_jitter)
                                 .factor(x_ref, y);
        ASSERT_TRUE(ref_f64.has_value()) << ref_f64.error().message();
        const auto ref_pred = ref_f64->predict(xq_ref);
        const auto ref_pred_small = ref_f64->predict(xq_small_ref);
        ASSERT_TRUE(ref_pred.has_value());
        ASSERT_TRUE(ref_pred_small.has_value());

        {
            const cxxgpr::test_support::AllocationBytesScope alloc_scope;
            const auto mp_reev =
                cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec)
                    .with_likelihood(small_lik)
                    .with_jitter(zero_jitter)
                    .factor(x_ref, y);
            ASSERT_TRUE(mp_reev.has_value()) << mp_reev.error().message();
            EXPECT_LT(alloc_scope.max_single_allocation_bytes(), full_double_matrix_bytes);
            EXPECT_EQ(mp_reev->factor_jitter(), 0.0);

            const auto summary = mp_reev->refinement_summary();
            ASSERT_TRUE(summary.has_value());
            EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
            EXPECT_FALSE(summary->used_fallback());
            EXPECT_GT(summary->preconditioner_shift, 0.0);
            EXPECT_TRUE(summary->used_krylov);
            EXPECT_LT(summary->final_relative_residual, cxxgpr::refinement_tolerance(n));

            // Independent exact double backward-error verification against B * alpha = y.
            constexpr double ell = 0.25;
            constexpr double inv_2l2 = 1.0 / (2.0 * ell * ell);
            double b_inf = 0.0;
            double r_inf = 0.0;
            double alpha_inf = 0.0;
            double y_inf = 0.0;
            for (std::size_t i = 0U; i < n; ++i)
            {
                alpha_inf = std::max(alpha_inf, std::abs(mp_reev->alpha()[i]));
                y_inf = std::max(y_inf, std::abs(y[i]));
                double row_sum = 0.0;
                double b_alpha_i = 0.0;
                for (std::size_t j = 0U; j < n; ++j)
                {
                    const double diff = x[i] - x[j];
                    const double entry = std::exp(-diff * diff * inv_2l2) + (i == j ? 1e-10 : 0.0);
                    row_sum += std::abs(entry);
                    b_alpha_i += entry * mp_reev->alpha()[j];
                }
                b_inf = std::max(b_inf, row_sum);
                r_inf = std::max(r_inf, std::abs(y[i] - b_alpha_i));
            }
            const double rel_res = r_inf / ((b_inf * alpha_inf) + y_inf);
            EXPECT_LT(rel_res, cxxgpr::refinement_tolerance(n));

            cxxgpr::Prediction<double> out_pred;
            ASSERT_TRUE(mp_reev->predict_into(xq_ref, out_pred).has_value());
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                ASSERT_TRUE(mp_reev->predict_into(xq_ref, out_pred).has_value());
            }
            double max_mean_diff = 0.0;
            for (std::size_t j = 0U; j < m; ++j)
            {
                max_mean_diff = std::max(max_mean_diff, std::abs(out_pred.mean[j] - ref_pred->mean[j]));
            }
            EXPECT_LT(max_mean_diff, 1e-10);

            // Shrunk query batch reuses warmed double column-block scratch with zero allocations.
            ASSERT_TRUE(mp_reev->predict_into(xq_small_ref, out_pred).has_value());
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                ASSERT_TRUE(mp_reev->predict_into(xq_small_ref, out_pred).has_value());
            }
            double max_small_diff = 0.0;
            for (std::size_t j = 0U; j < m_small; ++j)
            {
                max_small_diff = std::max(max_small_diff, std::abs(out_pred.mean[j] - ref_pred_small->mean[j]));
            }
            EXPECT_LT(max_small_diff, 1e-10);

            const auto cov = mp_reev->predict_covariance(xq_small_ref);
            ASSERT_TRUE(cov.has_value());
            double max_cov_mean_diff = 0.0;
            for (std::size_t j = 0U; j < m_small; ++j)
            {
                max_cov_mean_diff = std::max(max_cov_mean_diff, std::abs(cov->mean[j] - ref_pred_small->mean[j]));
            }
            EXPECT_LT(max_cov_mean_diff, 1e-10);

            const auto draws = mp_reev->sample(xq_small_ref, 3U, 77U);
            ASSERT_TRUE(draws.has_value());
            EXPECT_EQ(draws->size(), 3U * m_small);

            // Copy construction must preserve compiled_kernel_f64 and x_train_f64 for double mean prediction.
            const auto orig_pred = mp_reev->predict(xq_ref);
            const auto copied = *mp_reev;
            const auto copied_pred = copied.predict(xq_ref);
            ASSERT_TRUE(orig_pred.has_value());
            ASSERT_TRUE(copied_pred.has_value());
            EXPECT_EQ(*copied_pred, *orig_pred);
            double max_copied_diff = 0.0;
            for (std::size_t j = 0U; j < m; ++j)
            {
                max_copied_diff = std::max(max_copied_diff, std::abs(copied_pred->mean[j] - ref_pred->mean[j]));
            }
            EXPECT_LT(max_copied_diff, 1e-10);
        }

        // Preconditioner shift + FGMRES-IR must take precedence over Adaptive equation jitter.
        {
            const auto adaptive_jit = cxxgpr::JitterPolicy::adaptive(1e-6, 10.0, 5U, 1.0).value();
            const auto mp_adaptive =
                cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec)
                    .with_likelihood(small_lik)
                    .with_jitter(adaptive_jit)
                    .factor(x_ref, y);
            ASSERT_TRUE(mp_adaptive.has_value()) << mp_adaptive.error().message();
            EXPECT_EQ(mp_adaptive->factor_jitter(), 0.0);
            const auto summary = mp_adaptive->refinement_summary();
            ASSERT_TRUE(summary.has_value());
            EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
            EXPECT_FALSE(summary->used_fallback());
            EXPECT_GT(summary->preconditioner_shift, 0.0);
            EXPECT_TRUE(summary->used_krylov);

            const auto pred = mp_adaptive->predict(xq_ref);
            ASSERT_TRUE(pred.has_value());
            double max_mean_diff = 0.0;
            for (std::size_t j = 0U; j < m; ++j)
            {
                max_mean_diff = std::max(max_mean_diff, std::abs(pred->mean[j] - ref_pred->mean[j]));
            }
            EXPECT_LT(max_mean_diff, 1e-10);
        }

        // Well-conditioned unshifted float Cholesky (shift == 0) where stationary IR stagnates and escalates to
        // FGMRES-IR deterministically across platforms.
        {
            constexpr std::size_t n_esc = 16U;
            std::vector<double> x_esc(n_esc);
            std::vector<double> y_esc(n_esc);
            for (std::size_t i = 0U; i < n_esc; ++i)
            {
                const double xi = static_cast<double>(i) / static_cast<double>(n_esc - 1U);
                x_esc[i] = xi;
                y_esc[i] = std::sin(6.283185307179586 * xi) + (0.3 * std::cos(12.566370614359172 * xi));
            }
            const cxxgpr::MatrixRef<double> x_esc_ref(x_esc.data(), n_esc, d, n_esc);
            auto esc_custom = cxxgpr::CustomKernel::create(
                std::make_unique<ScaledRbfControlKernelTerm<double>>(2.0),
                std::make_unique<ScaledRbfControlKernelTerm<float>>(0.5)
            );
            ASSERT_TRUE(esc_custom.has_value());
            const cxxgpr::KernelSpec esc_spec(*esc_custom);
            const auto esc_lik = make_small_noise_likelihood(0.1);

            const auto esc_ref = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision>(esc_spec)
                                     .with_likelihood(esc_lik)
                                     .with_jitter(zero_jitter)
                                     .factor(x_esc_ref, y_esc);
            const auto esc_fit =
                cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(esc_spec)
                    .with_likelihood(esc_lik)
                    .with_jitter(zero_jitter)
                    .factor(x_esc_ref, y_esc);
            ASSERT_TRUE(esc_ref.has_value()) << esc_ref.error().message();
            ASSERT_TRUE(esc_fit.has_value()) << esc_fit.error().message();
            EXPECT_EQ(esc_fit->factor_jitter(), 0.0);
            const auto esc_summary = esc_fit->refinement_summary();
            ASSERT_TRUE(esc_summary.has_value());
            EXPECT_EQ(esc_summary->status, cxxgpr::RefinementStatus::Converged);
            EXPECT_FALSE(esc_summary->used_fallback());
            EXPECT_EQ(esc_summary->preconditioner_shift, 0.0);
            EXPECT_TRUE(esc_summary->used_krylov);

            const auto esc_ref_pred = esc_ref->predict(xq_small_ref);
            const auto esc_mp_pred = esc_fit->predict(xq_small_ref);
            ASSERT_TRUE(esc_ref_pred.has_value());
            ASSERT_TRUE(esc_mp_pred.has_value());
            double max_esc_diff = 0.0;
            for (std::size_t j = 0U; j < m_small; ++j)
            {
                max_esc_diff = std::max(max_esc_diff, std::abs(esc_mp_pred->mean[j] - esc_ref_pred->mean[j]));
            }
            EXPECT_LT(max_esc_diff, 1e-10);
        }

        // Negative control: genuinely indefinite system fails Fixed(0) and recovers only via Adaptive JitterPolicy.
        {
            auto custom = cxxgpr::CustomKernel::create(
                std::make_unique<IndefiniteControlKernelTerm<double>>(),
                std::make_unique<IndefiniteControlKernelTerm<float>>()
            );
            ASSERT_TRUE(custom.has_value());
            const cxxgpr::KernelSpec indef_spec(*custom);
            const std::array<double, 3> x_indef{0.0, 1.0, 2.0};
            const std::array<double, 3> y_indef{1.0, -0.5, 0.25};
            const auto indef_lik = make_small_noise_likelihood(1e-4);

            const auto trainer =
                cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(indef_spec)
                    .with_likelihood(indef_lik);
            const auto fail_fixed = trainer.with_jitter(zero_jitter).factor(x_indef, 3U, 1U, y_indef);
            ASSERT_FALSE(fail_fixed.has_value());
            EXPECT_EQ(fail_fixed.error().code(), cxxgpr::GprErrorCode::CholeskyFailed);

            const auto indef_jit = cxxgpr::JitterPolicy::adaptive(4.0, 2.0, 4U, 32.0).value();
            const auto rec = trainer.with_jitter(indef_jit).factor(x_indef, 3U, 1U, y_indef);
            ASSERT_TRUE(rec.has_value()) << rec.error().message();
            EXPECT_GT(rec->factor_jitter(), 0.0);
        }
    }

}  // namespace
