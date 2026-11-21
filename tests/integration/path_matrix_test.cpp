// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-04, REQ-05, REQ-07, REQ-08, REQ-09, REQ-10, DES-05, DES-10, DES-12, DES-15,
// DES-16, DES-18, DES-19
/// @file path_matrix_test.cpp
/// @brief Integration path-matrix tests covering kernels, optimizers, precisions, transforms, and persist.
/// @details Exercises the 10-kernel x 5-optimizer x 4-precision product matrix, composite kernel
/// persistence across all preprocessing transform pipelines, KernelExp x FitPolicy buffer matrices,
/// and DistanceKernel parity against coordinate evaluation under Accurate and FastApprox.

#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/persist/persist.hpp"
#include "cxxgpr/transform/transform.hpp"
#include "cxxgpr/workspace/mll.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace
{

    /// @brief Named kernel specification entry used in integration matrix sweeps.
    struct NamedKernel
    {
        /// @brief Human-readable kernel name attached to SCOPED_TRACE diagnostics.
        std::string name;
        /// @brief Kernel AST specification under test.
        cxxgpr::KernelSpec spec;
    };

    /// @brief Instantiates all ten built-in leaf kernel specifications for matrix testing.
    /// @details Includes RBF, RBF-ARD, Matern-5/2, Matern-ARD-3/2, Periodic, RationalQuadratic,
    /// RationalQuadratic-ARD, Constant, Linear, and White leaves with positive defaults.
    /// @return Vector containing one NamedKernel per built-in kernel family.
    [[nodiscard]] std::vector<NamedKernel> make_all_ten_built_in_kernels()
    {
        const std::array<double, 2> ard_ls{1.0, 1.25};
        std::vector<NamedKernel> out;
        // Register each of the 10 built-in kernel leaves with valid initial hyperparameters.
        out.push_back({.name = "Rbf", .spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value())});
        out.push_back({.name = "RbfArd", .spec = cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(ard_ls).value())});
        out.push_back(
            {.name = "Matern52",
                .spec = cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.0, cxxgpr::MaternNu::FiveHalves).value())}
        );
        out.push_back(
            {.name = "MaternArd32",
                .spec =
                    cxxgpr::KernelSpec(cxxgpr::MaternArdKernel::create(ard_ls, cxxgpr::MaternNu::ThreeHalves).value())}
        );
        out.push_back(
            {.name = "Periodic", .spec = cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(1.5, 4.0).value())}
        );
        out.push_back(
            {.name = "RationalQuadratic",
                .spec = cxxgpr::KernelSpec(cxxgpr::RationalQuadraticKernel::create(1.0, 1.5).value())}
        );
        out.push_back(
            {.name = "RationalQuadraticArd",
                .spec = cxxgpr::KernelSpec(cxxgpr::RationalQuadraticArdKernel::create(ard_ls, 1.5).value())}
        );
        out.push_back({.name = "Constant", .spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.5).value())});
        out.push_back({.name = "Linear", .spec = cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.8).value())});
        out.push_back({.name = "White", .spec = cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.25).value())});
        return out;
    }

    /// @brief Runs end-to-end fit, predict, covariance, sample, and LOO checks for every kernel.
    /// @details Executes fit(), predict_into(), predict_covariance(), sample(), and loo_predict()
    /// for a given optimizer instance and precision policy across all kernels in the supplied list.
    /// @tparam O Optimizer policy type (Fixed, Lbfgs, TrustRegion, NelderMead, or FSA).
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] opt Configured optimizer instance.
    /// @param[in] kernels Collection of named built-in kernel specs to test.
    /// @param[in] x_ref Column-major n x d training input view.
    /// @param[in] y_train Training target values of length n.
    /// @param[in] xq_ref Column-major m x d query input view.
    template <typename O, cxxgpr::PrecisionPolicyConcept P>
    void verify_kernel_matrix_for(
        const O& opt,
        const std::vector<NamedKernel>& kernels,
        cxxgpr::MatrixRef<double> x_ref,
        std::span<const double> y_train,
        cxxgpr::MatrixRef<double> xq_ref
    )
    {
        cxxgpr::Prediction<typename P::Refine> pred_out;
        for (const auto& nk : kernels)
        {
            SCOPED_TRACE(nk.name);
            // Fit with adaptive jitter so single-precision and periodic/linear Gram matrices stay SPD.
            const auto trainer = cxxgpr::Gpr<O, P>(nk.spec, opt)
                                     .with_noise_variance(5e-2)
                                     .value()
                                     .with_jitter(cxxgpr::JitterPolicy::adaptive(1e-6, 10.0, 6U, 1.0).value());
            auto fitted = trainer.fit(x_ref, y_train);
            ASSERT_TRUE(fitted.has_value()) << fitted.error().message();
            EXPECT_TRUE(std::isfinite(fitted->neg_log_marginal_likelihood()));

            ASSERT_TRUE(fitted->predict_into(xq_ref, pred_out).has_value());
            ASSERT_EQ(pred_out.mean.size(), xq_ref.rows());
            ASSERT_EQ(pred_out.variance.size(), xq_ref.rows());
            for (std::size_t j = 0; j < xq_ref.rows(); ++j)
            {
                EXPECT_TRUE(std::isfinite(pred_out.mean[j]));
                EXPECT_TRUE(std::isfinite(pred_out.variance[j]));
                EXPECT_GE(static_cast<double>(pred_out.variance[j]), 0.0);
            }

            auto cov = fitted->predict_covariance(xq_ref);
            ASSERT_TRUE(cov.has_value());
            for (std::size_t j = 0; j < xq_ref.rows(); ++j)
            {
                EXPECT_NEAR(
                    static_cast<double>(cov->covariance[(j * xq_ref.rows()) + j]),
                    static_cast<double>(pred_out.variance[j]), 1e-5
                );
            }

            auto draws = fitted->sample(xq_ref, 2U, 42ULL);
            ASSERT_TRUE(draws.has_value());
            EXPECT_EQ(draws->size(), 2U * xq_ref.rows());
            for (const auto v : *draws)
            {
                EXPECT_TRUE(std::isfinite(v));
            }

            auto loo = fitted->loo_predict();
            ASSERT_TRUE(loo.has_value());
            ASSERT_EQ(loo->mean.size(), x_ref.rows());
            for (std::size_t i = 0; i < x_ref.rows(); ++i)
            {
                EXPECT_TRUE(std::isfinite(loo->mean[i]));
                EXPECT_TRUE(std::isfinite(loo->variance[i]));
                EXPECT_GE(static_cast<double>(loo->variance[i]), 0.0);
            }
        }
    }

    /// @brief Sweeps all five optimizer policies for a single precision policy P.
    /// @details Invokes verify_kernel_matrix_for() with Fixed, Lbfgs, TrustRegion, NelderMead,
    /// and FastSimulatedAnnealing over the supplied kernel suite.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] kernels Collection of named kernel specs to test.
    /// @param[in] x_ref Column-major n x d training input view.
    /// @param[in] y_train Training target values of length n.
    /// @param[in] xq_ref Column-major m x d query input view.
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_all_optimizers_for_precision(
        const std::vector<NamedKernel>& kernels,
        cxxgpr::MatrixRef<double> x_ref,
        std::span<const double> y_train,
        cxxgpr::MatrixRef<double> xq_ref
    )
    {
        verify_kernel_matrix_for<cxxgpr::optimizer::Fixed, P>(
            cxxgpr::optimizer::Fixed(), kernels, x_ref, y_train, xq_ref
        );
        verify_kernel_matrix_for<cxxgpr::optimizer::Lbfgs, P>(
            cxxgpr::optimizer::Lbfgs().with_max_iterations(25U), kernels, x_ref, y_train, xq_ref
        );
        verify_kernel_matrix_for<cxxgpr::optimizer::TrustRegion, P>(
            cxxgpr::optimizer::TrustRegion().with_max_iterations(20U), kernels, x_ref, y_train, xq_ref
        );
        verify_kernel_matrix_for<cxxgpr::optimizer::NelderMead, P>(
            cxxgpr::optimizer::NelderMead().with_max_iterations(40U), kernels, x_ref, y_train, xq_ref
        );
        verify_kernel_matrix_for<cxxgpr::optimizer::FastSimulatedAnnealing, P>(
            cxxgpr::optimizer::FastSimulatedAnnealing().with_max_iterations(30U).with_seed(7ULL), kernels, x_ref,
            y_train, xq_ref
        );
    }

    /// @brief Verifies the full 10-kernel x 5-optimizer x 4-precision integration matrix.
    /// @details Exercises DoublePrecision, SinglePrecision, MixedPrecision<PromoteStorage>, and
    /// MixedPrecision<ReevaluateKernel> across every optimizer and built-in kernel family.
    TEST(PathMatrixTest, TenKernelsByFiveOptimizersByPrecisionPolicies)
    {
        // Spec: REQ-01, REQ-04, REQ-07, REQ-08, REQ-09, DES-05, DES-16, DES-19
        const std::size_t n = 8;
        const std::size_t d = 2;
        std::vector<double> x_train(n * d);
        std::vector<double> y_train(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double x0 = -1.5 + (3.0 * static_cast<double>(i) / static_cast<double>(n - 1U));
            const double x1 = 0.25 * static_cast<double>((i % 4U) + 1U);
            x_train[i] = x0;
            x_train[n + i] = x1;
            y_train[i] = std::sin(x0) + (0.5 * std::cos(x1));
        }
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);

        const std::array<double, 6> x_query{-0.8, 0.1, 1.1, 0.3, 0.7, 0.9};
        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 3U, d, 3U);

        const auto kernels = make_all_ten_built_in_kernels();
        ASSERT_EQ(kernels.size(), 10U);

        {
            SCOPED_TRACE("DoublePrecision");
            verify_all_optimizers_for_precision<cxxgpr::DoublePrecision>(kernels, x_ref, y_train, xq_ref);
        }
        {
            SCOPED_TRACE("SinglePrecision");
            verify_all_optimizers_for_precision<cxxgpr::SinglePrecision>(kernels, x_ref, y_train, xq_ref);
        }
        {
            SCOPED_TRACE("MixedPrecision<PromoteStorage>");
            verify_all_optimizers_for_precision<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(
                kernels, x_ref, y_train, xq_ref
            );
        }
        {
            SCOPED_TRACE("MixedPrecision<ReevaluateKernel>");
            verify_all_optimizers_for_precision<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(
                kernels, x_ref, y_train, xq_ref
            );
        }
    }

    /// @brief Verifies composite kernels with every transform mode survive save_with_factor()/load().
    /// @details Sweeps Identity, Standardize, MinMax, and multi-stage Pipeline preprocessing with a
    /// nested Constant * (RbfArd + Matern + Periodic) + Linear + White kernel and checks loaded
    /// predict(), loo_predict(), and neg_log_marginal_likelihood() parity.
    TEST(PathMatrixTest, CompositeKernelsTransformsAndPersistIntegration)
    {
        // Spec: REQ-04, REQ-10, DES-18, FMT-05
        const std::size_t n = 10;
        const std::size_t d = 2;
        std::vector<double> x_train(n * d);
        std::vector<double> y_train(n);
        // Generate an anisotropic 2D dataset with unscaled x1 and shifted y to stress transforms.
        for (std::size_t i = 0; i < n; ++i)
        {
            const double x0 = -1.0 + (2.0 * static_cast<double>(i) / static_cast<double>(n - 1U));
            const double x1 = 10.0 + (2.0 * static_cast<double>(i));
            x_train[i] = x0;
            x_train[n + i] = x1;
            y_train[i] = 100.0 + (5.0 * std::sin(x0)) + (0.2 * x1);
        }
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);
        const std::array<double, 4> x_query{-0.4, 0.5, 12.0, 22.0};
        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 2U, d, 2U);

        const auto composite =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
                * (cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 1.0}).value())
                    + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.0, cxxgpr::MaternNu::OneHalf).value())
                    + cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(1.0, 2.0).value())))
            + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value())
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value());

        cxxgpr::transform::Pipeline in_pipe;
        in_pipe.then(cxxgpr::transform::StandardizeInput());
        in_pipe.then(cxxgpr::transform::MinMaxInput::create(-1.0, 1.0).value());

        cxxgpr::transform::TargetPipeline tgt_pipe;
        tgt_pipe.then(cxxgpr::transform::StandardizeTarget());
        tgt_pipe.then(cxxgpr::transform::MinMaxTarget::create(0.0, 1.0).value());

        /// @brief Input/target transform configuration pair evaluated in the persistence loop.
        struct TransformCase
        {
            /// @brief Human-readable transform case label and checkpoint subdirectory name.
            std::string name;
            /// @brief Unfitted input transform pipeline or leaf.
            std::unique_ptr<cxxgpr::transform::UnfittedInputTransform> in_tf;
            /// @brief Unfitted target transform pipeline or leaf.
            std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform> tgt_tf;
        };

        std::vector<TransformCase> cases;
        cases.push_back(
            {.name = "Identity",
                .in_tf = cxxgpr::transform::IdentityInput().clone(),
                .tgt_tf = cxxgpr::transform::IdentityTarget().clone()}
        );
        cases.push_back(
            {.name = "Standardize",
                .in_tf = cxxgpr::transform::StandardizeInput().clone(),
                .tgt_tf = cxxgpr::transform::StandardizeTarget().clone()}
        );
        cases.push_back(
            {.name = "MinMax",
                .in_tf = cxxgpr::transform::MinMaxInput::create(-1.0, 1.0).value().clone(),
                .tgt_tf = cxxgpr::transform::MinMaxTarget::create(-1.0, 1.0).value().clone()}
        );
        cases.push_back({.name = "Pipeline", .in_tf = in_pipe.clone(), .tgt_tf = tgt_pipe.clone()});

        const auto temp_root = std::filesystem::temp_directory_path() / "cxxgpr_test_path_matrix_persist";
        std::error_code ec;
        std::filesystem::remove_all(temp_root, ec);

        for (const auto& tc : cases)
        {
            SCOPED_TRACE(tc.name);
            const auto trainer =
                cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(composite, cxxgpr::optimizer::Lbfgs().with_max_iterations(30U))
                    .with_input_transform(*tc.in_tf)
                    .with_target_transform(*tc.tgt_tf)
                    .with_noise_variance(1e-2)
                    .value();
            auto fitted = trainer.fit(x_ref, y_train);
            ASSERT_TRUE(fitted.has_value());
            auto orig_pred = fitted->predict(xq_ref);
            auto orig_loo = fitted->loo_predict();
            ASSERT_TRUE(orig_pred.has_value());
            ASSERT_TRUE(orig_loo.has_value());

            const auto ckpt_dir = temp_root / tc.name;
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fitted, ckpt_dir).has_value());
            {
                auto loaded = cxxgpr::persist::LoadedGpr::load(ckpt_dir);
                ASSERT_TRUE(loaded.has_value());
                auto loaded_pred = loaded->predict(xq_ref);
                auto loaded_loo = loaded->loo_predict();
                ASSERT_TRUE(loaded_pred.has_value());
                ASSERT_TRUE(loaded_loo.has_value());
                for (std::size_t j = 0; j < 2U; ++j)
                {
                    EXPECT_NEAR(loaded_pred->mean[j], orig_pred->mean[j], 1e-10);
                    EXPECT_NEAR(loaded_pred->variance[j], orig_pred->variance[j], 1e-10);
                }
                for (std::size_t i = 0; i < n; ++i)
                {
                    EXPECT_NEAR(loaded_loo->mean[i], orig_loo->mean[i], 1e-10);
                }
                auto restored = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
                ASSERT_TRUE(restored.has_value());
                EXPECT_NEAR(restored->neg_log_marginal_likelihood(), fitted->neg_log_marginal_likelihood(), 1e-10);
            }
        }
        std::filesystem::remove_all(temp_root, ec);
    }

    /// @brief Sweeps all KernelExp, DistanceCachePolicy, and CholeskyBuffer modes for precision P.
    /// @details Compares posterior predictions from every combination against the Accurate +
    /// prefer_speed() reference fit within exp_tol.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] spec Kernel specification factored across policy combinations.
    /// @param[in] x_ref Column-major n x d training input view.
    /// @param[in] y_train Training target values of length n.
    /// @param[in] xq_ref Column-major m x d query input view.
    /// @param[in] exp_tol Prediction tolerance permitted when KernelExp::FastApprox is active.
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_exp_and_policy_matrix(
        const cxxgpr::KernelSpec& spec,
        cxxgpr::MatrixRef<double> x_ref,
        std::span<const double> y_train,
        cxxgpr::MatrixRef<double> xq_ref,
        double exp_tol
    )
    {
        // Factor reference model with Accurate exp and prefer_speed() defaults.
        const std::array<cxxgpr::DistanceCachePolicy, 2> cache_policies{
            cxxgpr::DistanceCachePolicy::Cached, cxxgpr::DistanceCachePolicy::Uncached
        };
        const std::array<cxxgpr::CholeskyBuffer, 2> chol_policies{
            cxxgpr::CholeskyBuffer::Retain, cxxgpr::CholeskyBuffer::Reuse
        };

        auto ref_fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec)
                           .with_kernel_exp(cxxgpr::KernelExp::Accurate)
                           .with_prefer_speed()
                           .factor(x_ref, y_train);
        ASSERT_TRUE(ref_fit.has_value());
        auto ref_pred = ref_fit->predict(xq_ref);
        ASSERT_TRUE(ref_pred.has_value());

        const std::array<cxxgpr::KernelExp, 2> kexp_modes{cxxgpr::KernelExp::Accurate, cxxgpr::KernelExp::FastApprox};
        for (const auto kexp : kexp_modes)
        {
            for (const auto cp : cache_policies)
            {
                for (const auto cb : chol_policies)
                {
                    const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec)
                                         .with_kernel_exp(kexp)
                                         .with_distance_cache_policy(cp)
                                         .with_cholesky_buffer(cb)
                                         .factor(x_ref, y_train);
                    ASSERT_TRUE(fit.has_value());
                    auto pred = fit->predict(xq_ref);
                    ASSERT_TRUE(pred.has_value());
                    const double tol = (kexp == cxxgpr::KernelExp::Accurate) ? 1e-12 : exp_tol;
                    for (std::size_t j = 0; j < xq_ref.rows(); ++j)
                    {
                        EXPECT_NEAR(pred->mean[j], ref_pred->mean[j], tol);
                        EXPECT_NEAR(pred->variance[j], ref_pred->variance[j], tol);
                    }
                }
            }
        }
    }

    /// @brief Verifies KernelExp x DistanceCachePolicy x CholeskyBuffer matrix across precisions.
    /// @details Invokes verify_exp_and_policy_matrix() for DoublePrecision, SinglePrecision, and
    /// MixedPrecision<PromoteStorage> on a 1D RBF regression problem.
    TEST(PathMatrixTest, KernelExpAndFitPolicyMatrix)
    {
        // Spec: DES-12, DES-15
        const std::size_t n = 6;
        const std::size_t d = 1;
        const std::vector<double> x_train{0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
        const std::vector<double> y_train{1.0, 1.3, 1.2, 0.8, 0.4, 0.3};
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);
        const std::array<double, 2> x_query{0.45, 1.05};
        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 2U, d, 2U);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.8).value());

        verify_exp_and_policy_matrix<cxxgpr::DoublePrecision>(spec, x_ref, y_train, xq_ref, 1e-2);
        verify_exp_and_policy_matrix<cxxgpr::SinglePrecision>(spec, x_ref, y_train, xq_ref, 2e-2);
        verify_exp_and_policy_matrix<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(
            spec, x_ref, y_train, xq_ref, 2e-2
        );
    }

    /// @brief Verifies hybrid DistanceKernel scalar/ARD evaluation and MLL across KernelExp modes.
    /// @details Compares a combined ScalarDistance + ArdDistance + Linear kernel against coordinate
    /// apply_points() under Accurate and FastApprox and confirms finite MLL and gradients.
    TEST(PathMatrixTest, DistanceKernelScalarAndArdMatrixAcrossPrecisions)
    {
        // Spec: REQ-05, DES-10, DES-12
        const std::size_t n = 5;
        const std::size_t d = 2;
        std::vector<double> x_train{0.0, 0.4, 0.9, 1.3, 1.8, 0.1, 0.5, 0.2, 0.8, 1.1};
        std::vector<double> y_train{0.2, 0.7, 1.0, 0.6, 0.1};
        const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), n, d, n);

        cxxgpr::ScalarDistance s_slot;
        auto a_slot = cxxgpr::ArdDistance::create(d);
        ASSERT_TRUE(a_slot.has_value());
        const auto rbf = cxxgpr::RbfKernel::create(1.0).value();
        const auto rq_ard = cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{0.9, 1.3}, 1.5).value();
        const auto lin = cxxgpr::LinearKernel::create(0.5).value();

        const auto coord_spec = cxxgpr::KernelSpec(rbf) + cxxgpr::KernelSpec(rq_ard) + cxxgpr::KernelSpec(lin);
        auto rq_dist = a_slot->rational_quadratic_ard(rq_ard);
        ASSERT_TRUE(rq_dist.has_value());
        const auto dist_with_pts = (s_slot.rbf(rbf) + *rq_dist) + cxxgpr::KernelSpec(lin);

        // Precompute squared Euclidean scalar and per-dimension ARD distances.
        std::vector<double> d_scalar(n * n, 0.0);
        std::vector<double> d_ard0(n * n, 0.0);
        std::vector<double> d_ard1(n * n, 0.0);
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                const double dx0 = x_train[row] - x_train[col];
                const double dx1 = x_train[n + row] - x_train[n + col];
                d_ard0[(col * n) + row] = dx0 * dx0;
                d_ard1[(col * n) + row] = dx1 * dx1;
                d_scalar[(col * n) + row] = (dx0 * dx0) + (dx1 * dx1);
            }
        }

        cxxgpr::DistanceSource src;
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_mats{
            cxxgpr::MatrixRef<double>(d_ard0.data(), n, n, n), cxxgpr::MatrixRef<double>(d_ard1.data(), n, n, n)
        };
        src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(d_scalar.data(), n, n, n))
            .bind_ard(a_slot->slot(), ard_mats);

        const auto lik = cxxgpr::GaussianLikelihood::create(0.05).value();
        const auto jitter = cxxgpr::JitterPolicy::fixed(1e-8).value();
        std::vector<double> scratch_buf(n * n, 0.0);
        const auto scratch_mut = cxxgpr::MatrixMut<double>(scratch_buf.data(), n, n, n);

        const std::array<cxxgpr::KernelExp, 2> kexp_modes{cxxgpr::KernelExp::Accurate, cxxgpr::KernelExp::FastApprox};
        for (const auto kexp : kexp_modes)
        {
            auto ck_coord = cxxgpr::CompiledKernel<double>::from_spec(coord_spec);
            auto ck_dist = dist_with_pts.compile<double>();
            ck_coord.set_kernel_exp(kexp);
            ck_dist.set_kernel_exp(kexp);

            const auto store_f64 = ck_dist.prepare_square(src);
            ASSERT_TRUE(store_f64.has_value());

            std::vector<double> k_coord(n * n, 0.0);
            std::vector<double> k_dist(n * n, 0.0);
            const auto kc_mut = cxxgpr::MatrixMut<double>(k_coord.data(), n, n, n);
            const auto kd_mut = cxxgpr::MatrixMut<double>(k_dist.data(), n, n, n);

            ASSERT_TRUE(ck_coord.apply_points(x_ref, kc_mut, cxxgpr::Triangle::Lower, scratch_mut).has_value());
            ASSERT_TRUE(ck_dist.apply(*store_f64, x_ref, kd_mut, cxxgpr::Triangle::Lower, scratch_mut).has_value());
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    EXPECT_NEAR(k_dist[(col * n) + row], k_coord[(col * n) + row], 1e-12);
                }
            }

            // Verify MLL + grad on coordinate path matches finite values for each KernelExp.
            auto buf = cxxgpr::FitBuffers<cxxgpr::DoublePrecision>::create(n, cxxgpr::FitPolicy::prefer_speed());
            ASSERT_TRUE(buf.has_value());
            std::vector<double> grad(ck_coord.num_params() + 1U, 0.0);
            auto mll_res = cxxgpr::mll_and_gradient_into(ck_coord, lik, x_ref, y_train, jitter, *buf, grad);
            ASSERT_TRUE(mll_res.has_value());
            EXPECT_TRUE(std::isfinite(*mll_res));
            for (const double g : grad)
            {
                EXPECT_TRUE(std::isfinite(g));
            }
        }
    }

}  // namespace
