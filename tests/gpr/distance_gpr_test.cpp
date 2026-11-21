// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-05, DES-21
/// @file distance_gpr_test.cpp
/// @brief Verifies DistanceGpr / FittedDistanceGpr against the coordinate model fed Euclidean squared distances.
/// @details The independent reference is the coordinate Gpr / FittedGpr: a supplied-distance model whose tables
/// hold the Euclidean squared distances of a coordinate dataset must reproduce the coordinate model in objective,
/// weights, factor, optimum, predictions, covariance, samples and leave-one-out. Negative controls check slot
/// misuse, invalid tables and shape mismatches, and that failures leave the trainer usable.

#include "cxxgpr/gpr/distance_gpr.hpp"

#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/transform/transform.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace
{

    /// @brief Synthetic regression dataset with n points in d dimensions.
    struct Data
    {
        /// @brief Column-major training coordinates, n x d.
        std::vector<double> x;
        /// @brief Training targets.
        std::vector<double> y;
        /// @brief Column-major query coordinates, m x d.
        std::vector<double> q;
        /// @brief Number of training points.
        std::size_t n{0};
        /// @brief Number of query points.
        std::size_t m{0};
        /// @brief Dimension.
        std::size_t d{0};
    };

    /// @brief Builds a smooth noisy dataset deterministically.
    /// @param[in] n Number of training points.
    /// @param[in] m Number of query points.
    /// @param[in] d Dimension.
    /// @return Dataset.
    Data make_data(std::size_t n, std::size_t m, std::size_t d)
    {
        Data data;
        data.n = n;
        data.m = m;
        data.d = d;
        data.x.resize(n * d);
        data.y.resize(n);
        data.q.resize(m * d);
        cxxgpr::Xoshiro256PlusPlus rng(11);
        for (double& v : data.x)
        {
            v = 2.0 * rng.unit();
        }
        for (double& v : data.q)
        {
            v = 2.0 * rng.unit();
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            double s = 0.0;
            for (std::size_t k = 0; k < d; ++k)
            {
                s += std::sin(1.5 * data.x[(k * n) + i]);
            }
            data.y[i] = s + (0.05 * rng.unit_normal());
        }
        return data;
    }

    /// @brief Squared per-dimension differences or total squared distance between two column-major point sets.
    /// @param[in] a First set (na x d).
    /// @param[in] na Rows of the first set.
    /// @param[in] b Second set (nb x d).
    /// @param[in] nb Rows of the second set.
    /// @param[in] d Dimension.
    /// @param[in] dim Dimension index, or d for the total squared distance.
    /// @return Column-major na x nb table.
    std::vector<double> table(
        const std::vector<double>& a,
        std::size_t na,
        const std::vector<double>& b,
        std::size_t nb,
        std::size_t d,
        std::size_t dim
    )
    {
        std::vector<double> out(na * nb, 0.0);
        for (std::size_t j = 0; j < nb; ++j)
        {
            for (std::size_t i = 0; i < na; ++i)
            {
                double acc = 0.0;
                for (std::size_t k = 0; k < d; ++k)
                {
                    if (dim == d || dim == k)
                    {
                        const double diff = a[(k * na) + i] - b[(k * nb) + j];
                        acc += diff * diff;
                    }
                }
                out[(j * na) + i] = acc;
            }
        }
        return out;
    }

    /// @brief Relative difference of two vectors measured against the larger of their max-norms (and one).
    /// @tparam A First vector-like type.
    /// @tparam B Second vector-like type.
    /// @param[in] a First vector.
    /// @param[in] b Second vector.
    /// @return Max absolute difference divided by max(1, max |a|).
    template <typename A, typename B> double rel_diff(const A& a, const B& b)
    {
        double scale = 1.0;
        double diff = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            scale = std::max(scale, std::abs(static_cast<double>(a[i])));
            diff = std::max(diff, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        }
        return diff / scale;
    }

    /// @brief Coordinate reference kernel: Constant(1.3) * Rbf(0.8) + White(0.01).
    /// @return Kernel specification.
    cxxgpr::KernelSpec coordinate_scalar_kernel()
    {
        return (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.3).value())
                   * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.8).value()))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.01).value());
    }

    /// @brief Supplied-distance counterpart of coordinate_scalar_kernel().
    /// @param[in] slot Scalar distance slot of the Euclidean table.
    /// @return Distance kernel with the same leaves and parameters.
    cxxgpr::DistanceKernel<cxxgpr::DistanceOnly> distance_scalar_kernel(const cxxgpr::ScalarDistance& slot)
    {
        return (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.3).value())
                   * slot.rbf(cxxgpr::RbfKernel::create(0.8).value()))
            + cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::WhiteKernel::create(0.01).value());
    }

    /// @brief Fixture-free helper owning the Euclidean tables of a dataset.
    struct Tables
    {
        /// @brief Training table n x n.
        std::vector<double> train;
        /// @brief Cross table n x m.
        std::vector<double> cross;
        /// @brief Query table m x m.
        std::vector<double> query;
    };

    /// @brief Builds the Euclidean tables for the scalar slot.
    /// @param[in] data Dataset.
    /// @return Tables.
    Tables euclid_tables(const Data& data)
    {
        return Tables{
            .train = table(data.x, data.n, data.x, data.n, data.d, data.d),
            .cross = table(data.x, data.n, data.q, data.m, data.d, data.d),
            .query = table(data.q, data.m, data.q, data.m, data.d, data.d),
        };
    }

    TEST(DistanceGprTest, FactorMatchesTheCoordinateModelAtFixedHyperparameters)
    {
        const Data data = make_data(14, 5, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));

        const auto coord =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(coordinate_scalar_kernel()).with_noise_variance(0.02).value();
        const auto dist = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(slot))
                              .with_noise_variance(0.02)
                              .value();
        auto reference = coord.factor(data.x, data.n, data.d, data.y);
        auto model = dist.factor(train, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(model.has_value()) << model.error().message();

        EXPECT_NEAR(model->neg_log_marginal_likelihood(), reference->optimizer_result().value, 1e-10);
        EXPECT_LT(rel_diff(model->alpha(), reference->alpha()), 1e-10);
        // Compare the lower triangles (the strictly upper part of the stored factor is unused).
        double l_diff = 0.0;
        for (std::size_t col = 0; col < data.n; ++col)
        {
            for (std::size_t row = col; row < data.n; ++row)
            {
                l_diff = std::max(l_diff, std::abs(model->l_factor()(row, col) - reference->l_factor()(row, col)));
            }
        }
        EXPECT_LT(l_diff, 1e-10);
        EXPECT_EQ(model->n_train(), data.n);
        EXPECT_EQ(model->n_features(), 0U);
        EXPECT_TRUE(model->optimizer_result().converged);
        EXPECT_EQ(model->optimizer_result().iterations, 0U);
    }

    TEST(DistanceGprTest, FitReachesTheCoordinateOptimumForTheDerivativeBasedOptimizers)
    {
        const Data data = make_data(16, 4, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));

        auto check = [&]<typename O>(const O& optimizer)
        {
            const auto coord = cxxgpr::Gpr<O>(coordinate_scalar_kernel(), optimizer).with_noise_variance(0.05).value();
            const auto dist =
                cxxgpr::DistanceGpr<O>(distance_scalar_kernel(slot), optimizer).with_noise_variance(0.05).value();
            auto reference = coord.fit(data.x, data.n, data.d, data.y);
            auto model = dist.fit(train, data.y);
            ASSERT_TRUE(reference.has_value());
            ASSERT_TRUE(model.has_value()) << model.error().message();
            // The same algorithm with matching derivatives follows the same path: optimum and value agree.
            EXPECT_NEAR(model->neg_log_marginal_likelihood(), reference->optimizer_result().value, 1e-7);
            std::vector<double> p_ref(reference->kernel_spec().num_params());
            std::vector<double> p_dist(model->kernel().num_params());
            ASSERT_EQ(p_ref.size(), p_dist.size());
            ASSERT_TRUE(reference->kernel_spec().get_params(p_ref).has_value());
            ASSERT_TRUE(model->kernel().get_params(p_dist).has_value());
            EXPECT_LT(rel_diff(p_dist, p_ref), 1e-5);
            EXPECT_NEAR(
                model->likelihood().noise_variance(), reference->likelihood().noise_variance(),
                1e-5 * std::max(1.0, reference->likelihood().noise_variance())
            );
        };
        check(cxxgpr::optimizer::Lbfgs{});
        check(cxxgpr::optimizer::TrustRegion{});
    }

    TEST(DistanceGprTest, GradientFreeOptimizersImproveTheObjective)
    {
        const Data data = make_data(14, 4, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        auto start = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(slot))
                         .with_noise_variance(0.05)
                         .value()
                         .factor(train, data.y);
        ASSERT_TRUE(start.has_value());

        auto nm = cxxgpr::DistanceGpr<cxxgpr::optimizer::NelderMead>(distance_scalar_kernel(slot))
                      .with_noise_variance(0.05)
                      .value()
                      .fit(train, data.y);
        auto fsa = cxxgpr::DistanceGpr<cxxgpr::optimizer::FastSimulatedAnnealing>(distance_scalar_kernel(slot))
                       .with_noise_variance(0.05)
                       .value()
                       .fit(train, data.y);
        ASSERT_TRUE(nm.has_value()) << nm.error().message();
        ASSERT_TRUE(fsa.has_value()) << fsa.error().message();
        EXPECT_LT(nm->neg_log_marginal_likelihood(), start->neg_log_marginal_likelihood());
        EXPECT_LT(fsa->neg_log_marginal_likelihood(), start->neg_log_marginal_likelihood());
    }

    TEST(DistanceGprTest, PredictionsCovarianceSamplesAndLooMatchTheCoordinateModel)
    {
        const Data data = make_data(15, 6, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        cxxgpr::DistanceSource query;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        query.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.query.data(), data.m, data.m));

        const cxxgpr::transform::StandardizeTarget standardize;
        const auto coord = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(coordinate_scalar_kernel())
                               .with_noise_variance(0.02)
                               .value()
                               .with_target_transform(standardize);
        const auto dist = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(slot))
                              .with_noise_variance(0.02)
                              .value()
                              .with_target_transform(standardize);
        auto reference = coord.factor(data.x, data.n, data.d, data.y);
        auto model = dist.factor(train, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(model.has_value()) << model.error().message();

        for (const auto kind : {cxxgpr::VarianceKind::Observation, cxxgpr::VarianceKind::Latent})
        {
            const cxxgpr::PredictOptions options{.variance_kind = kind};
            auto p_ref = reference->predict(data.q, data.m, data.d, options);
            auto p_dist = model->predict(cross, options);
            ASSERT_TRUE(p_ref.has_value());
            ASSERT_TRUE(p_dist.has_value()) << p_dist.error().message();
            EXPECT_LT(rel_diff(p_dist->mean, p_ref->mean), 1e-10);
            EXPECT_LT(rel_diff(p_dist->variance, p_ref->variance), 1e-10);

            auto c_ref = reference->predict_covariance(data.q, data.m, data.d, options);
            auto c_dist = model->predict_covariance(cross, query, options);
            ASSERT_TRUE(c_ref.has_value());
            ASSERT_TRUE(c_dist.has_value()) << c_dist.error().message();
            EXPECT_LT(rel_diff(c_dist->covariance, c_ref->covariance), 1e-9);
            EXPECT_EQ(c_dist->n_points, data.m);

            auto l_ref = reference->loo_predict(options);
            auto l_dist = model->loo_predict(options);
            ASSERT_TRUE(l_ref.has_value());
            ASSERT_TRUE(l_dist.has_value());
            EXPECT_LT(rel_diff(l_dist->mean, l_ref->mean), 1e-9);
            EXPECT_LT(rel_diff(l_dist->variance, l_ref->variance), 1e-9);
        }

        auto s_ref = reference->sample(data.q, data.m, data.d, 5U, 7U);
        auto s_dist = model->sample(cross, query, 5U, 7U);
        ASSERT_TRUE(s_ref.has_value());
        ASSERT_TRUE(s_dist.has_value()) << s_dist.error().message();
        ASSERT_EQ(s_dist->size(), data.m * 5U);
        EXPECT_LT(rel_diff(*s_dist, *s_ref), 1e-8);
        // Deterministic for a fixed seed, different for another seed.
        auto again = model->sample(cross, query, 5U, 7U);
        auto other = model->sample(cross, query, 5U, 8U);
        ASSERT_TRUE(again.has_value());
        ASSERT_TRUE(other.has_value());
        EXPECT_EQ(*again, *s_dist);
        EXPECT_NE(*other, *s_dist);
        auto none = model->sample(cross, query, 0U, 7U);
        ASSERT_TRUE(none.has_value());
        EXPECT_TRUE(none->empty());
    }

    TEST(DistanceGprTest, ArdSlotMatchesTheCoordinateArdModel)
    {
        const Data data = make_data(13, 4, 2);
        const auto slot = cxxgpr::ArdDistance::create(2).value();
        const std::array<double, 2> length_scales{0.9, 1.4};
        const auto kernel_dist =
            cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(0.7).value())
            * slot.rbf_ard(cxxgpr::RbfArdKernel::create(length_scales).value()).value();
        const auto kernel_coord = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(0.7).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scales).value());

        const auto t0 = table(data.x, data.n, data.x, data.n, data.d, 0);
        const auto t1 = table(data.x, data.n, data.x, data.n, data.d, 1);
        const auto c0 = table(data.x, data.n, data.q, data.m, data.d, 0);
        const auto c1 = table(data.x, data.n, data.q, data.m, data.d, 1);
        const std::array<cxxgpr::MatrixRef<double>, 2> train_tables{
            cxxgpr::MatrixRef<double>(t0.data(), data.n, data.n), cxxgpr::MatrixRef<double>(t1.data(), data.n, data.n)
        };
        const std::array<cxxgpr::MatrixRef<double>, 2> cross_tables{
            cxxgpr::MatrixRef<double>(c0.data(), data.n, data.m), cxxgpr::MatrixRef<double>(c1.data(), data.n, data.m)
        };
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_ard(slot.slot(), train_tables);
        cross.bind_ard(slot.slot(), cross_tables);

        auto reference = cxxgpr::Gpr<cxxgpr::optimizer::TrustRegion>(kernel_coord)
                             .with_noise_variance(0.03)
                             .value()
                             .fit(data.x, data.n, data.d, data.y);
        auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::TrustRegion>(kernel_dist)
                         .with_noise_variance(0.03)
                         .value()
                         .fit(train, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(model.has_value()) << model.error().message();
        EXPECT_NEAR(model->neg_log_marginal_likelihood(), reference->optimizer_result().value, 1e-6);
        auto p_ref = reference->predict(data.q, data.m, data.d);
        auto p_dist = model->predict(cross);
        ASSERT_TRUE(p_ref.has_value());
        ASSERT_TRUE(p_dist.has_value()) << p_dist.error().message();
        EXPECT_LT(rel_diff(p_dist->mean, p_ref->mean), 1e-5);
        EXPECT_LT(rel_diff(p_dist->variance, p_ref->variance), 1e-5);
    }

    TEST(DistanceGprTest, WithPointsMixesSuppliedDistancesAndCoordinateLeaves)
    {
        const Data data = make_data(12, 4, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        // k = Constant(1.1) * Rbf(0.9)(supplied) + 0.5 * Linear on the coordinates.
        const cxxgpr::DistanceKernel<cxxgpr::WithPoints> kernel_dist =
            (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.1).value())
                * slot.rbf(cxxgpr::RbfKernel::create(0.9).value()))
                .with_points()
            + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value());
        const auto kernel_coord = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.1).value())
                                      * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.9).value()))
            + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value());
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        const cxxgpr::MatrixRef<double> x(data.x.data(), data.n, data.d, data.n);
        const cxxgpr::MatrixRef<double> q(data.q.data(), data.m, data.d, data.m);

        auto reference = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(kernel_coord)
                             .with_noise_variance(0.03)
                             .value()
                             .factor(data.x, data.n, data.d, data.y);
        auto model =
            cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(kernel_dist)
                .with_noise_variance(0.03)
                .value()
                .factor(train, x, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(model.has_value()) << model.error().message();
        EXPECT_EQ(model->n_features(), data.d);
        EXPECT_NEAR(model->neg_log_marginal_likelihood(), reference->optimizer_result().value, 1e-10);
        auto p_ref = reference->predict(data.q, data.m, data.d);
        auto p_dist = model->predict(cross, q);
        ASSERT_TRUE(p_ref.has_value());
        ASSERT_TRUE(p_dist.has_value()) << p_dist.error().message();
        EXPECT_LT(rel_diff(p_dist->mean, p_ref->mean), 1e-10);
        EXPECT_LT(rel_diff(p_dist->variance, p_ref->variance), 1e-10);

        // Query coordinates with the wrong feature count are rejected.
        const std::vector<double> bad(data.m * 3U, 0.5);
        auto wrong = model->predict(cross, cxxgpr::MatrixRef<double>(bad.data(), data.m, 3U, data.m));
        ASSERT_FALSE(wrong.has_value());
        EXPECT_EQ(wrong.error().code(), cxxgpr::GprErrorCode::DimensionMismatch);
    }

    TEST(DistanceGprTest, SinglePrecisionTracksTheDoubleModel)
    {
        const Data data = make_data(14, 4, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        auto reference = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(slot))
                             .with_noise_variance(0.05)
                             .value()
                             .factor(train, data.y);
        auto single =
            cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::SinglePrecision>(distance_scalar_kernel(slot))
                .with_noise_variance(0.05)
                .value()
                .factor(train, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(single.has_value()) << single.error().message();
        auto p_ref = reference->predict(cross);
        auto p_single = single->predict(cross);
        ASSERT_TRUE(p_ref.has_value());
        ASSERT_TRUE(p_single.has_value());
        EXPECT_LT(rel_diff(p_single->mean, p_ref->mean), 1e-3);
        EXPECT_LT(rel_diff(p_single->variance, p_ref->variance), 1e-3);
    }

    TEST(DistanceGprTest, RejectsInvalidInputsAndLeavesTheTrainerUsable)
    {
        const Data data = make_data(10, 3, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        const cxxgpr::ScalarDistance other_slot;
        const auto trainer = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(slot))
                                 .with_noise_variance(0.05)
                                 .value();
        cxxgpr::DistanceSource train;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));

        // Empty and mismatched targets, non-finite targets.
        EXPECT_EQ(trainer.fit(train, std::vector<double>{}).error().code(), cxxgpr::GprErrorCode::EmptyInput);
        EXPECT_EQ(
            trainer.fit(train, std::vector<double>(data.n + 1U, 0.0)).error().code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );
        std::vector<double> nan_y = data.y;
        nan_y[3] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ(trainer.fit(train, nan_y).error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

        // Missing, foreign and asymmetric tables.
        cxxgpr::DistanceSource empty;
        EXPECT_EQ(trainer.fit(empty, data.y).error().code(), cxxgpr::GprErrorCode::DistanceSlot);
        cxxgpr::DistanceSource foreign;
        foreign.bind_scalar(other_slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        EXPECT_FALSE(trainer.fit(foreign, data.y).has_value());
        std::vector<double> asym = t.train;
        asym[1] += 0.5;
        cxxgpr::DistanceSource bad;
        bad.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(asym.data(), data.n, data.n));
        EXPECT_EQ(trainer.fit(bad, data.y).error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        // The same table is accepted once the caller opts into repairing it.
        cxxgpr::DistanceSource repaired;
        repaired.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(asym.data(), data.n, data.n)).tidy(1.0);
        EXPECT_TRUE(trainer.fit(repaired, data.y).has_value());

        // Table order disagreeing with the targets.
        EXPECT_EQ(
            trainer.fit(train, std::vector<double>(data.n - 1U, 0.0)).error().code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );

        // The trainer is unchanged by the failures above and still fits.
        auto model = trainer.fit(train, data.y);
        ASSERT_TRUE(model.has_value());

        // Query supply checks: wrong row count, missing slot, non-finite table, mismatched query tables.
        std::vector<double> short_cross(data.m * (data.n - 1U), 0.5);
        cxxgpr::DistanceSource bad_cross;
        bad_cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(short_cross.data(), data.n - 1U, data.m));
        EXPECT_EQ(model->predict(bad_cross).error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        EXPECT_EQ(model->predict(empty).error().code(), cxxgpr::GprErrorCode::DistanceSlot);
        std::vector<double> nan_cross = t.cross;
        nan_cross[0] = std::numeric_limits<double>::quiet_NaN();
        cxxgpr::DistanceSource nan_source;
        nan_source.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(nan_cross.data(), data.n, data.m));
        EXPECT_EQ(model->predict(nan_source).error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        cxxgpr::DistanceSource cross;
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        cxxgpr::DistanceSource wrong_query;
        const std::vector<double> small_query(2U * 2U, 0.0);
        wrong_query.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(small_query.data(), 2U, 2U));
        EXPECT_EQ(model->predict_covariance(cross, wrong_query).error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        // A failed query leaves the model usable.
        EXPECT_TRUE(model->predict(cross).has_value());
    }

    TEST(DistanceGprTest, CopiesRefitAndIntoTrainerKeepTheFittedModel)
    {
        const Data data = make_data(12, 3, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs>(distance_scalar_kernel(slot))
                         .with_noise_variance(0.05)
                         .value()
                         .fit(train, data.y);
        ASSERT_TRUE(model.has_value()) << model.error().message();

        const auto copy = *model;
        auto p1 = model->predict(cross);
        auto p2 = copy.predict(cross);
        ASSERT_TRUE(p1.has_value());
        ASSERT_TRUE(p2.has_value());
        EXPECT_EQ(p1->mean, p2->mean);
        EXPECT_EQ(p1->variance, p2->variance);

        // Repeated predictions into the same buffer give identical results.
        cxxgpr::Prediction<double> out;
        ASSERT_TRUE(model->predict_into(cross, out).has_value());
        const auto first = out;
        ASSERT_TRUE(model->predict_into(cross, out).has_value());
        EXPECT_EQ(first, out);

        // Refit on the same data from the fitted hyperparameters reaches the same objective.
        auto refit = model->refit(train, data.y);
        ASSERT_TRUE(refit.has_value()) << refit.error().message();
        EXPECT_NEAR(refit->neg_log_marginal_likelihood(), model->neg_log_marginal_likelihood(), 1e-6);
        EXPECT_LE(refit->optimizer_result().iterations, model->optimizer_result().iterations);
        EXPECT_NEAR(model->into_trainer().likelihood().noise_variance(), model->likelihood().noise_variance(), 1e-15);
    }

    TEST(DistanceGprTest, MixedPrecisionRefinesAgainstTheExactSystemForBothResidualFormulas)
    {
        const Data data = make_data(60, 5, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        // A small noise variance makes the system badly enough conditioned for float storage to matter.
        auto reference = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(slot))
                             .with_noise_variance(1e-3)
                             .value()
                             .factor(train, data.y);
        auto promote = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(
            distance_scalar_kernel(slot)
        )
                           .with_noise_variance(1e-3)
                           .value()
                           .factor(train, data.y);
        auto reeval = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(
            distance_scalar_kernel(slot)
        )
                          .with_noise_variance(1e-3)
                          .value()
                          .factor(train, data.y);
        auto single =
            cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::SinglePrecision>(distance_scalar_kernel(slot))
                .with_noise_variance(1e-3)
                .value()
                .factor(train, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(promote.has_value()) << promote.error().message();
        ASSERT_TRUE(reeval.has_value()) << reeval.error().message();
        ASSERT_TRUE(single.has_value());
        EXPECT_FALSE(reference->refinement_summary().has_value());
        ASSERT_TRUE(promote->refinement_summary().has_value());
        ASSERT_TRUE(reeval->refinement_summary().has_value());
        EXPECT_FALSE(promote->refinement_summary()->used_fallback());
        EXPECT_EQ(reeval->refinement_summary()->status, cxxgpr::RefinementStatus::Converged);

        // ReevaluateKernel reaches the double-precision weights; PromoteStorage keeps the storage rounding.
        const double promote_err = rel_diff(promote->alpha(), reference->alpha());
        const double reeval_err = rel_diff(reeval->alpha(), reference->alpha());
        const double single_err = rel_diff(single->alpha(), reference->alpha());
        EXPECT_LT(reeval_err, 1e-5);
        EXPECT_LT(promote_err, 1e-2);
        EXPECT_LT(reeval_err, single_err);

        auto p_ref = reference->predict(cross);
        auto p_reeval = reeval->predict(cross);
        ASSERT_TRUE(p_ref.has_value());
        ASSERT_TRUE(p_reeval.has_value()) << p_reeval.error().message();
        EXPECT_LT(rel_diff(p_reeval->mean, p_ref->mean), 1e-3);
    }

}  // namespace
