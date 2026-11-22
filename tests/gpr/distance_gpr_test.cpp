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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "allocation_audit.hpp"

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

    /// @brief Returns true when two vectors hold bit-identical elements.
    /// @tparam A First vector-like type.
    /// @tparam B Second vector-like type.
    /// @param[in] a First vector.
    /// @param[in] b Second vector.
    /// @return True when sizes match and every element has the same bit pattern.
    template <typename A, typename B> bool bits_equal(const A& a, const B& b)
    {
        if (a.size() != b.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (std::bit_cast<std::uint64_t>(static_cast<double>(a[i]))
                != std::bit_cast<std::uint64_t>(static_cast<double>(b[i])))
            {
                return false;
            }
        }
        return true;
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

    /// @brief Verifies that factoring on supplied distances reproduces the coordinate model at fixed hyperparameters.
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

    /// @brief Verifies that fitting with each derivative-based optimizer reaches the same optimum as the coordinate
    /// model.
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

    /// @brief Verifies that the gradient-free optimizers lower the negative log marginal likelihood from the starting
    /// point.
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

    /// @brief Verifies that predictions, covariance, samples and leave-one-out results match the coordinate model.
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

    /// @brief Verifies that a per-dimension distance slot reproduces the coordinate ARD model.
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

    /// @brief Verifies that the with-points mode combines supplied-distance leaves with coordinate leaves correctly.
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

    /// @brief Verifies that the single-precision model tracks the double-precision model within a loose tolerance.
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

    /// @brief Verifies that invalid inputs are rejected with the expected error codes and the trainer remains usable.
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
        const std::vector<double> small_query(4U, 0.0);
        wrong_query.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(small_query.data(), 2U, 2U));
        EXPECT_EQ(model->predict_covariance(cross, wrong_query).error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        // A failed query leaves the model usable.
        EXPECT_TRUE(model->predict(cross).has_value());
    }

    /// @brief Verifies that copies, refit and conversion back into a trainer preserve the fitted model.
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

        // A copy owns its own query workspace: interleaving batches of different sizes on the copy and on the
        // original must give each the predictions of a fresh model fitted the same way (bitwise).
        auto fresh = cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs>(distance_scalar_kernel(slot))
                         .with_noise_variance(0.05)
                         .value()
                         .fit(train, data.y);
        ASSERT_TRUE(fresh.has_value());
        const std::vector<double> small_table(t.cross.begin(), t.cross.begin() + static_cast<std::ptrdiff_t>(data.n));
        cxxgpr::DistanceSource cross_small;
        cross_small.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(small_table.data(), data.n, 1U));
        auto expected_large = fresh->predict(cross);
        auto expected_small = fresh->predict(cross_small);
        ASSERT_TRUE(expected_large.has_value());
        ASSERT_TRUE(expected_small.has_value());
        // Copy-construct from another model, then copy-assign: the assigned copy must still own its workspace.
        auto copy = *fresh;
        copy = *model;
        cxxgpr::Prediction<double> on_copy;
        cxxgpr::Prediction<double> on_model;
        for (int round = 0; round < 2; ++round)
        {
            ASSERT_TRUE(copy.predict_into(cross_small, on_copy).has_value());
            ASSERT_TRUE(model->predict_into(cross, on_model).has_value());
            EXPECT_TRUE(bits_equal(on_copy.mean, expected_small->mean));
            EXPECT_TRUE(bits_equal(on_copy.variance, expected_small->variance));
            EXPECT_TRUE(bits_equal(on_model.mean, expected_large->mean));
            EXPECT_TRUE(bits_equal(on_model.variance, expected_large->variance));
            ASSERT_TRUE(copy.predict_into(cross, on_copy).has_value());
            ASSERT_TRUE(model->predict_into(cross_small, on_model).has_value());
            EXPECT_TRUE(bits_equal(on_copy.mean, expected_large->mean));
            EXPECT_TRUE(bits_equal(on_model.mean, expected_small->mean));
        }

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

    /// @brief Verifies that mixed-precision refinement converges against the exact system for both residual formulas.
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
        const auto promote_summary = promote->refinement_summary();
        const auto reeval_summary = reeval->refinement_summary();
        if (!promote_summary.has_value() || !reeval_summary.has_value())
        {
            FAIL() << "mixed-precision refinement summary is missing";
        }
        EXPECT_FALSE(promote_summary->used_fallback());
        EXPECT_EQ(reeval_summary->status, cxxgpr::RefinementStatus::Converged);

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

    /// @brief Verifies that results are bitwise identical regardless of the number of threads.
    TEST(DistanceGprTest, ResultsAreBitwiseIdenticalAcrossThreadCounts)
    {
        const Data data = make_data(420, 40, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        cxxgpr::DistanceSource query;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        query.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.query.data(), data.m, data.m));
        auto run = [&](int threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs>(distance_scalar_kernel(slot))
                             .with_noise_variance(0.05)
                             .value()
                             .with_optimizer(cxxgpr::optimizer::Lbfgs{}.with_max_iterations(4))
                             .fit(train, data.y);
            EXPECT_TRUE(model.has_value());
            return model;
        };
#ifdef _OPENMP
        const int saved_threads = omp_get_max_threads();
#endif
        auto reference = run(1);
        if (!reference.has_value())
        {
            FAIL() << "single-thread fit failed";
        }
        auto ref_pred = reference->predict_covariance(cross, query);
        if (!ref_pred.has_value())
        {
            FAIL() << "single-thread covariance failed";
        }
        for (const int threads : {2, 4})
        {
            auto other = run(threads);
            if (!other.has_value())
            {
                FAIL() << "fit failed for threads " << threads;
            }
            EXPECT_EQ(other->neg_log_marginal_likelihood(), reference->neg_log_marginal_likelihood());
            ASSERT_EQ(other->alpha().size(), reference->alpha().size());
            for (std::size_t i = 0; i < data.n; ++i)
            {
                EXPECT_EQ(other->alpha()[i], reference->alpha()[i]) << "threads " << threads << " index " << i;
            }
            auto pred = other->predict_covariance(cross, query);
            if (!pred.has_value())
            {
                FAIL() << "covariance failed for threads " << threads;
            }
            EXPECT_EQ(pred->mean, ref_pred->mean);
            EXPECT_EQ(pred->covariance, ref_pred->covariance);
        }
#ifdef _OPENMP
        omp_set_num_threads(saved_threads);
#endif
    }

    /// @brief Values, gradients and Hessians recorded by ProbeOptimizer at a few parameter vectors.
    struct ProbeRecord
    {
        /// @brief Objective values, one per probed parameter vector.
        std::vector<double> values;
        /// @brief Gradients, concatenated (dim entries per probed vector).
        std::vector<double> grads;
        /// @brief Hessians, concatenated (dim * dim entries per probed vector).
        std::vector<double> hessians;
    };

    /// @brief Optimizer that records the objective, its gradient and its Hessian and returns the start unchanged.
    /// @details It probes the start and two shifted parameter vectors so that the comparison covers more than one
    /// point; the shifts stay well inside the default intervals.
    struct ProbeOptimizer
    {
        /// @brief The probe evaluates whole objectives only.
        [[maybe_unused]] static constexpr bool uses_change_indices = false;
        /// @brief Destination of the recorded values (owned by the test).
        ProbeRecord* record = nullptr;

        /// @brief Records value, gradient and Hessian at three parameter vectors.
        /// @param[in,out] obj Objective under test.
        /// @param[in] initial Starting log-space parameters.
        /// @return The starting point as the optimum, or the first evaluation error.
        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable& obj, std::span<const double> initial
        ) const
        {
            const std::size_t p = obj.dim();
            std::vector<double> grad(p);
            std::vector<double> hess(p * p);
            double first_value = 0.0;
            for (const double shift : {0.0, 0.15, -0.2})
            {
                std::vector<double> theta(initial.begin(), initial.end());
                for (std::size_t i = 0; i < p; ++i)
                {
                    theta[i] += shift * (1.0 + (0.1 * static_cast<double>(i)));
                }
                auto value = obj.value_and_gradient_into(theta, grad);
                if (!value.has_value())
                {
                    return std::unexpected(std::move(value).error());
                }
                if (auto h = obj.hessian_into(theta, hess); !h.has_value())
                {
                    return std::unexpected(std::move(h).error());
                }
                if (shift == 0.0)
                {
                    first_value = *value;
                }
                record->values.push_back(*value);
                record->grads.insert(record->grads.end(), grad.begin(), grad.end());
                record->hessians.insert(record->hessians.end(), hess.begin(), hess.end());
            }
            return cxxgpr::optimizer::OptimizerResult{
                .params = std::vector<double>(initial.begin(), initial.end()),
                .value = first_value,
                .iterations = 0U,
                .converged = true,
            };
        }
    };

    /// @brief Expects that two probe records agree within a relative tolerance in every value and derivative.
    /// @param[in] dist Record of the supplied-distance objective.
    /// @param[in] coord Record of the coordinate objective (the independent reference).
    /// @param[in] tol Relative tolerance against max(1, max |reference|).
    void expect_probes_match(const ProbeRecord& dist, const ProbeRecord& coord, double tol)
    {
        ASSERT_EQ(dist.values.size(), coord.values.size());
        ASSERT_EQ(dist.grads.size(), coord.grads.size());
        ASSERT_EQ(dist.hessians.size(), coord.hessians.size());
        ASSERT_FALSE(coord.values.empty());
        EXPECT_LT(rel_diff(coord.values, dist.values), tol);
        EXPECT_LT(rel_diff(coord.grads, dist.grads), tol);
        EXPECT_LT(rel_diff(coord.hessians, dist.hessians), tol);
    }

    /// @brief Verifies the objective, gradient and Hessian against the coordinate model at fixed hyperparameters.
    /// @details Feeds the Euclidean squared distances of the same points; the coordinate objective is the
    /// independent reference. The tolerance 1e-10 is the completion criterion; both evaluate the same closed forms,
    /// so the observed differences are rounding only (about 1e-14 on these well-conditioned systems).
    TEST(DistanceGprTest, ObjectiveGradientAndHessianMatchTheCoordinateModel)
    {
        // Spec: DES-13, DES-14, DES-21
        constexpr double tol = 1e-10;
        const Data data = make_data(14, 3, 2);
        const Tables t = euclid_tables(data);

        // Scalar slot: Constant * Rbf + White.
        {
            const cxxgpr::ScalarDistance slot;
            cxxgpr::DistanceSource train;
            train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
            ProbeRecord coord;
            ProbeRecord dist;
            auto reference = cxxgpr::Gpr<ProbeOptimizer>(coordinate_scalar_kernel(), ProbeOptimizer{&coord})
                                 .with_noise_variance(0.02)
                                 .value()
                                 .fit(data.x, data.n, data.d, data.y);
            auto model = cxxgpr::DistanceGpr<ProbeOptimizer>(distance_scalar_kernel(slot), ProbeOptimizer{&dist})
                             .with_noise_variance(0.02)
                             .value()
                             .fit(train, data.y);
            ASSERT_TRUE(reference.has_value()) << reference.error().message();
            ASSERT_TRUE(model.has_value()) << model.error().message();
            expect_probes_match(dist, coord, tol);
            EXPECT_EQ(dist.grads.size(), 3U * 4U);
        }

        // ARD slot: Constant * RbfArd with two length scales.
        {
            const auto slot = cxxgpr::ArdDistance::create(2).value();
            const std::array<double, 2> length_scales{0.9, 1.4};
            const auto kernel_dist =
                cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(0.7).value())
                * slot.rbf_ard(cxxgpr::RbfArdKernel::create(length_scales).value()).value();
            const auto kernel_coord = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(0.7).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scales).value());
            const auto t0 = table(data.x, data.n, data.x, data.n, data.d, 0);
            const auto t1 = table(data.x, data.n, data.x, data.n, data.d, 1);
            const std::array<cxxgpr::MatrixRef<double>, 2> tables{
                cxxgpr::MatrixRef<double>(t0.data(), data.n, data.n),
                cxxgpr::MatrixRef<double>(t1.data(), data.n, data.n)
            };
            cxxgpr::DistanceSource train;
            train.bind_ard(slot.slot(), tables);
            ProbeRecord coord;
            ProbeRecord dist;
            auto reference = cxxgpr::Gpr<ProbeOptimizer>(kernel_coord, ProbeOptimizer{&coord})
                                 .with_noise_variance(0.03)
                                 .value()
                                 .fit(data.x, data.n, data.d, data.y);
            auto model = cxxgpr::DistanceGpr<ProbeOptimizer>(kernel_dist, ProbeOptimizer{&dist})
                             .with_noise_variance(0.03)
                             .value()
                             .fit(train, data.y);
            ASSERT_TRUE(reference.has_value()) << reference.error().message();
            ASSERT_TRUE(model.has_value()) << model.error().message();
            expect_probes_match(dist, coord, tol);
            EXPECT_EQ(dist.grads.size(), 3U * 4U);
        }

        // WithPoints: supplied Constant * Rbf plus a coordinate Linear leaf.
        {
            const cxxgpr::ScalarDistance slot;
            const cxxgpr::DistanceKernel<cxxgpr::WithPoints> kernel_dist =
                (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.1).value())
                    * slot.rbf(cxxgpr::RbfKernel::create(0.9).value()))
                    .with_points()
                + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value());
            const auto kernel_coord = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.1).value())
                                          * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.9).value()))
                + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value());
            cxxgpr::DistanceSource train;
            train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
            ProbeRecord coord;
            ProbeRecord dist;
            auto reference = cxxgpr::Gpr<ProbeOptimizer>(kernel_coord, ProbeOptimizer{&coord})
                                 .with_noise_variance(0.03)
                                 .value()
                                 .fit(data.x, data.n, data.d, data.y);
            auto model = cxxgpr::DistanceGpr<ProbeOptimizer, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(
                kernel_dist, ProbeOptimizer{&dist}
            )
                             .with_noise_variance(0.03)
                             .value()
                             .fit(train, cxxgpr::MatrixRef<double>(data.x.data(), data.n, data.d), data.y);
            ASSERT_TRUE(reference.has_value()) << reference.error().message();
            ASSERT_TRUE(model.has_value()) << model.error().message();
            expect_probes_match(dist, coord, tol);
            EXPECT_EQ(dist.grads.size(), 3U * 4U);
        }
    }

    /// @brief Verifies covariance, samples and leave-one-out of a WithPoints model against the coordinate model.
    /// @details Uses a standardized input transform on the coordinate leaf, so the transform path of the query
    /// coordinates is compared too. Tolerances: 1e-10 for predictions, covariance and LOO (completion criterion);
    /// 1e-8 for samples, whose Cholesky factor of the covariance amplifies the rounding difference.
    TEST(DistanceGprTest, WithPointsCovarianceSamplesAndLooMatchTheCoordinateModel)
    {
        // Spec: DES-16, DES-21
        const Data data = make_data(13, 5, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
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
        cxxgpr::DistanceSource query;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        query.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.query.data(), data.m, data.m));
        const cxxgpr::MatrixRef<double> x(data.x.data(), data.n, data.d);
        const cxxgpr::MatrixRef<double> q(data.q.data(), data.m, data.d);

        // Without an input transform the supplied Euclidean tables describe the same points as the coordinates.
        const cxxgpr::transform::StandardizeTarget standardize;
        auto reference = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(kernel_coord)
                             .with_noise_variance(0.03)
                             .value()
                             .with_target_transform(standardize)
                             .factor(data.x, data.n, data.d, data.y);
        auto model =
            cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(kernel_dist)
                .with_noise_variance(0.03)
                .value()
                .with_target_transform(standardize)
                .factor(train, x, data.y);
        ASSERT_TRUE(reference.has_value());
        ASSERT_TRUE(model.has_value()) << model.error().message();

        for (const auto kind : {cxxgpr::VarianceKind::Observation, cxxgpr::VarianceKind::Latent})
        {
            const cxxgpr::PredictOptions options{.variance_kind = kind};
            auto p_ref = reference->predict(data.q, data.m, data.d, options);
            auto p_dist = model->predict(cross, q, options);
            ASSERT_TRUE(p_ref.has_value());
            ASSERT_TRUE(p_dist.has_value()) << p_dist.error().message();
            EXPECT_LT(rel_diff(p_dist->mean, p_ref->mean), 1e-10);
            EXPECT_LT(rel_diff(p_dist->variance, p_ref->variance), 1e-10);

            auto c_ref = reference->predict_covariance(data.q, data.m, data.d, options);
            auto c_dist = model->predict_covariance(cross, query, q, options);
            ASSERT_TRUE(c_ref.has_value());
            ASSERT_TRUE(c_dist.has_value()) << c_dist.error().message();
            EXPECT_LT(rel_diff(c_dist->mean, c_ref->mean), 1e-10);
            EXPECT_LT(rel_diff(c_dist->covariance, c_ref->covariance), 1e-10);

            auto l_ref = reference->loo_predict(options);
            auto l_dist = model->loo_predict(options);
            ASSERT_TRUE(l_ref.has_value());
            ASSERT_TRUE(l_dist.has_value());
            EXPECT_LT(rel_diff(l_dist->mean, l_ref->mean), 1e-10);
            EXPECT_LT(rel_diff(l_dist->variance, l_ref->variance), 1e-10);
        }
        auto s_ref = reference->sample(data.q, data.m, data.d, 4U, 3U);
        auto s_dist = model->sample(cross, query, q, 4U, 3U);
        ASSERT_TRUE(s_ref.has_value());
        ASSERT_TRUE(s_dist.has_value()) << s_dist.error().message();
        ASSERT_EQ(s_dist->size(), data.m * 4U);
        EXPECT_LT(rel_diff(*s_dist, *s_ref), 1e-8);
    }

    /// @brief Verifies the single and mixed precision policies against the coordinate model of the same policy.
    /// @details The supplied tables are rounded to float from exact double distances, while the coordinate model
    /// rounds the coordinates and forms the distances in float, so the float systems differ by a few float epsilons
    /// amplified by the conditioning (noise 1e-2, n = 30). Tolerances: 1e-3 for SinglePrecision and
    /// PromoteStorage (storage rounding), 1e-6 for ReevaluateKernel, whose weights are refined against the exact
    /// double system in both models.
    TEST(DistanceGprTest, SingleAndMixedPrecisionMatchTheCoordinateModelOfTheSamePolicy)
    {
        // Spec: DES-05, DES-21
        const Data data = make_data(30, 5, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));

        auto check = [&]<typename P>(double tol)
        {
            auto reference = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(coordinate_scalar_kernel())
                                 .with_noise_variance(1e-2)
                                 .value()
                                 .factor(data.x, data.n, data.d, data.y);
            auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, P>(distance_scalar_kernel(slot))
                             .with_noise_variance(1e-2)
                             .value()
                             .factor(train, data.y);
            ASSERT_TRUE(reference.has_value());
            ASSERT_TRUE(model.has_value()) << model.error().message();
            EXPECT_LT(rel_diff(model->alpha(), reference->alpha()), tol);
            EXPECT_NEAR(
                model->neg_log_marginal_likelihood(), reference->optimizer_result().value,
                tol * std::max(1.0, std::abs(reference->optimizer_result().value))
            );
            auto p_ref = reference->predict(data.q, data.m, data.d);
            auto p_dist = model->predict(cross);
            ASSERT_TRUE(p_ref.has_value());
            ASSERT_TRUE(p_dist.has_value()) << p_dist.error().message();
            EXPECT_LT(rel_diff(p_dist->mean, p_ref->mean), tol);
            EXPECT_LT(rel_diff(p_dist->variance, p_ref->variance), tol);
            EXPECT_EQ(model->refinement_summary().has_value(), reference->refinement_summary().has_value());
        };
        check.template operator()<cxxgpr::SinglePrecision>(1e-3);
        check.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(1e-3);
        check.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(1e-6);
    }

    /// @brief Verifies that both engines apply the same PromoteStorage acceptance rule to the same system.
    /// @details On the data of the coordinate acceptance test (n = 400, noise 1e-3) the converged iterate misses the
    /// double tolerance 10 n eps64 but meets the storage-rounding bound; both models must accept it without the
    /// double-precision fallback.
    TEST(DistanceGprTest, PromoteStorageAcceptanceMatchesTheCoordinateEngine)
    {
        // Spec: DES-05, DES-21
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
        const auto train_table = table(x, n, x, n, d, d);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(train_table.data(), n, n));
        using Promote = cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>;
        const auto kernel_coord = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.7).value());
        const auto kernel_dist =
            cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.0).value())
            * slot.rbf(cxxgpr::RbfKernel::create(0.7).value());
        auto coord = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, Promote>(kernel_coord)
                         .with_noise_variance(1e-3)
                         .value()
                         .factor(x, n, d, y);
        auto dist = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, Promote>(kernel_dist)
                        .with_noise_variance(1e-3)
                        .value()
                        .factor(train, y);
        ASSERT_TRUE(coord.has_value());
        ASSERT_TRUE(dist.has_value()) << dist.error().message();
        const auto coord_summary = coord->refinement_summary();
        const auto dist_summary = dist->refinement_summary();
        if (!coord_summary.has_value() || !dist_summary.has_value())
        {
            FAIL() << "refinement summary is missing";
        }
        EXPECT_EQ(coord_summary->status, cxxgpr::RefinementStatus::Converged);
        EXPECT_EQ(dist_summary->status, coord_summary->status);
        EXPECT_FALSE(dist_summary->used_fallback());
    }

    /// @brief Builds a fitted model and the inputs used by the validation-order tests.
    struct OrderFixture
    {
        /// @brief Dataset (n = 8, m = 3, d = 2).
        Data data = make_data(8, 3, 2);
        /// @brief Euclidean tables of the dataset.
        Tables t = euclid_tables(data);
        /// @brief Scalar slot of the kernel.
        cxxgpr::ScalarDistance slot;
        /// @brief Slot of another kernel.
        cxxgpr::ScalarDistance foreign_slot;
    };

    /// @brief Verifies the training validation order: each adjacent pair of conditions reports the earlier error.
    TEST(DistanceGprTest, TrainingValidationFollowsTheSpecifiedOrder)
    {
        // Spec: DES-02, DES-21, REQ-03
        const OrderFixture f;
        const std::size_t n = f.data.n;
        const auto trainer = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(distance_scalar_kernel(f.slot))
                                 .with_noise_variance(0.05)
                                 .value();
        std::vector<double> negative = f.t.train;
        negative[1] = -1.0;
        negative[n] = -1.0;
        cxxgpr::DistanceSource bad_values;
        bad_values.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(negative.data(), n, n));
        cxxgpr::DistanceSource foreign;
        foreign.bind_scalar(f.foreign_slot.slot(), cxxgpr::MatrixRef<double>(f.t.train.data(), n, n));
        cxxgpr::DistanceSource good;
        good.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(f.t.train.data(), n, n));
        std::vector<double> nan_y = f.data.y;
        nan_y[2] = std::numeric_limits<double>::quiet_NaN();

        // Empty targets before a foreign slot.
        EXPECT_EQ(trainer.fit(foreign, std::vector<double>{}).error().code(), cxxgpr::GprErrorCode::EmptyInput);
        // A foreign slot is exactly UnsupportedKernelOperation.
        EXPECT_EQ(trainer.fit(foreign, f.data.y).error().code(), cxxgpr::GprErrorCode::UnsupportedKernelOperation);
        // Target length against the table order before the table values.
        EXPECT_EQ(
            trainer.fit(bad_values, std::vector<double>(n + 1U, 0.0)).error().code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );
        // Non-square table before its values.
        std::vector<double> rect(n * (n - 1U), -1.0);
        cxxgpr::DistanceSource non_square;
        non_square.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(rect.data(), n, n - 1U));
        EXPECT_EQ(trainer.fit(non_square, f.data.y).error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        // Non-finite targets before the table values.
        EXPECT_EQ(trainer.fit(bad_values, nan_y).error().code(), cxxgpr::GprErrorCode::NonFiniteInput);
        // The table values alone.
        EXPECT_EQ(trainer.fit(bad_values, f.data.y).error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        EXPECT_TRUE(trainer.fit(good, f.data.y).has_value());

        // ARD table count: after the target length, before non-finite targets.
        {
            const auto ard_slot = cxxgpr::ArdDistance::create(2).value();
            const auto ard_trainer = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(
                cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(0.7).value())
                * ard_slot.rbf_ard(cxxgpr::RbfArdKernel::create(std::array{0.9, 1.4}).value()).value()
            )
                                         .with_noise_variance(0.05)
                                         .value();
            const std::array<cxxgpr::MatrixRef<double>, 1> one{cxxgpr::MatrixRef<double>(negative.data(), n, n)};
            cxxgpr::DistanceSource wrong_count;
            wrong_count.bind_ard(ard_slot.slot(), one);
            EXPECT_EQ(
                ard_trainer.fit(wrong_count, std::vector<double>(n + 1U, 0.0)).error().code(),
                cxxgpr::GprErrorCode::LengthMismatch
            );
            EXPECT_EQ(ard_trainer.fit(wrong_count, nan_y).error().code(), cxxgpr::GprErrorCode::DimensionMismatch);
        }

        // WithPoints: empty, row count, column count, finiteness of the points, then the table values.
        {
            const auto ard_leaf = cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array{0.8, 1.2}).value());
            const cxxgpr::DistanceKernel<cxxgpr::WithPoints> kernel =
                distance_scalar_kernel(f.slot).with_points() + ard_leaf;
            const auto wp =
                cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(kernel)
                    .with_noise_variance(0.05)
                    .value();
            std::vector<double> nan_x = f.data.x;
            nan_x[3] = std::numeric_limits<double>::infinity();
            const std::vector<double> x3(n * 3U, 0.5);
            const cxxgpr::MatrixRef<double> x(f.data.x.data(), n, 2U);
            const cxxgpr::MatrixRef<double> x_nan(nan_x.data(), n, 2U);
            const cxxgpr::MatrixRef<double> x_short(nan_x.data(), n - 1U, 2U);
            const cxxgpr::MatrixRef<double> x_wide(x3.data(), n, 3U);
            EXPECT_EQ(
                wp.fit(foreign, cxxgpr::MatrixRef<double>(), f.data.y).error().code(), cxxgpr::GprErrorCode::EmptyInput
            );
            EXPECT_EQ(wp.fit(bad_values, x_short, f.data.y).error().code(), cxxgpr::GprErrorCode::LengthMismatch);
            EXPECT_EQ(wp.fit(bad_values, x_wide, f.data.y).error().code(), cxxgpr::GprErrorCode::DimensionMismatch);
            EXPECT_EQ(wp.fit(bad_values, x_nan, f.data.y).error().code(), cxxgpr::GprErrorCode::NonFiniteInput);
            EXPECT_EQ(wp.fit(bad_values, x, f.data.y).error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_TRUE(wp.fit(good, x, f.data.y).has_value());
        }
    }

    /// @brief Verifies the query validation order and that failures leave the fitted model bitwise unchanged.
    TEST(DistanceGprTest, QueryValidationFollowsTheSpecifiedOrderAndLeavesTheModelUnchanged)
    {
        // Spec: DES-02, DES-21, REQ-02
        const OrderFixture f;
        const std::size_t n = f.data.n;
        const std::size_t m = f.data.m;
        const auto ard_leaf = cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array{0.8, 1.2}).value());
        const cxxgpr::DistanceKernel<cxxgpr::WithPoints> kernel =
            distance_scalar_kernel(f.slot).with_points() + ard_leaf;
        const auto trainer =
            cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(kernel)
                .with_noise_variance(0.05)
                .value();
        cxxgpr::DistanceSource train;
        train.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(f.t.train.data(), n, n));
        const cxxgpr::MatrixRef<double> x(f.data.x.data(), n, 2U);
        auto model = trainer.fit(train, x, f.data.y);
        ASSERT_TRUE(model.has_value()) << model.error().message();

        cxxgpr::DistanceSource cross;
        cross.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(f.t.cross.data(), n, m));
        cxxgpr::DistanceSource query;
        query.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(f.t.query.data(), m, m));
        const cxxgpr::MatrixRef<double> q(f.data.q.data(), m, 2U);

        // Snapshot of the fitted state and of a prediction before the failures.
        const std::vector<double> alpha_before(model->alpha().begin(), model->alpha().end());
        std::vector<double> l_before(n * n);
        for (std::size_t c = 0; c < n; ++c)
        {
            for (std::size_t r = 0; r < n; ++r)
            {
                l_before[(c * n) + r] = model->l_factor()(r, c);
            }
        }
        const double nll_before = model->neg_log_marginal_likelihood();
        auto before = model->predict_covariance(cross, query, q);
        ASSERT_TRUE(before.has_value()) << before.error().message();

        std::vector<double> bad_cross = f.t.cross;
        bad_cross[1] = std::numeric_limits<double>::quiet_NaN();
        cxxgpr::DistanceSource bad_values;
        bad_values.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(bad_cross.data(), n, m));
        std::vector<double> short_cross(m * (n - 1U), -1.0);
        cxxgpr::DistanceSource short_rows;
        short_rows.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(short_cross.data(), n - 1U, m));
        cxxgpr::DistanceSource foreign;
        foreign.bind_scalar(f.foreign_slot.slot(), cxxgpr::MatrixRef<double>(f.t.cross.data(), n, m));
        std::vector<double> nan_q = f.data.q;
        nan_q[0] = std::numeric_limits<double>::quiet_NaN();
        const std::vector<double> q3(m * 3U, 0.5);
        const cxxgpr::MatrixRef<double> q_nan(nan_q.data(), m, 2U);
        const cxxgpr::MatrixRef<double> q_short(nan_q.data(), m - 1U, 2U);
        const cxxgpr::MatrixRef<double> q_wide(q3.data(), m, 3U);
        std::vector<double> bad_query = f.t.query;
        bad_query[1] = -1.0;
        cxxgpr::DistanceSource bad_query_values;
        bad_query_values.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(bad_query.data(), m, m));
        const std::vector<double> small_query(4U, -1.0);
        cxxgpr::DistanceSource wrong_order;
        wrong_order.bind_scalar(f.slot.slot(), cxxgpr::MatrixRef<double>(small_query.data(), 2U, 2U));

        cxxgpr::Prediction<double> out;
        // Empty query points before a foreign slot; the foreign slot is exactly UnsupportedKernelOperation.
        auto r1 = model->predict_into(foreign, cxxgpr::MatrixRef<double>(), out);
        ASSERT_FALSE(r1.has_value());
        EXPECT_EQ(r1.error().code(), cxxgpr::GprErrorCode::EmptyInput);
        auto r2 = model->predict_into(foreign, q, out);
        ASSERT_FALSE(r2.has_value());
        EXPECT_EQ(r2.error().code(), cxxgpr::GprErrorCode::UnsupportedKernelOperation);
        // Cross rows before the cross values.
        auto r3 = model->predict_into(short_rows, q, out);
        ASSERT_FALSE(r3.has_value());
        EXPECT_EQ(r3.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        // Query point rows before their finiteness and before the table values.
        auto r4 = model->predict_into(bad_values, q_short, out);
        ASSERT_FALSE(r4.has_value());
        EXPECT_EQ(r4.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        // Query point columns before their finiteness.
        auto r5 = model->predict_into(bad_values, q_wide, out);
        ASSERT_FALSE(r5.has_value());
        EXPECT_EQ(r5.error().code(), cxxgpr::GprErrorCode::DimensionMismatch);
        // Non-finite query points before the table values.
        auto r6 = model->predict_into(bad_values, q_nan, out);
        ASSERT_FALSE(r6.has_value());
        EXPECT_EQ(r6.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);
        auto r7 = model->predict_into(bad_values, q, out);
        ASSERT_FALSE(r7.has_value());
        EXPECT_EQ(r7.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        // Covariance: the query-to-query order before the cross values; cross values before query values.
        auto c1 = model->predict_covariance(bad_values, wrong_order, q);
        ASSERT_FALSE(c1.has_value());
        EXPECT_EQ(c1.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        auto c2 = model->predict_covariance(bad_values, bad_query_values, q);
        ASSERT_FALSE(c2.has_value());
        EXPECT_EQ(c2.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        EXPECT_NE(c2.error().message().find("finite"), std::string::npos) << c2.error().message();
        auto c3 = model->predict_covariance(cross, bad_query_values, q);
        ASSERT_FALSE(c3.has_value());
        EXPECT_EQ(c3.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        auto s1 = model->sample(cross, wrong_order, q, 2U, 1U);
        ASSERT_FALSE(s1.has_value());
        EXPECT_EQ(s1.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        // The fitted state and the results are bitwise unchanged after every failure above.
        EXPECT_TRUE(bits_equal(model->alpha(), alpha_before));
        std::vector<double> l_after(n * n);
        for (std::size_t c = 0; c < n; ++c)
        {
            for (std::size_t r = 0; r < n; ++r)
            {
                l_after[(c * n) + r] = model->l_factor()(r, c);
            }
        }
        EXPECT_TRUE(bits_equal(l_after, l_before));
        EXPECT_EQ(
            std::bit_cast<std::uint64_t>(model->neg_log_marginal_likelihood()), std::bit_cast<std::uint64_t>(nll_before)
        );
        auto after = model->predict_covariance(cross, query, q);
        ASSERT_TRUE(after.has_value());
        EXPECT_TRUE(bits_equal(after->mean, before->mean));
        EXPECT_TRUE(bits_equal(after->covariance, before->covariance));
        // The trainer is unchanged too: refitting gives the same model bit for bit.
        auto again = trainer.fit(train, x, f.data.y);
        ASSERT_TRUE(again.has_value());
        EXPECT_TRUE(bits_equal(again->alpha(), alpha_before));
    }

    /// @brief Verifies that predict_into allocates nothing after warm-up, including a smaller batch after a larger.
    /// @details Covers every precision policy, scalar and ARD slots, and a WithPoints model with a standardized
    /// input transform. After a warm-up call with m = 6, calls with m = 6 and m = 2 must not allocate, and the
    /// small batch must equal (bitwise) the prediction of a fresh copy that never saw the large batch.
    TEST(DistanceGprTest, PredictIntoAllocatesNothingAfterWarmUp)
    {
        // Spec: REQ-05, REQ-11, DES-17, DES-21
        const Data data = make_data(12, 6, 2);
        const Tables t = euclid_tables(data);
        constexpr std::size_t m_small = 2;
        // The first m_small query columns of the cross table, as a separate contiguous table.
        const std::vector<double> cross_small_table(
            t.cross.begin(), t.cross.begin() + static_cast<std::ptrdiff_t>(data.n * m_small)
        );
        const std::vector<double> q_small({data.q[0], data.q[1], data.q[data.m], data.q[data.m + 1U]});

        auto audit = [&]<typename P, typename C, typename K>(const char* label, const K& kernel, auto bind)
        {
            SCOPED_TRACE(label);
            cxxgpr::DistanceSource train;
            cxxgpr::DistanceSource cross;
            cxxgpr::DistanceSource cross_small;
            bind(train, cross, cross_small);
            const auto trainer = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, P, C>(kernel).with_noise_variance(0.05);
            ASSERT_TRUE(trainer.has_value());
            using Fitted = cxxgpr::FittedDistanceGpr<cxxgpr::optimizer::Fixed, P, C>;
            std::optional<Fitted> model;
            std::optional<Fitted> fresh;
            cxxgpr::Prediction<typename P::Refine> out;
            cxxgpr::Prediction<typename P::Refine> fresh_out;
            std::size_t delta_same = 0;
            std::size_t delta_small = 0;
            if constexpr (std::same_as<C, cxxgpr::WithPoints>)
            {
                const cxxgpr::transform::StandardizeInput standardize;
                const auto wp = trainer->with_input_transform(standardize);
                auto fitted = wp.factor(train, cxxgpr::MatrixRef<double>(data.x.data(), data.n, data.d), data.y);
                ASSERT_TRUE(fitted.has_value()) << fitted.error().message();
                model.emplace(*fitted);
                fresh.emplace(*fitted);
                const cxxgpr::MatrixRef<double> q(data.q.data(), data.m, data.d);
                const cxxgpr::MatrixRef<double> qs(q_small.data(), m_small, data.d);
                ASSERT_TRUE(model->predict_into(cross, q, out).has_value());
                {
                    const cxxgpr::test_support::NoAllocationScope scope;
                    const auto r = model->predict_into(cross, q, out);
                    delta_same = scope.delta_allocations();
                    ASSERT_TRUE(r.has_value());
                }
                {
                    const cxxgpr::test_support::NoAllocationScope scope;
                    const auto r = model->predict_into(cross_small, qs, out);
                    delta_small = scope.delta_allocations();
                    ASSERT_TRUE(r.has_value());
                }
                ASSERT_TRUE(fresh->predict_into(cross_small, qs, fresh_out).has_value());
            }
            else
            {
                auto fitted = trainer->factor(train, data.y);
                ASSERT_TRUE(fitted.has_value()) << fitted.error().message();
                model.emplace(*fitted);
                fresh.emplace(*fitted);
                ASSERT_TRUE(model->predict_into(cross, out).has_value());
                {
                    const cxxgpr::test_support::NoAllocationScope scope;
                    const auto r = model->predict_into(cross, out);
                    delta_same = scope.delta_allocations();
                    ASSERT_TRUE(r.has_value());
                }
                {
                    const cxxgpr::test_support::NoAllocationScope scope;
                    const auto r = model->predict_into(cross_small, out);
                    delta_small = scope.delta_allocations();
                    ASSERT_TRUE(r.has_value());
                }
                ASSERT_TRUE(fresh->predict_into(cross_small, fresh_out).has_value());
            }
            EXPECT_EQ(delta_same, 0U);
            EXPECT_EQ(delta_small, 0U);
            EXPECT_EQ(out.mean.size(), m_small);
            EXPECT_TRUE(bits_equal(out.mean, fresh_out.mean));
            EXPECT_TRUE(bits_equal(out.variance, fresh_out.variance));
        };

        const cxxgpr::ScalarDistance slot;
        auto bind_scalar =
            [&](cxxgpr::DistanceSource& train, cxxgpr::DistanceSource& cross, cxxgpr::DistanceSource& cross_small)
        {
            train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
            cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
            cross_small.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(cross_small_table.data(), data.n, m_small));
        };
        const auto scalar_kernel = distance_scalar_kernel(slot);
        audit.template operator()<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(
            "double scalar", scalar_kernel, bind_scalar
        );
        audit.template operator()<cxxgpr::SinglePrecision, cxxgpr::DistanceOnly>(
            "single scalar", scalar_kernel, bind_scalar
        );
        audit.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>, cxxgpr::DistanceOnly>(
            "promote scalar", scalar_kernel, bind_scalar
        );
        audit.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>, cxxgpr::DistanceOnly>(
            "reevaluate scalar", scalar_kernel, bind_scalar
        );

        // ARD slot.
        const auto ard_slot = cxxgpr::ArdDistance::create(2).value();
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
        const std::array<cxxgpr::MatrixRef<double>, 2> cross_small_tables{
            cxxgpr::MatrixRef<double>(c0.data(), data.n, m_small), cxxgpr::MatrixRef<double>(c1.data(), data.n, m_small)
        };
        const auto ard_kernel =
            cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(0.7).value())
            * ard_slot.rbf_ard(cxxgpr::RbfArdKernel::create(std::array{0.9, 1.4}).value()).value();
        auto bind_ard =
            [&](cxxgpr::DistanceSource& train, cxxgpr::DistanceSource& cross, cxxgpr::DistanceSource& cross_small)
        {
            train.bind_ard(ard_slot.slot(), train_tables);
            cross.bind_ard(ard_slot.slot(), cross_tables);
            cross_small.bind_ard(ard_slot.slot(), cross_small_tables);
        };
        audit.template operator()<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>("double ard", ard_kernel, bind_ard);
        audit.template operator()<cxxgpr::SinglePrecision, cxxgpr::DistanceOnly>("single ard", ard_kernel, bind_ard);

        // WithPoints with a coordinate ARD leaf and a standardized input transform.
        const cxxgpr::DistanceKernel<cxxgpr::WithPoints> wp_kernel = distance_scalar_kernel(slot).with_points()
            + cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array{0.8, 1.2}).value());
        audit.template operator()<cxxgpr::DoublePrecision, cxxgpr::WithPoints>(
            "double with-points", wp_kernel, bind_scalar
        );
        audit.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>, cxxgpr::WithPoints>(
            "reevaluate with-points", wp_kernel, bind_scalar
        );
    }

    /// @brief Optimizer that counts the heap allocations of one objective-and-gradient evaluation after warm-up.
    struct AllocationProbeOptimizer
    {
        /// @brief The probe evaluates whole objectives only.
        [[maybe_unused]] static constexpr bool uses_change_indices = false;
        /// @brief Destination of the allocation count (owned by the test).
        std::size_t* allocations = nullptr;

        /// @brief Evaluates once to warm up, then counts the allocations of an evaluation at shifted parameters.
        /// @param[in,out] obj Objective under test.
        /// @param[in] initial Starting log-space parameters.
        /// @return The starting point as the optimum, or the first evaluation error.
        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable& obj, std::span<const double> initial
        ) const
        {
            std::vector<double> theta(initial.begin(), initial.end());
            std::vector<double> grad(obj.dim());
            auto warm = obj.value_and_gradient_into(theta, grad);
            if (!warm.has_value())
            {
                return std::unexpected(std::move(warm).error());
            }
            for (double& v : theta)
            {
                v += 0.1;
            }
            {
                const cxxgpr::test_support::NoAllocationScope scope;
                auto value = obj.value_and_gradient_into(theta, grad);
                *allocations = scope.delta_allocations();
                if (!value.has_value())
                {
                    return std::unexpected(std::move(value).error());
                }
            }
            return cxxgpr::optimizer::OptimizerResult{
                .params = std::vector<double>(initial.begin(), initial.end()),
                .value = *warm,
                .iterations = 0U,
                .converged = true,
            };
        }
    };

    /// @brief Verifies that an objective-and-gradient evaluation allocates no more than the coordinate model's.
    /// @details Requirement: a supplied-distance model allocates no more than the coordinate model with the same
    /// kernel, precision and size. Both objectives evaluate after warm-up without growing any workspace; the only
    /// allocations left in either are those of the shared parameter rebinding (CompiledKernel::set_params and
    /// CompiledDistanceKernel::set_params stage a copy of the leaves, the ARD leaves stage their length scales, and
    /// the cached length scales are refreshed from a fresh vector), so the counts must be equal. Covers scalar and
    /// ARD slots, every precision policy and a WithPoints kernel with a coordinate ARD leaf.
    TEST(DistanceGprTest, ObjectiveAndGradientAllocateNoMoreThanTheCoordinateModel)
    {
        // Spec: REQ-05, REQ-11, DES-13, DES-21
        const Data data = make_data(12, 2, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        cxxgpr::DistanceSource train;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));

        auto check = [&]<typename P, typename C, typename K>(
                         const char* label, const K& kernel, const cxxgpr::KernelSpec& coord_kernel, const auto& source
                     )
        {
            SCOPED_TRACE(label);
            std::size_t dist_allocations = 1000;
            std::size_t coord_allocations = 2000;
            const auto trainer =
                cxxgpr::DistanceGpr<AllocationProbeOptimizer, P, C>(kernel, AllocationProbeOptimizer{&dist_allocations})
                    .with_noise_variance(0.05);
            ASSERT_TRUE(trainer.has_value());
            if constexpr (std::same_as<C, cxxgpr::WithPoints>)
            {
                auto fitted = trainer->fit(source, cxxgpr::MatrixRef<double>(data.x.data(), data.n, data.d), data.y);
                ASSERT_TRUE(fitted.has_value()) << fitted.error().message();
            }
            else
            {
                auto fitted = trainer->fit(source, data.y);
                ASSERT_TRUE(fitted.has_value()) << fitted.error().message();
            }
            auto coord =
                cxxgpr::Gpr<AllocationProbeOptimizer, P>(coord_kernel, AllocationProbeOptimizer{&coord_allocations})
                    .with_noise_variance(0.05);
            ASSERT_TRUE(coord.has_value());
            auto coord_fitted = coord->fit(data.x, data.n, data.d, data.y);
            ASSERT_TRUE(coord_fitted.has_value()) << coord_fitted.error().message();
            EXPECT_LE(dist_allocations, coord_allocations);
            EXPECT_EQ(dist_allocations, coord_allocations);
        };
        const auto scalar_kernel = distance_scalar_kernel(slot);
        const auto scalar_coord = coordinate_scalar_kernel();
        check.template operator()<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(
            "double", scalar_kernel, scalar_coord, train
        );
        check.template operator()<cxxgpr::SinglePrecision, cxxgpr::DistanceOnly>(
            "single", scalar_kernel, scalar_coord, train
        );
        check.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>, cxxgpr::DistanceOnly>(
            "promote", scalar_kernel, scalar_coord, train
        );
        check.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>, cxxgpr::DistanceOnly>(
            "reevaluate", scalar_kernel, scalar_coord, train
        );

        const auto ard_slot = cxxgpr::ArdDistance::create(2).value();
        const auto t0 = table(data.x, data.n, data.x, data.n, data.d, 0);
        const auto t1 = table(data.x, data.n, data.x, data.n, data.d, 1);
        const std::array<cxxgpr::MatrixRef<double>, 2> tables{
            cxxgpr::MatrixRef<double>(t0.data(), data.n, data.n), cxxgpr::MatrixRef<double>(t1.data(), data.n, data.n)
        };
        cxxgpr::DistanceSource ard_train;
        ard_train.bind_ard(ard_slot.slot(), tables);
        const std::array length_scales{0.9, 1.4};
        const auto ard_kernel =
            cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(0.7).value())
            * ard_slot.rbf_ard(cxxgpr::RbfArdKernel::create(length_scales).value()).value();
        const auto ard_coord = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(0.7).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scales).value());
        check.template operator()<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(
            "double ard", ard_kernel, ard_coord, ard_train
        );

        const std::array coord_scales{0.8, 1.2};
        const cxxgpr::DistanceKernel<cxxgpr::WithPoints> wp_kernel = distance_scalar_kernel(slot).with_points()
            + cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(coord_scales).value());
        const auto wp_coord =
            coordinate_scalar_kernel() + cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(coord_scales).value());
        check.template operator()<cxxgpr::DoublePrecision, cxxgpr::WithPoints>(
            "double with-points", wp_kernel, wp_coord, train
        );
        check.template operator()<cxxgpr::SinglePrecision, cxxgpr::WithPoints>(
            "single with-points", wp_kernel, wp_coord, train
        );
    }

    /// @brief Verifies that a non-finite latent variance is reported instead of being clamped to zero.
    /// @details Query coordinates of 1e200 make the Linear prior variance k(x*, x*) and ||L^-1 k*||^2 both overflow
    /// to +Inf, so their difference is NaN. Clamping max(0, NaN) would return 0 and report only the noise; the model
    /// must return NonFiniteKernelValue (the coordinate engine applies the same rule) and stay usable.
    TEST(DistanceGprTest, NonFiniteLatentVarianceIsReportedNotClamped)
    {
        // Spec: DES-02, DES-16, DES-21
        const Data data = make_data(8, 2, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance slot;
        const cxxgpr::DistanceKernel<cxxgpr::WithPoints> kernel =
            distance_scalar_kernel(slot).with_points() + cxxgpr::KernelSpec(cxxgpr::LinearKernel::create(0.5).value());
        cxxgpr::DistanceSource train;
        cxxgpr::DistanceSource cross;
        train.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        cross.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(t.cross.data(), data.n, data.m));
        auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(kernel)
                         .with_noise_variance(0.05)
                         .value()
                         .factor(train, cxxgpr::MatrixRef<double>(data.x.data(), data.n, data.d), data.y);
        ASSERT_TRUE(model.has_value()) << model.error().message();
        const std::vector<double> huge(data.m * data.d, 1e200);
        for (const auto kind : {cxxgpr::VarianceKind::Observation, cxxgpr::VarianceKind::Latent})
        {
            auto bad =
                model->predict(cross, cxxgpr::MatrixRef<double>(huge.data(), data.m, data.d), {.variance_kind = kind});
            ASSERT_FALSE(bad.has_value()) << "variance " << (bad.has_value() ? bad->variance[0] : 0.0);
            EXPECT_EQ(bad.error().code(), cxxgpr::GprErrorCode::NonFiniteKernelValue);
        }
        EXPECT_TRUE(model->predict(cross, cxxgpr::MatrixRef<double>(data.q.data(), data.m, data.d)).has_value());
    }

    /// @brief Custom optimizer auditing heap allocations on the second MLL+grad call.
    struct MllGradAuditOptimizer
    {
        /// @brief Opts out of incremental coordinate updates.
        [[maybe_unused]] static constexpr bool uses_change_indices = false;
        /// @brief Shared counter recording allocations observed on the second evaluation.
        std::shared_ptr<std::size_t> second_eval_allocs = std::make_shared<std::size_t>(999U);

        /// @brief Evaluates MLL+grad twice and records allocations on the second evaluation.
        /// @param[in,out] obj Objective supplied by DistanceGpr::fit().
        /// @param[in] init Initial log-space parameters.
        /// @return OptimizerResult on success, or GprError.
        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable& obj, std::span<const double> init
        ) const
        {
            std::vector<double> grad(obj.dim(), 0.0);
            std::vector<double> p1(init.begin(), init.end());
            auto v1 = obj.value_and_gradient_into(p1, grad);
            if (!v1.has_value())
            {
                return std::unexpected(std::move(v1).error());
            }
            std::vector<double> p2(init.begin(), init.end());
            for (double& x : p2)
            {
                x += 0.1;
            }
            {
                const cxxgpr::test_support::NoAllocationScope audit;
                auto v2 = obj.value_and_gradient_into(p2, grad);
                *second_eval_allocs = audit.delta_allocations();
                if (!v2.has_value())
                {
                    return std::unexpected(std::move(v2).error());
                }
                return cxxgpr::optimizer::OptimizerResult{
                    .params = std::move(p2),
                    .value = *v2,
                    .iterations = 2U,
                    .converged = true,
                };
            }
        }
    };

    /// @brief Verifies DistanceGpr MLL+grad evaluations (including set_params) perform zero heap allocations after
    /// warmup.
    TEST(DistanceGprTest, DistanceModelMllGradZeroAllocAfterWarmup)
    {
        // Spec: REQ-05, REQ-11, DES-21

        const Data data = make_data(12, 2, 2);
        const Tables t = euclid_tables(data);
        const cxxgpr::ScalarDistance s_slot;
        const auto a_slot = cxxgpr::ArdDistance::create(data.d).value();

        cxxgpr::DistanceSource train_s;
        train_s.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));

        const auto ard_t0 = table(data.x, data.n, data.x, data.n, data.d, 0);
        const auto ard_t1 = table(data.x, data.n, data.x, data.n, data.d, 1);
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_refs{
            cxxgpr::MatrixRef<double>(ard_t0.data(), data.n, data.n),
            cxxgpr::MatrixRef<double>(ard_t1.data(), data.n, data.n),
        };
        cxxgpr::DistanceSource train_a;
        train_a.bind_ard(a_slot.slot(), ard_refs);

        cxxgpr::DistanceSource train_comp;
        train_comp.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(t.train.data(), data.n, data.n));
        train_comp.bind_ard(a_slot.slot(), ard_refs);

        const auto k_scalar =
            cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(s_slot.rbf(cxxgpr::RbfKernel::create(0.8).value()));
        const auto k_ard = a_slot.rbf_ard(cxxgpr::RbfArdKernel::create(std::array{0.7, 1.2}).value()).value();
        const auto k_comp = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.1).value())
                * a_slot
                      .matern_ard(
                          cxxgpr::MaternArdKernel::create(std::array{0.6, 1.1}, cxxgpr::MaternNu::FiveHalves).value()
                      )
                      .value()
            + s_slot.rational_quadratic(cxxgpr::RationalQuadraticKernel::create(0.9, 1.5).value());

        auto run_case = [&]<typename P>(
                            const cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>& k, const cxxgpr::DistanceSource& src,
                            const char* label
                        )
        {
            SCOPED_TRACE(label);
            const MllGradAuditOptimizer opt;
            auto fit = cxxgpr::DistanceGpr<MllGradAuditOptimizer, P, cxxgpr::DistanceOnly>(k, opt)
                           .with_noise_variance(0.05)
                           .value()
                           .fit(src, data.y);
            ASSERT_TRUE(fit.has_value()) << fit.error().message();
            EXPECT_EQ(*opt.second_eval_allocs, 0U);
        };

        for (const auto& [name, k_ptr, s_ptr] : std::array<
                 std::tuple<
                     const char*, const cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>*, const cxxgpr::DistanceSource*>,
                 3>{
                 {{"scalar", &k_scalar, &train_s}, {"ard", &k_ard, &train_a}, {"composite", &k_comp, &train_comp}}
        })
        {
            SCOPED_TRACE(name);
            run_case.template operator()<cxxgpr::DoublePrecision>(*k_ptr, *s_ptr, "DoublePrecision");
            run_case.template operator()<cxxgpr::SinglePrecision>(*k_ptr, *s_ptr, "SinglePrecision");
            run_case.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(
                *k_ptr, *s_ptr, "MixedPromote"
            );
            run_case.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(
                *k_ptr, *s_ptr, "MixedReevaluate"
            );
        }
    }

}  // namespace
