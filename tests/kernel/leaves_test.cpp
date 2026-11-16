// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, DES-07

#include "cxxgpr/kernel/leaves.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

TEST(KernelLeavesTest, ConstructsAllTenBuiltInLeavesWithDefaultPositiveBounds) {
    const auto rbf = cxxgpr::RbfKernel::create(1.5);
    const auto rbf_ard = cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 2.0});
    const auto matern = cxxgpr::MaternKernel::create(1.2, cxxgpr::MaternNu::ThreeHalves);
    const auto matern_ard = cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.8, 1.4},
                                                            cxxgpr::MaternNu::FiveHalves);
    const auto periodic = cxxgpr::PeriodicKernel::create(1.0, 2.5);
    const auto rq = cxxgpr::RationalQuadraticKernel::create(1.0, 1.5);
    const auto rq_ard =
        cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.0, 2.0}, 2.0);
    const auto constant = cxxgpr::ConstantKernel::create(3.0);
    const auto linear = cxxgpr::LinearKernel::create(0.5);
    const auto white = cxxgpr::WhiteKernel::create(0.01);

    ASSERT_TRUE(rbf.has_value());
    ASSERT_TRUE(rbf_ard.has_value());
    ASSERT_TRUE(matern.has_value());
    ASSERT_TRUE(matern_ard.has_value());
    ASSERT_TRUE(periodic.has_value());
    ASSERT_TRUE(rq.has_value());
    ASSERT_TRUE(rq_ard.has_value());
    ASSERT_TRUE(constant.has_value());
    ASSERT_TRUE(linear.has_value());
    ASSERT_TRUE(white.has_value());

    EXPECT_EQ(rbf->num_params(), 1U);
    EXPECT_EQ(rbf_ard->num_params(), 2U);
    EXPECT_EQ(matern->num_params(), 1U);
    EXPECT_EQ(matern_ard->num_params(), 2U);
    EXPECT_EQ(periodic->num_params(), 2U);
    EXPECT_EQ(rq->num_params(), 2U);
    EXPECT_EQ(rq_ard->num_params(), 3U);
    EXPECT_EQ(constant->num_params(), 1U);
    EXPECT_EQ(linear->num_params(), 1U);
    EXPECT_EQ(white->num_params(), 1U);

    EXPECT_EQ(rbf->interval(), cxxgpr::Interval::default_positive());
    EXPECT_DOUBLE_EQ(rbf->lengthscale(), 1.5);
    EXPECT_EQ(matern->nu(), cxxgpr::MaternNu::ThreeHalves);
    EXPECT_EQ(matern_ard->nu(), cxxgpr::MaternNu::FiveHalves);
}

TEST(KernelLeavesTest, RejectsInvalidValuesIntervalsAndEmptyArd) {
    EXPECT_EQ(cxxgpr::RbfKernel::create(0.0).error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    EXPECT_EQ(cxxgpr::RbfKernel::create(-1.0).error().code(),
              cxxgpr::GprErrorCode::InvalidInterval);
    EXPECT_EQ(cxxgpr::RbfKernel::create(std::numeric_limits<double>::quiet_NaN()).error().code(),
              cxxgpr::GprErrorCode::InvalidInterval);

    const auto non_positive_iv = cxxgpr::Interval::create(-1.0, 5.0);
    ASSERT_TRUE(non_positive_iv.has_value());
    EXPECT_EQ(cxxgpr::ConstantKernel::create(1.0, *non_positive_iv).error().code(),
              cxxgpr::GprErrorCode::InvalidInterval);

    EXPECT_EQ(cxxgpr::RbfArdKernel::create(std::span<const double>{}).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::MaternArdKernel::create(std::span<const double>{}, cxxgpr::MaternNu::OneHalf)
                  .error()
                  .code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(
        cxxgpr::RationalQuadraticArdKernel::create(std::span<const double>{}, 1.0).error().code(),
        cxxgpr::GprErrorCode::EmptyInput);
}

TEST(KernelLeavesTest, SetParamsIsAtomicAndLogsRoundTrip) {
    auto rq_ard = cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.0, 2.0}, 1.5);
    ASSERT_TRUE(rq_ard.has_value());

    std::array<double, 3> theta{};
    ASSERT_TRUE(rq_ard->get_params(theta).has_value());
    EXPECT_DOUBLE_EQ(theta[0], std::log(1.0));
    EXPECT_DOUBLE_EQ(theta[1], std::log(2.0));
    EXPECT_DOUBLE_EQ(theta[2], std::log(1.5));

    // Third parameter is outside (1e-5, 1e5); first two must remain untouched.
    const std::array<double, 3> bad_theta{std::log(3.0), std::log(4.0), std::log(1e6)};
    const auto fail = rq_ard->set_params(bad_theta);
    ASSERT_FALSE(fail.has_value());
    EXPECT_EQ(fail.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    EXPECT_EQ(rq_ard->lengthscales(), (std::vector<double>{1.0, 2.0}));
    EXPECT_DOUBLE_EQ(rq_ard->alpha(), 1.5);

    const std::array<double, 3> good_theta{std::log(3.0), std::log(4.0), std::log(2.5)};
    ASSERT_TRUE(rq_ard->set_params(good_theta).has_value());
    EXPECT_NEAR(rq_ard->lengthscales()[0], 3.0, 1e-12);
    EXPECT_NEAR(rq_ard->lengthscales()[1], 4.0, 1e-12);
    EXPECT_NEAR(rq_ard->alpha(), 2.5, 1e-12);
}

} // namespace
