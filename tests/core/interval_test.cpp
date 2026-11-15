// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/interval.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

// Spec: REQ-02, DES-01, DES-02, DES-04
namespace {

TEST(IntervalTest, DefaultPositiveMatchesSpecification) {
    const auto iv = cxxgpr::Interval::default_positive();
    EXPECT_DOUBLE_EQ(iv.lo(), 1e-5);
    EXPECT_DOUBLE_EQ(iv.hi(), 1e5);
    EXPECT_NEAR(iv.width(), 1e5 - 1e-5, 1e-10);
    EXPECT_TRUE(iv.contains(1.0));
    EXPECT_FALSE(iv.contains(iv.lo()));
    EXPECT_FALSE(iv.contains(iv.hi()));
    EXPECT_FALSE(iv.contains(1e-6));
    EXPECT_FALSE(iv.contains(1e6));
    EXPECT_FALSE(iv.contains(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_FALSE(iv.contains(std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(iv.contains(-std::numeric_limits<double>::infinity()));
}

TEST(IntervalTest, CreateAcceptsValidOpenIntervalsAndRejectsInvalidBounds) {
    const auto negative_lo = cxxgpr::Interval::create(-2.0, 3.0);
    ASSERT_TRUE(negative_lo.has_value());
    EXPECT_DOUBLE_EQ(negative_lo->lo(), -2.0);
    EXPECT_DOUBLE_EQ(negative_lo->hi(), 3.0);
    EXPECT_TRUE(negative_lo->contains(0.0));
    EXPECT_FALSE(negative_lo->contains(-2.0));
    EXPECT_FALSE(negative_lo->contains(3.0));

    const auto inverted = cxxgpr::Interval::create(1.0, 0.0);
    ASSERT_FALSE(inverted.has_value());
    EXPECT_EQ(inverted.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    EXPECT_EQ(inverted.error().interval_error_kind(), cxxgpr::IntervalErrorKind::InvalidBounds);
    EXPECT_EQ(inverted.error().interval_lo(), 1.0);
    EXPECT_EQ(inverted.error().interval_hi(), 0.0);

    const auto degenerate = cxxgpr::Interval::create(1.0, 1.0);
    ASSERT_FALSE(degenerate.has_value());
    EXPECT_EQ(degenerate.error().code(), cxxgpr::GprErrorCode::InvalidInterval);

    const auto nan_lo = cxxgpr::Interval::create(std::numeric_limits<double>::quiet_NaN(), 1.0);
    ASSERT_FALSE(nan_lo.has_value());
    EXPECT_EQ(nan_lo.error().code(), cxxgpr::GprErrorCode::InvalidInterval);

    const auto inf_hi = cxxgpr::Interval::create(0.0, std::numeric_limits<double>::infinity());
    ASSERT_FALSE(inf_hi.has_value());
    EXPECT_EQ(inf_hi.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
}

TEST(IntervalTest, PositiveRequiresStrictlyPositiveLowerBound) {
    const auto valid = cxxgpr::Interval::positive(1e-3, 10.0);
    ASSERT_TRUE(valid.has_value());
    EXPECT_DOUBLE_EQ(valid->lo(), 1e-3);
    EXPECT_DOUBLE_EQ(valid->hi(), 10.0);

    for (const double bad_lo : {0.0, -0.0, -1e-6, -1.0}) {
        const auto err = cxxgpr::Interval::positive(bad_lo, 1.0);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
        EXPECT_EQ(err.error().interval_error_kind(), cxxgpr::IntervalErrorKind::InvalidBounds);
    }

    const auto inverted = cxxgpr::Interval::positive(5.0, 2.0);
    ASSERT_FALSE(inverted.has_value());
    EXPECT_EQ(inverted.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
}

TEST(BoundedParamTest, EnforcesOpenIntervalOnCreationAndUpdates) {
    const auto iv = cxxgpr::Interval::create(1.0, 3.0);
    ASSERT_TRUE(iv.has_value());

    const auto param = cxxgpr::BoundedParam::default_positive(1.0);
    ASSERT_TRUE(param.has_value());
    EXPECT_DOUBLE_EQ(param->value(), 1.0);
    EXPECT_NEAR(param->ln(), 0.0, 1e-15);
    EXPECT_EQ(param->interval(), cxxgpr::Interval::default_positive());

    const auto moved = param->with_value(2.0);
    ASSERT_TRUE(moved.has_value());
    EXPECT_DOUBLE_EQ(moved->value(), 2.0);
    EXPECT_NEAR(moved->ln(), std::log(2.0), 1e-15);

    const auto narrowed = moved->with_interval(*iv);
    ASSERT_TRUE(narrowed.has_value());
    EXPECT_DOUBLE_EQ(narrowed->interval().lo(), 1.0);
    EXPECT_DOUBLE_EQ(narrowed->interval().hi(), 3.0);

    const auto out_of_range = narrowed->with_value(3.0);
    ASSERT_FALSE(out_of_range.has_value());
    EXPECT_EQ(out_of_range.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    EXPECT_EQ(out_of_range.error().interval_error_kind(), cxxgpr::IntervalErrorKind::OutOfRange);
    EXPECT_EQ(out_of_range.error().interval_value(), 3.0);
    EXPECT_EQ(out_of_range.error().interval_lo(), 1.0);
    EXPECT_EQ(out_of_range.error().interval_hi(), 3.0);

    const auto outside_narrow = cxxgpr::BoundedParam::create(1.0, *iv);
    ASSERT_FALSE(outside_narrow.has_value());
    EXPECT_EQ(outside_narrow.error().code(), cxxgpr::GprErrorCode::InvalidInterval);

    const auto nan_param =
        cxxgpr::BoundedParam::default_positive(std::numeric_limits<double>::quiet_NaN());
    ASSERT_FALSE(nan_param.has_value());
    EXPECT_EQ(nan_param.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
}

} // namespace
