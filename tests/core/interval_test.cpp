// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-02, DES-01, DES-02, DES-04
/// @file interval_test.cpp
/// @brief Unit tests for open Interval bounds and BoundedParam hyperparameter wrappers.
/// @details Verifies default positive bounds, strict open-interval membership, factory rejection of
/// degenerate, inverted, non-positive, or non-finite bounds, and log-space updates on BoundedParam.

#include "cxxgpr/core/interval.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{

    /// @brief Verifies Interval::default_positive() matches (1e-5, 1e5) and open-endpoint rules.
    /// @details Confirms exact endpoint exclusivity at lo/hi and rejects out-of-range and non-finite inputs.
    TEST(IntervalTest, DefaultPositiveMatchesSpecification)
    {
        const auto iv = cxxgpr::Interval::default_positive();
        EXPECT_DOUBLE_EQ(iv.lo(), 1e-5);
        EXPECT_DOUBLE_EQ(iv.hi(), 1e5);
        EXPECT_NEAR(iv.width(), 1e5 - 1e-5, 1e-10);

        // Open interval (1e-5, 1e5) includes interior points and excludes endpoints and non-finites.
        EXPECT_TRUE(iv.contains(1.0));
        EXPECT_FALSE(iv.contains(iv.lo()));
        EXPECT_FALSE(iv.contains(iv.hi()));
        EXPECT_FALSE(iv.contains(1e-6));
        EXPECT_FALSE(iv.contains(1e6));
        EXPECT_FALSE(iv.contains(std::numeric_limits<double>::quiet_NaN()));
        EXPECT_FALSE(iv.contains(std::numeric_limits<double>::infinity()));
        EXPECT_FALSE(iv.contains(-std::numeric_limits<double>::infinity()));
    }

    /// @brief Verifies Interval::create() accepts general open intervals and rejects invalid bounds.
    /// @details Allows negative lower bounds while rejecting inverted (lo > hi), degenerate (lo == hi),
    /// NaN, and Inf bounds with IntervalErrorKind::InvalidBounds.
    TEST(IntervalTest, CreateAcceptsValidOpenIntervalsAndRejectsInvalidBounds)
    {
        const auto negative_lo = cxxgpr::Interval::create(-2.0, 3.0);
        ASSERT_TRUE(negative_lo.has_value());
        EXPECT_DOUBLE_EQ(negative_lo->lo(), -2.0);
        EXPECT_DOUBLE_EQ(negative_lo->hi(), 3.0);
        EXPECT_TRUE(negative_lo->contains(0.0));
        EXPECT_FALSE(negative_lo->contains(-2.0));
        EXPECT_FALSE(negative_lo->contains(3.0));

        // Negative control: inverted bounds record exact lo/hi metadata under InvalidBounds.
        const auto inverted = cxxgpr::Interval::create(1.0, 0.0);
        ASSERT_FALSE(inverted.has_value());
        EXPECT_EQ(inverted.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
        EXPECT_EQ(inverted.error().interval_error_kind(), cxxgpr::IntervalErrorKind::InvalidBounds);
        EXPECT_EQ(inverted.error().interval_lo(), 1.0);
        EXPECT_EQ(inverted.error().interval_hi(), 0.0);

        // Negative control: zero-width, NaN, and infinite intervals are invalid.
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

    /// @brief Verifies Interval::positive() requires a strictly positive lower bound (lo > 0).
    /// @details Accepts positive finite intervals and rejects +0.0, -0.0, negative lo, and lo > hi.
    TEST(IntervalTest, PositiveRequiresStrictlyPositiveLowerBound)
    {
        const auto valid = cxxgpr::Interval::positive(1e-3, 10.0);
        ASSERT_TRUE(valid.has_value());
        EXPECT_DOUBLE_EQ(valid->lo(), 1e-3);
        EXPECT_DOUBLE_EQ(valid->hi(), 10.0);

        // Zero (including -0.0) and negative lower bounds violate strict positivity.
        for (const double bad_lo : {0.0, -0.0, -1e-6, -1.0})
        {
            const auto err = cxxgpr::Interval::positive(bad_lo, 1.0);
            ASSERT_FALSE(err.has_value());
            EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
            EXPECT_EQ(err.error().interval_error_kind(), cxxgpr::IntervalErrorKind::InvalidBounds);
        }

        const auto inverted = cxxgpr::Interval::positive(5.0, 2.0);
        ASSERT_FALSE(inverted.has_value());
        EXPECT_EQ(inverted.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    }

    /// @brief Verifies BoundedParam enforces open-interval containment on construction and updates.
    /// @details Checks natural-log conversion via ln(), immutable value/interval updates via with_value()
    /// and with_interval(), and OutOfRange diagnostics on boundary/NaN violations.
    TEST(BoundedParamTest, EnforcesOpenIntervalOnCreationAndUpdates)
    {
        const auto iv = cxxgpr::Interval::create(1.0, 3.0);
        ASSERT_TRUE(iv.has_value());

        const auto param = cxxgpr::BoundedParam::default_positive(1.0);
        ASSERT_TRUE(param.has_value());
        EXPECT_DOUBLE_EQ(param->value(), 1.0);
        EXPECT_NEAR(param->ln(), 0.0, 1e-15);
        EXPECT_EQ(param->interval(), cxxgpr::Interval::default_positive());

        // Move value inside default bounds, then narrow the interval around 2.0 in (1.0, 3.0).
        const auto moved = param->with_value(2.0);
        ASSERT_TRUE(moved.has_value());
        EXPECT_DOUBLE_EQ(moved->value(), 2.0);
        EXPECT_NEAR(moved->ln(), std::log(2.0), 1e-15);

        const auto narrowed = moved->with_interval(*iv);
        ASSERT_TRUE(narrowed.has_value());
        EXPECT_DOUBLE_EQ(narrowed->interval().lo(), 1.0);
        EXPECT_DOUBLE_EQ(narrowed->interval().hi(), 3.0);

        // Open upper endpoint 3.0 must fail with OutOfRange and full interval metadata.
        const auto out_of_range = narrowed->with_value(3.0);
        ASSERT_FALSE(out_of_range.has_value());
        EXPECT_EQ(out_of_range.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
        EXPECT_EQ(out_of_range.error().interval_error_kind(), cxxgpr::IntervalErrorKind::OutOfRange);
        EXPECT_EQ(out_of_range.error().interval_value(), 3.0);
        EXPECT_EQ(out_of_range.error().interval_lo(), 1.0);
        EXPECT_EQ(out_of_range.error().interval_hi(), 3.0);

        // Open lower endpoint 1.0 and NaN values must also fail creation.
        const auto outside_narrow = cxxgpr::BoundedParam::create(1.0, *iv);
        ASSERT_FALSE(outside_narrow.has_value());
        EXPECT_EQ(outside_narrow.error().code(), cxxgpr::GprErrorCode::InvalidInterval);

        const auto nan_param = cxxgpr::BoundedParam::default_positive(std::numeric_limits<double>::quiet_NaN());
        ASSERT_FALSE(nan_param.has_value());
        EXPECT_EQ(nan_param.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    }

}  // namespace
