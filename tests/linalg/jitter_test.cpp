// SPDX-License-Identifier: Apache-2.0
// Spec: DES-01, DES-02, DES-04
/// @file jitter_test.cpp
/// @brief Unit tests for JitterPolicy construction, retry schedules, and invalid-config rejection.
/// @details Verifies default Fixed(0.0) empty schedules, single-retry Fixed(j > 0) policies,
/// geometrically escalating Adaptive schedules bounded by max_retries and max_jitter, and
/// InvalidConfig diagnostics across all parameter constraints.

#include "cxxgpr/linalg/jitter.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace
{

    /// @brief Brings GprErrorCode into the anonymous test namespace.
    using cxxgpr::GprErrorCode;
    /// @brief Brings JitterPolicy into the anonymous test namespace.
    using cxxgpr::JitterPolicy;

    /// @brief Verifies default-constructed JitterPolicy uses Fixed(0.0) with an empty retry schedule.
    /// @details Confirms kind(), is_fixed(), as_fixed()->jitter() == 0.0, unset as_adaptive(), and an
    /// immediately exhausted retry_schedule().
    TEST(JitterPolicyTest, DefaultIsFixedZeroWithEmptySchedule)
    {
        const JitterPolicy policy;
        EXPECT_TRUE(policy.is_fixed());
        EXPECT_FALSE(policy.is_adaptive());
        EXPECT_EQ(policy.kind(), JitterPolicy::Kind::Fixed);
        // Guard optional FixedJitter before inspecting its scalar value (clang-tidy check).
        const auto fixed = policy.as_fixed();
        if (!fixed.has_value())
        {
            ADD_FAILURE() << "expected Fixed jitter";
            return;
        }
        EXPECT_EQ(fixed->jitter(), 0.0);
        EXPECT_FALSE(policy.as_adaptive().has_value());

        // Zero fixed jitter permits only the unjittered j = 0 attempt and emits no retry entries.
        auto sched = policy.retry_schedule();
        EXPECT_FALSE(sched.next().has_value());
    }

    /// @brief Verifies strictly positive JitterPolicy::fixed() schedules yield exactly one retry jitter.
    /// @details Checks Fixed(1e-6) emits 1e-6 on the first next() call and std::nullopt thereafter.
    TEST(JitterPolicyTest, FixedPositiveYieldsSingleRetry)
    {
        const auto policy_res = JitterPolicy::fixed(1e-6);
        ASSERT_TRUE(policy_res.has_value());
        const JitterPolicy& policy = *policy_res;
        EXPECT_TRUE(policy.is_fixed());
        const auto fixed = policy.as_fixed();
        if (!fixed.has_value())
        {
            ADD_FAILURE() << "expected Fixed jitter";
            return;
        }
        EXPECT_DOUBLE_EQ(fixed->jitter(), 1e-6);

        // Positive fixed jitter schedules exactly one retry after j = 0 fails.
        auto sched = policy.retry_schedule();
        const auto first = sched.next();
        if (!first.has_value())
        {
            ADD_FAILURE() << "expected first jitter";
            return;
        }
        EXPECT_DOUBLE_EQ(*first, 1e-6);
        EXPECT_FALSE(sched.next().has_value());
    }

    /// @brief Verifies Adaptive jitter schedule escalates geometrically within retry and cap bounds.
    /// @details Confirms adaptive(1e-6, 10.0, 4, 1e-2) exposes exact metadata and yields [1e-6, 1e-5,
    /// 1e-4, 1e-3] in ascending geometric progression.
    TEST(JitterPolicyTest, AdaptiveEscalatesWithinRetriesAndMaxJitter)
    {
        const auto policy_res = JitterPolicy::adaptive(1e-6, 10.0, 4, 1e-2);
        ASSERT_TRUE(policy_res.has_value());
        const JitterPolicy& policy = *policy_res;
        EXPECT_TRUE(policy.is_adaptive());
        EXPECT_EQ(policy.kind(), JitterPolicy::Kind::Adaptive);
        const auto adaptive = policy.as_adaptive();
        if (!adaptive.has_value())
        {
            ADD_FAILURE() << "expected Adaptive jitter";
            return;
        }
        EXPECT_DOUBLE_EQ(adaptive->initial(), 1e-6);
        EXPECT_DOUBLE_EQ(adaptive->multiplier(), 10.0);
        EXPECT_EQ(adaptive->max_retries(), 4U);
        EXPECT_DOUBLE_EQ(adaptive->max_jitter(), 1e-2);

        // Exhaust the schedule and verify all four geometric steps match 10x escalation.
        auto sched = policy.retry_schedule();
        std::vector<double> yielded;
        while (const auto j = sched.next())
        {
            yielded.push_back(*j);
        }
        ASSERT_EQ(yielded.size(), 4U);
        EXPECT_DOUBLE_EQ(yielded[0], 1e-6);
        EXPECT_DOUBLE_EQ(yielded[1], 1e-5);
        EXPECT_DOUBLE_EQ(yielded[2], 1e-4);
        EXPECT_DOUBLE_EQ(yielded[3], 1e-3);
    }

    /// @brief Verifies Adaptive jitter stops early when the next scaled candidate exceeds max_jitter.
    /// @details With initial = 1e-3, multiplier = 10, max_retries = 5, and max_jitter = 5e-2, the third
    /// step (0.1) exceeds 5e-2 and terminates after yielding [1e-3, 1e-2].
    TEST(JitterPolicyTest, AdaptiveStopsWhenNextExceedsMaxJitter)
    {
        const auto policy_res = JitterPolicy::adaptive(1e-3, 10.0, 5, 5e-2);
        ASSERT_TRUE(policy_res.has_value());

        // Candidate 1e-1 exceeds max_jitter = 5e-2, truncating the 5-retry budget after 2 steps.
        auto sched = policy_res->retry_schedule();
        std::vector<double> yielded;
        while (const auto j = sched.next())
        {
            yielded.push_back(*j);
        }
        ASSERT_EQ(yielded.size(), 2U);
        EXPECT_DOUBLE_EQ(yielded[0], 1e-3);
        EXPECT_DOUBLE_EQ(yielded[1], 1e-2);
    }

    /// @brief Negative controls verifying invalid fixed and adaptive jitter configs return InvalidConfig.
    /// @details Tests negative/NaN/Inf fixed jitters, non-positive or NaN initial, multiplier <= 1 or Inf,
    /// zero max_retries, max_jitter < initial, and infinite max_jitter.
    TEST(JitterPolicyTest, RejectsInvalidFixedAndAdaptiveConfigs)
    {
        const double nan_val = std::numeric_limits<double>::quiet_NaN();
        const double inf_val = std::numeric_limits<double>::infinity();

        // Fixed jitter must be finite and >= 0.0.
        for (const double bad : {-1e-8, -1.0, nan_val, inf_val, -inf_val})
        {
            const auto res = JitterPolicy::fixed(bad);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), GprErrorCode::InvalidConfig);
        }

        // Adaptive jitter requires initial > 0, multiplier > 1, max_retries >= 1, and finite
        // max_jitter >= initial.
        const auto bad_init_zero = JitterPolicy::adaptive(0.0, 10.0, 3, 1.0);
        ASSERT_FALSE(bad_init_zero.has_value());
        EXPECT_EQ(bad_init_zero.error().code(), GprErrorCode::InvalidConfig);

        const auto bad_init_nan = JitterPolicy::adaptive(nan_val, 10.0, 3, 1.0);
        ASSERT_FALSE(bad_init_nan.has_value());
        EXPECT_EQ(bad_init_nan.error().code(), GprErrorCode::InvalidConfig);

        const auto bad_mult_one = JitterPolicy::adaptive(1e-8, 1.0, 3, 1.0);
        ASSERT_FALSE(bad_mult_one.has_value());
        EXPECT_EQ(bad_mult_one.error().code(), GprErrorCode::InvalidConfig);

        const auto bad_mult_inf = JitterPolicy::adaptive(1e-8, inf_val, 3, 1.0);
        ASSERT_FALSE(bad_mult_inf.has_value());
        EXPECT_EQ(bad_mult_inf.error().code(), GprErrorCode::InvalidConfig);

        const auto bad_retries = JitterPolicy::adaptive(1e-8, 10.0, 0, 1.0);
        ASSERT_FALSE(bad_retries.has_value());
        EXPECT_EQ(bad_retries.error().code(), GprErrorCode::InvalidConfig);

        const auto bad_max_lt = JitterPolicy::adaptive(1.0, 10.0, 3, 0.5);
        ASSERT_FALSE(bad_max_lt.has_value());
        EXPECT_EQ(bad_max_lt.error().code(), GprErrorCode::InvalidConfig);

        const auto bad_max_inf = JitterPolicy::adaptive(1.0, 10.0, 3, inf_val);
        ASSERT_FALSE(bad_max_inf.has_value());
        EXPECT_EQ(bad_max_inf.error().code(), GprErrorCode::InvalidConfig);
    }

}  // namespace
