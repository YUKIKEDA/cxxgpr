// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/linalg/jitter.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

// Spec: DES-01, DES-02, DES-04
namespace
{

    using cxxgpr::GprErrorCode;
    using cxxgpr::JitterPolicy;

    TEST(JitterPolicyTest, DefaultIsFixedZeroWithEmptySchedule)
    {
        const JitterPolicy policy;
        EXPECT_TRUE(policy.is_fixed());
        EXPECT_FALSE(policy.is_adaptive());
        EXPECT_EQ(policy.kind(), JitterPolicy::Kind::Fixed);
        const auto fixed = policy.as_fixed();
        if (!fixed.has_value())
        {
            ADD_FAILURE() << "expected Fixed jitter";
            return;
        }
        EXPECT_EQ(fixed->jitter(), 0.0);
        EXPECT_FALSE(policy.as_adaptive().has_value());

        auto sched = policy.retry_schedule();
        EXPECT_FALSE(sched.next().has_value());
    }

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

    TEST(JitterPolicyTest, AdaptiveStopsWhenNextExceedsMaxJitter)
    {
        const auto policy_res = JitterPolicy::adaptive(1e-3, 10.0, 5, 5e-2);
        ASSERT_TRUE(policy_res.has_value());

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

    TEST(JitterPolicyTest, RejectsInvalidFixedAndAdaptiveConfigs)
    {
        const double nan_val = std::numeric_limits<double>::quiet_NaN();
        const double inf_val = std::numeric_limits<double>::infinity();

        for (const double bad : {-1e-8, -1.0, nan_val, inf_val, -inf_val})
        {
            const auto res = JitterPolicy::fixed(bad);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), GprErrorCode::InvalidConfig);
        }

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
