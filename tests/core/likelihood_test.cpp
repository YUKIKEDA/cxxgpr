// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/likelihood.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include "allocation_audit.hpp"

// Spec: REQ-02, REQ-06, REQ-11, DES-01, DES-04
namespace
{

    TEST(GaussianLikelihoodTest, ConstructAndInspectDefaultBounds)
    {
        const auto lik = cxxgpr::GaussianLikelihood::create(0.25);
        ASSERT_TRUE(lik.has_value());
        EXPECT_DOUBLE_EQ(lik->noise_variance(), 0.25);
        EXPECT_NEAR(lik->log_noise_variance(), std::log(0.25), 1e-12);
        EXPECT_EQ(lik->bounds(), cxxgpr::Interval::default_positive());
        EXPECT_EQ(lik->num_params(), 1U);

        const auto from_log = cxxgpr::GaussianLikelihood::from_log_noise_variance(std::log(0.25));
        ASSERT_TRUE(from_log.has_value());
        EXPECT_NEAR(from_log->noise_variance(), lik->noise_variance(), 1e-12);
    }

    TEST(GaussianLikelihoodTest, RejectsInvalidValuesAndOutOfBoundsValues)
    {
        for (const double bad :
            {0.0, -0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()})
        {
            const auto err = cxxgpr::GaussianLikelihood::create(bad);
            ASSERT_FALSE(err.has_value());
            EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::InvalidNoiseVariance);
        }

        // Positive and finite, but outside the default open interval (1e-5, 1e5).
        for (const double out : {1e-5, 1e-6, 1e5, 1e6})
        {
            const auto err = cxxgpr::GaussianLikelihood::create(out);
            ASSERT_FALSE(err.has_value());
            EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
        }

        for (const double bad_log :
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(), 800.0, -800.0})
        {
            const auto err = cxxgpr::GaussianLikelihood::from_log_noise_variance(bad_log);
            ASSERT_FALSE(err.has_value());
            EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::InvalidNoiseVariance);
        }

        const auto log_out = cxxgpr::GaussianLikelihood::from_log_noise_variance(std::log(1e-6));
        ASSERT_FALSE(log_out.has_value());
        EXPECT_EQ(log_out.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    }

    TEST(GaussianLikelihoodTest, ParameterRoundtripAndAtomicFailure)
    {
        const auto bounded_iv = cxxgpr::Interval::create(1e-3, 1.0);
        ASSERT_TRUE(bounded_iv.has_value());
        const auto base = cxxgpr::GaussianLikelihood::create(0.5);
        ASSERT_TRUE(base.has_value());

        auto lik = base->with_bounds(*bounded_iv);
        ASSERT_TRUE(lik.has_value());
        EXPECT_EQ(lik->bounds(), *bounded_iv);

        // Current value outside new interval must fail cleanly.
        const auto too_narrow = cxxgpr::Interval::create(1e-3, 0.25);
        ASSERT_TRUE(too_narrow.has_value());
        const auto bad_bounds = lik->with_bounds(*too_narrow);
        ASSERT_FALSE(bad_bounds.has_value());
        EXPECT_EQ(bad_bounds.error().code(), cxxgpr::GprErrorCode::InvalidInterval);

        std::array<double, 1> theta{0.0};
        ASSERT_TRUE(lik->get_params(theta).has_value());
        EXPECT_NEAR(theta[0], std::log(0.5), 1e-12);

        std::array<double, 2> wrong_out{42.0, 99.0};
        const auto get_err = lik->get_params(wrong_out);
        ASSERT_FALSE(get_err.has_value());
        EXPECT_EQ(get_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        EXPECT_DOUBLE_EQ(wrong_out[0], 42.0);

        std::array<cxxgpr::Interval, 1> bounds_out{cxxgpr::Interval::default_positive()};
        ASSERT_TRUE(lik->bounds_into(bounds_out).has_value());
        EXPECT_EQ(bounds_out[0], *bounded_iv);

        // Successful update.
        const std::array<double, 1> next_theta{std::log(0.25)};
        ASSERT_TRUE(lik->set_params(next_theta).has_value());
        EXPECT_NEAR(lik->noise_variance(), 0.25, 1e-12);

        // Failed updates must leave *lik unchanged.
        const std::array<double, 2> bad_len{0.0, 0.0};
        const auto err_len = lik->set_params(bad_len);
        ASSERT_FALSE(err_len.has_value());
        EXPECT_EQ(err_len.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
        EXPECT_NEAR(lik->noise_variance(), 0.25, 1e-12);

        const std::array<double, 1> underflow{-800.0};
        const auto err_underflow = lik->set_params(underflow);
        ASSERT_FALSE(err_underflow.has_value());
        EXPECT_EQ(err_underflow.error().code(), cxxgpr::GprErrorCode::InvalidNoiseVariance);
        EXPECT_NEAR(lik->noise_variance(), 0.25, 1e-12);

        const std::array<double, 1> outside{std::numbers::ln2};
        const auto err_outside = lik->set_params(outside);
        ASSERT_FALSE(err_outside.has_value());
        EXPECT_EQ(err_outside.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
        EXPECT_NEAR(lik->noise_variance(), 0.25, 1e-12);
    }

    TEST(GaussianLikelihoodTest, KeepsSubnormalNoiseInsideWidenedBounds)
    {
        const auto wide = cxxgpr::Interval::create(0.0, 1.0);
        ASSERT_TRUE(wide.has_value());
        const auto base = cxxgpr::GaussianLikelihood::create(0.5);
        ASSERT_TRUE(base.has_value());
        auto lik = base->with_bounds(*wide);
        ASSERT_TRUE(lik.has_value());

        const std::array<double, 1> subnormal_theta{-740.0};
        ASSERT_TRUE(lik->set_params(subnormal_theta).has_value());
        const double noise = lik->noise_variance();
        EXPECT_GT(noise, 0.0);
        EXPECT_TRUE(std::isfinite(noise));
        EXPECT_FALSE(std::isnormal(noise));

        const auto rejected_default = cxxgpr::GaussianLikelihood::create(noise);
        ASSERT_FALSE(rejected_default.has_value());
        EXPECT_EQ(rejected_default.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    }

    TEST(GaussianLikelihoodTest, AddNoiseDiagAndGradientMatchExactDerivativeAndFiniteDiff)
    {
        const double variance = 0.25;
        const auto lik = cxxgpr::GaussianLikelihood::create(variance);
        ASSERT_TRUE(lik.has_value());

        std::array<double, 3> diag{1.0, 2.0, 3.0};
        lik->add_noise_diag(diag);
        EXPECT_NEAR(diag[0], 1.25, 1e-12);
        EXPECT_NEAR(diag[1], 2.25, 1e-12);
        EXPECT_NEAR(diag[2], 3.25, 1e-12);

        // Since theta = log(sigma_n^2), dK/dtheta = sigma_n^2 I (not 2 * sigma_n I).
        std::array<double, 2> grad{0.0, 0.0};
        ASSERT_TRUE(lik->noise_grad_diag(grad, 0).has_value());
        EXPECT_NEAR(grad[0], variance, 1e-12);
        EXPECT_NEAR(grad[1], variance, 1e-12);
        EXPECT_GT(std::abs(grad[0] - (2.0 * std::sqrt(variance))), 0.1);

        // Non-zero parameter index must return IndexOutOfRange without modifying grad.
        const auto idx_err = lik->noise_grad_diag(grad, 1);
        ASSERT_FALSE(idx_err.has_value());
        EXPECT_EQ(idx_err.error().code(), cxxgpr::GprErrorCode::IndexOutOfRange);
        EXPECT_NEAR(grad[0], variance, 1e-12);

        // Verify against central finite difference in log space.
        const auto probe = cxxgpr::GaussianLikelihood::from_log_noise_variance(-1.5);
        ASSERT_TRUE(probe.has_value());
        const double theta = probe->log_noise_variance();
        const double step = 1e-6;
        const auto plus = cxxgpr::GaussianLikelihood::from_log_noise_variance(theta + step);
        const auto minus = cxxgpr::GaussianLikelihood::from_log_noise_variance(theta - step);
        ASSERT_TRUE(plus.has_value());
        ASSERT_TRUE(minus.has_value());
        const double fd = (plus->noise_variance() - minus->noise_variance()) / (2.0 * step);

        std::array<double, 1> fd_check{0.0};
        ASSERT_TRUE(probe->noise_grad_diag(fd_check, 0).has_value());
        EXPECT_NEAR(fd_check[0], fd, 1e-10);
        EXPECT_NEAR(fd_check[0], probe->noise_variance(), 1e-12);
    }

    TEST(GaussianLikelihoodTest, HotPathOperationsAllocateZero)
    {
        auto lik = cxxgpr::GaussianLikelihood::create(0.2);
        ASSERT_TRUE(lik.has_value());

        std::array<double, 1> params{0.0};
        const std::array<double, 1> updated{std::log(0.3)};
        std::array<cxxgpr::Interval, 1> ivs{cxxgpr::Interval::default_positive()};
        std::array<double, 4> diag{1.0, 2.0, 3.0, 4.0};
        std::array<double, 4> grad{0.0, 0.0, 0.0, 0.0};

        std::size_t delta = 1;
        bool all_ok = false;
        {
            cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
            const bool ok1 = lik->get_params(params).has_value();
            const bool ok2 = lik->set_params(updated).has_value();
            const bool ok3 = lik->bounds_into(ivs).has_value();
            lik->add_noise_diag(diag);
            const bool ok4 = lik->noise_grad_diag(grad, 0).has_value();
            cxxgpr::test_support::touch_for_test(diag.data());
            cxxgpr::test_support::touch_for_test(grad.data());
            all_ok = ok1 && ok2 && ok3 && ok4;
            delta = scope.delta_allocations();
        }
        EXPECT_TRUE(all_ok);
        EXPECT_EQ(delta, 0U);
        EXPECT_NEAR(grad[0], 0.3, 1e-12);
    }

}  // namespace
