// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-02, REQ-06, REQ-11, DES-01, DES-04
/// @file likelihood_test.cpp
/// @brief Unit tests for GaussianLikelihood parameter bounds, derivatives, and zero-allocation calls.
/// @details Verifies construction from linear and log-noise variances, rejection of invalid and
/// out-of-bounds noise parameters, atomic failure guarantees on set_params(), subnormal handling,
/// analytical log-space gradients against central finite differences, and zero-allocation execution.

#include "cxxgpr/core/likelihood.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Verifies GaussianLikelihood construction from linear and log-noise variances.
    /// @details Confirms noise_variance(), log_noise_variance(), default_positive() bounds,
    /// num_params() == 1, and roundtrip equivalence with from_log_noise_variance().
    TEST(GaussianLikelihoodTest, ConstructAndInspectDefaultBounds)
    {
        const auto lik = cxxgpr::GaussianLikelihood::create(0.25);
        ASSERT_TRUE(lik.has_value());
        EXPECT_DOUBLE_EQ(lik->noise_variance(), 0.25);
        EXPECT_NEAR(lik->log_noise_variance(), std::log(0.25), 1e-12);
        EXPECT_EQ(lik->bounds(), cxxgpr::Interval::default_positive());
        EXPECT_EQ(lik->num_params(), 1U);

        // Constructing from log(0.25) must recover the identical noise variance.
        const auto from_log = cxxgpr::GaussianLikelihood::from_log_noise_variance(std::log(0.25));
        ASSERT_TRUE(from_log.has_value());
        EXPECT_NEAR(from_log->noise_variance(), lik->noise_variance(), 1e-12);
    }

    /// @brief Negative controls verifying invalid values and out-of-bounds noise fail appropriately.
    /// @details Checks InvalidNoiseVariance for non-positive, non-finite, and exp-overflow/underflow
    /// inputs, and InvalidInterval for finite positive values outside default bounds (1e-5, 1e5).
    TEST(GaussianLikelihoodTest, RejectsInvalidValuesAndOutOfBoundsValues)
    {
        // Zero, negative, NaN, and +/-Inf fail positivity/finiteness checks before interval bounds.
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

        // Non-finite log variances or magnitudes that underflow/overflow exp() must fail cleanly.
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

    /// @brief Verifies parameter get/set roundtrips, bounds updates, and atomic failure preservation.
    /// @details Confirms with_bounds() rejects intervals excluding the current value, get_params() does
    /// not mutate caller buffers on LengthMismatch, and failed set_params() calls leave state unchanged.
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

        // Wrong-length output buffer must fail without overwriting caller sentinel values.
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

    /// @brief Verifies positive IEEE 754 subnormal noise variances are permitted within widened bounds.
    /// @details Sets log(sigma_n^2) = -740 within (0.0, 1.0), checks the resulting variance is positive,
    /// finite, and subnormal, and confirms default bounds still reject it as out-of-range.
    TEST(GaussianLikelihoodTest, KeepsSubnormalNoiseInsideWidenedBounds)
    {
        const auto wide = cxxgpr::Interval::create(0.0, 1.0);
        ASSERT_TRUE(wide.has_value());
        const auto base = cxxgpr::GaussianLikelihood::create(0.5);
        ASSERT_TRUE(base.has_value());
        auto lik = base->with_bounds(*wide);
        ASSERT_TRUE(lik.has_value());

        // exp(-740) falls below DBL_MIN (~2.2e-308) into the positive IEEE 754 subnormal range.
        const std::array<double, 1> subnormal_theta{-740.0};
        ASSERT_TRUE(lik->set_params(subnormal_theta).has_value());
        const double noise = lik->noise_variance();
        EXPECT_GT(noise, 0.0);
        EXPECT_TRUE(std::isfinite(noise));
        EXPECT_FALSE(std::isnormal(noise));

        // Default bounds (1e-5, 1e5) reject the subnormal noise as out-of-range, not non-finite.
        const auto rejected_default = cxxgpr::GaussianLikelihood::create(noise);
        ASSERT_FALSE(rejected_default.has_value());
        EXPECT_EQ(rejected_default.error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    }

    /// @brief Verifies diagonal noise addition and analytical log-variance derivative against finite diffs.
    /// @details Checks add_noise_diag(), verifies d(sigma_n^2)/d(log(sigma_n^2)) == sigma_n^2 (and rejects
    /// nonzero parameter indices), and validates against a central finite-difference oracle.
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

    /// @brief Verifies all GaussianLikelihood hot-path operations allocate zero heap memory.
    /// @details Executes get_params, set_params, bounds_into, add_noise_diag, and noise_grad_diag
    /// inside AuditMode::CppAndEigen NoAllocationScope and asserts zero allocation delta.
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
        // Audit all parameter and diagonal operations in one scope without triggering GTest allocations.
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
