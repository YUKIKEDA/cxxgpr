// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-02, REQ-10, REQ-11, DES-01, DES-02, DES-18
/// @file target_test.cpp
/// @brief Unit tests for built-in and custom target preprocessing transforms.
/// @details Verifies IdentityTarget, StandardizeTarget, and MinMaxTarget mean/variance/covariance
/// forward and inverse mappings, constant-target unit fallback without NaNs, zero-allocation hot
/// paths, validation error precedence, and custom UnfittedTargetTransform extensions.

#include "cxxgpr/transform/target.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Brings GprError into the anonymous test namespace.
    using cxxgpr::GprError;
    /// @brief Brings GprErrorCode into the anonymous test namespace.
    using cxxgpr::GprErrorCode;
    /// @brief Brings PersistErrorKind into the anonymous test namespace.
    using cxxgpr::PersistErrorKind;
    /// @brief Brings allocation AuditMode enum into the anonymous test namespace.
    using cxxgpr::test_support::AuditMode;
    /// @brief Brings RAII NoAllocationScope guard into the anonymous test namespace.
    using cxxgpr::test_support::NoAllocationScope;
    /// @brief Brings FittedMinMaxTarget into the anonymous test namespace.
    using cxxgpr::transform::FittedMinMaxTarget;
    /// @brief Brings FittedStandardizeTarget into the anonymous test namespace.
    using cxxgpr::transform::FittedStandardizeTarget;
    /// @brief Brings FittedTargetTransform base interface into the anonymous test namespace.
    using cxxgpr::transform::FittedTargetTransform;
    /// @brief Brings IdentityTarget into the anonymous test namespace.
    using cxxgpr::transform::IdentityTarget;
    /// @brief Brings MinMaxTarget into the anonymous test namespace.
    using cxxgpr::transform::MinMaxTarget;
    /// @brief Brings StandardizeTarget into the anonymous test namespace.
    using cxxgpr::transform::StandardizeTarget;
    /// @brief Brings UnfittedTargetTransform base interface into the anonymous test namespace.
    using cxxgpr::transform::UnfittedTargetTransform;

    /// @brief Floating-point comparison tolerance used across target transform assertions.
    constexpr double k_tol = 1e-12;

    /// @brief Test fitted target transform shifting targets by a fitted empirical mean without scaling.
    /// @details Leaves predictive variance unscaled (unit Jacobian) to test custom FittedTargetTransform
    /// polymorphism and default inverse_transform_covariance() delegation.
    class FittedCenterY final : public FittedTargetTransform
    {
    public:
        /// @brief Constructs a mean-centering target transform with the given empirical offset.
        /// @details Stores mean so transform() subtracts it and inverse_transform_mean() adds it back.
        /// @param[in] mean Empirical training target mean.
        explicit FittedCenterY(double mean) : mean_(mean) {}

        /// @brief Subtracts the fitted mean from every element of y in place.
        /// @details Rejects empty spans with GprErrorCode::EmptyInput.
        /// @param[in,out] y Mutable span of target values to center.
        /// @return Empty expected on success, or EmptyInput when y is empty.
        [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override
        {
            if (y.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            for (double& v : y)
            {
                v -= mean_;
            }
            return {};
        }

        /// @brief Adds the fitted mean back to every predicted mean entry in place.
        /// @details Rejects empty spans with GprErrorCode::EmptyInput.
        /// @param[in,out] mean Mutable span of standardized predictive means.
        /// @return Empty expected on success, or EmptyInput when mean is empty.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean(std::span<double> mean) const override
        {
            if (mean.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            for (double& v : mean)
            {
                v += mean_;
            }
            return {};
        }

        /// @brief Leaves predictive variances unchanged because mean centering has unit Jacobian.
        /// @details Rejects empty spans with GprErrorCode::EmptyInput.
        /// @param[in,out] var Mutable span of predictive variances.
        /// @return Empty expected on success, or EmptyInput when var is empty.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance(std::span<double> var) const override
        {
            if (var.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            return {};
        }

        /// @brief Clones this fitted centering transform into a new unique_ptr.
        /// @details Preserves mean_ on the cloned instance.
        /// @return Heap-allocated copy of *this.
        [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override
        {
            return std::make_unique<FittedCenterY>(*this);
        }

    private:
        /// @brief Empirical training target mean subtracted during transform().
        double mean_ = 0.0;
    };

    /// @brief Test unfitted custom target transform fitting an empirical mean-centering FittedCenterY.
    /// @details Computes the sample mean over non-empty target spans and leaves persist_state_json() at default.
    class UnfittedCenterY final : public UnfittedTargetTransform
    {
    public:
        /// @brief Computes the empirical target mean and returns a FittedCenterY transform.
        /// @details Rejects empty spans with GprErrorCode::EmptyInput.
        /// @param[in] y Training target vector span.
        /// @return FittedCenterY wrapped in unique_ptr, or EmptyInput when y is empty.
        [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError> fit(
            std::span<const double> y
        ) const override
        {
            if (y.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            double sum = 0.0;
            for (const double v : y)
            {
                sum += v;
            }
            return std::make_unique<FittedCenterY>(sum / static_cast<double>(y.size()));
        }

        /// @brief Clones this unfitted centering transform into a new unique_ptr.
        /// @details Supports copying inside polymorphic target transform containers.
        /// @return Heap-allocated copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override
        {
            return std::make_unique<UnfittedCenterY>(*this);
        }
    };

    /// @brief Verifies IdentityTarget leaves targets, means, variances, and covariances unchanged.
    /// @details Confirms all four forward and inverse operations return exact copies of orig.
    TEST(TransformTargetTest, IdentityLeavesTargetsAndPredictionsUnchanged)
    {
        const std::array<double, 3> orig{1.0, 2.0, 3.0};
        auto fitted = IdentityTarget{}.fit_typed(orig);
        ASSERT_TRUE(fitted.has_value());

        // Forward transform and all predictive inverse methods must be exact no-ops.
        std::array<double, 3> buf = orig;
        ASSERT_TRUE(fitted->transform(buf).has_value());
        EXPECT_EQ(buf, orig);
        ASSERT_TRUE(fitted->inverse_transform_mean(buf).has_value());
        EXPECT_EQ(buf, orig);
        ASSERT_TRUE(fitted->inverse_transform_variance(buf).has_value());
        EXPECT_EQ(buf, orig);
        ASSERT_TRUE(fitted->inverse_transform_covariance(buf).has_value());
        EXPECT_EQ(buf, orig);
    }

    /// @brief Verifies StandardizeTarget centers/scales targets and scales variance/covariance by std^2.
    /// @details Checks mean = 3.0 and population std = sqrt(8/3) on [1, 3, 5], zero mean and unit
    /// variance after transform(), exact mean roundtrip, and s^2 scaling on variances and covariances.
    TEST(TransformTargetTest, StandardizeCentersScalesAndInvertsMeanVarianceCovariance)
    {
        const std::array<double, 3> y{1.0, 3.0, 5.0};
        auto fitted = StandardizeTarget{}.fit_typed(y);
        ASSERT_TRUE(fitted.has_value());
        const double s = fitted->std();
        EXPECT_NEAR(fitted->mean(), 3.0, k_tol);
        EXPECT_NEAR(s, std::sqrt(8.0 / 3.0), k_tol);

        // Transformed targets must have empirical mean 0 and empirical population variance 1.
        std::array<double, 3> z = y;
        ASSERT_TRUE(fitted->transform(z).has_value());
        EXPECT_NEAR((z[0] + z[1] + z[2]) / 3.0, 0.0, k_tol);
        EXPECT_NEAR(((z[0] * z[0]) + (z[1] * z[1]) + (z[2] * z[2])) / 3.0, 1.0, k_tol);

        ASSERT_TRUE(fitted->inverse_transform_mean(z).has_value());
        for (std::size_t i = 0; i < y.size(); ++i)
        {
            EXPECT_NEAR(z[i], y[i], k_tol);
        }

        // Predictive variance and covariance both scale quadratically by std^2.
        std::array<double, 3> var{0.25, 1.0, 4.0};
        ASSERT_TRUE(fitted->inverse_transform_variance(var).has_value());
        EXPECT_NEAR(var[0], 0.25 * s * s, k_tol);
        EXPECT_NEAR(var[1], 1.0 * s * s, k_tol);
        EXPECT_NEAR(var[2], 4.0 * s * s, k_tol);

        std::array<double, 4> cov{1.0, 0.5, 0.5, 4.0};
        ASSERT_TRUE(fitted->inverse_transform_covariance(cov).has_value());
        EXPECT_NEAR(cov[0], 1.0 * s * s, k_tol);
        EXPECT_NEAR(cov[1], 0.5 * s * s, k_tol);
        EXPECT_NEAR(cov[2], 0.5 * s * s, k_tol);
        EXPECT_NEAR(cov[3], 4.0 * s * s, k_tol);
    }

    /// @brief Verifies constant training targets fall back cleanly to unit scale without NaNs.
    /// @details Confirms both StandardizeTarget and MinMaxTarget map constant [4, 4, 4] to 0.0 and
    /// invert back to 4.0 without division by zero.
    TEST(TransformTargetTest, ConstantTargetsUseUnitFallbackWithoutNaN)
    {
        const std::array<double, 3> y{4.0, 4.0, 4.0};
        auto std_fitted = StandardizeTarget{}.fit_typed(y);
        ASSERT_TRUE(std_fitted.has_value());
        EXPECT_DOUBLE_EQ(std_fitted->mean(), 4.0);
        EXPECT_DOUBLE_EQ(std_fitted->std(), 1.0);

        // Constant targets center to 0.0 using unit std fallback and roundtrip back to 4.0.
        std::array<double, 3> z = y;
        ASSERT_TRUE(std_fitted->transform(z).has_value());
        EXPECT_DOUBLE_EQ(z[0], 0.0);
        ASSERT_TRUE(std_fitted->inverse_transform_mean(z).has_value());
        EXPECT_DOUBLE_EQ(z[0], 4.0);

        auto mm_fitted = MinMaxTarget{}.fit_typed(y);
        ASSERT_TRUE(mm_fitted.has_value());
        std::array<double, 3> zm = y;
        ASSERT_TRUE(mm_fitted->transform(zm).has_value());
        EXPECT_DOUBLE_EQ(zm[0], 0.0);
        ASSERT_TRUE(mm_fitted->inverse_transform_mean(zm).has_value());
        EXPECT_DOUBLE_EQ(zm[0], 4.0);
    }

    /// @brief Verifies MinMaxTarget custom-range scaling, mean inversion, and Jacobian variance scaling.
    /// @details Maps [1, 3, 5] into [-1, 1] (Jacobian dy/dz = 4 / 2 = 2) and checks inverse_transform_mean_into
    /// and inverse_transform_variance_into (scale^2 = 4).
    TEST(TransformTargetTest, MinMaxScalesAndInvertsMeanAndVariance)
    {
        const std::array<double, 3> y{1.0, 3.0, 5.0};
        auto spec = MinMaxTarget::create(-1.0, 1.0);
        ASSERT_TRUE(spec.has_value());
        auto fitted = spec->fit_typed(y);
        ASSERT_TRUE(fitted.has_value());
        EXPECT_DOUBLE_EQ(fitted->min(), 1.0);
        EXPECT_DOUBLE_EQ(fitted->max(), 5.0);

        std::array<double, 3> z{};
        ASSERT_TRUE(fitted->transform_into(y, z).has_value());
        EXPECT_NEAR(z[0], -1.0, k_tol);
        EXPECT_NEAR(z[1], 0.0, k_tol);
        EXPECT_NEAR(z[2], 1.0, k_tol);

        std::array<double, 3> rec{};
        ASSERT_TRUE(fitted->inverse_transform_mean_into(z, rec).has_value());
        for (std::size_t i = 0; i < y.size(); ++i)
        {
            EXPECT_NEAR(rec[i], y[i], k_tol);
        }

        // data_span = 4, out_span = 2 -> scale = 2 -> scale^2 = 4
        const std::array<double, 2> var_in{1.0, 0.25};
        std::array<double, 2> var_out{};
        ASSERT_TRUE(fitted->inverse_transform_variance_into(var_in, var_out).has_value());
        EXPECT_NEAR(var_out[0], 4.0, k_tol);
        EXPECT_NEAR(var_out[1], 1.0, k_tol);
    }

    /// @brief Verifies built-in IdentityTarget, StandardizeTarget, and MinMaxTarget allocate zero heap memory.
    /// @details Runs all eight in-place and out-of-place target operations inside AuditMode::CppOnly
    /// NoAllocationScope and checks delta_allocations() == 0.
    TEST(TransformTargetTest, HotPathZeroAllocationForBuiltins)
    {
        const std::array<double, 3> y{1.0, 3.0, 5.0};
        auto id = IdentityTarget{}.fit(y);
        auto st = StandardizeTarget{}.fit(y);
        auto mm = MinMaxTarget{}.fit(y);
        ASSERT_TRUE(id.has_value());
        ASSERT_TRUE(st.has_value());
        ASSERT_TRUE(mm.has_value());

        // Audit all eight FittedTargetTransform operations for each fitted built-in target map.
        std::array<double, 3> mut_buf = y;
        std::array<double, 3> out_buf{};
        for (const auto* map : {id->get(), st->get(), mm->get()})
        {
            const NoAllocationScope audit(AuditMode::CppOnly);
            ASSERT_TRUE(map->transform(mut_buf).has_value());
            ASSERT_TRUE(map->inverse_transform_mean(mut_buf).has_value());
            ASSERT_TRUE(map->inverse_transform_variance(mut_buf).has_value());
            ASSERT_TRUE(map->inverse_transform_covariance(mut_buf).has_value());
            ASSERT_TRUE(map->transform_into(y, out_buf).has_value());
            ASSERT_TRUE(map->inverse_transform_mean_into(out_buf, mut_buf).has_value());
            ASSERT_TRUE(map->inverse_transform_variance_into(out_buf, mut_buf).has_value());
            ASSERT_TRUE(map->inverse_transform_covariance_into(out_buf, mut_buf).has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }
    }

    /// @brief Negative controls verifying target validation precedence and invalid factory/from_parts args.
    /// @details Confirms EmptyInput > LengthMismatch > NonFiniteInput ordering plus InvalidConfig and
    /// InvalidHyperparameter checks across create() and from_parts().
    TEST(TransformTargetTest, RejectsInvalidInputsInOrder)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        // Empty and non-finite target arrays fail during fit_typed().
        EXPECT_EQ(IdentityTarget{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(StandardizeTarget{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(MinMaxTarget{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(
            StandardizeTarget{}.fit_typed(std::array<double, 2>{1.0, nan}).error().code(), GprErrorCode::NonFiniteInput
        );
        EXPECT_EQ(IdentityTarget{}.fit_typed(std::array<double, 1>{inf}).error().code(), GprErrorCode::NonFiniteInput);

        auto fitted = StandardizeTarget{}.fit_typed(std::array<double, 2>{1.0, 3.0});
        ASSERT_TRUE(fitted.has_value());

        std::array<double, 2> bad_y{1.0, nan};
        std::array<double, 1> short_out{};
        std::array<double, 2> good_out{};

        // EmptyInput -> LengthMismatch -> NonFiniteInput
        EXPECT_EQ(fitted->transform_into({}, short_out).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(fitted->transform_into(bad_y, short_out).error().code(), GprErrorCode::LengthMismatch);
        EXPECT_EQ(fitted->transform_into(bad_y, good_out).error().code(), GprErrorCode::NonFiniteInput);

        EXPECT_EQ(MinMaxTarget::create(1.0, 0.0).error().code(), GprErrorCode::InvalidConfig);
        EXPECT_EQ(MinMaxTarget::create(nan, 1.0).error().code(), GprErrorCode::InvalidConfig);

        EXPECT_EQ(FittedStandardizeTarget::from_parts(0.0, 0.0).error().code(), GprErrorCode::InvalidHyperparameter);
        EXPECT_EQ(FittedStandardizeTarget::from_parts(nan, 1.0).error().code(), GprErrorCode::NonFiniteInput);
        EXPECT_EQ(
            FittedMinMaxTarget::from_parts(2.0, 1.0, 0.0, 1.0).error().code(), GprErrorCode::InvalidHyperparameter
        );
    }

    /// @brief Verifies custom UnfittedTargetTransform / FittedTargetTransform polymorphism and roundtrips.
    /// @details Confirms default NotPersistable error from persist_state_json() and verifies
    /// transform_into() and inverse_transform_mean_into() center and reconstruct [1, 3].
    TEST(TransformTargetTest, CustomTargetInterfaceWorks)
    {
        UnfittedCenterY custom;
        // Custom target transforms without serialization overrides report PersistErrorKind::NotPersistable.
        EXPECT_EQ(custom.persist_state_json().error().persist_error_kind(), PersistErrorKind::NotPersistable);
        const std::array<double, 2> y{1.0, 3.0};
        auto fitted = custom.fit(y);
        ASSERT_TRUE(fitted.has_value());
        std::array<double, 2> out{};
        ASSERT_TRUE((*fitted)->transform_into(y, out).has_value());
        EXPECT_DOUBLE_EQ(out[0], -1.0);
        EXPECT_DOUBLE_EQ(out[1], 1.0);
        std::array<double, 2> rec{};
        ASSERT_TRUE((*fitted)->inverse_transform_mean_into(out, rec).has_value());
        EXPECT_EQ(rec, y);
    }

}  // namespace
