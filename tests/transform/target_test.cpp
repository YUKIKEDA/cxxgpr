// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/target.hpp"

#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>

// Spec: REQ-02, REQ-10, REQ-11, DES-01, DES-02, DES-18
namespace {

using cxxgpr::GprError;
using cxxgpr::GprErrorCode;
using cxxgpr::PersistErrorKind;
using cxxgpr::test_support::AuditMode;
using cxxgpr::test_support::NoAllocationScope;
using cxxgpr::transform::FittedMinMaxTarget;
using cxxgpr::transform::FittedStandardizeTarget;
using cxxgpr::transform::FittedTargetTransform;
using cxxgpr::transform::IdentityTarget;
using cxxgpr::transform::MinMaxTarget;
using cxxgpr::transform::StandardizeTarget;
using cxxgpr::transform::UnfittedTargetTransform;

constexpr double k_tol = 1e-12;

class FittedCenterY final : public FittedTargetTransform {
public:
    explicit FittedCenterY(double mean) : mean_(mean) {}

    [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override {
        if (y.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        for (double& v : y) {
            v -= mean_;
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean(std::span<double> mean) const override {
        if (mean.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        for (double& v : mean) {
            v += mean_;
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance(std::span<double> var) const override {
        if (var.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        return {};
    }

    [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override {
        return std::make_unique<FittedCenterY>(*this);
    }

private:
    double mean_ = 0.0;
};

class UnfittedCenterY final : public UnfittedTargetTransform {
public:
    [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
    fit(std::span<const double> y) const override {
        if (y.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        double sum = 0.0;
        for (const double v : y) {
            sum += v;
        }
        return std::make_unique<FittedCenterY>(sum / static_cast<double>(y.size()));
    }

    [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override {
        return std::make_unique<UnfittedCenterY>(*this);
    }
};

TEST(TransformTargetTest, IdentityLeavesTargetsAndPredictionsUnchanged) {
    const std::array<double, 3> orig{1.0, 2.0, 3.0};
    auto fitted = IdentityTarget{}.fit_typed(orig);
    ASSERT_TRUE(fitted.has_value());

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

TEST(TransformTargetTest, StandardizeCentersScalesAndInvertsMeanVarianceCovariance) {
    const std::array<double, 3> y{1.0, 3.0, 5.0};
    auto fitted = StandardizeTarget{}.fit_typed(y);
    ASSERT_TRUE(fitted.has_value());
    const double s = fitted->std();
    EXPECT_NEAR(fitted->mean(), 3.0, k_tol);
    EXPECT_NEAR(s, std::sqrt(8.0 / 3.0), k_tol);

    std::array<double, 3> z = y;
    ASSERT_TRUE(fitted->transform(z).has_value());
    EXPECT_NEAR((z[0] + z[1] + z[2]) / 3.0, 0.0, k_tol);
    EXPECT_NEAR(((z[0] * z[0]) + (z[1] * z[1]) + (z[2] * z[2])) / 3.0, 1.0, k_tol);

    ASSERT_TRUE(fitted->inverse_transform_mean(z).has_value());
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(z[i], y[i], k_tol);
    }

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

TEST(TransformTargetTest, ConstantTargetsUseUnitFallbackWithoutNaN) {
    const std::array<double, 3> y{4.0, 4.0, 4.0};
    auto std_fitted = StandardizeTarget{}.fit_typed(y);
    ASSERT_TRUE(std_fitted.has_value());
    EXPECT_DOUBLE_EQ(std_fitted->mean(), 4.0);
    EXPECT_DOUBLE_EQ(std_fitted->std(), 1.0);

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

TEST(TransformTargetTest, MinMaxScalesAndInvertsMeanAndVariance) {
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
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(rec[i], y[i], k_tol);
    }

    // data_span = 4, out_span = 2 -> scale = 2 -> scale^2 = 4
    const std::array<double, 2> var_in{1.0, 0.25};
    std::array<double, 2> var_out{};
    ASSERT_TRUE(fitted->inverse_transform_variance_into(var_in, var_out).has_value());
    EXPECT_NEAR(var_out[0], 4.0, k_tol);
    EXPECT_NEAR(var_out[1], 1.0, k_tol);
}

TEST(TransformTargetTest, HotPathZeroAllocationForBuiltins) {
    const std::array<double, 3> y{1.0, 3.0, 5.0};
    auto id = IdentityTarget{}.fit(y);
    auto st = StandardizeTarget{}.fit(y);
    auto mm = MinMaxTarget{}.fit(y);
    ASSERT_TRUE(id.has_value());
    ASSERT_TRUE(st.has_value());
    ASSERT_TRUE(mm.has_value());

    std::array<double, 3> mut_buf = y;
    std::array<double, 3> out_buf{};
    for (const auto* map : {id->get(), st->get(), mm->get()}) {
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

TEST(TransformTargetTest, RejectsInvalidInputsInOrder) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    EXPECT_EQ(IdentityTarget{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
    EXPECT_EQ(StandardizeTarget{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
    EXPECT_EQ(MinMaxTarget{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
    EXPECT_EQ(StandardizeTarget{}.fit_typed(std::array<double, 2>{1.0, nan}).error().code(),
              GprErrorCode::NonFiniteInput);
    EXPECT_EQ(IdentityTarget{}.fit_typed(std::array<double, 1>{inf}).error().code(),
              GprErrorCode::NonFiniteInput);

    auto fitted = StandardizeTarget{}.fit_typed(std::array<double, 2>{1.0, 3.0});
    ASSERT_TRUE(fitted.has_value());

    std::array<double, 2> bad_y{1.0, nan};
    std::array<double, 1> short_out{};
    std::array<double, 2> good_out{};

    // EmptyInput -> LengthMismatch -> NonFiniteInput
    EXPECT_EQ(fitted->transform_into({}, short_out).error().code(), GprErrorCode::EmptyInput);
    EXPECT_EQ(fitted->transform_into(bad_y, short_out).error().code(),
              GprErrorCode::LengthMismatch);
    EXPECT_EQ(fitted->transform_into(bad_y, good_out).error().code(), GprErrorCode::NonFiniteInput);

    EXPECT_EQ(MinMaxTarget::create(1.0, 0.0).error().code(), GprErrorCode::InvalidConfig);
    EXPECT_EQ(MinMaxTarget::create(nan, 1.0).error().code(), GprErrorCode::InvalidConfig);

    EXPECT_EQ(FittedStandardizeTarget::from_parts(0.0, 0.0).error().code(),
              GprErrorCode::InvalidHyperparameter);
    EXPECT_EQ(FittedStandardizeTarget::from_parts(nan, 1.0).error().code(),
              GprErrorCode::NonFiniteInput);
    EXPECT_EQ(FittedMinMaxTarget::from_parts(2.0, 1.0, 0.0, 1.0).error().code(),
              GprErrorCode::InvalidHyperparameter);
}

TEST(TransformTargetTest, CustomTargetInterfaceWorks) {
    UnfittedCenterY custom;
    EXPECT_EQ(custom.persist_state_json().error().persist_error_kind(),
              PersistErrorKind::NotPersistable);
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

} // namespace
