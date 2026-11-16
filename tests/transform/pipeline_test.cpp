// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/pipeline.hpp"

#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>

// Spec: REQ-02, REQ-03, REQ-10, REQ-11, DES-01, DES-02, DES-18
namespace {

using cxxgpr::GprErrorCode;
using cxxgpr::test_support::AuditMode;
using cxxgpr::test_support::NoAllocationScope;
using cxxgpr::transform::FittedPipeline;
using cxxgpr::transform::FittedTargetPipeline;
using cxxgpr::transform::MinMaxInput;
using cxxgpr::transform::MinMaxTarget;
using cxxgpr::transform::Pipeline;
using cxxgpr::transform::StandardizeInput;
using cxxgpr::transform::StandardizeTarget;
using cxxgpr::transform::TargetPipeline;

constexpr double k_tol = 1e-10;

TEST(TransformPipelineTest, EmptyInputPipelineIsIdentity) {
    const std::array<double, 4> orig{1.0, 2.0, 3.0, 4.0};
    const Pipeline empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0U);

    auto fitted = empty.fit_typed(orig, 2, 2);
    ASSERT_TRUE(fitted.has_value());
    EXPECT_TRUE(fitted->empty());

    std::array<double, 4> x = orig;
    ASSERT_TRUE(fitted->apply(x, 2, 2).has_value());
    EXPECT_EQ(x, orig);
    ASSERT_TRUE(fitted->inverse_apply(x, 2, 2).has_value());
    EXPECT_EQ(x, orig);
}

TEST(TransformPipelineTest, InputMinMaxThenStandardizeMatchesManualAndRoundTrips) {
    const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
    auto pipeline = Pipeline().then(MinMaxInput{}).then(StandardizeInput{});
    EXPECT_EQ(pipeline.size(), 2U);
    EXPECT_FALSE(pipeline.empty());

    auto fitted = pipeline.fit_typed(x, 3, 2);
    ASSERT_TRUE(fitted.has_value());
    EXPECT_EQ(fitted->size(), 2U);

    auto mm = MinMaxInput{}.fit_typed(x, 3, 2);
    ASSERT_TRUE(mm.has_value());
    std::array<double, 6> mid = x;
    ASSERT_TRUE(mm->apply(mid, 3, 2).has_value());
    auto st = StandardizeInput{}.fit_typed(mid, 3, 2);
    ASSERT_TRUE(st.has_value());
    std::array<double, 6> expected = mid;
    ASSERT_TRUE(st->apply(expected, 3, 2).has_value());

    std::array<double, 6> got = x;
    ASSERT_TRUE(fitted->apply(got, 3, 2).has_value());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], k_tol);
    }

    ASSERT_TRUE(fitted->inverse_apply(got, 3, 2).has_value());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(got[i], x[i], k_tol);
    }

    std::array<double, 6> out{};
    ASSERT_TRUE(fitted->apply_into(x, 3, 2, out).has_value());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], expected[i], k_tol);
    }
    std::array<double, 6> rec{};
    ASSERT_TRUE(fitted->inverse_apply_into(out, 3, 2, rec).has_value());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(rec[i], x[i], k_tol);
    }
}

TEST(TransformPipelineTest, EmptyTargetPipelineIsIdentity) {
    const std::array<double, 2> orig{1.0, 2.0};
    const TargetPipeline empty;
    EXPECT_TRUE(empty.empty());
    auto fitted = empty.fit_typed(orig);
    ASSERT_TRUE(fitted.has_value());
    EXPECT_TRUE(fitted->empty());

    std::array<double, 2> y = orig;
    ASSERT_TRUE(fitted->transform(y).has_value());
    EXPECT_EQ(y, orig);
    ASSERT_TRUE(fitted->inverse_transform_mean(y).has_value());
    EXPECT_EQ(y, orig);
    ASSERT_TRUE(fitted->inverse_transform_variance(y).has_value());
    EXPECT_EQ(y, orig);
    ASSERT_TRUE(fitted->inverse_transform_covariance(y).has_value());
    EXPECT_EQ(y, orig);
}

TEST(TransformPipelineTest, TargetMinMaxThenStandardizeRoundTrips) {
    const std::array<double, 3> y{1.0, 3.0, 5.0};
    auto pipeline = TargetPipeline().then(MinMaxTarget{}).then(StandardizeTarget{});
    EXPECT_EQ(pipeline.size(), 2U);

    auto fitted = pipeline.fit_typed(y);
    ASSERT_TRUE(fitted.has_value());

    auto mm = MinMaxTarget{}.fit_typed(y);
    ASSERT_TRUE(mm.has_value());
    std::array<double, 3> mid = y;
    ASSERT_TRUE(mm->transform(mid).has_value());
    auto st = StandardizeTarget{}.fit_typed(mid);
    ASSERT_TRUE(st.has_value());
    std::array<double, 3> expected = mid;
    ASSERT_TRUE(st->transform(expected).has_value());

    std::array<double, 3> got = y;
    ASSERT_TRUE(fitted->transform(got).has_value());
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], k_tol);
    }

    ASSERT_TRUE(fitted->inverse_transform_mean(got).has_value());
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(got[i], y[i], k_tol);
    }

    const double s = st->std();
    const double mm_scale = mm->max() - mm->min();
    const double scale2 = (s * mm_scale) * (s * mm_scale);

    std::array<double, 2> var{1.0, 0.25};
    ASSERT_TRUE(fitted->inverse_transform_variance(var).has_value());
    EXPECT_NEAR(var[0], 1.0 * scale2, k_tol);
    EXPECT_NEAR(var[1], 0.25 * scale2, k_tol);

    std::array<double, 2> cov_out{};
    ASSERT_TRUE(fitted->inverse_transform_covariance_into(std::array<double, 2>{1.0, 0.25}, cov_out)
                    .has_value());
    EXPECT_NEAR(cov_out[0], 1.0 * scale2, k_tol);
    EXPECT_NEAR(cov_out[1], 0.25 * scale2, k_tol);
}

TEST(TransformPipelineTest, CloneAndCopyPreserveSteps) {
    auto p = Pipeline().then(MinMaxInput{}).then(StandardizeInput{});
    Pipeline p_copy(p);
    p_copy.then(MinMaxInput{});
    EXPECT_EQ(p.size(), 2U);
    EXPECT_EQ(p_copy.size(), 3U);

    auto q = TargetPipeline().then(MinMaxTarget{}).then(StandardizeTarget{});
    TargetPipeline q_copy(q);
    q_copy.then(MinMaxTarget{});
    EXPECT_EQ(q.size(), 2U);
    EXPECT_EQ(q_copy.size(), 3U);

    auto fp = p.fit_typed(std::array<double, 4>{0.0, 1.0, 2.0, 3.0}, 2, 2);
    ASSERT_TRUE(fp.has_value());
    FittedPipeline fp_copy;
    fp_copy = *fp;
    EXPECT_EQ(fp_copy.size(), 2U);

    auto fq = q.fit_typed(std::array<double, 2>{1.0, 3.0});
    ASSERT_TRUE(fq.has_value());
    FittedTargetPipeline fq_copy;
    fq_copy = *fq;
    EXPECT_EQ(fq_copy.size(), 2U);
}

TEST(TransformPipelineTest, HotPathZeroAllocationForPipelines) {
    const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
    auto fp = Pipeline().then(MinMaxInput{}).then(StandardizeInput{}).fit_typed(x, 3, 2);
    ASSERT_TRUE(fp.has_value());

    std::array<double, 6> x_mut = x;
    std::array<double, 6> x_out{};
    {
        const NoAllocationScope audit(AuditMode::CppOnly);
        ASSERT_TRUE(fp->apply(x_mut, 3, 2).has_value());
        ASSERT_TRUE(fp->inverse_apply(x_mut, 3, 2).has_value());
        ASSERT_TRUE(fp->apply_into(x, 3, 2, x_out).has_value());
        ASSERT_TRUE(fp->inverse_apply_into(x_out, 3, 2, x_mut).has_value());
        EXPECT_EQ(audit.delta_allocations(), 0U);
    }

    const std::array<double, 3> y{1.0, 3.0, 5.0};
    auto fq = TargetPipeline().then(MinMaxTarget{}).then(StandardizeTarget{}).fit_typed(y);
    ASSERT_TRUE(fq.has_value());

    std::array<double, 3> y_mut = y;
    std::array<double, 3> y_out{};
    {
        const NoAllocationScope audit(AuditMode::CppOnly);
        ASSERT_TRUE(fq->transform(y_mut).has_value());
        ASSERT_TRUE(fq->inverse_transform_mean(y_mut).has_value());
        ASSERT_TRUE(fq->inverse_transform_variance(y_mut).has_value());
        ASSERT_TRUE(fq->inverse_transform_covariance(y_mut).has_value());
        ASSERT_TRUE(fq->transform_into(y, y_out).has_value());
        ASSERT_TRUE(fq->inverse_transform_mean_into(y_out, y_mut).has_value());
        ASSERT_TRUE(fq->inverse_transform_variance_into(y_out, y_mut).has_value());
        ASSERT_TRUE(fq->inverse_transform_covariance_into(y_out, y_mut).has_value());
        EXPECT_EQ(audit.delta_allocations(), 0U);
    }
}

TEST(TransformPipelineTest, RejectsInvalidInputsInOrder) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    EXPECT_EQ(Pipeline{}.fit_typed({}, 0, 1).error().code(), GprErrorCode::EmptyInput);
    EXPECT_EQ(Pipeline{}.fit_typed(std::array<double, 2>{1.0, nan}, 1, 2).error().code(),
              GprErrorCode::NonFiniteInput);

    EXPECT_EQ(TargetPipeline{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
    EXPECT_EQ(TargetPipeline{}.fit_typed(std::array<double, 1>{inf}).error().code(),
              GprErrorCode::NonFiniteInput);
}

} // namespace
