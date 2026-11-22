// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-02, REQ-03, REQ-10, REQ-11, DES-01, DES-02, DES-18
/// @file pipeline_test.cpp
/// @brief Unit tests for sequential input and target preprocessing pipelines.
/// @details Verifies empty identity pipelines, multi-stage MinMax->Standardize input and target
/// composition, exact roundtrip and variance/covariance scaling, deep copy/clone semantics,
/// zero-allocation hot paths, and input error ordering.

#include "cxxgpr/transform/pipeline.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "allocation_audit.hpp"
#include "small_stack.hpp"

namespace
{

    /// @brief Brings GprErrorCode into the anonymous test namespace.
    using cxxgpr::GprErrorCode;
    /// @brief Brings allocation AuditMode enum into the anonymous test namespace.
    using cxxgpr::test_support::AuditMode;
    /// @brief Brings RAII NoAllocationScope guard into the anonymous test namespace.
    using cxxgpr::test_support::NoAllocationScope;
    /// @brief Brings FittedPipeline into the anonymous test namespace.
    using cxxgpr::transform::FittedPipeline;
    /// @brief Brings FittedTargetPipeline into the anonymous test namespace.
    using cxxgpr::transform::FittedTargetPipeline;
    /// @brief Brings MinMaxInput into the anonymous test namespace.
    using cxxgpr::transform::MinMaxInput;
    /// @brief Brings MinMaxTarget into the anonymous test namespace.
    using cxxgpr::transform::MinMaxTarget;
    /// @brief Brings Pipeline into the anonymous test namespace.
    using cxxgpr::transform::Pipeline;
    /// @brief Brings StandardizeInput into the anonymous test namespace.
    using cxxgpr::transform::StandardizeInput;
    /// @brief Brings StandardizeTarget into the anonymous test namespace.
    using cxxgpr::transform::StandardizeTarget;
    /// @brief Brings TargetPipeline into the anonymous test namespace.
    using cxxgpr::transform::TargetPipeline;

    /// @brief Floating-point comparison tolerance used across pipeline assertions.
    constexpr double k_tol = 1e-10;

    /// @brief Verifies an empty input Pipeline fits to an identity transform leaving data untouched.
    /// @details Confirms empty() == true, size() == 0, and unchanged roundtrips through apply() and
    /// inverse_apply().
    TEST(TransformPipelineTest, EmptyInputPipelineIsIdentity)
    {
        const std::array<double, 4> orig{1.0, 2.0, 3.0, 4.0};
        const Pipeline empty;
        EXPECT_TRUE(empty.empty());
        EXPECT_EQ(empty.size(), 0U);

        // Fitting an empty pipeline produces an empty identity FittedPipeline.
        auto fitted = empty.fit_typed(orig, 2, 2);
        ASSERT_TRUE(fitted.has_value());
        EXPECT_TRUE(fitted->empty());

        std::array<double, 4> x = orig;
        ASSERT_TRUE(fitted->apply(x, 2, 2).has_value());
        EXPECT_EQ(x, orig);
        ASSERT_TRUE(fitted->inverse_apply(x, 2, 2).has_value());
        EXPECT_EQ(x, orig);
    }

    /// @brief Verifies sequential MinMaxInput->StandardizeInput fits stage-by-stage and roundtrips.
    /// @details Fits MinMaxInput on raw inputs and StandardizeInput on the MinMax output as an oracle,
    /// then checks in-place and out-of-place Pipeline apply/inverse parity.
    TEST(TransformPipelineTest, InputMinMaxThenStandardizeMatchesManualAndRoundTrips)
    {
        const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
        auto pipeline = Pipeline().then(MinMaxInput{}).then(StandardizeInput{});
        EXPECT_EQ(pipeline.size(), 2U);
        EXPECT_FALSE(pipeline.empty());

        auto fitted = pipeline.fit_typed(x, 3, 2);
        ASSERT_TRUE(fitted.has_value());
        EXPECT_EQ(fitted->size(), 2U);

        // Manually compose MinMax followed by Standardize as the ground-truth reference pipeline.
        auto mm = MinMaxInput{}.fit_typed(x, 3, 2);
        ASSERT_TRUE(mm.has_value());
        std::array<double, 6> mid = x;
        ASSERT_TRUE(mm->apply(mid, 3, 2).has_value());
        auto st = StandardizeInput{}.fit_typed(mid, 3, 2);
        ASSERT_TRUE(st.has_value());
        std::array<double, 6> expected = mid;
        ASSERT_TRUE(st->apply(expected, 3, 2).has_value());

        // Forward pipeline must match manual sequential application and invert in reverse order.
        std::array<double, 6> got = x;
        ASSERT_TRUE(fitted->apply(got, 3, 2).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(got[i], expected[i], k_tol);
        }

        ASSERT_TRUE(fitted->inverse_apply(got, 3, 2).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(got[i], x[i], k_tol);
        }

        std::array<double, 6> out{};
        ASSERT_TRUE(fitted->apply_into(x, 3, 2, out).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(out[i], expected[i], k_tol);
        }
        std::array<double, 6> rec{};
        ASSERT_TRUE(fitted->inverse_apply_into(out, 3, 2, rec).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(rec[i], x[i], k_tol);
        }
    }

    /// @brief Verifies an empty TargetPipeline leaves targets, means, variances, and covariances intact.
    /// @details Confirms empty() == true on both unfitted and fitted target pipelines and checks all
    /// four forward/inverse target operations are exact no-ops.
    TEST(TransformPipelineTest, EmptyTargetPipelineIsIdentity)
    {
        const std::array<double, 2> orig{1.0, 2.0};
        const TargetPipeline empty;
        EXPECT_TRUE(empty.empty());
        auto fitted = empty.fit_typed(orig);
        ASSERT_TRUE(fitted.has_value());
        EXPECT_TRUE(fitted->empty());

        // Every forward and inverse transform operation must preserve orig untouched.
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

    /// @brief Verifies TargetPipeline sequential MinMaxTarget->StandardizeTarget and Jacobian scaling.
    /// @details Checks sequential forward/inverse mean roundtrips and confirms predictive variance and
    /// covariance scale by the product Jacobian squared (s * (max - min))^2.
    TEST(TransformPipelineTest, TargetMinMaxThenStandardizeRoundTrips)
    {
        const std::array<double, 3> y{1.0, 3.0, 5.0};
        auto pipeline = TargetPipeline().then(MinMaxTarget{}).then(StandardizeTarget{});
        EXPECT_EQ(pipeline.size(), 2U);

        auto fitted = pipeline.fit_typed(y);
        ASSERT_TRUE(fitted.has_value());

        // Fit each stage sequentially to obtain the exact composite forward values and scale factors.
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
        for (std::size_t i = 0; i < y.size(); ++i)
        {
            EXPECT_NEAR(got[i], expected[i], k_tol);
        }

        ASSERT_TRUE(fitted->inverse_transform_mean(got).has_value());
        for (std::size_t i = 0; i < y.size(); ++i)
        {
            EXPECT_NEAR(got[i], y[i], k_tol);
        }

        // Composite inverse Jacobian dy/dz equals s * (max - min), so variance scales by (dy/dz)^2.
        const double s = st->std();
        const double mm_scale = mm->max() - mm->min();
        const double scale2 = (s * mm_scale) * (s * mm_scale);

        std::array<double, 2> var{1.0, 0.25};
        ASSERT_TRUE(fitted->inverse_transform_variance(var).has_value());
        EXPECT_NEAR(var[0], 1.0 * scale2, k_tol);
        EXPECT_NEAR(var[1], 0.25 * scale2, k_tol);

        std::array<double, 2> cov_out{};
        ASSERT_TRUE(fitted->inverse_transform_covariance_into(std::array<double, 2>{1.0, 0.25}, cov_out).has_value());
        EXPECT_NEAR(cov_out[0], 1.0 * scale2, k_tol);
        EXPECT_NEAR(cov_out[1], 0.25 * scale2, k_tol);
    }

    /// @brief Verifies copying unfitted and fitted pipelines preserves independent step sequences.
    /// @details Appends extra steps to copied Pipeline and TargetPipeline objects without mutating
    /// originals, and verifies copy assignment on FittedPipeline and FittedTargetPipeline.
    TEST(TransformPipelineTest, CloneAndCopyPreserveSteps)
    {
        auto p = Pipeline().then(MinMaxInput{}).then(StandardizeInput{});
        Pipeline p_copy(p);
        // Appending a third step to p_copy must leave p at size 2.
        p_copy.then(MinMaxInput{});
        EXPECT_EQ(p.size(), 2U);
        EXPECT_EQ(p_copy.size(), 3U);

        auto q = TargetPipeline().then(MinMaxTarget{}).then(StandardizeTarget{});
        TargetPipeline q_copy(q);
        q_copy.then(MinMaxTarget{});
        EXPECT_EQ(q.size(), 2U);
        EXPECT_EQ(q_copy.size(), 3U);

        // Verify fitted pipeline value copy assignment retains step counts cleanly.
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

    /// @brief Verifies FittedPipeline and FittedTargetPipeline execute with zero heap allocations.
    /// @details Audits all in-place and out-of-place methods on two-stage input and target pipelines
    /// inside AuditMode::CppOnly NoAllocationScope and asserts zero allocation deltas.
    TEST(TransformPipelineTest, HotPathZeroAllocationForPipelines)
    {
        const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
        auto fp = Pipeline().then(MinMaxInput{}).then(StandardizeInput{}).fit_typed(x, 3, 2);
        ASSERT_TRUE(fp.has_value());

        // Audit all four FittedPipeline forward/inverse methods inside NoAllocationScope.
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

        // Audit all eight FittedTargetPipeline mean/variance/covariance methods inside NoAllocationScope.
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

    /// @brief Negative controls verifying empty and non-finite inputs fail in order on pipelines.
    /// @details Confirms EmptyInput and NonFiniteInput rejections for both Pipeline and TargetPipeline.
    TEST(TransformPipelineTest, RejectsInvalidInputsInOrder)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        // Empty pipelines must still validate incoming spans before returning an identity fitted object.
        EXPECT_EQ(Pipeline{}.fit_typed({}, 0, 1).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(
            Pipeline{}.fit_typed(std::array<double, 2>{1.0, nan}, 1, 2).error().code(), GprErrorCode::NonFiniteInput
        );

        EXPECT_EQ(TargetPipeline{}.fit_typed({}).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(TargetPipeline{}.fit_typed(std::array<double, 1>{inf}).error().code(), GprErrorCode::NonFiniteInput);
    }

    /// @brief Verifies depth-128 nested Pipeline and TargetPipeline fit, apply, inverse, and destruction fit in small
    /// stack.
    TEST(TransformPipelineTest, DeepNestedPipelinesFitApplyAndDestroyInSmallStack)
    {
        // Spec: REQ-02, REQ-03
        const bool ran = cxxgpr::test_support::run_with_stack_size(
            cxxgpr::test_support::deep_model_stack_bytes(), []()
        {
            Pipeline pipe;
            pipe.then(StandardizeInput{});
            TargetPipeline tpipe;
            tpipe.then(StandardizeTarget{});
            for (int depth = 1; depth < 128; ++depth)
            {
                Pipeline outer;
                outer.then(pipe);
                pipe = std::move(outer);

                TargetPipeline t_outer;
                t_outer.then(tpipe);
                tpipe = std::move(t_outer);
            }

            std::array<double, 6> x{1.0, 2.0, 3.0, -1.0, 0.5, 2.5};
            const auto fitted_in = pipe.fit_typed(x, 3, 2);
            ASSERT_TRUE(fitted_in.has_value());
            std::array<double, 6> x_buf = x;
            ASSERT_TRUE(fitted_in->apply(x_buf, 3, 2).has_value());
            ASSERT_TRUE(fitted_in->inverse_apply(x_buf, 3, 2).has_value());
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                EXPECT_NEAR(x_buf[i], x[i], 1e-12);
            }

            std::array<double, 3> y{2.0, 4.0, 6.0};
            const auto fitted_tgt = tpipe.fit_typed(y);
            ASSERT_TRUE(fitted_tgt.has_value());
            std::array<double, 3> y_buf = y;
            ASSERT_TRUE(fitted_tgt->transform(y_buf).has_value());
            ASSERT_TRUE(fitted_tgt->inverse_transform_mean(y_buf).has_value());
            for (std::size_t i = 0; i < y.size(); ++i)
            {
                EXPECT_NEAR(y_buf[i], y[i], 1e-12);
            }
        }
        );
        ASSERT_TRUE(ran);
    }

    /// @brief Custom fitted input transform that optionally writes NaN without internal validation.
    class CustomFittedInput final : public cxxgpr::transform::FittedInputTransform
    {
    public:
        /// @brief Constructs the test transform.
        /// @param[in] emit_nan When true, writes quiet NaN into element 0; when false, acts as unvalidated identity.
        explicit CustomFittedInput(bool emit_nan) noexcept : emit_nan_(emit_nan) {}

        /// @brief Applies the transform in place.
        /// @param[in,out] x Feature matrix buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            std::span<double> x, std::size_t /*n_rows*/, std::size_t /*n_cols*/
        ) const override
        {
            if (emit_nan_)
            {
                x[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the transform out of place into out.
        /// @param[in] x Read-only input matrix buffer.
        /// @param[out] out Output matrix buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_into(
            std::span<const double> x, std::size_t /*n_rows*/, std::size_t /*n_cols*/, std::span<double> out
        ) const override
        {
            std::ranges::copy(x, out.begin());
            if (emit_nan_)
            {
                out[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse transform in place.
        /// @param[in,out] x Transformed feature matrix buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_apply(
            std::span<double> x, std::size_t /*n_rows*/, std::size_t /*n_cols*/
        ) const override
        {
            if (emit_nan_)
            {
                x[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse transform out of place into out.
        /// @param[in] x Read-only transformed feature matrix buffer.
        /// @param[out] out Output matrix buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_apply_into(
            std::span<const double> x, std::size_t /*n_rows*/, std::size_t /*n_cols*/, std::span<double> out
        ) const override
        {
            std::ranges::copy(x, out.begin());
            if (emit_nan_)
            {
                out[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Clones this transform.
        /// @return Owned deep copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::FittedInputTransform> clone() const override
        {
            return std::make_unique<CustomFittedInput>(*this);
        }

    private:
        /// @brief Whether to emit quiet NaN into element 0.
        bool emit_nan_{false};
    };

    /// @brief Custom unfitted input transform returning CustomFittedInput without validating inputs.
    class CustomUnfittedInput final : public cxxgpr::transform::UnfittedInputTransform
    {
    public:
        /// @brief Constructs the unfitted test transform.
        /// @param[in] emit_nan Passed to the fitted transform.
        explicit CustomUnfittedInput(bool emit_nan) noexcept : emit_nan_(emit_nan) {}

        /// @brief Returns a CustomFittedInput without inspecting x.
        /// @return Fitted transform pointer.
        [[nodiscard]] std::expected<std::unique_ptr<cxxgpr::transform::FittedInputTransform>, cxxgpr::GprError> fit(
            std::span<const double> /*x*/, std::size_t /*n_rows*/, std::size_t /*n_cols*/
        ) const override
        {
            return std::make_unique<CustomFittedInput>(emit_nan_);
        }

        /// @brief Clones this unfitted transform.
        /// @return Owned deep copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::UnfittedInputTransform> clone() const override
        {
            return std::make_unique<CustomUnfittedInput>(*this);
        }

    private:
        /// @brief Whether the fitted transform emits quiet NaN.
        bool emit_nan_{false};
    };

    /// @brief Custom fitted target transform that optionally writes NaN without internal validation.
    class CustomFittedTarget final : public cxxgpr::transform::FittedTargetTransform
    {
    public:
        /// @brief Constructs the test target transform.
        /// @param[in] emit_nan When true, writes quiet NaN into element 0; when false, acts as unvalidated identity.
        explicit CustomFittedTarget(bool emit_nan) noexcept : emit_nan_(emit_nan) {}

        /// @brief Applies the target transform in place.
        /// @param[in,out] y Target buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> transform(std::span<double> y) const override
        {
            if (emit_nan_)
            {
                y[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the target transform out of place into out.
        /// @param[in] y Read-only target buffer.
        /// @param[out] out Output buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> transform_into(
            std::span<const double> y, std::span<double> out
        ) const override
        {
            std::ranges::copy(y, out.begin());
            if (emit_nan_)
            {
                out[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse mean transform in place.
        /// @param[in,out] mean Predictive mean buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_mean(
            std::span<double> mean
        ) const override
        {
            if (emit_nan_)
            {
                mean[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse mean transform out of place into out.
        /// @param[in] mean Read-only predictive mean buffer.
        /// @param[out] out Output buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_mean_into(
            std::span<const double> mean, std::span<double> out
        ) const override
        {
            std::ranges::copy(mean, out.begin());
            if (emit_nan_)
            {
                out[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse variance transform in place.
        /// @param[in,out] var Predictive variance buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_variance(
            std::span<double> var
        ) const override
        {
            if (emit_nan_)
            {
                var[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse variance transform out of place into out.
        /// @param[in] var Read-only predictive variance buffer.
        /// @param[out] out Output buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_variance_into(
            std::span<const double> var, std::span<double> out
        ) const override
        {
            std::ranges::copy(var, out.begin());
            if (emit_nan_)
            {
                out[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse covariance transform in place.
        /// @param[in,out] cov Predictive covariance buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_covariance(
            std::span<double> cov
        ) const override
        {
            if (emit_nan_)
            {
                cov[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Applies the inverse covariance transform out of place into out.
        /// @param[in] cov Read-only predictive covariance buffer.
        /// @param[out] out Output buffer.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_covariance_into(
            std::span<const double> cov, std::span<double> out
        ) const override
        {
            std::ranges::copy(cov, out.begin());
            if (emit_nan_)
            {
                out[0] = std::numeric_limits<double>::quiet_NaN();
            }
            return {};
        }

        /// @brief Clones this target transform.
        /// @return Owned deep copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::FittedTargetTransform> clone() const override
        {
            return std::make_unique<CustomFittedTarget>(*this);
        }

    private:
        /// @brief Whether to emit quiet NaN into element 0.
        bool emit_nan_{false};
    };

    /// @brief Custom unfitted target transform returning CustomFittedTarget without validating inputs.
    class CustomUnfittedTarget final : public cxxgpr::transform::UnfittedTargetTransform
    {
    public:
        /// @brief Constructs the unfitted test target transform.
        /// @param[in] emit_nan Passed to the fitted target transform.
        explicit CustomUnfittedTarget(bool emit_nan) noexcept : emit_nan_(emit_nan) {}

        /// @brief Returns CustomFittedTarget without inspecting y.
        /// @return Fitted target transform pointer.
        [[nodiscard]] std::expected<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>, cxxgpr::GprError> fit(
            std::span<const double> /*y*/
        ) const override
        {
            return std::make_unique<CustomFittedTarget>(emit_nan_);
        }

        /// @brief Clones this unfitted target transform.
        /// @return Owned deep copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform> clone() const override
        {
            return std::make_unique<CustomUnfittedTarget>(*this);
        }

    private:
        /// @brief Whether the fitted transform emits quiet NaN.
        bool emit_nan_{false};
    };

    /// @brief Verifies nested pipeline boundaries still reject non-finite intermediate outputs after flattening.
    /// @details Guards both fit and apply/inverse paths when a custom step emits NaN before an empty or custom nested
    /// pipeline.
    TEST(TransformPipelineTest, NestedPipelineBoundariesRejectNonFiniteIntermediateOutputs)
    {
        // Spec: REQ-02, REQ-03, REQ-10
        const std::array<double, 4> x_src{1.0, 2.0, 3.0, 4.0};
        const std::array<double, 3> y_src{1.0, 2.0, 3.0};

        // 1. Fit stage: NaN emitted before an empty or passthrough nested pipeline must fail with NonFiniteInput.
        {
            const auto res_empty = Pipeline{}.then(CustomUnfittedInput{true}).then(Pipeline{}).fit_typed(x_src, 2, 2);
            ASSERT_FALSE(res_empty.has_value());
            EXPECT_EQ(res_empty.error().code(), GprErrorCode::NonFiniteInput);

            const auto res_nested = Pipeline{}
                                        .then(CustomUnfittedInput{true})
                                        .then(Pipeline{}.then(CustomUnfittedInput{false}))
                                        .fit_typed(x_src, 2, 2);
            ASSERT_FALSE(res_nested.has_value());
            EXPECT_EQ(res_nested.error().code(), GprErrorCode::NonFiniteInput);

            const auto tres_empty =
                TargetPipeline{}.then(CustomUnfittedTarget{true}).then(TargetPipeline{}).fit_typed(y_src);
            ASSERT_FALSE(tres_empty.has_value());
            EXPECT_EQ(tres_empty.error().code(), GprErrorCode::NonFiniteInput);

            const auto tres_nested = TargetPipeline{}
                                         .then(CustomUnfittedTarget{true})
                                         .then(TargetPipeline{}.then(CustomUnfittedTarget{false}))
                                         .fit_typed(y_src);
            ASSERT_FALSE(tres_nested.has_value());
            EXPECT_EQ(tres_nested.error().code(), GprErrorCode::NonFiniteInput);
        }

        // 2. FittedPipeline forward & inverse boundary validation.
        {
            auto make_input_pair = [](std::unique_ptr<cxxgpr::transform::FittedInputTransform> first,
                                       std::unique_ptr<cxxgpr::transform::FittedInputTransform> second)
            {
                std::vector<std::unique_ptr<cxxgpr::transform::FittedInputTransform>> steps;
                steps.push_back(std::move(first));
                steps.push_back(std::move(second));
                return FittedPipeline(std::move(steps));
            };

            std::vector<std::unique_ptr<cxxgpr::transform::FittedInputTransform>> pass_steps;
            pass_steps.push_back(std::make_unique<CustomFittedInput>(false));
            const FittedPipeline nested_pass(std::move(pass_steps));

            const FittedPipeline fwd_empty =
                make_input_pair(std::make_unique<CustomFittedInput>(true), std::make_unique<FittedPipeline>());
            const FittedPipeline fwd_pass =
                make_input_pair(std::make_unique<CustomFittedInput>(true), nested_pass.clone());

            for (const FittedPipeline* pipe : {&fwd_empty, &fwd_pass})
            {
                std::array<double, 4> x_mut = x_src;
                std::array<double, 4> x_out{};
                const auto r_apply = pipe->apply(x_mut, 2, 2);
                ASSERT_FALSE(r_apply.has_value());
                EXPECT_EQ(r_apply.error().code(), GprErrorCode::NonFiniteInput);

                const auto r_into = pipe->apply_into(x_src, 2, 2, x_out);
                ASSERT_FALSE(r_into.has_value());
                EXPECT_EQ(r_into.error().code(), GprErrorCode::NonFiniteInput);
            }

            const FittedPipeline inv_empty =
                make_input_pair(std::make_unique<FittedPipeline>(), std::make_unique<CustomFittedInput>(true));
            const FittedPipeline inv_pass =
                make_input_pair(nested_pass.clone(), std::make_unique<CustomFittedInput>(true));

            for (const FittedPipeline* pipe : {&inv_empty, &inv_pass})
            {
                std::array<double, 4> x_mut = x_src;
                std::array<double, 4> x_out{};
                const auto r_inv = pipe->inverse_apply(x_mut, 2, 2);
                ASSERT_FALSE(r_inv.has_value());
                EXPECT_EQ(r_inv.error().code(), GprErrorCode::NonFiniteInput);

                const auto r_inv_into = pipe->inverse_apply_into(x_src, 2, 2, x_out);
                ASSERT_FALSE(r_inv_into.has_value());
                EXPECT_EQ(r_inv_into.error().code(), GprErrorCode::NonFiniteInput);
            }

            // Nested-only empty FittedPipeline must still copy x into out on *_into calls.
            std::vector<std::unique_ptr<cxxgpr::transform::FittedInputTransform>> nested_empty_steps;
            nested_empty_steps.push_back(std::make_unique<FittedPipeline>());
            const FittedPipeline only_nested_empty(std::move(nested_empty_steps));
            std::array<double, 4> x_out{};
            ASSERT_TRUE(only_nested_empty.apply_into(x_src, 2, 2, x_out).has_value());
            EXPECT_EQ(x_out, x_src);
            x_out.fill(0.0);
            ASSERT_TRUE(only_nested_empty.inverse_apply_into(x_src, 2, 2, x_out).has_value());
            EXPECT_EQ(x_out, x_src);
        }

        // 3. FittedTargetPipeline forward & inverse boundary validation.
        {
            auto make_target_pair = [](std::unique_ptr<cxxgpr::transform::FittedTargetTransform> first,
                                        std::unique_ptr<cxxgpr::transform::FittedTargetTransform> second)
            {
                std::vector<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>> steps;
                steps.push_back(std::move(first));
                steps.push_back(std::move(second));
                return FittedTargetPipeline(std::move(steps));
            };

            std::vector<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>> pass_steps;
            pass_steps.push_back(std::make_unique<CustomFittedTarget>(false));
            const FittedTargetPipeline nested_pass(std::move(pass_steps));

            const FittedTargetPipeline fwd_empty =
                make_target_pair(std::make_unique<CustomFittedTarget>(true), std::make_unique<FittedTargetPipeline>());
            const FittedTargetPipeline fwd_pass =
                make_target_pair(std::make_unique<CustomFittedTarget>(true), nested_pass.clone());

            for (const FittedTargetPipeline* pipe : {&fwd_empty, &fwd_pass})
            {
                std::array<double, 3> y_mut = y_src;
                std::array<double, 3> y_out{};
                const auto r_tr = pipe->transform(y_mut);
                ASSERT_FALSE(r_tr.has_value());
                EXPECT_EQ(r_tr.error().code(), GprErrorCode::NonFiniteInput);

                const auto r_tr_into = pipe->transform_into(y_src, y_out);
                ASSERT_FALSE(r_tr_into.has_value());
                EXPECT_EQ(r_tr_into.error().code(), GprErrorCode::NonFiniteInput);
            }

            const FittedTargetPipeline inv_empty =
                make_target_pair(std::make_unique<FittedTargetPipeline>(), std::make_unique<CustomFittedTarget>(true));
            const FittedTargetPipeline inv_pass =
                make_target_pair(nested_pass.clone(), std::make_unique<CustomFittedTarget>(true));

            for (const FittedTargetPipeline* pipe : {&inv_empty, &inv_pass})
            {
                std::array<double, 3> y_mut = y_src;
                std::array<double, 3> y_out{};
                const auto r_inv = pipe->inverse_transform_mean(y_mut);
                ASSERT_FALSE(r_inv.has_value());
                EXPECT_EQ(r_inv.error().code(), GprErrorCode::NonFiniteInput);

                const auto r_inv_into = pipe->inverse_transform_mean_into(y_src, y_out);
                ASSERT_FALSE(r_inv_into.has_value());
                EXPECT_EQ(r_inv_into.error().code(), GprErrorCode::NonFiniteInput);
            }

            // Nested-only empty FittedTargetPipeline must still copy y into out on *_into calls.
            std::vector<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>> nested_empty_steps;
            nested_empty_steps.push_back(std::make_unique<FittedTargetPipeline>());
            const FittedTargetPipeline only_nested_empty(std::move(nested_empty_steps));
            std::array<double, 3> y_out{};
            ASSERT_TRUE(only_nested_empty.transform_into(y_src, y_out).has_value());
            EXPECT_EQ(y_out, y_src);
            y_out.fill(0.0);
            ASSERT_TRUE(only_nested_empty.inverse_transform_mean_into(y_src, y_out).has_value());
            EXPECT_EQ(y_out, y_src);
        }
    }

}  // namespace
