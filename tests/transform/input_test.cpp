// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-02, REQ-03, REQ-10, REQ-11, DES-01, DES-02, DES-18
/// @file input_test.cpp
/// @brief Unit tests for built-in and custom input feature preprocessing transforms.
/// @details Verifies IdentityInput, StandardizeInput, and MinMaxInput (with constant-column unit
/// fallback and custom feature ranges), IEEE 754 subnormal acceptance, zero-allocation hot paths,
/// strict validation error precedence, factory/from_parts negative controls, and custom extensions.

#include "cxxgpr/transform/input.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

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
    /// @brief Brings FittedIdentityInput into the anonymous test namespace.
    using cxxgpr::transform::FittedIdentityInput;
    /// @brief Brings FittedInputTransform base interface into the anonymous test namespace.
    using cxxgpr::transform::FittedInputTransform;
    /// @brief Brings FittedMinMaxInput into the anonymous test namespace.
    using cxxgpr::transform::FittedMinMaxInput;
    /// @brief Brings FittedStandardizeInput into the anonymous test namespace.
    using cxxgpr::transform::FittedStandardizeInput;
    /// @brief Brings IdentityInput into the anonymous test namespace.
    using cxxgpr::transform::IdentityInput;
    /// @brief Brings MinMaxInput into the anonymous test namespace.
    using cxxgpr::transform::MinMaxInput;
    /// @brief Brings StandardizeInput into the anonymous test namespace.
    using cxxgpr::transform::StandardizeInput;
    /// @brief Brings UnfittedInputTransform base interface into the anonymous test namespace.
    using cxxgpr::transform::UnfittedInputTransform;

    /// @brief Floating-point comparison tolerance used across input transform tests.
    constexpr double k_tol = 1e-12;

    /// @brief Test fitted custom input transform scaling features by 2.0 and recording expected_cols_.
    /// @details Exercises virtual FittedInputTransform dispatch, dimension checks, and default
    /// apply_into/inverse_apply_into fallbacks.
    class FittedTimesTwo final : public FittedInputTransform
    {
    public:
        /// @brief Constructs a fitted scaling transform bound to expected_cols feature columns.
        /// @details Stores expected_cols so apply() and inverse_apply() enforce dimension matching.
        /// @param expected_cols Expected number of feature columns.
        explicit FittedTimesTwo(std::size_t expected_cols) : expected_cols_(expected_cols) {}

        /// @brief Multiplies all feature entries by 2.0 after checking empty and column count.
        /// @details Returns EmptyInput on empty shapes and DimensionMismatch when n_cols != expected_cols_.
        /// @param x Mutable column-major feature matrix span.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or validation error on invalid dimensions.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (n_rows == 0 || n_cols == 0 || x.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            if (n_cols != expected_cols_)
            {
                return std::unexpected(GprError::dimension_mismatch(n_cols, expected_cols_));
            }
            for (double& v : x)
            {
                v *= 2.0;
            }
            return {};
        }

        /// @brief Halves all feature entries to invert apply().
        /// @details Returns EmptyInput on empty shapes and DimensionMismatch when n_cols != expected_cols_.
        /// @param x Mutable column-major feature matrix span.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or validation error on invalid dimensions.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (n_rows == 0 || n_cols == 0 || x.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            if (n_cols != expected_cols_)
            {
                return std::unexpected(GprError::dimension_mismatch(n_cols, expected_cols_));
            }
            for (double& v : x)
            {
                v *= 0.5;
            }
            return {};
        }

        /// @brief Clones this custom fitted transform into a new unique_ptr.
        /// @details Preserves expected_cols_ on the duplicated instance.
        /// @return Heap-allocated copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override
        {
            return std::make_unique<FittedTimesTwo>(*this);
        }

        /// @brief Returns the test persistence identifier for this custom transform.
        /// @details Used to verify default NotPersistable JSON rejection on custom subclasses.
        /// @return Static persistence identifier string "test.times_two".
        [[nodiscard]] std::string_view persist_id() const override { return "test.times_two"; }

    private:
        /// @brief Expected feature column dimension recorded during fit().
        std::size_t expected_cols_ = 0;
    };

    /// @brief Test unfitted custom input transform producing FittedTimesTwo instances.
    /// @details Captures training column count during fit() and leaves persist_state_json() at default.
    class UnfittedTimesTwo final : public UnfittedInputTransform
    {
    public:
        /// @brief Fits FittedTimesTwo by recording n_cols after guarding against empty inputs.
        /// @details Returns EmptyInput if x is empty or either extent is zero.
        /// @param x Training feature matrix span.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of training feature columns.
        /// @return FittedTimesTwo wrapped in unique_ptr, or EmptyInput on degenerate inputs.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (n_rows == 0 || n_cols == 0 || x.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            return std::make_unique<FittedTimesTwo>(n_cols);
        }

        /// @brief Clones this unfitted custom transform into a new unique_ptr.
        /// @details Enables polymorphic copying inside transform containers.
        /// @return Heap-allocated copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override
        {
            return std::make_unique<UnfittedTimesTwo>(*this);
        }

        /// @brief Returns the test persistence identifier for this custom transform.
        /// @details Matches FittedTimesTwo::persist_id() for custom extension tests.
        /// @return Static persistence identifier string "test.times_two".
        [[nodiscard]] std::string_view persist_id() const override { return "test.times_two"; }
    };

    /// @brief Verifies IdentityInput leaves feature values identical across apply and inverse_apply.
    /// @details Checks in-place apply() as well as out-of-place apply_into() and inverse_apply_into().
    TEST(TransformInputTest, IdentityLeavesFeaturesUnchanged)
    {
        const std::array<double, 4> orig{1.0, 2.0, 3.0, 4.0};
        auto fitted = IdentityInput{}.fit_typed(orig, 2, 2);
        ASSERT_TRUE(fitted.has_value());

        // Forward and inverse transforms must preserve every entry verbatim.
        std::array<double, 4> x = orig;
        ASSERT_TRUE(fitted->apply(x, 2, 2).has_value());
        EXPECT_DOUBLE_EQ(x[0], 1.0);
        EXPECT_DOUBLE_EQ(x[3], 4.0);

        std::array<double, 4> out{};
        ASSERT_TRUE(fitted->apply_into(orig, 2, 2, out).has_value());
        EXPECT_EQ(out, orig);
        ASSERT_TRUE(fitted->inverse_apply_into(orig, 2, 2, out).has_value());
        EXPECT_EQ(out, orig);
    }

    /// @brief Verifies StandardizeInput estimates per-column population mean/std with constant fallback.
    /// @details Confirms population std = sqrt(8/3) on [0, 2, 4], std fallback 1.0 on constant [5, 5, 5],
    /// zero-mean centering after apply(), and exact inverse_apply() recovery.
    TEST(TransformInputTest, StandardizeEachColumnWithConstantFallback)
    {
        // Column 0: [0, 2, 4] -> mean 2, pop std sqrt(8/3)
        // Column 1: [5, 5, 5] -> mean 5, constant fallback std = 1.0
        const std::array<double, 6> x{0.0, 2.0, 4.0, 5.0, 5.0, 5.0};
        auto fitted = StandardizeInput{}.fit_typed(x, 3, 2);
        ASSERT_TRUE(fitted.has_value());
        ASSERT_EQ(fitted->mean().size(), 2U);
        ASSERT_EQ(fitted->std().size(), 2U);
        EXPECT_NEAR(fitted->mean()[0], 2.0, k_tol);
        EXPECT_NEAR(fitted->std()[0], std::sqrt(8.0 / 3.0), k_tol);
        EXPECT_NEAR(fitted->mean()[1], 5.0, k_tol);
        EXPECT_DOUBLE_EQ(fitted->std()[1], 1.0);

        // Apply maps column 0 to zero mean and constant column 1 to zero without NaN.
        std::array<double, 6> z = x;
        ASSERT_TRUE(fitted->apply(z, 3, 2).has_value());
        EXPECT_NEAR((z[0] + z[1] + z[2]) / 3.0, 0.0, k_tol);
        EXPECT_NEAR(z[3], 0.0, k_tol);
        EXPECT_NEAR(z[4], 0.0, k_tol);
        EXPECT_NEAR(z[5], 0.0, k_tol);

        ASSERT_TRUE(fitted->inverse_apply(z, 3, 2).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(z[i], x[i], k_tol);
        }
    }

    /// @brief Verifies MinMaxInput default [0, 1] and custom [-1, 3] scaling with constant-column fallback.
    /// @details Confirms constant column [7, 7, 7] maps to the lower bound (0.0 or -1.0) without
    /// division by zero and inverse_apply_into() reconstructs original values accurately.
    TEST(TransformInputTest, MinMaxDefaultAndCustomRangeWithConstantFallback)
    {
        const std::array<double, 6> x{0.0, 2.0, 4.0, 7.0, 7.0, 7.0};
        auto fitted = MinMaxInput{}.fit_typed(x, 3, 2);
        ASSERT_TRUE(fitted.has_value());
        EXPECT_EQ(fitted->feature_range(), (std::pair<double, double>{0.0, 1.0}));
        EXPECT_DOUBLE_EQ(fitted->min()[0], 0.0);
        EXPECT_DOUBLE_EQ(fitted->max()[0], 4.0);
        EXPECT_DOUBLE_EQ(fitted->min()[1], 7.0);
        EXPECT_DOUBLE_EQ(fitted->max()[1], 7.0);

        // Constant column 1 maps to feature_range.first (0.0) instead of dividing by max - min = 0.
        std::array<double, 6> z = x;
        ASSERT_TRUE(fitted->apply(z, 3, 2).has_value());
        EXPECT_NEAR(z[0], 0.0, k_tol);
        EXPECT_NEAR(z[1], 0.5, k_tol);
        EXPECT_NEAR(z[2], 1.0, k_tol);
        EXPECT_NEAR(z[3], 0.0, k_tol);
        EXPECT_NEAR(z[4], 0.0, k_tol);
        EXPECT_NEAR(z[5], 0.0, k_tol);

        // Custom range [-1, 3] stretches [0, 4] 1:1 with -1 offset and recovers exact inputs.
        auto custom_spec = MinMaxInput::create(-1.0, 3.0);
        ASSERT_TRUE(custom_spec.has_value());
        auto custom_fitted = custom_spec->fit_typed(x, 3, 2);
        ASSERT_TRUE(custom_fitted.has_value());
        std::array<double, 6> out{};
        ASSERT_TRUE(custom_fitted->apply_into(x, 3, 2, out).has_value());
        EXPECT_NEAR(out[0], -1.0, k_tol);
        EXPECT_NEAR(out[1], 1.0, k_tol);
        EXPECT_NEAR(out[2], 3.0, k_tol);
        EXPECT_NEAR(out[3], -1.0, k_tol);

        std::array<double, 6> recovered{};
        ASSERT_TRUE(custom_fitted->inverse_apply_into(out, 3, 2, recovered).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(recovered[i], x[i], k_tol);
        }
    }

    /// @brief Verifies IEEE 754 subnormal inputs pass finite validation and transform cleanly.
    /// @details Fits and applies StandardizeInput on denorm_min() values and confirms finite outputs.
    TEST(TransformInputTest, SubnormalValuesAreAcceptedAsFinite)
    {
        // Positive IEEE 754 subnormals must pass std::isfinite() and yield finite standardized features.
        const double sub = std::numeric_limits<double>::denorm_min();
        const std::array<double, 2> x{sub, 2.0 * sub};
        auto std_fitted = StandardizeInput{}.fit_typed(x, 2, 1);
        ASSERT_TRUE(std_fitted.has_value());
        std::array<double, 2> out{};
        ASSERT_TRUE(std_fitted->apply_into(x, 2, 1, out).has_value());
        EXPECT_TRUE(std::isfinite(out[0]));
        EXPECT_TRUE(std::isfinite(out[1]));
    }

    /// @brief Verifies built-in Identity, Standardize, and MinMax transforms allocate zero heap memory.
    /// @details Executes all four forward/inverse methods on each fitted map inside AuditMode::CppOnly
    /// NoAllocationScope and checks delta_allocations() == 0.
    TEST(TransformInputTest, HotPathZeroAllocationForBuiltins)
    {
        const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
        auto id = IdentityInput{}.fit(x, 3, 2);
        auto st = StandardizeInput{}.fit(x, 3, 2);
        auto mm = MinMaxInput{}.fit(x, 3, 2);
        ASSERT_TRUE(id.has_value());
        ASSERT_TRUE(st.has_value());
        ASSERT_TRUE(mm.has_value());

        // Audit every fitted built-in input map across all four virtual operations.
        std::array<double, 6> mut_buf = x;
        std::array<double, 6> out_buf{};
        for (const auto* map : {id->get(), st->get(), mm->get()})
        {
            const NoAllocationScope audit(AuditMode::CppOnly);
            ASSERT_TRUE(map->apply(mut_buf, 3, 2).has_value());
            ASSERT_TRUE(map->inverse_apply(mut_buf, 3, 2).has_value());
            ASSERT_TRUE(map->apply_into(x, 3, 2, out_buf).has_value());
            ASSERT_TRUE(map->inverse_apply_into(out_buf, 3, 2, mut_buf).has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }
    }

    /// @brief Negative controls verifying FittedStandardizeInput enforces strict validation error precedence.
    /// @details Exercises compound violations across EmptyInput, SizeOverflow, LengthMismatch (x and out),
    /// DimensionMismatch, and NonFiniteInput in strict descending priority.
    TEST(TransformInputTest, ValidationEnforcesStrictTotalOrder)
    {
        const std::array<double, 4> train{0.0, 1.0, 2.0, 3.0};
        auto st = StandardizeInput{}.fit_typed(train, 2, 2);
        ASSERT_TRUE(st.has_value());

        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::array<double, 4> nan_buf{nan, 1.0, 2.0, 3.0};
        std::array<double, 2> short_buf{nan, 1.0};
        std::array<double, 4> out_buf{};
        std::array<double, 2> bad_out{};

        // 1. EmptyInput beats SizeOverflow, LengthMismatch, DimensionMismatch, NonFiniteInput
        auto r_empty = st->apply(short_buf, 0, 1);
        ASSERT_FALSE(r_empty.has_value());
        EXPECT_EQ(r_empty.error().code(), GprErrorCode::EmptyInput);

        // 2. SizeOverflow beats LengthMismatch, DimensionMismatch, NonFiniteInput
        const std::size_t huge = (std::numeric_limits<std::size_t>::max() / 2U) + 1U;
        auto r_ovf = st->apply(nan_buf, huge, 3);
        ASSERT_FALSE(r_ovf.has_value());
        EXPECT_EQ(r_ovf.error().code(), GprErrorCode::SizeOverflow);

        // 3. LengthMismatch (x) beats DimensionMismatch and NonFiniteInput
        auto r_len_x = st->apply(nan_buf, 3, 1);
        ASSERT_FALSE(r_len_x.has_value());
        EXPECT_EQ(r_len_x.error().code(), GprErrorCode::LengthMismatch);

        // 4. LengthMismatch (out) beats DimensionMismatch and NonFiniteInput
        auto r_len_out = st->apply_into(nan_buf, 4, 1, bad_out);
        ASSERT_FALSE(r_len_out.has_value());
        EXPECT_EQ(r_len_out.error().code(), GprErrorCode::LengthMismatch);

        // 5. DimensionMismatch beats NonFiniteInput
        auto r_dim = st->apply_into(nan_buf, 4, 1, out_buf);
        ASSERT_FALSE(r_dim.has_value());
        EXPECT_EQ(r_dim.error().code(), GprErrorCode::DimensionMismatch);
        EXPECT_EQ(r_dim.error().x_dim(), 1U);
        EXPECT_EQ(r_dim.error().expected_dim(), 2U);

        // 6. NonFiniteInput when shape and dimension match
        auto r_nan = st->apply_into(nan_buf, 2, 2, out_buf);
        ASSERT_FALSE(r_nan.has_value());
        EXPECT_EQ(r_nan.error().code(), GprErrorCode::NonFiniteInput);
    }

    /// @brief Negative controls verifying MinMaxInput::create() and from_parts() reject invalid arguments.
    /// @details Checks equal/inverted/non-finite MinMax ranges, empty/mismatched/non-finite/non-positive
    /// FittedStandardizeInput parts, and inverted/mismatched FittedMinMaxInput parts.
    TEST(TransformInputTest, MinMaxCreateAndFromPartsRejectInvalidInputs)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        // Degenerate, inverted, or non-finite feature ranges must fail with InvalidConfig.
        EXPECT_EQ(MinMaxInput::create(1.0, 1.0).error().code(), GprErrorCode::InvalidConfig);
        EXPECT_EQ(MinMaxInput::create(2.0, 1.0).error().code(), GprErrorCode::InvalidConfig);
        EXPECT_EQ(MinMaxInput::create(nan, 1.0).error().code(), GprErrorCode::InvalidConfig);
        EXPECT_EQ(MinMaxInput::create(0.0, inf).error().code(), GprErrorCode::InvalidConfig);

        // Reconstructing fitted statistics from parts must reject empty, mismatched, NaN, and std <= 0.
        EXPECT_EQ(FittedStandardizeInput::from_parts({}, {}).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(FittedStandardizeInput::from_parts({0.0}, {1.0, 2.0}).error().code(), GprErrorCode::LengthMismatch);
        EXPECT_EQ(FittedStandardizeInput::from_parts({nan}, {1.0}).error().code(), GprErrorCode::NonFiniteInput);
        EXPECT_EQ(FittedStandardizeInput::from_parts({0.0}, {0.0}).error().code(), GprErrorCode::InvalidHyperparameter);

        EXPECT_EQ(FittedMinMaxInput::from_parts({}, {}, 0.0, 1.0).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(
            FittedMinMaxInput::from_parts({0.0}, {1.0, 2.0}, 0.0, 1.0).error().code(), GprErrorCode::LengthMismatch
        );
        EXPECT_EQ(
            FittedMinMaxInput::from_parts({0.0, 3.0}, {1.0, 2.0}, 0.0, 1.0).error().code(),
            GprErrorCode::InvalidHyperparameter
        );
        EXPECT_TRUE(FittedMinMaxInput::from_parts({0.0, 2.0}, {1.0, 2.0}, 0.0, 1.0).has_value());
    }

    /// @brief Verifies user-defined UnfittedInputTransform and default apply_into/persist fallbacks.
    /// @details Checks persist_id(), default NotPersistable error from persist_state_json(), and
    /// polymorphic fit() + default out-of-place apply_into()/inverse_apply_into() roundtrip.
    TEST(TransformInputTest, CustomTransformInterfaceAndDefaultIntoFallback)
    {
        UnfittedTimesTwo custom;
        EXPECT_EQ(custom.persist_id(), "test.times_two");
        // Custom transforms without explicit persistence support report PersistErrorKind::NotPersistable.
        auto p_err = custom.persist_state_json();
        ASSERT_FALSE(p_err.has_value());
        EXPECT_EQ(p_err.error().code(), GprErrorCode::PersistFailed);
        EXPECT_EQ(p_err.error().persist_error_kind(), PersistErrorKind::NotPersistable);

        // Verify base-class apply_into/inverse_apply_into fallbacks copy and delegate properly.
        const std::array<double, 4> x{1.0, 2.0, 3.0, 4.0};
        auto fitted = custom.fit(x, 2, 2);
        ASSERT_TRUE(fitted.has_value());
        std::array<double, 4> out{};
        ASSERT_TRUE((*fitted)->apply_into(x, 2, 2, out).has_value());
        EXPECT_DOUBLE_EQ(out[0], 2.0);
        EXPECT_DOUBLE_EQ(out[3], 8.0);
        std::array<double, 4> rec{};
        ASSERT_TRUE((*fitted)->inverse_apply_into(out, 2, 2, rec).has_value());
        EXPECT_EQ(rec, x);
    }

}  // namespace
