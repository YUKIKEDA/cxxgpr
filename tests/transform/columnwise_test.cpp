// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-02, REQ-03, REQ-10, REQ-11, DES-01, DES-02, DES-18
/// @file columnwise_test.cpp
/// @brief Unit tests for ColumnwiseInput and FittedColumnwiseInput per-column feature mappings.
/// @details Verifies per-column MinMax/Standardize/Pipeline/custom transform composition, exact
/// roundtrip inversion, zero-allocation hot-path execution, strict input validation order, and
/// deep-copy clone independence.

#include "cxxgpr/transform/columnwise.hpp"

#include "cxxgpr/transform/pipeline.hpp"

#include <gtest/gtest.h>

#include <array>
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
    /// @brief Brings allocation AuditMode enum into the anonymous test namespace.
    using cxxgpr::test_support::AuditMode;
    /// @brief Brings RAII NoAllocationScope guard into the anonymous test namespace.
    using cxxgpr::test_support::NoAllocationScope;
    /// @brief Brings ColumnwiseInput specification builder into the anonymous test namespace.
    using cxxgpr::transform::ColumnwiseInput;
    /// @brief Brings FittedColumnwiseInput into the anonymous test namespace.
    using cxxgpr::transform::FittedColumnwiseInput;
    /// @brief Brings FittedInputTransform base interface into the anonymous test namespace.
    using cxxgpr::transform::FittedInputTransform;
    /// @brief Brings IdentityInput transform into the anonymous test namespace.
    using cxxgpr::transform::IdentityInput;
    /// @brief Brings MinMaxInput transform into the anonymous test namespace.
    using cxxgpr::transform::MinMaxInput;
    /// @brief Brings Pipeline transform composer into the anonymous test namespace.
    using cxxgpr::transform::Pipeline;
    /// @brief Brings StandardizeInput transform into the anonymous test namespace.
    using cxxgpr::transform::StandardizeInput;
    /// @brief Brings UnfittedInputTransform base interface into the anonymous test namespace.
    using cxxgpr::transform::UnfittedInputTransform;

    /// @brief Floating-point comparison tolerance used across columnwise transform assertions.
    constexpr double k_tol = 1e-10;

    /// @brief Test fitted input transform multiplying feature columns by 2.0 and halving on inverse.
    /// @details Used to verify ColumnwiseInput dispatches custom FittedInputTransform implementations per column.
    class FittedTimesTwoCol final : public FittedInputTransform
    {
    public:
        /// @brief Multiplies all entries of x by 2.0 in place.
        /// @details Rejects empty spans or zero dimensions with GprErrorCode::EmptyInput.
        /// @param x Mutable column-major feature matrix span.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or EmptyInput on degenerate inputs.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (n_rows == 0 || n_cols == 0 || x.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            for (double& v : x)
            {
                v *= 2.0;
            }
            return {};
        }

        /// @brief Halves all entries of x in place to invert apply().
        /// @details Rejects empty spans or zero dimensions with GprErrorCode::EmptyInput.
        /// @param x Mutable column-major feature matrix span.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or EmptyInput on degenerate inputs.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (n_rows == 0 || n_cols == 0 || x.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            for (double& v : x)
            {
                v *= 0.5;
            }
            return {};
        }

        /// @brief Clones this custom fitted column transform into a new unique_ptr.
        /// @details Supports value-semantic copying of ColumnwiseInput / FittedColumnwiseInput.
        /// @return Heap-allocated copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override
        {
            return std::make_unique<FittedTimesTwoCol>(*this);
        }
    };

    /// @brief Test unfitted input transform factory producing FittedTimesTwoCol instances.
    /// @details Used to test ColumnwiseInput::then() with user-defined UnfittedInputTransform subclasses.
    class UnfittedTimesTwoCol final : public UnfittedInputTransform
    {
    public:
        /// @brief Produces a stateless FittedTimesTwoCol instance after checking for empty inputs.
        /// @details Rejects empty spans or zero dimensions with GprErrorCode::EmptyInput.
        /// @param x Training feature matrix span.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return FittedTimesTwoCol wrapped in unique_ptr, or EmptyInput on degenerate inputs.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (n_rows == 0 || n_cols == 0 || x.empty())
            {
                return std::unexpected(GprError::empty_input());
            }
            return std::make_unique<FittedTimesTwoCol>();
        }

        /// @brief Clones this unfitted transform specification into a new unique_ptr.
        /// @details Supports copying ColumnwiseInput before fitting.
        /// @return Heap-allocated copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override
        {
            return std::make_unique<UnfittedTimesTwoCol>(*this);
        }
    };

    /// @brief Verifies ColumnwiseInput fits and applies distinct per-column transforms independently.
    /// @details Assigns MinMaxInput to column 0 and StandardizeInput to column 1, checks parity with
    /// single-column reference fits, and verifies both in-place and out-of-place roundtrips.
    TEST(TransformColumnwiseTest, MinMaxAndStandardizeEachColumn)
    {
        const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
        auto spec = ColumnwiseInput().then(MinMaxInput{}).then(StandardizeInput{});
        EXPECT_EQ(spec.size(), 2U);

        auto fitted = spec.fit_typed(x, 3, 2);
        ASSERT_TRUE(fitted.has_value());
        EXPECT_EQ(fitted->size(), 2U);

        // Fit each single-column transform independently as the ground-truth oracle.
        auto mm = MinMaxInput{}.fit_typed(std::span<const double>(x).subspan(0, 3), 3, 1);
        auto st = StandardizeInput{}.fit_typed(std::span<const double>(x).subspan(3, 3), 3, 1);
        ASSERT_TRUE(mm.has_value());
        ASSERT_TRUE(st.has_value());

        std::array<double, 6> expected = x;
        ASSERT_TRUE(mm->apply(std::span<double>(expected).subspan(0, 3), 3, 1).has_value());
        ASSERT_TRUE(st->apply(std::span<double>(expected).subspan(3, 3), 3, 1).has_value());

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

        // Confirm out-of-place *_into overloads produce identical forward and inverse values.
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

    /// @brief Verifies custom transforms and nested Pipeline objects execute correctly per column.
    /// @details Pairs IdentityInput with UnfittedTimesTwoCol and also nests a MinMax+Standardize
    /// Pipeline on column 0 alongside IdentityInput on column 1.
    TEST(TransformColumnwiseTest, CustomAndPipelineMapsWorkPerColumn)
    {
        const std::array<double, 4> x{1.0, 2.0, 3.0, 4.0};
        auto col_pipe = Pipeline().then(MinMaxInput{}).then(StandardizeInput{});
        auto spec = ColumnwiseInput().then(IdentityInput{}).then(UnfittedTimesTwoCol{});
        auto fitted = spec.fit_typed(x, 2, 2);
        ASSERT_TRUE(fitted.has_value());

        // Column 0 stays [1, 2]; column 1 doubles to [6, 8] then inverts back to [3, 4].
        std::array<double, 4> z = x;
        ASSERT_TRUE(fitted->apply(z, 2, 2).has_value());
        EXPECT_NEAR(z[0], 1.0, k_tol);
        EXPECT_NEAR(z[1], 2.0, k_tol);
        EXPECT_NEAR(z[2], 6.0, k_tol);
        EXPECT_NEAR(z[3], 8.0, k_tol);

        ASSERT_TRUE(fitted->inverse_apply(z, 2, 2).has_value());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(z[i], x[i], k_tol);
        }

        // Also verify Pipeline inside ColumnwiseInput
        auto nested = ColumnwiseInput().then(col_pipe).then(IdentityInput{}).fit_typed(x, 2, 2);
        ASSERT_TRUE(nested.has_value());
        std::array<double, 4> nz = x;
        ASSERT_TRUE(nested->apply(nz, 2, 2).has_value());
        EXPECT_NEAR(nz[0], -1.0, k_tol);
        EXPECT_NEAR(nz[1], 1.0, k_tol);
        EXPECT_NEAR(nz[2], 3.0, k_tol);
        EXPECT_NEAR(nz[3], 4.0, k_tol);
    }

    /// @brief Verifies FittedColumnwiseInput apply/inverse methods perform zero heap allocations.
    /// @details Runs apply, inverse_apply, apply_into, and inverse_apply_into inside AuditMode::CppOnly
    /// NoAllocationScope and checks delta_allocations() == 0.
    TEST(TransformColumnwiseTest, HotPathZeroAllocation)
    {
        const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
        auto fitted = ColumnwiseInput().then(MinMaxInput{}).then(StandardizeInput{}).fit_typed(x, 3, 2);
        ASSERT_TRUE(fitted.has_value());

        // Audit all four forward/inverse in-place and out-of-place methods under NoAllocationScope.
        std::array<double, 6> mut_buf = x;
        std::array<double, 6> out_buf{};
        {
            const NoAllocationScope audit(AuditMode::CppOnly);
            ASSERT_TRUE(fitted->apply(mut_buf, 3, 2).has_value());
            ASSERT_TRUE(fitted->inverse_apply(mut_buf, 3, 2).has_value());
            ASSERT_TRUE(fitted->apply_into(x, 3, 2, out_buf).has_value());
            ASSERT_TRUE(fitted->inverse_apply_into(out_buf, 3, 2, mut_buf).has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }
    }

    /// @brief Negative controls verifying ColumnwiseInput enforces strict input validation precedence.
    /// @details Confirms EmptyInput > SizeOverflow > LengthMismatch > DimensionMismatch > NonFiniteInput
    /// during both fit_typed() and apply_into() when multiple violations coincide.
    TEST(TransformColumnwiseTest, ValidationEnforcesStrictTotalOrder)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::array<double, 4> x{0.0, 1.0, 2.0, 3.0};
        const std::array<double, 4> nan_x{nan, 1.0, 2.0, 3.0};
        auto spec = ColumnwiseInput().then(MinMaxInput{});

        // EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch -> NonFiniteInput
        EXPECT_EQ(spec.fit_typed(x, 0, 1).error().code(), GprErrorCode::EmptyInput);

        const std::size_t huge = (std::numeric_limits<std::size_t>::max() / 2U) + 1U;
        EXPECT_EQ(spec.fit_typed(nan_x, huge, 3).error().code(), GprErrorCode::SizeOverflow);

        EXPECT_EQ(spec.fit_typed(nan_x, 3, 1).error().code(), GprErrorCode::LengthMismatch);

        // Dimension mismatch against the 1-column spec must fire before scanning nan_x for NaNs.
        auto dim_err = spec.fit_typed(nan_x, 2, 2);
        ASSERT_FALSE(dim_err.has_value());
        EXPECT_EQ(dim_err.error().code(), GprErrorCode::DimensionMismatch);
        EXPECT_EQ(dim_err.error().x_dim(), 2U);
        EXPECT_EQ(dim_err.error().expected_dim(), 1U);

        EXPECT_EQ(spec.fit_typed(nan_x, 4, 1).error().code(), GprErrorCode::NonFiniteInput);

        // Verify identical validation precedence on FittedColumnwiseInput::apply_into().
        auto fitted = spec.fit_typed(x, 4, 1);
        ASSERT_TRUE(fitted.has_value());
        std::array<double, 4> mut_x = nan_x;
        std::array<double, 4> out_x{};
        std::array<double, 2> short_out{};
        EXPECT_EQ(fitted->apply_into(mut_x, 3, 1, out_x).error().code(), GprErrorCode::LengthMismatch);
        EXPECT_EQ(fitted->apply_into(mut_x, 4, 1, short_out).error().code(), GprErrorCode::LengthMismatch);
        EXPECT_EQ(fitted->apply_into(mut_x, 2, 2, out_x).error().code(), GprErrorCode::DimensionMismatch);
        EXPECT_EQ(fitted->apply_into(mut_x, 4, 1, out_x).error().code(), GprErrorCode::NonFiniteInput);
    }

    /// @brief Verifies copying ColumnwiseInput and FittedColumnwiseInput preserves deep step ownership.
    /// @details Mutates a copied ColumnwiseInput via then() without affecting the original spec, and
    /// copy-assigns a FittedColumnwiseInput while verifying its column map count.
    TEST(TransformColumnwiseTest, ClonePreservesMapCount)
    {
        auto spec = ColumnwiseInput().then(MinMaxInput{}).then(StandardizeInput{});
        ColumnwiseInput copy(spec);
        // Appending a third column transform to the copy must not mutate the original spec.
        copy.then(IdentityInput{});
        EXPECT_EQ(spec.size(), 2U);
        EXPECT_EQ(copy.size(), 3U);

        auto fitted = spec.fit_typed(std::array<double, 4>{0.0, 1.0, 2.0, 3.0}, 2, 2);
        ASSERT_TRUE(fitted.has_value());
        FittedColumnwiseInput fitted_copy;
        fitted_copy = *fitted;
        EXPECT_EQ(fitted_copy.size(), 2U);
    }

}  // namespace
