// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/columnwise.hpp"
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

using cxxgpr::GprError;
using cxxgpr::GprErrorCode;
using cxxgpr::test_support::AuditMode;
using cxxgpr::test_support::NoAllocationScope;
using cxxgpr::transform::ColumnwiseInput;
using cxxgpr::transform::FittedColumnwiseInput;
using cxxgpr::transform::FittedInputTransform;
using cxxgpr::transform::IdentityInput;
using cxxgpr::transform::MinMaxInput;
using cxxgpr::transform::Pipeline;
using cxxgpr::transform::StandardizeInput;
using cxxgpr::transform::UnfittedInputTransform;

constexpr double k_tol = 1e-10;

class FittedTimesTwoCol final : public FittedInputTransform {
public:
    [[nodiscard]] std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows,
                                                      std::size_t n_cols) const override {
        if (n_rows == 0 || n_cols == 0 || x.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        for (double& v : x) {
            v *= 2.0;
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    inverse_apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override {
        if (n_rows == 0 || n_cols == 0 || x.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        for (double& v : x) {
            v *= 0.5;
        }
        return {};
    }

    [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override {
        return std::make_unique<FittedTimesTwoCol>(*this);
    }
};

class UnfittedTimesTwoCol final : public UnfittedInputTransform {
public:
    [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError>
    fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const override {
        if (n_rows == 0 || n_cols == 0 || x.empty()) {
            return std::unexpected(GprError::empty_input());
        }
        return std::make_unique<FittedTimesTwoCol>();
    }

    [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override {
        return std::make_unique<UnfittedTimesTwoCol>(*this);
    }
};

TEST(TransformColumnwiseTest, MinMaxAndStandardizeEachColumn) {
    const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
    auto spec = ColumnwiseInput().then(MinMaxInput{}).then(StandardizeInput{});
    EXPECT_EQ(spec.size(), 2U);

    auto fitted = spec.fit_typed(x, 3, 2);
    ASSERT_TRUE(fitted.has_value());
    EXPECT_EQ(fitted->size(), 2U);

    auto mm = MinMaxInput{}.fit_typed(std::span<const double>(x).subspan(0, 3), 3, 1);
    auto st = StandardizeInput{}.fit_typed(std::span<const double>(x).subspan(3, 3), 3, 1);
    ASSERT_TRUE(mm.has_value());
    ASSERT_TRUE(st.has_value());

    std::array<double, 6> expected = x;
    ASSERT_TRUE(mm->apply(std::span<double>(expected).subspan(0, 3), 3, 1).has_value());
    ASSERT_TRUE(st->apply(std::span<double>(expected).subspan(3, 3), 3, 1).has_value());

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

TEST(TransformColumnwiseTest, CustomAndPipelineMapsWorkPerColumn) {
    const std::array<double, 4> x{1.0, 2.0, 3.0, 4.0};
    auto col_pipe = Pipeline().then(MinMaxInput{}).then(StandardizeInput{});
    auto spec = ColumnwiseInput().then(IdentityInput{}).then(UnfittedTimesTwoCol{});
    auto fitted = spec.fit_typed(x, 2, 2);
    ASSERT_TRUE(fitted.has_value());

    std::array<double, 4> z = x;
    ASSERT_TRUE(fitted->apply(z, 2, 2).has_value());
    EXPECT_NEAR(z[0], 1.0, k_tol);
    EXPECT_NEAR(z[1], 2.0, k_tol);
    EXPECT_NEAR(z[2], 6.0, k_tol);
    EXPECT_NEAR(z[3], 8.0, k_tol);

    ASSERT_TRUE(fitted->inverse_apply(z, 2, 2).has_value());
    for (std::size_t i = 0; i < x.size(); ++i) {
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

TEST(TransformColumnwiseTest, HotPathZeroAllocation) {
    const std::array<double, 6> x{0.0, 2.0, 4.0, 1.0, 3.0, 5.0};
    auto fitted = ColumnwiseInput().then(MinMaxInput{}).then(StandardizeInput{}).fit_typed(x, 3, 2);
    ASSERT_TRUE(fitted.has_value());

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

TEST(TransformColumnwiseTest, ValidationEnforcesStrictTotalOrder) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::array<double, 4> x{0.0, 1.0, 2.0, 3.0};
    const std::array<double, 4> nan_x{nan, 1.0, 2.0, 3.0};
    auto spec = ColumnwiseInput().then(MinMaxInput{});

    // EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch -> NonFiniteInput
    EXPECT_EQ(spec.fit_typed(x, 0, 1).error().code(), GprErrorCode::EmptyInput);

    const std::size_t huge = (std::numeric_limits<std::size_t>::max() / 2U) + 1U;
    EXPECT_EQ(spec.fit_typed(nan_x, huge, 3).error().code(), GprErrorCode::SizeOverflow);

    EXPECT_EQ(spec.fit_typed(nan_x, 3, 1).error().code(), GprErrorCode::LengthMismatch);

    auto dim_err = spec.fit_typed(nan_x, 2, 2);
    ASSERT_FALSE(dim_err.has_value());
    EXPECT_EQ(dim_err.error().code(), GprErrorCode::DimensionMismatch);
    EXPECT_EQ(dim_err.error().x_dim(), 2U);
    EXPECT_EQ(dim_err.error().expected_dim(), 1U);

    EXPECT_EQ(spec.fit_typed(nan_x, 4, 1).error().code(), GprErrorCode::NonFiniteInput);

    auto fitted = spec.fit_typed(x, 4, 1);
    ASSERT_TRUE(fitted.has_value());
    std::array<double, 4> mut_x = nan_x;
    std::array<double, 4> out_x{};
    std::array<double, 2> short_out{};
    EXPECT_EQ(fitted->apply_into(mut_x, 3, 1, out_x).error().code(), GprErrorCode::LengthMismatch);
    EXPECT_EQ(fitted->apply_into(mut_x, 4, 1, short_out).error().code(),
              GprErrorCode::LengthMismatch);
    EXPECT_EQ(fitted->apply_into(mut_x, 2, 2, out_x).error().code(),
              GprErrorCode::DimensionMismatch);
    EXPECT_EQ(fitted->apply_into(mut_x, 4, 1, out_x).error().code(), GprErrorCode::NonFiniteInput);
}

TEST(TransformColumnwiseTest, ClonePreservesMapCount) {
    auto spec = ColumnwiseInput().then(MinMaxInput{}).then(StandardizeInput{});
    ColumnwiseInput copy(spec);
    copy.then(IdentityInput{});
    EXPECT_EQ(spec.size(), 2U);
    EXPECT_EQ(copy.size(), 3U);

    auto fitted = spec.fit_typed(std::array<double, 4>{0.0, 1.0, 2.0, 3.0}, 2, 2);
    ASSERT_TRUE(fitted.has_value());
    FittedColumnwiseInput fitted_copy;
    fitted_copy = *fitted;
    EXPECT_EQ(fitted_copy.size(), 2U);
}

} // namespace
