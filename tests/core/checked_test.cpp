// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/checked.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

// Spec: REQ-03, DES-02
namespace {

TEST(CheckedTest, ArithmeticRejectsOverflowAndHandlesOddEvenPackedLen) {
    constexpr std::size_t max_sz = std::numeric_limits<std::size_t>::max();

    EXPECT_EQ(cxxgpr::checked_add(3U, 4U), 7U);
    EXPECT_EQ(cxxgpr::checked_add(max_sz, 0U), max_sz);
    const auto add_ov = cxxgpr::checked_add(max_sz, 1U);
    ASSERT_FALSE(add_ov.has_value());
    EXPECT_EQ(add_ov.error().code(), cxxgpr::GprErrorCode::SizeOverflow);

    EXPECT_EQ(cxxgpr::checked_mul(3U, 4U), 12U);
    EXPECT_EQ(cxxgpr::checked_mul(max_sz, 0U), 0U);
    const auto mul_ov = cxxgpr::checked_mul(max_sz, 2U);
    ASSERT_FALSE(mul_ov.has_value());
    EXPECT_EQ(mul_ov.error().code(), cxxgpr::GprErrorCode::SizeOverflow);

    EXPECT_EQ(cxxgpr::column_major_len(3U, 2U), 6U);
    EXPECT_EQ(cxxgpr::column_major_len(0U, 2U).error().code(), cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::column_major_len(3U, 0U).error().code(), cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::column_major_len(max_sz, 2U).error().code(),
              cxxgpr::GprErrorCode::SizeOverflow);

    EXPECT_EQ(cxxgpr::packed_len(0U), 0U);
    EXPECT_EQ(cxxgpr::packed_len(1U), 1U);
    EXPECT_EQ(cxxgpr::packed_len(2U), 3U);
    EXPECT_EQ(cxxgpr::packed_len(3U), 6U);
    EXPECT_EQ(cxxgpr::packed_len(4U), 10U);
    EXPECT_EQ(cxxgpr::packed_len(100U), 5050U);

    // Ensure odd n == SIZE_MAX does not wrap around when evaluating (n + 1) / 2.
    EXPECT_EQ(cxxgpr::packed_len(max_sz).error().code(), cxxgpr::GprErrorCode::SizeOverflow);
    EXPECT_EQ(cxxgpr::packed_len(max_sz - 1U).error().code(), cxxgpr::GprErrorCode::SizeOverflow);
}

TEST(CheckedTest, RequireCountAndFiniteRespectMessagesAndSubnormals) {
    const auto count_ok = cxxgpr::require_count(3U, 3U, "kernel parameters");
    EXPECT_TRUE(count_ok.has_value());

    const auto count_err = cxxgpr::require_count(2U, 3U, "kernel parameters");
    ASSERT_FALSE(count_err.has_value());
    EXPECT_EQ(count_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);
    EXPECT_EQ(count_err.error().message(), "length mismatch: expected 3 kernel parameters, got 2");

    // Subnormal and signed zero values are valid IEEE 754 finite numbers.
    const std::array<double, 4> valid_d{
        0.0,
        -0.0,
        std::numeric_limits<double>::denorm_min(),
        -std::numeric_limits<double>::denorm_min(),
    };
    EXPECT_TRUE(cxxgpr::require_finite(valid_d).has_value());

    const std::array<float, 2> valid_f{
        std::numeric_limits<float>::denorm_min(),
        1.5F,
    };
    EXPECT_TRUE(cxxgpr::require_finite(valid_f).has_value());

    for (const double non_finite :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        const std::array<double, 2> bad{1.0, non_finite};
        const auto err = cxxgpr::require_finite(bad);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);
    }

    const std::array<float, 2> bad_f{1.0F, std::numeric_limits<float>::infinity()};
    const auto err_f = cxxgpr::require_finite(bad_f);
    ASSERT_FALSE(err_f.has_value());
    EXPECT_EQ(err_f.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);
}

TEST(CheckedTest, ValidateTrainingEnforcesStrictTotalValidationOrder) {
    constexpr std::size_t max_sz = std::numeric_limits<std::size_t>::max();
    constexpr std::size_t gram_overflow_n = std::size_t{1} << (sizeof(std::size_t) * 4U);
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const std::array<double, 1> nan_one{nan_val};
    const std::array<double, 1> finite_one{1.0};
    const std::array<double, 3> wrong_len_x{nan_val, 1.0, 2.0};
    const std::array<double, 2> mismatched_dim_nan{nan_val, 1.0};

    // 1. EmptyInput beats SizeOverflow, LengthMismatch, DimensionMismatch, NonFiniteInput,
    // InsufficientData.
    const std::array<double, 0> empty{};
    const auto e1 = cxxgpr::validate_training(empty, max_sz, 2U, nan_one, 1U, 10U);
    ASSERT_FALSE(e1.has_value());
    EXPECT_EQ(e1.error().code(), cxxgpr::GprErrorCode::EmptyInput);

    const auto e1_zero_rows = cxxgpr::validate_training(nan_one, 0U, 2U, nan_one, 1U, 10U);
    ASSERT_FALSE(e1_zero_rows.has_value());
    EXPECT_EQ(e1_zero_rows.error().code(), cxxgpr::GprErrorCode::EmptyInput);

    // 2a. SizeOverflow from n_rows * n_cols beats LengthMismatch, DimensionMismatch,
    // NonFiniteInput, InsufficientData.
    const auto e2a = cxxgpr::validate_training(wrong_len_x, max_sz, 2U, nan_one, 1U, 10U);
    ASSERT_FALSE(e2a.has_value());
    EXPECT_EQ(e2a.error().code(), cxxgpr::GprErrorCode::SizeOverflow);

    // 2b. SizeOverflow from Gram matrix n_rows * n_rows beats LengthMismatch and later checks.
    const auto e2b = cxxgpr::validate_training(wrong_len_x, gram_overflow_n, 2U, nan_one, 1U,
                                               gram_overflow_n + 1U);
    ASSERT_FALSE(e2b.has_value());
    EXPECT_EQ(e2b.error().code(), cxxgpr::GprErrorCode::SizeOverflow);

    // 3. LengthMismatch beats DimensionMismatch, NonFiniteInput, InsufficientData.
    const auto e3_x = cxxgpr::validate_training(wrong_len_x, 1U, 2U, nan_one, 1U, 10U);
    ASSERT_FALSE(e3_x.has_value());
    EXPECT_EQ(e3_x.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

    const auto e3_y = cxxgpr::validate_training(mismatched_dim_nan, 1U, 2U, wrong_len_x, 1U, 10U);
    ASSERT_FALSE(e3_y.has_value());
    EXPECT_EQ(e3_y.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

    // 4. DimensionMismatch beats NonFiniteInput and InsufficientData.
    const auto e4 = cxxgpr::validate_training(mismatched_dim_nan, 1U, 2U, nan_one, 1U, 10U);
    ASSERT_FALSE(e4.has_value());
    EXPECT_EQ(e4.error().code(), cxxgpr::GprErrorCode::DimensionMismatch);
    EXPECT_EQ(e4.error().x_dim(), 2U);
    EXPECT_EQ(e4.error().expected_dim(), 1U);

    // 5. NonFiniteInput beats InsufficientData.
    const auto e5_x = cxxgpr::validate_training(nan_one, 1U, 1U, finite_one, 1U, 2U);
    ASSERT_FALSE(e5_x.has_value());
    EXPECT_EQ(e5_x.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

    const auto e5_y = cxxgpr::validate_training(finite_one, 1U, 1U, nan_one, 1U, 2U);
    ASSERT_FALSE(e5_y.has_value());
    EXPECT_EQ(e5_y.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

    // 6. InsufficientData fires when earlier checks pass but n_rows < min_points.
    const auto e6 = cxxgpr::validate_training(finite_one, 1U, 1U, finite_one, 1U, 2U);
    ASSERT_FALSE(e6.has_value());
    EXPECT_EQ(e6.error().code(), cxxgpr::GprErrorCode::InsufficientData);
    EXPECT_EQ(e6.error().n_points(), 1U);
    EXPECT_EQ(e6.error().min_points(), 2U);

    // Valid call including subnormals succeeds.
    const std::array<double, 2> valid_subnormals{
        std::numeric_limits<double>::denorm_min(),
        2.0,
    };
    EXPECT_TRUE(
        cxxgpr::validate_training(valid_subnormals, 1U, 2U, finite_one, 2U, 1U).has_value());
}

TEST(CheckedTest, ValidateDistanceTargetsAndQueryEnforceExactOrdering) {
    constexpr std::size_t gram_overflow_n = std::size_t{1} << (sizeof(std::size_t) * 4U);
    const double inf_val = std::numeric_limits<double>::infinity();
    const std::array<double, 1> inf_one{inf_val};
    const std::array<double, 2> two_vals{1.0, 2.0};
    const std::array<double, 1> one_val{1.0};

    // Distance targets order: EmptyInput -> SizeOverflow -> LengthMismatch -> NonFiniteInput ->
    // InsufficientData.
    const std::array<double, 0> empty{};
    EXPECT_EQ(cxxgpr::validate_distance_targets(0U, inf_one, 2U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::validate_distance_targets(2U, empty, 2U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::validate_distance_targets(gram_overflow_n, two_vals, gram_overflow_n + 1U)
                  .error()
                  .code(),
              cxxgpr::GprErrorCode::SizeOverflow);
    EXPECT_EQ(cxxgpr::validate_distance_targets(1U, two_vals, 5U).error().code(),
              cxxgpr::GprErrorCode::LengthMismatch);
    EXPECT_EQ(cxxgpr::validate_distance_targets(1U, inf_one, 2U).error().code(),
              cxxgpr::GprErrorCode::NonFiniteInput);
    EXPECT_EQ(cxxgpr::validate_distance_targets(1U, one_val, 2U).error().code(),
              cxxgpr::GprErrorCode::InsufficientData);
    EXPECT_TRUE(cxxgpr::validate_distance_targets(2U, two_vals, 2U).has_value());

    // Query order: EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch ->
    // NonFiniteInput.
    constexpr std::size_t max_sz = std::numeric_limits<std::size_t>::max();
    EXPECT_EQ(cxxgpr::validate_query(empty, max_sz, 2U, 1U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::validate_query(two_vals, max_sz, 2U, 1U).error().code(),
              cxxgpr::GprErrorCode::SizeOverflow);
    EXPECT_EQ(cxxgpr::validate_query(two_vals, 1U, 3U, 1U).error().code(),
              cxxgpr::GprErrorCode::LengthMismatch);
    const std::array<double, 2> inf_two{inf_val, 1.0};
    EXPECT_EQ(cxxgpr::validate_query(inf_two, 1U, 2U, 1U).error().code(),
              cxxgpr::GprErrorCode::DimensionMismatch);
    EXPECT_EQ(cxxgpr::validate_query(inf_two, 1U, 2U, 2U).error().code(),
              cxxgpr::GprErrorCode::NonFiniteInput);
    EXPECT_TRUE(cxxgpr::validate_query(two_vals, 1U, 2U, 2U).has_value());

    // validate_matrix order: EmptyInput -> SizeOverflow -> LengthMismatch -> NonFiniteInput.
    EXPECT_EQ(cxxgpr::validate_matrix(empty, max_sz, 2U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::validate_matrix(inf_two, 0U, 2U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::validate_matrix(inf_two, 2U, 0U).error().code(),
              cxxgpr::GprErrorCode::EmptyInput);
    EXPECT_EQ(cxxgpr::validate_matrix(inf_two, max_sz, 2U).error().code(),
              cxxgpr::GprErrorCode::SizeOverflow);
    EXPECT_EQ(cxxgpr::validate_matrix(inf_two, 1U, 3U).error().code(),
              cxxgpr::GprErrorCode::LengthMismatch);
    EXPECT_EQ(cxxgpr::validate_matrix(inf_two, 1U, 2U).error().code(),
              cxxgpr::GprErrorCode::NonFiniteInput);
    const std::array<double, 2> valid_subnormal_mat{
        std::numeric_limits<double>::denorm_min(),
        2.0,
    };
    EXPECT_TRUE(cxxgpr::validate_matrix(valid_subnormal_mat, 2U, 1U).has_value());
}

} // namespace
