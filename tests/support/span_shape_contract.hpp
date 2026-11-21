// SPDX-License-Identifier: Apache-2.0
/// @file span_shape_contract.hpp
/// @brief Shared negative controls for public overloads taking a raw column-major span plus dimensions.
/// @details Drives one overload through every way a span can disagree with n_rows * n_cols and checks the
/// returned error code, so each overload is held to the same shape contract and validation order.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <vector>

/// @brief Test support utilities shared by the span-overload negative controls.
namespace cxxgpr::test_support
{

    /// @brief Reduces an expected result to its error code, or nullopt on success.
    /// @tparam T Success value type of the result.
    /// @param[in] result Result returned by the overload under test.
    /// @return Error code of a failed result, or std::nullopt when the call succeeded.
    template <typename T>
    [[nodiscard]] std::optional<GprErrorCode> error_code_of(const std::expected<T, GprError>& result)
    {
        if (result.has_value())
        {
            return std::nullopt;
        }
        return result.error().code();
    }

    /// @brief Checks one span overload against the shape contract EmptyInput -> SizeOverflow -> LengthMismatch.
    /// @details Covers a short span, a long span, an overflowing dimension product, an empty span with a null
    /// pointer, an empty span with a non-null pointer and non-zero dimensions, and a non-empty span with a zero
    /// dimension, plus the inputs where two violations hold at once. The non-null empty span points one past the
    /// end of a heap block, so an implementation that dereferences it is caught by AddressSanitizer. The valid
    /// shape must succeed before and after the rejected calls.
    /// @tparam Call Callable (std::span<const double>, std::size_t, std::size_t) -> std::optional<GprErrorCode>.
    /// @param[in] call Adapter invoking the overload under test and reducing the result with error_code_of().
    /// @param[in] valid Column-major data of exactly n_rows * n_cols finite values accepted by the overload.
    /// @param[in] n_rows Row count matching valid (at least 1).
    /// @param[in] n_cols Column count matching valid (at least 1).
    template <typename Call>
    void expect_span_shape_contract(
        const Call& call, std::span<const double> valid, std::size_t n_rows, std::size_t n_cols
    )
    {
        using Code = std::optional<GprErrorCode>;
        constexpr std::size_t huge = std::numeric_limits<std::size_t>::max();
        ASSERT_EQ(valid.size(), n_rows * n_cols);
        ASSERT_GE(valid.size(), 2U);

        std::vector<double> padded(valid.begin(), valid.end());
        padded.push_back(valid.front());
        const std::span<const double> too_long(padded);
        const std::span<const double> too_short = valid.first(valid.size() - 1U);
        // Size 0 with a non-null pointer one past the end of an exactly sized heap block.
        const std::vector<double> block(valid.begin(), valid.end());
        const std::span<const double> empty_non_null = std::span<const double>(block).last(0U);
        ASSERT_NE(empty_non_null.data(), nullptr);
        const std::span<const double> empty_null;

        EXPECT_EQ(call(valid, n_rows, n_cols), Code{}) << "valid shape before the rejected calls";

        EXPECT_EQ(call(too_short, n_rows, n_cols), Code{GprErrorCode::LengthMismatch}) << "short span";
        EXPECT_EQ(call(too_long, n_rows, n_cols), Code{GprErrorCode::LengthMismatch}) << "long span";
        EXPECT_EQ(call(valid, huge, 2U), Code{GprErrorCode::SizeOverflow}) << "overflowing n_rows * n_cols";
        EXPECT_EQ(call(valid, 2U, huge), Code{GprErrorCode::SizeOverflow}) << "overflowing n_rows * n_cols";
        EXPECT_EQ(call(empty_null, n_rows, n_cols), Code{GprErrorCode::EmptyInput}) << "empty span, null pointer";
        EXPECT_EQ(call(empty_non_null, n_rows, n_cols), Code{GprErrorCode::EmptyInput})
            << "empty span, non-null pointer, non-zero dimensions";
        EXPECT_EQ(call(valid, 0U, n_cols), Code{GprErrorCode::EmptyInput}) << "non-empty span, zero n_rows";
        EXPECT_EQ(call(valid, n_rows, 0U), Code{GprErrorCode::EmptyInput}) << "non-empty span, zero n_cols";

        // Two violations at once: the earlier one in the validation order is reported.
        EXPECT_EQ(call(empty_non_null, huge, 2U), Code{GprErrorCode::EmptyInput}) << "empty before overflow";
        EXPECT_EQ(call(valid, 0U, huge), Code{GprErrorCode::EmptyInput}) << "zero dimension before overflow";
        EXPECT_EQ(call(too_short, huge, 2U), Code{GprErrorCode::SizeOverflow}) << "overflow before length mismatch";

        EXPECT_EQ(call(valid, n_rows, n_cols), Code{}) << "valid shape after the rejected calls";
    }

}  // namespace cxxgpr::test_support
