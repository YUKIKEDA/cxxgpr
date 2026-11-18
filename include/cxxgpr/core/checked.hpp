// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

namespace cxxgpr
{

    /// Adds two std::size_t counts, returning SizeOverflow on unsigned wraparound.
    [[nodiscard]] std::expected<std::size_t, GprError> checked_add(std::size_t a, std::size_t b) noexcept;

    /// Multiplies two std::size_t counts, returning SizeOverflow on unsigned wraparound.
    [[nodiscard]] std::expected<std::size_t, GprError> checked_mul(std::size_t a, std::size_t b) noexcept;

    /// Computes n_rows * n_cols for non-empty column-major dimensions.
    [[nodiscard]] std::expected<std::size_t, GprError> column_major_len(
        std::size_t n_rows, std::size_t n_cols
    ) noexcept;

    /// Computes packed lower-triangular length n * (n + 1) / 2 without false intermediate overflow.
    [[nodiscard]] std::expected<std::size_t, GprError> packed_len(std::size_t n) noexcept;

    namespace detail
    {

        /// Formats a LengthMismatch error when actual != expected in require_count.
        [[nodiscard]] GprError make_count_mismatch_error(
            std::size_t actual, std::size_t expected, std::string_view what
        );

    }  // namespace detail

    /// Rejects a zero count with EmptyInput.
    [[nodiscard]] std::expected<void, GprError> require_nonempty(std::size_t n) noexcept;

    /// Validates that actual equals expected, returning LengthMismatch otherwise.
    [[nodiscard]] inline std::expected<void, GprError> require_count(
        std::size_t actual, std::size_t expected, std::string_view what
    )
    {
        if (actual == expected)
        {
            return {};
        }
        return std::unexpected(detail::make_count_mismatch_error(actual, expected, what));
    }

    /// Rejects any NaN or Inf entry in a double span with NonFiniteInput (subnormals are allowed).
    [[nodiscard]] std::expected<void, GprError> require_finite(std::span<const double> values) noexcept;

    /// Rejects any NaN or Inf entry in a float span with NonFiniteInput (subnormals are allowed).
    [[nodiscard]] std::expected<void, GprError> require_finite(std::span<const float> values) noexcept;

    /// Validates a non-empty, finite, column-major matrix span matching n_rows x n_cols.
    [[nodiscard]] std::expected<void, GprError> validate_matrix(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::string_view name = "X"
    );

    /// Validates training inputs (X, y) enforcing EmptyInput -> SizeOverflow -> LengthMismatch ->
    /// DimensionMismatch -> NonFiniteInput -> InsufficientData.
    [[nodiscard]] std::expected<void, GprError> validate_training(
        std::span<const double> x,
        std::size_t n_rows,
        std::size_t n_cols,
        std::span<const double> y,
        std::optional<std::size_t> expected_d = std::nullopt,
        std::size_t min_points = 1
    );

    /// Validates target vector y and sample count n for distance-only training.
    [[nodiscard]] std::expected<void, GprError> validate_distance_targets(
        std::size_t n, std::span<const double> y, std::size_t min_points = 1
    );

    /// Validates query inputs Xs against the expected feature dimension.
    [[nodiscard]] std::expected<void, GprError> validate_query(
        std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, std::size_t expected_d
    );

}  // namespace cxxgpr
