// SPDX-License-Identifier: Apache-2.0
/// @file detail.hpp
/// @brief Internal validation and numerical summary helpers for transform implementations.
/// @details Provides canonical input matrix and target vector validation, min-max interval validation, two-pass
/// population standard deviation, and safe zero-span fallback helpers.

#pragma once

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/core/error.hpp"

#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <utility>

/// @brief Internal validation and numerical summary helpers for transform implementations.
namespace cxxgpr::transform::detail
{

    /// @brief Validates that a min-max output interval [lo, hi] is finite and strictly ordered.
    /// @details Rejects NaN, infinity, or degenerate/inverted bounds where hi <= lo.
    /// @param[in] lo Lower endpoint of the target interval.
    /// @param[in] hi Upper endpoint of the target interval.
    /// @return Empty expected on success, or GprErrorCode::InvalidConfig on invalid bounds.
    /// @retval GprErrorCode::InvalidConfig When lo or hi is non-finite or hi <= lo.
    [[nodiscard]] inline std::expected<void, GprError> validate_feature_range(double lo, double hi)
    {
        // Reject non-finite or collapsed/inverted intervals so scaling denominators stay strictly positive.
        if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo)
        {
            return std::unexpected(
                GprError::invalid_config(
                    std::format("feature range must satisfy lo < hi and both finite, got [{}, {}]", lo, hi)
                )
            );
        }
        return {};
    }

    /// @brief Validates a column-major feature matrix buffer and optional column count and output buffer length.
    /// @details Enforces the canonical validation sequence EmptyInput -> SizeOverflow -> LengthMismatch ->
    /// DimensionMismatch -> NonFiniteInput.
    /// @param[in] x Read-only column-major feature buffer.
    /// @param[in] n_rows Number of sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @param[in] expected_cols Optional expected feature column count checked against n_cols.
    /// @param[in] out_size Optional output buffer length checked against n_rows * n_cols.
    /// @return Empty expected on success, or the highest-precedence GprError.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, or x is empty.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or *out_size differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When expected_cols is provided and n_cols != *expected_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    [[nodiscard]] inline std::expected<void, GprError> validate_input_matrix(
        std::span<const double> x,
        std::size_t n_rows,
        std::size_t n_cols,
        std::optional<std::size_t> expected_cols = std::nullopt,
        std::optional<std::size_t> out_size = std::nullopt
    )
    {
        // Validate in canonical order: EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch ->
        // NonFiniteInput.
        if (n_rows == 0 || n_cols == 0 || x.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        auto expected = checked_mul(n_rows, n_cols);
        if (!expected)
        {
            return std::unexpected(std::move(expected).error());
        }
        if (auto len_res = require_count(x.size(), *expected, "values"); !len_res)
        {
            return len_res;
        }
        if (out_size.has_value())
        {
            if (auto out_res = require_count(*out_size, *expected, "output values"); !out_res)
            {
                return out_res;
            }
        }
        if (expected_cols.has_value() && n_cols != *expected_cols)
        {
            return std::unexpected(GprError::dimension_mismatch(n_cols, *expected_cols));
        }
        return require_finite(x);
    }

    /// @brief Validates a target or predictive vector buffer and optional output buffer length.
    /// @details Enforces the canonical validation sequence EmptyInput -> LengthMismatch -> NonFiniteInput.
    /// @param[in] y Read-only span of target or prediction values.
    /// @param[in] out_size Optional output buffer length checked against y.size().
    /// @return Empty expected on success, or the highest-precedence GprError.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::LengthMismatch When out_size is provided and *out_size != y.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    [[nodiscard]] inline std::expected<void, GprError> validate_target_vector(
        std::span<const double> y, std::optional<std::size_t> out_size = std::nullopt
    )
    {
        // Validate in canonical order: EmptyInput -> LengthMismatch -> NonFiniteInput.
        if (y.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (out_size.has_value())
        {
            if (auto out_res = require_count(*out_size, y.size(), "output values"); !out_res)
            {
                return out_res;
            }
        }
        return require_finite(y);
    }

    /// @brief Computes empirical mean and population standard deviation for a non-empty finite span.
    /// @details Uses a two-pass calculation dividing squared deviations by N and falls back to 1.0 when the population
    /// standard deviation is zero or non-finite.
    /// @param[in] values Non-empty finite span of sample observations.
    /// @return Pair {mean, safe_population_std} where safe_population_std > 0.
    [[nodiscard]] inline std::pair<double, double> compute_mean_and_population_std(
        std::span<const double> values
    ) noexcept
    {
        // Two-pass evaluation keeps centered squared deviations accurate; constant inputs fall back to scale 1.0.
        const auto n = static_cast<double>(values.size());
        double sum = 0.0;
        for (const double v : values)
        {
            sum += v;
        }
        const double mean = sum / n;
        double sq_sum = 0.0;
        for (const double v : values)
        {
            const double delta = v - mean;
            sq_sum += delta * delta;
        }
        const double std_dev = std::sqrt(sq_sum / n);
        const double safe_std = (std_dev > 0.0 && std::isfinite(std_dev)) ? std_dev : 1.0;
        return {mean, safe_std};
    }

    /// @brief Computes max_val - min_val with a 1.0 fallback for zero or non-finite spans.
    /// @details Prevents division by zero when transforming or inverting constant features or targets.
    /// @param[in] max_val Observed maximum value.
    /// @param[in] min_val Observed minimum value.
    /// @return Strictly positive finite span (max_val - min_val when positive and finite, otherwise 1.0).
    [[nodiscard]] inline double safe_span(double max_val, double min_val) noexcept
    {
        // Fall back to unit span when extrema coincide or their difference overflows to infinity.
        const double span = max_val - min_val;
        if (span > 0.0 && std::isfinite(span))
        {
            return span;
        }
        return 1.0;
    }

}  // namespace cxxgpr::transform::detail
