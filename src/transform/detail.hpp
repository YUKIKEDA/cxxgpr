// SPDX-License-Identifier: Apache-2.0
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

namespace cxxgpr::transform::detail
{

    [[nodiscard]] inline std::expected<void, GprError> validate_feature_range(double lo, double hi)
    {
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

    [[nodiscard]] inline std::expected<void, GprError> validate_input_matrix(
        std::span<const double> x,
        std::size_t n_rows,
        std::size_t n_cols,
        std::optional<std::size_t> expected_cols = std::nullopt,
        std::optional<std::size_t> out_size = std::nullopt
    )
    {
        // Enforce DES-02 order: EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch ->
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

    [[nodiscard]] inline std::expected<void, GprError> validate_target_vector(
        std::span<const double> y, std::optional<std::size_t> out_size = std::nullopt
    )
    {
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

    [[nodiscard]] inline std::pair<double, double> compute_mean_and_population_std(
        std::span<const double> values
    ) noexcept
    {
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

    [[nodiscard]] inline double safe_span(double max_val, double min_val) noexcept
    {
        const double span = max_val - min_val;
        if (span > 0.0 && std::isfinite(span))
        {
            return span;
        }
        return 1.0;
    }

}  // namespace cxxgpr::transform::detail
