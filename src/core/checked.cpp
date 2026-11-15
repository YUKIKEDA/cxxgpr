// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <limits>

namespace cxxgpr {

namespace {

template <typename T>
std::expected<void, GprError> require_finite_impl(std::span<const T> values) noexcept {
    const bool all_finite = std::all_of(values.begin(), values.end(),
                                        [](T value) noexcept { return std::isfinite(value); });
    if (!all_finite) {
        return std::unexpected(GprError::non_finite_input());
    }
    return {};
}

std::expected<std::size_t, GprError> validate_shape_header(std::size_t data_size,
                                                           std::size_t n_rows, std::size_t n_cols,
                                                           std::string_view name) {
    if (n_rows == 0 || n_cols == 0 || data_size == 0) {
        return std::unexpected(GprError::empty_input());
    }
    auto expected = checked_mul(n_rows, n_cols);
    if (!expected) {
        return expected;
    }
    if (data_size != *expected) {
        return std::unexpected(GprError::length_mismatch(
            std::format("expected {} {} values, got {}", *expected, name, data_size)));
    }
    return *expected;
}

} // namespace

std::expected<std::size_t, GprError> checked_add(std::size_t a, std::size_t b) noexcept {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        return std::unexpected(GprError::size_overflow());
    }
    return a + b;
}

std::expected<std::size_t, GprError> checked_mul(std::size_t a, std::size_t b) noexcept {
    if (a == 0 || b == 0) {
        return std::size_t{0};
    }
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        return std::unexpected(GprError::size_overflow());
    }
    return a * b;
}

std::expected<std::size_t, GprError> column_major_len(std::size_t n_rows,
                                                      std::size_t n_cols) noexcept {
    if (n_rows == 0 || n_cols == 0) {
        return std::unexpected(GprError::empty_input());
    }
    return checked_mul(n_rows, n_cols);
}

std::expected<std::size_t, GprError> packed_len(std::size_t n) noexcept {
    if ((n % 2U) == 0U) {
        return checked_mul(n / 2U, n + 1U);
    }
    // Compute (n + 1) / 2 as n / 2 + 1 to avoid wrapping when n == SIZE_MAX.
    return checked_mul(n, (n / 2U) + 1U);
}

std::expected<void, GprError> require_nonempty(std::size_t n) noexcept {
    if (n == 0) {
        return std::unexpected(GprError::empty_input());
    }
    return {};
}

std::expected<void, GprError> require_count(std::size_t actual, std::size_t expected,
                                            std::string_view what) {
    if (actual == expected) {
        return {};
    }
    return std::unexpected(
        GprError::length_mismatch(std::format("expected {} {}, got {}", expected, what, actual)));
}

std::expected<void, GprError> require_finite(std::span<const double> values) noexcept {
    return require_finite_impl(values);
}

std::expected<void, GprError> require_finite(std::span<const float> values) noexcept {
    return require_finite_impl(values);
}

std::expected<void, GprError> validate_matrix(std::span<const double> x, std::size_t n_rows,
                                              std::size_t n_cols, std::string_view name) {
    auto len = validate_shape_header(x.size(), n_rows, n_cols, name);
    if (!len) {
        return std::unexpected(std::move(len.error()));
    }
    return require_finite(x);
}

std::expected<void, GprError> validate_training(std::span<const double> x, std::size_t n_rows,
                                                std::size_t n_cols, std::span<const double> y,
                                                std::optional<std::size_t> expected_d,
                                                std::size_t min_points) {
    // Enforce DES-02 strict validation total order:
    // EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch -> NonFiniteInput ->
    // InsufficientData.
    if (n_rows == 0 || n_cols == 0 || x.empty() || y.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    auto expected_x = checked_mul(n_rows, n_cols);
    if (!expected_x) {
        return std::unexpected(expected_x.error());
    }
    if (auto gram_len = checked_mul(n_rows, n_rows); !gram_len) {
        return std::unexpected(gram_len.error());
    }
    if (x.size() != *expected_x) {
        return std::unexpected(GprError::length_mismatch(
            std::format("expected {} feature values, got {}", *expected_x, x.size())));
    }
    if (y.size() != n_rows) {
        return std::unexpected(GprError::length_mismatch(
            std::format("expected {} targets, got {}", n_rows, y.size())));
    }
    if (expected_d.has_value() && n_cols != *expected_d) {
        return std::unexpected(GprError::dimension_mismatch(n_cols, *expected_d));
    }
    if (auto ok = require_finite(x); !ok) {
        return ok;
    }
    if (auto ok = require_finite(y); !ok) {
        return ok;
    }
    if (n_rows < min_points) {
        return std::unexpected(GprError::insufficient_data(n_rows, min_points));
    }
    return {};
}

std::expected<void, GprError> validate_distance_targets(std::size_t n, std::span<const double> y,
                                                        std::size_t min_points) {
    if (n == 0 || y.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (auto gram_len = checked_mul(n, n); !gram_len) {
        return std::unexpected(gram_len.error());
    }
    if (y.size() != n) {
        return std::unexpected(
            GprError::length_mismatch(std::format("expected {} targets, got {}", n, y.size())));
    }
    if (auto ok = require_finite(y); !ok) {
        return ok;
    }
    if (n < min_points) {
        return std::unexpected(GprError::insufficient_data(n, min_points));
    }
    return {};
}

std::expected<void, GprError> validate_query(std::span<const double> xs, std::size_t n_rows,
                                             std::size_t n_cols, std::size_t expected_d) {
    auto len = validate_shape_header(xs.size(), n_rows, n_cols, "query feature");
    if (!len) {
        return std::unexpected(std::move(len.error()));
    }
    if (n_cols != expected_d) {
        return std::unexpected(GprError::dimension_mismatch(n_cols, expected_d));
    }
    return require_finite(xs);
}

} // namespace cxxgpr
