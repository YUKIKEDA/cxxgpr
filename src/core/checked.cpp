// SPDX-License-Identifier: Apache-2.0
/// @file checked.cpp
/// @brief Implementation of overflow-safe arithmetic and strict input validation helpers.
/// @details Implements checked unsigned size math, finiteness checks, and ordered precondition
/// validators for feature matrices, targets, and query inputs.

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <limits>

namespace cxxgpr
{

    namespace
    {

        /// @brief Verifies that every element in a floating-point span is finite.
        /// @details Uses std::isfinite so normal, zero, and subnormal values pass while NaN and +/-Inf fail.
        /// @tparam T Floating-point scalar type (float or double).
        /// @param[in] values Input span to inspect.
        /// @return Empty expected if all elements are finite, or GprError otherwise.
        /// @retval GprErrorCode::NonFiniteInput When any element is NaN or +/-Inf.
        template <typename T> std::expected<void, GprError> require_finite_impl(std::span<const T> values) noexcept
        {
            // std::isfinite accepts subnormals and rejects NaN and +/-Inf.
            const bool all_finite =
                std::all_of(values.begin(), values.end(), [](T value) noexcept { return std::isfinite(value); });
            if (!all_finite)
            {
                return std::unexpected(GprError::non_finite_input());
            }
            return {};
        }

        /// @brief Validates non-emptiness, dimension product overflow, and buffer length for a matrix.
        /// @details Enforces EmptyInput -> SizeOverflow -> LengthMismatch before inspecting values.
        /// @param[in] data_size Supplied flat buffer length.
        /// @param[in] n_rows Matrix row count.
        /// @param[in] n_cols Matrix column count.
        /// @param[in] name Matrix label included in length-mismatch diagnostics.
        /// @return Product n_rows * n_cols on success, or GprError on shape violation.
        /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, or data_size == 0.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When data_size != n_rows * n_cols.
        std::expected<std::size_t, GprError> validate_shape_header(
            std::size_t data_size, std::size_t n_rows, std::size_t n_cols, std::string_view name
        )
        {
            // Enforce shape checks in order: EmptyInput -> SizeOverflow -> LengthMismatch.
            if (n_rows == 0 || n_cols == 0 || data_size == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            auto expected = checked_mul(n_rows, n_cols);
            if (!expected)
            {
                return expected;
            }
            if (data_size != *expected)
            {
                return std::unexpected(
                    GprError::length_mismatch(std::format("expected {} {} values, got {}", *expected, name, data_size))
                );
            }
            return *expected;
        }

    }  // namespace

    /// @brief Adds two std::size_t counts, returning SizeOverflow on unsigned wraparound.
    /// @details Tests a > max - b prior to adding so unsigned wraparound never occurs.
    /// @param[in] a First non-negative addend.
    /// @param[in] b Second non-negative addend.
    /// @return Sum a + b, or GprError on overflow.
    /// @retval GprErrorCode::SizeOverflow When a + b exceeds std::numeric_limits<std::size_t>::max().
    std::expected<std::size_t, GprError> checked_add(std::size_t a, std::size_t b) noexcept
    {
        // Rearrange a + b > SIZE_MAX as a > SIZE_MAX - b to prevent unsigned wraparound.
        if (a > std::numeric_limits<std::size_t>::max() - b)
        {
            return std::unexpected(GprError::size_overflow());
        }
        return a + b;
    }

    /// @brief Multiplies two std::size_t counts, returning SizeOverflow on unsigned wraparound.
    /// @details Short-circuits zero operands before dividing max by b.
    /// @param[in] a First non-negative factor.
    /// @param[in] b Second non-negative factor.
    /// @return Product a * b, or GprError on overflow.
    /// @retval GprErrorCode::SizeOverflow When a * b exceeds std::numeric_limits<std::size_t>::max().
    std::expected<std::size_t, GprError> checked_mul(std::size_t a, std::size_t b) noexcept
    {
        // Guard division by zero before comparing a > SIZE_MAX / b.
        if (a == 0 || b == 0)
        {
            return std::size_t{0};
        }
        if (a > std::numeric_limits<std::size_t>::max() / b)
        {
            return std::unexpected(GprError::size_overflow());
        }
        return a * b;
    }

    /// @brief Computes n_rows * n_cols for non-empty column-major dimensions.
    /// @details Rejects zero dimensions with EmptyInput before checking multiplication overflow.
    /// @param[in] n_rows Number of matrix rows.
    /// @param[in] n_cols Number of matrix columns.
    /// @return Total element count n_rows * n_cols, or GprError.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0 or n_cols == 0.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    std::expected<std::size_t, GprError> column_major_len(std::size_t n_rows, std::size_t n_cols) noexcept
    {
        // Reject zero dimensions before multiplying so callers can assume a non-empty buffer.
        if (n_rows == 0 || n_cols == 0)
        {
            return std::unexpected(GprError::empty_input());
        }
        return checked_mul(n_rows, n_cols);
    }

    /// @brief Computes packed lower-triangular length n * (n + 1) / 2 without false intermediate overflow.
    /// @details Divides whichever factor is even by 2 prior to calling checked_mul().
    /// @param[in] n Matrix order.
    /// @return Packed lower-triangular element count, or GprError on overflow.
    /// @retval GprErrorCode::SizeOverflow When n * (n + 1) / 2 overflows std::size_t.
    std::expected<std::size_t, GprError> packed_len(std::size_t n) noexcept
    {
        // Halve n when even so n * (n + 1) cannot overflow prematurely.
        if ((n % 2U) == 0U)
        {
            return checked_mul(n / 2U, n + 1U);
        }
        // Compute (n + 1) / 2 as n / 2 + 1 to avoid wrapping when n == SIZE_MAX.
        return checked_mul(n, (n / 2U) + 1U);
    }

    /// @brief Rejects a zero count with EmptyInput.
    /// @details Returns success when n != 0.
    /// @param[in] n Element or dimension count to validate.
    /// @return Empty expected on success, or GprError when n == 0.
    /// @retval GprErrorCode::EmptyInput When n == 0.
    std::expected<void, GprError> require_nonempty(std::size_t n) noexcept
    {
        // Treat zero size as an empty-input error across public validators.
        if (n == 0)
        {
            return std::unexpected(GprError::empty_input());
        }
        return {};
    }

    namespace detail
    {

        /// @brief Formats a LengthMismatch error when actual != expected in require_count.
        /// @details Produces a diagnostic string naming the checked item, expected count, and actual count.
        /// @param[in] actual Observed element count.
        /// @param[in] expected Required element count.
        /// @param[in] what Human-readable label describing the checked quantity.
        /// @return GprError with code GprErrorCode::LengthMismatch.
        GprError make_count_mismatch_error(std::size_t actual, std::size_t expected, std::string_view what)
        {
            return GprError::length_mismatch(std::format("expected {} {}, got {}", expected, what, actual));
        }

    }  // namespace detail

    /// @brief Rejects any NaN or Inf entry in a double span with NonFiniteInput.
    /// @details Delegates to require_finite_impl<double>().
    /// @param[in] values Double-precision scalar span to inspect.
    /// @return Empty expected if all entries are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any element is NaN or +/-Inf.
    std::expected<void, GprError> require_finite(std::span<const double> values) noexcept
    {
        return require_finite_impl(values);
    }

    /// @brief Rejects any NaN or Inf entry in a float span with NonFiniteInput.
    /// @details Delegates to require_finite_impl<float>().
    /// @param[in] values Single-precision scalar span to inspect.
    /// @return Empty expected if all entries are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any element is NaN or +/-Inf.
    std::expected<void, GprError> require_finite(std::span<const float> values) noexcept
    {
        return require_finite_impl(values);
    }

    /// @brief Validates a non-empty, finite, column-major matrix span matching n_rows x n_cols.
    /// @details Validates shape header preconditions first and then inspects finiteness of x.
    /// @param[in] x Flat column-major matrix buffer.
    /// @param[in] n_rows Matrix row count.
    /// @param[in] n_cols Matrix column count.
    /// @param[in] name Matrix label included in length-mismatch diagnostics.
    /// @return Empty expected if valid, or GprError describing the first violated precondition.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, or x is empty.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x is NaN or +/-Inf.
    std::expected<void, GprError> validate_matrix(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::string_view name
    )
    {
        // Validate non-emptiness, dimension overflow, and span length before scanning entries.
        auto len = validate_shape_header(x.size(), n_rows, n_cols, name);
        if (!len)
        {
            return std::unexpected(std::move(len.error()));
        }
        return require_finite(x);
    }

    /// @brief Validates training inputs (X, y) in strict error-precedence order.
    /// @details Enforces EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch ->
    /// NonFiniteInput -> InsufficientData, including Gram matrix n_rows * n_rows overflow checks.
    /// @param[in] x Flat column-major feature matrix buffer of shape n_rows x n_cols.
    /// @param[in] n_rows Number of training samples.
    /// @param[in] n_cols Number of input feature dimensions.
    /// @param[in] y Target vector span of length n_rows.
    /// @param[in] expected_d Optional required feature dimension for kernel compatibility.
    /// @param[in] min_points Minimum required number of training points.
    /// @return Empty expected if all preconditions pass, or the highest-precedence GprError.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, x is empty, or y is empty.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols or n_rows * n_rows overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols or y.size() != n_rows.
    /// @retval GprErrorCode::DimensionMismatch When expected_d is set and n_cols != *expected_d.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::InsufficientData When n_rows < min_points.
    std::expected<void, GprError> validate_training(
        std::span<const double> x,
        std::size_t n_rows,
        std::size_t n_cols,
        std::span<const double> y,
        std::optional<std::size_t> expected_d,
        std::size_t min_points
    )
    {
        // Enforce strict validation total order:
        // EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch -> NonFiniteInput ->
        // InsufficientData.
        if (n_rows == 0 || n_cols == 0 || x.empty() || y.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        // Check both feature matrix size and n_rows x n_rows Gram matrix capacity for overflow.
        auto expected_x = checked_mul(n_rows, n_cols);
        if (!expected_x)
        {
            return std::unexpected(expected_x.error());
        }
        if (auto gram_len = checked_mul(n_rows, n_rows); !gram_len)
        {
            return std::unexpected(gram_len.error());
        }
        // Verify buffer lengths match declared matrix and target dimensions before dimension check.
        if (x.size() != *expected_x)
        {
            return std::unexpected(
                GprError::length_mismatch(std::format("expected {} feature values, got {}", *expected_x, x.size()))
            );
        }
        if (y.size() != n_rows)
        {
            return std::unexpected(
                GprError::length_mismatch(std::format("expected {} targets, got {}", n_rows, y.size()))
            );
        }
        // Verify feature dimensionality before scanning all floating-point entries.
        if (expected_d.has_value() && n_cols != *expected_d)
        {
            return std::unexpected(GprError::dimension_mismatch(n_cols, *expected_d));
        }
        if (auto ok = require_finite(x); !ok)
        {
            return ok;
        }
        if (auto ok = require_finite(y); !ok)
        {
            return ok;
        }
        // Check minimum sample count last so structural and finiteness errors take precedence.
        if (n_rows < min_points)
        {
            return std::unexpected(GprError::insufficient_data(n_rows, min_points));
        }
        return {};
    }

    /// @brief Validates target vector y and sample count n for distance-only training.
    /// @details Enforces EmptyInput -> SizeOverflow (n * n) -> LengthMismatch -> NonFiniteInput ->
    /// InsufficientData without requiring a feature matrix X.
    /// @param[in] n Number of training samples implied by supplied distance matrices.
    /// @param[in] y Target vector span of expected length n.
    /// @param[in] min_points Minimum required number of training points.
    /// @return Empty expected if valid, or the highest-precedence GprError.
    /// @retval GprErrorCode::EmptyInput When n == 0 or y is empty.
    /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When y.size() != n.
    /// @retval GprErrorCode::NonFiniteInput When y contains NaN or +/-Inf.
    /// @retval GprErrorCode::InsufficientData When n < min_points.
    std::expected<void, GprError> validate_distance_targets(
        std::size_t n, std::span<const double> y, std::size_t min_points
    )
    {
        // Follow the same validation order as validate_training:
        // EmptyInput -> SizeOverflow -> LengthMismatch -> NonFiniteInput -> InsufficientData.
        if (n == 0 || y.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto gram_len = checked_mul(n, n); !gram_len)
        {
            return std::unexpected(gram_len.error());
        }
        if (y.size() != n)
        {
            return std::unexpected(GprError::length_mismatch(std::format("expected {} targets, got {}", n, y.size())));
        }
        if (auto ok = require_finite(y); !ok)
        {
            return ok;
        }
        if (n < min_points)
        {
            return std::unexpected(GprError::insufficient_data(n, min_points));
        }
        return {};
    }

    /// @brief Validates query inputs Xs against the expected feature dimension.
    /// @details Enforces EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch ->
    /// NonFiniteInput for column-major query features.
    /// @param[in] xs Flat column-major query matrix buffer of shape n_rows x n_cols.
    /// @param[in] n_rows Number of query points.
    /// @param[in] n_cols Number of query feature columns.
    /// @param[in] expected_d Feature dimension required by the fitted model.
    /// @return Empty expected if valid, or GprError describing the violated precondition.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, or xs is empty.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When xs.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != expected_d.
    /// @retval GprErrorCode::NonFiniteInput When any entry of xs is NaN or +/-Inf.
    std::expected<void, GprError> validate_query(
        std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, std::size_t expected_d
    )
    {
        // Check shape header first, then column count, and finally finiteness of query coordinates.
        auto len = validate_shape_header(xs.size(), n_rows, n_cols, "query feature");
        if (!len)
        {
            return std::unexpected(std::move(len.error()));
        }
        if (n_cols != expected_d)
        {
            return std::unexpected(GprError::dimension_mismatch(n_cols, expected_d));
        }
        return require_finite(xs);
    }

}  // namespace cxxgpr
