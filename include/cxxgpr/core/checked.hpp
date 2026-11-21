// SPDX-License-Identifier: Apache-2.0
/// @file checked.hpp
/// @brief Overflow-safe arithmetic and strict input validation helpers.
/// @details Provides checked size calculations, finite-value checks, and matrix/vector shape
/// validators used across cxxgpr APIs.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Adds two std::size_t counts, returning SizeOverflow on unsigned wraparound.
    /// @details Guards against unsigned integer wraparound by checking a <= max - b before adding.
    /// @param[in] a First non-negative addend.
    /// @param[in] b Second non-negative addend.
    /// @return Sum a + b, or GprError on unsigned overflow.
    /// @retval GprErrorCode::SizeOverflow When a + b exceeds std::numeric_limits<std::size_t>::max().
    [[nodiscard]] std::expected<std::size_t, GprError> checked_add(std::size_t a, std::size_t b) noexcept;

    /// @brief Multiplies two std::size_t counts, returning SizeOverflow on unsigned wraparound.
    /// @details Returns zero immediately if either factor is zero and otherwise checks a <= max / b.
    /// @param[in] a First non-negative factor.
    /// @param[in] b Second non-negative factor.
    /// @return Product a * b, or GprError on unsigned overflow.
    /// @retval GprErrorCode::SizeOverflow When a * b exceeds std::numeric_limits<std::size_t>::max().
    [[nodiscard]] std::expected<std::size_t, GprError> checked_mul(std::size_t a, std::size_t b) noexcept;

    /// @brief Computes n_rows * n_cols for non-empty column-major dimensions.
    /// @details Rejects zero dimensions before delegating to checked_mul().
    /// @param[in] n_rows Number of matrix rows (must be >= 1).
    /// @param[in] n_cols Number of matrix columns (must be >= 1).
    /// @return Total element count n_rows * n_cols, or GprError on empty or overflowing dimensions.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0 or n_cols == 0.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    [[nodiscard]] std::expected<std::size_t, GprError> column_major_len(
        std::size_t n_rows, std::size_t n_cols
    ) noexcept;

    /// @brief Computes packed lower-triangular length n * (n + 1) / 2 without false intermediate overflow.
    /// @details Halves the even factor before multiplying so n * (n + 1) never overflows prematurely.
    /// @param[in] n Matrix order (number of rows and columns).
    /// @return Packed triangular element count, or GprError on overflow.
    /// @retval GprErrorCode::SizeOverflow When n * (n + 1) / 2 overflows std::size_t.
    [[nodiscard]] std::expected<std::size_t, GprError> packed_len(std::size_t n) noexcept;

    /// @brief Internal implementation helpers for checked validation routines.
    namespace detail
    {

        /// @brief Formats a LengthMismatch error when actual != expected in require_count.
        /// @details Factored out of inline header checks so diagnostic string formatting stays out-of-line.
        /// @param[in] actual Observed element count.
        /// @param[in] expected Required element count.
        /// @param[in] what Human-readable label describing the checked quantity.
        /// @return GprError with code GprErrorCode::LengthMismatch.
        [[nodiscard]] GprError make_count_mismatch_error(
            std::size_t actual, std::size_t expected, std::string_view what
        );

        /// @brief Wraps a raw column-major span as a MatrixRef after checking its length against the dimensions.
        /// @details An empty span or a zero dimension is passed through unchanged so the downstream emptiness check
        /// reports EmptyInput. Otherwise n_rows * n_cols is computed with overflow checking and compared with
        /// data.size(), so a span shorter than the declared matrix is rejected instead of being read out of bounds.
        /// @param[in] data Contiguous column-major matrix data.
        /// @param[in] n_rows Declared number of rows.
        /// @param[in] n_cols Declared number of columns.
        /// @return View with leading dimension n_rows, or GprError on overflow or length mismatch.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When data.size() != n_rows * n_cols.
        [[nodiscard]] std::expected<MatrixRef<double>, GprError> matrix_view_from_span(
            std::span<const double> data, std::size_t n_rows, std::size_t n_cols
        );

    }  // namespace detail

    /// @brief Rejects a zero count with EmptyInput.
    /// @details Returns success whenever n > 0.
    /// @param[in] n Element or dimension count to validate.
    /// @return Empty expected on success, or GprError when n == 0.
    /// @retval GprErrorCode::EmptyInput When n == 0.
    [[nodiscard]] std::expected<void, GprError> require_nonempty(std::size_t n) noexcept;

    /// @brief Validates that actual equals expected, returning LengthMismatch otherwise.
    /// @details Fast-paths equality inline and delegates diagnostic formatting to detail::make_count_mismatch_error().
    /// @param[in] actual Observed count.
    /// @param[in] expected Required count.
    /// @param[in] what Human-readable label for the diagnostic message.
    /// @return Empty expected when counts match, or GprError otherwise.
    /// @retval GprErrorCode::LengthMismatch When actual != expected.
    [[nodiscard]] inline std::expected<void, GprError> require_count(
        std::size_t actual, std::size_t expected, std::string_view what
    )
    {
        // Fast-path exact match without constructing an error string.
        if (actual == expected)
        {
            return {};
        }
        return std::unexpected(detail::make_count_mismatch_error(actual, expected, what));
    }

    /// @brief Rejects any NaN or Inf entry in a double span with NonFiniteInput (subnormals are allowed).
    /// @details Inspects every element using std::isfinite; empty spans trivially succeed.
    /// @param[in] values Double-precision scalar span to inspect.
    /// @return Empty expected if all entries are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any element is NaN or +/-Inf.
    [[nodiscard]] std::expected<void, GprError> require_finite(std::span<const double> values) noexcept;

    /// @brief Rejects any NaN or Inf entry in a float span with NonFiniteInput (subnormals are allowed).
    /// @details Inspects every element using std::isfinite; empty spans trivially succeed.
    /// @param[in] values Single-precision scalar span to inspect.
    /// @return Empty expected if all entries are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any element is NaN or +/-Inf.
    [[nodiscard]] std::expected<void, GprError> require_finite(std::span<const float> values) noexcept;

    /// @brief Validates a non-empty, finite, column-major matrix span matching n_rows x n_cols.
    /// @details Checks emptiness, size overflow, buffer length, and finiteness in strict order.
    /// @param[in] x Flat column-major matrix buffer.
    /// @param[in] n_rows Matrix row count.
    /// @param[in] n_cols Matrix column count.
    /// @param[in] name Matrix label included in length-mismatch diagnostics.
    /// @return Empty expected if valid, or GprError describing the first violated precondition.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, or x is empty.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x is NaN or +/-Inf.
    [[nodiscard]] std::expected<void, GprError> validate_matrix(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::string_view name = "X"
    );

    /// @brief Validates training inputs (X, y) in strict error-precedence order.
    /// @details Enforces EmptyInput -> SizeOverflow -> LengthMismatch -> DimensionMismatch ->
    /// NonFiniteInput -> InsufficientData, including checking n_rows * n_rows Gram-matrix capacity.
    /// @param[in] x Flat column-major feature matrix buffer of shape n_rows x n_cols.
    /// @param[in] n_rows Number of training samples.
    /// @param[in] n_cols Number of input feature dimensions.
    /// @param[in] y Target vector span of length n_rows.
    /// @param[in] expected_d Optional required feature dimension for kernel compatibility.
    /// @param[in] min_points Minimum required number of training points.
    /// @return Empty expected if all training preconditions pass, or the highest-precedence GprError.
    /// @retval GprErrorCode::EmptyInput When n_rows == 0, n_cols == 0, x is empty, or y is empty.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols or n_rows * n_rows overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols or y.size() != n_rows.
    /// @retval GprErrorCode::DimensionMismatch When expected_d is set and n_cols != *expected_d.
    /// @retval GprErrorCode::NonFiniteInput When x or y contains NaN or +/-Inf.
    /// @retval GprErrorCode::InsufficientData When n_rows < min_points.
    [[nodiscard]] std::expected<void, GprError> validate_training(
        std::span<const double> x,
        std::size_t n_rows,
        std::size_t n_cols,
        std::span<const double> y,
        std::optional<std::size_t> expected_d = std::nullopt,
        std::size_t min_points = 1
    );

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
    [[nodiscard]] std::expected<void, GprError> validate_distance_targets(
        std::size_t n, std::span<const double> y, std::size_t min_points = 1
    );

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
    [[nodiscard]] std::expected<void, GprError> validate_query(
        std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, std::size_t expected_d
    );

}  // namespace cxxgpr
