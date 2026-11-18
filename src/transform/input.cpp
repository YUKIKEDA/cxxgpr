// SPDX-License-Identifier: Apache-2.0
/// @file input.cpp
/// @brief Implementations of built-in input feature transforms and base out-of-place fallbacks.
/// @details Implements IdentityInput, StandardizeInput, MinMaxInput, their fitted counterparts, and default base class
/// out-of-place and persistence helpers.

#include "cxxgpr/transform/input.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "transform/detail.hpp"

namespace cxxgpr::transform
{

    namespace
    {

        /// @brief Validates a column-major matrix and invokes col_fn(col, col_span) on each column in place.
        /// @details Enforces expected_cols parity before slicing each contiguous column [col * n_rows, (col + 1) *
        /// n_rows).
        /// @tparam Fn Callable invocable as col_fn(std::size_t col, std::span<double> col_span).
        /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @param expected_cols Expected number of feature columns.
        /// @param col_fn Per-column in-place transformation callback.
        /// @return Empty expected on success, or GprError from canonical matrix validation.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != expected_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        template <typename Fn>
        std::expected<void, GprError> apply_columns_in_place(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols, std::size_t expected_cols, Fn&& col_fn
        )
        {
            // Enforce canonical input matrix validation with fitted column count check.
            if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, expected_cols); !check)
            {
                return check;
            }
            // Slice contiguous column spans so inner loops run sequentially in memory.
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                auto col_span = x.subspan(col * n_rows, n_rows);
                col_fn(col, col_span);
            }
            return {};
        }

        /// @brief Validates input/output matrices and invokes col_fn(col, src_col, dst_col) out of place.
        /// @details Enforces expected_cols and out.size() parity before slicing contiguous input and output columns.
        /// @tparam Fn Callable invocable as col_fn(std::size_t col, std::span<const double> src, std::span<double>
        /// dst).
        /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @param expected_cols Expected number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @param col_fn Per-column out-of-place transformation callback.
        /// @return Empty expected on success, or GprError from canonical matrix validation.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != expected_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        template <typename Fn>
        std::expected<void, GprError> apply_columns_into(
            std::span<const double> x,
            std::size_t n_rows,
            std::size_t n_cols,
            std::size_t expected_cols,
            std::span<double> out,
            Fn&& col_fn
        )
        {
            if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, expected_cols, out.size()); !check)
            {
                return check;
            }
            // Slice matching contiguous column spans across x and out with zero allocations.
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                auto src_col = x.subspan(col * n_rows, n_rows);
                auto dst_col = out.subspan(col * n_rows, n_rows);
                col_fn(col, src_col, dst_col);
            }
            return {};
        }

    }  // namespace

    /// @brief Applies the fitted transform out of place into a preallocated output buffer.
    /// @details Validates inputs and output size, copies x into out when non-aliasing, and delegates to apply().
    /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns.
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation or transform failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols disagrees with the fitted column count.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedInputTransform::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size()); !check)
        {
            return check;
        }
        // Skip redundant copy when caller supplies an aliasing in-place buffer.
        if (out.data() != x.data())
        {
            std::ranges::copy(x, out.begin());
        }
        return apply(out, n_rows, n_cols);
    }

    /// @brief Maps transformed features back to original coordinates into a preallocated buffer.
    /// @details Validates inputs and output size, copies x into out when non-aliasing, and delegates to
    /// inverse_apply().
    /// @param x Read-only column-major transformed feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns.
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation or inverse transform failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols disagrees with the fitted column count.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedInputTransform::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size()); !check)
        {
            return check;
        }
        // Skip redundant copy when caller supplies an aliasing in-place buffer.
        if (out.data() != x.data())
        {
            std::ranges::copy(x, out.begin());
        }
        return inverse_apply(out, n_rows, n_cols);
    }

    /// @brief Returns the JSON state string paired with persist_id.
    /// @details Base fallback reporting NotPersistable for transforms without custom JSON serialization.
    /// @return Unexpected GprErrorCode::PersistFailed with PersistErrorKind::NotPersistable.
    /// @retval GprErrorCode::PersistFailed Always returned unless overridden by a persistable subclass.
    std::expected<std::string, GprError> FittedInputTransform::persist_state_json() const
    {
        return std::unexpected(
            GprError::persist_failed(
                PersistErrorKind::NotPersistable, "this input transform does not implement persist_state"
            )
        );
    }

    /// @brief Returns the JSON state string paired with persist_id.
    /// @details Base fallback reporting NotPersistable for transforms without custom JSON serialization.
    /// @return Unexpected GprErrorCode::PersistFailed with PersistErrorKind::NotPersistable.
    /// @retval GprErrorCode::PersistFailed Always returned unless overridden by a persistable subclass.
    std::expected<std::string, GprError> UnfittedInputTransform::persist_state_json() const
    {
        return std::unexpected(
            GprError::persist_failed(
                PersistErrorKind::NotPersistable, "this input transform does not implement persist_state"
            )
        );
    }

    // --- IdentityInput / FittedIdentityInput ---

    /// @brief Validates packing and finiteness without modifying x.
    /// @details Delegates directly to detail::validate_input_matrix.
    /// @param x Mutable column-major feature buffer validated in place.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns.
    /// @return Empty expected on success, or GprError on invalid dimensions, packing, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedIdentityInput::apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        return detail::validate_input_matrix(x, n_rows, n_cols);
    }

    /// @brief Validates packing and finiteness and copies x into out.
    /// @details Guards aliasing via out.data() != x.data() before copying.
    /// @param x Read-only column-major feature buffer.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns.
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on invalid dimensions, buffer size, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedIdentityInput::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size()); !check)
        {
            return check;
        }
        if (out.data() != x.data())
        {
            std::ranges::copy(x, out.begin());
        }
        return {};
    }

    /// @brief Validates packing and finiteness without modifying x.
    /// @details Identity inverse equals forward application.
    /// @param x Mutable column-major feature buffer validated in place.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns.
    /// @return Empty expected on success, or GprError on invalid dimensions, packing, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedIdentityInput::inverse_apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        return apply(x, n_rows, n_cols);
    }

    /// @brief Validates packing and finiteness and copies x into out.
    /// @details Identity inverse equals forward application.
    /// @param x Read-only column-major feature buffer.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns.
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on invalid dimensions, buffer size, or non-finite values.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedIdentityInput::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        return apply_into(x, n_rows, n_cols, out);
    }

    /// @brief Returns a polymorphic deep copy of this fitted identity transform.
    /// @details Allocates a new FittedIdentityInput copy.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<FittedInputTransform> FittedIdentityInput::clone() const
    {
        return std::make_unique<FittedIdentityInput>(*this);
    }

    /// @brief Validates packing and finiteness and returns a typed fitted identity transform.
    /// @details Ensures invalid training matrices still surface errors consistently during fit().
    /// @param x Read-only column-major training feature buffer.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns.
    /// @return FittedIdentityInput on success, or GprError on invalid training matrix.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<FittedIdentityInput, GprError> IdentityInput::fit_typed(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        return FittedIdentityInput{};
    }

    /// @brief Validates packing and finiteness and returns a boxed fitted identity transform.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param x Read-only column-major training feature buffer.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns.
    /// @return Boxed FittedIdentityInput on success, or GprError on invalid training matrix.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<std::unique_ptr<FittedInputTransform>, GprError> IdentityInput::fit(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        auto typed = fit_typed(x, n_rows, n_cols);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedIdentityInput>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted identity transform.
    /// @details Allocates a new IdentityInput copy.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<UnfittedInputTransform> IdentityInput::clone() const
    {
        return std::make_unique<IdentityInput>(*this);
    }

    // --- StandardizeInput / FittedStandardizeInput ---

    /// @brief Constructs a FittedStandardizeInput from validated per-column means and positive stds.
    /// @details Enforces validation order: EmptyInput -> LengthMismatch -> NonFiniteInput -> InvalidHyperparameter.
    /// @param mean Per-column mean offsets.
    /// @param std Per-column positive scale factors.
    /// @return Validated FittedStandardizeInput on success, or GprError on invalid parameters.
    /// @retval GprErrorCode::EmptyInput When mean or std is empty.
    /// @retval GprErrorCode::LengthMismatch When std.size() != mean.size().
    /// @retval GprErrorCode::NonFiniteInput When any entry of mean or std is non-finite.
    /// @retval GprErrorCode::InvalidHyperparameter When any entry of std is <= 0.
    std::expected<FittedStandardizeInput, GprError> FittedStandardizeInput::from_parts(
        std::vector<double> mean, std::vector<double> std
    )
    {
        // Enforce validation order: empty -> length mismatch -> finiteness -> positive scale.
        if (mean.empty() || std.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto len_res = require_count(std.size(), mean.size(), "std"); !len_res)
        {
            return std::unexpected(std::move(len_res).error());
        }
        if (auto fin_mean = require_finite(mean); !fin_mean)
        {
            return std::unexpected(std::move(fin_mean).error());
        }
        if (auto fin_std = require_finite(std); !fin_std)
        {
            return std::unexpected(std::move(fin_std).error());
        }
        for (const double s : std)
        {
            if (s <= 0.0)
            {
                return std::unexpected(GprError::invalid_hyperparameter("fitted standardize scale must be positive"));
            }
        }
        return FittedStandardizeInput(std::move(mean), std::move(std));
    }

    /// @brief Standardizes features in place: (x - mean) / std.
    /// @details Applies z = (x - m) / s per column slice with s guaranteed positive.
    /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match mean().size()).
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeInput::apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        return apply_columns_in_place(
            x, n_rows, n_cols, mean_.size(), [this](std::size_t col, std::span<double> col_span) noexcept
        {
            const double m = mean_[col];
            const double s = std_[col];
            for (double& v : col_span)
            {
                v = (v - m) / s;
            }
        }
        );
    }

    /// @brief Standardizes features into out: (x - mean) / std.
    /// @details Writes standardized column slices into out without allocation.
    /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match mean().size()).
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeInput::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        return apply_columns_into(
            x, n_rows, n_cols, mean_.size(), out,
            [this](std::size_t col, std::span<const double> src, std::span<double> dst) noexcept
        {
            const double m = mean_[col];
            const double s = std_[col];
            for (std::size_t i = 0; i < src.size(); ++i)
            {
                dst[i] = (src[i] - m) / s;
            }
        }
        );
    }

    /// @brief Restores original feature coordinates in place: x * std + mean.
    /// @details Applies the affine inverse transform per column slice in place.
    /// @param x Mutable column-major standardized feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match mean().size()).
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeInput::inverse_apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        return apply_columns_in_place(
            x, n_rows, n_cols, mean_.size(), [this](std::size_t col, std::span<double> col_span) noexcept
        {
            const double m = mean_[col];
            const double s = std_[col];
            for (double& v : col_span)
            {
                v = (v * s) + m;
            }
        }
        );
    }

    /// @brief Restores original feature coordinates into out: x * std + mean.
    /// @details Writes original-coordinate column slices into out without allocation.
    /// @param x Read-only column-major standardized feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match mean().size()).
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedStandardizeInput::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        return apply_columns_into(
            x, n_rows, n_cols, mean_.size(), out,
            [this](std::size_t col, std::span<const double> src, std::span<double> dst) noexcept
        {
            const double m = mean_[col];
            const double s = std_[col];
            for (std::size_t i = 0; i < src.size(); ++i)
            {
                dst[i] = (src[i] * s) + m;
            }
        }
        );
    }

    /// @brief Returns a polymorphic deep copy of this fitted standardize transform.
    /// @details Allocates a copy preserving per-column mean and std vectors.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<FittedInputTransform> FittedStandardizeInput::clone() const
    {
        return std::make_unique<FittedStandardizeInput>(*this);
    }

    /// @brief Estimates per-column means and population stds, using 1.0 for constant columns.
    /// @details Validates matrix packing and finiteness before computing two-pass column statistics.
    /// @param x Read-only column-major training feature matrix.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns.
    /// @return FittedStandardizeInput on success, or GprError on invalid training input.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<FittedStandardizeInput, GprError> StandardizeInput::fit_typed(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        // Compute empirical mean and population std independently for each contiguous column.
        std::vector<double> mean;
        std::vector<double> std_dev;
        mean.reserve(n_cols);
        std_dev.reserve(n_cols);
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            const auto [m, s] = detail::compute_mean_and_population_std(x.subspan(col * n_rows, n_rows));
            mean.push_back(m);
            std_dev.push_back(s);
        }
        return FittedStandardizeInput::from_parts(std::move(mean), std::move(std_dev));
    }

    /// @brief Estimates per-column means and population stds and returns a boxed fitted transform.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param x Read-only column-major training feature matrix.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns.
    /// @return Boxed FittedStandardizeInput on success, or GprError on invalid training input.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<std::unique_ptr<FittedInputTransform>, GprError> StandardizeInput::fit(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        auto typed = fit_typed(x, n_rows, n_cols);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedStandardizeInput>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted standardize transform.
    /// @details Allocates a new StandardizeInput copy.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<UnfittedInputTransform> StandardizeInput::clone() const
    {
        return std::make_unique<StandardizeInput>(*this);
    }

    // --- MinMaxInput / FittedMinMaxInput ---

    /// @brief Constructs a FittedMinMaxInput from validated extrema and finite range [range_lo, range_hi].
    /// @details Enforces validation order: EmptyInput -> LengthMismatch -> NonFiniteInput -> InvalidConfig ->
    /// InvalidHyperparameter.
    /// @param data_min Per-column training minima.
    /// @param data_max Per-column training maxima.
    /// @param range_lo Finite lower endpoint of the target interval.
    /// @param range_hi Finite upper endpoint of the target interval (must exceed range_lo).
    /// @return Validated FittedMinMaxInput on success, or GprError on invalid extrema or range.
    /// @retval GprErrorCode::EmptyInput When data_min or data_max is empty.
    /// @retval GprErrorCode::LengthMismatch When data_max.size() != data_min.size().
    /// @retval GprErrorCode::NonFiniteInput When any entry of data_min or data_max is non-finite.
    /// @retval GprErrorCode::InvalidConfig When range_lo or range_hi is non-finite or range_hi <= range_lo.
    /// @retval GprErrorCode::InvalidHyperparameter When data_min[c] > data_max[c] for any column c.
    std::expected<FittedMinMaxInput, GprError> FittedMinMaxInput::from_parts(
        std::vector<double> data_min, std::vector<double> data_max, double range_lo, double range_hi
    )
    {
        // Enforce validation order: empty -> length match -> finiteness -> valid output range -> min <= max.
        if (data_min.empty() || data_max.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto len_res = require_count(data_max.size(), data_min.size(), "data_max"); !len_res)
        {
            return std::unexpected(std::move(len_res).error());
        }
        if (auto fin_min = require_finite(data_min); !fin_min)
        {
            return std::unexpected(std::move(fin_min).error());
        }
        if (auto fin_max = require_finite(data_max); !fin_max)
        {
            return std::unexpected(std::move(fin_max).error());
        }
        if (auto range_res = detail::validate_feature_range(range_lo, range_hi); !range_res)
        {
            return std::unexpected(std::move(range_res).error());
        }
        for (std::size_t col = 0; col < data_min.size(); ++col)
        {
            if (data_min[col] > data_max[col])
            {
                return std::unexpected(
                    GprError::invalid_hyperparameter(
                        std::format(
                            "fitted min-max input needs data_min <= data_max, column {} has {} > {}", col,
                            data_min[col], data_max[col]
                        )
                    )
                );
            }
        }
        return FittedMinMaxInput(std::move(data_min), std::move(data_max), range_lo, range_hi);
    }

    /// @brief Scales features in place onto [range_lo, range_hi].
    /// @details Applies range_lo + (range_hi - range_lo) * (v - data_min) / safe_span(data_max, data_min) per column.
    /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match min().size()).
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxInput::apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        // Scale by out_span / safe_span; constant columns use unit span 1.0 and map min_v to range_lo_.
        const double out_span = range_hi_ - range_lo_;
        return apply_columns_in_place(
            x, n_rows, n_cols, data_min_.size(), [this, out_span](std::size_t col, std::span<double> col_span) noexcept
        {
            const double min_v = data_min_[col];
            const double span = detail::safe_span(data_max_[col], min_v);
            for (double& v : col_span)
            {
                v = range_lo_ + (out_span * (v - min_v) / span);
            }
        }
        );
    }

    /// @brief Scales features into out onto [range_lo, range_hi].
    /// @details Writes scaled columns directly into out without allocation.
    /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match min().size()).
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxInput::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        const double out_span = range_hi_ - range_lo_;
        return apply_columns_into(
            x, n_rows, n_cols, data_min_.size(), out,
            [this, out_span](std::size_t col, std::span<const double> src, std::span<double> dst) noexcept
        {
            const double min_v = data_min_[col];
            const double span = detail::safe_span(data_max_[col], min_v);
            for (std::size_t i = 0; i < src.size(); ++i)
            {
                dst[i] = range_lo_ + (out_span * (src[i] - min_v) / span);
            }
        }
        );
    }

    /// @brief Restores original feature coordinates in place.
    /// @details Applies data_min + safe_span(data_max, data_min) * (v - range_lo) / (range_hi - range_lo) per column.
    /// @param x Mutable column-major scaled feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match min().size()).
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxInput::inverse_apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        // Invert affine min-max scaling using the reciprocal Jacobian factor span / out_span.
        const double out_span = range_hi_ - range_lo_;
        return apply_columns_in_place(
            x, n_rows, n_cols, data_min_.size(), [this, out_span](std::size_t col, std::span<double> col_span) noexcept
        {
            const double min_v = data_min_[col];
            const double span = detail::safe_span(data_max_[col], min_v);
            for (double& v : col_span)
            {
                v = min_v + (span * (v - range_lo_) / out_span);
            }
        }
        );
    }

    /// @brief Restores original feature coordinates into out.
    /// @details Writes unscaled feature columns directly into out without allocation.
    /// @param x Read-only column-major scaled feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must match min().size()).
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<void, GprError> FittedMinMaxInput::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        const double out_span = range_hi_ - range_lo_;
        return apply_columns_into(
            x, n_rows, n_cols, data_min_.size(), out,
            [this, out_span](std::size_t col, std::span<const double> src, std::span<double> dst) noexcept
        {
            const double min_v = data_min_[col];
            const double span = detail::safe_span(data_max_[col], min_v);
            for (std::size_t i = 0; i < src.size(); ++i)
            {
                dst[i] = min_v + (span * (src[i] - range_lo_) / out_span);
            }
        }
        );
    }

    /// @brief Returns a polymorphic deep copy of this fitted min-max transform.
    /// @details Allocates a copy preserving stored extrema and target interval bounds.
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<FittedInputTransform> FittedMinMaxInput::clone() const
    {
        return std::make_unique<FittedMinMaxInput>(*this);
    }

    /// @brief Creates a MinMaxInput mapping onto [lo, hi], rejecting non-finite bounds or hi <= lo.
    /// @details Validates target interval bounds via detail::validate_feature_range.
    /// @param lo Finite lower endpoint of the target output interval.
    /// @param hi Finite upper endpoint of the target output interval (must exceed lo).
    /// @return Validated MinMaxInput on success, or GprErrorCode::InvalidConfig on invalid bounds.
    /// @retval GprErrorCode::InvalidConfig When lo or hi is non-finite or hi <= lo.
    std::expected<MinMaxInput, GprError> MinMaxInput::create(double lo, double hi)
    {
        if (auto range_res = detail::validate_feature_range(lo, hi); !range_res)
        {
            return std::unexpected(std::move(range_res).error());
        }
        return MinMaxInput(lo, hi);
    }

    /// @brief Estimates per-column minima and maxima and returns a typed fitted transform.
    /// @details Validates matrix packing and finiteness before finding per-column extrema with
    /// std::ranges::minmax_element.
    /// @param x Read-only column-major training feature matrix.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns.
    /// @return FittedMinMaxInput on success, or GprError on invalid training input.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<FittedMinMaxInput, GprError> MinMaxInput::fit_typed(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        // Record per-column min and max across contiguous column-major slices.
        std::vector<double> data_min;
        std::vector<double> data_max;
        data_min.reserve(n_cols);
        data_max.reserve(n_cols);
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            const auto col_span = x.subspan(col * n_rows, n_rows);
            const auto [min_it, max_it] = std::ranges::minmax_element(col_span);
            data_min.push_back(*min_it);
            data_max.push_back(*max_it);
        }
        return FittedMinMaxInput::from_parts(std::move(data_min), std::move(data_max), range_lo_, range_hi_);
    }

    /// @brief Estimates per-column minima and maxima and returns a boxed fitted transform.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param x Read-only column-major training feature matrix.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns.
    /// @return Boxed FittedMinMaxInput on success, or GprError on invalid training input.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    std::expected<std::unique_ptr<FittedInputTransform>, GprError> MinMaxInput::fit(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        auto typed = fit_typed(x, n_rows, n_cols);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedMinMaxInput>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted min-max transform.
    /// @details Allocates a new MinMaxInput copy preserving feature_range().
    /// @return Unique pointer owning a copy of *this.
    std::unique_ptr<UnfittedInputTransform> MinMaxInput::clone() const
    {
        return std::make_unique<MinMaxInput>(*this);
    }

}  // namespace cxxgpr::transform
