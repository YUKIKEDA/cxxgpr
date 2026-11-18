// SPDX-License-Identifier: Apache-2.0
/// @file columnwise.cpp
/// @brief Implementations of ColumnwiseInput and FittedColumnwiseInput feature transforms.
/// @details Implements deep copying, fluent builder chaining, per-column fitting, and contiguous column-slice dispatch
/// across column-major feature matrices.

#include "cxxgpr/transform/columnwise.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "transform/detail.hpp"

namespace cxxgpr::transform
{

    namespace
    {

        /// @brief Deep-clones a vector of polymorphic transform unique pointers.
        /// @details Skips any null pointers when populating the destination vector.
        /// @tparam Ptr Smart pointer type exposing clone().
        /// @param src Source vector of polymorphic transform pointers.
        /// @return Vector holding cloned copies of all non-null elements in src.
        template <typename Ptr> std::vector<Ptr> clone_ptr_vec(const std::vector<Ptr>& src)
        {
            // Reserve full capacity upfront and skip null entries while cloning each element.
            std::vector<Ptr> out;
            out.reserve(src.size());
            for (const auto& item : src)
            {
                if (item != nullptr)
                {
                    out.push_back(item->clone());
                }
            }
            return out;
        }

    }  // namespace

    // --- FittedColumnwiseInput ---

    /// @brief Constructs a fitted columnwise transform from per-feature fitted maps.
    /// @details Takes ownership of maps without modifying or validating individual entries until execution.
    /// @param maps Ordered vector of fitted per-column input transforms.
    FittedColumnwiseInput::FittedColumnwiseInput(std::vector<std::unique_ptr<FittedInputTransform>> maps) noexcept :
        maps_(std::move(maps))
    {
    }

    /// @brief Deep-copies the fitted columnwise transform.
    /// @details Clones every non-null per-column map via clone_ptr_vec.
    /// @param other Source FittedColumnwiseInput to copy.
    FittedColumnwiseInput::FittedColumnwiseInput(const FittedColumnwiseInput& other) : maps_(clone_ptr_vec(other.maps_))
    {
    }

    /// @brief Deep-copy assigns the fitted columnwise transform.
    /// @details Replaces owned column maps with deep clones of other's maps when &other != this.
    /// @param other Source FittedColumnwiseInput to copy from.
    /// @return Reference to *this.
    FittedColumnwiseInput& FittedColumnwiseInput::operator=(const FittedColumnwiseInput& other)
    {
        if (this != &other)
        {
            maps_ = clone_ptr_vec(other.maps_);
        }
        return *this;
    }

    /// @brief Applies each column's fitted map to its corresponding column in place.
    /// @details Validates matrix packing and column parity against maps_.size(), then dispatches each contiguous column
    /// slice as an n_rows x 1 view.
    /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must equal size()).
    /// @return Empty expected on success, or GprError on validation failure or null map entry.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any entry of maps() is null.
    std::expected<void, GprError> FittedColumnwiseInput::apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        // Enforce canonical matrix validation and require n_cols == maps_.size() before slicing columns.
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size()); !check)
        {
            return check;
        }
        // In column-major storage each column occupies a contiguous block [col * n_rows, (col + 1) * n_rows).
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->apply(x.subspan(col * n_rows, n_rows), n_rows, 1); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Applies each column's fitted map from x into out without allocation.
    /// @details Validates matrix packing, column parity, and out.size(), then dispatches each contiguous column slice
    /// to apply_into.
    /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must equal size()).
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation failure or null map entry.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any entry of maps() is null.
    std::expected<void, GprError> FittedColumnwiseInput::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size(), out.size()); !check)
        {
            return check;
        }
        // Dispatch each contiguous column slice directly into the matching output column slice.
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->apply_into(
                    x.subspan(col * n_rows, n_rows), n_rows, 1, out.subspan(col * n_rows, n_rows)
                );
                !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Inverts each column's fitted map on its corresponding column in place.
    /// @details Validates matrix packing and column parity against maps_.size(), then dispatches inverse_apply per
    /// column slice.
    /// @param x Mutable column-major transformed feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must equal size()).
    /// @return Empty expected on success, or GprError on validation failure or null map entry.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any entry of maps() is null.
    std::expected<void, GprError> FittedColumnwiseInput::inverse_apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size()); !check)
        {
            return check;
        }
        // Invert each column independently since columnwise maps have no cross-column coupling.
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->inverse_apply(x.subspan(col * n_rows, n_rows), n_rows, 1); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Inverts each column's fitted map from x into out without allocation.
    /// @details Validates matrix packing, column parity, and out.size(), then dispatches inverse_apply_into per column
    /// slice.
    /// @param x Read-only column-major transformed feature buffer of size n_rows * n_cols.
    /// @param n_rows Number of sample rows.
    /// @param n_cols Number of feature columns (must equal size()).
    /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError on validation failure or null map entry.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any entry of maps() is null.
    std::expected<void, GprError> FittedColumnwiseInput::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size(), out.size()); !check)
        {
            return check;
        }
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->inverse_apply_into(
                    x.subspan(col * n_rows, n_rows), n_rows, 1, out.subspan(col * n_rows, n_rows)
                );
                !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Returns a polymorphic deep copy of this fitted columnwise transform.
    /// @details Allocates a new FittedColumnwiseInput using the copy constructor.
    /// @return Unique pointer owning a deep copy of *this.
    std::unique_ptr<FittedInputTransform> FittedColumnwiseInput::clone() const
    {
        return std::make_unique<FittedColumnwiseInput>(*this);
    }

    // --- ColumnwiseInput ---

    /// @brief Constructs a columnwise transform from per-feature unfitted maps.
    /// @details Takes ownership of the provided vector of unfitted column maps.
    /// @param maps Ordered vector of unfitted per-column input transforms.
    ColumnwiseInput::ColumnwiseInput(std::vector<std::unique_ptr<UnfittedInputTransform>> maps) noexcept :
        maps_(std::move(maps))
    {
    }

    /// @brief Deep-copies the unfitted columnwise transform.
    /// @details Clones every non-null unfitted map via clone_ptr_vec.
    /// @param other Source ColumnwiseInput to copy.
    ColumnwiseInput::ColumnwiseInput(const ColumnwiseInput& other) : maps_(clone_ptr_vec(other.maps_)) {}

    /// @brief Deep-copy assigns the unfitted columnwise transform.
    /// @details Replaces owned maps with deep clones of other's maps when &other != this.
    /// @param other Source ColumnwiseInput to copy from.
    /// @return Reference to *this.
    ColumnwiseInput& ColumnwiseInput::operator=(const ColumnwiseInput& other)
    {
        if (this != &other)
        {
            maps_ = clone_ptr_vec(other.maps_);
        }
        return *this;
    }

    /// @brief Appends a map for the next feature column by cloning it.
    /// @details Clones map and pushes it to maps_ on an lvalue instance.
    /// @param map Unfitted input transform to clone and append for the next feature column.
    /// @return Reference to *this.
    ColumnwiseInput& ColumnwiseInput::then(const UnfittedInputTransform& map) &
    {
        maps_.push_back(map.clone());
        return *this;
    }

    /// @brief Appends a map for the next feature column by cloning it onto an rvalue.
    /// @details Clones map, pushes it to maps_, and returns std::move(*this).
    /// @param map Unfitted input transform to clone and append for the next feature column.
    /// @return Rvalue reference to *this.
    ColumnwiseInput&& ColumnwiseInput::then(const UnfittedInputTransform& map) &&
    {
        maps_.push_back(map.clone());
        return std::move(*this);
    }

    /// @brief Appends an owned map for the next feature column.
    /// @details Moves non-null map into maps_ and ignores null pointers.
    /// @param map Owned unfitted input transform for the next feature column.
    /// @return Reference to *this.
    ColumnwiseInput& ColumnwiseInput::then(std::unique_ptr<UnfittedInputTransform> map) &
    {
        // Ignore null pointers so valid fluent chains never introduce empty slots.
        if (map != nullptr)
        {
            maps_.push_back(std::move(map));
        }
        return *this;
    }

    /// @brief Appends an owned map for the next feature column onto an rvalue.
    /// @details Moves non-null map into maps_ and returns std::move(*this).
    /// @param map Owned unfitted input transform for the next feature column.
    /// @return Rvalue reference to *this.
    ColumnwiseInput&& ColumnwiseInput::then(std::unique_ptr<UnfittedInputTransform> map) &&
    {
        // Ignore null pointers so valid fluent chains never introduce empty slots.
        if (map != nullptr)
        {
            maps_.push_back(std::move(map));
        }
        return std::move(*this);
    }

    /// @brief Fits each column map independently on its column of x (requires n_cols == size()).
    /// @details Validates matrix packing and column count against maps_.size(), then fits each column map on its
    /// contiguous n_rows x 1 slice.
    /// @param x Read-only column-major training feature matrix.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns (must equal size()).
    /// @return FittedColumnwiseInput on success, or GprError on validation failure or null map entry.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any entry of maps() is null.
    std::expected<FittedColumnwiseInput, GprError> ColumnwiseInput::fit_typed(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        // Validate matrix shape, expected column count, and finiteness before fitting each column map.
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size()); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        std::vector<std::unique_ptr<FittedInputTransform>> fitted;
        fitted.reserve(maps_.size());
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            // Fit each column transform independently on its contiguous n_rows x 1 slice.
            auto fitted_col = maps_[col]->fit(x.subspan(col * n_rows, n_rows), n_rows, 1);
            if (!fitted_col)
            {
                return std::unexpected(std::move(fitted_col).error());
            }
            fitted.push_back(std::move(*fitted_col));
        }
        return FittedColumnwiseInput(std::move(fitted));
    }

    /// @brief Fits each column map independently and returns a boxed fitted columnwise transform.
    /// @details Delegates to fit_typed and wraps the typed result in std::unique_ptr<FittedInputTransform>.
    /// @param x Read-only column-major training feature matrix.
    /// @param n_rows Number of training sample rows.
    /// @param n_cols Number of feature columns (must equal size()).
    /// @return Boxed FittedColumnwiseInput on success, or GprError on validation failure or null map entry.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When n_cols != size().
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any entry of maps() is null.
    std::expected<std::unique_ptr<FittedInputTransform>, GprError> ColumnwiseInput::fit(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        auto typed = fit_typed(x, n_rows, n_cols);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedColumnwiseInput>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted columnwise transform.
    /// @details Allocates a new ColumnwiseInput using the copy constructor.
    /// @return Unique pointer owning a deep copy of *this.
    std::unique_ptr<UnfittedInputTransform> ColumnwiseInput::clone() const
    {
        return std::make_unique<ColumnwiseInput>(*this);
    }

}  // namespace cxxgpr::transform
