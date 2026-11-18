// SPDX-License-Identifier: Apache-2.0
/// @file columnwise.hpp
/// @brief Per-column heterogeneous input feature transform adapters.
/// @details Defines ColumnwiseInput and FittedColumnwiseInput to assign and apply distinct
/// FittedInputTransform/UnfittedInputTransform instances independently to each column of a column-major feature matrix.

#pragma once

#include "cxxgpr/transform/input.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

/// @brief Feature and target preprocessing transforms and composable pipelines.
namespace cxxgpr::transform
{

    /// @brief Fitted per-column input feature transform applying an independent map to each feature column.
    /// @details Dispatches each contiguous column slice (length n_rows) as an n_rows x 1 submatrix to the corresponding
    /// element of maps_.
    class FittedColumnwiseInput final : public FittedInputTransform
    {
    public:
        /// @brief Constructs an empty fitted columnwise transform.
        /// @details Holds zero column maps until populated or replaced by assignment.
        FittedColumnwiseInput() noexcept = default;

        /// @brief Constructs a fitted columnwise transform from per-feature fitted maps.
        /// @details Takes ownership of the provided per-column transform pointers.
        /// @param maps Ordered vector of fitted per-column input transforms.
        explicit FittedColumnwiseInput(std::vector<std::unique_ptr<FittedInputTransform>> maps) noexcept;

        /// @brief Deep-copies the fitted columnwise transform.
        /// @details Clones each non-null per-column map into a newly owned vector.
        /// @param other Source FittedColumnwiseInput to copy.
        FittedColumnwiseInput(const FittedColumnwiseInput& other);

        /// @brief Deep-copy assigns the fitted columnwise transform.
        /// @details Replaces owned column maps with deep clones of other's maps, guarding self-assignment.
        /// @param other Source FittedColumnwiseInput to copy from.
        /// @return Reference to *this.
        FittedColumnwiseInput& operator=(const FittedColumnwiseInput& other);

        /// @brief Moves the fitted columnwise transform.
        /// @details Transfers ownership of the underlying vector of column maps without cloning.
        FittedColumnwiseInput(FittedColumnwiseInput&&) noexcept = default;

        /// @brief Move-assigns the fitted columnwise transform.
        /// @details Transfers ownership of the underlying vector of column maps without cloning.
        /// @return Reference to *this.
        FittedColumnwiseInput& operator=(FittedColumnwiseInput&&) noexcept = default;

        /// @brief Destroys the fitted columnwise transform and its owned per-column maps.
        /// @details Overrides FittedInputTransform virtual destructor.
        ~FittedColumnwiseInput() override = default;

        /// @brief Returns the number of per-column maps.
        /// @details Must match n_cols passed to apply(), apply_into(), inverse_apply(), and inverse_apply_into().
        /// @return Count of configured feature column maps.
        [[nodiscard]] std::size_t size() const noexcept { return maps_.size(); }

        /// @brief Returns true if no column maps are configured.
        /// @details Equivalent to size() == 0.
        /// @return True when zero column maps are present.
        [[nodiscard]] bool empty() const noexcept { return maps_.empty(); }

        /// @brief Returns a read-only view of the fitted per-column maps.
        /// @details Preserves column order [0, size()).
        /// @return Read-only span of unique pointers to per-column fitted transforms.
        [[nodiscard]] std::span<const std::unique_ptr<FittedInputTransform>> maps() const noexcept { return maps_; }

        /// @brief Applies each column's fitted map to its corresponding column in place.
        /// @details Validates matrix dimensions against size() and invokes maps_[col]->apply on each contiguous column
        /// slice.
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
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Applies each column's fitted map from x into out without allocation.
        /// @details Validates input/output sizes against size() and invokes maps_[col]->apply_into per column slice.
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
        [[nodiscard]] std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Inverts each column's fitted map on its corresponding column in place.
        /// @details Validates matrix dimensions against size() and invokes maps_[col]->inverse_apply per column slice.
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
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Inverts each column's fitted map from x into out without allocation.
        /// @details Validates input/output sizes against size() and invokes maps_[col]->inverse_apply_into per column
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
        [[nodiscard]] std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted columnwise transform.
        /// @details Clones all per-column fitted transforms.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

    private:
        /// @brief Ordered list of fitted transforms, one per input feature column.
        std::vector<std::unique_ptr<FittedInputTransform>> maps_;
    };

    /// @brief Unfitted per-column input feature transform assigning one map per feature column.
    /// @details Enables composing heterogeneous columnwise preprocessing specs via fluent then() builder methods.
    class ColumnwiseInput final : public UnfittedInputTransform
    {
    public:
        /// @brief Constructs an empty columnwise transform specification.
        /// @details Must have at least one map added matching training n_cols before fitting.
        ColumnwiseInput() noexcept = default;

        /// @brief Constructs a columnwise transform from per-feature unfitted maps.
        /// @details Takes ownership of the provided vector of unfitted column maps.
        /// @param maps Ordered vector of unfitted per-column input transforms.
        explicit ColumnwiseInput(std::vector<std::unique_ptr<UnfittedInputTransform>> maps) noexcept;

        /// @brief Deep-copies the unfitted columnwise transform.
        /// @details Clones each non-null unfitted column map into a newly owned vector.
        /// @param other Source ColumnwiseInput to copy.
        ColumnwiseInput(const ColumnwiseInput& other);

        /// @brief Deep-copy assigns the unfitted columnwise transform.
        /// @details Replaces owned column maps with deep clones of other's maps, guarding self-assignment.
        /// @param other Source ColumnwiseInput to copy from.
        /// @return Reference to *this.
        ColumnwiseInput& operator=(const ColumnwiseInput& other);

        /// @brief Moves the unfitted columnwise transform.
        /// @details Transfers ownership of the underlying vector of column maps without cloning.
        ColumnwiseInput(ColumnwiseInput&&) noexcept = default;

        /// @brief Move-assigns the unfitted columnwise transform.
        /// @details Transfers ownership of the underlying vector of column maps without cloning.
        /// @return Reference to *this.
        ColumnwiseInput& operator=(ColumnwiseInput&&) noexcept = default;

        /// @brief Destroys the unfitted columnwise transform and its owned per-column specs.
        /// @details Overrides UnfittedInputTransform virtual destructor.
        ~ColumnwiseInput() override = default;

        /// @brief Appends a map for the next feature column by cloning it.
        /// @details Lvalue overload supporting chained calls on a named ColumnwiseInput variable.
        /// @param map Unfitted input transform to clone and append for the next feature column.
        /// @return Reference to *this.
        ColumnwiseInput& then(const UnfittedInputTransform& map) &;

        /// @brief Appends a map for the next feature column by cloning it onto an rvalue.
        /// @details Rvalue overload supporting fluent construction expressions.
        /// @param map Unfitted input transform to clone and append for the next feature column.
        /// @return Rvalue reference to *this.
        ColumnwiseInput&& then(const UnfittedInputTransform& map) &&;

        /// @brief Appends an owned map for the next feature column.
        /// @details Ignores null pointers rather than storing empty slots.
        /// @param map Owned unfitted input transform for the next feature column.
        /// @return Reference to *this.
        ColumnwiseInput& then(std::unique_ptr<UnfittedInputTransform> map) &;

        /// @brief Appends an owned map for the next feature column onto an rvalue.
        /// @details Ignores null pointers rather than storing empty slots.
        /// @param map Owned unfitted input transform for the next feature column.
        /// @return Rvalue reference to *this.
        ColumnwiseInput&& then(std::unique_ptr<UnfittedInputTransform> map) &&;

        /// @brief Returns the number of configured column maps.
        /// @details Training input n_cols must equal this value when fitting.
        /// @return Count of configured per-column maps.
        [[nodiscard]] std::size_t size() const noexcept { return maps_.size(); }

        /// @brief Returns true if no column maps are configured.
        /// @details Equivalent to size() == 0.
        /// @return True when zero column maps are present.
        [[nodiscard]] bool empty() const noexcept { return maps_.empty(); }

        /// @brief Returns a read-only view of the unfitted per-column maps.
        /// @details Preserves column order [0, size()).
        /// @return Read-only span of unique pointers to per-column unfitted transforms.
        [[nodiscard]] std::span<const std::unique_ptr<UnfittedInputTransform>> maps() const noexcept { return maps_; }

        /// @brief Fits each column map independently on its column of x (requires n_cols == size()).
        /// @details Validates matrix packing and column count against size(), then fits each column map on its
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
        [[nodiscard]] std::expected<FittedColumnwiseInput, GprError> fit_typed(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const;

        /// @brief Fits each column map independently and returns a boxed fitted columnwise transform.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedInputTransform>.
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
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted columnwise transform.
        /// @details Clones all per-column unfitted transform specs.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

    private:
        /// @brief Ordered list of unfitted transforms, one per expected feature column.
        std::vector<std::unique_ptr<UnfittedInputTransform>> maps_;
    };

}  // namespace cxxgpr::transform
