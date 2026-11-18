// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/transform/input.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace cxxgpr::transform
{

    /// Fitted per-column input feature transform applying an independent map to each feature column.
    class FittedColumnwiseInput final : public FittedInputTransform
    {
    public:
        /// Constructs an empty fitted columnwise transform.
        FittedColumnwiseInput() noexcept = default;

        /// Constructs a fitted columnwise transform from per-feature fitted maps.
        explicit FittedColumnwiseInput(std::vector<std::unique_ptr<FittedInputTransform>> maps) noexcept;

        /// Deep-copies the fitted columnwise transform.
        FittedColumnwiseInput(const FittedColumnwiseInput& other);

        /// Deep-copy assigns the fitted columnwise transform.
        FittedColumnwiseInput& operator=(const FittedColumnwiseInput& other);

        /// Moves the fitted columnwise transform.
        FittedColumnwiseInput(FittedColumnwiseInput&&) noexcept = default;

        /// Move-assigns the fitted columnwise transform.
        FittedColumnwiseInput& operator=(FittedColumnwiseInput&&) noexcept = default;

        ~FittedColumnwiseInput() override = default;

        /// Returns the number of per-column maps.
        [[nodiscard]] std::size_t size() const noexcept { return maps_.size(); }

        /// Returns true if no column maps are configured.
        [[nodiscard]] bool empty() const noexcept { return maps_.empty(); }

        /// Returns a read-only view of the fitted per-column maps.
        [[nodiscard]] std::span<const std::unique_ptr<FittedInputTransform>> maps() const noexcept { return maps_; }

        /// Applies each column's fitted map to its corresponding column in place.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// Applies each column's fitted map from x into out without allocation.
        [[nodiscard]] std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// Inverts each column's fitted map on its corresponding column in place.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// Inverts each column's fitted map from x into out without allocation.
        [[nodiscard]] std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// Returns a polymorphic deep copy of this fitted columnwise transform.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

    private:
        std::vector<std::unique_ptr<FittedInputTransform>> maps_;
    };

    /// Unfitted per-column input feature transform assigning one map per feature column.
    class ColumnwiseInput final : public UnfittedInputTransform
    {
    public:
        /// Constructs an empty columnwise transform specification.
        ColumnwiseInput() noexcept = default;

        /// Constructs a columnwise transform from per-feature unfitted maps.
        explicit ColumnwiseInput(std::vector<std::unique_ptr<UnfittedInputTransform>> maps) noexcept;

        /// Deep-copies the unfitted columnwise transform.
        ColumnwiseInput(const ColumnwiseInput& other);

        /// Deep-copy assigns the unfitted columnwise transform.
        ColumnwiseInput& operator=(const ColumnwiseInput& other);

        /// Moves the unfitted columnwise transform.
        ColumnwiseInput(ColumnwiseInput&&) noexcept = default;

        /// Move-assigns the unfitted columnwise transform.
        ColumnwiseInput& operator=(ColumnwiseInput&&) noexcept = default;

        ~ColumnwiseInput() override = default;

        /// Appends a map for the next feature column by cloning it.
        ColumnwiseInput& then(const UnfittedInputTransform& map) &;

        /// Appends a map for the next feature column by cloning it onto an rvalue.
        ColumnwiseInput&& then(const UnfittedInputTransform& map) &&;

        /// Appends an owned map for the next feature column.
        ColumnwiseInput& then(std::unique_ptr<UnfittedInputTransform> map) &;

        /// Appends an owned map for the next feature column onto an rvalue.
        ColumnwiseInput&& then(std::unique_ptr<UnfittedInputTransform> map) &&;

        /// Returns the number of configured column maps.
        [[nodiscard]] std::size_t size() const noexcept { return maps_.size(); }

        /// Returns true if no column maps are configured.
        [[nodiscard]] bool empty() const noexcept { return maps_.empty(); }

        /// Returns a read-only view of the unfitted per-column maps.
        [[nodiscard]] std::span<const std::unique_ptr<UnfittedInputTransform>> maps() const noexcept { return maps_; }

        /// Fits each column map independently on its column of x (requires n_cols == size()).
        [[nodiscard]] std::expected<FittedColumnwiseInput, GprError> fit_typed(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const;

        /// Fits each column map independently and returns a boxed fitted columnwise transform.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// Returns a polymorphic deep copy of this unfitted columnwise transform.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

    private:
        std::vector<std::unique_ptr<UnfittedInputTransform>> maps_;
    };

}  // namespace cxxgpr::transform
