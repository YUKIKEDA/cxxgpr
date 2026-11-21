// SPDX-License-Identifier: Apache-2.0
/// @file distance_store.hpp
/// @brief Internal staged validation of supplied-distance tables and reusable stores for the model engines.
/// @details The public prepare_square() and prepare_cross() validate a supply in one call and return a fresh store.
/// The model engines need the same checks split into phases, so that the shape checks of every input precede the
/// value checks of any table, and need to refill one store per fitted model without allocating after warm-up.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/distance.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <vector>

/// @brief Internal implementation details of cxxgpr.
namespace cxxgpr::detail
{

    /// @brief Common shape of the tables of a supply.
    struct DistanceTableShape
    {
        /// @brief Rows of every table.
        std::size_t rows{0};
        /// @brief Columns of every table.
        std::size_t cols{0};
    };

    /// @brief Checks the slots, emptiness, size products and shapes of a supply without reading table values.
    /// @details Order: InvalidConfig (tidy tolerance), UnsupportedKernelOperation (foreign slot), DistanceSlot
    /// (wrong kind, duplicate, missing), UnsupportedKernelOperation (kernel without slots), EmptyInput (empty table),
    /// SizeOverflow, LengthMismatch (non-square training table or tables of different shapes). Allocates nothing.
    /// @tparam T Storage scalar type of the compiled kernel.
    /// @param[in] kernel Compiled kernel defining the expected slots.
    /// @param[in] source Unchecked supply.
    /// @param[in] is_square True for a training supply, false for a cross supply.
    /// @return Common table shape, or the first violated GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<DistanceTableShape, GprError> check_distance_structure(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, bool is_square
    );

    /// @brief Checks that every ARD entry supplies the number of per-dimension tables its slot declares.
    /// @tparam T Storage scalar type of the compiled kernel.
    /// @param[in] kernel Compiled kernel defining the expected slots.
    /// @param[in] source Supply accepted by check_distance_structure().
    /// @return Empty expected, or DimensionMismatch.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> check_distance_ard_counts(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source
    );

    /// @brief Checks the column count of a point matrix against the coordinate leaves of a kernel.
    /// @tparam T Storage scalar type of the compiled kernel.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] cols Number of point columns.
    /// @return Empty expected, or DimensionMismatch when an ARD coordinate leaf needs another dimension.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> check_distance_point_cols(
        const CompiledDistanceKernel<T>& kernel, std::size_t cols
    );

    /// @brief Validates the table values of a supply into a reusable store.
    /// @details Requires that check_distance_structure() returned @p shape and check_distance_ard_counts()
    /// accepted @p source. The first call creates the store; later calls refill it in place and allocate only when
    /// the tables grow. After a failure the store holds partial data and must be refilled before use.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply to validate.
    /// @param[in] shape Shape returned by check_distance_structure().
    /// @param[in] is_square True for a training-shaped (square) store, false for a cross store.
    /// @param[in,out] store Reusable store.
    /// @return Pointer to the refilled store (never null), or InvalidDistance for the first invalid entry.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<const CheckedDistanceStore<T>*, GprError> fill_distance_store(
        const CompiledDistanceKernel<T>& kernel,
        const DistanceSource& source,
        DistanceTableShape shape,
        bool is_square,
        std::optional<CheckedDistanceStore<T>>& store
    );

    /// @brief Validates the table values of a supply into a new store.
    /// @details Requires that check_distance_structure() returned @p shape and check_distance_ard_counts()
    /// accepted @p source; equivalent to the value phase of prepare_square() or prepare_cross().
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply to validate.
    /// @param[in] shape Shape returned by check_distance_structure().
    /// @param[in] is_square True for a training-shaped (square) store, false for a cross store.
    /// @return New store, or InvalidDistance for the first invalid entry.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<CheckedDistanceStore<T>, GprError> make_distance_store(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, DistanceTableShape shape, bool is_square
    );

    /// @brief Evaluates the kernel diagonal with a caller-owned, grow-only scratch vector.
    /// @details Same contract as CompiledDistanceKernel::fill_diag(); repeated calls of the same or a smaller
    /// @p n reuse @p scratch and allocate nothing.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] n Number of diagonal entries.
    /// @param[in] x Optional point matrix of @p n rows.
    /// @param[out] out Destination of length @p n.
    /// @param[in,out] scratch Grow-only scratch storage.
    /// @return Empty expected, or GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> fill_distance_diag(
        const CompiledDistanceKernel<T>& kernel,
        std::size_t n,
        std::optional<MatrixRef<T>> x,
        std::span<T> out,
        std::vector<T>& scratch
    );

}  // namespace cxxgpr::detail
