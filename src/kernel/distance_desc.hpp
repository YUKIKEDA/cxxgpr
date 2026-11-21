// SPDX-License-Identifier: Apache-2.0

/// @file distance_desc.hpp
/// @brief Library-private structural description of a DistanceKernel and of its validated distance tables.
/// @details Lets the persistence layer serialize a DistanceKernel and rebuild it with freshly issued slots without
/// exposing the kernel's internal tree in a public header.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/kernel/distance.hpp"
#include "cxxgpr/kernel/spec.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <vector>

/// @brief Library-internal helpers of the cxxgpr library.
namespace cxxgpr::detail
{

    /// @brief One node of the structural description of a DistanceKernel tree.
    struct DistanceNodeDesc
    {
        /// @brief Node category.
        enum class Kind
        {
            /// @brief A leaf reading a supplied distance slot.
            Supplied,
            /// @brief A leaf evaluated from coordinates, or a parameter-free constant or white leaf.
            Coord,
            /// @brief Sum of two subtrees.
            Sum,
            /// @brief Product of two subtrees.
            Product
        };

        /// @brief Node category.
        Kind kind{Kind::Coord};
        /// @brief Slot ordinal (position in DistanceKernel::slots()) of a Supplied leaf.
        std::size_t slot{0};
        /// @brief Kernel leaf of a Supplied or Coord node.
        std::optional<KernelLeaf> leaf;
        /// @brief Left subtree of a Sum or Product node.
        std::shared_ptr<const DistanceNodeDesc> lhs;
        /// @brief Right subtree of a Sum or Product node.
        std::shared_ptr<const DistanceNodeDesc> rhs;
    };

    /// @brief Kind of one distance slot of a described kernel.
    struct DistanceSlotDesc
    {
        /// @brief Number of ARD dimensions, or std::nullopt for a scalar slot.
        std::optional<std::size_t> ard_dims;
    };

    /// @brief Structural description of a DistanceKernel: its tree and its slots in DistanceKernel::slots() order.
    struct DistanceKernelDesc
    {
        /// @brief Root of the tree.
        std::shared_ptr<const DistanceNodeDesc> root;
        /// @brief Slots in depth-first order of first appearance.
        std::vector<DistanceSlotDesc> slots;
    };

    /// @brief Describes the tree and the slots of a DistanceKernel.
    /// @tparam C Distance mode.
    /// @param[in] kernel Kernel to describe.
    /// @return Structural description.
    template <DistanceModeConcept C>
    [[nodiscard]] DistanceKernelDesc describe_distance_kernel(const DistanceKernel<C>& kernel);

    /// @brief Rebuilds a DistanceKernel from its description, issuing a fresh slot for every described slot.
    /// @tparam C Distance mode.
    /// @param[in] desc Structural description.
    /// @return The kernel, or GprError when the description is inconsistent with the mode or the slots.
    /// @retval GprErrorCode::PersistFailed When a leaf does not fit its slot or the mode.
    template <DistanceModeConcept C>
    [[nodiscard]] std::expected<DistanceKernel<C>, GprError> build_distance_kernel(const DistanceKernelDesc& desc);

    /// @brief Exports the validated training tables of a store as lower-triangular packed arrays, one per slot.
    /// @details Slot k of the result follows the slot order of the kernel that built the store. A scalar slot holds
    /// n(n+1)/2 values at the packed position j*n - j(j-1)/2 + (i-j); an ARD slot holds dims blocks of that length.
    /// @tparam T Storage scalar of the store.
    /// @param[in] store Square store built by CompiledDistanceKernel::prepare_square.
    /// @return Packed arrays in slot order.
    template <KernelScalarConcept T>
    [[nodiscard]] std::vector<std::vector<T>> export_packed_distance_tables(const CheckedDistanceStore<T>& store);

}  // namespace cxxgpr::detail
