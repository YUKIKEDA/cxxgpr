// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/kernel/custom.hpp"
#include "cxxgpr/kernel/leaves.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace cxxgpr {

/// Compiled execution-layer kernel tree parameterized by scalar type T.
template <KernelScalarConcept T> class CompiledKernel;

/// Maps an index in the flattened optimizer parameter vector to a leaf and local parameter slot.
struct ParameterBinding {
    /// Index in the flattened kernel parameter vector.
    std::size_t index = 0;
    /// Leaf index in depth-first, left-to-right order.
    std::size_t leaf_id = 0;
    /// Parameter index within that leaf.
    std::size_t local_index = 0;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const ParameterBinding& lhs,
                                                   const ParameterBinding& rhs) noexcept = default;
};

/// Input representation required to evaluate a compiled kernel tree.
enum class CoordMode : std::uint8_t {
    /// Isotropic distance leaves only; accepts squared-distance matrices.
    Dist,
    /// Coordinate-dependent leaves (ARD or Linear); requires input point matrices.
    Points,
    /// Constant and White leaves only; accepts either distance shapes or coordinates.
    Either,
    /// Mix of isotropic distance leaves and coordinate-dependent leaves.
    Mixed,
};

/// Variant over all built-in kernel leaf types and user-defined custom leaves.
using KernelLeaf = std::variant<RbfKernel, RbfArdKernel, MaternKernel, MaternArdKernel,
                                PeriodicKernel, RationalQuadraticKernel, RationalQuadraticArdKernel,
                                ConstantKernel, LinearKernel, WhiteKernel, CustomKernel>;

/// Declaration-layer kernel expression tree holding double-precision parameters until compile.
class KernelSpec {
public:
    /// Constructs a leaf specification from any built-in or custom kernel leaf.
    /* implicit */ KernelSpec(KernelLeaf leaf);

    /// Constructs a leaf specification from a user-defined CustomKernel.
    [[nodiscard]] static KernelSpec custom(CustomKernel leaf);

    /// Constructs a sum of two kernel specifications.
    [[nodiscard]] static KernelSpec sum(KernelSpec lhs, KernelSpec rhs);

    /// Constructs a product of two kernel specifications.
    [[nodiscard]] static KernelSpec product(KernelSpec lhs, KernelSpec rhs);

    /// Returns the total number of log-space hyperparameters in the tree.
    [[nodiscard]] std::size_t num_params() const noexcept;

    /// Writes flattened log-space hyperparameters in depth-first left-to-right leaf order.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates all leaf parameters from log-space params; leaves state unchanged on
    /// error.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes flattened open intervals in depth-first left-to-right leaf order.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Returns flattened parameter-to-leaf bindings in depth-first left-to-right order.
    [[nodiscard]] std::vector<ParameterBinding> parameter_bindings() const;

    /// Returns the input coordinate mode required by the tree.
    [[nodiscard]] CoordMode coord_mode() const noexcept;

    /// Compiles this specification into an associative-flattened execution kernel for scalar T.
    template <KernelScalarConcept T> [[nodiscard]] CompiledKernel<T> compile() const;

    /// Combines two kernel specifications by addition.
    friend KernelSpec operator+(KernelSpec lhs, KernelSpec rhs);

    /// Combines two kernel specifications by multiplication.
    friend KernelSpec operator*(KernelSpec lhs, KernelSpec rhs);

private:
    struct SumNode {
        std::shared_ptr<const KernelSpec> lhs;
        std::shared_ptr<const KernelSpec> rhs;
    };

    struct ProductNode {
        std::shared_ptr<const KernelSpec> lhs;
        std::shared_ptr<const KernelSpec> rhs;
    };

    struct InternalTag {};

    using NodeVariant = std::variant<KernelLeaf, SumNode, ProductNode>;

    KernelSpec(InternalTag, NodeVariant node) noexcept : node_(std::move(node)) {}

    NodeVariant node_;

    template <KernelScalarConcept T> friend class CompiledKernel;
    friend struct KernelSpecAccess;
    friend struct KernelSpecPersistAccess;
    friend struct DistanceKernelAccess;
};

/// Combines two kernel specifications by addition.
[[nodiscard]] KernelSpec operator+(KernelSpec lhs, KernelSpec rhs);

/// Combines two kernel specifications by multiplication.
[[nodiscard]] KernelSpec operator*(KernelSpec lhs, KernelSpec rhs);

} // namespace cxxgpr
