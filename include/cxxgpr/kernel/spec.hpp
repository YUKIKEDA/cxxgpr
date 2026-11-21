// SPDX-License-Identifier: Apache-2.0
/// @file spec.hpp
/// @brief Declaration-layer composable kernel expression tree specification.
/// @details Declares ParameterBinding, CoordMode, KernelLeaf, and KernelSpec for composing arbitrary
/// sums and products of built-in and custom kernel leaves in double precision prior to compilation
/// into CompiledKernel<T>.
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

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Forward declaration of the execution-layer compiled kernel tree.
    /// @details Produced by KernelSpec::compile<T>() with associative flattening and leaf caches.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class CompiledKernel;

    /// @brief Maps an index in the flattened optimizer parameter vector to a leaf and local slot.
    /// @details Records depth-first left-to-right flattening metadata shared by KernelSpec,
    /// CompiledKernel, DistanceKernel, and CompiledDistanceKernel.
    struct ParameterBinding
    {
        /// @brief Zero-based index in the flattened kernel parameter vector.
        std::size_t index = 0;
        /// @brief Zero-based leaf index in depth-first, left-to-right traversal order.
        std::size_t leaf_id = 0;
        /// @brief Zero-based parameter index within that leaf.
        std::size_t local_index = 0;

        /// @brief Checks value equality between two ParameterBinding descriptors.
        /// @details Compares flattened index, leaf_id, and local_index.
        /// @param[in] lhs Left-hand ParameterBinding operand.
        /// @param[in] rhs Right-hand ParameterBinding operand.
        /// @return True if all three binding indices match.
        [[nodiscard]] friend constexpr bool operator==(
            const ParameterBinding& lhs, const ParameterBinding& rhs
        ) noexcept = default;
    };

    /// @brief Input representation required to evaluate a compiled kernel tree.
    /// @details Summarizes the coordinate-vs-distance capability across all leaves in an expression tree.
    enum class CoordMode : std::uint8_t
    {
        /// @brief Isotropic distance leaves only; accepts squared-distance matrices.
        Dist,
        /// @brief Coordinate-dependent leaves (ARD or Linear); requires input point matrices.
        Points,
        /// @brief Constant and White leaves only; accepts either distance shapes or coordinates.
        Either,
        /// @brief Mix of isotropic distance leaves and coordinate-dependent leaves.
        Mixed,
    };

    /// @brief Variant over all 10 built-in kernel leaf types and user-defined custom leaves.
    /// @details Holds value-semantic leaf configurations stored at the leaves of a KernelSpec tree.
    using KernelLeaf = std::variant<
        RbfKernel,
        RbfArdKernel,
        MaternKernel,
        MaternArdKernel,
        PeriodicKernel,
        RationalQuadraticKernel,
        RationalQuadraticArdKernel,
        ConstantKernel,
        LinearKernel,
        WhiteKernel,
        CustomKernel>;

    /// @brief Declaration-layer kernel expression tree holding double-precision parameters until compile.
    /// @details Supports immutable binary sum/product composition (+, *), atomic copy-on-write
    /// parameter updates in unconstrained log space, and compilation into CompiledKernel<T>.
    class KernelSpec
    {
    public:
        /// @brief Constructs a leaf specification from any built-in or custom kernel leaf.
        /// @details Implicitly converts any leaf type in KernelLeaf into a single-leaf KernelSpec.
        /// @param[in] leaf Built-in or custom kernel leaf value.
        /* implicit */ KernelSpec(KernelLeaf leaf);

        /// @brief Copy-constructs a kernel specification sharing child subtrees.
        /// @param[in] other Source kernel specification.
        KernelSpec(const KernelSpec& other) = default;

        /// @brief Move-constructs a kernel specification.
        /// @param[in,out] other Source kernel specification.
        KernelSpec(KernelSpec&& other) noexcept = default;

        /// @brief Copy-assigns a kernel specification.
        /// @param[in] other Source kernel specification.
        /// @return Reference to *this.
        KernelSpec& operator=(const KernelSpec& other) = default;

        /// @brief Move-assigns a kernel specification.
        /// @param[in,out] other Source kernel specification.
        /// @return Reference to *this.
        KernelSpec& operator=(KernelSpec&& other) noexcept = default;

        /// @brief Destroys the expression tree iteratively to bound stack usage on deep trees.
        ~KernelSpec() noexcept;

        /// @brief Constructs a leaf specification from a user-defined CustomKernel.
        /// @details Convenience factory wrapping a CustomKernel inside a leaf KernelSpec.
        /// @param[in] leaf Validated CustomKernel value wrapper.
        /// @return Single-leaf KernelSpec holding leaf.
        [[nodiscard]] static KernelSpec custom(CustomKernel leaf);

        /// @brief Constructs a sum of two kernel specifications.
        /// @details Combines k_sum(x, x') = lhs(x, x') + rhs(x, x') as an immutable binary AST node.
        /// @param[in] lhs Left kernel addend.
        /// @param[in] rhs Right kernel addend.
        /// @return Composite KernelSpec representing lhs + rhs.
        [[nodiscard]] static KernelSpec sum(KernelSpec lhs, KernelSpec rhs);

        /// @brief Constructs a product of two kernel specifications.
        /// @details Combines k_prod(x, x') = lhs(x, x') * rhs(x, x') as an immutable binary AST node.
        /// @param[in] lhs Left kernel factor.
        /// @param[in] rhs Right kernel factor.
        /// @return Composite KernelSpec representing lhs * rhs.
        [[nodiscard]] static KernelSpec product(KernelSpec lhs, KernelSpec rhs);

        /// @brief Returns the total number of log-space hyperparameters in the tree.
        /// @details Sums parameter counts across all leaves in the expression tree.
        /// @return Total number of kernel hyperparameters.
        [[nodiscard]] std::size_t num_params() const noexcept;

        /// @brief Writes flattened log-space hyperparameters in depth-first left-to-right leaf order.
        /// @details Verifies out.size() == num_params() before populating out.
        /// @param[out] out Destination span of length num_params().
        /// @return Empty expected on success or GprError on length mismatch or leaf query error.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates all leaf parameters from log-space params.
        /// @details Rebuilds a validated copy of the expression tree and replaces *this only if every
        /// leaf accepts its parameter slice, leaving *this unchanged on any error.
        /// @param[in] params Input span of unconstrained log-space parameters of length num_params().
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound violation.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any log-space parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exponentiated value falls outside its leaf interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes flattened open intervals in depth-first left-to-right leaf order.
        /// @details Order matches get_params() and set_params().
        /// @param[out] out Destination span of length num_params().
        /// @return Empty expected on success or GprError on length mismatch or leaf query error.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Returns flattened parameter-to-leaf bindings in depth-first left-to-right order.
        /// @details Each ParameterBinding maps a flat parameter index to its owning leaf and local slot.
        /// @return Vector of ParameterBinding descriptors of length num_params().
        [[nodiscard]] std::vector<ParameterBinding> parameter_bindings() const;

        /// @brief Returns the input coordinate mode required by the tree.
        /// @details Combines leaf CoordMode values recursively across sums and products.
        /// @return Aggregate CoordMode for the entire expression tree.
        [[nodiscard]] CoordMode coord_mode() const noexcept;

        /// @brief Compiles this specification into an associative-flattened execution kernel for scalar T.
        /// @details Invokes CompiledKernel<T>::from_spec(*this).
        /// @tparam T Kernel evaluation scalar type (double or float).
        /// @return Compiled execution-layer CompiledKernel<T>.
        template <KernelScalarConcept T> [[nodiscard]] CompiledKernel<T> compile() const;

        /// @brief Combines two kernel specifications by addition.
        /// @details Equivalent to KernelSpec::sum(lhs, rhs).
        /// @param[in] lhs Left kernel addend.
        /// @param[in] rhs Right kernel addend.
        /// @return Composite sum KernelSpec.
        friend KernelSpec operator+(KernelSpec lhs, KernelSpec rhs);

        /// @brief Combines two kernel specifications by multiplication.
        /// @details Equivalent to KernelSpec::product(lhs, rhs).
        /// @param[in] lhs Left kernel factor.
        /// @param[in] rhs Right kernel factor.
        /// @return Composite product KernelSpec.
        friend KernelSpec operator*(KernelSpec lhs, KernelSpec rhs);

    private:
        /// @brief Internal AST node representing lhs + rhs.
        /// @details Stores shared immutable pointers to both child specifications.
        struct SumNode
        {
            /// @brief Left addend subtree.
            mutable std::shared_ptr<const KernelSpec> lhs;
            /// @brief Right addend subtree.
            mutable std::shared_ptr<const KernelSpec> rhs;
        };

        /// @brief Internal AST node representing lhs * rhs.
        /// @details Stores shared immutable pointers to both child specifications.
        struct ProductNode
        {
            /// @brief Left factor subtree.
            mutable std::shared_ptr<const KernelSpec> lhs;
            /// @brief Right factor subtree.
            mutable std::shared_ptr<const KernelSpec> rhs;
        };

        /// @brief Empty tag disambiguating internal composite-node construction.
        struct InternalTag
        {
        };

        /// @brief Discriminated union over a leaf node, sum node, or product node.
        using NodeVariant = std::variant<KernelLeaf, SumNode, ProductNode>;

        /// @brief Private constructor initializing a composite SumNode or ProductNode.
        /// @details Disambiguated by InternalTag for internal factory methods.
        /// @param[in] node Constructed NodeVariant payload.
        KernelSpec(InternalTag, NodeVariant node) noexcept : node_(std::move(node)) {}

        /// @brief Stored AST variant node (leaf, sum, or product).
        NodeVariant node_;

        template <KernelScalarConcept T> friend class CompiledKernel;
        friend struct KernelSpecAccess;
        friend struct KernelSpecPersistAccess;
        friend struct DistanceKernelAccess;
    };

    /// @brief Combines two kernel specifications by addition.
    /// @details Equivalent to KernelSpec::sum(lhs, rhs).
    /// @param[in] lhs Left kernel addend.
    /// @param[in] rhs Right kernel addend.
    /// @return Composite sum KernelSpec.
    [[nodiscard]] KernelSpec operator+(KernelSpec lhs, KernelSpec rhs);

    /// @brief Combines two kernel specifications by multiplication.
    /// @details Equivalent to KernelSpec::product(lhs, rhs).
    /// @param[in] lhs Left kernel factor.
    /// @param[in] rhs Right kernel factor.
    /// @return Composite product KernelSpec.
    [[nodiscard]] KernelSpec operator*(KernelSpec lhs, KernelSpec rhs);

}  // namespace cxxgpr
