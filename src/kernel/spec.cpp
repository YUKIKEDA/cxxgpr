// SPDX-License-Identifier: Apache-2.0
/// @file spec.cpp
/// @brief Implementation of KernelSpec expression-tree composition and parameter traversal.
/// @details Implements depth-first parameter flattening, copy-on-write atomic parameter updates,
/// open interval aggregation, CoordMode lattice combination, and compilation dispatch.
#include "cxxgpr/kernel/spec.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/kernel/compiled.hpp"

#include <utility>

#include "kernel/leaf_eval.hpp"

namespace cxxgpr
{

    namespace
    {

        /// @brief Combines two child CoordMode requirements into their composite node requirement.
        /// @details CoordMode::Either acts as the identity element; combining Dist with Points yields
        /// CoordMode::Mixed.
        /// @param[in] a Coordinate mode of the left child subtree.
        /// @param[in] b Coordinate mode of the right child subtree.
        /// @return Combined CoordMode covering both subtrees.
        [[nodiscard]] CoordMode combine_coord_mode(CoordMode a, CoordMode b) noexcept
        {
            // Either (Constant/White) imposes no input representation constraint of its own.
            if (a == b)
            {
                return a;
            }
            if (a == CoordMode::Either)
            {
                return b;
            }
            if (b == CoordMode::Either)
            {
                return a;
            }
            return CoordMode::Mixed;
        }

        /// @brief Delegates leaf log-space parameter extraction to detail::leaf_get_params.
        /// @details Preserves uniform variant dispatch for KernelSpecAccess.
        /// @param[in] leaf Kernel leaf variant to query.
        /// @param[out] out Destination span sized to detail::leaf_num_params(leaf).
        /// @return Empty expected on success or GprError from the leaf.
        /// @retval GprErrorCode::LengthMismatch When out.size() does not match leaf parameter count.
        [[nodiscard]] std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out)
        {
            return detail::leaf_get_params(leaf, out);
        }

        /// @brief Delegates leaf open-interval extraction to detail::leaf_bounds_into.
        /// @details Preserves uniform variant dispatch for KernelSpecAccess.
        /// @param[in] leaf Kernel leaf variant to query.
        /// @param[out] out Destination span sized to detail::leaf_num_params(leaf).
        /// @return Empty expected on success or GprError from the leaf.
        /// @retval GprErrorCode::LengthMismatch When out.size() does not match leaf parameter count.
        [[nodiscard]] std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out)
        {
            return detail::leaf_bounds_into(leaf, out);
        }

    }  // namespace

    /// @brief Internal friend helper traversing KernelSpec trees for parameter and binding queries.
    /// @details Encapsulates recursive depth-first AST walkers without exposing internal tree nodes in
    /// the public header.
    struct KernelSpecAccess
    {
        /// @brief Recursively writes flattened log-space parameters starting at offset.
        /// @details Short-circuits subsequent subtrees once status enters an error state.
        /// @param[in] spec Subtree root specification to traverse.
        /// @param[out] out Full flattened output parameter span.
        /// @param[in] offset Starting index in out for this subtree.
        /// @param[in,out] status In-out status updated if any leaf parameter query fails.
        /// @return Updated offset immediately following this subtree's parameters.
        static std::size_t fill_params(
            const KernelSpec& spec, std::span<double> out, std::size_t offset, std::expected<void, GprError>& status
        )
        {
            if (!status.has_value())
            {
                return offset;
            }
            // Traverse depth-first left-to-right so parameter offsets stay canonical.
            if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_))
            {
                const std::size_t count = detail::leaf_num_params(*leaf);
                status = leaf_get_params(*leaf, out.subspan(offset, count));
                return offset + count;
            }
            if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_))
            {
                offset = fill_params(*s->lhs, out, offset, status);
                return fill_params(*s->rhs, out, offset, status);
            }
            const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
            offset = fill_params(*p.lhs, out, offset, status);
            return fill_params(*p.rhs, out, offset, status);
        }

        /// @brief Recursively writes flattened open parameter intervals starting at offset.
        /// @details Short-circuits subsequent subtrees once status enters an error state.
        /// @param[in] spec Subtree root specification to traverse.
        /// @param[out] out Full flattened output interval span.
        /// @param[in] offset Starting index in out for this subtree.
        /// @param[in,out] status In-out status updated if any leaf bounds query fails.
        /// @return Updated offset immediately following this subtree's intervals.
        static std::size_t fill_bounds(
            const KernelSpec& spec, std::span<Interval> out, std::size_t offset, std::expected<void, GprError>& status
        )
        {
            if (!status.has_value())
            {
                return offset;
            }
            // Traverse depth-first left-to-right to match flattened parameter indices.
            if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_))
            {
                const std::size_t count = detail::leaf_num_params(*leaf);
                status = leaf_bounds_into(*leaf, out.subspan(offset, count));
                return offset + count;
            }
            if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_))
            {
                offset = fill_bounds(*s->lhs, out, offset, status);
                return fill_bounds(*s->rhs, out, offset, status);
            }
            const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
            offset = fill_bounds(*p.lhs, out, offset, status);
            return fill_bounds(*p.rhs, out, offset, status);
        }

        /// @brief Constructs a new KernelSpec subtree with updated log-space leaf parameters.
        /// @details Performs copy-on-write traversal so any validation failure aborts without mutating
        /// the original shared AST nodes.
        /// @param[in] spec Source subtree root specification.
        /// @param[in] params Full flattened input parameter span.
        /// @param[in,out] offset In-out cursor tracking the current position in params.
        /// @return Updated KernelSpec subtree or GprError if any leaf rejects its slice.
        /// @retval GprErrorCode::NonFiniteInput When any log-space parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exponentiated value falls outside leaf bounds.
        static std::expected<KernelSpec, GprError> with_updated_params(
            const KernelSpec& spec, std::span<const double> params, std::size_t& offset
        )
        {
            // Clone each visited leaf before mutating so failed siblings never leave partial updates.
            if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_))
            {
                KernelLeaf copy = *leaf;
                const std::size_t count = detail::leaf_num_params(copy);
                auto res = std::visit([sub = params.subspan(offset, count)](auto& typed_leaf) {
                    return typed_leaf.set_params(sub);
                }, copy);
                if (!res.has_value())
                {
                    return std::unexpected(std::move(res).error());
                }
                offset += count;
                return KernelSpec(std::move(copy));
            }
            if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_))
            {
                auto lhs = with_updated_params(*s->lhs, params, offset);
                if (!lhs.has_value())
                {
                    return std::unexpected(std::move(lhs).error());
                }
                auto rhs = with_updated_params(*s->rhs, params, offset);
                if (!rhs.has_value())
                {
                    return std::unexpected(std::move(rhs).error());
                }
                return KernelSpec::sum(std::move(*lhs), std::move(*rhs));
            }
            const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
            auto lhs = with_updated_params(*p.lhs, params, offset);
            if (!lhs.has_value())
            {
                return std::unexpected(std::move(lhs).error());
            }
            auto rhs = with_updated_params(*p.rhs, params, offset);
            if (!rhs.has_value())
            {
                return std::unexpected(std::move(rhs).error());
            }
            return KernelSpec::product(std::move(*lhs), std::move(*rhs));
        }

        /// @brief Recursively emits ParameterBinding entries in depth-first left-to-right leaf order.
        /// @details Assigns monotonically increasing leaf_id and global parameter index values.
        /// @param[in] spec Subtree root specification.
        /// @param[in,out] next_leaf_id In-out leaf counter incremented once per visited leaf.
        /// @param[in,out] next_param_idx In-out global parameter counter incremented once per parameter.
        /// @param[in,out] out Destination vector receiving ParameterBinding descriptors.
        static void collect_bindings(
            const KernelSpec& spec,
            std::size_t& next_leaf_id,
            std::size_t& next_param_idx,
            std::vector<ParameterBinding>& out
        )
        {
            // Assign canonical depth-first leaf IDs and contiguous local parameter indices.
            if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_))
            {
                const std::size_t count = detail::leaf_num_params(*leaf);
                const std::size_t leaf_id = next_leaf_id++;
                for (std::size_t local = 0; local < count; ++local)
                {
                    out.push_back(
                        ParameterBinding{
                            .index = next_param_idx++,
                            .leaf_id = leaf_id,
                            .local_index = local,
                        }
                    );
                }
                return;
            }
            if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_))
            {
                collect_bindings(*s->lhs, next_leaf_id, next_param_idx, out);
                collect_bindings(*s->rhs, next_leaf_id, next_param_idx, out);
                return;
            }
            const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
            collect_bindings(*p.lhs, next_leaf_id, next_param_idx, out);
            collect_bindings(*p.rhs, next_leaf_id, next_param_idx, out);
        }
    };

    /// @brief Constructs a leaf specification from any built-in or custom kernel leaf.
    /// @details Moves leaf into the underlying NodeVariant.
    /// @param[in] leaf Built-in or custom kernel leaf value.
    KernelSpec::KernelSpec(KernelLeaf leaf) : node_(std::move(leaf)) {}

    /// @brief Constructs a leaf specification from a user-defined CustomKernel.
    /// @details Wraps leaf in KernelLeaf and returns a single-leaf KernelSpec.
    /// @param[in] leaf Validated CustomKernel value wrapper.
    /// @return Single-leaf KernelSpec holding leaf.
    KernelSpec KernelSpec::custom(CustomKernel leaf)
    {
        return {KernelLeaf(std::move(leaf))};
    }

    /// @brief Constructs a sum of two kernel specifications.
    /// @details Allocates shared immutable child nodes and wraps them in SumNode.
    /// @param[in] lhs Left kernel addend.
    /// @param[in] rhs Right kernel addend.
    /// @return Composite KernelSpec representing lhs + rhs.
    KernelSpec KernelSpec::sum(KernelSpec lhs, KernelSpec rhs)
    {
        auto lhs_ptr = std::make_shared<const KernelSpec>(std::move(lhs));
        auto rhs_ptr = std::make_shared<const KernelSpec>(std::move(rhs));
        SumNode node{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
        return {InternalTag{}, std::move(node)};
    }

    /// @brief Constructs a product of two kernel specifications.
    /// @details Allocates shared immutable child nodes and wraps them in ProductNode.
    /// @param[in] lhs Left kernel factor.
    /// @param[in] rhs Right kernel factor.
    /// @return Composite KernelSpec representing lhs * rhs.
    KernelSpec KernelSpec::product(KernelSpec lhs, KernelSpec rhs)
    {
        auto lhs_ptr = std::make_shared<const KernelSpec>(std::move(lhs));
        auto rhs_ptr = std::make_shared<const KernelSpec>(std::move(rhs));
        ProductNode node{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
        return {InternalTag{}, std::move(node)};
    }

    /// @brief Returns the total number of log-space hyperparameters in the tree.
    /// @details Sums parameter counts recursively across leaves, sums, and products.
    /// @return Total number of hyperparameters.
    std::size_t KernelSpec::num_params() const noexcept
    {
        if (const auto* leaf = std::get_if<KernelLeaf>(&node_))
        {
            return detail::leaf_num_params(*leaf);
        }
        if (const auto* s = std::get_if<SumNode>(&node_))
        {
            return s->lhs->num_params() + s->rhs->num_params();
        }
        if (const auto* p = std::get_if<ProductNode>(&node_))
        {
            return p->lhs->num_params() + p->rhs->num_params();
        }
        return 0;
    }

    /// @brief Writes flattened log-space hyperparameters in depth-first left-to-right leaf order.
    /// @details Validates out length before dispatching recursive depth-first traversal.
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch or leaf failure.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> KernelSpec::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        std::expected<void, GprError> status{};
        KernelSpecAccess::fill_params(*this, out, 0, status);
        return status;
    }

    /// @brief Atomically updates all leaf parameters from log-space params.
    /// @details Validates input length, builds a complete replacement tree via copy-on-write, and
    /// commits *this only when all leaves accept their new values.
    /// @param[in] params Input span of unconstrained log-space parameters of length num_params().
    /// @return Empty expected on success or GprError if any validation step fails.
    /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
    /// @retval GprErrorCode::NonFiniteInput When any log-space parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exponentiated parameter falls outside its leaf interval.
    std::expected<void, GprError> KernelSpec::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Build a complete updated tree before replacing *this to guarantee atomic rollback on failure.
        std::size_t offset = 0;
        auto updated = KernelSpecAccess::with_updated_params(*this, params, offset);
        if (!updated.has_value())
        {
            return std::unexpected(std::move(updated).error());
        }
        *this = std::move(*updated);
        return {};
    }

    /// @brief Writes flattened open intervals in depth-first left-to-right leaf order.
    /// @details Validates out length before traversing the tree in canonical parameter order.
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch or leaf failure.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> KernelSpec::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        std::expected<void, GprError> status{};
        KernelSpecAccess::fill_bounds(*this, out, 0, status);
        return status;
    }

    /// @brief Returns flattened parameter-to-leaf bindings in depth-first left-to-right order.
    /// @details Pre-allocates the exact vector capacity num_params() before collecting bindings.
    /// @return Vector of ParameterBinding descriptors matching the flattened parameter order.
    std::vector<ParameterBinding> KernelSpec::parameter_bindings() const
    {
        std::vector<ParameterBinding> bindings;
        bindings.reserve(num_params());
        std::size_t next_leaf_id = 0;
        std::size_t next_param_idx = 0;
        KernelSpecAccess::collect_bindings(*this, next_leaf_id, next_param_idx, bindings);
        return bindings;
    }

    /// @brief Returns the input coordinate mode required by the tree.
    /// @details Combines left and right subtree CoordMode requirements recursively.
    /// @return Aggregate CoordMode of the expression tree.
    CoordMode KernelSpec::coord_mode() const noexcept
    {
        if (const auto* leaf = std::get_if<KernelLeaf>(&node_))
        {
            return detail::leaf_coord_mode(*leaf);
        }
        if (const auto* s = std::get_if<SumNode>(&node_))
        {
            return combine_coord_mode(s->lhs->coord_mode(), s->rhs->coord_mode());
        }
        if (const auto* p = std::get_if<ProductNode>(&node_))
        {
            return combine_coord_mode(p->lhs->coord_mode(), p->rhs->coord_mode());
        }
        return CoordMode::Either;
    }

    /// @brief Compiles this specification into an associative-flattened execution kernel for scalar T.
    /// @details Delegates to CompiledKernel<T>::from_spec(*this).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @return Execution-layer CompiledKernel<T>.
    template <KernelScalarConcept T> CompiledKernel<T> KernelSpec::compile() const
    {
        return CompiledKernel<T>::from_spec(*this);
    }

    template CompiledKernel<double> KernelSpec::compile<double>() const;
    template CompiledKernel<float> KernelSpec::compile<float>() const;

    /// @brief Combines two kernel specifications by addition.
    /// @details Moves both operands into KernelSpec::sum().
    /// @param[in] lhs Left kernel addend.
    /// @param[in] rhs Right kernel addend.
    /// @return Composite sum KernelSpec.
    KernelSpec operator+(KernelSpec lhs, KernelSpec rhs)
    {
        return KernelSpec::sum(std::move(lhs), std::move(rhs));
    }

    /// @brief Combines two kernel specifications by multiplication.
    /// @details Moves both operands into KernelSpec::product().
    /// @param[in] lhs Left kernel factor.
    /// @param[in] rhs Right kernel factor.
    /// @return Composite product KernelSpec.
    KernelSpec operator*(KernelSpec lhs, KernelSpec rhs)
    {
        return KernelSpec::product(std::move(lhs), std::move(rhs));
    }

}  // namespace cxxgpr
