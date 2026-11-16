// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/spec.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "kernel/leaf_eval.hpp"

#include <utility>

namespace cxxgpr {

namespace {

[[nodiscard]] CoordMode combine_coord_mode(CoordMode a, CoordMode b) noexcept {
    if (a == b) {
        return a;
    }
    if (a == CoordMode::Either) {
        return b;
    }
    if (b == CoordMode::Either) {
        return a;
    }
    return CoordMode::Mixed;
}

[[nodiscard]] std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf,
                                                            std::span<double> out) {
    return detail::leaf_get_params(leaf, out);
}

[[nodiscard]] std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf,
                                                             std::span<Interval> out) {
    return detail::leaf_bounds_into(leaf, out);
}

} // namespace

struct KernelSpecAccess {
    static std::size_t fill_params(const KernelSpec& spec, std::span<double> out,
                                   std::size_t offset, std::expected<void, GprError>& status) {
        if (!status.has_value()) {
            return offset;
        }
        if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_)) {
            const std::size_t count = detail::leaf_num_params(*leaf);
            status = leaf_get_params(*leaf, out.subspan(offset, count));
            return offset + count;
        }
        if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_)) {
            offset = fill_params(*s->lhs, out, offset, status);
            return fill_params(*s->rhs, out, offset, status);
        }
        const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
        offset = fill_params(*p.lhs, out, offset, status);
        return fill_params(*p.rhs, out, offset, status);
    }

    static std::size_t fill_bounds(const KernelSpec& spec, std::span<Interval> out,
                                   std::size_t offset, std::expected<void, GprError>& status) {
        if (!status.has_value()) {
            return offset;
        }
        if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_)) {
            const std::size_t count = detail::leaf_num_params(*leaf);
            status = leaf_bounds_into(*leaf, out.subspan(offset, count));
            return offset + count;
        }
        if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_)) {
            offset = fill_bounds(*s->lhs, out, offset, status);
            return fill_bounds(*s->rhs, out, offset, status);
        }
        const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
        offset = fill_bounds(*p.lhs, out, offset, status);
        return fill_bounds(*p.rhs, out, offset, status);
    }

    static std::expected<KernelSpec, GprError> with_updated_params(const KernelSpec& spec,
                                                                   std::span<const double> params,
                                                                   std::size_t& offset) {
        if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_)) {
            KernelLeaf copy = *leaf;
            const std::size_t count = detail::leaf_num_params(copy);
            auto res = std::visit([sub = params.subspan(offset, count)](
                                      auto& typed_leaf) { return typed_leaf.set_params(sub); },
                                  copy);
            if (!res.has_value()) {
                return std::unexpected(std::move(res).error());
            }
            offset += count;
            return KernelSpec(std::move(copy));
        }
        if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_)) {
            auto lhs = with_updated_params(*s->lhs, params, offset);
            if (!lhs.has_value()) {
                return std::unexpected(std::move(lhs).error());
            }
            auto rhs = with_updated_params(*s->rhs, params, offset);
            if (!rhs.has_value()) {
                return std::unexpected(std::move(rhs).error());
            }
            return KernelSpec::sum(std::move(*lhs), std::move(*rhs));
        }
        const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
        auto lhs = with_updated_params(*p.lhs, params, offset);
        if (!lhs.has_value()) {
            return std::unexpected(std::move(lhs).error());
        }
        auto rhs = with_updated_params(*p.rhs, params, offset);
        if (!rhs.has_value()) {
            return std::unexpected(std::move(rhs).error());
        }
        return KernelSpec::product(std::move(*lhs), std::move(*rhs));
    }

    static void collect_bindings(const KernelSpec& spec, std::size_t& next_leaf_id,
                                 std::size_t& next_param_idx, std::vector<ParameterBinding>& out) {
        if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_)) {
            const std::size_t count = detail::leaf_num_params(*leaf);
            const std::size_t leaf_id = next_leaf_id++;
            for (std::size_t local = 0; local < count; ++local) {
                out.push_back(ParameterBinding{
                    .index = next_param_idx++,
                    .leaf_id = leaf_id,
                    .local_index = local,
                });
            }
            return;
        }
        if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_)) {
            collect_bindings(*s->lhs, next_leaf_id, next_param_idx, out);
            collect_bindings(*s->rhs, next_leaf_id, next_param_idx, out);
            return;
        }
        const auto& p = std::get<KernelSpec::ProductNode>(spec.node_);
        collect_bindings(*p.lhs, next_leaf_id, next_param_idx, out);
        collect_bindings(*p.rhs, next_leaf_id, next_param_idx, out);
    }
};

KernelSpec::KernelSpec(KernelLeaf leaf) : node_(std::move(leaf)) {}

KernelSpec KernelSpec::custom(CustomKernel leaf) {
    return {KernelLeaf(std::move(leaf))};
}

KernelSpec KernelSpec::sum(KernelSpec lhs, KernelSpec rhs) {
    auto lhs_ptr = std::make_shared<const KernelSpec>(std::move(lhs));
    auto rhs_ptr = std::make_shared<const KernelSpec>(std::move(rhs));
    SumNode node{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
    return {InternalTag{}, std::move(node)};
}

KernelSpec KernelSpec::product(KernelSpec lhs, KernelSpec rhs) {
    auto lhs_ptr = std::make_shared<const KernelSpec>(std::move(lhs));
    auto rhs_ptr = std::make_shared<const KernelSpec>(std::move(rhs));
    ProductNode node{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
    return {InternalTag{}, std::move(node)};
}

std::size_t KernelSpec::num_params() const noexcept {
    if (const auto* leaf = std::get_if<KernelLeaf>(&node_)) {
        return detail::leaf_num_params(*leaf);
    }
    if (const auto* s = std::get_if<SumNode>(&node_)) {
        return s->lhs->num_params() + s->rhs->num_params();
    }
    if (const auto* p = std::get_if<ProductNode>(&node_)) {
        return p->lhs->num_params() + p->rhs->num_params();
    }
    return 0;
}

std::expected<void, GprError> KernelSpec::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::expected<void, GprError> status{};
    KernelSpecAccess::fill_params(*this, out, 0, status);
    return status;
}

std::expected<void, GprError> KernelSpec::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::size_t offset = 0;
    auto updated = KernelSpecAccess::with_updated_params(*this, params, offset);
    if (!updated.has_value()) {
        return std::unexpected(std::move(updated).error());
    }
    *this = std::move(*updated);
    return {};
}

std::expected<void, GprError> KernelSpec::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    std::expected<void, GprError> status{};
    KernelSpecAccess::fill_bounds(*this, out, 0, status);
    return status;
}

std::vector<ParameterBinding> KernelSpec::parameter_bindings() const {
    std::vector<ParameterBinding> bindings;
    bindings.reserve(num_params());
    std::size_t next_leaf_id = 0;
    std::size_t next_param_idx = 0;
    KernelSpecAccess::collect_bindings(*this, next_leaf_id, next_param_idx, bindings);
    return bindings;
}

CoordMode KernelSpec::coord_mode() const noexcept {
    if (const auto* leaf = std::get_if<KernelLeaf>(&node_)) {
        return detail::leaf_coord_mode(*leaf);
    }
    if (const auto* s = std::get_if<SumNode>(&node_)) {
        return combine_coord_mode(s->lhs->coord_mode(), s->rhs->coord_mode());
    }
    if (const auto* p = std::get_if<ProductNode>(&node_)) {
        return combine_coord_mode(p->lhs->coord_mode(), p->rhs->coord_mode());
    }
    return CoordMode::Either;
}

template <KernelScalarConcept T> CompiledKernel<T> KernelSpec::compile() const {
    return CompiledKernel<T>::from_spec(*this);
}

template CompiledKernel<double> KernelSpec::compile<double>() const;
template CompiledKernel<float> KernelSpec::compile<float>() const;

KernelSpec operator+(KernelSpec lhs, KernelSpec rhs) {
    return KernelSpec::sum(std::move(lhs), std::move(rhs));
}

KernelSpec operator*(KernelSpec lhs, KernelSpec rhs) {
    return KernelSpec::product(std::move(lhs), std::move(rhs));
}

} // namespace cxxgpr
