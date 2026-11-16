// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/distance.hpp"

#include "cxxgpr/core/checked.hpp"
#include "kernel/leaf_eval.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <utility>
#include <variant>

namespace cxxgpr {

namespace {

std::atomic<std::uint64_t> g_next_slot_id{1};

[[nodiscard]] constexpr std::size_t packed_lower_index(std::size_t n, std::size_t row,
                                                       std::size_t col) noexcept {
    const std::size_t r = std::max(row, col);
    const std::size_t c = std::min(row, col);
    const std::size_t tri = (c > 0U) ? ((c * (c - 1U)) / 2U) : 0U;
    return ((c * n) - tri) + (r - c);
}

template <KernelScalarConcept T>
void add_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept {
    const std::size_t n_rows = dst.rows();
    const std::size_t n_cols = dst.cols();
    if (uplo == Triangle::Lower && n_rows == n_cols) {
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = col; row < n_rows; ++row) {
                dst(row, col) += src(row, col);
            }
        }
        return;
    }
    if (uplo == Triangle::Upper && n_rows == n_cols) {
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = 0; row <= col; ++row) {
                dst(row, col) += src(row, col);
            }
        }
        return;
    }
    for (std::size_t col = 0; col < n_cols; ++col) {
        for (std::size_t row = 0; row < n_rows; ++row) {
            dst(row, col) += src(row, col);
        }
    }
}

template <KernelScalarConcept T>
void mul_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept {
    const std::size_t n_rows = dst.rows();
    const std::size_t n_cols = dst.cols();
    if (uplo == Triangle::Lower && n_rows == n_cols) {
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = col; row < n_rows; ++row) {
                dst(row, col) *= src(row, col);
            }
        }
        return;
    }
    if (uplo == Triangle::Upper && n_rows == n_cols) {
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = 0; row <= col; ++row) {
                dst(row, col) *= src(row, col);
            }
        }
        return;
    }
    for (std::size_t col = 0; col < n_cols; ++col) {
        for (std::size_t row = 0; row < n_rows; ++row) {
            dst(row, col) *= src(row, col);
        }
    }
}

template <KernelScalarConcept T, typename Fn>
void visit_triangle(std::size_t n, Triangle uplo, Fn&& fn) {
    switch (uplo) {
    case Triangle::Lower:
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = col; row < n; ++row) {
                fn(row, col);
            }
        }
        return;
    case Triangle::Upper:
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = 0; row <= col; ++row) {
                fn(row, col);
            }
        }
        return;
    case Triangle::Full:
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = 0; row < n; ++row) {
                fn(row, col);
            }
        }
        return;
    }
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> require_scratch(MatrixRef<T> out,
                                                            MatrixMut<T> scratch) {
    if (scratch.rows() != out.rows() || scratch.cols() != out.cols()) {
        return std::unexpected(GprError::workspace_too_small());
    }
    if (scratch.data() == out.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffer must not alias output"));
    }
    return {};
}

[[nodiscard]] constexpr double matern_scale_sq_d(MaternNu nu) noexcept {
    switch (nu) {
    case MaternNu::OneHalf:
        return 1.0;
    case MaternNu::ThreeHalves:
        return 3.0;
    case MaternNu::FiveHalves:
        return 5.0;
    }
    return 1.0;
}

template <KernelScalarConcept T>
[[nodiscard]] T eval_matern_u_d(T u, MaternNu nu, KernelExp exp_mode) noexcept {
    if (u <= T{0}) {
        return T{1};
    }
    const T e = eval_kernel_exp(-u, exp_mode);
    switch (nu) {
    case MaternNu::OneHalf:
        return e;
    case MaternNu::ThreeHalves:
        return (T{1} + u) * e;
    case MaternNu::FiveHalves:
        return (T{1} + u + ((u * u) / T{3})) * e;
    }
    return e;
}

template <KernelScalarConcept T>
[[nodiscard]] T eval_matern_grad_g1_d(T u, MaternNu nu, KernelExp exp_mode) noexcept {
    if (u <= T{0}) {
        return T{0};
    }
    if (exp_mode == KernelExp::Accurate) {
        const T e = std::exp(-u);
        switch (nu) {
        case MaternNu::OneHalf:
            return u * e;
        case MaternNu::ThreeHalves:
            return (u * u) * e;
        case MaternNu::FiveHalves:
            return (((u * u) * (T{1} + u)) / T{3}) * e;
        }
        return T{0};
    }
    const ExpJet<T> jet = eval_kernel_exp_jet(-u, exp_mode);
    switch (nu) {
    case MaternNu::OneHalf:
        return u * jet.d1;
    case MaternNu::ThreeHalves:
        return u * (((T{1} + u) * jet.d1) - jet.v);
    case MaternNu::FiveHalves:
        return u *
               (((T{1} + u + ((u * u) / T{3})) * jet.d1) - ((T{1} + ((T{2} * u) / T{3})) * jet.v));
    }
    return T{0};
}

template <KernelScalarConcept T>
[[nodiscard]] T eval_matern_ard_cross_d(T u, MaternNu nu, KernelExp exp_mode) noexcept {
    if (u <= T{0}) {
        return T{0};
    }
    if (exp_mode == KernelExp::Accurate) {
        const T e = std::exp(-u);
        switch (nu) {
        case MaternNu::OneHalf:
            return u * (u + T{1}) * e;
        case MaternNu::ThreeHalves:
            return (u * u * u) * e;
        case MaternNu::FiveHalves:
            return ((u * u * u * u) / T{3}) * e;
        }
        return T{0};
    }
    const ExpJet<T> jet = eval_kernel_exp_jet(-u, exp_mode);
    switch (nu) {
    case MaternNu::OneHalf:
        return u * ((u * jet.d2) + jet.d1);
    case MaternNu::ThreeHalves: {
        const T d2k = ((T{1} + u) * jet.d2) - (T{2} * jet.d1);
        const T dk = jet.v - ((T{1} + u) * jet.d1);
        return (u * u * d2k) - (u * dk);
    }
    case MaternNu::FiveHalves: {
        const T poly = T{1} + u + ((u * u) / T{3});
        const T dpoly = T{1} + ((T{2} * u) / T{3});
        const T d2k = (poly * jet.d2) - (T{2} * dpoly * jet.d1) + ((T{2} / T{3}) * jet.v);
        const T dk = (dpoly * jet.v) - (poly * jet.d1);
        return (u * u * d2k) - (u * dk);
    }
    }
    return T{0};
}

template <KernelScalarConcept T>
void fill_square_sq_dist(MatrixRef<T> x, MatrixMut<T> dist) noexcept {
    const std::size_t n = x.rows();
    const std::size_t d = x.cols();
    for (std::size_t col = 0; col < n; ++col) {
        dist(col, col) = T{0};
        for (std::size_t row = col + 1U; row < n; ++row) {
            const std::size_t r1 = row;
            const std::size_t r2 = col;
            T acc = T{0};
            for (std::size_t k = 0; k < d; ++k) {
                const T diff = x(r1, k) - x(r2, k);
                acc += diff * diff;
            }
            dist(r1, r2) = acc;
            dist(r2, r1) = acc;
        }
    }
}

template <KernelScalarConcept T>
void fill_cross_sq_dist(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> dist) noexcept {
    const std::size_t n1 = x1.rows();
    const std::size_t n2 = x2.rows();
    const std::size_t d = x1.cols();
    for (std::size_t col = 0; col < n2; ++col) {
        for (std::size_t row = 0; row < n1; ++row) {
            T acc = T{0};
            for (std::size_t k = 0; k < d; ++k) {
                const T diff = x1(row, k) - x2(col, k);
                acc += diff * diff;
            }
            dist(row, col) = acc;
        }
    }
}

} // namespace

// --- Internal tree representation for DistanceKernel ---

struct SuppliedScalarNode {
    SlotId slot;
    KernelLeaf leaf;
};

struct SuppliedArdNode {
    SlotId slot;
    std::size_t dims = 0;
    KernelLeaf leaf;
};

struct CoordLeafNode {
    KernelLeaf leaf;
};

template <DistanceModeConcept Mode> struct DistanceKernel<Mode>::InternalNode {
    struct Sum {
        std::shared_ptr<const InternalNode> lhs;
        std::shared_ptr<const InternalNode> rhs;
    };
    struct Product {
        std::shared_ptr<const InternalNode> lhs;
        std::shared_ptr<const InternalNode> rhs;
    };

    std::variant<SuppliedScalarNode, SuppliedArdNode, CoordLeafNode, Sum, Product> data;
};

struct DistanceKernelAccess {
    template <DistanceModeConcept Mode> using Node = DistanceKernel<Mode>::InternalNode;

    template <DistanceModeConcept Mode>
    static std::shared_ptr<const Node<Mode>>
    promote_node(const DistanceKernel<DistanceOnly>::InternalNode& src) {
        if constexpr (std::same_as<Mode, DistanceOnly>) {
            return std::make_shared<const Node<Mode>>(src);
        } else {
            if (const auto* s = std::get_if<SuppliedScalarNode>(&src.data)) {
                return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = *s});
            }
            if (const auto* a = std::get_if<SuppliedArdNode>(&src.data)) {
                return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = *a});
            }
            if (const auto* c = std::get_if<CoordLeafNode>(&src.data)) {
                return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = *c});
            }
            if (const auto* sum =
                    std::get_if<DistanceKernel<DistanceOnly>::InternalNode::Sum>(&src.data)) {
                auto lhs_ptr = promote_node<WithPoints>(*sum->lhs);
                auto rhs_ptr = promote_node<WithPoints>(*sum->rhs);
                Node<WithPoints>::Sum snode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
                return std::make_shared<const Node<WithPoints>>(
                    Node<WithPoints>{.data = std::move(snode)});
            }
            const auto& prod =
                std::get<DistanceKernel<DistanceOnly>::InternalNode::Product>(src.data);
            auto lhs_ptr = promote_node<WithPoints>(*prod.lhs);
            auto rhs_ptr = promote_node<WithPoints>(*prod.rhs);
            Node<WithPoints>::Product pnode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
            return std::make_shared<const Node<WithPoints>>(
                Node<WithPoints>{.data = std::move(pnode)});
        }
    }

    static std::shared_ptr<const Node<WithPoints>> from_coord_spec(const KernelSpec& spec) {
        if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_)) {
            return std::make_shared<const Node<WithPoints>>(
                Node<WithPoints>{.data = CoordLeafNode{.leaf = *leaf}});
        }
        if (const auto* sum = std::get_if<KernelSpec::SumNode>(&spec.node_)) {
            auto lhs_ptr = from_coord_spec(*sum->lhs);
            auto rhs_ptr = from_coord_spec(*sum->rhs);
            Node<WithPoints>::Sum snode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
            return std::make_shared<const Node<WithPoints>>(
                Node<WithPoints>{.data = std::move(snode)});
        }
        const auto& prod = std::get<KernelSpec::ProductNode>(spec.node_);
        auto lhs_ptr = from_coord_spec(*prod.lhs);
        auto rhs_ptr = from_coord_spec(*prod.rhs);
        Node<WithPoints>::Product pnode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
        return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = std::move(pnode)});
    }

    template <DistanceModeConcept Mode>
    static std::size_t count_params(const Node<Mode>& node) noexcept {
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            return detail::leaf_num_params(s->leaf);
        }
        if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            return detail::leaf_num_params(a->leaf);
        }
        if (const auto* c = std::get_if<CoordLeafNode>(&node.data)) {
            return detail::leaf_num_params(c->leaf);
        }
        if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data)) {
            return count_params<Mode>(*sum->lhs) + count_params<Mode>(*sum->rhs);
        }
        if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data)) {
            return count_params<Mode>(*prod->lhs) + count_params<Mode>(*prod->rhs);
        }
        return 0;
    }

    template <DistanceModeConcept Mode>
    static std::size_t fill_params(const Node<Mode>& node, std::span<double> out,
                                   std::size_t offset, std::expected<void, GprError>& status) {
        if (!status.has_value()) {
            return offset;
        }
        const KernelLeaf* leaf = nullptr;
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            leaf = &s->leaf;
        } else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            leaf = &a->leaf;
        } else if (const auto* c = std::get_if<CoordLeafNode>(&node.data)) {
            leaf = &c->leaf;
        }
        if (leaf != nullptr) {
            const std::size_t count = detail::leaf_num_params(*leaf);
            status = detail::leaf_get_params(*leaf, out.subspan(offset, count));
            return offset + count;
        }
        if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data)) {
            offset = fill_params<Mode>(*sum->lhs, out, offset, status);
            return fill_params<Mode>(*sum->rhs, out, offset, status);
        }
        if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data)) {
            offset = fill_params<Mode>(*prod->lhs, out, offset, status);
            return fill_params<Mode>(*prod->rhs, out, offset, status);
        }
        return offset;
    }

    template <DistanceModeConcept Mode>
    static std::size_t fill_bounds(const Node<Mode>& node, std::span<Interval> out,
                                   std::size_t offset, std::expected<void, GprError>& status) {
        if (!status.has_value()) {
            return offset;
        }
        const KernelLeaf* leaf = nullptr;
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            leaf = &s->leaf;
        } else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            leaf = &a->leaf;
        } else if (const auto* c = std::get_if<CoordLeafNode>(&node.data)) {
            leaf = &c->leaf;
        }
        if (leaf != nullptr) {
            const std::size_t count = detail::leaf_num_params(*leaf);
            status = detail::leaf_bounds_into(*leaf, out.subspan(offset, count));
            return offset + count;
        }
        if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data)) {
            offset = fill_bounds<Mode>(*sum->lhs, out, offset, status);
            return fill_bounds<Mode>(*sum->rhs, out, offset, status);
        }
        if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data)) {
            offset = fill_bounds<Mode>(*prod->lhs, out, offset, status);
            return fill_bounds<Mode>(*prod->rhs, out, offset, status);
        }
        return offset;
    }

    template <DistanceModeConcept Mode>
    static std::expected<std::shared_ptr<const Node<Mode>>, GprError>
    with_updated_params(const Node<Mode>& node, std::span<const double> params,
                        std::size_t& offset) {
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            KernelLeaf copy = s->leaf;
            const std::size_t count = detail::leaf_num_params(copy);
            if (auto res = detail::leaf_set_params(copy, params.subspan(offset, count));
                !res.has_value()) {
                return std::unexpected(std::move(res).error());
            }
            offset += count;
            return std::make_shared<const Node<Mode>>(
                Node<Mode>{.data = SuppliedScalarNode{.slot = s->slot, .leaf = std::move(copy)}});
        }
        if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            KernelLeaf copy = a->leaf;
            const std::size_t count = detail::leaf_num_params(copy);
            if (auto res = detail::leaf_set_params(copy, params.subspan(offset, count));
                !res.has_value()) {
                return std::unexpected(std::move(res).error());
            }
            offset += count;
            return std::make_shared<const Node<Mode>>(Node<Mode>{
                .data =
                    SuppliedArdNode{.slot = a->slot, .dims = a->dims, .leaf = std::move(copy)}});
        }
        if (const auto* c = std::get_if<CoordLeafNode>(&node.data)) {
            KernelLeaf copy = c->leaf;
            const std::size_t count = detail::leaf_num_params(copy);
            if (auto res = detail::leaf_set_params(copy, params.subspan(offset, count));
                !res.has_value()) {
                return std::unexpected(std::move(res).error());
            }
            offset += count;
            return std::make_shared<const Node<Mode>>(
                Node<Mode>{.data = CoordLeafNode{.leaf = std::move(copy)}});
        }
        if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data)) {
            auto lhs = with_updated_params<Mode>(*sum->lhs, params, offset);
            if (!lhs.has_value()) {
                return std::unexpected(std::move(lhs).error());
            }
            auto rhs = with_updated_params<Mode>(*sum->rhs, params, offset);
            if (!rhs.has_value()) {
                return std::unexpected(std::move(rhs).error());
            }
            return std::make_shared<const Node<Mode>>(
                Node<Mode>{.data = typename Node<Mode>::Sum{.lhs = *lhs, .rhs = *rhs}});
        }
        const auto& prod = std::get<typename Node<Mode>::Product>(node.data);
        auto lhs = with_updated_params<Mode>(*prod.lhs, params, offset);
        if (!lhs.has_value()) {
            return std::unexpected(std::move(lhs).error());
        }
        auto rhs = with_updated_params<Mode>(*prod.rhs, params, offset);
        if (!rhs.has_value()) {
            return std::unexpected(std::move(rhs).error());
        }
        return std::make_shared<const Node<Mode>>(
            Node<Mode>{.data = typename Node<Mode>::Product{.lhs = *lhs, .rhs = *rhs}});
    }

    template <DistanceModeConcept Mode>
    static void collect_bindings(const Node<Mode>& node, std::size_t& next_leaf_id,
                                 std::size_t& next_param_idx, std::vector<ParameterBinding>& out) {
        const KernelLeaf* leaf = nullptr;
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            leaf = &s->leaf;
        } else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            leaf = &a->leaf;
        } else if (const auto* c = std::get_if<CoordLeafNode>(&node.data)) {
            leaf = &c->leaf;
        }
        if (leaf != nullptr) {
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
        if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data)) {
            collect_bindings<Mode>(*sum->lhs, next_leaf_id, next_param_idx, out);
            collect_bindings<Mode>(*sum->rhs, next_leaf_id, next_param_idx, out);
            return;
        }
        if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data)) {
            collect_bindings<Mode>(*prod->lhs, next_leaf_id, next_param_idx, out);
            collect_bindings<Mode>(*prod->rhs, next_leaf_id, next_param_idx, out);
        }
    }

    template <DistanceModeConcept Mode>
    static void collect_slots(const Node<Mode>& node, std::vector<SlotDescriptor>& out,
                              std::size_t& next_scalar, std::size_t& next_ard) {
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            for (const auto& existing : out) {
                if (existing.id == s->slot) {
                    return;
                }
            }
            out.push_back(
                SlotDescriptor{.id = s->slot, .ard_dims = std::nullopt, .at = next_scalar++});
            return;
        }
        if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            for (const auto& existing : out) {
                if (existing.id == a->slot) {
                    return;
                }
            }
            out.push_back(SlotDescriptor{.id = a->slot, .ard_dims = a->dims, .at = next_ard++});
            return;
        }
        if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data)) {
            collect_slots<Mode>(*sum->lhs, out, next_scalar, next_ard);
            collect_slots<Mode>(*sum->rhs, out, next_scalar, next_ard);
            return;
        }
        if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data)) {
            collect_slots<Mode>(*prod->lhs, out, next_scalar, next_ard);
            collect_slots<Mode>(*prod->rhs, out, next_scalar, next_ard);
        }
    }

    template <DistanceModeConcept Mode>
    static DistanceKernel<Mode> make_sum(DistanceKernel<Mode> lhs, DistanceKernel<Mode> rhs) {
        return DistanceKernel<Mode>(std::make_shared<const Node<Mode>>(
            Node<Mode>{.data = typename Node<Mode>::Sum{.lhs = std::move(lhs.root_),
                                                        .rhs = std::move(rhs.root_)}}));
    }

    template <DistanceModeConcept Mode>
    static DistanceKernel<Mode> make_product(DistanceKernel<Mode> lhs, DistanceKernel<Mode> rhs) {
        return DistanceKernel<Mode>(std::make_shared<const Node<Mode>>(
            Node<Mode>{.data = typename Node<Mode>::Product{.lhs = std::move(lhs.root_),
                                                            .rhs = std::move(rhs.root_)}}));
    }
};

// --- ScalarDistance & ArdDistance ---

ScalarDistance::ScalarDistance()
    : id_(SlotId(g_next_slot_id.fetch_add(1U, std::memory_order_relaxed))) {}

DistanceKernel<DistanceOnly> ScalarDistance::rbf(RbfKernel leaf) const {
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}}));
}

DistanceKernel<DistanceOnly> ScalarDistance::matern(MaternKernel leaf) const {
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}}));
}

DistanceKernel<DistanceOnly> ScalarDistance::periodic(PeriodicKernel leaf) const {
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}}));
}

DistanceKernel<DistanceOnly>
ScalarDistance::rational_quadratic(RationalQuadraticKernel leaf) const {
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}}));
}

DistanceKernel<DistanceOnly> ScalarDistance::custom(CustomKernel leaf) const {
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = std::move(leaf)}}));
}

std::expected<ArdDistance, GprError> ArdDistance::create(std::size_t dims) {
    if (dims == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    return ArdDistance(SlotId(g_next_slot_id.fetch_add(1U, std::memory_order_relaxed)), dims);
}

std::expected<DistanceKernel<DistanceOnly>, GprError>
ArdDistance::rbf_ard(RbfArdKernel leaf) const {
    if (leaf.num_params() != dims_) {
        return std::unexpected(GprError::dimension_mismatch(leaf.num_params(), dims_));
    }
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedArdNode{.slot = id_, .dims = dims_, .leaf = std::move(leaf)}}));
}

std::expected<DistanceKernel<DistanceOnly>, GprError>
ArdDistance::matern_ard(MaternArdKernel leaf) const {
    if (leaf.num_params() != dims_) {
        return std::unexpected(GprError::dimension_mismatch(leaf.num_params(), dims_));
    }
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedArdNode{.slot = id_, .dims = dims_, .leaf = std::move(leaf)}}));
}

std::expected<DistanceKernel<DistanceOnly>, GprError>
ArdDistance::rational_quadratic_ard(RationalQuadraticArdKernel leaf) const {
    const std::size_t leaf_dims = leaf.lengthscales().size();
    if (leaf_dims != dims_) {
        return std::unexpected(GprError::dimension_mismatch(leaf_dims, dims_));
    }
    return DistanceKernel<DistanceOnly>(
        std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedArdNode{.slot = id_, .dims = dims_, .leaf = std::move(leaf)}}));
}

// --- DistanceKernel<Mode> ---

template <DistanceModeConcept Mode>
DistanceKernel<Mode>::DistanceKernel(ConstantKernel leaf)
    : root_(
          std::make_shared<const InternalNode>(InternalNode{.data = CoordLeafNode{.leaf = leaf}})) {
}

template <DistanceModeConcept Mode>
DistanceKernel<Mode>::DistanceKernel(WhiteKernel leaf)
    : root_(
          std::make_shared<const InternalNode>(InternalNode{.data = CoordLeafNode{.leaf = leaf}})) {
}

template <DistanceModeConcept Mode>
DistanceKernel<Mode>::DistanceKernel(const DistanceKernel<DistanceOnly>& other)
    requires std::same_as<Mode, WithPoints>
    : root_(DistanceKernelAccess::promote_node<WithPoints>(*other.root_)) {}

template <DistanceModeConcept Mode>
DistanceKernel<Mode>::DistanceKernel(const KernelSpec& coord_spec)
    requires std::same_as<Mode, WithPoints>
    : root_(DistanceKernelAccess::from_coord_spec(coord_spec)) {}

template <DistanceModeConcept Mode>
DistanceKernel<WithPoints> DistanceKernel<Mode>::with_points() const {
    if constexpr (std::same_as<Mode, WithPoints>) {
        return *this;
    } else {
        return DistanceKernel<WithPoints>(*this);
    }
}

template <DistanceModeConcept Mode> std::size_t DistanceKernel<Mode>::num_params() const noexcept {
    return DistanceKernelAccess::count_params<Mode>(*root_);
}

template <DistanceModeConcept Mode>
std::expected<void, GprError> DistanceKernel<Mode>::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value()) {
        return cnt;
    }
    std::expected<void, GprError> status{};
    DistanceKernelAccess::fill_params<Mode>(*root_, out, 0, status);
    return status;
}

template <DistanceModeConcept Mode>
std::expected<void, GprError> DistanceKernel<Mode>::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value()) {
        return cnt;
    }
    std::size_t offset = 0;
    auto next = DistanceKernelAccess::with_updated_params<Mode>(*root_, params, offset);
    if (!next.has_value()) {
        return std::unexpected(std::move(next).error());
    }
    root_ = std::move(*next);
    return {};
}

template <DistanceModeConcept Mode>
std::expected<void, GprError> DistanceKernel<Mode>::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value()) {
        return cnt;
    }
    std::expected<void, GprError> status{};
    DistanceKernelAccess::fill_bounds<Mode>(*root_, out, 0, status);
    return status;
}

template <DistanceModeConcept Mode>
std::vector<ParameterBinding> DistanceKernel<Mode>::parameter_bindings() const {
    std::vector<ParameterBinding> out;
    out.reserve(num_params());
    std::size_t next_leaf = 0;
    std::size_t next_param = 0;
    DistanceKernelAccess::collect_bindings<Mode>(*root_, next_leaf, next_param, out);
    return out;
}

template <DistanceModeConcept Mode>
std::vector<SlotDescriptor> DistanceKernel<Mode>::slots() const {
    std::vector<SlotDescriptor> out;
    std::size_t next_scalar = 0;
    std::size_t next_ard = 0;
    DistanceKernelAccess::collect_slots<Mode>(*root_, out, next_scalar, next_ard);
    return out;
}

template <DistanceModeConcept Mode> std::size_t DistanceKernel<Mode>::num_scalar_slots() const {
    std::size_t count = 0;
    for (const auto& s : slots()) {
        if (!s.ard_dims.has_value()) {
            ++count;
        }
    }
    return count;
}

template <DistanceModeConcept Mode> std::size_t DistanceKernel<Mode>::num_ard_slots() const {
    std::size_t count = 0;
    for (const auto& s : slots()) {
        if (s.ard_dims.has_value()) {
            ++count;
        }
    }
    return count;
}

DistanceKernel<DistanceOnly> operator+(DistanceKernel<DistanceOnly> lhs,
                                       DistanceKernel<DistanceOnly> rhs) {
    return DistanceKernelAccess::make_sum<DistanceOnly>(std::move(lhs), std::move(rhs));
}

DistanceKernel<DistanceOnly> operator*(DistanceKernel<DistanceOnly> lhs,
                                       DistanceKernel<DistanceOnly> rhs) {
    return DistanceKernelAccess::make_product<DistanceOnly>(std::move(lhs), std::move(rhs));
}

DistanceKernel<WithPoints> operator+(DistanceKernel<WithPoints> lhs,
                                     DistanceKernel<WithPoints> rhs) {
    return DistanceKernelAccess::make_sum<WithPoints>(std::move(lhs), std::move(rhs));
}

DistanceKernel<WithPoints> operator*(DistanceKernel<WithPoints> lhs,
                                     DistanceKernel<WithPoints> rhs) {
    return DistanceKernelAccess::make_product<WithPoints>(std::move(lhs), std::move(rhs));
}

DistanceKernel<WithPoints> operator+(const DistanceKernel<DistanceOnly>& lhs,
                                     const KernelSpec& rhs) {
    return DistanceKernel<WithPoints>(lhs) + DistanceKernel<WithPoints>(rhs);
}

DistanceKernel<WithPoints> operator+(const KernelSpec& lhs,
                                     const DistanceKernel<DistanceOnly>& rhs) {
    return DistanceKernel<WithPoints>(lhs) + DistanceKernel<WithPoints>(rhs);
}

DistanceKernel<WithPoints> operator*(const DistanceKernel<DistanceOnly>& lhs,
                                     const KernelSpec& rhs) {
    return DistanceKernel<WithPoints>(lhs) * DistanceKernel<WithPoints>(rhs);
}

DistanceKernel<WithPoints> operator*(const KernelSpec& lhs,
                                     const DistanceKernel<DistanceOnly>& rhs) {
    return DistanceKernel<WithPoints>(lhs) * DistanceKernel<WithPoints>(rhs);
}

// --- DistanceSource ---

DistanceSource& DistanceSource::bind_scalar(SlotId slot, MatrixRef<double> d2) {
    scalars_.push_back(ScalarEntry{.slot = slot, .d2 = d2});
    return *this;
}

DistanceSource& DistanceSource::bind_ard(SlotId slot,
                                         std::span<const MatrixRef<double>> d2_per_dim) {
    ards_.push_back(ArdEntry{
        .slot = slot,
        .d2_per_dim = std::vector<MatrixRef<double>>(d2_per_dim.begin(), d2_per_dim.end())});
    return *this;
}

DistanceSource& DistanceSource::tidy(double rel_tol) {
    tidy_rel_tol_ = rel_tol;
    return *this;
}

// --- CheckedDistanceStore<T> ---

template <KernelScalarConcept T> struct CheckedDistanceStore<T>::Impl {
    std::vector<SlotDescriptor> kernel_slots;
    std::size_t n_rows = 0;
    std::size_t n_cols = 0;
    bool square = true;
    // Per scalar slot: dense n_rows x n_cols column-major data.
    std::vector<std::vector<T>> scalar_tables;
    // Per ARD slot (when square): packed lower triangles m * per_dim + p.
    // Per ARD slot (when cross): dims * n_rows * n_cols dense blocks.
    std::vector<std::size_t> ard_dims;
    std::vector<std::vector<T>> ard_data;
};

template <KernelScalarConcept T>
CheckedDistanceStore<T>::CheckedDistanceStore(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

template <KernelScalarConcept T> CheckedDistanceStore<T>::~CheckedDistanceStore() = default;

template <KernelScalarConcept T>
CheckedDistanceStore<T>::CheckedDistanceStore(const CheckedDistanceStore& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

template <KernelScalarConcept T>
CheckedDistanceStore<T>& CheckedDistanceStore<T>::operator=(const CheckedDistanceStore& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

template <KernelScalarConcept T> std::size_t CheckedDistanceStore<T>::rows() const noexcept {
    return impl_->n_rows;
}

template <KernelScalarConcept T> std::size_t CheckedDistanceStore<T>::cols() const noexcept {
    return impl_->n_cols;
}

template <KernelScalarConcept T> bool CheckedDistanceStore<T>::is_square() const noexcept {
    return impl_->square;
}

// --- CompiledDistanceKernel<T>::Impl ---

template <KernelScalarConcept T> struct CompiledDistanceKernel<T>::Impl {
    struct SuppliedScalarRef {
        std::size_t leaf_idx = 0;
        std::size_t slot_at = 0;
        std::size_t param_begin = 0;
        std::size_t param_count = 0;
    };

    struct SuppliedArdRef {
        std::size_t leaf_idx = 0;
        std::size_t slot_at = 0;
        std::size_t dims = 0;
        std::size_t param_begin = 0;
        std::size_t param_count = 0;
    };

    struct CoordLeafRef {
        std::size_t leaf_idx = 0;
        std::size_t param_begin = 0;
        std::size_t param_count = 0;
    };

    struct SumTerm;
    struct ProductTerm;
    using Term =
        std::variant<SuppliedScalarRef, SuppliedArdRef, CoordLeafRef, SumTerm, ProductTerm>;

    struct SumTerm {
        std::vector<Term> terms;
        std::size_t param_begin = 0;
        std::size_t param_end = 0;
    };

    struct ProductTerm {
        std::vector<Term> terms;
        std::size_t param_begin = 0;
        std::size_t param_end = 0;
    };

    std::vector<KernelLeaf> leaves;
    std::vector<std::vector<double>> cached_ls;
    std::vector<ParameterBinding> bindings;
    std::vector<SlotDescriptor> slots;
    std::size_t num_scalar_slots = 0;
    std::size_t num_ard_slots = 0;
    Term root;
    bool has_coord_points = false;
    std::size_t nested_depth = 0;
    KernelExp exp_mode = KernelExp::Accurate;

    void refresh_cached_ls() {
        cached_ls.resize(leaves.size());
        for (std::size_t idx = 0; idx < leaves.size(); ++idx) {
            if (const auto* k = std::get_if<RbfArdKernel>(&leaves[idx])) {
                cached_ls[idx] = k->lengthscales();
            } else if (const auto* km = std::get_if<MaternArdKernel>(&leaves[idx])) {
                cached_ls[idx] = km->lengthscales();
            } else if (const auto* krq = std::get_if<RationalQuadraticArdKernel>(&leaves[idx])) {
                cached_ls[idx] = krq->lengthscales();
            } else {
                cached_ls[idx].clear();
            }
        }
    }

    static std::pair<std::size_t, std::size_t> param_range(const Term& term) noexcept {
        if (const auto* s = std::get_if<SuppliedScalarRef>(&term)) {
            return {s->param_begin, s->param_begin + s->param_count};
        }
        if (const auto* a = std::get_if<SuppliedArdRef>(&term)) {
            return {a->param_begin, a->param_begin + a->param_count};
        }
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            return {c->param_begin, c->param_begin + c->param_count};
        }
        if (const auto* sum = std::get_if<SumTerm>(&term)) {
            return {sum->param_begin, sum->param_end};
        }
        if (const auto* prod = std::get_if<ProductTerm>(&term)) {
            return {prod->param_begin, prod->param_end};
        }
        return {0, 0};
    }

    static bool contains_param(const Term& term, std::size_t param_idx) noexcept {
        const auto [begin, end] = param_range(term);
        return param_idx >= begin && param_idx < end;
    }

    static bool is_leaf_term(const Term& term) noexcept {
        return std::holds_alternative<SuppliedScalarRef>(term) ||
               std::holds_alternative<SuppliedArdRef>(term) ||
               std::holds_alternative<CoordLeafRef>(term);
    }

    static std::size_t compute_nested_depth(const Term& term) noexcept {
        const std::vector<Term>* children = nullptr;
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            children = &s->terms;
        } else if (const auto* p = std::get_if<ProductTerm>(&term)) {
            children = &p->terms;
        } else {
            return 0;
        }
        std::size_t max_child = 0;
        for (const auto& child : *children) {
            if (!is_leaf_term(child)) {
                max_child = std::max(max_child, 1U + compute_nested_depth(child));
            }
        }
        return max_child;
    }

    static std::size_t find_slot_at(const std::vector<SlotDescriptor>& slot_list,
                                    SlotId id) noexcept {
        for (const auto& desc : slot_list) {
            if (desc.id == id) {
                return desc.at;
            }
        }
        return 0;
    }

    template <DistanceModeConcept Mode>
    static Term compile_node(const DistanceKernel<Mode>::InternalNode& node, Impl& out,
                             std::size_t& next_param_idx) {
        using Node = DistanceKernel<Mode>::InternalNode;
        if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data)) {
            const std::size_t leaf_idx = out.leaves.size();
            const std::size_t p_begin = next_param_idx;
            const std::size_t p_count = detail::leaf_num_params(s->leaf);
            out.leaves.push_back(s->leaf);
            for (std::size_t local = 0; local < p_count; ++local) {
                out.bindings.push_back(ParameterBinding{
                    .index = next_param_idx++, .leaf_id = leaf_idx, .local_index = local});
            }
            return SuppliedScalarRef{.leaf_idx = leaf_idx,
                                     .slot_at = find_slot_at(out.slots, s->slot),
                                     .param_begin = p_begin,
                                     .param_count = p_count};
        }
        if (const auto* a = std::get_if<SuppliedArdNode>(&node.data)) {
            const std::size_t leaf_idx = out.leaves.size();
            const std::size_t p_begin = next_param_idx;
            const std::size_t p_count = detail::leaf_num_params(a->leaf);
            out.leaves.push_back(a->leaf);
            for (std::size_t local = 0; local < p_count; ++local) {
                out.bindings.push_back(ParameterBinding{
                    .index = next_param_idx++, .leaf_id = leaf_idx, .local_index = local});
            }
            return SuppliedArdRef{.leaf_idx = leaf_idx,
                                  .slot_at = find_slot_at(out.slots, a->slot),
                                  .dims = a->dims,
                                  .param_begin = p_begin,
                                  .param_count = p_count};
        }
        if (const auto* c = std::get_if<CoordLeafNode>(&node.data)) {
            const std::size_t leaf_idx = out.leaves.size();
            const std::size_t p_begin = next_param_idx;
            const std::size_t p_count = detail::leaf_num_params(c->leaf);
            out.leaves.push_back(c->leaf);
            const CoordMode leaf_mode = detail::leaf_coord_mode(c->leaf);
            if (leaf_mode != CoordMode::Either) {
                out.has_coord_points = true;
            }
            for (std::size_t local = 0; local < p_count; ++local) {
                out.bindings.push_back(ParameterBinding{
                    .index = next_param_idx++, .leaf_id = leaf_idx, .local_index = local});
            }
            return CoordLeafRef{
                .leaf_idx = leaf_idx, .param_begin = p_begin, .param_count = p_count};
        }
        if (std::holds_alternative<typename Node::Sum>(node.data)) {
            std::vector<Term> flat;
            const std::size_t p_begin = next_param_idx;
            collect_sum<Mode>(node, flat, out, next_param_idx);
            return SumTerm{
                .terms = std::move(flat), .param_begin = p_begin, .param_end = next_param_idx};
        }
        std::vector<Term> flat;
        const std::size_t p_begin = next_param_idx;
        collect_product<Mode>(node, flat, out, next_param_idx);
        return ProductTerm{
            .terms = std::move(flat), .param_begin = p_begin, .param_end = next_param_idx};
    }

    template <DistanceModeConcept Mode>
    static void collect_sum(const DistanceKernel<Mode>::InternalNode& node,
                            std::vector<Term>& out_terms, Impl& out, std::size_t& next_param_idx) {
        using Node = DistanceKernel<Mode>::InternalNode;
        if (const auto* sum = std::get_if<typename Node::Sum>(&node.data)) {
            collect_sum<Mode>(*sum->lhs, out_terms, out, next_param_idx);
            collect_sum<Mode>(*sum->rhs, out_terms, out, next_param_idx);
            return;
        }
        out_terms.push_back(compile_node<Mode>(node, out, next_param_idx));
    }

    template <DistanceModeConcept Mode>
    static void collect_product(const DistanceKernel<Mode>::InternalNode& node,
                                std::vector<Term>& out_terms, Impl& out,
                                std::size_t& next_param_idx) {
        using Node = DistanceKernel<Mode>::InternalNode;
        if (const auto* prod = std::get_if<typename Node::Product>(&node.data)) {
            collect_product<Mode>(*prod->lhs, out_terms, out, next_param_idx);
            collect_product<Mode>(*prod->rhs, out_terms, out, next_param_idx);
            return;
        }
        out_terms.push_back(compile_node<Mode>(node, out, next_param_idx));
    }

    // --- Validation helpers for evaluation ---

    [[nodiscard]] std::expected<void, GprError> validate_store(const CheckedDistanceStore<T>& store,
                                                               bool expect_square,
                                                               MatrixRef<T> out) const {
        if (store.impl_->kernel_slots != slots) {
            return std::unexpected(GprError::unsupported_kernel_operation(
                "distance store does not match this kernel's slots"));
        }
        if (store.impl_->square != expect_square) {
            return std::unexpected(GprError::unsupported_kernel_operation(
                expect_square ? "expected square training distance store"
                              : "expected rectangular cross distance store"));
        }
        if (store.rows() == 0 || store.cols() == 0) {
            return std::unexpected(GprError::empty_input());
        }
        if (out.rows() != store.rows() || out.cols() != store.cols()) {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError> validate_ard_coord_dims(std::size_t x_dim) const {
        for (std::size_t idx = 0; idx < leaves.size(); ++idx) {
            // Check coordinate ARD leaves only (SuppliedArd leaves read their own distance slot).
            if (is_coord_leaf_idx(root, idx)) {
                const auto& leaf = leaves[idx];
                std::size_t expected_dim = 0;
                if (const auto* k = std::get_if<RbfArdKernel>(&leaf)) {
                    expected_dim = k->num_params();
                } else if (const auto* km = std::get_if<MaternArdKernel>(&leaf)) {
                    expected_dim = km->num_params();
                } else if (const auto* krq = std::get_if<RationalQuadraticArdKernel>(&leaf)) {
                    expected_dim = krq->lengthscales().size();
                }
                if (expected_dim > 0 && x_dim != expected_dim) {
                    return std::unexpected(GprError::dimension_mismatch(x_dim, expected_dim));
                }
            }
        }
        return {};
    }

    static bool is_coord_leaf_idx(const Term& term, std::size_t target_idx) noexcept {
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            return c->leaf_idx == target_idx;
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                if (is_coord_leaf_idx(child, target_idx)) {
                    return true;
                }
            }
        } else if (const auto* p = std::get_if<ProductTerm>(&term)) {
            for (const auto& child : p->terms) {
                if (is_coord_leaf_idx(child, target_idx)) {
                    return true;
                }
            }
        }
        return false;
    }

    [[nodiscard]] std::expected<void, GprError>
    validate_optional_points(std::optional<MatrixRef<T>> x, std::size_t expected_rows) const {
        if (!has_coord_points) {
            if (x.has_value()) {
                if (x->rows() == 0 || x->cols() == 0) {
                    return std::unexpected(GprError::empty_input());
                }
                if (x->rows() != expected_rows) {
                    return std::unexpected(GprError::length_mismatch("point row count mismatch"));
                }
            }
            return {};
        }
        if (!x.has_value()) {
            return std::unexpected(GprError::unsupported_kernel_operation(
                "WithPoints distance kernel with coordinate leaves requires point matrix"));
        }
        if (x->rows() == 0 || x->cols() == 0) {
            return std::unexpected(GprError::empty_input());
        }
        if (x->rows() != expected_rows) {
            return std::unexpected(GprError::length_mismatch("point row count mismatch"));
        }
        if (auto ard = validate_ard_coord_dims(x->cols()); !ard.has_value()) {
            return ard;
        }
        for (std::size_t col = 0; col < x->cols(); ++col) {
            for (std::size_t row = 0; row < x->rows(); ++row) {
                if (!std::isfinite((*x)(row, col))) {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        return {};
    }

    // --- ARD supplied evaluation / grad / hess ---

    [[nodiscard]] static T ard_scaled_sq_square(std::span<const T> packed, std::size_t n,
                                                std::size_t per_dim, std::span<const double> ls,
                                                std::size_t row, std::size_t col) noexcept {
        const std::size_t p = packed_lower_index(n, row, col);
        T acc = T{0};
        for (std::size_t m = 0; m < ls.size(); ++m) {
            const T ell = static_cast<T>(ls[m]);
            acc += packed[(m * per_dim) + p] / (ell * ell);
        }
        return acc;
    }

    [[nodiscard]] static T ard_scaled_sq_cross(std::span<const T> blocks, std::size_t n_rows,
                                               std::size_t n_cols, std::span<const double> ls,
                                               std::size_t row, std::size_t col) noexcept {
        const std::size_t block_len = n_rows * n_cols;
        const std::size_t elem_offset = (col * n_rows) + row;
        T acc = T{0};
        for (std::size_t m = 0; m < ls.size(); ++m) {
            const T ell = static_cast<T>(ls[m]);
            acc += blocks[(m * block_len) + elem_offset] / (ell * ell);
        }
        return acc;
    }

    void eval_supplied_ard_square(const SuppliedArdRef& ref, const CheckedDistanceStore<T>& store,
                                  MatrixMut<T> out, Triangle uplo) const {
        const std::size_t n = store.rows();
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        const std::span<const T> packed(store.impl_->ard_data[ref.slot_at]);
        const std::span<const double> ls(cached_ls[ref.leaf_idx]);
        const auto& leaf = leaves[ref.leaf_idx];
        if (std::holds_alternative<RbfArdKernel>(leaf)) {
            visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
                const T s2 = ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
                out(row, col) = eval_kernel_exp(-T{0.5} * s2, exp_mode);
            });
            return;
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf)) {
            const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
            const MaternNu nu = mat->nu();
            visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
                const T u2 = c * ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
                const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                out(row, col) = eval_matern_u_d(u, nu, exp_mode);
            });
            return;
        }
        const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
        const T alpha = static_cast<T>(rq.alpha());
        visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
            const T s2 = ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
            const T w = s2 / (T{2} * alpha);
            out(row, col) = eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
        });
    }

    void eval_supplied_ard_cross(const SuppliedArdRef& ref, const CheckedDistanceStore<T>& store,
                                 MatrixMut<T> out) const {
        const std::size_t n_rows = store.rows();
        const std::size_t n_cols = store.cols();
        const std::span<const T> blocks(store.impl_->ard_data[ref.slot_at]);
        const std::span<const double> ls(cached_ls[ref.leaf_idx]);
        const auto& leaf = leaves[ref.leaf_idx];
        if (std::holds_alternative<RbfArdKernel>(leaf)) {
            for (std::size_t col = 0; col < n_cols; ++col) {
                for (std::size_t row = 0; row < n_rows; ++row) {
                    const T s2 = ard_scaled_sq_cross(blocks, n_rows, n_cols, ls, row, col);
                    out(row, col) = eval_kernel_exp(-T{0.5} * s2, exp_mode);
                }
            }
            return;
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf)) {
            const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
            const MaternNu nu = mat->nu();
            for (std::size_t col = 0; col < n_cols; ++col) {
                for (std::size_t row = 0; row < n_rows; ++row) {
                    const T u2 = c * ard_scaled_sq_cross(blocks, n_rows, n_cols, ls, row, col);
                    const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                    out(row, col) = eval_matern_u_d(u, nu, exp_mode);
                }
            }
            return;
        }
        const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
        const T alpha = static_cast<T>(rq.alpha());
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = 0; row < n_rows; ++row) {
                const T s2 = ard_scaled_sq_cross(blocks, n_rows, n_cols, ls, row, col);
                const T w = s2 / (T{2} * alpha);
                out(row, col) = eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
            }
        }
    }

    void eval_supplied_ard_grad(const SuppliedArdRef& ref, const CheckedDistanceStore<T>& store,
                                MatrixMut<T> d_k, std::size_t local_idx, Triangle uplo) const {
        const std::size_t n = store.rows();
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        const std::span<const T> packed(store.impl_->ard_data[ref.slot_at]);
        const std::span<const double> ls(cached_ls[ref.leaf_idx]);
        const auto& leaf = leaves[ref.leaf_idx];
        if (std::holds_alternative<RbfArdKernel>(leaf)) {
            const T ell_p = static_cast<T>(ls[local_idx]);
            const T inv_l2 = T{1} / (ell_p * ell_p);
            visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
                const std::size_t p = packed_lower_index(n, row, col);
                const T s2 = ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
                const T sp2 = packed[(local_idx * per_dim) + p] * inv_l2;
                d_k(row, col) = sp2 * eval_kernel_exp_jet(-T{0.5} * s2, exp_mode).d1;
            });
            return;
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf)) {
            const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
            const MaternNu nu = mat->nu();
            const T ell_p = static_cast<T>(ls[local_idx]);
            const T inv_l2 = T{1} / (ell_p * ell_p);
            visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
                const std::size_t p = packed_lower_index(n, row, col);
                const T u2 = c * ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
                if (u2 <= T{0}) {
                    d_k(row, col) = T{0};
                    return;
                }
                const T u = std::sqrt(u2);
                const T wp = (c * packed[(local_idx * per_dim) + p] * inv_l2) / u2;
                d_k(row, col) = eval_matern_grad_g1_d(u, nu, exp_mode) * wp;
            });
            return;
        }
        const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
        const T alpha = static_cast<T>(rq.alpha());
        const std::size_t alpha_idx = ls.size();
        visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
            const std::size_t p = packed_lower_index(n, row, col);
            const T s2 = ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
            const T w = s2 / (T{2} * alpha);
            const T b = T{1} + w;
            const T d1_exp = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode).d1;
            if (local_idx < alpha_idx) {
                const T ell_p = static_cast<T>(ls[local_idx]);
                const T sp2 = packed[(local_idx * per_dim) + p] / (ell_p * ell_p);
                d_k(row, col) = (sp2 / b) * d1_exp;
            } else {
                d_k(row, col) = alpha * ((w / b) - std::log1p(w)) * d1_exp;
            }
        });
    }

    void eval_supplied_ard_hess(const SuppliedArdRef& ref, const CheckedDistanceStore<T>& store,
                                MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                                Triangle uplo) const {
        const std::size_t n = store.rows();
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        const std::span<const T> packed(store.impl_->ard_data[ref.slot_at]);
        const std::span<const double> ls(cached_ls[ref.leaf_idx]);
        const auto& leaf = leaves[ref.leaf_idx];
        if (std::holds_alternative<RbfArdKernel>(leaf)) {
            const T inv_li2 = T{1} / static_cast<T>(ls[i] * ls[i]);
            const T inv_lj2 = T{1} / static_cast<T>(ls[j] * ls[j]);
            visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
                const std::size_t p = packed_lower_index(n, row, col);
                const T s2 = ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
                const ExpJet<T> jet = eval_kernel_exp_jet(-T{0.5} * s2, exp_mode);
                const T si2 = packed[(i * per_dim) + p] * inv_li2;
                if (i == j) {
                    d2_k(row, col) = (si2 * si2 * jet.d2) - (T{2} * si2 * jet.d1);
                } else {
                    const T sj2 = packed[(j * per_dim) + p] * inv_lj2;
                    d2_k(row, col) = si2 * sj2 * jet.d2;
                }
            });
            return;
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf)) {
            const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
            const MaternNu nu = mat->nu();
            const T inv_li2 = T{1} / static_cast<T>(ls[i] * ls[i]);
            const T inv_lj2 = T{1} / static_cast<T>(ls[j] * ls[j]);
            visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
                const std::size_t p = packed_lower_index(n, row, col);
                const T u2 = c * ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
                if (u2 <= T{0}) {
                    d2_k(row, col) = T{0};
                    return;
                }
                const T u = std::sqrt(u2);
                const T wi = (c * packed[(i * per_dim) + p] * inv_li2) / u2;
                const T wj = (c * packed[(j * per_dim) + p] * inv_lj2) / u2;
                T val = eval_matern_ard_cross_d(u, nu, exp_mode) * wi * wj;
                if (i == j) {
                    val -= T{2} * eval_matern_grad_g1_d(u, nu, exp_mode) * wi;
                }
                d2_k(row, col) = val;
            });
            return;
        }
        const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
        const T alpha = static_cast<T>(rq.alpha());
        const std::size_t alpha_idx = ls.size();
        visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
            const std::size_t p = packed_lower_index(n, row, col);
            const T s2 = ard_scaled_sq_square(packed, n, per_dim, ls, row, col);
            const T w = s2 / (T{2} * alpha);
            const T b = T{1} + w;
            const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
            const T g_alpha = alpha * ((w / b) - std::log1p(w));
            if (i == alpha_idx && j == alpha_idx) {
                d2_k(row, col) = ((g_alpha * g_alpha) * jet.d2) +
                                 ((g_alpha + ((alpha * w * w) / (b * b))) * jet.d1);
            } else if (i == alpha_idx || j == alpha_idx) {
                const std::size_t ls_idx = (i == alpha_idx) ? j : i;
                const T ell_ls = static_cast<T>(ls[ls_idx]);
                const T s_ls2 = packed[(ls_idx * per_dim) + p] / (ell_ls * ell_ls);
                d2_k(row, col) = (s_ls2 / b) * ((g_alpha * jet.d2) + ((w / b) * jet.d1));
            } else {
                const T ell_i = static_cast<T>(ls[i]);
                const T ell_j = static_cast<T>(ls[j]);
                const T si2 = packed[(i * per_dim) + p] / (ell_i * ell_i);
                const T sj2 = packed[(j * per_dim) + p] / (ell_j * ell_j);
                const T sq_term = (si2 * sj2) / (b * b);
                T val = (sq_term * jet.d2) + ((sq_term / alpha) * jet.d1);
                if (i == j) {
                    val -= ((T{2} * si2) / b) * jet.d1;
                }
                d2_k(row, col) = val;
            }
        });
    }

    // --- Recursive tree evaluation ---

    [[nodiscard]] std::expected<void, GprError> ensure_nested(NestedBuffers<T>& nested,
                                                              std::size_t level_idx,
                                                              std::size_t rows,
                                                              std::size_t cols) const {
        if (nested.depth() <= level_idx || nested.rows() < rows || nested.cols() < cols) {
            return std::unexpected(GprError::workspace_too_small());
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_term_square(const Term& term, const CheckedDistanceStore<T>& store,
                     std::optional<MatrixRef<T>> x, MatrixMut<T> out, Triangle uplo,
                     MatrixMut<T> scratch, NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* s = std::get_if<SuppliedScalarRef>(&term)) {
            const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(),
                                    store.cols());
            return detail::leaf_apply_dist(leaves[s->leaf_idx], dist, out, uplo, exp_mode);
        }
        if (const auto* a = std::get_if<SuppliedArdRef>(&term)) {
            eval_supplied_ard_square(*a, store, out, uplo);
            return {};
        }
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            if (x.has_value()) {
                const KernelLeaf& kl = leaves[c->leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                    if (auto ok = ensure_nested(nested, level_idx, x->rows(), x->rows());
                        !ok.has_value()) {
                        return ok;
                    }
                    auto dist_buf = nested.level(level_idx, x->rows(), x->rows());
                    fill_square_sq_dist(*x, dist_buf);
                    return custom->template term<T>().apply(dist_buf.as_ref(), out, uplo);
                }
                return detail::leaf_apply_points(kl, *x, out, uplo, exp_mode);
            }
            // Constant or White only: pass dummy 0-safe shape via Constant/White on square n x n.
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c->leaf_idx])) {
                const T val = static_cast<T>(ck->constant());
                visit_triangle<T>(store.rows(), uplo,
                                  [&](std::size_t row, std::size_t col) { out(row, col) = val; });
                return {};
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c->leaf_idx]);
            const T var = static_cast<T>(wk.variance());
            visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = (row == col) ? var : T{0};
            });
            return {};
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_square(terms[0], store, x, out, uplo, scratch, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (is_leaf_term(terms[k])) {
                if (auto res =
                        eval_term_square(terms[k], store, x, scratch, uplo, out, nested, level_idx);
                    !res.has_value()) {
                    return res;
                }
                if (is_sum) {
                    add_into(out, scratch.as_ref(), uplo);
                } else {
                    mul_into(out, scratch.as_ref(), uplo);
                }
            } else {
                if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols());
                    !ok.has_value()) {
                    return ok;
                }
                auto buf = nested.level(level_idx, out.rows(), out.cols());
                if (auto res = eval_term_square(terms[k], store, x, buf, uplo, scratch, nested,
                                                level_idx + 1U);
                    !res.has_value()) {
                    return res;
                }
                if (is_sum) {
                    add_into(out, buf.as_ref(), uplo);
                } else {
                    mul_into(out, buf.as_ref(), uplo);
                }
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_term_cross(const Term& term, const CheckedDistanceStore<T>& store,
                    std::optional<MatrixRef<T>> x1, std::optional<MatrixRef<T>> x2,
                    MatrixMut<T> out, MatrixMut<T> scratch, NestedBuffers<T>& nested,
                    std::size_t level_idx) const {
        if (const auto* s = std::get_if<SuppliedScalarRef>(&term)) {
            const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(),
                                    store.cols());
            return detail::leaf_apply_cross_dist(leaves[s->leaf_idx], dist, out, exp_mode);
        }
        if (const auto* a = std::get_if<SuppliedArdRef>(&term)) {
            eval_supplied_ard_cross(*a, store, out);
            return {};
        }
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            if (x1.has_value() && x2.has_value()) {
                const KernelLeaf& kl = leaves[c->leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                    if (auto ok = ensure_nested(nested, level_idx, x1->rows(), x2->rows());
                        !ok.has_value()) {
                        return ok;
                    }
                    auto dist_buf = nested.level(level_idx, x1->rows(), x2->rows());
                    fill_cross_sq_dist(*x1, *x2, dist_buf);
                    return custom->template term<T>().apply_cross(dist_buf.as_ref(), out);
                }
                return detail::leaf_apply_cross_points(kl, *x1, *x2, out, exp_mode);
            }
            const T val =
                std::holds_alternative<ConstantKernel>(leaves[c->leaf_idx])
                    ? static_cast<T>(std::get<ConstantKernel>(leaves[c->leaf_idx]).constant())
                    : T{0};
            for (std::size_t col = 0; col < out.cols(); ++col) {
                for (std::size_t row = 0; row < out.rows(); ++row) {
                    out(row, col) = val;
                }
            }
            return {};
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_cross(terms[0], store, x1, x2, out, scratch, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (is_leaf_term(terms[k])) {
                if (auto res =
                        eval_term_cross(terms[k], store, x1, x2, scratch, out, nested, level_idx);
                    !res.has_value()) {
                    return res;
                }
                if (is_sum) {
                    add_into(out, scratch.as_ref(), Triangle::Full);
                } else {
                    mul_into(out, scratch.as_ref(), Triangle::Full);
                }
            } else {
                if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols());
                    !ok.has_value()) {
                    return ok;
                }
                auto buf = nested.level(level_idx, out.rows(), out.cols());
                if (auto res = eval_term_cross(terms[k], store, x1, x2, buf, scratch, nested,
                                               level_idx + 1U);
                    !res.has_value()) {
                    return res;
                }
                if (is_sum) {
                    add_into(out, buf.as_ref(), Triangle::Full);
                } else {
                    mul_into(out, buf.as_ref(), Triangle::Full);
                }
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_fill_diag(const Term& term, std::size_t n, std::optional<MatrixRef<T>> x, std::span<T> out,
                   std::vector<T>& scratch_vec, std::size_t level_idx) const {
        if (std::holds_alternative<SuppliedScalarRef>(term) ||
            std::holds_alternative<SuppliedArdRef>(term)) {
            std::fill(out.begin(), out.end(), T{1});
            return {};
        }
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            if (x.has_value()) {
                return detail::leaf_fill_diag(leaves[c->leaf_idx], *x, out);
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c->leaf_idx])) {
                std::fill(out.begin(), out.end(), static_cast<T>(ck->constant()));
                return {};
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c->leaf_idx]);
            std::fill(out.begin(), out.end(), static_cast<T>(wk.variance()));
            return {};
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_fill_diag(terms[0], n, x, out, scratch_vec, level_idx);
            !res.has_value()) {
            return res;
        }
        if (terms.size() == 1U) {
            return {};
        }
        const std::size_t slice_offset = level_idx * out.size();
        if (scratch_vec.size() < slice_offset + out.size()) {
            scratch_vec.resize(slice_offset + out.size());
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            const std::span<T> sub(scratch_vec.data() + slice_offset, out.size());
            if (auto res = eval_fill_diag(terms[k], n, x, sub, scratch_vec, level_idx + 1U);
                !res.has_value()) {
                return res;
            }
            for (std::size_t idx = 0; idx < out.size(); ++idx) {
                if (is_sum) {
                    out[idx] += sub[idx];
                } else {
                    out[idx] *= sub[idx];
                }
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_grad(const Term& term, const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x,
              MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo, MatrixMut<T> scratch_a,
              MatrixMut<T> scratch_b, NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* s = std::get_if<SuppliedScalarRef>(&term)) {
            const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(),
                                    store.cols());
            return detail::leaf_grad_dist(leaves[s->leaf_idx], dist, d_k,
                                          param_idx - s->param_begin, uplo, exp_mode);
        }
        if (const auto* a = std::get_if<SuppliedArdRef>(&term)) {
            eval_supplied_ard_grad(*a, store, d_k, param_idx - a->param_begin, uplo);
            return {};
        }
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            if (x.has_value()) {
                const KernelLeaf& kl = leaves[c->leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                    if (auto ok = ensure_nested(nested, level_idx, x->rows(), x->rows());
                        !ok.has_value()) {
                        return ok;
                    }
                    auto dist_buf = nested.level(level_idx, x->rows(), x->rows());
                    fill_square_sq_dist(*x, dist_buf);
                    return custom->template term<T>().grad(dist_buf.as_ref(), d_k,
                                                           param_idx - c->param_begin, uplo);
                }
                return detail::leaf_grad_points(kl, *x, d_k, param_idx - c->param_begin, uplo,
                                                exp_mode);
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c->leaf_idx])) {
                const T val = static_cast<T>(ck->constant());
                visit_triangle<T>(store.rows(), uplo,
                                  [&](std::size_t row, std::size_t col) { d_k(row, col) = val; });
                return {};
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c->leaf_idx]);
            const T var = static_cast<T>(wk.variance());
            visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                d_k(row, col) = (row == col) ? var : T{0};
            });
            return {};
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                if (contains_param(child, param_idx)) {
                    return eval_grad(child, store, x, d_k, param_idx, uplo, scratch_a, scratch_b,
                                     nested, level_idx);
                }
            }
            return std::unexpected(
                GprError::index_out_of_range("kernel parameter index out of range"));
        }
        const auto& p = std::get<ProductTerm>(term);
        std::size_t owner = p.terms.size();
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (contains_param(p.terms[m], param_idx)) {
                owner = m;
                break;
            }
        }
        if (owner == p.terms.size()) {
            return std::unexpected(
                GprError::index_out_of_range("kernel parameter index out of range"));
        }
        if (auto res = eval_grad(p.terms[owner], store, x, d_k, param_idx, uplo, scratch_a,
                                 scratch_b, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner) {
                continue;
            }
            if (auto res = eval_term_square(p.terms[m], store, x, scratch_a, uplo, scratch_b,
                                            nested, level_idx);
                !res.has_value()) {
                return res;
            }
            mul_into(d_k, scratch_a.as_ref(), uplo);
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_hess(const Term& term, const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x,
              MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo,
              MatrixMut<T> scratch_a, MatrixMut<T> scratch_b, MatrixMut<T> scratch_c,
              NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* s = std::get_if<SuppliedScalarRef>(&term)) {
            const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(),
                                    store.cols());
            return detail::leaf_hess_dist(leaves[s->leaf_idx], dist, d2_k, i - s->param_begin,
                                          j - s->param_begin, uplo, exp_mode);
        }
        if (const auto* a = std::get_if<SuppliedArdRef>(&term)) {
            eval_supplied_ard_hess(*a, store, d2_k, i - a->param_begin, j - a->param_begin, uplo);
            return {};
        }
        if (const auto* c = std::get_if<CoordLeafRef>(&term)) {
            if (x.has_value()) {
                const KernelLeaf& kl = leaves[c->leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                    if (auto ok = ensure_nested(nested, level_idx, x->rows(), x->rows());
                        !ok.has_value()) {
                        return ok;
                    }
                    auto dist_buf = nested.level(level_idx, x->rows(), x->rows());
                    fill_square_sq_dist(*x, dist_buf);
                    return custom->template term<T>().hess(
                        dist_buf.as_ref(), d2_k, i - c->param_begin, j - c->param_begin, uplo);
                }
                return detail::leaf_hess_points(kl, *x, d2_k, i - c->param_begin,
                                                j - c->param_begin, uplo, exp_mode);
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c->leaf_idx])) {
                const T val = static_cast<T>(ck->constant());
                visit_triangle<T>(store.rows(), uplo,
                                  [&](std::size_t row, std::size_t col) { d2_k(row, col) = val; });
                return {};
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c->leaf_idx]);
            const T var = static_cast<T>(wk.variance());
            visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                d2_k(row, col) = (row == col) ? var : T{0};
            });
            return {};
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                const bool has_i = contains_param(child, i);
                const bool has_j = contains_param(child, j);
                if (has_i && has_j) {
                    return eval_hess(child, store, x, d2_k, i, j, uplo, scratch_a, scratch_b,
                                     scratch_c, nested, level_idx);
                }
                if (has_i || has_j) {
                    visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                        d2_k(row, col) = T{0};
                    });
                    return {};
                }
            }
            return std::unexpected(
                GprError::index_out_of_range("kernel parameter index out of range"));
        }
        const auto& p = std::get<ProductTerm>(term);
        std::size_t owner_i = p.terms.size();
        std::size_t owner_j = p.terms.size();
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (contains_param(p.terms[m], i)) {
                owner_i = m;
            }
            if (contains_param(p.terms[m], j)) {
                owner_j = m;
            }
        }
        if (owner_i == p.terms.size() || owner_j == p.terms.size()) {
            return std::unexpected(
                GprError::index_out_of_range("kernel parameter index out of range"));
        }
        if (owner_i == owner_j) {
            if (auto res = eval_hess(p.terms[owner_i], store, x, d2_k, i, j, uplo, scratch_a,
                                     scratch_b, scratch_c, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m) {
                if (m == owner_i) {
                    continue;
                }
                if (auto res = eval_term_square(p.terms[m], store, x, scratch_b, uplo, scratch_c,
                                                nested, level_idx);
                    !res.has_value()) {
                    return res;
                }
                mul_into(d2_k, scratch_b.as_ref(), uplo);
            }
            return {};
        }
        MatrixMut<T> grad_rhs = scratch_a;
        MatrixMut<T> work_a = scratch_b;
        MatrixMut<T> work_b = scratch_c;
        if (auto res = eval_grad(p.terms[owner_i], store, x, d2_k, i, uplo, work_a, work_b, nested,
                                 level_idx);
            !res.has_value()) {
            return res;
        }
        if (auto res = eval_grad(p.terms[owner_j], store, x, grad_rhs, j, uplo, work_a, work_b,
                                 nested, level_idx);
            !res.has_value()) {
            return res;
        }
        mul_into(d2_k, grad_rhs.as_ref(), uplo);
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner_i || m == owner_j) {
                continue;
            }
            if (auto res = eval_term_square(p.terms[m], store, x, scratch_b, uplo, scratch_c,
                                            nested, level_idx);
                !res.has_value()) {
                return res;
            }
            mul_into(d2_k, scratch_b.as_ref(), uplo);
        }
        return {};
    }
};

// --- DistanceStoreBuilder ---

struct DistanceStoreBuilder {
    static constexpr std::size_t k_pair_tile = 64U;

    template <KernelScalarConcept T>
    [[nodiscard]] static std::expected<T, GprError>
    checked_cast(double val, std::optional<std::size_t> slot, std::optional<std::size_t> dim,
                 std::size_t row, std::size_t col) {
        const auto cast = static_cast<T>(val);
        if (!std::isfinite(cast)) {
            return std::unexpected(GprError::invalid_distance(
                slot, dim, std::pair{row, col}, "value overflows storage scalar range"));
        }
        return cast;
    }

    template <KernelScalarConcept T>
    [[nodiscard]] static std::expected<void, GprError> validate_and_pack_square_block(
        MatrixRef<double> src, std::optional<double> tidy_tol, std::optional<std::size_t> slot_idx,
        std::optional<std::size_t> dim_idx, bool pack_lower, std::vector<T>& dst) {
        const std::size_t n = src.rows();
        double max_abs = 0.0;
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = 0; row < n; ++row) {
                const double v = src(row, col);
                if (!std::isfinite(v)) {
                    return std::unexpected(GprError::invalid_distance(
                        slot_idx, dim_idx, std::pair{row, col}, "distance is not finite"));
                }
                const double av = std::abs(v);
                max_abs = std::max(av, max_abs);
            }
        }
        const double tol = tidy_tol.has_value() ? (*tidy_tol * max_abs) : 0.0;

        if (!tidy_tol.has_value()) {
            // Strict 1-pass check.
            for (std::size_t col = 0; col < n; ++col) {
                const double diag = src(col, col);
                if (diag < 0.0) {
                    return std::unexpected(GprError::invalid_distance(
                        slot_idx, dim_idx, std::pair{col, col}, "distance is negative"));
                }
                if (diag != 0.0) {
                    return std::unexpected(GprError::invalid_distance(
                        slot_idx, dim_idx, std::pair{col, col}, "diagonal must be zero"));
                }
            }
            for (std::size_t j0 = 0; j0 < n; j0 += k_pair_tile) {
                const std::size_t j1 = std::min(j0 + k_pair_tile, n);
                for (std::size_t i0 = j0; i0 < n; i0 += k_pair_tile) {
                    const std::size_t i1 = std::min(i0 + k_pair_tile, n);
                    for (std::size_t j = j0; j < j1; ++j) {
                        for (std::size_t i = std::max(i0, j + 1U); i < i1; ++i) {
                            const double a = src(i, j);
                            const double b = src(j, i);
                            if (a < 0.0) {
                                return std::unexpected(GprError::invalid_distance(
                                    slot_idx, dim_idx, std::pair{i, j}, "distance is negative"));
                            }
                            if (b < 0.0) {
                                return std::unexpected(GprError::invalid_distance(
                                    slot_idx, dim_idx, std::pair{j, i}, "distance is negative"));
                            }
                            if (a != b) {
                                return std::unexpected(GprError::invalid_distance(
                                    slot_idx, dim_idx, std::pair{i, j},
                                    "square distance matrix is not symmetric"));
                            }
                        }
                    }
                }
            }
        } else {
            // Tidy tolerance check: track worst violation or first out-of-tolerance violation.
            for (std::size_t col = 0; col < n; ++col) {
                for (std::size_t row = 0; row < n; ++row) {
                    const double v = src(row, col);
                    if (v < -tol) {
                        return std::unexpected(
                            GprError::invalid_distance(slot_idx, dim_idx, std::pair{row, col},
                                                       "negative distance exceeds tidy tolerance"));
                    }
                }
                if (std::abs(src(col, col)) > tol) {
                    return std::unexpected(
                        GprError::invalid_distance(slot_idx, dim_idx, std::pair{col, col},
                                                   "nonzero diagonal exceeds tidy tolerance"));
                }
            }
            for (std::size_t j0 = 0; j0 < n; j0 += k_pair_tile) {
                const std::size_t j1 = std::min(j0 + k_pair_tile, n);
                for (std::size_t i0 = j0; i0 < n; i0 += k_pair_tile) {
                    const std::size_t i1 = std::min(i0 + k_pair_tile, n);
                    for (std::size_t j = j0; j < j1; ++j) {
                        for (std::size_t i = std::max(i0, j + 1U); i < i1; ++i) {
                            const double gap = std::abs(src(i, j) - src(j, i));
                            if (gap > tol) {
                                return std::unexpected(
                                    GprError::invalid_distance(slot_idx, dim_idx, std::pair{i, j},
                                                               "asymmetry exceeds tidy tolerance"));
                            }
                        }
                    }
                }
            }
        }

        if (pack_lower) {
            const auto per_dim_exp = packed_len(n);
            if (!per_dim_exp.has_value()) {
                return std::unexpected(std::move(per_dim_exp).error());
            }
            const std::size_t per_dim = *per_dim_exp;
            const std::size_t base_offset = dst.size();
            dst.resize(base_offset + per_dim);
            for (std::size_t j = 0; j < n; ++j) {
                const std::size_t diag_p = packed_lower_index(n, j, j);
                dst[base_offset + diag_p] = T{0};
                for (std::size_t i = j + 1U; i < n; ++i) {
                    const double a = src(i, j);
                    const double b = src(j, i);
                    const double mean = (a == b) ? a : (a + (0.5 * (b - a)));
                    const double repaired = std::max(0.0, mean);
                    auto cast = checked_cast<T>(repaired, slot_idx, dim_idx, i, j);
                    if (!cast.has_value()) {
                        return std::unexpected(std::move(cast).error());
                    }
                    dst[base_offset + packed_lower_index(n, i, j)] = *cast;
                }
            }
        } else {
            const auto len_exp = column_major_len(n, n);
            if (!len_exp.has_value()) {
                return std::unexpected(std::move(len_exp).error());
            }
            dst.resize(*len_exp);
            for (std::size_t j = 0; j < n; ++j) {
                dst[(j * n) + j] = T{0};
                for (std::size_t i = j + 1U; i < n; ++i) {
                    const double a = src(i, j);
                    const double b = src(j, i);
                    const double mean = (a == b) ? a : (a + (0.5 * (b - a)));
                    const double repaired = std::max(0.0, mean);
                    auto cast = checked_cast<T>(repaired, slot_idx, dim_idx, i, j);
                    if (!cast.has_value()) {
                        return std::unexpected(std::move(cast).error());
                    }
                    dst[(j * n) + i] = *cast;
                    dst[(i * n) + j] = *cast;
                }
            }
        }
        return {};
    }

    template <KernelScalarConcept T>
    [[nodiscard]] static std::expected<void, GprError>
    validate_and_copy_rect_block(MatrixRef<double> src, std::optional<double> tidy_tol,
                                 std::optional<std::size_t> slot_idx,
                                 std::optional<std::size_t> dim_idx, std::vector<T>& dst) {
        const std::size_t n_rows = src.rows();
        const std::size_t n_cols = src.cols();
        double max_abs = 0.0;
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = 0; row < n_rows; ++row) {
                const double v = src(row, col);
                if (!std::isfinite(v)) {
                    return std::unexpected(GprError::invalid_distance(
                        slot_idx, dim_idx, std::pair{row, col}, "distance is not finite"));
                }
                const double av = std::abs(v);
                max_abs = std::max(av, max_abs);
            }
        }
        const double tol = tidy_tol.has_value() ? (*tidy_tol * max_abs) : 0.0;
        const std::size_t base_offset = dst.size();
        const auto len_exp = column_major_len(n_rows, n_cols);
        if (!len_exp.has_value()) {
            return std::unexpected(std::move(len_exp).error());
        }
        dst.resize(base_offset + *len_exp);
        for (std::size_t col = 0; col < n_cols; ++col) {
            for (std::size_t row = 0; row < n_rows; ++row) {
                const double v = src(row, col);
                if (!tidy_tol.has_value() && v < 0.0) {
                    return std::unexpected(GprError::invalid_distance(
                        slot_idx, dim_idx, std::pair{row, col}, "distance is negative"));
                }
                if (tidy_tol.has_value() && v < -tol) {
                    return std::unexpected(
                        GprError::invalid_distance(slot_idx, dim_idx, std::pair{row, col},
                                                   "negative distance exceeds tidy tolerance"));
                }
                const double repaired = std::max(0.0, v);
                auto cast = checked_cast<T>(repaired, slot_idx, dim_idx, row, col);
                if (!cast.has_value()) {
                    return std::unexpected(std::move(cast).error());
                }
                dst[base_offset + (col * n_rows) + row] = *cast;
            }
        }
        return {};
    }

    template <KernelScalarConcept T>
    [[nodiscard]] static std::expected<CheckedDistanceStore<T>, GprError>
    build(const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, bool is_square) {
        if (source.tidy_rel_tol_.has_value()) {
            const double tol = *source.tidy_rel_tol_;
            if (!std::isfinite(tol) || tol < 0.0) {
                return std::unexpected(
                    GprError::invalid_config("tidy relative tolerance must be finite and >= 0"));
            }
        }

        const auto& kslots = kernel.slots();

        // Check duplicates and unknown/mismatched slots in source.
        std::vector<SlotId> seen_ids;
        seen_ids.reserve(source.scalars_.size() + source.ards_.size());

        auto find_desc = [&](SlotId id) -> const SlotDescriptor* {
            for (const auto& d : kslots) {
                if (d.id == id) {
                    return &d;
                }
            }
            return nullptr;
        };

        for (const auto& entry : source.scalars_) {
            const auto* desc = find_desc(entry.slot);
            if (desc == nullptr) {
                return std::unexpected(GprError::unsupported_kernel_operation(
                    "distance source supplies a slot from another kernel"));
            }
            if (desc->ard_dims.has_value()) {
                return std::unexpected(GprError::distance_slot(SlotErrorKind::NotRead, desc->at));
            }
            if (std::ranges::find(seen_ids, entry.slot) != seen_ids.end()) {
                return std::unexpected(GprError::distance_slot(SlotErrorKind::Duplicate, desc->at));
            }
            seen_ids.push_back(entry.slot);
        }

        for (const auto& entry : source.ards_) {
            const auto* desc = find_desc(entry.slot);
            if (desc == nullptr) {
                return std::unexpected(GprError::unsupported_kernel_operation(
                    "distance source supplies a slot from another kernel"));
            }
            if (!desc->ard_dims.has_value()) {
                return std::unexpected(GprError::distance_slot(SlotErrorKind::NotRead, desc->at));
            }
            if (std::ranges::find(seen_ids, entry.slot) != seen_ids.end()) {
                return std::unexpected(GprError::distance_slot(SlotErrorKind::Duplicate, desc->at));
            }
            seen_ids.push_back(entry.slot);
        }

        // Check every required slot is present.
        for (const auto& desc : kslots) {
            if (std::ranges::find(seen_ids, desc.id) == seen_ids.end()) {
                return std::unexpected(GprError::distance_slot(SlotErrorKind::Missing, desc.at));
            }
        }

        if (kslots.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("kernel has no supplied distance slots"));
        }

        // Determine shape from the first slot entry.
        std::size_t n_rows = 0;
        std::size_t n_cols = 0;
        if (!source.scalars_.empty()) {
            n_rows = source.scalars_[0].d2.rows();
            n_cols = source.scalars_[0].d2.cols();
        } else {
            if (source.ards_[0].d2_per_dim.empty()) {
                return std::unexpected(GprError::empty_input());
            }
            n_rows = source.ards_[0].d2_per_dim[0].rows();
            n_cols = source.ards_[0].d2_per_dim[0].cols();
        }
        if (n_rows == 0 || n_cols == 0) {
            return std::unexpected(GprError::empty_input());
        }
        if (is_square && n_rows != n_cols) {
            return std::unexpected(
                GprError::length_mismatch("square distance matrix must have equal rows and cols"));
        }

        auto impl = std::make_unique<typename CheckedDistanceStore<T>::Impl>();
        impl->kernel_slots = kslots;
        impl->n_rows = n_rows;
        impl->n_cols = n_cols;
        impl->square = is_square;
        impl->scalar_tables.resize(kernel.impl_->num_scalar_slots);
        impl->ard_dims.resize(kernel.impl_->num_ard_slots, 0U);
        impl->ard_data.resize(kernel.impl_->num_ard_slots);

        for (const auto& entry : source.scalars_) {
            const auto* desc = find_desc(entry.slot);
            if (entry.d2.rows() != n_rows || entry.d2.cols() != n_cols) {
                return std::unexpected(
                    GprError::length_mismatch("scalar distance block shape mismatch"));
            }
            if (is_square) {
                if (auto res = validate_and_pack_square_block<T>(entry.d2, source.tidy_rel_tol_,
                                                                 desc->at, std::nullopt, false,
                                                                 impl->scalar_tables[desc->at]);
                    !res.has_value()) {
                    return std::unexpected(std::move(res).error());
                }
            } else {
                if (auto res = validate_and_copy_rect_block<T>(entry.d2, source.tidy_rel_tol_,
                                                               desc->at, std::nullopt,
                                                               impl->scalar_tables[desc->at]);
                    !res.has_value()) {
                    return std::unexpected(std::move(res).error());
                }
            }
        }

        for (const auto& entry : source.ards_) {
            const auto* desc = find_desc(entry.slot);
            const std::size_t expected_dims = *desc->ard_dims;
            if (entry.d2_per_dim.size() != expected_dims) {
                return std::unexpected(
                    GprError::dimension_mismatch(entry.d2_per_dim.size(), expected_dims));
            }
            impl->ard_dims[desc->at] = expected_dims;
            if (is_square) {
                const auto per_dim_exp = packed_len(n_rows);
                if (!per_dim_exp.has_value()) {
                    return std::unexpected(std::move(per_dim_exp).error());
                }
                const auto total_exp = checked_mul(*per_dim_exp, expected_dims);
                if (!total_exp.has_value()) {
                    return std::unexpected(std::move(total_exp).error());
                }
                impl->ard_data[desc->at].reserve(*total_exp);
            } else {
                const auto block_exp = column_major_len(n_rows, n_cols);
                if (!block_exp.has_value()) {
                    return std::unexpected(std::move(block_exp).error());
                }
                const auto total_exp = checked_mul(*block_exp, expected_dims);
                if (!total_exp.has_value()) {
                    return std::unexpected(std::move(total_exp).error());
                }
                impl->ard_data[desc->at].reserve(*total_exp);
            }
            for (std::size_t dim = 0; dim < expected_dims; ++dim) {
                const auto& blk = entry.d2_per_dim[dim];
                if (blk.rows() != n_rows || blk.cols() != n_cols) {
                    return std::unexpected(
                        GprError::length_mismatch("ARD distance block shape mismatch"));
                }
                if (is_square) {
                    if (auto res =
                            validate_and_pack_square_block<T>(blk, source.tidy_rel_tol_, desc->at,
                                                              dim, true, impl->ard_data[desc->at]);
                        !res.has_value()) {
                        return std::unexpected(std::move(res).error());
                    }
                } else {
                    if (auto res = validate_and_copy_rect_block<T>(
                            blk, source.tidy_rel_tol_, desc->at, dim, impl->ard_data[desc->at]);
                        !res.has_value()) {
                        return std::unexpected(std::move(res).error());
                    }
                }
            }
        }

        return CheckedDistanceStore<T>(std::move(impl));
    }
};

// --- CompiledDistanceKernel<T> ---

template <KernelScalarConcept T>
CompiledDistanceKernel<T>::CompiledDistanceKernel(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

template <KernelScalarConcept T> CompiledDistanceKernel<T>::~CompiledDistanceKernel() = default;

template <KernelScalarConcept T>
CompiledDistanceKernel<T>::CompiledDistanceKernel(const CompiledDistanceKernel& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

template <KernelScalarConcept T>
CompiledDistanceKernel<T>&
CompiledDistanceKernel<T>::operator=(const CompiledDistanceKernel& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

template <DistanceModeConcept Mode>
template <KernelScalarConcept T>
CompiledDistanceKernel<T> DistanceKernel<Mode>::compile() const {
    auto impl = std::make_unique<typename CompiledDistanceKernel<T>::Impl>();
    impl->slots = slots();
    impl->num_scalar_slots = num_scalar_slots();
    impl->num_ard_slots = num_ard_slots();
    std::size_t next_param = 0;
    impl->root =
        CompiledDistanceKernel<T>::Impl::template compile_node<Mode>(*root_, *impl, next_param);
    bool has_custom_coord = false;
    if (impl->has_coord_points) {
        for (const auto& leaf : impl->leaves) {
            if (std::holds_alternative<CustomKernel>(leaf)) {
                has_custom_coord = true;
                break;
            }
        }
    }
    impl->nested_depth = CompiledDistanceKernel<T>::Impl::compute_nested_depth(impl->root) +
                         (has_custom_coord ? 1U : 0U);
    impl->refresh_cached_ls();
    return CompiledDistanceKernel<T>(std::move(impl));
}

template <KernelScalarConcept T> KernelExp CompiledDistanceKernel<T>::kernel_exp() const noexcept {
    return impl_->exp_mode;
}

template <KernelScalarConcept T>
void CompiledDistanceKernel<T>::set_kernel_exp(KernelExp mode) noexcept {
    impl_->exp_mode = mode;
}

template <KernelScalarConcept T>
std::size_t CompiledDistanceKernel<T>::num_params() const noexcept {
    return impl_->bindings.size();
}

template <KernelScalarConcept T>
std::size_t CompiledDistanceKernel<T>::num_leaves() const noexcept {
    return impl_->leaves.size();
}

template <KernelScalarConcept T> bool CompiledDistanceKernel<T>::requires_points() const noexcept {
    return impl_->has_coord_points;
}

template <KernelScalarConcept T>
std::size_t CompiledDistanceKernel<T>::nested_depth() const noexcept {
    return impl_->nested_depth;
}

template <KernelScalarConcept T>
NestedBuffers<T> CompiledDistanceKernel<T>::make_nested_buffers(std::size_t rows,
                                                                std::size_t cols) const {
    return NestedBuffers<T>(impl_->nested_depth, rows, cols);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledDistanceKernel<T>::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value()) {
        return cnt;
    }
    std::size_t offset = 0;
    for (const auto& leaf : impl_->leaves) {
        const std::size_t count = detail::leaf_num_params(leaf);
        if (auto res = detail::leaf_get_params(leaf, out.subspan(offset, count));
            !res.has_value()) {
            return res;
        }
        offset += count;
    }
    return {};
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), impl_->bindings.size(), "params");
        !cnt.has_value()) {
        return cnt;
    }
    std::vector<KernelLeaf> updated = impl_->leaves;
    std::size_t offset = 0;
    for (auto& leaf : updated) {
        const std::size_t count = detail::leaf_num_params(leaf);
        if (auto res = detail::leaf_set_params(leaf, params.subspan(offset, count));
            !res.has_value()) {
            return res;
        }
        offset += count;
    }
    impl_->leaves = std::move(updated);
    impl_->refresh_cached_ls();
    return {};
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value()) {
        return cnt;
    }
    std::size_t offset = 0;
    for (const auto& leaf : impl_->leaves) {
        const std::size_t count = detail::leaf_num_params(leaf);
        if (auto res = detail::leaf_bounds_into(leaf, out.subspan(offset, count));
            !res.has_value()) {
            return res;
        }
        offset += count;
    }
    return {};
}

template <KernelScalarConcept T>
const std::vector<ParameterBinding>&
CompiledDistanceKernel<T>::parameter_bindings() const noexcept {
    return impl_->bindings;
}

template <KernelScalarConcept T>
const std::vector<SlotDescriptor>& CompiledDistanceKernel<T>::slots() const noexcept {
    return impl_->slots;
}

template <KernelScalarConcept T>
std::expected<CheckedDistanceStore<T>, GprError>
CompiledDistanceKernel<T>::prepare_square(const DistanceSource& source) const {
    return DistanceStoreBuilder::build<T>(*this, source, true);
}

template <KernelScalarConcept T>
std::expected<CheckedDistanceStore<T>, GprError>
CompiledDistanceKernel<T>::prepare_cross(const DistanceSource& source) const {
    return DistanceStoreBuilder::build<T>(*this, source, false);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::apply(const CheckedDistanceStore<T>& store,
                                 std::optional<MatrixRef<T>> x, MatrixMut<T> out, Triangle uplo,
                                 MatrixMut<T> scratch, NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch); !sc.has_value()) {
        return sc;
    }
    if (auto vs = impl_->validate_store(store, true, out.as_ref()); !vs.has_value()) {
        return vs;
    }
    if (auto vp = impl_->validate_optional_points(x, store.rows()); !vp.has_value()) {
        return vp;
    }
    return impl_->eval_term_square(impl_->root, store, x, out, uplo, scratch, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledDistanceKernel<T>::apply(const CheckedDistanceStore<T>& store,
                                                               std::optional<MatrixRef<T>> x,
                                                               MatrixMut<T> out, Triangle uplo,
                                                               MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply(store, x, out, uplo, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::apply_cross(const CheckedDistanceStore<T>& store,
                                       std::optional<MatrixRef<T>> x1,
                                       std::optional<MatrixRef<T>> x2, MatrixMut<T> out,
                                       MatrixMut<T> scratch, NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch); !sc.has_value()) {
        return sc;
    }
    if (auto vs = impl_->validate_store(store, false, out.as_ref()); !vs.has_value()) {
        return vs;
    }
    if (auto vp1 = impl_->validate_optional_points(x1, store.rows()); !vp1.has_value()) {
        return vp1;
    }
    if (auto vp2 = impl_->validate_optional_points(x2, store.cols()); !vp2.has_value()) {
        return vp2;
    }
    if (x1.has_value() && x2.has_value() && x1->cols() != x2->cols()) {
        return std::unexpected(GprError::dimension_mismatch(x2->cols(), x1->cols()));
    }
    return impl_->eval_term_cross(impl_->root, store, x1, x2, out, scratch, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledDistanceKernel<T>::apply_cross(
    const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x1,
    std::optional<MatrixRef<T>> x2, MatrixMut<T> out, MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply_cross(store, x1, x2, out, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledDistanceKernel<T>::fill_diag(std::size_t n,
                                                                   std::optional<MatrixRef<T>> x,
                                                                   std::span<T> out) const {
    if (n == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    if (auto cnt = require_count(out.size(), n, "out"); !cnt.has_value()) {
        return cnt;
    }
    if (auto vp = impl_->validate_optional_points(x, n); !vp.has_value()) {
        return vp;
    }
    std::vector<T> scratch_vec((impl_->nested_depth + 1U) * out.size());
    return impl_->eval_fill_diag(impl_->root, n, x, out, scratch_vec, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::grad(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x,
                                MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo,
                                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                                NestedBuffers<T>& nested) const {
    if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a); !sc_a.has_value()) {
        return sc_a;
    }
    if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b); !sc_b.has_value()) {
        return sc_b;
    }
    if (scratch_a.data() == scratch_b.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
    }
    if (auto vs = impl_->validate_store(store, true, d_k.as_ref()); !vs.has_value()) {
        return vs;
    }
    if (auto vp = impl_->validate_optional_points(x, store.rows()); !vp.has_value()) {
        return vp;
    }
    if (param_idx >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_grad(impl_->root, store, x, d_k, param_idx, uplo, scratch_a, scratch_b,
                            nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::grad(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x,
                                MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo,
                                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b) const {
    auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
    return grad(store, x, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::hess(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x,
                                MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo,
                                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                                MatrixMut<T> scratch_c, NestedBuffers<T>& nested) const {
    if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a); !sc_a.has_value()) {
        return sc_a;
    }
    if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b); !sc_b.has_value()) {
        return sc_b;
    }
    if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c); !sc_c.has_value()) {
        return sc_c;
    }
    if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data() ||
        scratch_b.data() == scratch_c.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
    }
    if (auto vs = impl_->validate_store(store, true, d2_k.as_ref()); !vs.has_value()) {
        return vs;
    }
    if (auto vp = impl_->validate_optional_points(x, store.rows()); !vp.has_value()) {
        return vp;
    }
    if (i >= impl_->bindings.size() || j >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_hess(impl_->root, store, x, d2_k, i, j, uplo, scratch_a, scratch_b,
                            scratch_c, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledDistanceKernel<T>::hess(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x,
                                MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo,
                                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                                MatrixMut<T> scratch_c) const {
    auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
    return hess(store, x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
}

template class DistanceKernel<DistanceOnly>;
template class DistanceKernel<WithPoints>;
template CompiledDistanceKernel<double> DistanceKernel<DistanceOnly>::compile<double>() const;
template CompiledDistanceKernel<float> DistanceKernel<DistanceOnly>::compile<float>() const;
template CompiledDistanceKernel<double> DistanceKernel<WithPoints>::compile<double>() const;
template CompiledDistanceKernel<float> DistanceKernel<WithPoints>::compile<float>() const;
template class CheckedDistanceStore<double>;
template class CheckedDistanceStore<float>;
template class CompiledDistanceKernel<double>;
template class CompiledDistanceKernel<float>;

} // namespace cxxgpr
