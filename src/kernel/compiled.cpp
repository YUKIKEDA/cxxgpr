// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/compiled.hpp"

#include "cxxgpr/core/checked.hpp"
#include "kernel/leaf_eval.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

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

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> require_scratch(MatrixRef<T> out, MatrixMut<T> scratch,
                                                            std::string_view /*label*/) {
    if (scratch.rows() != out.rows() || scratch.cols() != out.cols()) {
        return std::unexpected(GprError::workspace_too_small());
    }
    if (scratch.data() == out.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffer must not alias output"));
    }
    return {};
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

} // namespace

// --- NestedBuffers ---

template <KernelScalarConcept T>
NestedBuffers<T>::NestedBuffers(std::size_t depth, std::size_t rows, std::size_t cols) {
    resize(depth, rows, cols);
}

template <KernelScalarConcept T>
void NestedBuffers<T>::resize(std::size_t depth, std::size_t rows, std::size_t cols) {
    depth_ = depth;
    rows_ = rows;
    cols_ = cols;
    const std::size_t needed = depth * rows * cols;
    if (storage_.size() < needed) {
        storage_.resize(needed);
    }
}

template <KernelScalarConcept T>
MatrixMut<T> NestedBuffers<T>::level(std::size_t index, std::size_t rows, std::size_t cols) {
    T* ptr = storage_.data() + (index * rows_ * cols_);
    return {ptr, rows, cols, rows};
}

template <KernelScalarConcept T> struct CompiledKernel<T>::Impl {
    struct LeafRef {
        std::size_t leaf_idx = 0;
        std::size_t param_begin = 0;
        std::size_t param_count = 0;
    };

    struct SumTerm;
    struct ProductTerm;
    using Term = std::variant<LeafRef, SumTerm, ProductTerm>;

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
    std::vector<ParameterBinding> bindings;
    Term root;
    CoordMode coord_mode = CoordMode::Either;
    std::size_t nested_depth = 0;

    static std::pair<std::size_t, std::size_t> param_range(const Term& term) noexcept {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return {leaf->param_begin, leaf->param_begin + leaf->param_count};
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            return {s->param_begin, s->param_end};
        }
        if (const auto* p = std::get_if<ProductTerm>(&term)) {
            return {p->param_begin, p->param_end};
        }
        return {0, 0};
    }

    static bool contains_param(const Term& term, std::size_t param_idx) noexcept {
        const auto [begin, end] = param_range(term);
        return param_idx >= begin && param_idx < end;
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
            if (!std::holds_alternative<LeafRef>(child)) {
                max_child = std::max(max_child, 1U + compute_nested_depth(child));
            }
        }
        return max_child;
    }

    static Term compile_node(const KernelSpec& spec, std::vector<KernelLeaf>& out_leaves,
                             std::vector<ParameterBinding>& out_bindings,
                             std::size_t& next_param_idx) {
        if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_)) {
            const std::size_t leaf_idx = out_leaves.size();
            const std::size_t p_begin = next_param_idx;
            const std::size_t p_count = detail::leaf_num_params(*leaf);
            out_leaves.push_back(*leaf);
            for (std::size_t local = 0; local < p_count; ++local) {
                out_bindings.push_back(ParameterBinding{
                    .index = next_param_idx++,
                    .leaf_id = leaf_idx,
                    .local_index = local,
                });
            }
            return LeafRef{.leaf_idx = leaf_idx, .param_begin = p_begin, .param_count = p_count};
        }
        if (std::holds_alternative<KernelSpec::SumNode>(spec.node_)) {
            std::vector<Term> flat;
            const std::size_t p_begin = next_param_idx;
            collect_sum(spec, flat, out_leaves, out_bindings, next_param_idx);
            return SumTerm{
                .terms = std::move(flat),
                .param_begin = p_begin,
                .param_end = next_param_idx,
            };
        }
        std::vector<Term> flat;
        const std::size_t p_begin = next_param_idx;
        collect_product(spec, flat, out_leaves, out_bindings, next_param_idx);
        return ProductTerm{
            .terms = std::move(flat),
            .param_begin = p_begin,
            .param_end = next_param_idx,
        };
    }

    static void collect_sum(const KernelSpec& spec, std::vector<Term>& out_terms,
                            std::vector<KernelLeaf>& out_leaves,
                            std::vector<ParameterBinding>& out_bindings,
                            std::size_t& next_param_idx) {
        if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_)) {
            collect_sum(*s->lhs, out_terms, out_leaves, out_bindings, next_param_idx);
            collect_sum(*s->rhs, out_terms, out_leaves, out_bindings, next_param_idx);
            return;
        }
        out_terms.push_back(compile_node(spec, out_leaves, out_bindings, next_param_idx));
    }

    static void collect_product(const KernelSpec& spec, std::vector<Term>& out_terms,
                                std::vector<KernelLeaf>& out_leaves,
                                std::vector<ParameterBinding>& out_bindings,
                                std::size_t& next_param_idx) {
        if (const auto* p = std::get_if<KernelSpec::ProductNode>(&spec.node_)) {
            collect_product(*p->lhs, out_terms, out_leaves, out_bindings, next_param_idx);
            collect_product(*p->rhs, out_terms, out_leaves, out_bindings, next_param_idx);
            return;
        }
        out_terms.push_back(compile_node(spec, out_leaves, out_bindings, next_param_idx));
    }

    [[nodiscard]] std::expected<void, GprError> validate_ard_dims(std::size_t x_dim) const {
        for (const auto& leaf : leaves) {
            std::size_t expected_dim = 0;
            if (const auto* k = std::get_if<RbfArdKernel>(&leaf)) {
                expected_dim = k->num_params();
            } else if (const auto* k_m = std::get_if<MaternArdKernel>(&leaf)) {
                expected_dim = k_m->num_params();
            } else if (const auto* k_rq = std::get_if<RationalQuadraticArdKernel>(&leaf)) {
                expected_dim = k_rq->lengthscales().size();
            }
            if (expected_dim > 0 && x_dim != expected_dim) {
                return std::unexpected(GprError::dimension_mismatch(x_dim, expected_dim));
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    validate_dist_call(MatrixRef<T> dist, MatrixRef<T> out, Triangle uplo, bool square) const {
        if (coord_mode == CoordMode::Points || coord_mode == CoordMode::Mixed) {
            return std::unexpected(GprError::unsupported_kernel_operation(
                "coordinate-dependent kernel cannot be evaluated from squared distances"));
        }
        if (dist.rows() == 0 || dist.cols() == 0) {
            return std::unexpected(GprError::empty_input());
        }
        if (square && dist.rows() != dist.cols()) {
            return std::unexpected(
                GprError::length_mismatch("distance matrix for square evaluation must be square"));
        }
        if (out.rows() != dist.rows() || out.cols() != dist.cols()) {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        const std::size_t n_rows = dist.rows();
        const std::size_t n_cols = dist.cols();
        for (std::size_t col = 0; col < n_cols; ++col) {
            const std::size_t row_begin = (square && uplo == Triangle::Lower) ? col : 0U;
            const std::size_t row_end = (square && uplo == Triangle::Upper) ? (col + 1U) : n_rows;
            for (std::size_t row = row_begin; row < row_end; ++row) {
                if (!std::isfinite(dist(row, col))) {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError> validate_points_call(MatrixRef<T> x,
                                                                     MatrixRef<T> out) const {
        if (x.rows() == 0 || x.cols() == 0) {
            return std::unexpected(GprError::empty_input());
        }
        for (std::size_t col = 0; col < x.cols(); ++col) {
            for (std::size_t row = 0; row < x.rows(); ++row) {
                if (!std::isfinite(x(row, col))) {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        if (auto ard = validate_ard_dims(x.cols()); !ard.has_value()) {
            return ard;
        }
        if (out.rows() != x.rows() || out.cols() != x.rows()) {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        return {};
    }

    // --- Evaluation helpers ---

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
    eval_term_dist(const Term& term, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo,
                   MatrixMut<T> scratch, NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_apply_dist(leaves[leaf->leaf_idx], dist, out, uplo);
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_dist(terms[0], dist, out, uplo, scratch, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (std::holds_alternative<LeafRef>(terms[k])) {
                if (auto res =
                        eval_term_dist(terms[k], dist, scratch, uplo, out, nested, level_idx);
                    !res.has_value()) {
                    return res;
                }
            } else {
                if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols());
                    !ok.has_value()) {
                    return ok;
                }
                auto buf = nested.level(level_idx, out.rows(), out.cols());
                if (auto res =
                        eval_term_dist(terms[k], dist, buf, uplo, scratch, nested, level_idx + 1);
                    !res.has_value()) {
                    return res;
                }
                if (is_sum) {
                    add_into(out, buf.as_ref(), uplo);
                } else {
                    mul_into(out, buf.as_ref(), uplo);
                }
                continue;
            }
            if (is_sum) {
                add_into(out, scratch.as_ref(), uplo);
            } else {
                mul_into(out, scratch.as_ref(), uplo);
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_term_points(const Term& term, MatrixRef<T> x, MatrixMut<T> out, Triangle uplo,
                     MatrixMut<T> scratch, NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_apply_points(leaves[leaf->leaf_idx], x, out, uplo);
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_points(terms[0], x, out, uplo, scratch, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (std::holds_alternative<LeafRef>(terms[k])) {
                if (auto res = eval_term_points(terms[k], x, scratch, uplo, out, nested, level_idx);
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
                if (auto res =
                        eval_term_points(terms[k], x, buf, uplo, scratch, nested, level_idx + 1);
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
    eval_term_cross_dist(const Term& term, MatrixRef<T> dist, MatrixMut<T> out,
                         MatrixMut<T> scratch, NestedBuffers<T>& nested,
                         std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_apply_cross_dist(leaves[leaf->leaf_idx], dist, out);
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_cross_dist(terms[0], dist, out, scratch, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (std::holds_alternative<LeafRef>(terms[k])) {
                if (auto res =
                        eval_term_cross_dist(terms[k], dist, scratch, out, nested, level_idx);
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
                if (auto res =
                        eval_term_cross_dist(terms[k], dist, buf, scratch, nested, level_idx + 1);
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
    eval_term_cross_points(const Term& term, MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out,
                           MatrixMut<T> scratch, NestedBuffers<T>& nested,
                           std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_apply_cross_points(leaves[leaf->leaf_idx], x1, x2, out);
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_cross_points(terms[0], x1, x2, out, scratch, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (std::holds_alternative<LeafRef>(terms[k])) {
                if (auto res =
                        eval_term_cross_points(terms[k], x1, x2, scratch, out, nested, level_idx);
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
                if (auto res = eval_term_cross_points(terms[k], x1, x2, buf, scratch, nested,
                                                      level_idx + 1);
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

    [[nodiscard]] std::expected<void, GprError> eval_fill_diag(const Term& term, MatrixRef<T> x,
                                                               std::span<T> out,
                                                               std::vector<T>& scratch_vec,
                                                               std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_fill_diag(leaves[leaf->leaf_idx], x, out);
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_fill_diag(terms[0], x, out, scratch_vec, level_idx); !res.has_value()) {
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
            if (auto res = eval_fill_diag(terms[k], x, sub, scratch_vec, level_idx + 1U);
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
    eval_grad_dist(const Term& term, MatrixRef<T> dist, MatrixMut<T> d_k, std::size_t param_idx,
                   Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                   NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_grad_dist(leaves[leaf->leaf_idx], dist, d_k,
                                          param_idx - leaf->param_begin, uplo);
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                if (contains_param(child, param_idx)) {
                    return eval_grad_dist(child, dist, d_k, param_idx, uplo, scratch_a, scratch_b,
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
        if (auto res = eval_grad_dist(p.terms[owner], dist, d_k, param_idx, uplo, scratch_a,
                                      scratch_b, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner) {
                continue;
            }
            if (auto res =
                    eval_term_dist(p.terms[m], dist, scratch_a, uplo, scratch_b, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            mul_into(d_k, scratch_a.as_ref(), uplo);
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_grad_points(const Term& term, MatrixRef<T> x, MatrixMut<T> d_k, std::size_t param_idx,
                     Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                     NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_grad_points(leaves[leaf->leaf_idx], x, d_k,
                                            param_idx - leaf->param_begin, uplo);
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                if (contains_param(child, param_idx)) {
                    return eval_grad_points(child, x, d_k, param_idx, uplo, scratch_a, scratch_b,
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
        if (auto res = eval_grad_points(p.terms[owner], x, d_k, param_idx, uplo, scratch_a,
                                        scratch_b, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner) {
                continue;
            }
            if (auto res =
                    eval_term_points(p.terms[m], x, scratch_a, uplo, scratch_b, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            mul_into(d_k, scratch_a.as_ref(), uplo);
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_hess_dist(const Term& term, MatrixRef<T> dist, MatrixMut<T> d2_k, std::size_t i,
                   std::size_t j, Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                   MatrixMut<T> scratch_c, NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_hess_dist(leaves[leaf->leaf_idx], dist, d2_k, i - leaf->param_begin,
                                          j - leaf->param_begin, uplo);
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                const bool has_i = contains_param(child, i);
                const bool has_j = contains_param(child, j);
                if (has_i && has_j) {
                    return eval_hess_dist(child, dist, d2_k, i, j, uplo, scratch_a, scratch_b,
                                          scratch_c, nested, level_idx);
                }
                if (has_i || has_j) {
                    // Input shape, finiteness, and CoordMode were already validated at entry.
                    for (std::size_t col = 0; col < d2_k.cols(); ++col) {
                        for (std::size_t row = 0; row < d2_k.rows(); ++row) {
                            if (uplo == Triangle::Lower && row < col) {
                                continue;
                            }
                            if (uplo == Triangle::Upper && row > col) {
                                continue;
                            }
                            d2_k(row, col) = T{0};
                        }
                    }
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
            if (auto res = eval_hess_dist(p.terms[owner_i], dist, d2_k, i, j, uplo, scratch_a,
                                          scratch_b, scratch_c, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m) {
                if (m == owner_i) {
                    continue;
                }
                if (auto res = eval_term_dist(p.terms[m], dist, scratch_b, uplo, scratch_c, nested,
                                              level_idx);
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
        if (auto res = eval_grad_dist(p.terms[owner_i], dist, d2_k, i, uplo, work_a, work_b, nested,
                                      level_idx);
            !res.has_value()) {
            return res;
        }
        if (auto res = eval_grad_dist(p.terms[owner_j], dist, grad_rhs, j, uplo, work_a, work_b,
                                      nested, level_idx);
            !res.has_value()) {
            return res;
        }
        mul_into(d2_k, grad_rhs.as_ref(), uplo);
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner_i || m == owner_j) {
                continue;
            }
            if (auto res =
                    eval_term_dist(p.terms[m], dist, scratch_b, uplo, scratch_c, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            mul_into(d2_k, scratch_b.as_ref(), uplo);
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_hess_points(const Term& term, MatrixRef<T> x, MatrixMut<T> d2_k, std::size_t i,
                     std::size_t j, Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                     MatrixMut<T> scratch_c, NestedBuffers<T>& nested,
                     std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_hess_points(leaves[leaf->leaf_idx], x, d2_k, i - leaf->param_begin,
                                            j - leaf->param_begin, uplo);
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                const bool has_i = contains_param(child, i);
                const bool has_j = contains_param(child, j);
                if (has_i && has_j) {
                    return eval_hess_points(child, x, d2_k, i, j, uplo, scratch_a, scratch_b,
                                            scratch_c, nested, level_idx);
                }
                if (has_i || has_j) {
                    // Input shape, finiteness, and all ARD dimensions were already validated at
                    // entry.
                    for (std::size_t col = 0; col < d2_k.cols(); ++col) {
                        for (std::size_t row = 0; row < d2_k.rows(); ++row) {
                            if (uplo == Triangle::Lower && row < col) {
                                continue;
                            }
                            if (uplo == Triangle::Upper && row > col) {
                                continue;
                            }
                            d2_k(row, col) = T{0};
                        }
                    }
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
            if (auto res = eval_hess_points(p.terms[owner_i], x, d2_k, i, j, uplo, scratch_a,
                                            scratch_b, scratch_c, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m) {
                if (m == owner_i) {
                    continue;
                }
                if (auto res = eval_term_points(p.terms[m], x, scratch_b, uplo, scratch_c, nested,
                                                level_idx);
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
        if (auto res = eval_grad_points(p.terms[owner_i], x, d2_k, i, uplo, work_a, work_b, nested,
                                        level_idx);
            !res.has_value()) {
            return res;
        }
        if (auto res = eval_grad_points(p.terms[owner_j], x, grad_rhs, j, uplo, work_a, work_b,
                                        nested, level_idx);
            !res.has_value()) {
            return res;
        }
        mul_into(d2_k, grad_rhs.as_ref(), uplo);
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner_i || m == owner_j) {
                continue;
            }
            if (auto res =
                    eval_term_points(p.terms[m], x, scratch_b, uplo, scratch_c, nested, level_idx);
                !res.has_value()) {
                return res;
            }
            mul_into(d2_k, scratch_b.as_ref(), uplo);
        }
        return {};
    }
};

// --- CompiledKernel<T> public methods ---

template <KernelScalarConcept T>
CompiledKernel<T>::CompiledKernel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

template <KernelScalarConcept T> CompiledKernel<T>::~CompiledKernel() = default;

template <KernelScalarConcept T>
CompiledKernel<T>::CompiledKernel(const CompiledKernel& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

template <KernelScalarConcept T>
CompiledKernel<T>& CompiledKernel<T>::operator=(const CompiledKernel& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

template <KernelScalarConcept T>
CompiledKernel<T> CompiledKernel<T>::from_spec(const KernelSpec& spec) {
    auto impl = std::make_unique<Impl>();
    std::size_t next_param_idx = 0;
    impl->root = Impl::compile_node(spec, impl->leaves, impl->bindings, next_param_idx);
    CoordMode mode = CoordMode::Either;
    for (const auto& leaf : impl->leaves) {
        mode = combine_coord_mode(mode, detail::leaf_coord_mode(leaf));
    }
    impl->coord_mode = mode;
    impl->nested_depth = Impl::compute_nested_depth(impl->root);
    return CompiledKernel<T>(std::move(impl));
}

template <KernelScalarConcept T> std::size_t CompiledKernel<T>::num_params() const noexcept {
    return impl_->bindings.size();
}

template <KernelScalarConcept T> std::size_t CompiledKernel<T>::num_leaves() const noexcept {
    return impl_->leaves.size();
}

template <KernelScalarConcept T> CoordMode CompiledKernel<T>::coord_mode() const noexcept {
    return impl_->coord_mode;
}

template <KernelScalarConcept T> std::size_t CompiledKernel<T>::nested_depth() const noexcept {
    return impl_->nested_depth;
}

template <KernelScalarConcept T>
NestedBuffers<T> CompiledKernel<T>::make_nested_buffers(std::size_t rows, std::size_t cols) const {
    return NestedBuffers<T>(impl_->nested_depth, rows, cols);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::get_params(std::span<double> out) const {
    if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
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
std::expected<void, GprError> CompiledKernel<T>::set_params(std::span<const double> params) {
    if (auto cnt = require_count(params.size(), impl_->bindings.size(), "params");
        !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
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
    return {};
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::bounds_into(std::span<Interval> out) const {
    if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
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
const std::vector<ParameterBinding>& CompiledKernel<T>::parameter_bindings() const noexcept {
    return impl_->bindings;
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::apply(MatrixRef<T> dist, MatrixMut<T> out,
                                                       Triangle uplo, MatrixMut<T> scratch,
                                                       NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (auto v = impl_->validate_dist_call(dist, out.as_ref(), uplo, true); !v.has_value()) {
        return v;
    }
    return impl_->eval_term_dist(impl_->root, dist, out, uplo, scratch, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::apply(MatrixRef<T> dist, MatrixMut<T> out,
                                                       Triangle uplo, MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply(dist, out, uplo, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::apply_points(MatrixRef<T> x, MatrixMut<T> out,
                                                              Triangle uplo, MatrixMut<T> scratch,
                                                              NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value()) {
        return v;
    }
    return impl_->eval_term_points(impl_->root, x, out, uplo, scratch, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::apply_points(MatrixRef<T> x, MatrixMut<T> out,
                                                              Triangle uplo,
                                                              MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply_points(x, out, uplo, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::apply_cross(MatrixRef<T> dist, MatrixMut<T> out,
                                                             MatrixMut<T> scratch,
                                                             NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (auto v = impl_->validate_dist_call(dist, out.as_ref(), Triangle::Full, false);
        !v.has_value()) {
        return v;
    }
    return impl_->eval_term_cross_dist(impl_->root, dist, out, scratch, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::apply_cross(MatrixRef<T> dist, MatrixMut<T> out,
                                                             MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply_cross(dist, out, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_cross_points(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out,
                                      MatrixMut<T> scratch, NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    return impl_->eval_term_cross_points(impl_->root, x1, x2, out, scratch, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_cross_points(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out,
                                      MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply_cross_points(x1, x2, out, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::fill_diag(MatrixRef<T> x, std::span<T> out) const {
    if (auto ard = impl_->validate_ard_dims(x.cols()); !ard.has_value()) {
        return ard;
    }
    std::vector<T> scratch_vec((impl_->nested_depth + 1U) * out.size());
    return impl_->eval_fill_diag(impl_->root, x, out, scratch_vec, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::grad(MatrixRef<T> dist, MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo,
                        MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                        NestedBuffers<T>& nested) const {
    if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value()) {
        return sc_a;
    }
    if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value()) {
        return sc_b;
    }
    if (scratch_a.data() == scratch_b.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
    }
    if (auto v = impl_->validate_dist_call(dist, d_k.as_ref(), uplo, true); !v.has_value()) {
        return v;
    }
    if (param_idx >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_grad_dist(impl_->root, dist, d_k, param_idx, uplo, scratch_a, scratch_b,
                                 nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::grad(MatrixRef<T> dist, MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo,
                        MatrixMut<T> scratch_a, MatrixMut<T> scratch_b) const {
    auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
    return grad(dist, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::grad_points(MatrixRef<T> x, MatrixMut<T> d_k, std::size_t param_idx,
                               Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                               NestedBuffers<T>& nested) const {
    if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value()) {
        return sc_a;
    }
    if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value()) {
        return sc_b;
    }
    if (scratch_a.data() == scratch_b.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
    }
    if (auto v = impl_->validate_points_call(x, d_k.as_ref()); !v.has_value()) {
        return v;
    }
    if (param_idx >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_grad_points(impl_->root, x, d_k, param_idx, uplo, scratch_a, scratch_b,
                                   nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::grad_points(MatrixRef<T> x, MatrixMut<T> d_k,
                                                             std::size_t param_idx, Triangle uplo,
                                                             MatrixMut<T> scratch_a,
                                                             MatrixMut<T> scratch_b) const {
    auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
    return grad_points(x, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::hess(MatrixRef<T> dist, MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                        Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                        MatrixMut<T> scratch_c, NestedBuffers<T>& nested) const {
    if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value()) {
        return sc_a;
    }
    if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value()) {
        return sc_b;
    }
    if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c, "scratch_c"); !sc_c.has_value()) {
        return sc_c;
    }
    if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data() ||
        scratch_b.data() == scratch_c.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
    }
    if (auto v = impl_->validate_dist_call(dist, d2_k.as_ref(), uplo, true); !v.has_value()) {
        return v;
    }
    if (i >= impl_->bindings.size() || j >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_hess_dist(impl_->root, dist, d2_k, i, j, uplo, scratch_a, scratch_b,
                                 scratch_c, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::hess(MatrixRef<T> dist, MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                        Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                        MatrixMut<T> scratch_c) const {
    auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
    return hess(dist, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::hess_points(MatrixRef<T> x, MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                               Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                               MatrixMut<T> scratch_c, NestedBuffers<T>& nested) const {
    if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value()) {
        return sc_a;
    }
    if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value()) {
        return sc_b;
    }
    if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c, "scratch_c"); !sc_c.has_value()) {
        return sc_c;
    }
    if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data() ||
        scratch_b.data() == scratch_c.data()) {
        return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
    }
    if (auto v = impl_->validate_points_call(x, d2_k.as_ref()); !v.has_value()) {
        return v;
    }
    if (i >= impl_->bindings.size() || j >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_hess_points(impl_->root, x, d2_k, i, j, uplo, scratch_a, scratch_b,
                                   scratch_c, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::hess_points(MatrixRef<T> x, MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                               Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                               MatrixMut<T> scratch_c) const {
    auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
    return hess_points(x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
}

template class NestedBuffers<double>;
template class NestedBuffers<float>;
template class CompiledKernel<double>;
template class CompiledKernel<float>;

} // namespace cxxgpr
