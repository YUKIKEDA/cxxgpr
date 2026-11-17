// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/compiled.hpp"

#include "cxxgpr/core/checked.hpp"
#include "kernel/leaf_eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <variant>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cxxgpr {

namespace {

[[nodiscard]] bool leaf_reads_distances(const KernelLeaf& leaf) noexcept {
    return std::holds_alternative<RbfKernel>(leaf) || std::holds_alternative<MaternKernel>(leaf) ||
           std::holds_alternative<PeriodicKernel>(leaf) ||
           std::holds_alternative<RationalQuadraticKernel>(leaf) ||
           std::holds_alternative<CustomKernel>(leaf);
}

[[nodiscard]] bool leaf_needs_ard(const KernelLeaf& leaf) noexcept {
    return std::holds_alternative<RbfArdKernel>(leaf) ||
           std::holds_alternative<MaternArdKernel>(leaf) ||
           std::holds_alternative<RationalQuadraticArdKernel>(leaf);
}

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

template <KernelScalarConcept T, typename Op>
void apply_binary_op(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo, Op&& op) noexcept {
    const std::size_t n_rows = dst.rows();
    const std::size_t n_cols = dst.cols();
    const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
    if (uplo == Triangle::Lower && n_rows == n_cols) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_cols >= 64U)
#endif
        for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s) {
            const auto col = static_cast<std::size_t>(col_s);
            for (std::size_t row = col; row < n_rows; ++row) {
                op(dst(row, col), src(row, col));
            }
        }
        return;
    }
    if (uplo == Triangle::Upper && n_rows == n_cols) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_cols >= 64U)
#endif
        for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s) {
            const auto col = static_cast<std::size_t>(col_s);
            for (std::size_t row = 0; row <= col; ++row) {
                op(dst(row, col), src(row, col));
            }
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_cols >= 64U)
#endif
    for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s) {
        const auto col = static_cast<std::size_t>(col_s);
        for (std::size_t row = 0; row < n_rows; ++row) {
            op(dst(row, col), src(row, col));
        }
    }
}

template <KernelScalarConcept T>
void copy_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept {
    apply_binary_op(dst, src, uplo, [](T& d, T s) { d = s; });
}

template <KernelScalarConcept T>
void add_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept {
    apply_binary_op(dst, src, uplo, [](T& d, T s) { d += s; });
}

template <KernelScalarConcept T>
void mul_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept {
    apply_binary_op(dst, src, uplo, [](T& d, T s) { d *= s; });
}

template <KernelScalarConcept T> void scale_lower(MatrixMut<T> dst, T scale) noexcept {
    const std::size_t n = dst.rows();
    const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 64U)
#endif
    for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s) {
        const auto col = static_cast<std::size_t>(col_s);
        for (std::size_t row = col; row < n; ++row) {
            dst(row, col) *= scale;
        }
    }
}

template <KernelScalarConcept T>
[[nodiscard]] double lower_frobenius(MatrixRef<T> a, MatrixRef<T> b,
                                     std::span<double> col_partials) noexcept {
    const std::size_t n = a.rows();
    const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 64U)
#endif
    for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s) {
        const auto col = static_cast<std::size_t>(col_s);
        const double diag = static_cast<double>(a(col, col)) * static_cast<double>(b(col, col));
        double off = 0.0;
        for (std::size_t row = col + 1U; row < n; ++row) {
            off += static_cast<double>(a(row, col)) * static_cast<double>(b(row, col));
        }
        col_partials[col] = diag + (2.0 * off);
    }
    double total = 0.0;
    for (std::size_t col = 0; col < n; ++col) {
        total += col_partials[col];
    }
    return total;
}

template <KernelScalarConcept T>
[[nodiscard]] double lower_sum(MatrixRef<T> w, std::span<double> col_partials) noexcept {
    const std::size_t n = w.rows();
    const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 64U)
#endif
    for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s) {
        const auto col = static_cast<std::size_t>(col_s);
        const auto diag = static_cast<double>(w(col, col));
        double off = 0.0;
        for (std::size_t row = col + 1U; row < n; ++row) {
            off += static_cast<double>(w(row, col));
        }
        col_partials[col] = diag + (2.0 * off);
    }
    double total = 0.0;
    for (std::size_t col = 0; col < n; ++col) {
        total += col_partials[col];
    }
    return total;
}

template <KernelScalarConcept T> [[nodiscard]] double lower_diag_sum(MatrixRef<T> w) noexcept {
    const std::size_t n = w.rows();
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total += static_cast<double>(w(i, i));
    }
    return total;
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
    std::vector<std::vector<double>> cached_ls;
    std::vector<ParameterBinding> bindings;
    Term root;
    CoordMode coord_mode = CoordMode::Either;
    std::size_t nested_depth = 0;
    KernelExp exp_mode = KernelExp::Accurate;
    bool reads_dist = false;
    bool needs_ard = false;

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

    [[nodiscard]] std::expected<void, GprError>
    validate_cache_shape(MatrixRef<T> x, const DistanceCache<T>& cache) const {
        if (cache.has_dist()) {
            const auto dist = cache.dist_view();
            if (dist.rows() != x.rows() || dist.cols() != x.rows()) {
                return std::unexpected(GprError::length_mismatch("distance cache shape mismatch"));
            }
        }
        if (cache.has_ard()) {
            if (cache.num_points() != x.rows()) {
                return std::unexpected(GprError::length_mismatch("ARD cache row count mismatch"));
            }
            if (cache.num_dims() != x.cols()) {
                return std::unexpected(GprError::dimension_mismatch(cache.num_dims(), x.cols()));
            }
            const std::size_t per_dim = (x.rows() * (x.rows() + 1U)) / 2U;
            if (cache.ard_packed().size() != cache.num_dims() * per_dim) {
                return std::unexpected(GprError::length_mismatch("ARD cache length mismatch"));
            }
        }
        return {};
    }

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
        if (out.rows() != x.rows() || out.cols() != x.rows()) {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        if (auto ard = validate_ard_dims(x.cols()); !ard.has_value()) {
            return ard;
        }
        for (std::size_t col = 0; col < x.cols(); ++col) {
            for (std::size_t row = 0; row < x.rows(); ++row) {
                if (!std::isfinite(x(row, col))) {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GprError>
    validate_cross_points_call(MatrixRef<T> x1, MatrixRef<T> x2, MatrixRef<T> out) const {
        if (x1.rows() == 0 || x1.cols() == 0 || x2.rows() == 0 || x2.cols() == 0) {
            return std::unexpected(GprError::empty_input());
        }
        if (out.rows() != x1.rows() || out.cols() != x2.rows()) {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        if (x1.cols() != x2.cols()) {
            return std::unexpected(GprError::dimension_mismatch(x2.cols(), x1.cols()));
        }
        if (auto ard = validate_ard_dims(x1.cols()); !ard.has_value()) {
            return ard;
        }
        for (std::size_t col = 0; col < x1.cols(); ++col) {
            for (std::size_t row = 0; row < x1.rows(); ++row) {
                if (!std::isfinite(x1(row, col))) {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        for (std::size_t col = 0; col < x2.cols(); ++col) {
            for (std::size_t row = 0; row < x2.rows(); ++row) {
                if (!std::isfinite(x2(row, col))) {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
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
            return detail::leaf_apply_dist(leaves[leaf->leaf_idx], dist, out, uplo, exp_mode);
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
    eval_leaf_at(std::size_t leaf_idx, MatrixRef<T> x, MatrixMut<T> out, Triangle uplo,
                 NestedBuffers<T>& nested, std::size_t level_idx,
                 const DistanceCache<T>* cache = nullptr) const {
        const KernelLeaf& kl = leaves[leaf_idx];
        if (cache != nullptr) {
            if (cache->has_dist() && leaf_reads_distances(kl)) {
                return detail::leaf_apply_dist(kl, cache->dist_view(), out, uplo, exp_mode);
            }
            if (cache->has_ard() && leaf_needs_ard(kl)) {
                return detail::leaf_apply_ard_packed(kl, cached_ls[leaf_idx], cache->ard_packed(),
                                                     x.rows(), out, uplo, exp_mode);
            }
        }
        if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
            if (auto ok = ensure_nested(nested, level_idx, x.rows(), x.rows()); !ok.has_value()) {
                return ok;
            }
            auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
            fill_square_sq_dist(x, dist_buf);
            return custom->template term<T>().apply(dist_buf.as_ref(), out, uplo);
        }
        return detail::leaf_apply_points(kl, x, out, uplo, exp_mode, cached_ls[leaf_idx]);
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_combine_term(const Term& term, std::span<const MatrixRef<T>> leaf_grams, MatrixMut<T> out,
                      Triangle uplo, NestedBuffers<T>& nested, std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            copy_into(out, leaf_grams[leaf->leaf_idx], uplo);
            return {};
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_combine_term(terms[0], leaf_grams, out, uplo, nested, level_idx);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (const auto* leaf = std::get_if<LeafRef>(&terms[k])) {
                if (is_sum) {
                    add_into(out, leaf_grams[leaf->leaf_idx], uplo);
                } else {
                    mul_into(out, leaf_grams[leaf->leaf_idx], uplo);
                }
            } else {
                if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols());
                    !ok.has_value()) {
                    return ok;
                }
                auto buf = nested.level(level_idx, out.rows(), out.cols());
                if (auto res =
                        eval_combine_term(terms[k], leaf_grams, buf, uplo, nested, level_idx + 1U);
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
    eval_term_points(const Term& term, MatrixRef<T> x, MatrixMut<T> out, Triangle uplo,
                     MatrixMut<T> scratch, NestedBuffers<T>& nested, std::size_t level_idx,
                     const DistanceCache<T>* cache = nullptr) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return eval_leaf_at(leaf->leaf_idx, x, out, uplo, nested, level_idx, cache);
        }
        const bool is_sum = std::holds_alternative<SumTerm>(term);
        const auto& terms =
            is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
        if (terms.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        if (auto res = eval_term_points(terms[0], x, out, uplo, scratch, nested, level_idx, cache);
            !res.has_value()) {
            return res;
        }
        for (std::size_t k = 1; k < terms.size(); ++k) {
            if (std::holds_alternative<LeafRef>(terms[k])) {
                if (auto res =
                        eval_term_points(terms[k], x, scratch, uplo, out, nested, level_idx, cache);
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
                if (auto res = eval_term_points(terms[k], x, buf, uplo, scratch, nested,
                                                level_idx + 1, cache);
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
            return detail::leaf_apply_cross_dist(leaves[leaf->leaf_idx], dist, out, exp_mode);
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
            const KernelLeaf& kl = leaves[leaf->leaf_idx];
            if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                if (auto ok = ensure_nested(nested, level_idx, x1.rows(), x2.rows());
                    !ok.has_value()) {
                    return ok;
                }
                auto dist_buf = nested.level(level_idx, x1.rows(), x2.rows());
                fill_cross_sq_dist(x1, x2, dist_buf);
                return custom->template term<T>().apply_cross(dist_buf.as_ref(), out);
            }
            return detail::leaf_apply_cross_points(kl, x1, x2, out, exp_mode,
                                                   cached_ls[leaf->leaf_idx]);
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
                                                               std::span<T> scratch_vec,
                                                               std::size_t level_idx) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return detail::leaf_fill_diag(leaves[leaf->leaf_idx], x, out,
                                          cached_ls[leaf->leaf_idx]);
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
            return std::unexpected(
                GprError::length_mismatch("diagonal evaluation scratch buffer is too small"));
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
                                          param_idx - leaf->param_begin, uplo, exp_mode);
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
                     NestedBuffers<T>& nested, std::size_t level_idx,
                     const DistanceCache<T>* cache = nullptr) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            const KernelLeaf& kl = leaves[leaf->leaf_idx];
            if (cache != nullptr) {
                if (cache->has_dist() && leaf_reads_distances(kl)) {
                    return detail::leaf_grad_dist(kl, cache->dist_view(), d_k,
                                                  param_idx - leaf->param_begin, uplo, exp_mode);
                }
                if (cache->has_ard() && leaf_needs_ard(kl)) {
                    return detail::leaf_grad_ard_packed(
                        kl, cached_ls[leaf->leaf_idx], cache->ard_packed(), x.rows(), d_k,
                        param_idx - leaf->param_begin, uplo, exp_mode);
                }
            }
            if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                if (auto ok = ensure_nested(nested, level_idx, x.rows(), x.rows());
                    !ok.has_value()) {
                    return ok;
                }
                auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
                fill_square_sq_dist(x, dist_buf);
                return custom->template term<T>().grad(dist_buf.as_ref(), d_k,
                                                       param_idx - leaf->param_begin, uplo);
            }
            return detail::leaf_grad_points(kl, x, d_k, param_idx - leaf->param_begin, uplo,
                                            exp_mode, cached_ls[leaf->leaf_idx]);
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                if (contains_param(child, param_idx)) {
                    return eval_grad_points(child, x, d_k, param_idx, uplo, scratch_a, scratch_b,
                                            nested, level_idx, cache);
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
                                        scratch_b, nested, level_idx, cache);
            !res.has_value()) {
            return res;
        }
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner) {
                continue;
            }
            if (auto res = eval_term_points(p.terms[m], x, scratch_a, uplo, scratch_b, nested,
                                            level_idx, cache);
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
                                          j - leaf->param_begin, uplo, exp_mode);
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
                     MatrixMut<T> scratch_c, NestedBuffers<T>& nested, std::size_t level_idx,
                     const DistanceCache<T>* cache = nullptr) const {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            const KernelLeaf& kl = leaves[leaf->leaf_idx];
            if (cache != nullptr) {
                if (cache->has_dist() && leaf_reads_distances(kl)) {
                    return detail::leaf_hess_dist(kl, cache->dist_view(), d2_k,
                                                  i - leaf->param_begin, j - leaf->param_begin,
                                                  uplo, exp_mode);
                }
                if (cache->has_ard() && leaf_needs_ard(kl)) {
                    return detail::leaf_hess_ard_packed(
                        kl, cached_ls[leaf->leaf_idx], cache->ard_packed(), x.rows(), d2_k,
                        i - leaf->param_begin, j - leaf->param_begin, uplo, exp_mode);
                }
            }
            if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
                if (auto ok = ensure_nested(nested, level_idx, x.rows(), x.rows());
                    !ok.has_value()) {
                    return ok;
                }
                auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
                fill_square_sq_dist(x, dist_buf);
                return custom->template term<T>().hess(
                    dist_buf.as_ref(), d2_k, i - leaf->param_begin, j - leaf->param_begin, uplo);
            }
            return detail::leaf_hess_points(kl, x, d2_k, i - leaf->param_begin,
                                            j - leaf->param_begin, uplo, exp_mode,
                                            cached_ls[leaf->leaf_idx]);
        }
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            for (const auto& child : s->terms) {
                const bool has_i = contains_param(child, i);
                const bool has_j = contains_param(child, j);
                if (has_i && has_j) {
                    return eval_hess_points(child, x, d2_k, i, j, uplo, scratch_a, scratch_b,
                                            scratch_c, nested, level_idx, cache);
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
                                            scratch_b, scratch_c, nested, level_idx, cache);
                !res.has_value()) {
                return res;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m) {
                if (m == owner_i) {
                    continue;
                }
                if (auto res = eval_term_points(p.terms[m], x, scratch_b, uplo, scratch_c, nested,
                                                level_idx, cache);
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
                                        level_idx, cache);
            !res.has_value()) {
            return res;
        }
        if (auto res = eval_grad_points(p.terms[owner_j], x, grad_rhs, j, uplo, work_a, work_b,
                                        nested, level_idx, cache);
            !res.has_value()) {
            return res;
        }
        mul_into(d2_k, grad_rhs.as_ref(), uplo);
        for (std::size_t m = 0; m < p.terms.size(); ++m) {
            if (m == owner_i || m == owner_j) {
                continue;
            }
            if (auto res = eval_term_points(p.terms[m], x, scratch_b, uplo, scratch_c, nested,
                                            level_idx, cache);
                !res.has_value()) {
                return res;
            }
            mul_into(d2_k, scratch_b.as_ref(), uplo);
        }
        return {};
    }

    // --- keep_plan and weighted_grads ---

    [[nodiscard]] bool is_constant_term(const Term& term) const noexcept {
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            return std::holds_alternative<ConstantKernel>(leaves[leaf->leaf_idx]);
        }
        return false;
    }

    [[nodiscard]] std::size_t varying_factors(const std::vector<Term>& terms) const noexcept {
        std::size_t count = 0;
        for (const auto& t : terms) {
            if (!is_constant_term(t)) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] T constant_scale(const std::vector<Term>& terms) const noexcept {
        T scale = T{1};
        for (const auto& t : terms) {
            if (const auto* leaf = std::get_if<LeafRef>(&t)) {
                if (const auto* ck = std::get_if<ConstantKernel>(&leaves[leaf->leaf_idx])) {
                    scale *= static_cast<T>(ck->constant());
                }
            }
        }
        return scale;
    }

    [[nodiscard]] std::span<const Term> root_terms() const noexcept {
        if (const auto* s = std::get_if<SumTerm>(&root)) {
            return s->terms;
        }
        return std::span<const Term>(&root, 1U);
    }

    [[nodiscard]] static std::size_t unkept_buffers_of(const Term& term) noexcept {
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            std::size_t mx = 0;
            for (const auto& child : s->terms) {
                mx = std::max(mx, unkept_buffers_of(child));
            }
            return mx;
        }
        if (const auto* p = std::get_if<ProductTerm>(&term)) {
            std::size_t mx = 0;
            for (const auto& child : p->terms) {
                mx = std::max(mx, unkept_buffers_of(child));
            }
            return p->terms.size() + 1U + mx;
        }
        return 1U;
    }

    [[nodiscard]] std::size_t node_buffers_of(const Term& term, bool keeps) const noexcept {
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            std::size_t mx = 0;
            for (const auto& child : s->terms) {
                mx = std::max(mx, node_buffers_of(child, false));
            }
            return mx;
        }
        if (const auto* p = std::get_if<ProductTerm>(&term)) {
            std::size_t mx = 0;
            for (const auto& child : p->terms) {
                mx = std::max(mx, node_buffers_of(child, false));
            }
            if (keeps) {
                return 1U + mx;
            }
            return varying_factors(p->terms) + 1U + mx;
        }
        return 1U;
    }

    [[nodiscard]] std::size_t kept_grams_count(std::size_t products) const noexcept {
        std::size_t kept = 0;
        std::size_t grams = 0;
        for (const auto& t : root_terms()) {
            if (const auto* p = std::get_if<ProductTerm>(&t)) {
                const std::size_t v = varying_factors(p->terms);
                if (v >= 2U && kept < products) {
                    ++kept;
                    grams += v;
                }
            }
        }
        return grams;
    }

    [[nodiscard]] std::size_t walk_buffers_count(std::size_t products) const noexcept {
        std::size_t kept = 0;
        std::size_t mx = 0;
        for (const auto& t : root_terms()) {
            bool keeps = false;
            if (const auto* p = std::get_if<ProductTerm>(&t)) {
                if (varying_factors(p->terms) >= 2U && kept < products) {
                    ++kept;
                    keeps = true;
                }
            }
            mx = std::max(mx, node_buffers_of(t, keeps));
        }
        return mx;
    }

    [[nodiscard]] std::size_t compute_kept_products() const noexcept {
        const std::size_t budget = unkept_buffers_of(root);
        std::size_t candidates = 0;
        for (const auto& t : root_terms()) {
            if (const auto* p = std::get_if<ProductTerm>(&t)) {
                if (varying_factors(p->terms) >= 2U) {
                    ++candidates;
                }
            }
        }
        for (std::size_t p = candidates; p > 0U; --p) {
            if (kept_grams_count(p) + walk_buffers_count(p) <= budget) {
                return p;
            }
        }
        return 0U;
    }

    [[nodiscard]] std::expected<void, GprError>
    grad_leaf_at(std::size_t leaf_idx, MatrixRef<T> x, const DistanceCache<T>* cache,
                 MatrixMut<T> d_k, std::size_t local_idx, Triangle uplo, NestedBuffers<T>& nested,
                 std::size_t level_idx) const {
        const KernelLeaf& kl = leaves[leaf_idx];
        if (cache != nullptr) {
            if (cache->has_dist() && leaf_reads_distances(kl)) {
                return detail::leaf_grad_dist(kl, cache->dist_view(), d_k, local_idx, uplo,
                                              exp_mode);
            }
            if (cache->has_ard() && leaf_needs_ard(kl)) {
                return detail::leaf_grad_ard_packed(kl, cached_ls[leaf_idx], cache->ard_packed(),
                                                    x.rows(), d_k, local_idx, uplo, exp_mode);
            }
        }
        if (const auto* custom = std::get_if<CustomKernel>(&kl)) {
            if (auto ok = ensure_nested(nested, level_idx, x.rows(), x.rows()); !ok.has_value()) {
                return ok;
            }
            auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
            fill_square_sq_dist(x, dist_buf);
            return custom->template term<T>().grad(dist_buf.as_ref(), d_k, local_idx, uplo);
        }
        return detail::leaf_grad_points(kl, x, d_k, local_idx, uplo, exp_mode, cached_ls[leaf_idx]);
    }

    [[nodiscard]] std::expected<void, GprError>
    eval_keeping_impl(MatrixRef<T> x, const DistanceCache<T>* cache, MatrixMut<T> out,
                      Triangle uplo, MatrixMut<T> scratch, NestedBuffers<T>& nested,
                      NestedBuffers<T>& kept_bufs, std::size_t products) const {
        const auto roots = root_terms();
        if (roots.empty()) {
            return std::unexpected(
                GprError::unsupported_kernel_operation("composite kernel has no terms"));
        }
        const std::size_t n = out.rows();
        std::size_t kept = 0;
        std::size_t offset = 0;
        for (std::size_t r_idx = 0; r_idx < roots.size(); ++r_idx) {
            const Term& t = roots[r_idx];
            MatrixMut<T> target = (r_idx == 0U) ? out : scratch;
            bool keeps = false;
            if (const auto* p = std::get_if<ProductTerm>(&t)) {
                const std::size_t v = varying_factors(p->terms);
                if (v >= 2U && kept < products) {
                    keeps = true;
                    ++kept;
                    const T scale = constant_scale(p->terms);
                    std::size_t v_idx = 0;
                    for (const auto& f : p->terms) {
                        if (is_constant_term(f)) {
                            continue;
                        }
                        if (kept_bufs.depth() <= (offset + v_idx) || kept_bufs.rows() < n ||
                            kept_bufs.cols() < n) {
                            return std::unexpected(GprError::workspace_too_small());
                        }
                        auto kbuf = kept_bufs.level(offset + v_idx, n, n);
                        MatrixMut<T> tmp = scratch;
                        std::size_t sub_level = 0U;
                        if (r_idx > 0U && v_idx > 0U && !std::holds_alternative<LeafRef>(f)) {
                            if (auto ok = ensure_nested(nested, 0U, n, n); !ok.has_value()) {
                                return ok;
                            }
                            tmp = nested.level(0U, n, n);
                            sub_level = 1U;
                        }
                        if (auto res =
                                eval_term_points(f, x, kbuf, uplo, tmp, nested, sub_level, cache);
                            !res.has_value()) {
                            return res;
                        }
                        if (v_idx == 0U) {
                            copy_into(target, kbuf.as_ref(), uplo);
                            if (scale != T{1}) {
                                scale_lower(target, scale);
                            }
                        } else {
                            mul_into(target, kbuf.as_ref(), uplo);
                        }
                        ++v_idx;
                    }
                    offset += v;
                }
            }
            if (!keeps) {
                if (r_idx == 0U || std::holds_alternative<LeafRef>(t)) {
                    if (auto res = eval_term_points(t, x, target, uplo, scratch, nested, 0U, cache);
                        !res.has_value()) {
                        return res;
                    }
                } else {
                    if (auto ok = ensure_nested(nested, 0U, n, n); !ok.has_value()) {
                        return ok;
                    }
                    auto nbuf = nested.level(0U, n, n);
                    if (auto res = eval_term_points(t, x, nbuf, uplo, scratch, nested, 1U, cache);
                        !res.has_value()) {
                        return res;
                    }
                    add_into(out, nbuf.as_ref(), uplo);
                    continue;
                }
            }
            if (r_idx > 0U) {
                add_into(out, target.as_ref(), uplo);
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<double, GprError>
    walk_weighted(const Term& term, MatrixRef<T> x, const DistanceCache<T>* cache,
                  MatrixRef<T> weight, std::span<double> out_grads, MatrixMut<T> kernel_scratch,
                  NestedBuffers<T>& nested, NestedBuffers<T>& weighted_bufs,
                  std::size_t kept_offset, std::size_t kept_count, bool has_kept,
                  std::size_t walk_cursor, std::span<double> col_partials, bool need_value) const {
        const std::size_t n = x.rows();
        if (const auto* s = std::get_if<SumTerm>(&term)) {
            std::size_t p_off = 0;
            double value = 0.0;
            for (const auto& child : s->terms) {
                const auto [cb, ce] = param_range(child);
                const std::size_t count = ce - cb;
                auto res = walk_weighted(child, x, cache, weight, out_grads.subspan(p_off, count),
                                         kernel_scratch, nested, weighted_bufs, 0, 0, false,
                                         walk_cursor, col_partials, need_value);
                if (!res.has_value()) {
                    return res;
                }
                value += *res;
                p_off += count;
            }
            return value;
        }
        if (const auto* leaf = std::get_if<LeafRef>(&term)) {
            const KernelLeaf& kl = leaves[leaf->leaf_idx];
            if (const auto* ck = std::get_if<ConstantKernel>(&kl)) {
                const double val = ck->constant() * lower_sum(weight, col_partials);
                out_grads[0] = val;
                return val;
            }
            if (const auto* wk = std::get_if<WhiteKernel>(&kl)) {
                const double val = wk->variance() * lower_diag_sum(weight);
                out_grads[0] = val;
                return need_value ? val : 0.0;
            }
            for (std::size_t p = 0; p < leaf->param_count; ++p) {
                if (auto res = grad_leaf_at(leaf->leaf_idx, x, cache, kernel_scratch, p,
                                            Triangle::Lower, nested, 0);
                    !res.has_value()) {
                    return std::unexpected(std::move(res).error());
                }
                out_grads[p] = lower_frobenius(weight, kernel_scratch.as_ref(), col_partials);
            }
            if (need_value) {
                if (auto res = eval_leaf_at(leaf->leaf_idx, x, kernel_scratch, Triangle::Lower,
                                            nested, 0, cache);
                    !res.has_value()) {
                    return std::unexpected(std::move(res).error());
                }
                return lower_frobenius(weight, kernel_scratch.as_ref(), col_partials);
            }
            return 0.0;
        }

        const auto& p = std::get<ProductTerm>(term);
        const std::size_t varying = varying_factors(p.terms);
        const T scale = constant_scale(p.terms);
        const bool has_const = (varying < p.terms.size());

        if (varying == 0U) {
            const double val = static_cast<double>(scale) * lower_sum(weight, col_partials);
            std::size_t p_off = 0;
            for (const auto& f : p.terms) {
                const auto [fb, fe] = param_range(f);
                out_grads[p_off] = val;
                p_off += (fe - fb);
            }
            return val;
        }

        if (varying == 1U) {
            double val = 0.0;
            if (has_const) {
                if (auto ok = ensure_nested(weighted_bufs, walk_cursor, n, n); !ok.has_value()) {
                    return std::unexpected(std::move(ok).error());
                }
            }
            std::size_t p_off = 0;
            for (const auto& f : p.terms) {
                const auto [fb, fe] = param_range(f);
                const std::size_t count = fe - fb;
                if (!is_constant_term(f)) {
                    MatrixRef<T> eff_w = weight;
                    if (has_const) {
                        auto w_buf = weighted_bufs.level(walk_cursor, n, n);
                        copy_into(w_buf, weight, Triangle::Lower);
                        if (scale != T{1}) {
                            scale_lower(w_buf, scale);
                        }
                        eff_w = w_buf.as_ref();
                    }
                    auto res = walk_weighted(f, x, cache, eff_w, out_grads.subspan(p_off, count),
                                             kernel_scratch, nested, weighted_bufs, 0, 0, false,
                                             has_const ? (walk_cursor + 1U) : walk_cursor,
                                             col_partials, need_value || has_const);
                    if (!res.has_value()) {
                        return res;
                    }
                    val = *res;
                }
                p_off += count;
            }
            if (has_const) {
                std::size_t c_off = 0;
                for (const auto& f : p.terms) {
                    const auto [fb, fe] = param_range(f);
                    if (is_constant_term(f)) {
                        out_grads[c_off] = val;
                    }
                    c_off += (fe - fb);
                }
            }
            return val;
        }

        // varying >= 2
        const std::size_t g_base = has_kept ? kept_offset : walk_cursor;
        const std::size_t next_cursor = has_kept ? walk_cursor : (walk_cursor + varying);
        if (!has_kept) {
            std::size_t v_idx = 0;
            for (const auto& f : p.terms) {
                if (is_constant_term(f)) {
                    continue;
                }
                if (auto ok = ensure_nested(weighted_bufs, g_base + v_idx, n, n); !ok.has_value()) {
                    return std::unexpected(std::move(ok).error());
                }
                auto g_buf = weighted_bufs.level(g_base + v_idx, n, n);
                if (auto res = eval_term_points(f, x, g_buf, Triangle::Lower, kernel_scratch,
                                                nested, 0, cache);
                    !res.has_value()) {
                    return std::unexpected(std::move(res).error());
                }
                ++v_idx;
            }
        } else if (kept_count < varying) {
            return std::unexpected(GprError::workspace_too_small());
        }

        if (auto ok = ensure_nested(weighted_bufs, next_cursor, n, n); !ok.has_value()) {
            return std::unexpected(std::move(ok).error());
        }
        auto w_eff = weighted_bufs.level(next_cursor, n, n);

        double val = 0.0;
        std::size_t p_off = 0;
        std::size_t cur_v = 0;
        for (const auto& f : p.terms) {
            const auto [fb, fe] = param_range(f);
            const std::size_t count = fe - fb;
            if (!is_constant_term(f)) {
                copy_into(w_eff, weight, Triangle::Lower);
                if (scale != T{1}) {
                    scale_lower(w_eff, scale);
                }
                for (std::size_t r = 0; r < varying; ++r) {
                    if (r == cur_v) {
                        continue;
                    }
                    mul_into(w_eff, weighted_bufs.level(g_base + r, n, n).as_ref(),
                             Triangle::Lower);
                }
                if (auto res =
                        walk_weighted(f, x, cache, w_eff.as_ref(), out_grads.subspan(p_off, count),
                                      kernel_scratch, nested, weighted_bufs, 0, 0, false,
                                      next_cursor + 1U, col_partials, false);
                    !res.has_value()) {
                    return res;
                }
                ++cur_v;
            }
            p_off += count;
        }

        if (need_value || has_const) {
            copy_into(w_eff, weight, Triangle::Lower);
            if (scale != T{1}) {
                scale_lower(w_eff, scale);
            }
            for (std::size_t r = 1U; r < varying; ++r) {
                mul_into(w_eff, weighted_bufs.level(g_base + r, n, n).as_ref(), Triangle::Lower);
            }
            val = lower_frobenius(w_eff.as_ref(), weighted_bufs.level(g_base, n, n).as_ref(),
                                  col_partials);
            if (has_const) {
                std::size_t c_off = 0;
                for (const auto& f : p.terms) {
                    const auto [fb, fe] = param_range(f);
                    if (is_constant_term(f)) {
                        out_grads[c_off] = val;
                    }
                    c_off += (fe - fb);
                }
            }
        }
        return val;
    }
};

// --- CompiledKernel<T> public methods ---

template <KernelScalarConcept T>
CompiledKernel<T>::CompiledKernel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

template <KernelScalarConcept T>
CompiledKernel<T>::CompiledKernel(CompiledKernel&& other) noexcept = default;
template <KernelScalarConcept T>
CompiledKernel<T>& CompiledKernel<T>::operator=(CompiledKernel&& other) noexcept = default;
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
    bool has_custom_leaf = false;
    bool reads_dist = false;
    bool needs_ard = false;
    for (const auto& leaf : impl->leaves) {
        mode = combine_coord_mode(mode, detail::leaf_coord_mode(leaf));
        if (std::holds_alternative<CustomKernel>(leaf)) {
            has_custom_leaf = true;
        }
        if (leaf_reads_distances(leaf)) {
            reads_dist = true;
        }
        if (leaf_needs_ard(leaf)) {
            needs_ard = true;
        }
    }
    impl->coord_mode = mode;
    impl->reads_dist = reads_dist;
    impl->needs_ard = needs_ard;
    impl->nested_depth = Impl::compute_nested_depth(impl->root) + (has_custom_leaf ? 1U : 0U);
    impl->refresh_cached_ls();
    return CompiledKernel<T>(std::move(impl));
}

template <KernelScalarConcept T> KernelExp CompiledKernel<T>::kernel_exp() const noexcept {
    return impl_->exp_mode;
}

template <KernelScalarConcept T> void CompiledKernel<T>::set_kernel_exp(KernelExp mode) noexcept {
    impl_->exp_mode = mode;
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

template <KernelScalarConcept T> bool CompiledKernel<T>::reads_distances() const noexcept {
    return impl_->reads_dist;
}

template <KernelScalarConcept T> bool CompiledKernel<T>::needs_ard_sq_diff() const noexcept {
    return impl_->needs_ard;
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
    impl_->refresh_cached_ls();
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
    if (auto v = impl_->validate_cross_points_call(x1, x2, out.as_ref()); !v.has_value()) {
        return v;
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
std::expected<void, GprError> CompiledKernel<T>::fill_diag(MatrixRef<T> x, std::span<T> out,
                                                           std::span<T> scratch) const {
    if (x.rows() == 0 || x.cols() == 0) {
        return std::unexpected(GprError::empty_input());
    }
    if (auto cnt = require_count(out.size(), x.rows(), "out"); !cnt.has_value()) {
        return cnt;
    }
    if (auto ard = impl_->validate_ard_dims(x.cols()); !ard.has_value()) {
        return ard;
    }
    const std::size_t min_scratch = (impl_->nested_depth + 1U) * out.size();
    if (scratch.size() < min_scratch) {
        return std::unexpected(
            GprError::length_mismatch("diagonal evaluation scratch buffer is too small"));
    }
    for (std::size_t col = 0; col < x.cols(); ++col) {
        for (std::size_t row = 0; row < x.rows(); ++row) {
            if (!std::isfinite(x(row, col))) {
                return std::unexpected(GprError::non_finite_input());
            }
        }
    }
    return impl_->eval_fill_diag(impl_->root, x, out, scratch, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::fill_diag(MatrixRef<T> x, std::span<T> out) const {
    const std::size_t min_scratch = (impl_->nested_depth + 1U) * out.size();
    std::vector<T> scratch_vec(min_scratch);
    return fill_diag(x, out, scratch_vec);
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

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_cached(MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> out,
                                Triangle uplo, MatrixMut<T> scratch,
                                NestedBuffers<T>& nested) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value()) {
        return v;
    }
    if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value()) {
        return cs;
    }
    return impl_->eval_term_points(impl_->root, x, out, uplo, scratch, nested, 0, &cache);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_cached(MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> out,
                                Triangle uplo, MatrixMut<T> scratch) const {
    auto nested = make_nested_buffers(out.rows(), out.cols());
    return apply_cached(x, cache, out, uplo, scratch, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::grad_cached(MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> d_k,
                               std::size_t param_idx, Triangle uplo, MatrixMut<T> scratch_a,
                               MatrixMut<T> scratch_b, NestedBuffers<T>& nested) const {
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
    if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value()) {
        return cs;
    }
    if (param_idx >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_grad_points(impl_->root, x, d_k, param_idx, uplo, scratch_a, scratch_b,
                                   nested, 0, &cache);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::grad_cached(MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> d_k,
                               std::size_t param_idx, Triangle uplo, MatrixMut<T> scratch_a,
                               MatrixMut<T> scratch_b) const {
    auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
    return grad_cached(x, cache, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::hess_cached(MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> d2_k,
                               std::size_t i, std::size_t j, Triangle uplo, MatrixMut<T> scratch_a,
                               MatrixMut<T> scratch_b, MatrixMut<T> scratch_c,
                               NestedBuffers<T>& nested) const {
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
    if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value()) {
        return cs;
    }
    if (i >= impl_->bindings.size() || j >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->eval_hess_points(impl_->root, x, d2_k, i, j, uplo, scratch_a, scratch_b,
                                   scratch_c, nested, 0, &cache);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::hess_cached(MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> d2_k,
                               std::size_t i, std::size_t j, Triangle uplo, MatrixMut<T> scratch_a,
                               MatrixMut<T> scratch_b, MatrixMut<T> scratch_c) const {
    auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
    return hess_cached(x, cache, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
}

template <KernelScalarConcept T>
std::expected<std::size_t, GprError>
CompiledKernel<T>::leaf_index_for_param(std::size_t param_idx) const {
    if (param_idx >= impl_->bindings.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
    }
    return impl_->bindings[param_idx].leaf_id;
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_leaf_points(std::size_t leaf_idx, MatrixRef<T> x, MatrixMut<T> out,
                                     Triangle uplo, NestedBuffers<T>& nested) const {
    if (leaf_idx >= impl_->leaves.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel leaf index out of range"));
    }
    if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value()) {
        return v;
    }
    return impl_->eval_leaf_at(leaf_idx, x, out, uplo, nested, 0);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_leaf_cached(std::size_t leaf_idx, MatrixRef<T> x,
                                     const DistanceCache<T>& cache, MatrixMut<T> out, Triangle uplo,
                                     NestedBuffers<T>& nested) const {
    if (leaf_idx >= impl_->leaves.size()) {
        return std::unexpected(GprError::index_out_of_range("kernel leaf index out of range"));
    }
    if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value()) {
        return v;
    }
    if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value()) {
        return cs;
    }
    return impl_->eval_leaf_at(leaf_idx, x, out, uplo, nested, 0, &cache);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::combine_from_leaf_grams(std::span<const MatrixRef<T>> leaf_grams,
                                           MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch,
                                           NestedBuffers<T>& nested) const {
    if (out.is_empty() || out.data() == nullptr) {
        return std::unexpected(GprError::empty_input());
    }
    if (out.rows() != out.cols()) {
        return std::unexpected(GprError::length_mismatch("output matrix must be square"));
    }
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (leaf_grams.size() != impl_->leaves.size()) {
        return std::unexpected(
            GprError::length_mismatch("leaf Gram count does not match compiled kernel leaves"));
    }
    for (const auto& gram : leaf_grams) {
        if (gram.rows() != out.rows() || gram.cols() != out.cols() || gram.data() == nullptr) {
            return std::unexpected(GprError::length_mismatch("leaf Gram matrix shape mismatch"));
        }
    }
    return impl_->eval_combine_term(impl_->root, leaf_grams, out, uplo, nested, 0);
}

template <KernelScalarConcept T> std::size_t CompiledKernel<T>::kept_products() const noexcept {
    return impl_->compute_kept_products();
}

template <KernelScalarConcept T>
std::size_t CompiledKernel<T>::kept_buffers(std::size_t products) const noexcept {
    return impl_->kept_grams_count(products);
}

template <KernelScalarConcept T>
std::size_t CompiledKernel<T>::weighted_buffers(std::size_t products) const noexcept {
    return impl_->kept_grams_count(products) + impl_->walk_buffers_count(products);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_keeping(MatrixRef<T> x, MatrixMut<T> out, Triangle uplo,
                                 MatrixMut<T> scratch, NestedBuffers<T>& nested,
                                 NestedBuffers<T>& kept_bufs, std::size_t products) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value()) {
        return v;
    }
    return impl_->eval_keeping_impl(x, nullptr, out, uplo, scratch, nested, kept_bufs, products);
}

template <KernelScalarConcept T>
std::expected<void, GprError>
CompiledKernel<T>::apply_cached_keeping(MatrixRef<T> x, const DistanceCache<T>& cache,
                                        MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch,
                                        NestedBuffers<T>& nested, NestedBuffers<T>& kept_bufs,
                                        std::size_t products) const {
    if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value()) {
        return sc;
    }
    if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value()) {
        return v;
    }
    if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value()) {
        return cs;
    }
    return impl_->eval_keeping_impl(x, &cache, out, uplo, scratch, nested, kept_bufs, products);
}

template <KernelScalarConcept T>
std::expected<void, GprError> CompiledKernel<T>::weighted_grads(
    MatrixRef<T> x, const DistanceCache<T>* cache, MatrixRef<T> weight_lower,
    std::span<double> out_grads, MatrixMut<T> kernel_scratch, NestedBuffers<T>& nested,
    NestedBuffers<T>& weighted_bufs, std::size_t products, std::span<double> col_partials) const {
    if (out_grads.size() != impl_->bindings.size()) {
        return std::unexpected(GprError::length_mismatch("weighted_grads output length mismatch"));
    }
    if (auto v = impl_->validate_points_call(x, weight_lower); !v.has_value()) {
        return v;
    }
    if (cache != nullptr) {
        if (auto cs = impl_->validate_cache_shape(x, *cache); !cs.has_value()) {
            return cs;
        }
    }
    if (auto sc = require_scratch(weight_lower, kernel_scratch, "kernel_scratch");
        !sc.has_value()) {
        return sc;
    }
    if (col_partials.size() < x.rows()) {
        return std::unexpected(GprError::workspace_too_small());
    }
    const std::size_t needed_bufs = weighted_buffers(products);
    if (weighted_bufs.depth() < needed_bufs || weighted_bufs.rows() < x.rows() ||
        weighted_bufs.cols() < x.rows()) {
        return std::unexpected(GprError::workspace_too_small());
    }

    const std::size_t kept_total = impl_->kept_grams_count(products);
    std::size_t p_off = 0;
    std::size_t kept_idx = 0;
    std::size_t kept_off = 0;
    for (const auto& t : impl_->root_terms()) {
        const auto [tb, te] = Impl::param_range(t);
        const std::size_t count = te - tb;
        std::size_t v = 0;
        bool has_kept = false;
        if (const auto* p = std::get_if<typename Impl::ProductTerm>(&t)) {
            v = impl_->varying_factors(p->terms);
            if (v >= 2U && kept_idx < products) {
                has_kept = true;
                ++kept_idx;
            }
        }
        auto res =
            impl_->walk_weighted(t, x, cache, weight_lower, out_grads.subspan(p_off, count),
                                 kernel_scratch, nested, weighted_bufs, has_kept ? kept_off : 0U,
                                 has_kept ? v : 0U, has_kept, kept_total, col_partials, false);
        if (!res.has_value()) {
            return std::unexpected(std::move(res).error());
        }
        if (has_kept) {
            kept_off += v;
        }
        p_off += count;
    }
    return {};
}

template class NestedBuffers<double>;
template class NestedBuffers<float>;
template class CompiledKernel<double>;
template class CompiledKernel<float>;

} // namespace cxxgpr
