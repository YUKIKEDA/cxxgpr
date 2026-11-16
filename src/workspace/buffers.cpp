// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/workspace/buffers.hpp"

#include "cxxgpr/core/checked.hpp"

#include <utility>

namespace cxxgpr {

template <KernelScalarConcept S>
std::expected<void, GprError> HessianScratch<S>::ensure(std::size_t n, std::size_t p) {
    if (n == 0U || p == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    auto n2 = checked_mul(n, n);
    if (!n2.has_value()) {
        return std::unexpected(std::move(n2).error());
    }
    auto pn2 = checked_mul(p, *n2);
    if (!pn2.has_value()) {
        return std::unexpected(std::move(pn2).error());
    }
    auto np = checked_mul(n, p);
    if (!np.has_value()) {
        return std::unexpected(std::move(np).error());
    }

    if (s_storage.size() < *pn2) {
        s_storage.resize(*pn2);
    }
    if (v_storage.size() < *np) {
        v_storage.resize(*np);
    }
    if (u.size() < n) {
        u.resize(n);
    }
    if (scratch_c.size() < *n2) {
        scratch_c.resize(*n2);
    }
    order_ = n;
    num_params_ = p;
    return {};
}

template <KernelScalarConcept S> MatrixMut<S> HessianScratch<S>::s_mut(std::size_t i) noexcept {
    const std::size_t n2 = order_ * order_;
    return {s_storage.data() + (i * n2), order_, order_, order_};
}

template <KernelScalarConcept S>
MatrixRef<S> HessianScratch<S>::s_ref(std::size_t i) const noexcept {
    const std::size_t n2 = order_ * order_;
    return {s_storage.data() + (i * n2), order_, order_, order_};
}

template <KernelScalarConcept S> std::span<S> HessianScratch<S>::v_col_mut(std::size_t i) noexcept {
    return {v_storage.data() + (i * order_), order_};
}

template <KernelScalarConcept S>
std::span<const S> HessianScratch<S>::v_col_ref(std::size_t i) const noexcept {
    return {v_storage.data() + (i * order_), order_};
}

template <KernelScalarConcept S> MatrixMut<S> HessianScratch<S>::scratch_c_mut() noexcept {
    return {scratch_c.data(), order_, order_, order_};
}

template <PrecisionPolicyConcept P>
std::expected<WorkspaceCore<P>, GprError> WorkspaceCore<P>::create(std::size_t n) {
    if (n == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    auto n2 = checked_mul(n, n);
    if (!n2.has_value()) {
        return std::unexpected(std::move(n2).error());
    }

    WorkspaceCore<P> core;
    core.order_ = n;
    core.k_matrix.assign(*n2, Storage{0});
    core.exp_buf.assign(*n2, Storage{0});
    core.rhs.assign(n, Storage{0});
    core.col_partials.assign(n, 0.0);
    return core;
}

template <PrecisionPolicyConcept P>
MatrixMut<typename P::Storage> WorkspaceCore<P>::k_mut() noexcept {
    return {k_matrix.data(), order_, order_, order_};
}

template <PrecisionPolicyConcept P>
MatrixRef<typename P::Storage> WorkspaceCore<P>::k_ref() const noexcept {
    return {k_matrix.data(), order_, order_, order_};
}

template <PrecisionPolicyConcept P>
MatrixMut<typename P::Storage> WorkspaceCore<P>::exp_mut() noexcept {
    return {exp_buf.data(), order_, order_, order_};
}

template <PrecisionPolicyConcept P>
MatrixRef<typename P::Storage> WorkspaceCore<P>::exp_ref() const noexcept {
    return {exp_buf.data(), order_, order_, order_};
}

template <PrecisionPolicyConcept P>
MatrixMut<typename P::Storage> WorkspaceCore<P>::kernel_scratch_mut() noexcept {
    if (kernel_scratch.empty()) {
        return {};
    }
    return {kernel_scratch.data(), order_, order_, order_};
}

template <PrecisionPolicyConcept P>
std::expected<void, GprError> WorkspaceCore<P>::ensure_kernel_scratch() {
    if (order_ == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    auto n2 = checked_mul(order_, order_);
    if (!n2.has_value()) {
        return std::unexpected(std::move(n2).error());
    }
    if (kernel_scratch.size() < *n2) {
        kernel_scratch.assign(*n2, Storage{0});
    }
    return {};
}

template <PrecisionPolicyConcept P>
std::expected<void, GprError> WorkspaceCore<P>::ensure_weighted(std::size_t n_bufs) {
    if (order_ == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    if (weighted.depth() < n_bufs || weighted.rows() < order_ || weighted.cols() < order_) {
        weighted.resize(n_bufs, order_, order_);
    }
    return {};
}

template <PrecisionPolicyConcept P>
std::expected<FitBuffers<P>, GprError>
FitBuffers<P>::create(std::size_t n, DistanceCachePolicy cache, CholeskyBuffer buffer) {
    auto core = WorkspaceCore<P>::create(n);
    if (!core.has_value()) {
        return std::unexpected(std::move(core).error());
    }

    FitBuffers<P> out;
    out.core = std::move(*core);
    out.cache_policy = cache;
    out.buffer_policy = buffer;
    if (cache == DistanceCachePolicy::Cached) {
        out.dist.emplace();
    }
    if (buffer == CholeskyBuffer::Retain) {
        out.w_matrix.assign(out.core.k_matrix.size(), Storage{0});
    }
    return out;
}

template <PrecisionPolicyConcept P>
std::expected<FitBuffers<P>, GprError> FitBuffers<P>::create(std::size_t n, FitPolicy policy) {
    return create(n, policy.distance_cache, policy.cholesky_buffer);
}

template <PrecisionPolicyConcept P> MatrixMut<typename P::Storage> FitBuffers<P>::w_mut() noexcept {
    if (buffer_policy == CholeskyBuffer::Retain) {
        return {w_matrix.data(), core.order_, core.order_, core.order_};
    }
    return core.k_mut();
}

template <PrecisionPolicyConcept P>
MatrixRef<typename P::Storage> FitBuffers<P>::w_ref() const noexcept {
    if (buffer_policy == CholeskyBuffer::Retain) {
        return {w_matrix.data(), core.order_, core.order_, core.order_};
    }
    return core.k_ref();
}

template <PrecisionPolicyConcept P>
std::expected<void, GprError> QueryWorkspace<P>::ensure(std::size_t n, std::size_t m, std::size_t d,
                                                        std::size_t nested_depth) {
    if (n == 0U || m == 0U || d == 0U) {
        return std::unexpected(GprError::empty_input());
    }
    auto md = checked_mul(m, d);
    if (!md.has_value()) {
        return std::unexpected(std::move(md).error());
    }
    auto nm = checked_mul(n, m);
    if (!nm.has_value()) {
        return std::unexpected(std::move(nm).error());
    }

    if (query_xs.size() < *md) {
        query_xs.resize(*md);
    }
    if (query_x.size() < *md) {
        query_x.resize(*md);
    }
    if (query_k_star.size() < *nm) {
        query_k_star.resize(*nm);
    }
    if (query_scratch.size() < *nm) {
        query_scratch.resize(*nm);
    }
    if (query_dist.size() < *nm) {
        query_dist.resize(*nm);
    }
    if (query_kss.size() < m) {
        query_kss.resize(m);
    }
    if (query_nested.depth() < nested_depth || query_nested.rows() < n || query_nested.cols() < m) {
        query_nested.resize(nested_depth, n, m);
    }
    train_n = n;
    query_m = m;
    dims = d;
    return {};
}

template struct HessianScratch<double>;
template struct HessianScratch<float>;

template struct WorkspaceCore<DoublePrecision>;
template struct WorkspaceCore<SinglePrecision>;
template struct WorkspaceCore<MixedPrecision<PromoteStorage>>;
template struct WorkspaceCore<MixedPrecision<ReevaluateKernel>>;

template struct FitBuffers<DoublePrecision>;
template struct FitBuffers<SinglePrecision>;
template struct FitBuffers<MixedPrecision<PromoteStorage>>;
template struct FitBuffers<MixedPrecision<ReevaluateKernel>>;

template struct QueryWorkspace<DoublePrecision>;
template struct QueryWorkspace<SinglePrecision>;
template struct QueryWorkspace<MixedPrecision<PromoteStorage>>;
template struct QueryWorkspace<MixedPrecision<ReevaluateKernel>>;

} // namespace cxxgpr
