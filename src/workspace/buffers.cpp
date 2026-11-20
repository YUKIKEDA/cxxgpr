// SPDX-License-Identifier: Apache-2.0
/// @file buffers.cpp
/// @brief Implementations and explicit instantiations for Exact GPR workspace buffers.
/// @details Implements checked capacity sizing, view accessors, and lazy buffer initialization for
/// HessianScratch, WorkspaceCore, FitBuffers, and QueryWorkspace across all precision policies.

#include "cxxgpr/workspace/buffers.hpp"

#include "cxxgpr/core/checked.hpp"

#include <utility>

namespace cxxgpr
{

    /// @brief Ensures storage for p matrices S_i of size n x n, matrix V (n x p), and scratch vectors.
    /// @details Guards n * n, p * n * n, and n * p against std::size_t overflow before monotonically
    /// growing vectors so existing capacity is retained across repeated Hessian evaluations.
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] n Training sample count (matrix order).
    /// @param[in] p Total hyperparameter count (kernel parameters plus noise parameter).
    /// @return Empty expected on success, or GprError on empty dimensions or size overflow.
    /// @retval GprErrorCode::EmptyInput When n == 0 or p == 0.
    /// @retval GprErrorCode::SizeOverflow When n * n, p * n * n, or n * p overflows std::size_t.
    template <KernelScalarConcept S>
    std::expected<void, GprError> HessianScratch<S>::ensure(std::size_t n, std::size_t p)
    {
        // Reject degenerate zero dimensions before checking buffer products.
        if (n == 0U || p == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        // Check n^2, p * n^2, and n * p against std::size_t overflow before resizing.
        auto n2 = checked_mul(n, n);
        if (!n2.has_value())
        {
            return std::unexpected(std::move(n2).error());
        }
        auto pn2 = checked_mul(p, *n2);
        if (!pn2.has_value())
        {
            return std::unexpected(std::move(pn2).error());
        }
        auto np = checked_mul(n, p);
        if (!np.has_value())
        {
            return std::unexpected(std::move(np).error());
        }

        // Grow buffers monotonically so matching or smaller Hessian calls avoid reallocations.
        if (s_storage.size() < *pn2)
        {
            s_storage.resize(*pn2);
        }
        if (v_storage.size() < *np)
        {
            v_storage.resize(*np);
        }
        if (u.size() < n)
        {
            u.resize(n);
        }
        if (scratch_c.size() < *n2)
        {
            scratch_c.resize(*n2);
        }
        order_ = n;
        num_params_ = p;
        return {};
    }

    /// @brief Returns a mutable n x n column-major view of S_i for parameter index i.
    /// @details Indexes contiguous p * n * n storage at offset i * (order_ * order_).
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] i Zero-based parameter index in [0, num_params_).
    /// @return Mutable n x n column-major view of S_i.
    template <KernelScalarConcept S> MatrixMut<S> HessianScratch<S>::s_mut(std::size_t i) noexcept
    {
        const std::size_t n2 = order_ * order_;
        return {s_storage.data() + (i * n2), order_, order_, order_};
    }

    /// @brief Returns an immutable n x n column-major view of S_i for parameter index i.
    /// @details Indexes contiguous p * n * n storage at offset i * (order_ * order_).
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] i Zero-based parameter index in [0, num_params_).
    /// @return Immutable n x n column-major view of S_i.
    template <KernelScalarConcept S> MatrixRef<S> HessianScratch<S>::s_ref(std::size_t i) const noexcept
    {
        const std::size_t n2 = order_ * order_;
        return {s_storage.data() + (i * n2), order_, order_, order_};
    }

    /// @brief Returns a mutable span over column i of V (length n).
    /// @details Indexes contiguous n x p storage at offset i * order_.
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] i Zero-based parameter index in [0, num_params_).
    /// @return Mutable length-n span over column i of V.
    template <KernelScalarConcept S> std::span<S> HessianScratch<S>::v_col_mut(std::size_t i) noexcept
    {
        return {v_storage.data() + (i * order_), order_};
    }

    /// @brief Returns an immutable span over column i of V (length n).
    /// @details Indexes contiguous n x p storage at offset i * order_.
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] i Zero-based parameter index in [0, num_params_).
    /// @return Immutable length-n span over column i of V.
    template <KernelScalarConcept S> std::span<const S> HessianScratch<S>::v_col_ref(std::size_t i) const noexcept
    {
        return {v_storage.data() + (i * order_), order_};
    }

    /// @brief Returns a mutable n x n column-major view of the auxiliary third scratch matrix.
    /// @details Wraps scratch_c storage with leading dimension equal to order_.
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @return Mutable n x n column-major view of scratch_c.
    template <KernelScalarConcept S> MatrixMut<S> HessianScratch<S>::scratch_c_mut() noexcept
    {
        return {scratch_c.data(), order_, order_, order_};
    }

    /// @brief Allocates core buffers sized for n training points.
    /// @details Validates n > 0 and checks n * n overflow before zero-initializing k_matrix, exp_buf,
    /// rhs, and col_partials.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] n Training sample count (matrix order).
    /// @return Allocated WorkspaceCore<P>, or GprError if n == 0 or n * n overflows.
    /// @retval GprErrorCode::EmptyInput When n == 0.
    /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
    template <PrecisionPolicyConcept P>
    std::expected<WorkspaceCore<P>, GprError> WorkspaceCore<P>::create(std::size_t n)
    {
        // Reject zero-sized workspaces and guard n * n against integer overflow.
        if (n == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        auto n2 = checked_mul(n, n);
        if (!n2.has_value())
        {
            return std::unexpected(std::move(n2).error());
        }

        auto cp_len = checked_mul(n, std::size_t{3});
        if (!cp_len.has_value())
        {
            return std::unexpected(std::move(cp_len).error());
        }

        // Pre-allocate primary n x n and length-n/3*n reduction buffers; derivative scratch remains lazy.
        WorkspaceCore<P> core;
        core.order_ = n;
        core.k_matrix.assign(*n2, Storage{0});
        core.exp_buf.assign(*n2, Storage{0});
        core.rhs.assign(n, Storage{0});
        core.col_partials.assign(*cp_len, 0.0);
        return core;
    }

    /// @brief Returns a mutable n x n column-major view of k_matrix.
    /// @details Wraps k_matrix with leading dimension equal to order_.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Mutable n x n column-major view of k_matrix.
    template <PrecisionPolicyConcept P> MatrixMut<typename P::Storage> WorkspaceCore<P>::k_mut() noexcept
    {
        return {k_matrix.data(), order_, order_, order_};
    }

    /// @brief Returns an immutable n x n column-major view of k_matrix.
    /// @details Wraps k_matrix with leading dimension equal to order_.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Immutable n x n column-major view of k_matrix.
    template <PrecisionPolicyConcept P> MatrixRef<typename P::Storage> WorkspaceCore<P>::k_ref() const noexcept
    {
        return {k_matrix.data(), order_, order_, order_};
    }

    /// @brief Returns a mutable n x n column-major view of exp_buf.
    /// @details Wraps exp_buf with leading dimension equal to order_.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Mutable n x n column-major view of exp_buf.
    template <PrecisionPolicyConcept P> MatrixMut<typename P::Storage> WorkspaceCore<P>::exp_mut() noexcept
    {
        return {exp_buf.data(), order_, order_, order_};
    }

    /// @brief Returns an immutable n x n column-major view of exp_buf.
    /// @details Wraps exp_buf with leading dimension equal to order_.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Immutable n x n column-major view of exp_buf.
    template <PrecisionPolicyConcept P> MatrixRef<typename P::Storage> WorkspaceCore<P>::exp_ref() const noexcept
    {
        return {exp_buf.data(), order_, order_, order_};
    }

    /// @brief Returns a mutable n x n column-major view of kernel_scratch (empty until ensured).
    /// @details Returns a default empty MatrixMut when kernel_scratch has not been allocated yet.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Mutable n x n column-major view of kernel_scratch, or an empty view.
    template <PrecisionPolicyConcept P> MatrixMut<typename P::Storage> WorkspaceCore<P>::kernel_scratch_mut() noexcept
    {
        if (kernel_scratch.empty())
        {
            return {};
        }
        return {kernel_scratch.data(), order_, order_, order_};
    }

    /// @brief Ensures kernel_scratch is allocated to n x n without reallocating if already sized.
    /// @details Allocates order_ * order_ elements lazily when kernel_scratch is undersized.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Empty expected on success, or GprError if order_ == 0 or size overflows.
    /// @retval GprErrorCode::EmptyInput When order_ == 0.
    /// @retval GprErrorCode::SizeOverflow When order_ * order_ overflows std::size_t.
    template <PrecisionPolicyConcept P> std::expected<void, GprError> WorkspaceCore<P>::ensure_kernel_scratch()
    {
        if (order_ == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        auto n2 = checked_mul(order_, order_);
        if (!n2.has_value())
        {
            return std::unexpected(std::move(n2).error());
        }
        // Only allocate when current vector size is smaller than n^2.
        if (kernel_scratch.size() < *n2)
        {
            kernel_scratch.assign(*n2, Storage{0});
        }
        return {};
    }

    /// @brief Ensures weighted holds at least n_bufs n x n buffers without reallocating when sufficient.
    /// @details Grows NestedBuffers only if depth, rows, or cols fall below the requested shape.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] n_bufs Minimum number of n x n buffer levels required in weighted.
    /// @return Empty expected on success, or GprError if order_ == 0.
    /// @retval GprErrorCode::EmptyInput When order_ == 0.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> WorkspaceCore<P>::ensure_weighted(std::size_t n_bufs)
    {
        if (order_ == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        // Preserve existing NestedBuffers storage whenever depth and matrix dims already suffice.
        if (weighted.depth() < n_bufs || weighted.rows() < order_ || weighted.cols() < order_)
        {
            weighted.resize(n_bufs, order_, order_);
        }
        return {};
    }

    /// @brief Allocates training buffers for n points under the given cache and Cholesky buffer policies.
    /// @details Creates WorkspaceCore<P>, optionally constructs DistanceCache, and allocates w_matrix
    /// only when buffer == CholeskyBuffer::Retain.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] n Training sample count.
    /// @param[in] cache Distance caching policy (Cached or Uncached).
    /// @param[in] buffer Cholesky buffer retention policy (Retain or Reuse).
    /// @return Allocated FitBuffers<P>, or GprError if n == 0 or size calculation overflows.
    /// @retval GprErrorCode::EmptyInput When n == 0.
    /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
    template <PrecisionPolicyConcept P>
    std::expected<FitBuffers<P>, GprError> FitBuffers<P>::create(
        std::size_t n, DistanceCachePolicy cache, CholeskyBuffer buffer
    )
    {
        auto core = WorkspaceCore<P>::create(n);
        if (!core.has_value())
        {
            return std::unexpected(std::move(core).error());
        }

        FitBuffers<P> out;
        out.core = std::move(*core);
        out.cache_policy = cache;
        out.buffer_policy = buffer;
        // Construct the optional DistanceCache lazily; actual distance matrices are filled on first fit.
        if (cache == DistanceCachePolicy::Cached)
        {
            out.dist.emplace();
        }
        // Allocate separate W matrix only under Retain; Reuse aliases core.k_matrix instead.
        if (buffer == CholeskyBuffer::Retain)
        {
            out.w_matrix.assign(out.core.k_matrix.size(), Storage{0});
        }
        return out;
    }

    /// @brief Allocates training buffers for n points using a FitPolicy preset.
    /// @details Delegates to the policy-pair overload of FitBuffers<P>::create().
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] n Training sample count.
    /// @param[in] policy Combined distance-cache and Cholesky-buffer preset.
    /// @return Allocated FitBuffers<P>, or GprError if n == 0 or size calculation overflows.
    /// @retval GprErrorCode::EmptyInput When n == 0.
    /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
    template <PrecisionPolicyConcept P>
    std::expected<FitBuffers<P>, GprError> FitBuffers<P>::create(std::size_t n, FitPolicy policy)
    {
        return create(n, policy.distance_cache, policy.cholesky_buffer);
    }

    /// @brief Returns a mutable n x n column-major view of the W buffer (dedicated w_matrix or k_matrix).
    /// @details Returns w_matrix under CholeskyBuffer::Retain and aliases core.k_mut() under Reuse.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Mutable n x n column-major view of W.
    template <PrecisionPolicyConcept P> MatrixMut<typename P::Storage> FitBuffers<P>::w_mut() noexcept
    {
        if (buffer_policy == CholeskyBuffer::Retain)
        {
            return {w_matrix.data(), core.order_, core.order_, core.order_};
        }
        return core.k_mut();
    }

    /// @brief Returns an immutable n x n column-major view of the W buffer.
    /// @details Returns w_matrix under CholeskyBuffer::Retain and aliases core.k_ref() under Reuse.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @return Immutable n x n column-major view of W.
    template <PrecisionPolicyConcept P> MatrixRef<typename P::Storage> FitBuffers<P>::w_ref() const noexcept
    {
        if (buffer_policy == CholeskyBuffer::Retain)
        {
            return {w_matrix.data(), core.order_, core.order_, core.order_};
        }
        return core.k_ref();
    }

    /// @brief Ensures buffers can accommodate n training points, m queries, d features, and depth.
    /// @details Checks overflow bounds on m * d, n * m, and (nested_depth + 1) * m, then grows each
    /// workspace vector only if its current size is insufficient so warmed queries perform zero allocations.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    /// @param[in] n Training sample count.
    /// @param[in] m Query point count.
    /// @param[in] d Feature dimension count.
    /// @param[in] nested_depth Kernel tree nested buffer depth.
    /// @return Empty expected on success, or GprError on empty extents or size overflow.
    /// @retval GprErrorCode::EmptyInput When n == 0, m == 0, or d == 0.
    /// @retval GprErrorCode::SizeOverflow When m * d, n * m, or (nested_depth + 1) * m overflows std::size_t.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> QueryWorkspace<P>::ensure(
        std::size_t n, std::size_t m, std::size_t d, std::size_t nested_depth
    )
    {
        if (auto result = ensure_prediction(n, m, d, nested_depth); !result)
        {
            return result;
        }
        if (query_dist.size() < n * m)
        {
            query_dist.resize(n * m);
        }
        return {};
    }

    /// @brief Grows the buffers used by point-based prediction, leaving query_dist untouched.
    /// @tparam P Training precision policy.
    /// @param[in] n Training point count.
    /// @param[in] m Query point count.
    /// @param[in] d Feature count.
    /// @param[in] nested_depth Kernel nesting depth.
    /// @return Success or a dimension error.
    template <PrecisionPolicyConcept P>
    std::expected<void, GprError> QueryWorkspace<P>::ensure_prediction(
        std::size_t n, std::size_t m, std::size_t d, std::size_t nested_depth
    )
    {
        if (n == 0U || m == 0U || d == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        // Validate all query buffer element counts before growing any workspace storage.
        auto md = checked_mul(m, d);
        if (!md.has_value())
        {
            return std::unexpected(std::move(md).error());
        }
        auto nm = checked_mul(n, m);
        if (!nm.has_value())
        {
            return std::unexpected(std::move(nm).error());
        }
        auto diag_scratch_len = checked_mul(nested_depth + 1U, m);
        if (!diag_scratch_len.has_value())
        {
            return std::unexpected(std::move(diag_scratch_len).error());
        }

        // Grow monotonically so subsequent queries of equal or smaller size avoid heap allocations.
        if (query_xs.size() < *md)
        {
            query_xs.resize(*md);
        }
        if (query_x.size() < *md)
        {
            query_x.resize(*md);
        }
        if (query_k_star.size() < *nm)
        {
            query_k_star.resize(*nm);
        }
        if (query_scratch.size() < *nm)
        {
            query_scratch.resize(*nm);
        }
        if (query_kss.size() < m)
        {
            query_kss.resize(m);
        }
        if (query_diag_scratch.size() < *diag_scratch_len)
        {
            query_diag_scratch.resize(*diag_scratch_len);
        }
        if (query_nested.depth() < nested_depth || query_nested.rows() < n || query_nested.cols() < m)
        {
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

}  // namespace cxxgpr
