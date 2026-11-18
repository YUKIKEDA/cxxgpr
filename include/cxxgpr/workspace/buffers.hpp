// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <vector>

namespace cxxgpr
{

    /// Selects whether the inverse/gradient matrix W is kept separately or overwrites L.
    enum class CholeskyBuffer : std::uint8_t
    {
        /// Allocates a dedicated n x n buffer for W = alpha * alpha^T - A^-1 and preserves L.
        Retain,
        /// Reuses k_matrix for W during gradient evaluation and refactors L afterwards.
        Reuse,
    };

    /// Combined distance-cache and Cholesky-buffer preference preset for model fitting.
    struct FitPolicy
    {
        /// Distance cache strategy used during optimization.
        DistanceCachePolicy distance_cache = DistanceCachePolicy::Cached;
        /// Cholesky / W buffer retention strategy used during gradient and Hessian evaluations.
        CholeskyBuffer cholesky_buffer = CholeskyBuffer::Retain;

        /// Returns the default speed-preferring preset (Cached distances + Retained W buffer).
        [[nodiscard]] static constexpr FitPolicy prefer_speed() noexcept
        {
            return {.distance_cache = DistanceCachePolicy::Cached, .cholesky_buffer = CholeskyBuffer::Retain};
        }

        /// Returns the memory-preferring preset (Uncached distances + Reused Cholesky buffer).
        [[nodiscard]] static constexpr FitPolicy prefer_memory() noexcept
        {
            return {.distance_cache = DistanceCachePolicy::Uncached, .cholesky_buffer = CholeskyBuffer::Reuse};
        }

        /// Checks value equality between two FitPolicy values.
        [[nodiscard]] friend constexpr bool operator==(const FitPolicy&, const FitPolicy&) noexcept = default;
    };

    /// On-demand scratch buffers for analytical Hessian computation, empty until first use.
    template <KernelScalarConcept S> struct HessianScratch
    {
        /// Ensures storage for p matrices S_i of size n x n, matrix V (n x p), and scratch vectors.
        [[nodiscard]] std::expected<void, GprError> ensure(std::size_t n, std::size_t p);

        /// Returns true if buffers have been allocated for non-zero dimensions.
        [[nodiscard]] bool is_allocated() const noexcept { return order_ > 0U && num_params_ > 0U; }

        /// Returns a mutable n x n view of S_i for parameter index i.
        [[nodiscard]] MatrixMut<S> s_mut(std::size_t i) noexcept;

        /// Returns an immutable n x n view of S_i for parameter index i.
        [[nodiscard]] MatrixRef<S> s_ref(std::size_t i) const noexcept;

        /// Returns a mutable span over column i of V (length n).
        [[nodiscard]] std::span<S> v_col_mut(std::size_t i) noexcept;

        /// Returns an immutable span over column i of V (length n).
        [[nodiscard]] std::span<const S> v_col_ref(std::size_t i) const noexcept;

        /// Returns a mutable n x n view of the auxiliary third scratch matrix.
        [[nodiscard]] MatrixMut<S> scratch_c_mut() noexcept;

        /// Training order n.
        std::size_t order_ = 0;
        /// Total parameter count p.
        std::size_t num_params_ = 0;
        /// Contiguous storage for p matrices S_i of shape n x n (column-major).
        std::vector<S> s_storage;
        /// Contiguous storage for n x p matrix V (column-major).
        std::vector<S> v_storage;
        /// Auxiliary length-n vector for A_i * alpha.
        std::vector<S> u;
        /// Auxiliary n x n scratch matrix for second-derivative tree evaluation.
        std::vector<S> scratch_c;
    };

    /// Reusable core training buffers shared across MLL, gradient, and Hessian evaluations.
    template <PrecisionPolicyConcept P> struct WorkspaceCore
    {
        /// Storage scalar type for Cholesky factors and kernel matrices.
        using Storage = typename P::Storage;

        /// Allocates core buffers sized for n training points.
        [[nodiscard]] static std::expected<WorkspaceCore<P>, GprError> create(std::size_t n);

        /// Returns the training sample count n this workspace was allocated for.
        [[nodiscard]] std::size_t order() const noexcept { return order_; }

        /// Returns a mutable n x n column-major view of k_matrix.
        [[nodiscard]] MatrixMut<Storage> k_mut() noexcept;
        /// Returns an immutable n x n column-major view of k_matrix.
        [[nodiscard]] MatrixRef<Storage> k_ref() const noexcept;
        /// Returns a mutable n x n column-major view of exp_buf.
        [[nodiscard]] MatrixMut<Storage> exp_mut() noexcept;
        /// Returns an immutable n x n column-major view of exp_buf.
        [[nodiscard]] MatrixRef<Storage> exp_ref() const noexcept;
        /// Returns a mutable n x n column-major view of kernel_scratch (empty until ensured).
        [[nodiscard]] MatrixMut<Storage> kernel_scratch_mut() noexcept;

        /// Ensures kernel_scratch is allocated to n x n without reallocating if already sized.
        [[nodiscard]] std::expected<void, GprError> ensure_kernel_scratch();

        /// Ensures weighted holds at least n_bufs n x n buffers without reallocating when sufficient.
        [[nodiscard]] std::expected<void, GprError> ensure_weighted(std::size_t n_bufs);

        /// Training order n.
        std::size_t order_ = 0;
        /// System matrix A = K + sigma_n^2 I, then lower Cholesky factor L.
        std::vector<Storage> k_matrix;
        /// Reusable n x n scratch for kernel evaluation, Cholesky jitter backup, and solves.
        std::vector<Storage> exp_buf;
        /// Distinct n x n scratch for leaf gradients during joint gradient walks.
        std::vector<Storage> kernel_scratch;
        /// Right-hand side y then Cholesky solution alpha (length n).
        std::vector<Storage> rhs;
        /// Scratch parameter vector for rollback on failed factorizations.
        std::vector<double> theta;
        /// Per-column partial sums for deterministic OpenMP reductions (length n).
        std::vector<double> col_partials;
        /// Pre-allocated nested intermediate buffers for tree evaluation.
        NestedBuffers<Storage> nested;
        /// Pre-allocated buffers for keep_plan and weighted_grads walks.
        NestedBuffers<Storage> weighted;
        /// On-demand analytical Hessian scratch buffers (empty until first Hessian evaluation).
        HessianScratch<Storage> hessian;
        /// Diagonal jitter added during the last successful Cholesky factorization.
        double factor_jitter = 0.0;
    };

    /// Training buffers combining WorkspaceCore with optional DistanceCache and W matrix.
    template <PrecisionPolicyConcept P> struct FitBuffers
    {
        /// Storage scalar type.
        using Storage = typename P::Storage;

        /// Allocates training buffers for n points under the given cache and Cholesky buffer policies.
        [[nodiscard]] static std::expected<FitBuffers<P>, GprError> create(
            std::size_t n,
            DistanceCachePolicy cache = DistanceCachePolicy::Cached,
            CholeskyBuffer buffer = CholeskyBuffer::Retain
        );

        /// Allocates training buffers for n points using a FitPolicy preset.
        [[nodiscard]] static std::expected<FitBuffers<P>, GprError> create(std::size_t n, FitPolicy policy);

        /// Returns true if gradient evaluation overwrites k_matrix and requires refactoring L.
        [[nodiscard]] bool overwrites_cholesky() const noexcept { return buffer_policy == CholeskyBuffer::Reuse; }

        /// Returns a mutable n x n view of the W buffer (dedicated w_matrix or k_matrix).
        [[nodiscard]] MatrixMut<Storage> w_mut() noexcept;

        /// Returns an immutable n x n view of the W buffer.
        [[nodiscard]] MatrixRef<Storage> w_ref() const noexcept;

        /// Core factorization and gradient scratch buffers.
        WorkspaceCore<P> core;
        /// Optional training distance cache (present when cache_policy == Cached).
        std::optional<DistanceCache<Storage>> dist;
        /// Dedicated W = alpha * alpha^T - A^-1 buffer when buffer_policy == Retain.
        std::vector<Storage> w_matrix;
        /// Distance cache policy for this fit buffer set.
        DistanceCachePolicy cache_policy = DistanceCachePolicy::Cached;
        /// Cholesky buffer retention policy.
        CholeskyBuffer buffer_policy = CholeskyBuffer::Retain;
    };

    /// Reusable query-time prediction workspace owned by fitted models.
    template <PrecisionPolicyConcept P> struct QueryWorkspace
    {
        /// Storage scalar type.
        using Storage = typename P::Storage;

        /// Ensures buffers can accommodate n training points, m queries, d features, and depth.
        [[nodiscard]] std::expected<void, GprError> ensure(
            std::size_t n, std::size_t m, std::size_t d, std::size_t nested_depth
        );

        /// Transformed query features in double precision (m x d, column-major).
        std::vector<double> query_xs;
        /// Query features cast to Storage scalar (m x d, column-major).
        std::vector<Storage> query_x;
        /// Cross-covariance k(X, X*) then L^-1 K_* (n x m, column-major).
        std::vector<Storage> query_k_star;
        /// Scratch for cross-covariance tree evaluation (n x m, column-major).
        std::vector<Storage> query_scratch;
        /// Nested intermediate buffers for cross-covariance evaluation.
        NestedBuffers<Storage> query_nested;
        /// Train-test squared Euclidean distances (n x m, column-major).
        std::vector<Storage> query_dist;
        /// Prior diagonal k(x*_j, x*_j) for each query point (length m).
        std::vector<Storage> query_kss;
        /// Scratch for zero-allocation diagonal tree evaluation (length >= (nested_depth + 1) * m).
        std::vector<Storage> query_diag_scratch;
        /// Current training point capacity n.
        std::size_t train_n = 0;
        /// Current query point capacity m.
        std::size_t query_m = 0;
        /// Current feature dimension capacity d.
        std::size_t dims = 0;
    };

    extern template struct HessianScratch<double>;
    extern template struct HessianScratch<float>;

    extern template struct WorkspaceCore<DoublePrecision>;
    extern template struct WorkspaceCore<SinglePrecision>;
    extern template struct WorkspaceCore<MixedPrecision<PromoteStorage>>;
    extern template struct WorkspaceCore<MixedPrecision<ReevaluateKernel>>;

    extern template struct FitBuffers<DoublePrecision>;
    extern template struct FitBuffers<SinglePrecision>;
    extern template struct FitBuffers<MixedPrecision<PromoteStorage>>;
    extern template struct FitBuffers<MixedPrecision<ReevaluateKernel>>;

    extern template struct QueryWorkspace<DoublePrecision>;
    extern template struct QueryWorkspace<SinglePrecision>;
    extern template struct QueryWorkspace<MixedPrecision<PromoteStorage>>;
    extern template struct QueryWorkspace<MixedPrecision<ReevaluateKernel>>;

}  // namespace cxxgpr
