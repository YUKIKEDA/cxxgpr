// SPDX-License-Identifier: Apache-2.0
/// @file buffers.hpp
/// @brief Pre-allocated training and query-time workspace buffers for Exact GPR.
/// @details Declares CholeskyBuffer, FitPolicy, HessianScratch, WorkspaceCore, FitBuffers, and
/// QueryWorkspace so marginal likelihood, analytical gradient, analytical Hessian, and posterior
/// prediction routines execute with zero heap allocations inside optimization and query hot paths.
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

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Selects whether the gradient weight matrix W is kept separately or overwrites L.
    /// @details Trades O(n^2) memory for avoiding a second Cholesky factorization when evaluating
    /// negative marginal log-likelihood gradients or Hessians.
    enum class CholeskyBuffer : std::uint8_t
    {
        /// @brief Allocates a dedicated n x n buffer for W = alpha * alpha^T - A^-1 and preserves L.
        Retain,
        /// @brief Reuses k_matrix for W during gradient or Hessian computation and refactors L afterwards.
        Reuse,
    };

    /// @brief Combined distance-cache and Cholesky-buffer preference preset for model fitting.
    /// @details Bundles DistanceCachePolicy and CholeskyBuffer into speed- vs memory-oriented presets.
    struct FitPolicy
    {
        /// @brief Distance cache strategy used during optimization and fitting.
        DistanceCachePolicy distance_cache = DistanceCachePolicy::Cached;
        /// @brief Cholesky / W buffer retention strategy used during gradient and Hessian evaluations.
        CholeskyBuffer cholesky_buffer = CholeskyBuffer::Retain;

        /// @brief Returns the default speed-preferring preset (Cached distances + Retained W buffer).
        /// @details Avoids recomputing distances and refactoring Cholesky at the expense of extra n x n storage.
        /// @return FitPolicy configured with DistanceCachePolicy::Cached and CholeskyBuffer::Retain.
        [[nodiscard]] static constexpr FitPolicy prefer_speed() noexcept
        {
            return {.distance_cache = DistanceCachePolicy::Cached, .cholesky_buffer = CholeskyBuffer::Retain};
        }

        /// @brief Returns the memory-preferring preset (Uncached distances + Reused Cholesky buffer).
        /// @details Minimizes workspace memory footprint during fitting by avoiding distance caches and dedicated W
        /// storage.
        /// @return FitPolicy configured with DistanceCachePolicy::Uncached and CholeskyBuffer::Reuse.
        [[nodiscard]] static constexpr FitPolicy prefer_memory() noexcept
        {
            return {.distance_cache = DistanceCachePolicy::Uncached, .cholesky_buffer = CholeskyBuffer::Reuse};
        }

        /// @brief Checks value equality between two FitPolicy values.
        /// @details Compares both distance_cache and cholesky_buffer policies.
        /// @param[in] lhs Left-hand FitPolicy operand.
        /// @param[in] rhs Right-hand FitPolicy operand.
        /// @return True if both policy fields match.
        [[nodiscard]] friend constexpr bool operator==(const FitPolicy& lhs, const FitPolicy& rhs) noexcept = default;
    };

    /// @brief On-demand scratch buffers for analytical Hessian computation, empty until first use.
    /// @details Holds p column-major n x n matrices S_i = L^-1 A_i L^-T, the n x p matrix V with columns
    /// v_i = L^-1 A_i alpha, the auxiliary vector u = A_i alpha, and a third n x n scratch matrix for
    /// second-derivative tree evaluation.
    /// @tparam S Floating-point kernel scalar type satisfying KernelScalarConcept.
    template <KernelScalarConcept S> struct HessianScratch
    {
        /// @brief Ensures storage for p matrices S_i of size n x n, matrix V (n x p), and scratch vectors.
        /// @details Checks integer multiplication bounds before growing buffers; existing allocations are
        /// preserved when capacity already suffices.
        /// @param[in] n Training sample count (matrix order).
        /// @param[in] p Total hyperparameter count (kernel parameters plus noise parameter).
        /// @return Empty expected on success, or GprError on empty dimensions or overflow.
        /// @retval GprErrorCode::EmptyInput When n == 0 or p == 0.
        /// @retval GprErrorCode::SizeOverflow When n * n, p * n * n, or n * p overflows std::size_t.
        [[nodiscard]] std::expected<void, GprError> ensure(std::size_t n, std::size_t p);

        /// @brief Returns true if buffers have been allocated for non-zero dimensions.
        /// @details Checks order_ > 0 and num_params_ > 0 after a successful ensure() call.
        /// @return True when S, V, u, and scratch_c storage is ready for use.
        [[nodiscard]] bool is_allocated() const noexcept { return order_ > 0U && num_params_ > 0U; }

        /// @brief Returns a mutable n x n column-major view of S_i for parameter index i.
        /// @details Caller must ensure is_allocated() is true and i < num_params_.
        /// @param[in] i Zero-based parameter index in [0, num_params_).
        /// @return Mutable n x n column-major view of S_i.
        [[nodiscard]] MatrixMut<S> s_mut(std::size_t i) noexcept;

        /// @brief Returns an immutable n x n column-major view of S_i for parameter index i.
        /// @details Caller must ensure is_allocated() is true and i < num_params_.
        /// @param[in] i Zero-based parameter index in [0, num_params_).
        /// @return Immutable n x n column-major view of S_i.
        [[nodiscard]] MatrixRef<S> s_ref(std::size_t i) const noexcept;

        /// @brief Returns a mutable span over column i of V (length n).
        /// @details Stores v_i = L^-1 A_i alpha for parameter index i.
        /// @param[in] i Zero-based parameter index in [0, num_params_).
        /// @return Mutable length-n span over column i of V.
        [[nodiscard]] std::span<S> v_col_mut(std::size_t i) noexcept;

        /// @brief Returns an immutable span over column i of V (length n).
        /// @details Reads v_i = L^-1 A_i alpha populated during analytical Hessian evaluation.
        /// @param[in] i Zero-based parameter index in [0, num_params_).
        /// @return Immutable length-n span over column i of V.
        [[nodiscard]] std::span<const S> v_col_ref(std::size_t i) const noexcept;

        /// @brief Returns a mutable n x n column-major view of the auxiliary third scratch matrix.
        /// @details Used as scratch_b/scratch_c when calling CompiledKernel::hess_* or grad_*.
        /// @return Mutable n x n column-major view of scratch_c.
        [[nodiscard]] MatrixMut<S> scratch_c_mut() noexcept;

        /// @brief Training order n configured by the last successful ensure() call.
        std::size_t order_ = 0;
        /// @brief Total parameter count p configured by the last successful ensure() call.
        std::size_t num_params_ = 0;
        /// @brief Contiguous storage for p matrices S_i of shape n x n (column-major).
        std::vector<S> s_storage;
        /// @brief Contiguous storage for n x p matrix V (column-major).
        std::vector<S> v_storage;
        /// @brief Auxiliary length-n vector for A_i * alpha before triangular solve.
        std::vector<S> u;
        /// @brief Auxiliary n x n scratch matrix for second-derivative tree evaluation.
        std::vector<S> scratch_c;
    };

    /// @brief Reusable core training buffers shared across MLL, gradient, and Hessian evaluations.
    /// @details Owns k_matrix, exp_buf, lazy kernel_scratch, Cholesky solve rhs/alpha, parameter rollback
    /// buffer, deterministic column reduction partials, tree evaluation buffers, and on-demand Hessian scratch.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    template <PrecisionPolicyConcept P> struct WorkspaceCore
    {
        /// @brief Storage scalar type for Cholesky factors and kernel matrices.
        using Storage = typename P::Storage;

        /// @brief Allocates core buffers sized for n training points.
        /// @details Validates n > 0 and checks n * n against std::size_t overflow before allocating k_matrix,
        /// exp_buf, rhs, and col_partials; kernel_scratch, weighted, and hessian remain lazily allocated.
        /// @param[in] n Training sample count (matrix order).
        /// @return Allocated WorkspaceCore<P>, or GprError if n == 0 or n * n overflows.
        /// @retval GprErrorCode::EmptyInput When n == 0.
        /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
        [[nodiscard]] static std::expected<WorkspaceCore<P>, GprError> create(std::size_t n);

        /// @brief Returns the training sample count n this workspace was allocated for.
        /// @details Zero for default-constructed uninitialized instances.
        /// @return Training order n.
        [[nodiscard]] std::size_t order() const noexcept { return order_; }

        /// @brief Returns a mutable n x n column-major view of k_matrix.
        /// @details Addresses the system matrix A = K + sigma_n^2 I or its lower Cholesky factor L.
        /// @return Mutable n x n column-major view of k_matrix.
        [[nodiscard]] MatrixMut<Storage> k_mut() noexcept;
        /// @brief Returns an immutable n x n column-major view of k_matrix.
        /// @details Addresses the factored lower Cholesky triangle L after factor_and_mll().
        /// @return Immutable n x n column-major view of k_matrix.
        [[nodiscard]] MatrixRef<Storage> k_ref() const noexcept;
        /// @brief Returns a mutable n x n column-major view of exp_buf.
        /// @details Used as kernel evaluation scratch, Cholesky jitter backup, and derivative/inverse scratch.
        /// @return Mutable n x n column-major view of exp_buf.
        [[nodiscard]] MatrixMut<Storage> exp_mut() noexcept;
        /// @brief Returns an immutable n x n column-major view of exp_buf.
        /// @details Reads intermediate kernel derivative or inverse scratch populated in exp_buf.
        /// @return Immutable n x n column-major view of exp_buf.
        [[nodiscard]] MatrixRef<Storage> exp_ref() const noexcept;
        /// @brief Returns a mutable n x n column-major view of kernel_scratch (empty until ensured).
        /// @details Returns a default empty view when ensure_kernel_scratch() has not yet been called.
        /// @return Mutable n x n column-major view of kernel_scratch, or an empty view if unallocated.
        [[nodiscard]] MatrixMut<Storage> kernel_scratch_mut() noexcept;

        /// @brief Ensures kernel_scratch is allocated to n x n without reallocating if already sized.
        /// @details Lazily initializes kernel_scratch on the first gradient or Hessian call.
        /// @return Empty expected on success, or GprError if workspace order is zero or size overflows.
        /// @retval GprErrorCode::EmptyInput When order_ == 0.
        /// @retval GprErrorCode::SizeOverflow When order_ * order_ overflows std::size_t.
        [[nodiscard]] std::expected<void, GprError> ensure_kernel_scratch();

        /// @brief Ensures weighted holds at least n_bufs n x n buffers without reallocating when sufficient.
        /// @details Sizes NestedBuffers for keep_plan factor-Gram retention and single-pass weighted_grads().
        /// @param[in] n_bufs Minimum number of n x n buffer levels required in weighted.
        /// @return Empty expected on success, or GprError if workspace order is zero.
        /// @retval GprErrorCode::EmptyInput When order_ == 0.
        [[nodiscard]] std::expected<void, GprError> ensure_weighted(std::size_t n_bufs);

        /// @brief Training order n.
        std::size_t order_ = 0;
        /// @brief System matrix A = K + sigma_n^2 I, then lower Cholesky factor L.
        std::vector<Storage> k_matrix;
        /// @brief Reusable n x n scratch for kernel evaluation, Cholesky jitter backup, and solves.
        std::vector<Storage> exp_buf;
        /// @brief Distinct n x n scratch for leaf gradients during joint gradient and Hessian walks.
        std::vector<Storage> kernel_scratch;
        /// @brief Right-hand side y then Cholesky solution alpha (length n).
        std::vector<Storage> rhs;
        /// @brief Scratch parameter vector for rollback on failed factorizations.
        std::vector<double> theta;
        /// @brief Per-column partial sums for deterministic OpenMP reductions (length 3 * n).
        std::vector<double> col_partials;
        /// @brief Pre-allocated nested intermediate buffers for tree evaluation.
        NestedBuffers<Storage> nested;
        /// @brief Pre-allocated buffers for keep_plan and weighted_grads walks.
        NestedBuffers<Storage> weighted;
        /// @brief On-demand analytical Hessian scratch buffers (empty until first Hessian evaluation).
        HessianScratch<Storage> hessian;
        /// @brief Diagonal jitter added during the last successful Cholesky factorization.
        double factor_jitter = 0.0;
    };

    /// @brief Training buffers combining WorkspaceCore with optional DistanceCache and W matrix.
    /// @details Orchestrates memory/speed tradeoffs across distance caching and Cholesky factor retention.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    template <PrecisionPolicyConcept P> struct FitBuffers
    {
        /// @brief Storage scalar type.
        using Storage = typename P::Storage;

        /// @brief Allocates training buffers for n points under the given cache and Cholesky buffer policies.
        /// @details Allocates WorkspaceCore<P>, constructs an empty DistanceCache if cache == Cached, and
        /// allocates w_matrix of size n x n if buffer == Retain.
        /// @param[in] n Training sample count.
        /// @param[in] cache Distance caching policy (Cached or Uncached).
        /// @param[in] buffer Cholesky buffer retention policy (Retain or Reuse).
        /// @return Allocated FitBuffers<P>, or GprError if n == 0 or size calculation overflows.
        /// @retval GprErrorCode::EmptyInput When n == 0.
        /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
        [[nodiscard]] static std::expected<FitBuffers<P>, GprError> create(
            std::size_t n,
            DistanceCachePolicy cache = DistanceCachePolicy::Cached,
            CholeskyBuffer buffer = CholeskyBuffer::Retain
        );

        /// @brief Allocates training buffers for n points using a FitPolicy preset.
        /// @details Convenience overload forwarding policy.distance_cache and policy.cholesky_buffer.
        /// @param[in] n Training sample count.
        /// @param[in] policy Combined distance-cache and Cholesky-buffer preset.
        /// @return Allocated FitBuffers<P>, or GprError if n == 0 or size calculation overflows.
        /// @retval GprErrorCode::EmptyInput When n == 0.
        /// @retval GprErrorCode::SizeOverflow When n * n overflows std::size_t.
        [[nodiscard]] static std::expected<FitBuffers<P>, GprError> create(std::size_t n, FitPolicy policy);

        /// @brief Returns true if gradient or Hessian evaluation overwrites k_matrix and requires refactoring L.
        /// @details True exclusively when buffer_policy == CholeskyBuffer::Reuse.
        /// @return True when w_mut() aliases core.k_matrix.
        [[nodiscard]] bool overwrites_cholesky() const noexcept { return buffer_policy == CholeskyBuffer::Reuse; }

        /// @brief Returns a mutable n x n column-major view of the W buffer (dedicated w_matrix or k_matrix).
        /// @details Points to w_matrix when buffer_policy == Retain, or core.k_mut() when Reuse.
        /// @return Mutable n x n column-major view for storing W = alpha * alpha^T - A^-1.
        [[nodiscard]] MatrixMut<Storage> w_mut() noexcept;

        /// @brief Returns an immutable n x n column-major view of the W buffer.
        /// @details Points to w_matrix when buffer_policy == Retain, or core.k_ref() when Reuse.
        /// @return Immutable n x n column-major view of W.
        [[nodiscard]] MatrixRef<Storage> w_ref() const noexcept;

        /// @brief Core factorization and gradient scratch buffers.
        WorkspaceCore<P> core;
        /// @brief Optional training distance cache (present when cache_policy == Cached).
        std::optional<DistanceCache<Storage>> dist;
        /// @brief Dedicated W = alpha * alpha^T - A^-1 buffer when buffer_policy == Retain.
        std::vector<Storage> w_matrix;
        /// @brief Distance cache policy for this fit buffer set.
        DistanceCachePolicy cache_policy = DistanceCachePolicy::Cached;
        /// @brief Cholesky buffer retention policy.
        CholeskyBuffer buffer_policy = CholeskyBuffer::Retain;
    };

    /// @brief Reusable query-time prediction workspace owned by fitted models.
    /// @details Grows monotonically to hold query features, cross-covariance K_*, triangular solve V,
    /// prior diagonals k_**, and nested kernel scratch so repeated predictions allocate nothing once warmed.
    /// @tparam P Precision policy type satisfying PrecisionPolicyConcept.
    template <PrecisionPolicyConcept P> struct QueryWorkspace
    {
        /// @brief Storage scalar type.
        using Storage = typename P::Storage;

        /// @brief Ensures buffers can accommodate n training points, m queries, d features, and depth.
        /// @details Checks dimension multiplication bounds first and resizes vectors only when current sizes
        /// are smaller than required, preserving existing heap capacity across matching or smaller queries.
        /// @param[in] n Training sample count.
        /// @param[in] m Query point count.
        /// @param[in] d Feature dimension count.
        /// @param[in] nested_depth Kernel tree nested buffer depth.
        /// @return Empty expected on success, or GprError on empty extents or size overflow.
        /// @retval GprErrorCode::EmptyInput When n == 0, m == 0, or d == 0.
        /// @retval GprErrorCode::SizeOverflow When m * d, n * m, or (nested_depth + 1) * m overflows std::size_t.
        [[nodiscard]] std::expected<void, GprError> ensure(
            std::size_t n, std::size_t m, std::size_t d, std::size_t nested_depth
        );

        /// @brief Transformed query features in double precision (m x d, column-major).
        std::vector<double> query_xs;
        /// @brief Query features cast to Storage scalar (m x d, column-major).
        std::vector<Storage> query_x;
        /// @brief Cross-covariance k(X, X*) then L^-1 K_* (n x m, column-major).
        std::vector<Storage> query_k_star;
        /// @brief Scratch for cross-covariance tree evaluation (n x m, column-major).
        std::vector<Storage> query_scratch;
        /// @brief Nested intermediate buffers for cross-covariance evaluation.
        NestedBuffers<Storage> query_nested;
        /// @brief Prior diagonal k(x*_j, x*_j) for each query point (length m).
        std::vector<Storage> query_kss;
        /// @brief Scratch for zero-allocation diagonal tree evaluation (length >= (nested_depth + 1) * m).
        std::vector<Storage> query_diag_scratch;
        /// @brief Current training point capacity n.
        std::size_t train_n = 0;
        /// @brief Current query point capacity m.
        std::size_t query_m = 0;
        /// @brief Current feature dimension capacity d.
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
