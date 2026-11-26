// SPDX-License-Identifier: Apache-2.0
/// @file leaf_cache.hpp
/// @brief Incremental per-leaf Gram cache and optimizer change-index traits.
/// @details Defines OptimizerHasChangeIndicesConcept, OptimizerTraits, require_change_indices(), and
/// LeafCache<T> for validating bit-exact parameter change masks and recomputing only dirty leaves
/// during single-coordinate optimization proposals such as FastSimulatedAnnealing.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Concept matching optimizer types that expose a static boolean uses_change_indices member.
    /// @details Enables compile-time dispatch for incremental single-coordinate objective evaluation.
    /// @tparam O Candidate optimizer type to inspect.
    template <typename O>
    concept OptimizerHasChangeIndicesConcept = requires {
        { O::uses_change_indices } -> std::convertible_to<bool>;
    };

    /// @brief Traits describing incremental coordinate-step support for an optimizer type.
    /// @details Defaults uses_change_indices to false for optimizers that update all parameters simultaneously.
    /// @tparam O Optimizer type queried by model objectives.
    template <typename O> struct OptimizerTraits
    {
        /// @brief True when the optimizer supplies changed parameter indices at each evaluation step.
        static constexpr bool uses_change_indices = false;
    };

    /// @brief Specialization for optimizers that declare static constexpr bool uses_change_indices.
    /// @details Forwards O::uses_change_indices to enable incremental leaf-Gram caching when true.
    /// @tparam O Optimizer type satisfying OptimizerHasChangeIndicesConcept.
    template <OptimizerHasChangeIndicesConcept O> struct OptimizerTraits<O>
    {
        /// @brief Mirrors O::uses_change_indices for incremental leaf cache dispatch.
        static constexpr bool uses_change_indices = static_cast<bool>(O::uses_change_indices);
    };

    /// @brief Convenience variable template reading OptimizerTraits<O>::uses_change_indices.
    /// @details Evaluates to true when optimizer O supports incremental value_at_changes() updates.
    /// @tparam O Optimizer type queried at compile time.
    template <typename O>
    inline constexpr bool optimizer_uses_change_indices_v = OptimizerTraits<O>::uses_change_indices;

    /// @brief Validates a non-empty, in-bounds, duplicate-free span of changed parameter indices.
    /// @details Rejects empty spans, indices >= n_params, and repeated indices before incremental leaf lookup.
    /// @param[in] indices Span of zero-based parameter indices claimed to have changed.
    /// @param[in] n_params Total number of objective parameters (kernel + likelihood).
    /// @return Empty expected on success, or GprError if indices are empty, out of range, or duplicated.
    /// @retval GprErrorCode::IndexOutOfRange When indices is empty, any index >= n_params, or duplicates exist.
    [[nodiscard]] std::expected<void, GprError> require_change_indices(
        std::span<const std::size_t> indices, std::size_t n_params
    );

    /// @brief Per-leaf Gram matrix cache supporting bit-exact change-index validation and incremental updates.
    /// @details Stores one n x n column-major Gram matrix per compiled kernel leaf, verifies via std::bit_cast
    /// that no parameter outside change_indices was modified since the last recorded step, recomputes only
    /// leaves owning changed kernel parameters, and combines all leaf Grams into the composite Gram matrix.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    template <KernelScalarConcept T> class LeafCache
    {
    public:
        /// @brief Constructs an empty, unprimed leaf cache.
        /// @details Allocates no heap memory until resize() or evaluate_incremental() is called.
        LeafCache() noexcept = default;

        /// @brief Sizes internal storage for n_leaves square matrices of order n; preserves capacity if unchanged.
        /// @details Checks n * n and n_leaves * n * n against std::size_t overflow, builds MatrixRef<T>
        /// views into contiguous storage, sizes ARD $s^2$ and lengthscale snapshot buffers dynamically (using exact
        /// per-leaf counts when ard_dims_per_leaf is provided), and marks all leaves dirty while resetting primed
        /// state.
        /// @param[in] n_leaves Number of kernel leaves in the flattened CompiledKernel tree.
        /// @param[in] n Training sample count (matrix order).
        /// @param[in] ard_dims_per_leaf Optional span of length n_leaves giving ARD dim counts (0 for non-ARD leaves).
        /// @return Empty expected on success, or GprError if total element count overflows or dims count mismatches.
        /// @retval GprErrorCode::SizeOverflow When buffer element counts overflow std::size_t.
        /// @retval GprErrorCode::LengthMismatch When ard_dims_per_leaf is non-empty and its size != n_leaves.
        [[nodiscard]] std::expected<void, GprError> resize(
            std::size_t n_leaves, std::size_t n, std::span<const std::size_t> ard_dims_per_leaf = {}
        );

        /// @brief Validates params against change_indices using bit-exact comparison and marks dirty leaves.
        /// @details Verifies params.size() == total_params and validates change_indices; if primed, checks
        /// every coordinate via bit_cast equality against last_params_ so undeclared changes fail cleanly,
        /// then flags only leaves owning indices in [0, kernel.num_params()).
        /// @param[in] params Candidate full log-space parameter vector (kernel parameters + noise parameter).
        /// @param[in] change_indices Non-empty duplicate-free span of parameter indices modified since last step.
        /// @param[in] kernel Compiled kernel tree mapping parameter indices to owning leaf indices.
        /// @param[in] total_params Expected full parameter count (kernel.num_params() + 1).
        /// @return Empty expected on success, or GprError on length mismatch or undeclared parameter changes.
        /// @retval GprErrorCode::LengthMismatch When params.size() != total_params.
        /// @retval GprErrorCode::IndexOutOfRange When change_indices is invalid or an unlisted parameter changed.
        [[nodiscard]] std::expected<void, GprError> prepare_step(
            std::span<const double> params,
            std::span<const std::size_t> change_indices,
            const CompiledKernel<T>& kernel,
            std::size_t total_params
        );

        /// @brief Marks every leaf as dirty so the next evaluation recomputes all leaf Gram matrices.
        /// @details Leaves primed_ state intact; use invalidate() when baseline parameter state is stale.
        void mark_all_dirty() noexcept;

        /// @brief Invalidates the primed state and marks all leaves dirty without releasing allocated storage.
        /// @details Called automatically if leaf evaluation or tree combination fails mid-step.
        void invalidate() noexcept;

        /// @brief Records params as the baseline parameter vector for subsequent bit-exact change checks.
        /// @details Copies params into last_params_ and marks the cache primed.
        /// @param[in] params Full log-space parameter vector matching the current cached leaf Grams.
        /// @return Empty expected on completion.
        [[nodiscard]] std::expected<void, GprError> record(std::span<const double> params);

        /// @brief Recomputes only dirty leaves and combines all leaf Grams into out without heap allocation.
        /// @details Sizes storage if needed, runs bit-exact prepare_step(), evaluates dirty leaves incrementally across
        /// both cached DistanceCache and uncached coordinate paths, combines leaf Grams with combine_from_leaf_grams(),
        /// and either commits params immediately (`commit_immediately == true`) or stages them as a non-destructive
        /// trial until commit_candidate() is invoked (`commit_immediately == false`).
        /// @param[in] kernel Compiled execution-layer kernel tree.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] dist_cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
        /// @param[in] params Candidate full log-space parameter vector of length total_params.
        /// @param[in] change_indices Non-empty duplicate-free span of changed parameter indices.
        /// @param[in] total_params Expected length of params (kernel parameters plus noise parameter).
        /// @param[out] out Column-major n x n destination composite Gram matrix.
        /// @param[in] uplo Triangular or full matrix region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix view for tree combination.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= kernel.nested_depth() and shape >= n x n.
        /// @param[in] commit_immediately True to commit ARD $s^2$ and baseline immediately; false for trial mode.
        /// @return Empty expected on success, or GprError if validation, leaf evaluation, or combination fails.
        /// @retval GprErrorCode::EmptyInput When x is empty or null.
        /// @retval GprErrorCode::SizeOverflow When leaf cache sizing overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When params or output/cache dimensions mismatch.
        /// @retval GprErrorCode::IndexOutOfRange When change_indices is invalid or an unlisted parameter changed.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD or cache dimensions.
        /// @retval GprErrorCode::NonFiniteInput When x contains non-finite values.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        [[nodiscard]] std::expected<void, GprError> evaluate_incremental(
            const CompiledKernel<T>& kernel,
            MatrixRef<T> x,
            const DistanceCache<T>* dist_cache,
            std::span<const double> params,
            std::span<const std::size_t> change_indices,
            std::size_t total_params,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            bool commit_immediately = true
        );

        /// @brief Primes all leaf Grams and ARD $s^2$ baselines from a full baseline evaluation at params.
        /// @details Evaluates every leaf, combines them into out, and records params as the accepted baseline so the
        /// very first incremental proposal starts with warm per-leaf Grams and ARD $s^2$ caches.
        /// @param[in] kernel Compiled execution-layer kernel tree.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] dist_cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
        /// @param[in] params Baseline full log-space parameter vector matching kernel.
        /// @param[out] out Column-major n x n destination composite Gram matrix.
        /// @param[in] uplo Triangular or full matrix region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix view for tree combination.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= kernel.nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError if validation, leaf evaluation, or combination fails.
        [[nodiscard]] std::expected<void, GprError> prime(
            const CompiledKernel<T>& kernel,
            MatrixRef<T> x,
            const DistanceCache<T>* dist_cache,
            std::span<const double> params,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested
        );

        /// @brief Commits or discards the pending non-destructive trial staged by evaluate_incremental().
        /// @details When accepted is true, commits accepted ARD $s^2$ updates in $O(n^2)$ per changed dim (or rebuilds
        /// cleanly on step saturation) and advances last_params(); when accepted is false, discards in $O(1)$ while
        /// marking only overwritten leaf Grams stale without mutating the accepted ARD $s^2$ baseline.
        /// @param[in] accepted True if the pending trial became the accepted state; false to discard it.
        /// @param[in] kernel Compiled kernel currently configured with the pending candidate parameters.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] dist_cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
        /// @return Empty expected on success or GprError if ARD baseline commit validation fails.
        [[nodiscard]] std::expected<void, GprError> commit_candidate(
            bool accepted, const CompiledKernel<T>& kernel, MatrixRef<T> x, const DistanceCache<T>* dist_cache
        );

        /// @brief Returns true when the cached leaf Grams match the recorded parameter vector.
        /// @details False before the first record() call or after invalidate() or resize().
        /// @return True if primed against last_params().
        [[nodiscard]] bool is_primed() const noexcept { return primed_; }

        /// @brief Returns true if leaf_idx is marked dirty and needs recomputation.
        /// @details Returns false when leaf_idx >= num_leaves().
        /// @param[in] leaf_idx Zero-based leaf index.
        /// @return True if dirty_[leaf_idx] is non-zero.
        [[nodiscard]] bool is_leaf_dirty(std::size_t leaf_idx) const noexcept;

        /// @brief Returns the number of cached leaves.
        /// @details Matches CompiledKernel::num_leaves() after resize() or evaluate_incremental().
        /// @return Cached leaf count.
        [[nodiscard]] std::size_t num_leaves() const noexcept { return n_leaves_; }

        /// @brief Returns the matrix order n of each cached leaf Gram matrix.
        /// @details Zero before the first resize() or evaluate_incremental() call.
        /// @return Square Gram matrix order n.
        [[nodiscard]] std::size_t order() const noexcept { return order_; }

        /// @brief Returns a mutable n x n column-major view of the cached Gram matrix for leaf_idx.
        /// @details Returns an empty view when leaf_idx >= num_leaves().
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @return Mutable n x n column-major view of leaf_idx storage.
        [[nodiscard]] MatrixMut<T> leaf_gram_mut(std::size_t leaf_idx) noexcept;

        /// @brief Returns an immutable n x n column-major view of the cached Gram matrix for leaf_idx.
        /// @details Returns an empty view when leaf_idx >= num_leaves().
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @return Immutable n x n column-major view of leaf_idx storage.
        [[nodiscard]] MatrixRef<T> leaf_gram(std::size_t leaf_idx) const noexcept;

        /// @brief Returns the baseline parameter vector recorded on the last successful step.
        /// @details Empty prior to the first successful record() call.
        /// @return Immutable span over the recorded parameter vector.
        [[nodiscard]] std::span<const double> last_params() const noexcept { return last_params_; }

    private:
        /// @brief Default per-leaf lengthscale snapshot capacity when caller uses 2-arg resize without ARD dims.
        static constexpr std::size_t k_default_ls_cap_ = 64U;
        /// @brief Sentinel indicating a leaf does not own an ARD $s^2$ cache slot.
        static constexpr std::size_t k_no_s2_slot_ = ~std::size_t{0};
        /// @brief Number of cached leaf Gram slots.
        std::size_t n_leaves_ = 0;
        /// @brief Square matrix order n for each cached leaf Gram slot.
        std::size_t order_ = 0;
        /// @brief Number of packed lower-triangular entries per ARD leaf (`order_ * (order_ + 1) / 2`).
        std::size_t ard_per_dim_ = 0;
        /// @brief Contiguous column-major storage for all n_leaves_ n x n leaf Gram matrices.
        std::vector<T> storage_;
        /// @brief Pre-built immutable MatrixRef views into each leaf slice of storage_.
        std::vector<MatrixRef<T>> gram_views_;
        /// @brief Per-leaf dirty flags (1 = needs recomputation, 0 = clean).
        std::vector<std::uint8_t> dirty_;
        /// @brief Baseline accepted log-space parameters corresponding to the accepted baseline.
        std::vector<double> last_params_;
        /// @brief Last evaluated parameter vector (committed or trial) for bit-exact change-index checks.
        std::vector<double> last_eval_params_;
        /// @brief Trial candidate parameters awaiting commit_candidate().
        std::vector<double> pending_params_;
        /// @brief Per-leaf flags indicating Gram slots touched by an uncommitted trial.
        std::vector<std::uint8_t> trial_modified_;
        /// @brief Per-leaf flags indicating Gram slots overwritten by a rejected trial.
        std::vector<std::uint8_t> stale_after_reject_;
        /// @brief Pre-allocated packed lower-triangular ARD s^2 cache sized O(n^2) per ARD leaf.
        std::vector<T> ard_s2_storage_;
        /// @brief Starting index into `ard_s2_storage_` per leaf, or `k_no_s2_slot_` for non-ARD leaves.
        std::vector<std::size_t> ard_s2_offsets_;
        /// @brief Pre-allocated lengthscale snapshots packed dynamically across leaves (`O(d)` total).
        std::vector<double> ard_prev_ls_;
        /// @brief Starting index into `ard_prev_ls_` per leaf.
        std::vector<std::size_t> ard_ls_offsets_;
        /// @brief Allocated double capacity inside `ard_prev_ls_` per leaf.
        std::vector<std::size_t> ard_ls_capacities_;
        /// @brief Per-leaf validity flags for `ard_s2_storage_` (1 = valid, 0 = rebuild required).
        std::vector<std::uint8_t> ard_s2_valid_;
        /// @brief Per-leaf counters of consecutive incremental delta updates since the last full rebuild.
        std::vector<std::size_t> ard_inc_steps_;
        /// @brief True once at least one accepted baseline has been recorded.
        bool primed_ = false;
        /// @brief True when a non-destructive trial evaluation is awaiting commit_candidate().
        bool has_pending_ = false;
    };

    extern template class LeafCache<double>;
    extern template class LeafCache<float>;

}  // namespace cxxgpr
