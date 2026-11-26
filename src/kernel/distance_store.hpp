// SPDX-License-Identifier: Apache-2.0
/// @file distance_store.hpp
/// @brief Internal staged validation of supplied-distance tables and reusable stores for the model engines.
/// @details The public prepare_square() and prepare_cross() validate a supply in one call and return a fresh store.
/// The model engines need the same checks split into phases, so that the shape checks of every input precede the
/// value checks of any table, and need to refill one store per fitted model without allocating after warm-up.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/distance.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <vector>

/// @brief Internal implementation details of cxxgpr.
namespace cxxgpr::detail
{

    /// @brief Common shape of the tables of a supply.
    struct DistanceTableShape
    {
        /// @brief Rows of every table.
        std::size_t rows{0};
        /// @brief Columns of every table.
        std::size_t cols{0};
    };

    /// @brief Checks the slots, emptiness, size products and shapes of a supply without reading table values.
    /// @details Order: InvalidConfig (tidy tolerance), UnsupportedKernelOperation (foreign slot), DistanceSlot
    /// (wrong kind, duplicate, missing), UnsupportedKernelOperation (kernel without slots), EmptyInput (empty table),
    /// SizeOverflow, LengthMismatch (non-square training table or tables of different shapes). Allocates nothing.
    /// @tparam T Storage scalar type of the compiled kernel.
    /// @param[in] kernel Compiled kernel defining the expected slots.
    /// @param[in] source Unchecked supply.
    /// @param[in] is_square True for a training supply, false for a cross supply.
    /// @return Common table shape, or the first violated GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<DistanceTableShape, GprError> check_distance_structure(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, bool is_square
    );

    /// @brief Checks that every ARD entry supplies the number of per-dimension tables its slot declares.
    /// @tparam T Storage scalar type of the compiled kernel.
    /// @param[in] kernel Compiled kernel defining the expected slots.
    /// @param[in] source Supply accepted by check_distance_structure().
    /// @return Empty expected, or DimensionMismatch.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> check_distance_ard_counts(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source
    );

    /// @brief Checks the column count of a point matrix against the coordinate leaves of a kernel.
    /// @tparam T Storage scalar type of the compiled kernel.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] cols Number of point columns.
    /// @return Empty expected, or DimensionMismatch when an ARD coordinate leaf needs another dimension.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> check_distance_point_cols(
        const CompiledDistanceKernel<T>& kernel, std::size_t cols
    );

    /// @brief Validates the table values of a supply into a reusable store.
    /// @details Requires that check_distance_structure() returned @p shape and check_distance_ard_counts()
    /// accepted @p source. The first call creates the store; later calls refill it in place and allocate only when
    /// the tables grow. When @p allow_borrow is true and the bound input scalar matches @c T without @c tidy(),
    /// scalar blocks (and cross ARD blocks) borrow caller views directly without copying. After a failure the store
    /// holds partial data and must be refilled before use.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply to validate.
    /// @param[in] shape Shape returned by check_distance_structure().
    /// @param[in] is_square True for a training-shaped (square) store, false for a cross store.
    /// @param[in,out] store Reusable store.
    /// @param[in] allow_borrow True when zero-copy borrowing of matching caller views is permitted.
    /// @return Pointer to the refilled store (never null), or InvalidDistance for the first invalid entry.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<const CheckedDistanceStore<T>*, GprError> fill_distance_store(
        const CompiledDistanceKernel<T>& kernel,
        const DistanceSource& source,
        DistanceTableShape shape,
        bool is_square,
        std::optional<CheckedDistanceStore<T>>& store,
        bool allow_borrow = false
    );

    /// @brief Validates the table values of a supply into a new store.
    /// @details Requires that check_distance_structure() returned @p shape and check_distance_ard_counts()
    /// accepted @p source; equivalent to the value phase of prepare_square() or prepare_cross().
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply to validate.
    /// @param[in] shape Shape returned by check_distance_structure().
    /// @param[in] is_square True for a training-shaped (square) store, false for a cross store.
    /// @param[in] allow_borrow True when zero-copy borrowing of matching caller views is permitted for transient use.
    /// @return New store, or InvalidDistance for the first invalid entry.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<CheckedDistanceStore<T>, GprError> make_distance_store(
        const CompiledDistanceKernel<T>& kernel,
        const DistanceSource& source,
        DistanceTableShape shape,
        bool is_square,
        bool allow_borrow = false
    );

    /// @brief Validates that a caller-supplied pre-validated @c CheckedDistanceStore matches @p kernel and shape mode.
    /// @details Returns EmptyInput for empty or moved-from stores and UnsupportedKernelOperation on slot/mode mismatch.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel defining the expected slots.
    /// @param[in] store Pre-validated store supplied by caller.
    /// @param[in] is_square True for square training stores, false for cross stores.
    /// @return Store dimensions on success, or GprError on empty or incompatible store.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<DistanceTableShape, GprError> validate_store_for_kernel(
        const CompiledDistanceKernel<T>& kernel, const CheckedDistanceStore<T>& store, bool is_square
    );

    /// @brief Widens a pre-validated @c float store into a @c double store for mixed-precision residual refinement.
    /// @details Preserves packed lower-triangular layout where allowed by @p kernel64; expands to dense if a
    /// @c CustomKernel leaf requires a full matrix view.
    /// @param[in] kernel64 Double-precision compiled kernel matching the store's slots.
    /// @param[in] store32 Validated single-precision training store.
    /// @return Double-precision store with identical validated entries, or GprError on empty/allocation error.
    [[nodiscard]] std::expected<CheckedDistanceStore<double>, GprError> promote_store_to_double(
        const CompiledDistanceKernel<double>& kernel64, const CheckedDistanceStore<float>& store32
    );

    /// @brief Creates a shared handle referencing @p store's immutable validated state without copying tables.
    /// @tparam T Storage scalar type.
    /// @param[in] store Pre-validated store to share internally.
    /// @return Shared handle referencing @p store's internal state, or null when @p store is moved-from.
    template <KernelScalarConcept T>
    [[nodiscard]] std::shared_ptr<const CheckedDistanceStore<T>> share_distance_store(
        const CheckedDistanceStore<T>& store
    );

    /// @brief Evaluates the kernel diagonal with a caller-owned, grow-only scratch vector.
    /// @details Same contract as CompiledDistanceKernel::fill_diag(); repeated calls of the same or a smaller
    /// @p n reuse @p scratch and allocate nothing.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] n Number of diagonal entries.
    /// @param[in] x Optional point matrix of @p n rows.
    /// @param[out] out Destination of length @p n.
    /// @param[in,out] scratch Grow-only scratch storage.
    /// @return Empty expected, or GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> fill_distance_diag(
        const CompiledDistanceKernel<T>& kernel,
        std::size_t n,
        std::optional<MatrixRef<T>> x,
        std::span<T> out,
        std::vector<T>& scratch
    );

    /// @brief Returns the number of flattened kernel leaves in @p kernel.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @return Total compiled leaf count.
    template <KernelScalarConcept T>
    [[nodiscard]] std::size_t distance_num_leaves(const CompiledDistanceKernel<T>& kernel) noexcept;

    /// @brief Returns the number of ARD kernel leaves in @p kernel.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @return ARD leaf count (counting both supplied-ARD and coordinate ARD leaves).
    template <KernelScalarConcept T>
    [[nodiscard]] std::size_t distance_ard_leaf_count(const CompiledDistanceKernel<T>& kernel) noexcept;

    /// @brief Returns per-leaf ARD dimension counts (`0` for non-ARD leaves).
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @return Immutable span of length `distance_num_leaves(kernel)`.
    template <KernelScalarConcept T>
    [[nodiscard]] std::span<const std::size_t> distance_ard_dims_per_leaf(
        const CompiledDistanceKernel<T>& kernel
    ) noexcept;

    /// @brief Returns the peak `NestedBuffers` depth required by `distance_weighted_grads`.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @return Required number of n x n weight-accumulation buffers for product subtrees.
    template <KernelScalarConcept T>
    [[nodiscard]] std::size_t distance_weighted_depth(const CompiledDistanceKernel<T>& kernel) noexcept;

    /// @brief Maps a global kernel hyperparameter index to its owning leaf index.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @param[in] param_idx Global hyperparameter index in `[0, kernel.num_params())`.
    /// @return Zero-based leaf index, or `IndexOutOfRange`.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<std::size_t, GprError> distance_leaf_index_for_param(
        const CompiledDistanceKernel<T>& kernel, std::size_t param_idx
    );

    /// @brief Evaluates @p kernel over @p store and @p x while capturing packed lower-triangular ARD $s^2$ buffers.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @param[in] store Pre-validated square supplied-distance store.
    /// @param[in] x Optional coordinate matrix for `WithPoints` trees.
    /// @param[out] out Destination square Gram matrix `(n, n)`.
    /// @param[in] uplo Triangle to populate.
    /// @param[in,out] scratch Non-aliasing n x n scratch matrix.
    /// @param[in,out] nested Pre-allocated intermediate buffer pool.
    /// @param[out] ard_s2_buf Pre-allocated packed $s^2$ storage sized for `distance_ard_leaf_count(kernel)` leaves.
    /// @param[in,out] ard_s2_ready Readiness flags cleared and populated with `1` for captured ARD leaves.
    /// @return Empty expected on success or GprError on validation/workspace failure.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> distance_apply_capturing_s2(
        const CompiledDistanceKernel<T>& kernel,
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested,
        std::span<T> ard_s2_buf,
        std::span<std::uint8_t> ard_s2_ready
    );

    /// @brief Contracts `<weight_lower, dK/dtheta_p>_F` across all parameters in a single adjoint tree walk.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @param[in] store Pre-validated square supplied-distance store.
    /// @param[in] x Optional coordinate matrix for `WithPoints` trees.
    /// @param[in] weight_lower Lower-triangular symmetric adjoint matrix $W$ (`n x n`).
    /// @param[out] out_grads Destination span of length `kernel.num_params()` receiving Frobenius products.
    /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix.
    /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix.
    /// @param[in,out] nested Pre-allocated intermediate buffer pool (`>= kernel.nested_depth()`).
    /// @param[in,out] work_bufs Pre-allocated product adjoint buffer pool (`>= distance_weighted_depth(kernel)`).
    /// @param[in,out] col_partials Per-column partial sum scratch of length `>= n`.
    /// @param[in] ard_s2_buf Optional forward-captured ARD $s^2$ storage.
    /// @param[in] ard_s2_ready Optional readiness bytes indicating valid entries inside `ard_s2_buf`.
    /// @return Empty expected on success or GprError on invalid arguments/buffers.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> distance_weighted_grads(
        const CompiledDistanceKernel<T>& kernel,
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixRef<T> weight_lower,
        std::span<double> out_grads,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        NestedBuffers<T>& nested,
        NestedBuffers<T>& work_bufs,
        std::span<double> col_partials,
        std::span<const T> ard_s2_buf = {},
        std::span<const std::uint8_t> ard_s2_ready = {}
    );

    /// @brief Evaluates one distance-kernel leaf incrementally with optional non-destructive ARD $s^2$ trials.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, distance_num_leaves(kernel))`.
    /// @param[in] store Pre-validated square supplied-distance store.
    /// @param[in] x Optional coordinate matrix for `WithPoints` trees.
    /// @param[out] out Destination leaf Gram matrix `(n, n)`.
    /// @param[in] uplo Triangle to populate.
    /// @param[in,out] nested Intermediate buffer pool.
    /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
    /// @param[in,out] prev_ls Caller-owned baseline lengthscale snapshot for this leaf.
    /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
    /// @param[in,out] inc_steps Consecutive incremental delta updates applied since last rebuild.
    /// @param[in] commit_s2 True to mutate `ard_s2_cache`/`prev_ls`; false for non-destructive trials.
    /// @return Empty expected on success or GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> distance_apply_leaf_incremental(
        const CompiledDistanceKernel<T>& kernel,
        std::size_t leaf_idx,
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> out,
        Triangle uplo,
        NestedBuffers<T>& nested,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps,
        bool commit_s2 = true
    );

    /// @brief Commits accepted lengthscales of `leaf_idx` to its `ard_s2_cache` baseline.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, distance_num_leaves(kernel))`.
    /// @param[in] store Pre-validated square supplied-distance store.
    /// @param[in] x Optional coordinate matrix for `WithPoints` trees.
    /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
    /// @param[in,out] prev_ls Caller-owned baseline lengthscale snapshot for this leaf.
    /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
    /// @param[in,out] inc_steps Consecutive incremental delta updates applied since last rebuild.
    /// @return Empty expected on success or GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> distance_commit_leaf_ard_s2(
        const CompiledDistanceKernel<T>& kernel,
        std::size_t leaf_idx,
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps
    );

    /// @brief Reconstructs the composite Gram matrix from precomputed per-leaf Gram matrices.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled distance kernel.
    /// @param[in] leaf_grams Span of `distance_num_leaves(kernel)` square n x n leaf Gram views.
    /// @param[out] out Destination composite Gram matrix `(n, n)`.
    /// @param[in] uplo Triangle to populate.
    /// @param[in,out] scratch Non-aliasing n x n scratch matrix.
    /// @param[in,out] nested Pre-allocated intermediate buffer pool.
    /// @return Empty expected on success or GprError.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> distance_combine_from_leaf_grams(
        const CompiledDistanceKernel<T>& kernel,
        std::span<const MatrixRef<T>> leaf_grams,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested
    );

}  // namespace cxxgpr::detail
