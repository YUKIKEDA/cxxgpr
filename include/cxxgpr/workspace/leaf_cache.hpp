// SPDX-License-Identifier: Apache-2.0
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

namespace cxxgpr
{

    /// Concept matching optimizer types that declare a static boolean uses_change_indices member.
    template <typename O>
    concept OptimizerHasChangeIndicesConcept = requires {
        { O::uses_change_indices } -> std::convertible_to<bool>;
    };

    /// Traits describing incremental coordinate-step support for an optimizer type.
    template <typename O> struct OptimizerTraits
    {
        /// True when the optimizer supplies changed parameter indices at each evaluation step.
        static constexpr bool uses_change_indices = false;
    };

    /// Specialization for optimizers that declare static constexpr bool uses_change_indices.
    template <OptimizerHasChangeIndicesConcept O> struct OptimizerTraits<O>
    {
        /// Mirrors O::uses_change_indices for incremental leaf cache dispatch.
        static constexpr bool uses_change_indices = static_cast<bool>(O::uses_change_indices);
    };

    /// Convenience variable template reading OptimizerTraits<O>::uses_change_indices.
    template <typename O>
    inline constexpr bool optimizer_uses_change_indices_v = OptimizerTraits<O>::uses_change_indices;

    /// Validates a non-empty, in-bounds, duplicate-free span of changed parameter indices.
    [[nodiscard]] std::expected<void, GprError> require_change_indices(
        std::span<const std::size_t> indices, std::size_t n_params
    );

    /// Per-leaf Gram matrix cache supporting bit-exact change-index validation and incremental
    /// updates.
    template <KernelScalarConcept T> class LeafCache
    {
    public:
        /// Constructs an empty, unprimed leaf cache.
        LeafCache() noexcept = default;

        /// Sizes internal storage for n_leaves square matrices of order n; preserves capacity if
        /// unchanged.
        [[nodiscard]] std::expected<void, GprError> resize(std::size_t n_leaves, std::size_t n);

        /// Validates params against change_indices using bit-exact comparison and marks dirty leaves.
        [[nodiscard]] std::expected<void, GprError> prepare_step(
            std::span<const double> params,
            std::span<const std::size_t> change_indices,
            const CompiledKernel<T>& kernel,
            std::size_t total_params
        );

        /// Marks every leaf as dirty so the next evaluation recomputes all leaf Gram matrices.
        void mark_all_dirty() noexcept;

        /// Invalidates the primed state and marks all leaves dirty without releasing allocated storage.
        void invalidate() noexcept;

        /// Records params as the baseline parameter vector for subsequent bit-exact change checks.
        [[nodiscard]] std::expected<void, GprError> record(std::span<const double> params);

        /// Recomputes only dirty leaves and combines all leaf Grams into out without heap allocation.
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
            NestedBuffers<T>& nested
        );

        /// Returns true when the cached leaf Grams match the recorded parameter vector.
        [[nodiscard]] bool is_primed() const noexcept { return primed_; }

        /// Returns true if leaf_idx is marked dirty and needs recomputation.
        [[nodiscard]] bool is_leaf_dirty(std::size_t leaf_idx) const noexcept;

        /// Returns the number of cached leaves.
        [[nodiscard]] std::size_t num_leaves() const noexcept { return n_leaves_; }

        /// Returns the matrix order n of each cached leaf Gram matrix.
        [[nodiscard]] std::size_t order() const noexcept { return order_; }

        /// Returns a mutable n x n column-major view of the cached Gram matrix for leaf_idx.
        [[nodiscard]] MatrixMut<T> leaf_gram_mut(std::size_t leaf_idx) noexcept;

        /// Returns an immutable n x n column-major view of the cached Gram matrix for leaf_idx.
        [[nodiscard]] MatrixRef<T> leaf_gram(std::size_t leaf_idx) const noexcept;

        /// Returns the baseline parameter vector recorded on the last successful step.
        [[nodiscard]] std::span<const double> last_params() const noexcept { return last_params_; }

    private:
        std::size_t n_leaves_ = 0;
        std::size_t order_ = 0;
        std::vector<T> storage_;
        std::vector<MatrixRef<T>> gram_views_;
        std::vector<std::uint8_t> dirty_;
        std::vector<double> last_params_;
        bool primed_ = false;
    };

    extern template class LeafCache<double>;
    extern template class LeafCache<float>;

}  // namespace cxxgpr
