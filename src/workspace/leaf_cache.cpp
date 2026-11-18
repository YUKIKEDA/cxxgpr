// SPDX-License-Identifier: Apache-2.0
/// @file leaf_cache.cpp
/// @brief Implementations and explicit instantiations for per-leaf incremental Gram caching.
/// @details Implements duplicate/bounds validation of changed parameter indices, bit-exact
/// parameter tracking via std::bit_cast, and selective dirty-leaf Gram recomputation.

#include "cxxgpr/workspace/leaf_cache.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>

namespace cxxgpr
{

    /// @brief Validates a non-empty, in-bounds, duplicate-free span of changed parameter indices.
    /// @details Verifies that indices is non-empty, each entry is strictly less than n_params, and
    /// no index appears more than once.
    /// @param indices Span of zero-based parameter indices claimed to have changed.
    /// @param n_params Total number of objective parameters (kernel + likelihood).
    /// @return Empty expected on success, or GprError if validation fails.
    /// @retval GprErrorCode::IndexOutOfRange When indices is empty, out of range, or contains duplicates.
    std::expected<void, GprError> require_change_indices(std::span<const std::size_t> indices, std::size_t n_params)
    {
        // Reject empty change lists and verify every index is unique and within [0, n_params).
        if (indices.empty())
        {
            return std::unexpected(GprError::index_out_of_range("change indices must not be empty"));
        }
        for (std::size_t k = 0; k < indices.size(); ++k)
        {
            const std::size_t idx = indices[k];
            if (idx >= n_params)
            {
                return std::unexpected(GprError::index_out_of_range("change index is out of parameter range"));
            }
            for (std::size_t prev = 0; prev < k; ++prev)
            {
                if (indices[prev] == idx)
                {
                    return std::unexpected(GprError::index_out_of_range("change indices must not contain duplicates"));
                }
            }
        }
        return {};
    }

    /// @brief Sizes internal storage for n_leaves square matrices of order n; preserves capacity if unchanged.
    /// @details Short-circuits when n_leaves, order_, and metadata vectors already match; otherwise checks
    /// overflow, allocates contiguous storage, populates MatrixRef views, and resets primed_ to false.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param n_leaves Number of kernel leaves in the flattened CompiledKernel tree.
    /// @param n Training sample count (matrix order).
    /// @return Empty expected on success, or GprError if element counts overflow std::size_t.
    /// @retval GprErrorCode::SizeOverflow When n * n or n_leaves * n * n overflows std::size_t.
    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::resize(std::size_t n_leaves, std::size_t n)
    {
        // Avoid reallocating or invalidating primed state when leaf count and matrix order match.
        if (n_leaves_ == n_leaves && order_ == n && dirty_.size() == n_leaves && gram_views_.size() == n_leaves)
        {
            return {};
        }

        auto mat_len = checked_mul(n, n);
        if (!mat_len.has_value())
        {
            return std::unexpected(std::move(mat_len).error());
        }
        auto total_len = checked_mul(n_leaves, *mat_len);
        if (!total_len.has_value())
        {
            return std::unexpected(std::move(total_len).error());
        }

        // Partition contiguous storage into n_leaves column-major n x n MatrixRef views.
        n_leaves_ = n_leaves;
        order_ = n;
        storage_.assign(*total_len, T{0});
        gram_views_.resize(n_leaves_);
        for (std::size_t idx = 0; idx < n_leaves_; ++idx)
        {
            gram_views_[idx] = MatrixRef<T>(storage_.data() + (idx * *mat_len), order_, order_, order_);
        }
        dirty_.assign(n_leaves_, 1U);
        primed_ = false;
        return {};
    }

    /// @brief Validates params against change_indices using bit-exact comparison and marks dirty leaves.
    /// @details When unprimed or parameter length changed, marks all leaves dirty; when primed, checks
    /// every parameter outside change_indices via bit_cast<std::uint64_t> equality against last_params_
    /// and marks only leaves owning changed kernel indices dirty.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param params Candidate full log-space parameter vector.
    /// @param change_indices Non-empty duplicate-free span of changed parameter indices.
    /// @param kernel Compiled kernel tree mapping parameter indices to owning leaf indices.
    /// @param total_params Expected full parameter count.
    /// @return Empty expected on success, or GprError on length mismatch or undeclared parameter mutation.
    /// @retval GprErrorCode::LengthMismatch When params.size() != total_params.
    /// @retval GprErrorCode::IndexOutOfRange When change_indices is invalid or an unlisted parameter changed.
    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::prepare_step(
        std::span<const double> params,
        std::span<const std::size_t> change_indices,
        const CompiledKernel<T>& kernel,
        std::size_t total_params
    )
    {
        if (params.size() != total_params)
        {
            return std::unexpected(GprError::length_mismatch("parameter vector length mismatch"));
        }
        if (auto valid = require_change_indices(change_indices, total_params); !valid.has_value())
        {
            return valid;
        }
        if (dirty_.size() != n_leaves_)
        {
            dirty_.assign(n_leaves_, 1U);
            primed_ = false;
        }

        // Force full leaf recomputation on the first step or if parameter dimension changed.
        if (!primed_ || last_params_.size() != total_params)
        {
            std::ranges::fill(dirty_, 1U);
            return {};
        }

        // Enforce bit-exact invariance on all parameters omitted from change_indices.
        for (std::size_t j = 0; j < total_params; ++j)
        {
            const auto prev_bits = std::bit_cast<std::uint64_t>(last_params_[j]);
            const auto next_bits = std::bit_cast<std::uint64_t>(params[j]);
            if (prev_bits != next_bits)
            {
                const bool listed = std::ranges::find(change_indices, j) != change_indices.end();
                if (!listed)
                {
                    return std::unexpected(
                        GprError::index_out_of_range("parameter changed outside declared change indices")
                    );
                }
            }
        }

        // Mark only leaves whose kernel parameter index appears in change_indices; noise-only steps leave all clean.
        std::ranges::fill(dirty_, 0U);
        const std::size_t n_kernel = kernel.num_params();
        for (const std::size_t j : change_indices)
        {
            if (j < n_kernel)
            {
                auto leaf_idx = kernel.leaf_index_for_param(j);
                if (!leaf_idx.has_value())
                {
                    return std::unexpected(std::move(leaf_idx).error());
                }
                if (*leaf_idx < dirty_.size())
                {
                    dirty_[*leaf_idx] = 1U;
                }
            }
        }
        return {};
    }

    /// @brief Marks every leaf as dirty so the next evaluation recomputes all leaf Gram matrices.
    /// @details Sets all entries of dirty_ to 1 without altering primed_.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @return None (void).
    template <KernelScalarConcept T> void LeafCache<T>::mark_all_dirty() noexcept
    {
        std::ranges::fill(dirty_, 1U);
    }

    /// @brief Invalidates the primed state and marks all leaves dirty without releasing allocated storage.
    /// @details Resets primed_ to false and flags all leaves dirty.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @return None (void).
    template <KernelScalarConcept T> void LeafCache<T>::invalidate() noexcept
    {
        primed_ = false;
        mark_all_dirty();
    }

    /// @brief Records params as the baseline parameter vector for subsequent bit-exact change checks.
    /// @details Copies params into last_params_ and marks primed_ = true.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param params Full log-space parameter vector matching the current cached leaf Grams.
    /// @return Empty expected on completion.
    template <KernelScalarConcept T> std::expected<void, GprError> LeafCache<T>::record(std::span<const double> params)
    {
        last_params_.resize(params.size());
        std::ranges::copy(params, last_params_.begin());
        primed_ = true;
        return {};
    }

    /// @brief Recomputes only dirty leaves and combines all leaf Grams into out without heap allocation.
    /// @details Resizes cache storage if needed, validates change_indices bit-exactly via prepare_step(),
    /// evaluates each dirty leaf using dist_cache when available, combines all leaf Grams via
    /// combine_from_leaf_grams(), and records params upon success (invalidating on any mid-step error).
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param kernel Compiled execution-layer kernel tree.
    /// @param x Column-major n x d training coordinate matrix.
    /// @param dist_cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
    /// @param params Candidate full log-space parameter vector of length total_params.
    /// @param change_indices Non-empty duplicate-free span of changed parameter indices.
    /// @param total_params Expected length of params.
    /// @param out Column-major n x n destination composite Gram matrix.
    /// @param uplo Triangular or full matrix region to populate.
    /// @param scratch Non-aliasing n x n scratch matrix view for tree combination.
    /// @param nested Pre-allocated NestedBuffers with depth >= kernel.nested_depth() and shape >= n x n.
    /// @return Empty expected on success, or GprError if validation, leaf evaluation, or combination fails.
    /// @retval GprErrorCode::EmptyInput When x is empty or null.
    /// @retval GprErrorCode::SizeOverflow When leaf cache sizing overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When params or output/cache dimensions mismatch.
    /// @retval GprErrorCode::IndexOutOfRange When change_indices is invalid or an unlisted parameter changed.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD or cache dimensions.
    /// @retval GprErrorCode::NonFiniteInput When x contains non-finite values.
    /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
    /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::evaluate_incremental(
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
    )
    {
        if (x.is_empty() || x.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto sized = resize(kernel.num_leaves(), x.rows()); !sized.has_value())
        {
            return sized;
        }
        if (auto prep = prepare_step(params, change_indices, kernel, total_params); !prep.has_value())
        {
            return prep;
        }

        // Recompute only leaves marked dirty, clearing each flag once its Gram slot succeeds.
        for (std::size_t leaf_idx = 0; leaf_idx < n_leaves_; ++leaf_idx)
        {
            if (dirty_[leaf_idx] == 0U)
            {
                continue;
            }
            MatrixMut<T> slot = leaf_gram_mut(leaf_idx);
            std::expected<void, GprError> eval_res;
            if (dist_cache != nullptr)
            {
                eval_res = kernel.apply_leaf_cached(leaf_idx, x, *dist_cache, slot, uplo, nested);
            }
            else
            {
                eval_res = kernel.apply_leaf_points(leaf_idx, x, slot, uplo, nested);
            }
            if (!eval_res.has_value())
            {
                // Invalidate whole cache on failure because previous clean leaves remain at old params.
                invalidate();
                return eval_res;
            }
            dirty_[leaf_idx] = 0U;
        }

        // Recombine all leaf Grams according to the sum/product tree and commit params as new baseline.
        if (auto comb = kernel.combine_from_leaf_grams(gram_views_, out, uplo, scratch, nested); !comb.has_value())
        {
            invalidate();
            return comb;
        }
        return record(params);
    }

    /// @brief Returns true if leaf_idx is marked dirty and needs recomputation.
    /// @details Bounds-checks leaf_idx against dirty_.size() and returns false if out of range.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param leaf_idx Zero-based leaf index.
    /// @return True if leaf_idx is dirty, false otherwise.
    template <KernelScalarConcept T> bool LeafCache<T>::is_leaf_dirty(std::size_t leaf_idx) const noexcept
    {
        if (leaf_idx >= dirty_.size())
        {
            return false;
        }
        return dirty_[leaf_idx] != 0U;
    }

    /// @brief Returns a mutable n x n column-major view of the cached Gram matrix for leaf_idx.
    /// @details Returns an empty MatrixMut<T> when leaf_idx >= n_leaves_.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param leaf_idx Zero-based leaf index in [0, num_leaves()).
    /// @return Mutable n x n column-major view of leaf_idx storage.
    template <KernelScalarConcept T> MatrixMut<T> LeafCache<T>::leaf_gram_mut(std::size_t leaf_idx) noexcept
    {
        if (leaf_idx >= n_leaves_)
        {
            return {};
        }
        const std::size_t mat_len = order_ * order_;
        return {storage_.data() + (leaf_idx * mat_len), order_, order_, order_};
    }

    /// @brief Returns an immutable n x n column-major view of the cached Gram matrix for leaf_idx.
    /// @details Returns an empty MatrixRef<T> when leaf_idx >= n_leaves_.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param leaf_idx Zero-based leaf index in [0, num_leaves()).
    /// @return Immutable n x n column-major view of leaf_idx storage.
    template <KernelScalarConcept T> MatrixRef<T> LeafCache<T>::leaf_gram(std::size_t leaf_idx) const noexcept
    {
        if (leaf_idx >= n_leaves_)
        {
            return {};
        }
        return gram_views_[leaf_idx];
    }

    template class LeafCache<double>;
    template class LeafCache<float>;

}  // namespace cxxgpr
