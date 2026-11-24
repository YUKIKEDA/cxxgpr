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
    /// @param[in] indices Span of zero-based parameter indices claimed to have changed.
    /// @param[in] n_params Total number of objective parameters (kernel + likelihood).
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

    /// @brief Sizes internal storage for n_leaves square matrices of order n; preserves capacity if sufficient.
    /// @details Short-circuits when n_leaves, order_, metadata vectors, and per-leaf ARD capacities already suffice;
    /// otherwise checks overflow, allocates O(n^2) Gram + ARD s^2 slots and O(d) lengthscale snapshots dynamically,
    /// and resets primed state.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] n_leaves Number of kernel leaves in the flattened CompiledKernel tree.
    /// @param[in] n Training sample count (matrix order).
    /// @param[in] ard_dims_per_leaf Optional per-leaf ARD dimension counts (`0` for non-ARD leaves).
    /// @return Empty expected on success, or GprError on overflow or dimension metadata mismatch.
    /// @retval GprErrorCode::SizeOverflow When buffer element counts overflow std::size_t.
    /// @retval GprErrorCode::LengthMismatch When ard_dims_per_leaf is non-empty and its size != n_leaves.
    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::resize(
        std::size_t n_leaves, std::size_t n, std::span<const std::size_t> ard_dims_per_leaf
    )
    {
        if (!ard_dims_per_leaf.empty() && ard_dims_per_leaf.size() != n_leaves)
        {
            return std::unexpected(GprError::length_mismatch("ARD dimension metadata count mismatch"));
        }

        // Preserve primed state and existing capacity whenever shape and ARD capacities already suffice.
        bool can_reuse =
            (n_leaves_ == n_leaves && order_ == n && dirty_.size() == n_leaves && gram_views_.size() == n_leaves
                && ard_s2_offsets_.size() == n_leaves && ard_ls_offsets_.size() == n_leaves
                && ard_ls_capacities_.size() == n_leaves && ard_s2_valid_.size() == n_leaves
                && ard_inc_steps_.size() == n_leaves && trial_modified_.size() == n_leaves
                && stale_after_reject_.size() == n_leaves);
        if (can_reuse)
        {
            for (std::size_t i = 0U; i < n_leaves; ++i)
            {
                const std::size_t req_dim = ard_dims_per_leaf.empty() ? k_default_ls_cap_ : ard_dims_per_leaf[i];
                const bool need_s2 = ard_dims_per_leaf.empty() || (req_dim > 0U);
                if (ard_ls_capacities_[i] < req_dim || (need_s2 && ard_s2_offsets_[i] == k_no_s2_slot_))
                {
                    can_reuse = false;
                    break;
                }
            }
            if (can_reuse)
            {
                return {};
            }
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
        auto n_plus_one = checked_add(n, std::size_t{1});
        if (!n_plus_one.has_value())
        {
            return std::unexpected(std::move(n_plus_one).error());
        }
        auto n_np1 = checked_mul(n, *n_plus_one);
        if (!n_np1.has_value())
        {
            return std::unexpected(std::move(n_np1).error());
        }
        const std::size_t per_dim = *n_np1 / 2U;

        std::vector<std::size_t> next_ls_caps(n_leaves, 0U);
        std::vector<std::size_t> next_ls_offsets(n_leaves, 0U);
        std::vector<std::size_t> next_s2_offsets(n_leaves, k_no_s2_slot_);
        std::size_t ls_total = 0U;
        std::size_t ard_total = 0U;
        for (std::size_t i = 0U; i < n_leaves; ++i)
        {
            const std::size_t req_dim = ard_dims_per_leaf.empty() ? k_default_ls_cap_ : ard_dims_per_leaf[i];
            const std::size_t cap = (n_leaves == n_leaves_ && i < ard_ls_capacities_.size())
                ? std::max(ard_ls_capacities_[i], req_dim)
                : req_dim;
            next_ls_caps[i] = cap;
            next_ls_offsets[i] = ls_total;
            auto ls_next = checked_add(ls_total, cap);
            if (!ls_next.has_value())
            {
                return std::unexpected(std::move(ls_next).error());
            }
            ls_total = *ls_next;

            const bool need_s2 = ard_dims_per_leaf.empty() || (cap > 0U);
            if (need_s2)
            {
                next_s2_offsets[i] = ard_total;
                auto ard_next = checked_add(ard_total, per_dim);
                if (!ard_next.has_value())
                {
                    return std::unexpected(std::move(ard_next).error());
                }
                ard_total = *ard_next;
            }
        }

        // Partition contiguous storage into n_leaves column-major n x n MatrixRef views and O(n^2) ARD s^2 slots.
        n_leaves_ = n_leaves;
        order_ = n;
        ard_per_dim_ = per_dim;
        storage_.assign(*total_len, T{0});
        gram_views_.resize(n_leaves_);
        for (std::size_t idx = 0; idx < n_leaves_; ++idx)
        {
            gram_views_[idx] = MatrixRef<T>(storage_.data() + (idx * *mat_len), order_, order_, order_);
        }
        dirty_.assign(n_leaves_, 1U);
        trial_modified_.assign(n_leaves_, 0U);
        stale_after_reject_.assign(n_leaves_, 0U);
        ard_s2_storage_.assign(ard_total, T{0});
        ard_s2_offsets_ = std::move(next_s2_offsets);
        ard_prev_ls_.assign(ls_total, 0.0);
        ard_ls_offsets_ = std::move(next_ls_offsets);
        ard_ls_capacities_ = std::move(next_ls_caps);
        ard_s2_valid_.assign(n_leaves_, 0U);
        ard_inc_steps_.assign(n_leaves_, 0U);
        primed_ = false;
        has_pending_ = false;
        return {};
    }

    /// @brief Validates params against change_indices using bit-exact comparison and marks dirty leaves.
    /// @details When unprimed or parameter length changed, marks all leaves dirty; when primed, checks
    /// every parameter outside change_indices via bit_cast<std::uint64_t> equality against last_eval_params_
    /// and marks leaves retained in stale_after_reject_ plus leaves owning changed kernel indices dirty.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] params Candidate full log-space parameter vector.
    /// @param[in] change_indices Non-empty duplicate-free span of changed parameter indices.
    /// @param[in] kernel Compiled kernel tree mapping parameter indices to owning leaf indices.
    /// @param[in] total_params Expected full parameter count.
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
        if (dirty_.size() != n_leaves_ || trial_modified_.size() != n_leaves_
            || stale_after_reject_.size() != n_leaves_)
        {
            dirty_.assign(n_leaves_, 1U);
            trial_modified_.assign(n_leaves_, 0U);
            stale_after_reject_.assign(n_leaves_, 0U);
            ard_s2_valid_.assign(n_leaves_, 0U);
            ard_inc_steps_.assign(n_leaves_, 0U);
            primed_ = false;
            has_pending_ = false;
        }

        // Force full leaf recomputation on the first step or if parameter dimension changed.
        if (!primed_ || last_eval_params_.size() != total_params)
        {
            std::ranges::fill(dirty_, 1U);
            std::ranges::fill(ard_s2_valid_, 0U);
            std::ranges::fill(ard_inc_steps_, 0U);
            std::ranges::fill(stale_after_reject_, 0U);
            return {};
        }

        // Enforce bit-exact invariance on all parameters omitted from change_indices against the last evaluated vector.
        for (std::size_t j = 0; j < total_params; ++j)
        {
            const auto prev_bits = std::bit_cast<std::uint64_t>(last_eval_params_[j]);
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

        // Keep leaves marked stale by an earlier rejected trial dirty alongside leaves changed in this step.
        std::ranges::copy(stale_after_reject_, dirty_.begin());
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
    /// @details Sets all entries of dirty_ to 1 and invalidates cached ARD s^2 state without altering primed_.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @return None (void).
    template <KernelScalarConcept T> void LeafCache<T>::mark_all_dirty() noexcept
    {
        std::ranges::fill(dirty_, 1U);
        std::ranges::fill(trial_modified_, 0U);
        std::ranges::fill(stale_after_reject_, 0U);
        std::ranges::fill(ard_s2_valid_, 0U);
        std::ranges::fill(ard_inc_steps_, 0U);
        has_pending_ = false;
    }

    /// @brief Invalidates the primed state and marks all leaves dirty without releasing allocated storage.
    /// @details Resets primed_ to false, clears ARD s^2 cache validity, and flags all leaves dirty.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @return None (void).
    template <KernelScalarConcept T> void LeafCache<T>::invalidate() noexcept
    {
        primed_ = false;
        mark_all_dirty();
    }

    /// @brief Records params as the baseline parameter vector for subsequent bit-exact change checks.
    /// @details Copies params into last_params_ and last_eval_params_, reserves trial staging capacity, clears stale
    /// trial flags, and marks primed_ = true.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] params Full log-space parameter vector matching the current cached leaf Grams.
    /// @return Empty expected on completion.
    template <KernelScalarConcept T> std::expected<void, GprError> LeafCache<T>::record(std::span<const double> params)
    {
        last_params_.resize(params.size());
        std::ranges::copy(params, last_params_.begin());
        last_eval_params_.resize(params.size());
        std::ranges::copy(params, last_eval_params_.begin());
        pending_params_.reserve(params.size());
        std::ranges::fill(trial_modified_, 0U);
        std::ranges::fill(stale_after_reject_, 0U);
        has_pending_ = false;
        primed_ = true;
        return {};
    }

    /// @brief Recomputes only dirty leaves and combines all leaf Grams into out without heap allocation.
    /// @details Sizes storage if needed, runs bit-exact invariance validation via prepare_step(), evaluates each dirty
    /// leaf via `kernel.apply_leaf_incremental` across both cached `DistanceCache` and uncached coordinate paths,
    /// combines all leaf Grams via `combine_from_leaf_grams()`, and either commits immediately or stages the candidate
    /// until `commit_candidate()` runs.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree.
    /// @param[in] x Column-major n x d training coordinate matrix.
    /// @param[in] dist_cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
    /// @param[in] params Candidate full log-space parameter vector of length total_params.
    /// @param[in] change_indices Non-empty duplicate-free span of changed parameter indices.
    /// @param[in] total_params Expected length of params.
    /// @param[out] out Column-major n x n destination composite Gram matrix.
    /// @param[in] uplo Triangular or full matrix region to populate.
    /// @param[in,out] scratch Non-aliasing n x n scratch matrix view for tree combination.
    /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= kernel.nested_depth() and shape >= n x n.
    /// @param[in] commit_immediately True to commit ARD $s^2$ and baseline immediately; false for non-destructive
    /// trial evaluation.
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
        NestedBuffers<T>& nested,
        bool commit_immediately
    )
    {
        if (x.is_empty() || x.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto sized = resize(kernel.num_leaves(), x.rows(), kernel.ard_dims_per_leaf()); !sized.has_value())
        {
            return sized;
        }
        if (auto prep = prepare_step(params, change_indices, kernel, total_params); !prep.has_value())
        {
            return prep;
        }

        std::ranges::fill(trial_modified_, 0U);

        // Recompute only leaves marked dirty, routing both Cached and Uncached ARD leaves incrementally.
        for (std::size_t leaf_idx = 0; leaf_idx < n_leaves_; ++leaf_idx)
        {
            if (dirty_[leaf_idx] == 0U)
            {
                continue;
            }
            MatrixMut<T> slot = leaf_gram_mut(leaf_idx);
            const std::span<T> s2_buf = (ard_s2_offsets_[leaf_idx] != k_no_s2_slot_)
                ? std::span<T>(ard_s2_storage_.data() + ard_s2_offsets_[leaf_idx], ard_per_dim_)
                : std::span<T>{};
            const std::span<double> prev_ls(
                ard_prev_ls_.data() + ard_ls_offsets_[leaf_idx], ard_ls_capacities_[leaf_idx]
            );
            bool s2_valid = ard_s2_valid_[leaf_idx] != 0U;
            std::size_t inc_steps = ard_inc_steps_[leaf_idx];
            auto eval_res = kernel.apply_leaf_incremental(
                leaf_idx, x, dist_cache, slot, uplo, nested, s2_buf, prev_ls, s2_valid, inc_steps, commit_immediately
            );
            if (!eval_res.has_value())
            {
                // Invalidate whole cache on failure because previous clean leaves may belong to an earlier baseline.
                invalidate();
                return eval_res;
            }
            if (commit_immediately)
            {
                ard_s2_valid_[leaf_idx] = s2_valid ? 1U : 0U;
                ard_inc_steps_[leaf_idx] = inc_steps;
                stale_after_reject_[leaf_idx] = 0U;
            }
            else
            {
                trial_modified_[leaf_idx] = 1U;
            }
            dirty_[leaf_idx] = 0U;
        }

        // Recombine all leaf Grams according to the sum/product tree.
        if (auto comb = kernel.combine_from_leaf_grams(gram_views_, out, uplo, scratch, nested); !comb.has_value())
        {
            invalidate();
            return comb;
        }
        if (commit_immediately)
        {
            return record(params);
        }
        last_eval_params_.resize(params.size());
        std::ranges::copy(params, last_eval_params_.begin());
        pending_params_.resize(params.size());
        std::ranges::copy(params, pending_params_.begin());
        has_pending_ = true;
        return {};
    }

    /// @brief Evaluates all leaf Grams and primes ARD $s^2$ baselines from a full baseline parameter vector.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] kernel Compiled execution-layer kernel tree.
    /// @param[in] x Column-major n x d training coordinate matrix.
    /// @param[in] dist_cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
    /// @param[in] params Full baseline parameter vector matching kernel.
    /// @param[out] out Column-major n x n destination composite Gram matrix.
    /// @param[in] uplo Triangular or full matrix region to populate.
    /// @param[in,out] scratch Non-aliasing n x n scratch matrix view for tree combination.
    /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= kernel.nested_depth() and shape >= n x n.
    /// @return Empty expected on success or GprError if evaluation fails.
    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::prime(
        const CompiledKernel<T>& kernel,
        MatrixRef<T> x,
        const DistanceCache<T>* dist_cache,
        std::span<const double> params,
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
        if (auto sized = resize(kernel.num_leaves(), x.rows(), kernel.ard_dims_per_leaf()); !sized.has_value())
        {
            return sized;
        }
        mark_all_dirty();
        for (std::size_t leaf_idx = 0; leaf_idx < n_leaves_; ++leaf_idx)
        {
            MatrixMut<T> slot = leaf_gram_mut(leaf_idx);
            const std::span<T> s2_buf = (ard_s2_offsets_[leaf_idx] != k_no_s2_slot_)
                ? std::span<T>(ard_s2_storage_.data() + ard_s2_offsets_[leaf_idx], ard_per_dim_)
                : std::span<T>{};
            const std::span<double> prev_ls(
                ard_prev_ls_.data() + ard_ls_offsets_[leaf_idx], ard_ls_capacities_[leaf_idx]
            );
            bool s2_valid = false;
            std::size_t inc_steps = 0U;
            if (auto res = kernel.apply_leaf_incremental(
                    leaf_idx, x, dist_cache, slot, uplo, nested, s2_buf, prev_ls, s2_valid, inc_steps, true
                );
                !res.has_value())
            {
                invalidate();
                return res;
            }
            ard_s2_valid_[leaf_idx] = s2_valid ? 1U : 0U;
            ard_inc_steps_[leaf_idx] = inc_steps;
            dirty_[leaf_idx] = 0U;
        }
        if (auto comb = kernel.combine_from_leaf_grams(gram_views_, out, uplo, scratch, nested); !comb.has_value())
        {
            invalidate();
            return comb;
        }
        return record(params);
    }

    /// @brief Commits or discards the trial candidate staged by `evaluate_incremental(..., false)`.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] accepted True if accepted by optimizer; false to discard in O(1).
    /// @param[in] kernel Compiled kernel holding pending candidate parameters.
    /// @param[in] x Column-major n x d training coordinate matrix.
    /// @param[in] dist_cache Optional precomputed DistanceCache pointer, or nullptr.
    /// @return Empty expected on success or GprError if committing ARD $s^2$ fails.
    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::commit_candidate(
        bool accepted, const CompiledKernel<T>& kernel, MatrixRef<T> x, const DistanceCache<T>* dist_cache
    )
    {
        if (!has_pending_)
        {
            return {};
        }
        if (!accepted)
        {
            for (std::size_t leaf_idx = 0U; leaf_idx < n_leaves_; ++leaf_idx)
            {
                if (trial_modified_[leaf_idx] != 0U)
                {
                    stale_after_reject_[leaf_idx] = 1U;
                    trial_modified_[leaf_idx] = 0U;
                }
            }
            has_pending_ = false;
            return {};
        }

        for (std::size_t leaf_idx = 0U; leaf_idx < n_leaves_; ++leaf_idx)
        {
            if (trial_modified_[leaf_idx] == 0U)
            {
                continue;
            }
            if (ard_s2_offsets_[leaf_idx] != k_no_s2_slot_ && ard_ls_capacities_[leaf_idx] > 0U)
            {
                const std::span<T> s2_buf(ard_s2_storage_.data() + ard_s2_offsets_[leaf_idx], ard_per_dim_);
                const std::span<double> prev_ls(
                    ard_prev_ls_.data() + ard_ls_offsets_[leaf_idx], ard_ls_capacities_[leaf_idx]
                );
                bool s2_valid = ard_s2_valid_[leaf_idx] != 0U;
                std::size_t inc_steps = ard_inc_steps_[leaf_idx];
                if (auto res = kernel.commit_leaf_ard_s2(leaf_idx, x, dist_cache, s2_buf, prev_ls, s2_valid, inc_steps);
                    !res.has_value())
                {
                    invalidate();
                    return res;
                }
                ard_s2_valid_[leaf_idx] = s2_valid ? 1U : 0U;
                ard_inc_steps_[leaf_idx] = inc_steps;
            }
            trial_modified_[leaf_idx] = 0U;
            stale_after_reject_[leaf_idx] = 0U;
        }
        last_params_.resize(pending_params_.size());
        std::ranges::copy(pending_params_, last_params_.begin());
        primed_ = true;
        has_pending_ = false;
        return {};
    }

    /// @brief Returns true if leaf_idx is marked dirty and needs recomputation.
    /// @details Bounds-checks leaf_idx against dirty_.size() and returns false if out of range.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] leaf_idx Zero-based leaf index.
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
    /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
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
    /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
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
