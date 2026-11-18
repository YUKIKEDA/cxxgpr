// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/workspace/leaf_cache.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>

namespace cxxgpr
{

    std::expected<void, GprError> require_change_indices(std::span<const std::size_t> indices, std::size_t n_params)
    {
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

    template <KernelScalarConcept T>
    std::expected<void, GprError> LeafCache<T>::resize(std::size_t n_leaves, std::size_t n)
    {
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

        if (!primed_ || last_params_.size() != total_params)
        {
            std::ranges::fill(dirty_, 1U);
            return {};
        }

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

    template <KernelScalarConcept T> void LeafCache<T>::mark_all_dirty() noexcept
    {
        std::ranges::fill(dirty_, 1U);
    }

    template <KernelScalarConcept T> void LeafCache<T>::invalidate() noexcept
    {
        primed_ = false;
        mark_all_dirty();
    }

    template <KernelScalarConcept T> std::expected<void, GprError> LeafCache<T>::record(std::span<const double> params)
    {
        last_params_.resize(params.size());
        std::ranges::copy(params, last_params_.begin());
        primed_ = true;
        return {};
    }

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
                invalidate();
                return eval_res;
            }
            dirty_[leaf_idx] = 0U;
        }

        if (auto comb = kernel.combine_from_leaf_grams(gram_views_, out, uplo, scratch, nested); !comb.has_value())
        {
            invalidate();
            return comb;
        }
        return record(params);
    }

    template <KernelScalarConcept T> bool LeafCache<T>::is_leaf_dirty(std::size_t leaf_idx) const noexcept
    {
        if (leaf_idx >= dirty_.size())
        {
            return false;
        }
        return dirty_[leaf_idx] != 0U;
    }

    template <KernelScalarConcept T> MatrixMut<T> LeafCache<T>::leaf_gram_mut(std::size_t leaf_idx) noexcept
    {
        if (leaf_idx >= n_leaves_)
        {
            return {};
        }
        const std::size_t mat_len = order_ * order_;
        return {storage_.data() + (leaf_idx * mat_len), order_, order_, order_};
    }

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
