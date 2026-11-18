// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace cxxgpr
{

    /// Controls whether training distance caches are precomputed and reused across iterations.
    enum class DistanceCachePolicy : std::uint8_t
    {
        /// Precomputes and reuses squared-distance and per-dimension squared-difference caches.
        Cached,
        /// Computes distances directly from input coordinates without allocating cache buffers.
        Uncached,
    };

    /// Precomputed training-point squared Euclidean distances and packed ARD squared differences.
    template <KernelScalarConcept T> class DistanceCache
    {
    public:
        /// Constructs an empty distance cache (Uncached / unallocated state).
        DistanceCache() noexcept = default;

        /// Builds a distance cache for training points x according to cache flags and policy.
        [[nodiscard]] static std::expected<DistanceCache<T>, GprError> build(
            MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy = DistanceCachePolicy::Cached
        );

        /// Populates or reuses this cache in place for training points x.
        [[nodiscard]] std::expected<void, GprError> ensure(
            MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy = DistanceCachePolicy::Cached
        );

        /// Clears cached views while releasing or resetting stored entries.
        void reset() noexcept;

        /// Returns the policy recorded during the last build or ensure call.
        [[nodiscard]] DistanceCachePolicy policy() const noexcept { return policy_; }

        /// Returns the number of training points represented by the cache.
        [[nodiscard]] std::size_t num_points() const noexcept { return num_points_; }

        /// Returns the number of input dimensions represented by the cache.
        [[nodiscard]] std::size_t num_dims() const noexcept { return num_dims_; }

        /// Returns true when the dense squared-distance matrix is populated.
        [[nodiscard]] bool has_dist() const noexcept { return !dist_.empty(); }

        /// Returns true when the packed lower-triangular ARD squared-difference array is populated.
        [[nodiscard]] bool has_ard() const noexcept { return !ard_sq_diff_.empty(); }

        /// Returns an immutable column-major view of the n x n squared-distance matrix.
        [[nodiscard]] MatrixRef<T> dist_view() const noexcept;

        /// Returns a mutable column-major view of the n x n squared-distance matrix.
        [[nodiscard]] MatrixMut<T> dist_mut() noexcept;

        /// Returns an immutable span over the packed lower-triangular ARD squared differences.
        [[nodiscard]] std::span<const T> ard_packed() const noexcept { return ard_sq_diff_; }

        /// Returns a mutable span over the packed lower-triangular ARD squared differences.
        [[nodiscard]] std::span<T> ard_packed_mut() noexcept { return ard_sq_diff_; }

    private:
        DistanceCachePolicy policy_ = DistanceCachePolicy::Cached;
        std::size_t num_points_ = 0;
        std::size_t num_dims_ = 0;
        std::vector<T> dist_;
        std::vector<T> ard_sq_diff_;
    };

    extern template class DistanceCache<double>;
    extern template class DistanceCache<float>;

}  // namespace cxxgpr
