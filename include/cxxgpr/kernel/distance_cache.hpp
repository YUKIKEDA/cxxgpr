// SPDX-License-Identifier: Apache-2.0
/// @file distance_cache.hpp
/// @brief Precomputed squared-distance and packed ARD difference cache for kernel evaluation.
/// @details Defines DistanceCachePolicy and DistanceCache<T>, which store the dense n x n Euclidean
/// squared-distance matrix and packed lower-triangular per-dimension squared differences (dx_d)^2
/// across optimizer iterations to avoid repeated coordinate scans.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Controls whether training distance caches are precomputed and reused across iterations.
    /// @details Tradeoff selector between memory footprint and repeated coordinate-to-distance
    /// computation during hyperparameter optimization.
    enum class DistanceCachePolicy : std::uint8_t
    {
        /// @brief Precomputes and reuses squared-distance and per-dimension squared-difference caches.
        Cached,
        /// @brief Computes distances directly from input coordinates without allocating cache buffers.
        Uncached,
    };

    /// @brief Precomputed training-point squared Euclidean distances and packed ARD squared differences.
    /// @details Stores an optional n x n column-major squared-distance table for isotropic leaves and an
    /// optional packed lower-triangular per-dimension (dx_d)^2 buffer of size d * n * (n + 1) / 2 for
    /// ARD leaves.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class DistanceCache
    {
    public:
        /// @brief Constructs an empty distance cache in unallocated state.
        /// @details Policy defaults to DistanceCachePolicy::Cached with zero points and dimensions until
        /// populated by ensure() or build().
        DistanceCache() noexcept = default;

        /// @brief Builds a distance cache for training points x according to cache flags and policy.
        /// @details Validates x and populates requested distance and/or ARD buffers when policy is Cached.
        /// @param x Training input matrix of shape n x d in column-major order.
        /// @param cache_dist Whether to populate the dense n x n squared-distance matrix.
        /// @param cache_ard Whether to populate the packed lower-triangular ARD squared-difference array.
        /// @param policy Caching policy governing whether buffers are allocated or left empty.
        /// @return Populated DistanceCache<T> on success or GprError on invalid input.
        /// @retval GprErrorCode::EmptyInput When x has zero rows, zero columns, or null data.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is NaN or infinite.
        /// @retval GprErrorCode::SizeOverflow When cache element counts overflow std::size_t.
        [[nodiscard]] static std::expected<DistanceCache<T>, GprError> build(
            MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy = DistanceCachePolicy::Cached
        );

        /// @brief Populates or reuses this cache in place for training points x.
        /// @details Validates x before modifying internal buffers; when policy is Uncached or a flag is
        /// false, the corresponding internal vector is cleared.
        /// @param x Training input matrix of shape n x d in column-major order.
        /// @param cache_dist Whether to populate the dense n x n squared-distance matrix.
        /// @param cache_ard Whether to populate the packed lower-triangular ARD squared-difference array.
        /// @param policy Caching policy governing whether buffers are allocated or cleared.
        /// @return Empty expected on success or GprError if x is invalid or allocation size overflows.
        /// @retval GprErrorCode::EmptyInput When x has zero rows, zero columns, or null data.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is NaN or infinite.
        /// @retval GprErrorCode::SizeOverflow When cache element counts overflow std::size_t.
        [[nodiscard]] std::expected<void, GprError> ensure(
            MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy = DistanceCachePolicy::Cached
        );

        /// @brief Clears cached entries and resets stored point and dimension counts to zero.
        /// @details Retains vector capacity for reuse while clearing logical contents.
        void reset() noexcept;

        /// @brief Returns the policy recorded during the last build or ensure call.
        /// @details Defaults to DistanceCachePolicy::Cached before any build or ensure call.
        /// @return Active DistanceCachePolicy.
        [[nodiscard]] DistanceCachePolicy policy() const noexcept { return policy_; }

        /// @brief Returns the number of training points represented by the cache.
        /// @details Equal to x.rows() from the last successful ensure() or build(), or 0 if reset.
        /// @return Training point count n.
        [[nodiscard]] std::size_t num_points() const noexcept { return num_points_; }

        /// @brief Returns the number of input dimensions represented by the cache.
        /// @details Equal to x.cols() from the last successful ensure() or build(), or 0 if reset.
        /// @return Input feature dimension count d.
        [[nodiscard]] std::size_t num_dims() const noexcept { return num_dims_; }

        /// @brief Returns true when the dense squared-distance matrix is populated.
        /// @details Checks whether the internal n x n squared-distance storage is non-empty.
        /// @return True if dist_view() addresses valid n x n cached distances.
        [[nodiscard]] bool has_dist() const noexcept { return !dist_.empty(); }

        /// @brief Returns true when the packed lower-triangular ARD squared-difference array is populated.
        /// @details Checks whether the internal d * n * (n + 1) / 2 ARD storage is non-empty.
        /// @return True if ard_packed() addresses valid packed per-dimension squared differences.
        [[nodiscard]] bool has_ard() const noexcept { return !ard_sq_diff_.empty(); }

        /// @brief Returns an immutable column-major view of the n x n squared-distance matrix.
        /// @details Returns an empty view when has_dist() is false.
        /// @return Immutable n x n column-major view over cached squared Euclidean distances.
        [[nodiscard]] MatrixRef<T> dist_view() const noexcept;

        /// @brief Returns a mutable column-major view of the n x n squared-distance matrix.
        /// @details Returns an empty view when has_dist() is false.
        /// @return Mutable n x n column-major view over cached squared Euclidean distances.
        [[nodiscard]] MatrixMut<T> dist_mut() noexcept;

        /// @brief Returns an immutable span over the packed lower-triangular ARD squared differences.
        /// @details Layout stores d contiguous dimension blocks of length n * (n + 1) / 2, each storing
        /// column-major lower-triangular entries (row >= col) at col * (2n - col + 1) / 2 + (row - col).
        /// @return Immutable span of length d * n * (n + 1) / 2 (or empty if unpopulated).
        [[nodiscard]] std::span<const T> ard_packed() const noexcept { return ard_sq_diff_; }

        /// @brief Returns a mutable span over the packed lower-triangular ARD squared differences.
        /// @details Uses the same dimension-major, column-major lower-triangular packed layout as ard_packed().
        /// @return Mutable span of length d * n * (n + 1) / 2 (or empty if unpopulated).
        [[nodiscard]] std::span<T> ard_packed_mut() noexcept { return ard_sq_diff_; }

    private:
        /// @brief Active caching policy recorded by the last build or ensure call.
        DistanceCachePolicy policy_ = DistanceCachePolicy::Cached;
        /// @brief Number of training points n currently represented by the cache.
        std::size_t num_points_ = 0;
        /// @brief Number of input feature dimensions d currently represented by the cache.
        std::size_t num_dims_ = 0;
        /// @brief Column-major dense n x n buffer holding pairwise squared Euclidean distances.
        std::vector<T> dist_;
        /// @brief Packed lower-triangular per-dimension (dx_d)^2 buffer of length d * n * (n + 1) / 2.
        std::vector<T> ard_sq_diff_;
    };

    extern template class DistanceCache<double>;
    extern template class DistanceCache<float>;

}  // namespace cxxgpr
