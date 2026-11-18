// SPDX-License-Identifier: Apache-2.0
/// @file distance_cache.cpp
/// @brief Implementation of DistanceCache<T> dense squared-distance and packed ARD cache builders.
/// @details Computes symmetric n x n squared Euclidean distances and packed lower-triangular
/// per-dimension squared differences (dx_d)^2 with OpenMP column-parallel loops and overflow checks.
#include "cxxgpr/kernel/distance_cache.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace cxxgpr
{

    namespace
    {

        /// @brief Validates that the point matrix x is non-empty and contains only finite values.
        /// @details Scans column-major columns sequentially so cache building fails early before allocating.
        /// @tparam T Kernel scalar type (double or float).
        /// @param x Input coordinate matrix of shape n x d.
        /// @return Empty expected on success or GprError on invalid input.
        /// @retval GprErrorCode::EmptyInput When x has zero size or null data pointer.
        /// @retval GprErrorCode::NonFiniteInput When any coordinate entry is NaN or infinite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_cache_points(MatrixRef<T> x)
        {
            if (x.is_empty() || x.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            // Scan columns in contiguous memory order to reject NaN/inf before any cache allocation.
            for (std::size_t col = 0; col < x.cols(); ++col)
            {
                const auto column = x.col(col);
                for (std::size_t row = 0; row < x.rows(); ++row)
                {
                    if (!std::isfinite(column[row]))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        /// @brief Computes d * n * (n + 1) / 2 with std::size_t overflow checking.
        /// @details Guards n + 1, n * (n + 1), and d * tri separately before allocating ARD cache storage.
        /// @param n Number of training points.
        /// @param d Number of feature dimensions.
        /// @return Total packed element count or GprError on overflow.
        /// @retval GprErrorCode::SizeOverflow When any intermediate product or sum exceeds std::size_t.
        [[nodiscard]] std::expected<std::size_t, GprError> checked_packed_ard_len(std::size_t n, std::size_t d)
        {
            // Prevent wrap-around on n + 1 before multiplying n * (n + 1).
            if (n > (std::numeric_limits<std::size_t>::max() - 1U))
            {
                return std::unexpected(GprError::size_overflow());
            }
            auto n_times_n1 = checked_mul(n, n + 1U);
            if (!n_times_n1.has_value())
            {
                return std::unexpected(std::move(n_times_n1).error());
            }
            const std::size_t tri = *n_times_n1 / 2U;
            return checked_mul(d, tri);
        }

        /// @brief Populates the symmetric n x n squared Euclidean distance matrix from points x.
        /// @details Computes each lower-triangular entry sum_k (x_{ik} - x_{jk})^2 once and mirrors to
        /// both triangles, zeroing the diagonal.
        /// @tparam T Kernel scalar type (double or float).
        /// @param x Validated training coordinate matrix of shape n x d.
        /// @param dst Pre-sized destination buffer of length n * n in column-major layout.
        template <KernelScalarConcept T> void fill_square_dist_matrix(MatrixRef<T> x, std::span<T> dst) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            MatrixMut<T> dist(dst.data(), n, n, n);
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Parallelize across columns for sufficiently large matrices; each column pair writes unique cells.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                dist(col, col) = T{0};
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const std::size_t r1 = row;
                    const std::size_t r2 = col;
                    T acc = T{0};
                    for (std::size_t k = 0; k < d; ++k)
                    {
                        const T diff = x(r1, k) - x(r2, k);
                        acc += diff * diff;
                    }
                    // Mirror lower-triangular result into upper triangle so either Triangle view reads it directly.
                    dist(r1, r2) = acc;
                    dist(r2, r1) = acc;
                }
            }
        }

        /// @brief Populates packed lower-triangular per-dimension squared differences (dx_d)^2.
        /// @details For each dimension dim in [0, d), stores n * (n + 1) / 2 entries in packed
        /// column-major lower-triangular order where column col starts at dim_base + col * (2n - col + 1) / 2.
        /// @tparam T Kernel scalar type (double or float).
        /// @param x Validated training coordinate matrix of shape n x d.
        /// @param dst Pre-sized packed destination buffer of length d * n * (n + 1) / 2.
        template <KernelScalarConcept T> void fill_packed_ard_sq_diff(MatrixRef<T> x, std::span<T> dst) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            const std::size_t per_dim = (n * (n + 1U)) / 2U;
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Lay out one contiguous lower-triangular block per feature dimension for sequential leaf reads.
            for (std::size_t dim = 0; dim < d; ++dim)
            {
                const auto x_col = x.col(dim);
                const std::size_t dim_base = dim * per_dim;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
#endif
                for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
                {
                    const auto col = static_cast<std::size_t>(col_s);
                    // Column col has (n - col) lower entries; preceding columns sum to col*(2n - col + 1)/2.
                    const std::size_t col_base = dim_base + ((col * ((2U * n) - col + 1U)) / 2U);
                    const T x_col_val = x_col[col];
                    dst[col_base] = T{0};
                    for (std::size_t row = col + 1U; row < n; ++row)
                    {
                        const T diff = x_col[row] - x_col_val;
                        dst[col_base + (row - col)] = diff * diff;
                    }
                }
            }
        }

    }  // namespace

    /// @brief Builds a distance cache for training points x according to cache flags and policy.
    /// @details Allocates a fresh DistanceCache<T> and delegates validation and population to ensure().
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param x Training input matrix of shape n x d.
    /// @param cache_dist Whether to populate the dense n x n squared-distance matrix.
    /// @param cache_ard Whether to populate the packed lower-triangular ARD squared-difference buffer.
    /// @param policy Caching policy governing allocation vs uncached operation.
    /// @return Populated DistanceCache<T> on success or GprError on invalid input or overflow.
    /// @retval GprErrorCode::EmptyInput When x is empty or null.
    /// @retval GprErrorCode::NonFiniteInput When any coordinate of x is NaN or infinite.
    /// @retval GprErrorCode::SizeOverflow When requested buffer size exceeds std::size_t.
    template <KernelScalarConcept T>
    std::expected<DistanceCache<T>, GprError> DistanceCache<T>::build(
        MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy
    )
    {
        DistanceCache<T> out;
        if (auto ok = out.ensure(x, cache_dist, cache_ard, policy); !ok.has_value())
        {
            return std::unexpected(std::move(ok).error());
        }
        return out;
    }

    /// @brief Populates or reuses this cache in place for training points x.
    /// @details Validates points and checks buffer sizes before mutating existing cache state.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param x Training input matrix of shape n x d.
    /// @param cache_dist Whether to populate the dense n x n squared-distance matrix.
    /// @param cache_ard Whether to populate the packed lower-triangular ARD squared-difference buffer.
    /// @param policy Caching policy governing allocation vs uncached clearing.
    /// @return Empty expected on success or GprError if validation or size checks fail.
    /// @retval GprErrorCode::EmptyInput When x is empty or null.
    /// @retval GprErrorCode::NonFiniteInput When any coordinate of x is NaN or infinite.
    /// @retval GprErrorCode::SizeOverflow When requested buffer size exceeds std::size_t.
    template <KernelScalarConcept T>
    std::expected<void, GprError> DistanceCache<T>::ensure(
        MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy
    )
    {
        // Validate input coordinates and dimensions before computing buffer sizes or mutating state.
        if (auto valid = validate_cache_points(x); !valid.has_value())
        {
            return valid;
        }

        const std::size_t n = x.rows();
        const std::size_t d = x.cols();

        std::size_t dist_len = 0;
        std::size_t ard_len = 0;
        if (policy == DistanceCachePolicy::Cached)
        {
            if (cache_dist)
            {
                auto n2 = checked_mul(n, n);
                if (!n2.has_value())
                {
                    return std::unexpected(std::move(n2).error());
                }
                dist_len = *n2;
            }
            if (cache_ard)
            {
                auto alen = checked_packed_ard_len(n, d);
                if (!alen.has_value())
                {
                    return std::unexpected(std::move(alen).error());
                }
                ard_len = *alen;
            }
        }

        // Commit dimensions and populate or clear each cache buffer independently.
        policy_ = policy;
        num_points_ = n;
        num_dims_ = d;

        if (dist_len == 0U)
        {
            dist_.clear();
        }
        else
        {
            dist_.resize(dist_len);
            fill_square_dist_matrix(x, std::span<T>(dist_));
        }

        if (ard_len == 0U)
        {
            ard_sq_diff_.clear();
        }
        else
        {
            ard_sq_diff_.resize(ard_len);
            fill_packed_ard_sq_diff(x, std::span<T>(ard_sq_diff_));
        }

        return {};
    }

    /// @brief Clears cached entries and resets stored point and dimension counts to zero.
    /// @details Retains vector capacity while clearing elements so later ensure() calls can reuse memory.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @return None (void).
    template <KernelScalarConcept T> void DistanceCache<T>::reset() noexcept
    {
        // Reset logical dimensions and clear vectors without deallocating capacity.
        num_points_ = 0;
        num_dims_ = 0;
        dist_.clear();
        ard_sq_diff_.clear();
    }

    /// @brief Returns an immutable column-major view of the n x n squared-distance matrix.
    /// @details Returns an empty view when the distance matrix is unpopulated.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @return Immutable column-major n x n view or empty MatrixRef<T>.
    template <KernelScalarConcept T> MatrixRef<T> DistanceCache<T>::dist_view() const noexcept
    {
        if (dist_.empty())
        {
            return {};
        }
        return {dist_.data(), num_points_, num_points_, num_points_};
    }

    /// @brief Returns a mutable column-major view of the n x n squared-distance matrix.
    /// @details Returns an empty view when the distance matrix is unpopulated.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @return Mutable column-major n x n view or empty MatrixMut<T>.
    template <KernelScalarConcept T> MatrixMut<T> DistanceCache<T>::dist_mut() noexcept
    {
        if (dist_.empty())
        {
            return {};
        }
        return {dist_.data(), num_points_, num_points_, num_points_};
    }

    template class DistanceCache<double>;
    template class DistanceCache<float>;

}  // namespace cxxgpr
