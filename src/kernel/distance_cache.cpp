// SPDX-License-Identifier: Apache-2.0
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

        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_cache_points(MatrixRef<T> x)
        {
            if (x.is_empty() || x.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
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

        [[nodiscard]] std::expected<std::size_t, GprError> checked_packed_ard_len(std::size_t n, std::size_t d)
        {
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

        template <KernelScalarConcept T> void fill_square_dist_matrix(MatrixRef<T> x, std::span<T> dst) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            MatrixMut<T> dist(dst.data(), n, n, n);
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
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
                    dist(r1, r2) = acc;
                    dist(r2, r1) = acc;
                }
            }
        }

        template <KernelScalarConcept T> void fill_packed_ard_sq_diff(MatrixRef<T> x, std::span<T> dst) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            const std::size_t per_dim = (n * (n + 1U)) / 2U;
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
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

    template <KernelScalarConcept T>
    std::expected<void, GprError> DistanceCache<T>::ensure(
        MatrixRef<T> x, bool cache_dist, bool cache_ard, DistanceCachePolicy policy
    )
    {
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

    template <KernelScalarConcept T> void DistanceCache<T>::reset() noexcept
    {
        num_points_ = 0;
        num_dims_ = 0;
        dist_.clear();
        ard_sq_diff_.clear();
    }

    template <KernelScalarConcept T> MatrixRef<T> DistanceCache<T>::dist_view() const noexcept
    {
        if (dist_.empty())
        {
            return {};
        }
        return {dist_.data(), num_points_, num_points_, num_points_};
    }

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
