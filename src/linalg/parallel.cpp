// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/linalg/parallel.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace cxxgpr::linalg
{

    namespace
    {

        struct Uint128
        {
            std::uint64_t hi = 0;
            std::uint64_t lo = 0;
        };

        [[nodiscard]] constexpr Uint128 mul_u64_u128(std::uint64_t a, std::uint64_t b) noexcept
        {
            const std::uint64_t a_lo = a & 0xffffffffULL;
            const std::uint64_t a_hi = a >> 32U;
            const std::uint64_t b_lo = b & 0xffffffffULL;
            const std::uint64_t b_hi = b >> 32U;

            const std::uint64_t p00 = a_lo * b_lo;
            const std::uint64_t p01 = a_lo * b_hi;
            const std::uint64_t p10 = a_hi * b_lo;
            const std::uint64_t p11 = a_hi * b_hi;

            const std::uint64_t mid = (p00 >> 32U) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
            const std::uint64_t lo = (p00 & 0xffffffffULL) | (mid << 32U);
            const std::uint64_t hi = p11 + (p01 >> 32U) + (p10 >> 32U) + (mid >> 32U);
            return {.hi = hi, .lo = lo};
        }

        [[nodiscard]] constexpr bool u128_ge(Uint128 a, Uint128 b) noexcept
        {
            return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
        }

        [[nodiscard]] constexpr bool u128_lt(Uint128 a, Uint128 b) noexcept
        {
            return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
        }

        [[nodiscard]] constexpr Uint128 u128_sub(Uint128 a, Uint128 b) noexcept
        {
            const std::uint64_t borrow = (a.lo < b.lo) ? 1ULL : 0ULL;
            return {.hi = a.hi - b.hi - borrow, .lo = a.lo - b.lo};
        }

        [[nodiscard]] constexpr std::uint64_t lower_area_u64(std::uint64_t n, std::uint64_t col) noexcept
        {
            const std::uint64_t c = std::min(col, n);
            const std::uint64_t tri = (c > 0) ? (c * (c - 1)) / 2 : 0;
            return (c * n) - tri;
        }

        [[nodiscard]] std::size_t resolve_threads(std::size_t num_threads) noexcept
        {
            if (num_threads == 0)
            {
                return openmp_max_threads();
            }
            return num_threads;
        }

    }  // namespace

    std::size_t lower_area(std::size_t n, std::size_t col) noexcept
    {
        return static_cast<std::size_t>(lower_area_u64(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(col)));
    }

    std::size_t lower_block_start(std::size_t n, std::size_t block_idx, std::size_t n_blocks) noexcept
    {
        const std::size_t blocks = std::max<std::size_t>(n_blocks, 1);
        if (block_idx >= blocks || n == 0)
        {
            return n;
        }

        const auto n64 = static_cast<std::uint64_t>(n);
        const auto k64 = static_cast<std::uint64_t>(blocks);
        const Uint128 target = mul_u64_u128(static_cast<std::uint64_t>(block_idx), lower_area_u64(n64, n64));

        std::size_t lo = 0;
        std::size_t hi = n;
        while (lo < hi)
        {
            const std::size_t mid = lo + ((hi - lo) / 2);
            const Uint128 lhs = mul_u64_u128(lower_area_u64(n64, static_cast<std::uint64_t>(mid)), k64);
            if (u128_ge(lhs, target))
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo > 0)
        {
            const Uint128 prev_scaled = mul_u64_u128(lower_area_u64(n64, static_cast<std::uint64_t>(lo - 1)), k64);
            const Uint128 curr_scaled = mul_u64_u128(lower_area_u64(n64, static_cast<std::uint64_t>(lo)), k64);
            const Uint128 dist_prev = u128_sub(target, prev_scaled);
            const Uint128 dist_curr = u128_sub(curr_scaled, target);
            if (u128_lt(dist_prev, dist_curr))
            {
                return lo - 1;
            }
        }
        return lo;
    }

    std::size_t openmp_max_threads() noexcept
    {
#ifdef _OPENMP
        const int threads = omp_get_max_threads();
        return threads > 1 ? static_cast<std::size_t>(threads) : 1;
#else
        return 1;
#endif
    }

    void parallel_for_lower_blocks_raw(
        std::size_t n,
        std::size_t num_threads,
        void (*fn)(std::size_t col_begin, std::size_t col_end, void* ctx),
        void* ctx
    )
    {
        if (n == 0 || fn == nullptr)
        {
            return;
        }
#ifdef _OPENMP
        const std::size_t effective_threads = resolve_threads(num_threads);
        const std::size_t n_blocks = std::min(effective_threads, n);
        if (n_blocks > 1)
        {
            const int blocks_i = static_cast<int>(
                std::min<std::size_t>(n_blocks, static_cast<std::size_t>(std::numeric_limits<int>::max()))
            );
#    pragma omp parallel for schedule(static, 1) num_threads(blocks_i)
            for (int b = 0; b < blocks_i; ++b)
            {
                const auto ub = static_cast<std::size_t>(b);
                const std::size_t col_begin = lower_block_start(n, ub, n_blocks);
                const std::size_t col_end = lower_block_start(n, ub + 1, n_blocks);
                if (col_begin < col_end)
                {
                    fn(col_begin, col_end, ctx);
                }
            }
            return;
        }
#else
        (void)resolve_threads(num_threads);
#endif

        fn(0, n, ctx);
    }

    std::expected<double, GprError> parallel_lower_reduce_cols_raw(
        std::size_t n,
        std::size_t num_threads,
        std::span<double> col_partials,
        double (*fn)(std::size_t col, void* ctx),
        void* ctx
    )
    {
        if (n == 0 || col_partials.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (const auto cnt = require_count(col_partials.size(), n, "col_partials length"); !cnt.has_value())
        {
            return std::unexpected(cnt.error());
        }
        if (fn == nullptr)
        {
            return std::unexpected(GprError::invalid_config("column callback must not be null"));
        }

        struct ReduceCtx
        {
            double (*cb)(std::size_t col, void* ctx) = nullptr;
            void* user_ctx = nullptr;
            double* partials = nullptr;
        };

        ReduceCtx reduce_ctx{.cb = fn, .user_ctx = ctx, .partials = col_partials.data()};

        parallel_for_lower_blocks_raw(
            n, num_threads, [](std::size_t col_begin, std::size_t col_end, void* raw_ctx)
        {
            const auto* rctx = static_cast<const ReduceCtx*>(raw_ctx);
            for (std::size_t col = col_begin; col < col_end; ++col)
            {
                rctx->partials[col] = rctx->cb(col, rctx->user_ctx);
            }
        }, &reduce_ctx
        );

        double sum = 0.0;
        double comp = 0.0;
        for (std::size_t col = 0; col < n; ++col)
        {
            const double term = col_partials[col];
            const double t = sum + term;
            if (std::abs(sum) >= std::abs(term))
            {
                comp += (sum - t) + term;
            }
            else
            {
                comp += (term - t) + sum;
            }
            sum = t;
        }
        return sum + comp;
    }

}  // namespace cxxgpr::linalg
