// SPDX-License-Identifier: Apache-2.0
/// @file parallel.cpp
/// @brief Implementations of area-balanced OpenMP lower-triangle partitioning and deterministic column reductions.
/// @details Uses exact 128-bit integer arithmetic for overflow-free lower-triangular partitioning and a serial Neumaier
/// reduction over per-column partial results for thread-count determinism.

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

        /// @brief Unsigned 128-bit integer pair used for exact partition target comparisons.
        /// @details Prevents 64-bit overflow when multiplying lower_area(n, n) by block counts.
        struct Uint128
        {
            /// @brief Upper 64 bits of the 128-bit unsigned integer.
            std::uint64_t hi = 0;
            /// @brief Lower 64 bits of the 128-bit unsigned integer.
            std::uint64_t lo = 0;
        };

        /// @brief Computes the exact 128-bit product of two 64-bit unsigned integers.
        /// @details Splits each operand into 32-bit halves and propagates carries from low to high words.
        /// @param[in] a First 64-bit unsigned multiplicand.
        /// @param[in] b Second 64-bit unsigned multiplicand.
        /// @return Exact 128-bit unsigned product a * b.
        [[nodiscard]] constexpr Uint128 mul_u64_u128(std::uint64_t a, std::uint64_t b) noexcept
        {
            // Decompose into 32-bit limbs so every partial product fits cleanly within uint64_t.
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

        /// @brief Compares two 128-bit unsigned integers for a >= b.
        /// @details Evaluates lexicographic order on (hi, lo).
        /// @param[in] a Left-hand 128-bit unsigned integer.
        /// @param[in] b Right-hand 128-bit unsigned integer.
        /// @return True if a >= b.
        [[nodiscard]] constexpr bool u128_ge(Uint128 a, Uint128 b) noexcept
        {
            return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
        }

        /// @brief Compares two 128-bit unsigned integers for a < b.
        /// @details Evaluates lexicographic order on (hi, lo).
        /// @param[in] a Left-hand 128-bit unsigned integer.
        /// @param[in] b Right-hand 128-bit unsigned integer.
        /// @return True if a < b.
        [[nodiscard]] constexpr bool u128_lt(Uint128 a, Uint128 b) noexcept
        {
            return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
        }

        /// @brief Subtracts b from a assuming a >= b in 128-bit unsigned arithmetic.
        /// @details Propagates a single borrow bit from the low 64-bit word to the high 64-bit word.
        /// @param[in] a Minuend (must satisfy a >= b).
        /// @param[in] b Subtrahend.
        /// @return Difference a - b.
        [[nodiscard]] constexpr Uint128 u128_sub(Uint128 a, Uint128 b) noexcept
        {
            const std::uint64_t borrow = (a.lo < b.lo) ? 1ULL : 0ULL;
            return {.hi = a.hi - b.hi - borrow, .lo = a.lo - b.lo};
        }

        /// @brief Computes lower-triangular entry count in columns [0, min(col, n)) in 64-bit arithmetic.
        /// @details Evaluates c * n - c * (c - 1) / 2 where c = min(col, n).
        /// @param[in] n Matrix order.
        /// @param[in] col Exclusive ending column index.
        /// @return Number of lower-triangular entries in columns [0, min(col, n)).
        [[nodiscard]] constexpr std::uint64_t lower_area_u64(std::uint64_t n, std::uint64_t col) noexcept
        {
            const std::uint64_t c = std::min(col, n);
            const std::uint64_t tri = (c > 0) ? (c * (c - 1)) / 2 : 0;
            return (c * n) - tri;
        }

        /// @brief Resolves requested thread count, falling back to openmp_max_threads() when zero.
        /// @details Leaves positive thread counts unchanged.
        /// @param[in] num_threads Requested thread count, or 0 for automatic OpenMP query.
        /// @return Effective positive thread count.
        [[nodiscard]] std::size_t resolve_threads(std::size_t num_threads) noexcept
        {
            if (num_threads == 0)
            {
                return openmp_max_threads();
            }
            return num_threads;
        }

    }  // namespace

    /// @brief Returns the count of lower-triangular entries (including diagonal) in columns [0, col) of an n x n
    /// matrix.
    /// @details Delegates to lower_area_u64 with col clamped to n.
    /// @param[in] n Matrix order (number of rows and columns).
    /// @param[in] col Exclusive ending column index.
    /// @return Number of lower-triangular entries covered by columns [0, min(col, n)).
    std::size_t lower_area(std::size_t n, std::size_t col) noexcept
    {
        return static_cast<std::size_t>(lower_area_u64(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(col)));
    }

    /// @brief Returns the starting column index of an area-balanced partition block in an n x n lower triangle.
    /// @details Uses binary search with exact 128-bit cross-multiplication to locate the column whose cumulative area
    /// best approximates block_idx / n_blocks of the total lower triangle.
    /// @param[in] n Matrix order (number of rows and columns).
    /// @param[in] block_idx Zero-based block index in [0, n_blocks].
    /// @param[in] n_blocks Total number of partition blocks (clamped to at least 1).
    /// @return Starting column index in [0, n].
    std::size_t lower_block_start(std::size_t n, std::size_t block_idx, std::size_t n_blocks) noexcept
    {
        const std::size_t blocks = std::max<std::size_t>(n_blocks, 1);
        if (block_idx >= blocks || n == 0)
        {
            return n;
        }

        // Cross-multiply target = block_idx * area(n, n) vs area(n, mid) * blocks in 128 bits to avoid overflow.
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

        // Pick whichever of lo - 1 or lo is strictly closer to the fractional area target.
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

    /// @brief Returns the maximum number of OpenMP threads available for execution.
    /// @details Queries omp_get_max_threads() when compiled with _OPENMP, clamping to at least 1.
    /// @return Positive thread count (always at least 1).
    std::size_t openmp_max_threads() noexcept
    {
#ifdef _OPENMP
        const int threads = omp_get_max_threads();
        return threads > 1 ? static_cast<std::size_t>(threads) : 1;
#else
        return 1;
#endif
    }

    /// @brief Executes fn(col_begin, col_end, ctx) across area-balanced column blocks covering [0, n).
    /// @details Uses OpenMP static-1 scheduling over area-balanced lower-triangle column intervals when OpenMP is
    /// active and more than one block is used; otherwise executes serially on [0, n).
    /// @param[in] n Matrix order specifying column range [0, n).
    /// @param[in] num_threads Requested thread count, or 0 to query openmp_max_threads().
    /// @param[in] fn Callback invoked once per non-empty column block [col_begin, col_end).
    /// @param[in,out] ctx Opaque context pointer forwarded unchanged to fn.
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
        // Cap blocks at min(effective_threads, n) so every thread gets an area-balanced column slice.
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

        // Single-threaded or non-OpenMP fallback executes the entire range [0, n) sequentially.
        fn(0, n, ctx);
    }

    /// @brief Evaluates fn(col, ctx) in parallel into col_partials and reduces deterministically in ascending order.
    /// @details Validates inputs, computes independent per-column partials concurrently across area-balanced blocks,
    /// and sums col_partials[0..n) serially with Neumaier compensation for bit-identical reproducibility.
    /// @param[in] n Number of columns to evaluate and reduce.
    /// @param[in] num_threads Requested thread count, or 0 to query openmp_max_threads().
    /// @param[in,out] col_partials Preallocated workspace of length n storing per-column partial results.
    /// @param[in] fn Callback returning the partial double contribution for column col.
    /// @param[in,out] ctx Opaque context pointer forwarded unchanged to fn.
    /// @return Deterministically reduced sum in double precision, or GprError on invalid arguments.
    /// @retval GprErrorCode::EmptyInput When n == 0 or col_partials is empty.
    /// @retval GprErrorCode::LengthMismatch When col_partials.size() != n.
    /// @retval GprErrorCode::InvalidConfig When fn is null.
    std::expected<double, GprError> parallel_lower_reduce_cols_raw(
        std::size_t n,
        std::size_t num_threads,
        std::span<double> col_partials,
        double (*fn)(std::size_t col, void* ctx),
        void* ctx
    )
    {
        // Enforce validation order: empty inputs -> partial buffer length -> callback non-null.
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

        // Local context passed through parallel_for_lower_blocks_raw to write disjoint column slots.
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

        // Reduce col_partials serially in ascending column order with Neumaier compensation so
        // the result is bit-identical across any thread count or OpenMP scheduling.
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
