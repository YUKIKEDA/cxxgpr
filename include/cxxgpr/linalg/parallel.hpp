// SPDX-License-Identifier: Apache-2.0
/// @file parallel.hpp
/// @brief Area-balanced lower-triangular OpenMP partitioning and deterministic column reductions.
/// @details Provides triangular column partitioning helpers and zero-allocation parallel dispatch primitives whose
/// reductions remain bit-identical across thread counts.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>

/// @brief Dense and factorized linear algebra primitives for cxxgpr.
namespace cxxgpr::linalg
{

    /// @brief Returns the count of lower-triangular entries (including diagonal) in columns [0, col) of an n x n
    /// matrix.
    /// @details Computes col * n - col * (col - 1) / 2 with col clamped to n.
    /// @param n Matrix order (number of rows and columns).
    /// @param col Exclusive ending column index.
    /// @return Number of lower-triangular entries covered by columns [0, min(col, n)).
    [[nodiscard]] std::size_t lower_area(std::size_t n, std::size_t col) noexcept;

    /// @brief Returns the starting column index of an area-balanced partition block in an n x n lower triangle.
    /// @details Searches for the column boundary whose cumulative lower-triangular area is closest to block_idx /
    /// n_blocks of the total triangle area.
    /// @param n Matrix order (number of rows and columns).
    /// @param block_idx Zero-based block index in [0, n_blocks].
    /// @param n_blocks Total number of partition blocks (clamped to at least 1).
    /// @return Starting column index in [0, n].
    [[nodiscard]] std::size_t lower_block_start(std::size_t n, std::size_t block_idx, std::size_t n_blocks) noexcept;

    /// @brief Returns the maximum number of OpenMP threads available for execution.
    /// @details Queries omp_get_max_threads() when OpenMP is enabled and returns 1 otherwise.
    /// @return Positive thread count (always at least 1).
    [[nodiscard]] std::size_t openmp_max_threads() noexcept;

    /// @brief Executes fn(col_begin, col_end, ctx) across area-balanced column blocks covering [0, n).
    /// @details Partitioning balances lower-triangular work across threads while keeping each column exclusive to one
    /// block; passing num_threads == 0 selects openmp_max_threads().
    /// @param n Matrix order specifying column range [0, n).
    /// @param num_threads Requested thread count, or 0 to query openmp_max_threads().
    /// @param fn Callback invoked once per non-empty column block [col_begin, col_end).
    /// @param ctx Opaque context pointer forwarded unchanged to fn.
    void parallel_for_lower_blocks_raw(
        std::size_t n,
        std::size_t num_threads,
        void (*fn)(std::size_t col_begin, std::size_t col_end, void* ctx),
        void* ctx
    );

    /// @brief Executes fn(col_begin, col_end) across area-balanced column blocks covering [0, n).
    /// @details Forwards the callable through a stateless trampoline pointer to parallel_for_lower_blocks_raw with zero
    /// heap allocations.
    /// @tparam Fn Callable type invocable as fn(std::size_t col_begin, std::size_t col_end).
    /// @param n Matrix order specifying column range [0, n).
    /// @param num_threads Requested thread count, or 0 to query openmp_max_threads().
    /// @param fn Callable invoked once per non-empty column block [col_begin, col_end).
    template <typename Fn> void parallel_for_lower_blocks(std::size_t n, std::size_t num_threads, Fn&& fn)
    {
        // Forward Fn by raw pointer through a captureless lambda to avoid std::function heap allocations.
        auto* fn_ptr = std::addressof(fn);
        using FnPtr = decltype(fn_ptr);
        parallel_for_lower_blocks_raw(n, num_threads, [](std::size_t col_begin, std::size_t col_end, void* ctx) {
            (*static_cast<FnPtr>(ctx))(col_begin, col_end);
        }, fn_ptr);
    }

    /// @brief Evaluates fn(col, ctx) in parallel into col_partials and reduces deterministically in ascending order.
    /// @details Computes each column scalar independently across area-balanced OpenMP blocks into col_partials[col],
    /// then performs a serial Neumaier compensated sum over columns 0..n-1 for bit-identical determinism.
    /// @param n Number of columns to evaluate and reduce.
    /// @param num_threads Requested thread count, or 0 to query openmp_max_threads().
    /// @param col_partials Preallocated workspace of length n storing per-column partial results.
    /// @param fn Callback returning the partial double contribution for column col.
    /// @param ctx Opaque context pointer forwarded unchanged to fn.
    /// @return Deterministically reduced sum in double precision, or GprError on invalid arguments.
    /// @retval GprErrorCode::EmptyInput When n == 0 or col_partials is empty.
    /// @retval GprErrorCode::LengthMismatch When col_partials.size() != n.
    /// @retval GprErrorCode::InvalidConfig When fn is null.
    [[nodiscard]] std::expected<double, GprError> parallel_lower_reduce_cols_raw(
        std::size_t n,
        std::size_t num_threads,
        std::span<double> col_partials,
        double (*fn)(std::size_t col, void* ctx),
        void* ctx
    );

    /// @brief Evaluates fn(col) in parallel into col_partials and reduces deterministically in ascending order.
    /// @details Wraps fn in a stateless trampoline and delegates to parallel_lower_reduce_cols_raw with zero heap
    /// allocations.
    /// @tparam Fn Callable type returning double when invoked as fn(std::size_t col).
    /// @param n Number of columns to evaluate and reduce.
    /// @param num_threads Requested thread count, or 0 to query openmp_max_threads().
    /// @param col_partials Preallocated workspace of length n storing per-column partial results.
    /// @param fn Callable returning the partial double contribution for column col.
    /// @return Deterministically reduced sum in double precision, or GprError on invalid arguments.
    /// @retval GprErrorCode::EmptyInput When n == 0 or col_partials is empty.
    /// @retval GprErrorCode::LengthMismatch When col_partials.size() != n.
    template <typename Fn>
    [[nodiscard]] std::expected<double, GprError> parallel_lower_reduce_cols(
        std::size_t n, std::size_t num_threads, std::span<double> col_partials, Fn&& fn
    )
    {
        // Forward Fn by raw pointer through a captureless lambda to avoid std::function heap allocations.
        auto* fn_ptr = std::addressof(fn);
        using FnPtr = decltype(fn_ptr);
        return parallel_lower_reduce_cols_raw(n, num_threads, col_partials, [](std::size_t col, void* ctx) -> double {
            return (*static_cast<FnPtr>(ctx))(col);
        }, fn_ptr);
    }

}  // namespace cxxgpr::linalg
