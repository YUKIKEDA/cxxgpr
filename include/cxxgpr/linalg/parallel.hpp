// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>

namespace cxxgpr::linalg
{

    /// Returns the number of lower-triangular entries (including diagonal) in columns [0, col) of
    /// an n x n matrix: col * n - col * (col - 1) / 2.
    [[nodiscard]] std::size_t lower_area(std::size_t n, std::size_t col) noexcept;

    /// Returns the starting column index of block idx out of n_blocks that partition the lower
    /// triangle of an n x n matrix into roughly equal entry counts.
    [[nodiscard]] std::size_t lower_block_start(std::size_t n, std::size_t block_idx, std::size_t n_blocks) noexcept;

    /// Returns the maximum number of OpenMP threads available, or 1 if OpenMP is disabled.
    [[nodiscard]] std::size_t openmp_max_threads() noexcept;

    /// Executes fn(col_begin, col_end, ctx) across area-balanced column blocks covering [0, n).
    /// Passing num_threads == 0 selects openmp_max_threads().
    void parallel_for_lower_blocks_raw(
        std::size_t n,
        std::size_t num_threads,
        void (*fn)(std::size_t col_begin, std::size_t col_end, void* ctx),
        void* ctx
    );

    /// Executes fn(col_begin, col_end) across area-balanced column blocks covering [0, n).
    /// Passing num_threads == 0 selects openmp_max_threads().
    template <typename Fn> void parallel_for_lower_blocks(std::size_t n, std::size_t num_threads, Fn&& fn)
    {
        auto* fn_ptr = std::addressof(fn);
        using FnPtr = decltype(fn_ptr);
        parallel_for_lower_blocks_raw(n, num_threads, [](std::size_t col_begin, std::size_t col_end, void* ctx) {
            (*static_cast<FnPtr>(ctx))(col_begin, col_end);
        }, fn_ptr);
    }

    /// Evaluates fn(col, ctx) in parallel across [0, n) into col_partials, then reduces
    /// col_partials[0..n) serially in ascending column order for bit-identical determinism.
    [[nodiscard]] std::expected<double, GprError> parallel_lower_reduce_cols_raw(
        std::size_t n,
        std::size_t num_threads,
        std::span<double> col_partials,
        double (*fn)(std::size_t col, void* ctx),
        void* ctx
    );

    /// Evaluates fn(col) in parallel across [0, n) into col_partials, then reduces
    /// col_partials[0..n) serially in ascending column order for bit-identical determinism.
    template <typename Fn>
    [[nodiscard]] std::expected<double, GprError> parallel_lower_reduce_cols(
        std::size_t n, std::size_t num_threads, std::span<double> col_partials, Fn&& fn
    )
    {
        auto* fn_ptr = std::addressof(fn);
        using FnPtr = decltype(fn_ptr);
        return parallel_lower_reduce_cols_raw(n, num_threads, col_partials, [](std::size_t col, void* ctx) -> double {
            return (*static_cast<FnPtr>(ctx))(col);
        }, fn_ptr);
    }

}  // namespace cxxgpr::linalg
