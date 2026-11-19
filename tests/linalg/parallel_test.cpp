// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-11, DES-02, DES-20
/// @file parallel_test.cpp
/// @brief Unit and property tests for deterministic OpenMP lower-triangular parallel helpers.
/// @details Verifies balanced lower-triangular column partitioning, single-visit parallel_for
/// coverage, bitwise-identical thread-count invariance in parallel_lower_reduce_cols(), zero heap
/// allocations on hot loops, and negative validation checks.

#include "cxxgpr/linalg/parallel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Brings GprErrorCode into the anonymous test namespace.
    using cxxgpr::GprErrorCode;
    /// @brief Brings allocation AuditMode enum into the anonymous test namespace.
    using cxxgpr::test_support::AuditMode;
    /// @brief Brings RAII NoAllocationScope guard into the anonymous test namespace.
    using cxxgpr::test_support::NoAllocationScope;

    /// @brief Verifies lower_block_start() and lower_area() partition columns monotonically with no gaps.
    /// @details Sweeps matrix sizes n and worker counts k to ensure block boundaries stay monotonic from
    /// 0 to n and lower_area(10, 10) equals 10 * 11 / 2 = 55.
    TEST(ParallelLinalgTest, LowerBlockStartPartitionsColumnsWithoutGapsOrOverlaps)
    {
        // Sweep degenerate and prime matrix dimensions across single and multi-thread counts.
        for (const std::size_t n : {0U, 1U, 2U, 7U, 64U, 127U, 256U})
        {
            for (const std::size_t k : {1U, 2U, 3U, 4U, 8U, 16U})
            {
                EXPECT_EQ(cxxgpr::linalg::lower_block_start(n, 0, k), 0U);
                EXPECT_EQ(cxxgpr::linalg::lower_block_start(n, k, k), n);
                std::size_t prev = 0;
                for (std::size_t b = 0; b <= k; ++b)
                {
                    const std::size_t cur = cxxgpr::linalg::lower_block_start(n, b, k);
                    EXPECT_GE(cur, prev);
                    EXPECT_LE(cur, n);
                    prev = cur;
                }
            }
        }
        EXPECT_EQ(cxxgpr::linalg::lower_area(10, 0), 0U);
        EXPECT_EQ(cxxgpr::linalg::lower_area(10, 10), 55U);
    }

    /// @brief Verifies parallel_for_lower_blocks() visits every column exactly once with zero allocations.
    /// @details Executes across thread counts {1, 2, 4, 8} inside AuditMode::CppAndEigen
    /// NoAllocationScope and asserts each column in [0, 129) is visited once.
    TEST(ParallelLinalgTest, ParallelForLowerBlocksVisitsEachColumnOnceWithZeroAllocations)
    {
        constexpr std::size_t n = 129;
        for (const std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}})
        {
            std::vector<int> visits(n, 0);
            // Audit the parallel loop body for zero dynamic allocations across worker dispatches.
            {
                const NoAllocationScope audit(AuditMode::CppAndEigen);
                cxxgpr::linalg::parallel_for_lower_blocks(
                    n, threads, [&](std::size_t col_begin, std::size_t col_end)
                {
                    for (std::size_t col = col_begin; col < col_end; ++col)
                    {
                        visits[col] += 1;
                    }
                }
                );
                EXPECT_EQ(audit.delta_allocations(), 0U);
            }
            for (std::size_t col = 0; col < n; ++col)
            {
                EXPECT_EQ(visits[col], 1) << "col=" << col << " threads=" << threads;
            }
        }
    }

    /// @brief Verifies parallel_lower_reduce_cols() yields bit-identical totals across thread counts.
    /// @details Accumulates an alternating non-associative series over n = 257 lower-triangular columns
    /// and compares std::bit_cast<uint64_t> sums across 2, 3, 4, and 8 threads against 1 thread.
    TEST(ParallelLinalgTest, ParallelLowerReduceColsIsBitIdenticalAcrossThreadCounts)
    {
        constexpr std::size_t n = 257;
        std::vector<double> partials(n, 0.0);

        // Alternating signs magnify floating-point summation order differences across workers.
        auto eval_col = [](std::size_t col) -> double
        {
            double col_sum = 0.0;
            for (std::size_t row = col; row < n; ++row)
            {
                const double sign = ((row + col) % 2 == 0) ? 1.0 : -1.0;
                col_sum += sign / (1.0 + static_cast<double>((row * 17) + (col * 31)));
            }
            return col_sum;
        };

        // Capture the single-thread baseline bit pattern, then assert multi-thread runs match bits.
        const auto seq_res = cxxgpr::linalg::parallel_lower_reduce_cols(n, 1, partials, eval_col);
        ASSERT_TRUE(seq_res.has_value());
        const auto seq_bits = std::bit_cast<std::uint64_t>(*seq_res);

        for (const std::size_t threads : {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{8}})
        {
            std::ranges::fill(partials, 0.0);
            const NoAllocationScope audit(AuditMode::CppAndEigen);
            const auto par_res = cxxgpr::linalg::parallel_lower_reduce_cols(n, threads, partials, eval_col);
            ASSERT_TRUE(par_res.has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(*par_res), seq_bits) << "threads=" << threads;
        }
    }

    /// @brief Negative controls verifying parallel reduction checks empty, mismatched, and null callbacks.
    /// @details Confirms EmptyInput when n == 0, LengthMismatch when partials.size() != n, and
    /// InvalidConfig when a null raw callback pointer is passed to parallel_lower_reduce_cols_raw().
    TEST(ParallelLinalgTest, NegativeValidation)
    {
        std::vector<double> empty_vec;
        auto dummy = [](std::size_t col) -> double
        {
            return static_cast<double>(col);
        };
        // Reject zero dimension before checking partial buffer size.
        EXPECT_EQ(
            cxxgpr::linalg::parallel_lower_reduce_cols(0, 1, empty_vec, dummy).error().code(), GprErrorCode::EmptyInput
        );

        std::vector<double> wrong_len(3, 0.0);
        EXPECT_EQ(
            cxxgpr::linalg::parallel_lower_reduce_cols(4, 1, wrong_len, dummy).error().code(),
            GprErrorCode::LengthMismatch
        );

        // Raw reduction entry point must reject null function pointers with InvalidConfig.
        std::vector<double> ok_len(4, 0.0);
        EXPECT_EQ(
            cxxgpr::linalg::parallel_lower_reduce_cols_raw(4, 1, ok_len, nullptr, nullptr).error().code(),
            GprErrorCode::InvalidConfig
        );
    }

}  // namespace
