// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/rng.hpp"

#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// Spec: REQ-11, REQ-12, DES-20
namespace {

// Pinned first 4 outputs for seeds 0 and 42 matching gprx/src/rng.rs (rand_xoshiro 0.7).
constexpr std::array<std::uint64_t, 4> expected_seed_0{
    0x53175d61490b23dfULL,
    0x61da6f3dc380d507ULL,
    0x5c0fdf91ec9a7bfcULL,
    0x02eebf8c3bbe5e1aULL,
};

constexpr std::array<std::uint64_t, 4> expected_seed_42{
    0xd0764d4f4476689fULL,
    0x519e4174576f3791ULL,
    0xfbe07cfb0c24ed8cULL,
    0xb37d9f600cd835b8ULL,
};

TEST(RngTest, MatchesPinnedReferenceGoldenSequencesBitForBit) {
    cxxgpr::Xoshiro256PlusPlus rng0(0);
    for (std::size_t i = 0; i < expected_seed_0.size(); ++i) {
        EXPECT_EQ(rng0.next_u64(), expected_seed_0.at(i)) << "mismatch at index " << i;
    }

    cxxgpr::SeededRng rng42(42);
    for (std::size_t i = 0; i < expected_seed_42.size(); ++i) {
        EXPECT_EQ(rng42.next_u64(), expected_seed_42.at(i)) << "mismatch at index " << i;
    }

    // Same seed reproduces bit-identical state and variates; different seed diverges.
    cxxgpr::SeededRng copy0(0);
    cxxgpr::SeededRng fresh0(0);
    for (std::size_t i = 0; i < expected_seed_0.size(); ++i) {
        EXPECT_EQ(copy0.next_u64(), fresh0.next_u64());
    }
    EXPECT_EQ(copy0, rng0);
    EXPECT_NE(rng0, rng42);
}

TEST(RngTest, UnitOpenUnitUpToAndNormalStayInContractDomains) {
    cxxgpr::Xoshiro256PlusPlus rng(7);
    constexpr double eps = 1.0 / static_cast<double>(1ULL << 53U);

    for (const std::size_t bound :
         {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{9}, std::size_t{1000}}) {
        for (int i = 0; i < 200; ++i) {
            const std::size_t v = rng.up_to(bound);
            if (bound <= 1) {
                EXPECT_EQ(v, 0U);
            } else {
                EXPECT_LT(v, bound);
            }
        }
    }

    for (int i = 0; i < 2000; ++i) {
        const double u = rng.unit();
        EXPECT_GE(u, 0.0);
        EXPECT_LT(u, 1.0);

        const double ou = rng.open_unit();
        EXPECT_GE(ou, eps);
        EXPECT_LE(ou, 1.0 - eps);
        EXPECT_GT(ou, 0.0);
        EXPECT_LT(ou, 1.0);
    }

    // Statistical sanity check on deterministic Box-Muller variates.
    constexpr int n_samples = 10000;
    cxxgpr::Xoshiro256PlusPlus norm_rng(12345);
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        const double z = norm_rng.unit_normal();
        ASSERT_TRUE(std::isfinite(z));
        sum += z;
        sum_sq += z * z;
    }
    const double mean = sum / static_cast<double>(n_samples);
    const double var = (sum_sq / static_cast<double>(n_samples)) - (mean * mean);
    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(var, 1.0, 0.08);
}

TEST(RngTest, GenerationAllocatesZero) {
    cxxgpr::Xoshiro256PlusPlus rng(99);
    std::uint64_t word = 0;
    double acc = 0.0;
    std::size_t idx = 0;
    std::size_t delta = 1;

    {
        cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
        for (int i = 0; i < 64; ++i) {
            word ^= rng.next_u64();
            acc += rng.unit() + rng.open_unit() + rng.unit_normal();
            idx += rng.up_to(17);
        }
        cxxgpr::test_support::touch_for_test(&word);
        cxxgpr::test_support::touch_for_test(&acc);
        cxxgpr::test_support::touch_for_test(&idx);
        delta = scope.delta_allocations();
    }

    EXPECT_EQ(delta, 0U);
    EXPECT_TRUE(std::isfinite(acc));
}

} // namespace
