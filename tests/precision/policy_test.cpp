// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-07, REQ-11, DES-01, DES-02, DES-05, FMT-03
/// @file policy_test.cpp
/// @brief Unit and compile-time tests for precision policies, residual formulas, and storage helpers.
/// @details Verifies precision and residual concepts via static_assert, persistence string tags,
/// zero-allocation copy_storage_to_refine(), double-precision strided compute_storage_means(), and
/// validation error precedence.

#include "cxxgpr/precision/policy.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Negative-control stub intentionally omitting ResidualFormulaConcept members.
    /// @details Used in static_assert checks to verify ResidualFormulaConcept rejects empty structs.
    struct BogusResidual
    {
    };

    /// @brief Negative-control stub intentionally omitting PrecisionPolicyConcept and GpScalarConcept members.
    /// @details Used in static_assert checks to verify concept constraints reject non-policy types.
    struct BogusPolicy
    {
    };

    // Compile-time checks verifying valid policies and negative-control concept rejections.
    static_assert(cxxgpr::ResidualFormulaConcept<cxxgpr::PromoteStorage>);
    static_assert(cxxgpr::ResidualFormulaConcept<cxxgpr::ReevaluateKernel>);
    static_assert(!cxxgpr::ResidualFormulaConcept<BogusResidual>);
    static_assert(!cxxgpr::ResidualFormulaConcept<int>);

    static_assert(cxxgpr::KernelScalarConcept<float>);
    static_assert(cxxgpr::KernelScalarConcept<double>);
    static_assert(!cxxgpr::KernelScalarConcept<long double>);
    static_assert(!cxxgpr::KernelScalarConcept<int>);

    static_assert(cxxgpr::PrecisionPolicyConcept<cxxgpr::DoublePrecision>);
    static_assert(cxxgpr::PrecisionPolicyConcept<cxxgpr::SinglePrecision>);
    static_assert(cxxgpr::PrecisionPolicyConcept<cxxgpr::MixedPrecision<>>);
    static_assert(cxxgpr::PrecisionPolicyConcept<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>);
    static_assert(cxxgpr::PrecisionPolicyConcept<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>);
    static_assert(!cxxgpr::PrecisionPolicyConcept<BogusPolicy>);
    static_assert(!cxxgpr::PrecisionPolicyConcept<cxxgpr::PromoteStorage>);
    static_assert(!cxxgpr::PrecisionPolicyConcept<double>);
    static_assert(cxxgpr::GpScalarConcept<cxxgpr::DoublePrecision>);
    static_assert(cxxgpr::GpScalarConcept<cxxgpr::SinglePrecision>);
    static_assert(cxxgpr::GpScalarConcept<cxxgpr::MixedPrecision<>>);
    static_assert(!cxxgpr::GpScalarConcept<BogusPolicy>);

    // Verify associated Storage, Refine, and default Residual types for each precision policy.
    static_assert(std::same_as<cxxgpr::DoublePrecision::Storage, double>);
    static_assert(std::same_as<cxxgpr::DoublePrecision::Refine, double>);
    static_assert(std::same_as<cxxgpr::SinglePrecision::Storage, float>);
    static_assert(std::same_as<cxxgpr::SinglePrecision::Refine, float>);
    static_assert(std::same_as<cxxgpr::MixedPrecision<>::Storage, float>);
    static_assert(std::same_as<cxxgpr::MixedPrecision<>::Refine, double>);
    static_assert(std::same_as<cxxgpr::MixedPrecision<>::Residual, cxxgpr::PromoteStorage>);
    static_assert(std::same_as<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>::Residual, cxxgpr::ReevaluateKernel>);

    /// @brief Verifies policy traits, PersistPrecisionKind tags, and persistence JSON string mappings.
    /// @details Confirms reads_storage, refines_in_f64, equality, persist_kind(), to_precision_string(),
    /// and to_residual_string() across Double, Single, MixedPromote, and MixedReevaluate.
    TEST(PrecisionPolicyTest, PolicyMetadataAndPersistenceTagsMatchSpec)
    {
        // Check residual formula traits and empty-struct equality.
        EXPECT_TRUE(cxxgpr::PromoteStorage::reads_storage);
        EXPECT_FALSE(cxxgpr::ReevaluateKernel::reads_storage);
        EXPECT_EQ(cxxgpr::PromoteStorage::name(), "promote_storage");
        EXPECT_EQ(cxxgpr::ReevaluateKernel::name(), "reevaluate_kernel");
        EXPECT_EQ(cxxgpr::PromoteStorage{}, cxxgpr::PromoteStorage{});
        EXPECT_EQ(cxxgpr::ReevaluateKernel{}, cxxgpr::ReevaluateKernel{});

        // Verify refinement flags and persistence tag serialization across all four policy modes.
        EXPECT_FALSE(cxxgpr::DoublePrecision::refines_in_f64);
        EXPECT_FALSE(cxxgpr::SinglePrecision::refines_in_f64);
        EXPECT_TRUE(cxxgpr::MixedPrecision<>::refines_in_f64);
        EXPECT_TRUE(cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>::refines_in_f64);

        EXPECT_EQ(cxxgpr::DoublePrecision::persist_kind(), cxxgpr::PersistPrecisionKind::Double);
        EXPECT_EQ(cxxgpr::SinglePrecision::persist_kind(), cxxgpr::PersistPrecisionKind::Single);
        EXPECT_EQ(cxxgpr::MixedPrecision<>::persist_kind(), cxxgpr::PersistPrecisionKind::MixedPromote);
        EXPECT_EQ(
            cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>::persist_kind(),
            cxxgpr::PersistPrecisionKind::MixedReevaluate
        );

        EXPECT_EQ(cxxgpr::to_precision_string(cxxgpr::PersistPrecisionKind::Double), "double");
        EXPECT_EQ(cxxgpr::to_precision_string(cxxgpr::PersistPrecisionKind::Single), "single");
        EXPECT_EQ(cxxgpr::to_precision_string(cxxgpr::PersistPrecisionKind::MixedPromote), "mixed");
        EXPECT_EQ(cxxgpr::to_precision_string(cxxgpr::PersistPrecisionKind::MixedReevaluate), "mixed");

        EXPECT_EQ(cxxgpr::to_residual_string(cxxgpr::PersistPrecisionKind::Double), "promote_storage");
        EXPECT_EQ(cxxgpr::to_residual_string(cxxgpr::PersistPrecisionKind::Single), "promote_storage");
        EXPECT_EQ(cxxgpr::to_residual_string(cxxgpr::PersistPrecisionKind::MixedPromote), "promote_storage");
        EXPECT_EQ(cxxgpr::to_residual_string(cxxgpr::PersistPrecisionKind::MixedReevaluate), "reevaluate_kernel");
    }

    /// @brief Verifies copy_storage_to_refine() promotes values bit-identically with zero heap allocation.
    /// @details Audits float-to-double promotion and same-type copy inside NoAllocationScope, checks
    /// bit-cast equality, and verifies EmptyInput precedes LengthMismatch on invalid spans.
    TEST(PrecisionPolicyTest, CopyStorageToRefinePreservesValuesAndChecksLengthsZeroAlloc)
    {
        const std::array<float, 4> src_f32{1.5F, -2.25F, 0.125F, 3.75F};
        std::array<double, 4> dst_f64{};
        std::array<float, 4> dst_f32{};

        // Audit mixed float->double promotion and single->single copy inside NoAllocationScope.
        {
            const cxxgpr::test_support::NoAllocationScope scope;
            ASSERT_TRUE(cxxgpr::copy_storage_to_refine<cxxgpr::MixedPrecision<>>(src_f32, dst_f64).has_value());
            ASSERT_TRUE(cxxgpr::copy_storage_to_refine<cxxgpr::SinglePrecision>(src_f32, dst_f32).has_value());
            EXPECT_EQ(scope.delta_allocations(), 0U);
        }

        // Verify bit-for-bit preservation of widened IEEE 754 float values.
        for (std::size_t i = 0; i < src_f32.size(); ++i)
        {
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(dst_f64[i]), std::bit_cast<std::uint64_t>(static_cast<double>(src_f32[i]))
            );
            EXPECT_EQ(std::bit_cast<std::uint32_t>(dst_f32[i]), std::bit_cast<std::uint32_t>(src_f32[i]));
        }

        const std::array<double, 3> src_f64{-1.0, 2.5, 4.0};
        std::array<double, 3> out_f64{};
        ASSERT_TRUE(cxxgpr::copy_storage_to_refine<cxxgpr::DoublePrecision>(src_f64, out_f64).has_value());
        EXPECT_EQ(out_f64, src_f64);

        // Negative control: mismatched non-empty buffer returns LengthMismatch.
        std::array<double, 2> short_dst{};
        const auto err = cxxgpr::copy_storage_to_refine<cxxgpr::DoublePrecision>(src_f64, short_dst);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        // EmptyInput takes precedence over LengthMismatch when either or both spans are empty.
        const std::span<const double> empty_src{};
        const std::span<double> empty_dst{};
        const auto empty_src_err = cxxgpr::copy_storage_to_refine<cxxgpr::DoublePrecision>(empty_src, short_dst);
        ASSERT_FALSE(empty_src_err.has_value());
        EXPECT_EQ(empty_src_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const auto empty_dst_err = cxxgpr::copy_storage_to_refine<cxxgpr::DoublePrecision>(src_f64, empty_dst);
        ASSERT_FALSE(empty_dst_err.has_value());
        EXPECT_EQ(empty_dst_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const auto both_empty_err = cxxgpr::copy_storage_to_refine<cxxgpr::DoublePrecision>(empty_src, empty_dst);
        ASSERT_FALSE(both_empty_err.has_value());
        EXPECT_EQ(both_empty_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);
    }

    /// @brief Verifies compute_storage_means() accumulates in double precision across strided columns.
    /// @details Uses catastrophic cancellation in float ([16777216.0F, 0.25F, -16777216.0F]) plus
    /// column padding sentinels (999.0F) to prove double accumulation and stride respect with zero allocs.
    TEST(PrecisionPolicyTest, ComputeStorageMeansAccumulatesInDoubleAndHandlesStridesZeroAlloc)
    {
        // Column-major 3x2 matrix stored with stride 4 (1 padding element after each column):
        // col 0: [16777216.0F, 0.25F, -16777216.0F], pad
        // col 1: [1.5F, -0.5F, 2.25F], pad
        // In single-precision left-to-right addition, 16777216.0F + 0.25F rounds away 0.25F,
        // whereas double accumulation preserves 0.25F.
        const std::array<float, 8> storage_raw{
            16777216.0F,
            0.25F,
            -16777216.0F,
            999.0F,
            1.5F,
            -0.5F,
            2.25F,
            999.0F,
        };
        const cxxgpr::MatrixRef<float> k_f32{storage_raw.data(), 3U, 2U, 4U};
        const std::array<float, 3> alpha_f32{1.0F, 1.0F, 1.0F};
        std::array<float, 2> out_f32{};

        const std::array<double, 3> alpha_f64{1.0, 1.0, 1.0};
        std::array<double, 2> out_f64{};

        // Audit both SinglePrecision and MixedPrecision predictive mean projections inside one scope.
        {
            const cxxgpr::test_support::NoAllocationScope scope;
            ASSERT_TRUE(cxxgpr::compute_storage_means<cxxgpr::SinglePrecision>(k_f32, alpha_f32, out_f32).has_value());
            ASSERT_TRUE(cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(k_f32, alpha_f64, out_f64).has_value());
            EXPECT_EQ(scope.delta_allocations(), 0U);
        }

        EXPECT_FLOAT_EQ(out_f32[0], 0.25F);
        EXPECT_FLOAT_EQ(out_f32[1], 3.25F);
        EXPECT_DOUBLE_EQ(out_f64[0], 0.25);
        EXPECT_DOUBLE_EQ(out_f64[1], 3.25);

        // Verify DoublePrecision path against explicit column dot products.
        const std::array<double, 6> k64_raw{0.5, -1.5, 2.0, 3.0, 4.0, -0.5};
        const cxxgpr::MatrixRef<double> k_f64{k64_raw.data(), 3U, 2U};
        const std::array<double, 3> weights{2.0, -1.0, 0.5};
        std::array<double, 2> pred64{};
        ASSERT_TRUE(cxxgpr::compute_storage_means<cxxgpr::DoublePrecision>(k_f64, weights, pred64).has_value());
        EXPECT_DOUBLE_EQ(pred64[0], (0.5 * 2.0) + ((-1.5) * (-1.0)) + (2.0 * 0.5));
        EXPECT_DOUBLE_EQ(pred64[1], (3.0 * 2.0) + (4.0 * (-1.0)) + ((-0.5) * 0.5));
    }

    /// @brief Negative controls verifying compute_storage_means() rejects empty/null and mismatched spans.
    /// @details Checks LengthMismatch for wrong alpha/out lengths and verifies EmptyInput precedence for
    /// null matrix pointers, zero row extents, and empty alpha or output spans even when lengths disagree.
    TEST(PrecisionPolicyTest, ComputeStorageMeansRejectsInvalidShapesAndNullBuffer)
    {
        const std::array<float, 6> data{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        const cxxgpr::MatrixRef<float> k_f32{data.data(), 3U, 2U};
        const std::array<double, 2> bad_alpha{1.0, 2.0};
        const std::array<double, 3> ok_alpha{1.0, 2.0, 3.0};
        std::array<double, 2> ok_out{};
        std::array<double, 1> bad_out{};

        // Non-empty mismatched alpha or output spans must fail with LengthMismatch.
        const auto alpha_err = cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(k_f32, bad_alpha, ok_out);
        ASSERT_FALSE(alpha_err.has_value());
        EXPECT_EQ(alpha_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        const auto out_err = cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(k_f32, ok_alpha, bad_out);
        ASSERT_FALSE(out_err.has_value());
        EXPECT_EQ(out_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        const cxxgpr::MatrixRef<float> null_k{nullptr, 3U, 2U};
        const auto null_err = cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(null_k, ok_alpha, ok_out);
        ASSERT_FALSE(null_err.has_value());
        EXPECT_EQ(null_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        // DES-02 precedence: EmptyInput precedes LengthMismatch on compound violations.
        const auto null_and_mismatch =
            cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(null_k, bad_alpha, ok_out);
        ASSERT_FALSE(null_and_mismatch.has_value());
        EXPECT_EQ(null_and_mismatch.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const cxxgpr::MatrixRef<float> zero_rows_k{data.data(), 0U, 2U};
        const auto zero_rows_err =
            cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(zero_rows_k, bad_alpha, ok_out);
        ASSERT_FALSE(zero_rows_err.has_value());
        EXPECT_EQ(zero_rows_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const std::span<const double> empty_alpha{};
        const auto empty_alpha_err =
            cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(k_f32, empty_alpha, bad_out);
        ASSERT_FALSE(empty_alpha_err.has_value());
        EXPECT_EQ(empty_alpha_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const std::span<double> empty_out{};
        const auto empty_out_err = cxxgpr::compute_storage_means<cxxgpr::MixedPrecision<>>(k_f32, bad_alpha, empty_out);
        ASSERT_FALSE(empty_out_err.has_value());
        EXPECT_EQ(empty_out_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);
    }

}  // namespace
