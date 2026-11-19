// SPDX-License-Identifier: Apache-2.0
// Spec: DES-02, DES-03
/// @file dense_test.cpp
/// @brief Unit tests for dense column-major linear algebra helpers.
/// @details Verifies add_to_diag(), symmetrize_lower(), copy_lower_triangle(), frobenius_lower(),
/// dot(), and matvec_columns() correctness, upper-triangle preservation, zero-allocation hot paths,
/// and negative input validation.

#include "cxxgpr/linalg/dense.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Brings GprErrorCode into the anonymous test namespace.
    using cxxgpr::GprErrorCode;
    /// @brief Brings mutable MatrixMut view into the anonymous test namespace.
    using cxxgpr::MatrixMut;
    /// @brief Brings read-only MatrixRef view into the anonymous test namespace.
    using cxxgpr::MatrixRef;
    /// @brief Brings allocation AuditMode enum into the anonymous test namespace.
    using cxxgpr::test_support::AuditMode;
    /// @brief Brings RAII NoAllocationScope guard into the anonymous test namespace.
    using cxxgpr::test_support::NoAllocationScope;

    /// @brief Verifies add_to_diag() shifts only diagonal entries and symmetrize_lower() mirrors them.
    /// @details Places sentinel garbage in the strictly upper triangle and confirms add_to_diag() leaves
    /// it untouched before symmetrize_lower() overwrites it from the lower triangle.
    TEST(DenseLinalgTest, AddToDiagAndSymmetrizeLower)
    {
        std::array<double, 9> mat = {
            1.0, 2.0, 3.0,  // col 0
            9.0, 4.0, 5.0,  // col 1 (upper entry 9.0 is garbage)
            8.0, 7.0, 6.0   // col 2 (upper entries 8.0, 7.0 are garbage)
        };
        const MatrixMut<double> mut_view(mat.data(), 3, 3);
        // Diagonal shift must not disturb strictly upper-triangular sentinel entries.
        ASSERT_TRUE(cxxgpr::linalg::add_to_diag(mut_view, 0.5).has_value());
        EXPECT_DOUBLE_EQ(mat[0], 1.5);
        EXPECT_DOUBLE_EQ(mat[4], 4.5);
        EXPECT_DOUBLE_EQ(mat[8], 6.5);
        EXPECT_DOUBLE_EQ(mat[3], 9.0);

        // Mirroring lower triangle overwrites the upper triangle with exact subdiagonal values.
        ASSERT_TRUE(cxxgpr::linalg::symmetrize_lower(mut_view).has_value());
        EXPECT_DOUBLE_EQ(mat[3], 2.0);
        EXPECT_DOUBLE_EQ(mat[6], 3.0);
        EXPECT_DOUBLE_EQ(mat[7], 5.0);
    }

    /// @brief Verifies copy_lower_triangle() copies diagonal and subdiagonal entries without upper writes.
    /// @details Seeds the destination upper entry with -7.0 and confirms it stays untouched after copying.
    TEST(DenseLinalgTest, CopyLowerTrianglePreservesDestinationUpper)
    {
        const std::array<double, 4> src = {1.0, 2.0, 99.0, 3.0};
        std::array<double, 4> dst = {0.0, 0.0, -7.0, 0.0};
        ASSERT_TRUE(
            cxxgpr::linalg::copy_lower_triangle(
                MatrixRef<double>(src.data(), 2, 2), MatrixMut<double>(dst.data(), 2, 2)
            )
                .has_value()
        );
        // Only lower-triangular indices (0, 1, 3) change; upper index 2 retains -7.0.
        EXPECT_DOUBLE_EQ(dst[0], 1.0);
        EXPECT_DOUBLE_EQ(dst[1], 2.0);
        EXPECT_DOUBLE_EQ(dst[2], -7.0);
        EXPECT_DOUBLE_EQ(dst[3], 3.0);
    }

    /// @brief Verifies frobenius_lower() computes tr(A * B) using only lower-triangular entries.
    /// @details Doubles strictly lower-triangular products and adds diagonal products, confirming an
    /// exact match against the analytical sum (16 - 5.5 = 10.5).
    TEST(DenseLinalgTest, FrobeniusLowerMatchesFullSymmetricInnerProduct)
    {
        const std::array<double, 9> a = {
            2.0, -1.0, 0.5,  // col 0
            0.0, 3.0, 1.5,   // col 1 (upper ignored)
            0.0, 0.0, 4.0    // col 2 (upper ignored)
        };
        const std::array<double, 9> b = {
            1.0, 2.0, -3.0,  // col 0
            0.0, -2.0, 0.5,  // col 1
            0.0, 0.0, 5.0    // col 2
        };
        // Only the lower triangles of A and B are read; upper zeros must not alter the symmetric trace.
        const auto inner =
            cxxgpr::linalg::frobenius_lower(MatrixRef<double>(a.data(), 3, 3), MatrixRef<double>(b.data(), 3, 3));
        ASSERT_TRUE(inner.has_value());
        // Diag: 2*1 + 3*(-2) + 4*5 = 2 - 6 + 20 = 16
        // Strict lower x2: 2 * ((-1)*2 + 0.5*(-3) + 1.5*0.5) = 2 * (-2 - 1.5 + 0.75) = -5.5
        // Total: 10.5
        EXPECT_DOUBLE_EQ(*inner, 10.5);
    }

    /// @brief Verifies float dot() and column-major matvec_columns() results and zero-allocation execution.
    /// @details Evaluates a 2x3 column-major matrix-vector product and self-dot product inside
    /// AuditMode::CppAndEigen NoAllocationScope and checks results against analytical values.
    TEST(DenseLinalgTest, DotAndMatvecColumnsMatchReferenceWithZeroAllocations)
    {
        const std::array<float, 6> a = {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};  // 2x3 col-major
        const std::array<float, 3> x = {0.5F, -1.0F, 2.0F};
        std::array<float, 2> out = {0.0F, 0.0F};

        // Audit both matvec_columns() and dot() inside a single zero-allocation scope.
        {
            const NoAllocationScope audit(AuditMode::CppAndEigen);
            const auto mv = cxxgpr::linalg::matvec_columns(
                MatrixRef<float>(a.data(), 2, 3), std::span<const float>(x), std::span<float>(out)
            );
            ASSERT_TRUE(mv.has_value());
            const auto d = cxxgpr::linalg::dot<float>(std::span<const float>(x), std::span<const float>(x));
            ASSERT_TRUE(d.has_value());
            EXPECT_DOUBLE_EQ(*d, 5.25);
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }

        // row 0: 1*0.5 + 2*(-1) + 3*2 = 4.5
        // row 1: 4*0.5 + 5*(-1) + 6*2 = 9.0
        EXPECT_FLOAT_EQ(out[0], 4.5F);
        EXPECT_FLOAT_EQ(out[1], 9.0F);
    }

    /// @brief Negative controls verifying dense helpers reject empty, non-square, and non-finite inputs.
    /// @details Confirms EmptyInput for null/empty buffers, LengthMismatch for rectangular matrices in
    /// symmetrize_lower(), and NonFiniteInput when adding a NaN scalar to the diagonal.
    TEST(DenseLinalgTest, NegativeInputValidation)
    {
        std::vector<double> empty_vec;
        const MatrixMut<double> empty_mat(nullptr, 0, 0);
        // Empty matrix/vector inputs must return EmptyInput before geometry or scalar checks.
        EXPECT_EQ(cxxgpr::linalg::add_to_diag(empty_mat, 1.0).error().code(), GprErrorCode::EmptyInput);
        EXPECT_EQ(cxxgpr::linalg::dot<double>(empty_vec, empty_vec).error().code(), GprErrorCode::EmptyInput);

        std::array<double, 6> rect = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        EXPECT_EQ(
            cxxgpr::linalg::symmetrize_lower(MatrixMut<double>(rect.data(), 2, 3)).error().code(),
            GprErrorCode::LengthMismatch
        );

        const double nan_val = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ(
            cxxgpr::linalg::add_to_diag(MatrixMut<double>(rect.data(), 2, 2), nan_val).error().code(),
            GprErrorCode::NonFiniteInput
        );
    }

}  // namespace
