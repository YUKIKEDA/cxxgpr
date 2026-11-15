// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/views.hpp"

#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <span>

// Spec: REQ-11, DES-01, DES-08, DES-16
namespace {

TEST(ViewsTest, ColumnMajorIndexingAndStridedSubmatrixWorkCorrectly) {
    // Packed 3x2 column-major matrix:
    // col 0: [1, 2, 3], col 1: [4, 5, 6]
    std::array<double, 6> storage{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const cxxgpr::MatrixMut<double> mat(std::span<double>{storage}, 3U, 2U);

    EXPECT_FALSE(mat.is_empty());
    EXPECT_FALSE(mat.is_square());
    EXPECT_TRUE(mat.is_contiguous());
    EXPECT_EQ(mat.rows(), 3U);
    EXPECT_EQ(mat.cols(), 2U);
    EXPECT_EQ(mat.col_stride(), 3U);
    EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 5.0);

    // Strided 2x2 bottom-row subview starting at row 1 with col_stride = 3.
    cxxgpr::MatrixMut<double> sub(&storage[1], 2U, 2U, 3U);
    EXPECT_TRUE(sub.is_square());
    EXPECT_FALSE(sub.is_contiguous());
    EXPECT_DOUBLE_EQ(sub(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(sub(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(sub(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(sub(1, 1), 6.0);

    sub(1, 1) = 42.0;
    EXPECT_DOUBLE_EQ(storage[5], 42.0);

    const cxxgpr::MatrixRef<double> ref = sub;
    EXPECT_DOUBLE_EQ(ref(1, 1), 42.0);
    const auto col1 = ref.col(1);
    ASSERT_EQ(col1.size(), 2U);
    EXPECT_DOUBLE_EQ(col1[0], 5.0);
    EXPECT_DOUBLE_EQ(col1[1], 42.0);

    const cxxgpr::MatrixRef<double> empty_view;
    EXPECT_TRUE(empty_view.is_empty());
    EXPECT_TRUE(empty_view.col(0).empty());
}

TEST(ViewsTest, PredictionTypesAndClosedEnumsUseSpecifiedDefaults) {
    EXPECT_EQ(cxxgpr::PredictOptions{}.variance_kind, cxxgpr::VarianceKind::Observation);
    EXPECT_NE(cxxgpr::Triangle::Lower, cxxgpr::Triangle::Upper);
    EXPECT_EQ(cxxgpr::kernel::Triangle::Full, cxxgpr::Triangle::Full);

    cxxgpr::Prediction<double> pred_d;
    EXPECT_EQ(pred_d.variance_kind, cxxgpr::VarianceKind::Observation);
    pred_d.mean = {1.0, 2.0};
    pred_d.variance = {0.1, 0.2};
    EXPECT_EQ(pred_d.mean.size(), 2U);

    cxxgpr::Prediction<float> pred_f;
    pred_f.variance_kind = cxxgpr::VarianceKind::Latent;
    EXPECT_EQ(pred_f.variance_kind, cxxgpr::VarianceKind::Latent);

    cxxgpr::PredictiveCovariance<double> cov;
    EXPECT_EQ(cov.n_points, 0U);
    EXPECT_EQ(cov.variance_kind, cxxgpr::VarianceKind::Observation);
}

TEST(ViewsTest, ViewAccessAndColumnSlicingAllocateZero) {
    std::array<double, 9> buf{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double acc = 0.0;
    std::size_t delta = 1;
    {
        cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
        cxxgpr::MatrixMut<double> mut(std::span<double>{buf}, 3U, 3U);
        mut(2, 1) = -7.0;
        const cxxgpr::MatrixRef<double> ref = mut.as_ref();
        for (std::size_t col = 0; col < ref.cols(); ++col) {
            for (const double v : ref.col(col)) {
                acc += v;
            }
        }
        cxxgpr::test_support::touch_for_test(&acc);
        delta = scope.delta_allocations();
    }
    EXPECT_EQ(delta, 0U);
    EXPECT_DOUBLE_EQ(acc, 32.0);
}

} // namespace
