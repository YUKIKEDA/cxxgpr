// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/linalg/cholesky.hpp"

#include "cxxgpr/linalg/dense.hpp"
#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

// Spec: REQ-06, REQ-11, DES-02, DES-03, DES-04
namespace {

using cxxgpr::CholeskyStage;
using cxxgpr::GprErrorCode;
using cxxgpr::JitterPolicy;
using cxxgpr::MatrixMut;
using cxxgpr::MatrixRef;
using cxxgpr::test_support::AuditMode;
using cxxgpr::test_support::NoAllocationScope;

template <typename Scalar> std::vector<Scalar> make_rbf_gram(std::size_t n, Scalar diag_offset) {
    std::vector<Scalar> a(n * n, Scalar{0});
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = j; i < n; ++i) {
            const double diff = static_cast<double>(i) - static_cast<double>(j);
            const double d = diff * 0.35;
            double val = std::exp(-0.5 * d * d);
            if (i == j) {
                val += static_cast<double>(diag_offset);
            }
            a[(j * n) + i] = static_cast<Scalar>(val);
            a[(i * n) + j] = static_cast<Scalar>(val);
        }
    }
    return a;
}

TEST(CholeskyLinalgTest, LowerFactorAndSolvesMatchAnalyticalReferenceDouble) {
    // 3x3 SPD matrix:
    // [ 4  2  -2 ]
    // [ 2 10   5 ]
    // [-2  5  14 ]
    // Exact L:
    // [ 2  0  0 ]
    // [ 1  3  0 ]
    // [-1  2  3 ]
    std::array<double, 9> a = {
        4.0,  2.0,  -2.0, // col 0
        99.0, 10.0, 5.0,  // col 1 (upper sentinel 99.0)
        88.0, 77.0, 14.0  // col 2 (upper sentinels 88.0, 77.0)
    };
    const MatrixMut<double> a_view(a.data(), 3, 3);
    ASSERT_TRUE(cxxgpr::linalg::cholesky_lower_in_place(a_view).has_value());

    EXPECT_NEAR(a[0], 2.0, 1e-14);
    EXPECT_NEAR(a[1], 1.0, 1e-14);
    EXPECT_NEAR(a[2], -1.0, 1e-14);
    EXPECT_NEAR(a[4], 3.0, 1e-14);
    EXPECT_NEAR(a[5], 2.0, 1e-14);
    EXPECT_NEAR(a[8], 3.0, 1e-14);
    // Strictly upper triangle must remain untouched.
    EXPECT_DOUBLE_EQ(a[3], 99.0);
    EXPECT_DOUBLE_EQ(a[6], 88.0);
    EXPECT_DOUBLE_EQ(a[7], 77.0);

    const MatrixRef<double> l_ref(a.data(), 3, 3);
    // Exact det(A) = (2 * 3 * 3)^2 = 324 -> log|A| = log(324)
    const auto log_det = cxxgpr::linalg::log_det_from_l(l_ref);
    ASSERT_TRUE(log_det.has_value());
    EXPECT_NEAR(*log_det, std::log(324.0), 1e-14);

    const auto half_ld = cxxgpr::linalg::half_log_det_from_l(l_ref);
    ASSERT_TRUE(half_ld.has_value());
    EXPECT_NEAR(*half_ld, 0.5 * std::log(324.0), 1e-14);

    // Solve A * x = b where true x = [1, -2, 3]^T -> b = [-6, -3, 30]^T
    std::array<double, 3> rhs = {-6.0, -3.0, 30.0};
    ASSERT_TRUE(cxxgpr::linalg::solve_llt_in_place(l_ref, std::span<double>(rhs)).has_value());
    EXPECT_NEAR(rhs[0], 1.0, 1e-13);
    EXPECT_NEAR(rhs[1], -2.0, 1e-13);
    EXPECT_NEAR(rhs[2], 3.0, 1e-13);

    // Verify diag(A^-1)
    // L^-1 = [0.5, 0, 0; -1/6, 1/3, 0; 5/18, -2/9, 1/3]
    // diag(A^-1)_0 = 0.25 + 1/36 + 25/324 = (81 + 9 + 25)/324 = 115/324
    // diag(A^-1)_1 = 1/9 + 4/81 = 13/81
    // diag(A^-1)_2 = 1/9
    std::array<double, 3> inv_diag{};
    std::array<double, 9> scratch{};
    ASSERT_TRUE(cxxgpr::linalg::inv_diag_from_chol_l(l_ref, std::span<double>(inv_diag),
                                                     MatrixMut<double>(scratch.data(), 3, 3))
                    .has_value());
    EXPECT_NEAR(inv_diag[0], 115.0 / 324.0, 1e-14);
    EXPECT_NEAR(inv_diag[1], 13.0 / 81.0, 1e-14);
    EXPECT_NEAR(inv_diag[2], 1.0 / 9.0, 1e-14);
}

TEST(CholeskyLinalgTest, LowerFactorAndSolvesFloatPrecision) {
    constexpr std::size_t n = 12;
    auto a = make_rbf_gram<float>(n, 1e-2F);
    const auto a_orig = a;
    std::vector<float> backup(n * n);

    const auto policy = JitterPolicy::fixed(0.0);
    ASSERT_TRUE(policy.has_value());
    const auto jitter_res = cxxgpr::linalg::cholesky_lower_with_jitter(
        MatrixMut<float>(a.data(), n, n), MatrixMut<float>(backup.data(), n, n), *policy);
    ASSERT_TRUE(jitter_res.has_value());
    EXPECT_DOUBLE_EQ(*jitter_res, 0.0);

    std::vector<float> b(n, 1.0F);
    const std::vector<float> b_orig = b;
    ASSERT_TRUE(
        cxxgpr::linalg::solve_llt_in_place(MatrixRef<float>(a.data(), n, n), std::span<float>(b))
            .has_value());

    std::vector<float> ax(n, 0.0F);
    ASSERT_TRUE(cxxgpr::linalg::matvec_columns(MatrixRef<float>(a_orig.data(), n, n),
                                               std::span<const float>(b), std::span<float>(ax))
                    .has_value());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(ax[i], b_orig[i], 1e-4F);
    }
}

TEST(CholeskyLinalgTest, JitterEscalatesOnIndefiniteAndPreservesFactorJitter) {
    // Rank-1 / singular 2x2 matrix: [1, 1; 1, 1] -> eigenvalues 2 and 0.
    // Adding j > 0 makes it SPD.
    const std::array<double, 4> singular = {1.0, 1.0, 1.0, 1.0};
    std::array<double, 4> work = {1.0, 1.0, 1.0, 1.0};
    std::array<double, 4> backup{};

    // Default Fixed(0.0) fails with jitter() == 0.0 and stage == Predict
    const auto default_res = cxxgpr::linalg::cholesky_lower_with_jitter(
        MatrixMut<double>(work.data(), 2, 2), MatrixMut<double>(backup.data(), 2, 2),
        JitterPolicy{}, CholeskyStage::Predict);
    ASSERT_FALSE(default_res.has_value());
    EXPECT_EQ(default_res.error().code(), GprErrorCode::CholeskyFailed);
    EXPECT_EQ(default_res.error().jitter(), 0.0);
    EXPECT_EQ(default_res.error().matrix_size(), 2U);
    EXPECT_EQ(default_res.error().cholesky_stage(), CholeskyStage::Predict);

    // Slightly negative-definite: [1, 1; 1, 1 - 5e-4] -> min eigenvalue approx -2.5e-4
    // Adaptive(1e-5, 10.0, 5, 1.0) should fail at 1e-5, 1e-4, and succeed at 1e-3.
    work = {1.0, 1.0, 1.0, 1.0 - 5e-4};
    const auto adaptive_policy = JitterPolicy::adaptive(1e-5, 10.0, 5, 1.0);
    ASSERT_TRUE(adaptive_policy.has_value());
    const auto adap_res = cxxgpr::linalg::cholesky_lower_with_jitter(
        MatrixMut<double>(work.data(), 2, 2), MatrixMut<double>(backup.data(), 2, 2),
        *adaptive_policy, CholeskyStage::Fit);
    ASSERT_TRUE(adap_res.has_value());
    EXPECT_DOUBLE_EQ(*adap_res, 1e-3);

    // Exhaustion reports the last jitter tried.
    work = {-5.0, 0.0, 0.0, -5.0};
    const auto exhaust_policy = JitterPolicy::adaptive(1e-6, 10.0, 3, 1.0);
    ASSERT_TRUE(exhaust_policy.has_value());
    const auto exhaust_res = cxxgpr::linalg::cholesky_lower_with_jitter(
        MatrixMut<double>(work.data(), 2, 2), MatrixMut<double>(backup.data(), 2, 2),
        *exhaust_policy, CholeskyStage::Fit);
    ASSERT_FALSE(exhaust_res.has_value());
    EXPECT_EQ(exhaust_res.error().code(), GprErrorCode::CholeskyFailed);
    ASSERT_TRUE(exhaust_res.error().jitter().has_value());
    EXPECT_DOUBLE_EQ(*exhaust_res.error().jitter(), 1e-6 * 10.0 * 10.0);
    EXPECT_EQ(exhaust_res.error().matrix_size(), 2U);
    EXPECT_EQ(exhaust_res.error().cholesky_stage(), CholeskyStage::Fit);
    (void)singular;
}

TEST(CholeskyLinalgTest, ZeroHeapAllocationsInPreallocatedWorkspace) {
    constexpr std::size_t n = 32;
    auto a = make_rbf_gram<double>(n, 1e-2);
    std::vector<double> backup(n * n);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> inv_diag(n, 0.0);
    std::vector<double> scratch(n * n, 0.0);

    const auto policy = JitterPolicy::adaptive(1e-8, 10.0, 4, 1e-2);
    ASSERT_TRUE(policy.has_value());

    const NoAllocationScope audit(AuditMode::CppAndEigen);
    const auto jitter_res = cxxgpr::linalg::cholesky_lower_with_jitter(
        MatrixMut<double>(a.data(), n, n), MatrixMut<double>(backup.data(), n, n), *policy);
    ASSERT_TRUE(jitter_res.has_value());

    const MatrixRef<double> l_ref(a.data(), n, n);
    ASSERT_TRUE(cxxgpr::linalg::solve_lower_in_place(l_ref, std::span<double>(rhs)).has_value());
    ASSERT_TRUE(
        cxxgpr::linalg::solve_lower_transpose_in_place(l_ref, std::span<double>(rhs)).has_value());
    ASSERT_TRUE(cxxgpr::linalg::solve_llt_in_place(l_ref, std::span<double>(rhs)).has_value());
    ASSERT_TRUE(cxxgpr::linalg::inv_diag_from_chol_l(l_ref, std::span<double>(inv_diag),
                                                     MatrixMut<double>(scratch.data(), n, n))
                    .has_value());
    const auto ld = cxxgpr::linalg::log_det_from_l(l_ref);
    ASSERT_TRUE(ld.has_value());
    EXPECT_EQ(audit.delta_allocations(), 0U);
}

TEST(CholeskyLinalgTest, NegativeValidationAndErrorOrdering) {
    std::array<double, 4> mat = {1.0, 0.0, 0.0, 1.0};
    std::array<double, 2> vec = {1.0, 2.0};
    std::array<double, 9> wrong_mat{};

    EXPECT_EQ(
        cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(nullptr, 0, 0)).error().code(),
        GprErrorCode::EmptyInput);
    EXPECT_EQ(
        cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(mat.data(), 2, 1)).error().code(),
        GprErrorCode::LengthMismatch);
    EXPECT_EQ(cxxgpr::linalg::cholesky_lower_with_jitter(MatrixMut<double>(mat.data(), 2, 2),
                                                         MatrixMut<double>(wrong_mat.data(), 3, 3),
                                                         JitterPolicy{})
                  .error()
                  .code(),
              GprErrorCode::LengthMismatch);

    mat[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(
        cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(mat.data(), 2, 2)).error().code(),
        GprErrorCode::NonFiniteInput);

    mat = {1.0, 0.0, 0.0, 0.0};
    EXPECT_EQ(cxxgpr::linalg::log_det_from_l(MatrixRef<double>(mat.data(), 2, 2)).error().code(),
              GprErrorCode::NonPositiveDefiniteMatrix);
    EXPECT_EQ(cxxgpr::linalg::solve_llt_in_place(MatrixRef<double>(mat.data(), 2, 2),
                                                 std::span<double>(vec))
                  .error()
                  .code(),
              GprErrorCode::NonPositiveDefiniteMatrix);
}

} // namespace
