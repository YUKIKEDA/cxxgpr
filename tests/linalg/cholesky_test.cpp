// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-06, REQ-11, DES-02, DES-03, DES-04
/// @file cholesky_test.cpp
/// @brief Unit tests for lower Cholesky factorization, jitter escalation, triangular solves, and inv_diag.
/// @details Verifies analytical Cholesky factors, log-determinant, solves, and diagonal inverse
/// formulas in double precision, float-precision jittered solves, adaptive jitter escalation and
/// failure diagnostics, zero-allocation execution, and negative validation checks.

#include "cxxgpr/linalg/cholesky.hpp"

#include "cxxgpr/linalg/dense.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "allocation_audit.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace
{

    /// @brief Brings CholeskyStage into the anonymous test namespace.
    using cxxgpr::CholeskyStage;
    /// @brief Brings GprErrorCode into the anonymous test namespace.
    using cxxgpr::GprErrorCode;
    /// @brief Brings JitterPolicy into the anonymous test namespace.
    using cxxgpr::JitterPolicy;
    /// @brief Brings mutable MatrixMut view into the anonymous test namespace.
    using cxxgpr::MatrixMut;
    /// @brief Brings read-only MatrixRef view into the anonymous test namespace.
    using cxxgpr::MatrixRef;
    /// @brief Brings allocation AuditMode enum into the anonymous test namespace.
    using cxxgpr::test_support::AuditMode;
    /// @brief Brings RAII NoAllocationScope guard into the anonymous test namespace.
    using cxxgpr::test_support::NoAllocationScope;

    /// @brief Constructs an n x n symmetric RBF Gram matrix with an explicit diagonal shift.
    /// @details Evaluates exp(-0.5 * ((i - j) * 0.35)^2) across all pairs and adds diag_offset to
    /// diagonal entries so test systems can tune conditioning across float and double precisions.
    /// @tparam Scalar Matrix scalar type (float or double).
    /// @param[in] n Square matrix order.
    /// @param[in] diag_offset Non-negative diagonal shift added to every diagonal entry.
    /// @return Column-major n x n SPD Gram matrix stored in a flat vector.
    template <typename Scalar> std::vector<Scalar> make_rbf_gram(std::size_t n, Scalar diag_offset)
    {
        std::vector<Scalar> a(n * n, Scalar{0});
        // Fill symmetric lower and upper entries using deterministic 1D grid spacing 0.35.
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t i = j; i < n; ++i)
            {
                const double diff = static_cast<double>(i) - static_cast<double>(j);
                const double d = diff * 0.35;
                double val = std::exp(-0.5 * d * d);
                if (i == j)
                {
                    val += static_cast<double>(diag_offset);
                }
                a[(j * n) + i] = static_cast<Scalar>(val);
                a[(i * n) + j] = static_cast<Scalar>(val);
            }
        }
        return a;
    }

    /// @brief Verifies double-precision Cholesky factor, log|A|, LLT solve, and diag(A^-1) against exact math.
    /// @details Confirms cholesky_lower_in_place() leaves upper-triangle sentinels untouched and matches
    /// exact closed-form entries for L, log(324), x = [1, -2, 3]^T, and diag(A^-1) = [115/324, 13/81, 1/9].
    TEST(CholeskyLinalgTest, LowerFactorAndSolvesMatchAnalyticalReferenceDouble)
    {
        // 3x3 SPD matrix:
        // [ 4  2  -2 ]
        // [ 2 10   5 ]
        // [-2  5  14 ]
        // Exact L:
        // [ 2  0  0 ]
        // [ 1  3  0 ]
        // [-1  2  3 ]
        std::array<double, 9> a = {
            4.0, 2.0, -2.0,   // col 0
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
        ASSERT_TRUE(
            cxxgpr::linalg::inv_diag_from_chol_l(
                l_ref, std::span<double>(inv_diag), MatrixMut<double>(scratch.data(), 3, 3)
            )
                .has_value()
        );
        EXPECT_NEAR(inv_diag[0], 115.0 / 324.0, 1e-14);
        EXPECT_NEAR(inv_diag[1], 13.0 / 81.0, 1e-14);
        EXPECT_NEAR(inv_diag[2], 1.0 / 9.0, 1e-14);
    }

    /// @brief Verifies single-precision jittered Cholesky factorization and linear solve residual.
    /// @details Factors a 12x12 float RBF matrix with fixed(0.0) policy, solves A * x = 1, and checks
    /// forward reconstruction A * x against the original right-hand side.
    TEST(CholeskyLinalgTest, LowerFactorAndSolvesFloatPrecision)
    {
        constexpr std::size_t n = 12;
        auto a = make_rbf_gram<float>(n, 1e-2F);
        const auto a_orig = a;
        std::vector<float> backup(n * n);

        // Unjittered factorization (j = 0) should succeed directly on the diagonally shifted RBF matrix.
        const auto policy = JitterPolicy::fixed(0.0);
        ASSERT_TRUE(policy.has_value());
        const auto jitter_res = cxxgpr::linalg::cholesky_lower_with_jitter(
            MatrixMut<float>(a.data(), n, n), MatrixMut<float>(backup.data(), n, n), *policy
        );
        ASSERT_TRUE(jitter_res.has_value());
        EXPECT_DOUBLE_EQ(*jitter_res, 0.0);

        std::vector<float> b(n, 1.0F);
        const std::vector<float> b_orig = b;
        ASSERT_TRUE(
            cxxgpr::linalg::solve_llt_in_place(MatrixRef<float>(a.data(), n, n), std::span<float>(b)).has_value()
        );

        // Verify forward residual A_orig * x ≈ b_orig within single-precision tolerance.
        std::vector<float> ax(n, 0.0F);
        ASSERT_TRUE(
            cxxgpr::linalg::matvec_columns(
                MatrixRef<float>(a_orig.data(), n, n), std::span<const float>(b), std::span<float>(ax)
            )
                .has_value()
        );
        for (std::size_t i = 0; i < n; ++i)
        {
            EXPECT_NEAR(ax[i], b_orig[i], 1e-4F);
        }
    }

    /// @brief Verifies adaptive jitter escalation, stage metadata, and failure diagnostics on indefinites.
    /// @details Checks Fixed(0.0) immediate Predict failure, Adaptive escalation succeeding at the third
    /// step (1e-3), and retry exhaustion reporting the final attempted jitter and Fit stage.
    TEST(CholeskyLinalgTest, JitterEscalatesOnIndefiniteAndPreservesFactorJitter)
    {
        // Rank-1 / singular 2x2 matrix: [1, 1; 1, 1] -> eigenvalues 2 and 0.
        // Adding j > 0 makes it SPD.
        const std::array<double, 4> singular = {1.0, 1.0, 1.0, 1.0};
        std::array<double, 4> work = {1.0, 1.0, 1.0, 1.0};
        std::array<double, 4> backup{};

        // Default Fixed(0.0) fails with jitter() == 0.0 and stage == Predict
        const auto default_res = cxxgpr::linalg::cholesky_lower_with_jitter(
            MatrixMut<double>(work.data(), 2, 2), MatrixMut<double>(backup.data(), 2, 2), JitterPolicy{},
            CholeskyStage::Predict
        );
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
            MatrixMut<double>(work.data(), 2, 2), MatrixMut<double>(backup.data(), 2, 2), *adaptive_policy,
            CholeskyStage::Fit
        );
        ASSERT_TRUE(adap_res.has_value());
        EXPECT_DOUBLE_EQ(*adap_res, 1e-3);

        // Exhaustion reports the last jitter tried.
        work = {-5.0, 0.0, 0.0, -5.0};
        const auto exhaust_policy = JitterPolicy::adaptive(1e-6, 10.0, 3, 1.0);
        ASSERT_TRUE(exhaust_policy.has_value());
        const auto exhaust_res = cxxgpr::linalg::cholesky_lower_with_jitter(
            MatrixMut<double>(work.data(), 2, 2), MatrixMut<double>(backup.data(), 2, 2), *exhaust_policy,
            CholeskyStage::Fit
        );
        ASSERT_FALSE(exhaust_res.has_value());
        EXPECT_EQ(exhaust_res.error().code(), GprErrorCode::CholeskyFailed);
        // Guard std::optional dereference before inspecting the reported jitter value.
        const auto err_jitter = exhaust_res.error().jitter();
        if (!err_jitter.has_value())
        {
            ADD_FAILURE() << "expected CholeskyFailed jitter";
            return;
        }
        EXPECT_DOUBLE_EQ(*err_jitter, 1e-6 * 10.0 * 10.0);
        EXPECT_EQ(exhaust_res.error().matrix_size(), 2U);
        EXPECT_EQ(exhaust_res.error().cholesky_stage(), CholeskyStage::Fit);
        (void)singular;
    }

    /// @brief Verifies Cholesky factorization, solves, inv_diag, and log_det allocate zero heap memory.
    /// @details Preallocates all n = 32 matrices and buffers before entering AuditMode::CppAndEigen
    /// NoAllocationScope and asserts delta_allocations() == 0 across every linear-algebra primitive.
    TEST(CholeskyLinalgTest, ZeroHeapAllocationsInPreallocatedWorkspace)
    {
        constexpr std::size_t n = 32;
        auto a = make_rbf_gram<double>(n, 1e-2);
        std::vector<double> backup(n * n);
        std::vector<double> rhs(n, 1.0);
        std::vector<double> inv_diag(n, 0.0);
        std::vector<double> scratch(n * n, 0.0);

        const auto policy = JitterPolicy::adaptive(1e-8, 10.0, 4, 1e-2);
        ASSERT_TRUE(policy.has_value());

        // Enter allocation audit scope only after all test vectors are sized.
        const NoAllocationScope audit(AuditMode::CppAndEigen);
        const auto jitter_res = cxxgpr::linalg::cholesky_lower_with_jitter(
            MatrixMut<double>(a.data(), n, n), MatrixMut<double>(backup.data(), n, n), *policy
        );
        ASSERT_TRUE(jitter_res.has_value());

        const MatrixRef<double> l_ref(a.data(), n, n);
        ASSERT_TRUE(cxxgpr::linalg::solve_lower_in_place(l_ref, std::span<double>(rhs)).has_value());
        ASSERT_TRUE(cxxgpr::linalg::solve_lower_transpose_in_place(l_ref, std::span<double>(rhs)).has_value());
        ASSERT_TRUE(cxxgpr::linalg::solve_llt_in_place(l_ref, std::span<double>(rhs)).has_value());
        ASSERT_TRUE(
            cxxgpr::linalg::inv_diag_from_chol_l(
                l_ref, std::span<double>(inv_diag), MatrixMut<double>(scratch.data(), n, n)
            )
                .has_value()
        );
        const auto ld = cxxgpr::linalg::log_det_from_l(l_ref);
        ASSERT_TRUE(ld.has_value());
        EXPECT_EQ(audit.delta_allocations(), 0U);
    }

    /// @brief Negative controls verifying Cholesky validation errors and strict non-SPD factor checks.
    /// @details Checks EmptyInput, LengthMismatch on non-square or mismatched backup matrices,
    /// NonFiniteInput on NaN entries, and NonPositiveDefiniteMatrix on zero-diagonal factors.
    TEST(CholeskyLinalgTest, NegativeValidationAndErrorOrdering)
    {
        std::array<double, 4> mat = {1.0, 0.0, 0.0, 1.0};
        std::array<double, 2> vec = {1.0, 2.0};
        std::array<double, 9> wrong_mat{};

        // Null/empty and non-square/mismatched dimension errors fire before reading matrix entries.
        EXPECT_EQ(
            cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(nullptr, 0, 0)).error().code(),
            GprErrorCode::EmptyInput
        );
        EXPECT_EQ(
            cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(mat.data(), 2, 1)).error().code(),
            GprErrorCode::LengthMismatch
        );
        EXPECT_EQ(
            cxxgpr::linalg::cholesky_lower_with_jitter(
                MatrixMut<double>(mat.data(), 2, 2), MatrixMut<double>(wrong_mat.data(), 3, 3), JitterPolicy{}
            )
                .error()
                .code(),
            GprErrorCode::LengthMismatch
        );

        // NaN inside the lower triangle must return NonFiniteInput.
        mat[1] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ(
            cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(mat.data(), 2, 2)).error().code(),
            GprErrorCode::NonFiniteInput
        );

        // Zero diagonal in a supplied Cholesky factor must fail strict positive-definiteness checks.
        mat = {1.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(
            cxxgpr::linalg::log_det_from_l(MatrixRef<double>(mat.data(), 2, 2)).error().code(),
            GprErrorCode::NonPositiveDefiniteMatrix
        );
        EXPECT_EQ(
            cxxgpr::linalg::solve_llt_in_place(MatrixRef<double>(mat.data(), 2, 2), std::span<double>(vec))
                .error()
                .code(),
            GprErrorCode::NonPositiveDefiniteMatrix
        );
    }

    /// @brief Verifies blocked Cholesky (n * sizeof(T) >= 2048) stays bit-identical across 1/2/4 threads with zero
    /// allocs.
    /// @details Factors n = 260 double and n = 520 float SPD RBF matrices under omp_set_num_threads(1), (2), and (4)
    /// inside NoAllocationScope(AuditMode::CppAndEigen) and checks exact bit equality across all lower-triangle
    /// entries.
    TEST(CholeskyLinalgTest, ParallelBlockedCholeskyBitwiseDeterministicAndZeroAlloc)
    {
        auto check_blocked = []<typename Scalar>(std::size_t n, Scalar shift)
        {
            const auto src = make_rbf_gram<Scalar>(n, shift);
            auto run_factor = [&](int threads)
            {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#else
                (void)threads;
#endif
                std::vector<Scalar> work = src;
                {
                    const NoAllocationScope audit(AuditMode::CppAndEigen);
                    EXPECT_TRUE(
                        cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<Scalar>(work.data(), n, n)).has_value()
                    );
                    EXPECT_EQ(audit.delta_allocations(), 0U);
                }
                return work;
            };

#ifdef _OPENMP
            const int saved_threads = omp_get_max_threads();
#endif
            const auto r1 = run_factor(1);
            const auto r2 = run_factor(2);
            const auto r4 = run_factor(4);
#ifdef _OPENMP
            omp_set_num_threads(saved_threads);
#endif

            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    const std::size_t idx = (col * n) + row;
                    if constexpr (std::same_as<Scalar, double>)
                    {
                        EXPECT_EQ(std::bit_cast<std::uint64_t>(r1[idx]), std::bit_cast<std::uint64_t>(r2[idx]));
                        EXPECT_EQ(std::bit_cast<std::uint64_t>(r1[idx]), std::bit_cast<std::uint64_t>(r4[idx]));
                    }
                    else
                    {
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(r1[idx]), std::bit_cast<std::uint32_t>(r2[idx]));
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(r1[idx]), std::bit_cast<std::uint32_t>(r4[idx]));
                    }
                }
            }
        };

        check_blocked.template operator()<double>(260U, 1e-2);
        check_blocked.template operator()<float>(520U, 5e-2F);
    }

    /// @brief Verifies parallel fixed-block triangular solves (n^2 * m >= 2^24) stay bit-identical across 1/2/4
    /// threads.
    /// @details Factors a 256 x 256 double SPD matrix and solves 256 right-hand side columns across Forward, Backward,
    /// and Llt modes under 1, 2, and 4 OpenMP threads with zero heap allocations and exact residual accuracy.
    TEST(CholeskyLinalgTest, ParallelTriSolveFixedBlockBitwiseDeterministicAndZeroAlloc)
    {
        constexpr std::size_t n = 256U;
        constexpr std::size_t m = 256U;
        const auto gram = make_rbf_gram<double>(n, 1e-2);
        std::vector<double> l_buf = gram;
        ASSERT_TRUE(cxxgpr::linalg::cholesky_lower_in_place(MatrixMut<double>(l_buf.data(), n, n)).has_value());
        const MatrixRef<double> l_ref(l_buf.data(), n, n);

        std::vector<double> rhs_init(n * m, 0.0);
        for (std::size_t c = 0U; c < m; ++c)
        {
            for (std::size_t r = 0U; r < n; ++r)
            {
                rhs_init[(c * n) + r] =
                    std::sin((0.13 * static_cast<double>(r + 1U)) + (0.07 * static_cast<double>(c)));
            }
        }

        auto run_solves = [&](int threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
            std::vector<double> fwd = rhs_init;
            std::vector<double> bwd = rhs_init;
            std::vector<double> llt = rhs_init;
            {
                const NoAllocationScope audit(AuditMode::CppAndEigen);
                EXPECT_TRUE(
                    cxxgpr::linalg::solve_lower_in_place(l_ref, MatrixMut<double>(fwd.data(), n, m)).has_value()
                );
                EXPECT_TRUE(
                    cxxgpr::linalg::solve_lower_transpose_in_place(l_ref, MatrixMut<double>(bwd.data(), n, m))
                        .has_value()
                );
                EXPECT_TRUE(cxxgpr::linalg::solve_llt_in_place(l_ref, MatrixMut<double>(llt.data(), n, m)).has_value());
                EXPECT_EQ(audit.delta_allocations(), 0U);
            }
            return std::array<std::vector<double>, 3>{std::move(fwd), std::move(bwd), std::move(llt)};
        };

#ifdef _OPENMP
        const int saved_threads = omp_get_max_threads();
#endif
        const auto s1 = run_solves(1);
        const auto s2 = run_solves(2);
        const auto s4 = run_solves(4);
#ifdef _OPENMP
        omp_set_num_threads(saved_threads);
#endif

        for (std::size_t mode = 0U; mode < 3U; ++mode)
        {
            for (std::size_t idx = 0U; idx < n * m; ++idx)
            {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(s1[mode][idx]), std::bit_cast<std::uint64_t>(s2[mode][idx]));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(s1[mode][idx]), std::bit_cast<std::uint64_t>(s4[mode][idx]));
            }
        }

        // Spot-check A * x = b residual on the first column of LLT solve.
        for (std::size_t row = 0U; row < n; ++row)
        {
            double ax = 0.0;
            for (std::size_t col = 0U; col < n; ++col)
            {
                ax += gram[(col * n) + row] * s1[2][col];
            }
            EXPECT_NEAR(ax, rhs_init[row], 1e-11);
        }
    }

}  // namespace
