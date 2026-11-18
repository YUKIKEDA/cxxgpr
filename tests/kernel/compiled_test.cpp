// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, REQ-11, DES-06, DES-07
/// @file compiled_test.cpp
/// @brief Unit and property tests for compiled kernel evaluation, derivatives, and validation.
/// @details Verifies compiled composite kernels over all ten leaf families against squared-distance
/// and point-coordinate evaluation, central finite-difference gradients and Hessians, single-precision
/// zero-allocation execution, and error precedence across invalid inputs and aliased buffers.

#include "cxxgpr/kernel/kernel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Computes a dense column-major pairwise squared Euclidean distance matrix.
    /// @details Evaluates ||x_i - x_j||_2^2 for all pairs (i, j) so compiled distance-based
    /// kernel evaluation can be cross-checked directly against coordinate evaluation.
    /// @param x Column-major n x d coordinate matrix.
    /// @return Column-major n x n vector of squared Euclidean distances.
    std::vector<double> make_sq_dist(cxxgpr::MatrixRef<double> x)
    {
        const std::size_t n = x.rows();
        std::vector<double> d2(n * n, 0.0);
        // Accumulate squared coordinate differences per column-major output entry.
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                double acc = 0.0;
                for (std::size_t d = 0; d < x.cols(); ++d)
                {
                    const double diff = x(row, d) - x(col, d);
                    acc += diff * diff;
                }
                d2[(col * n) + row] = acc;
            }
        }
        return d2;
    }

    /// @brief Verifies distance vs coordinate Gram parity, triangle masking, and White cross-covariance.
    /// @details Evaluates a composite Sum/Product tree via apply() and apply_points() with
    /// Triangle::Lower, confirms strict upper sentinels stay unmodified, and checks WhiteKernel
    /// yields exact zeros on cross-covariance queries.
    TEST(CompiledKernelTest, MatchesDistAndPointsAndRespectsTriangles)
    {
        // Build a 3x2 point cloud and its explicit pairwise squared-distance matrix.
        const std::array<double, 6> x_buf{0.0, 1.0, 2.5, -0.5, 0.5, 1.5};
        const cxxgpr::MatrixRef<double> x_ref(x_buf.data(), 3, 2);
        const auto d2_buf = make_sq_dist(x_ref);
        const cxxgpr::MatrixRef<double> d2_ref(d2_buf.data(), 3, 3);

        // Compose a mixed isotropic tree covering Constant, RBF, Matern-5/2, Periodic, RQ, and White.
        const auto c = cxxgpr::ConstantKernel::create(1.8);
        const auto rbf = cxxgpr::RbfKernel::create(1.3);
        const auto m52 = cxxgpr::MaternKernel::create(0.9, cxxgpr::MaternNu::FiveHalves);
        const auto per = cxxgpr::PeriodicKernel::create(1.1, 2.2);
        const auto rq = cxxgpr::RationalQuadraticKernel::create(1.4, 2.0);
        const auto white = cxxgpr::WhiteKernel::create(0.05);
        ASSERT_TRUE(c && rbf && m52 && per && rq && white);

        const cxxgpr::KernelSpec spec = (cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf))
            + (cxxgpr::KernelSpec(*m52) + cxxgpr::KernelSpec(*per) + cxxgpr::KernelSpec(*rq)
                + cxxgpr::KernelSpec(*white));
        const auto compiled = spec.compile<double>();

        // Pre-fill outputs with sentinel -999.0 to verify Triangle::Lower preserves strict upper entries.
        std::vector<double> out_dist(9, -999.0);
        std::vector<double> out_pts(9, -999.0);
        std::vector<double> scratch(9, 0.0);
        auto nested = compiled.make_nested_buffers(3, 3);

        ASSERT_TRUE(compiled
                .apply(
                    d2_ref, cxxgpr::MatrixMut<double>(out_dist.data(), 3, 3), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(scratch.data(), 3, 3), nested
                )
                .has_value());
        ASSERT_TRUE(compiled
                .apply_points(
                    x_ref, cxxgpr::MatrixMut<double>(out_pts.data(), 3, 3), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(scratch.data(), 3, 3), nested
                )
                .has_value());

        for (std::size_t col = 0; col < 3; ++col)
        {
            for (std::size_t row = 0; row < 3; ++row)
            {
                if (row < col)
                {
                    // Strict upper entries remain untouched when Triangle::Lower is requested.
                    EXPECT_DOUBLE_EQ(out_dist[(col * 3) + row], -999.0);
                    EXPECT_DOUBLE_EQ(out_pts[(col * 3) + row], -999.0);
                }
                else
                {
                    EXPECT_NEAR(out_dist[(col * 3) + row], out_pts[(col * 3) + row], 1e-13);
                }
            }
        }

        // WhiteKernel is zero on cross-covariance.
        const auto w_compiled = cxxgpr::KernelSpec(*white).compile<double>();
        std::vector<double> w_cross(6, -1.0);
        std::vector<double> w_scratch(6, 0.0);
        const std::array<double, 4> x2_buf{0.0, 1.0, -0.5, 0.5};
        ASSERT_TRUE(w_compiled
                .apply_cross_points(
                    x_ref, cxxgpr::MatrixRef<double>(x2_buf.data(), 2, 2),
                    cxxgpr::MatrixMut<double>(w_cross.data(), 3, 2), cxxgpr::MatrixMut<double>(w_scratch.data(), 3, 2)
                )
                .has_value());
        for (double val : w_cross)
        {
            EXPECT_DOUBLE_EQ(val, 0.0);
        }
    }

    /// @brief Verifies analytical gradients and Hessians of all leaf families against finite differences.
    /// @details Exercises a nested Product/Sum composite spanning all ten kernel leaf variants
    /// including coincident points (row 0 == row 1) and checks every parameter derivative and
    /// cross-Hessian against central finite differences.
    TEST(CompiledKernelTest, AllLeavesAnalyticalGradAndHessMatchFiniteDifferences)
    {
        const std::array<double, 6> x_buf{0.2, 0.2, 1.7, -0.4, -0.4, 0.9};
        const cxxgpr::MatrixRef<double> x_ref(x_buf.data(), 3, 2);

        const auto c = cxxgpr::ConstantKernel::create(1.7);
        const auto rbf = cxxgpr::RbfKernel::create(1.1);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.9, 1.6});
        const auto m12 = cxxgpr::MaternKernel::create(1.2, cxxgpr::MaternNu::OneHalf);
        const auto m32 = cxxgpr::MaternKernel::create(1.3, cxxgpr::MaternNu::ThreeHalves);
        const auto m52 = cxxgpr::MaternKernel::create(1.4, cxxgpr::MaternNu::FiveHalves);
        const auto m_ard12 =
            cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.8, 1.5}, cxxgpr::MaternNu::OneHalf);
        const auto m_ard32 =
            cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.8, 1.5}, cxxgpr::MaternNu::ThreeHalves);
        const auto m_ard52 =
            cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.8, 1.5}, cxxgpr::MaternNu::FiveHalves);
        const auto per = cxxgpr::PeriodicKernel::create(1.2, 2.4);
        const auto rq = cxxgpr::RationalQuadraticKernel::create(1.1, 1.8);
        const auto rq_ard = cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.0, 1.7}, 2.2);
        const auto lin = cxxgpr::LinearKernel::create(0.7);
        const auto white = cxxgpr::WhiteKernel::create(0.15);
        ASSERT_TRUE(
            c && rbf && rbf_ard && m12 && m32 && m52 && m_ard12 && m_ard32 && m_ard52 && per && rq && rq_ard && lin
            && white
        );

        // Composite expression exercises nested Product + Sum across isotropic, ARD, Linear, White.
        const cxxgpr::KernelSpec tree =
            (cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf_ard) * cxxgpr::KernelSpec(*m_ard52))
            + (cxxgpr::KernelSpec(*rbf) * cxxgpr::KernelSpec(*per)) + cxxgpr::KernelSpec(*m12)
            + cxxgpr::KernelSpec(*m32) + cxxgpr::KernelSpec(*m52) + cxxgpr::KernelSpec(*m_ard12)
            + cxxgpr::KernelSpec(*m_ard32) + cxxgpr::KernelSpec(*rq) + cxxgpr::KernelSpec(*rq_ard)
            + cxxgpr::KernelSpec(*lin) + cxxgpr::KernelSpec(*white);

        auto compiled = tree.compile<double>();
        const std::size_t np = compiled.num_params();
        std::vector<double> theta(np);
        ASSERT_TRUE(compiled.get_params(theta).has_value());

        std::vector<double> k_mat(9, 0.0);
        std::vector<double> dk(9, 0.0);
        std::vector<double> d2k(9, 0.0);
        std::vector<double> sa(9, 0.0);
        std::vector<double> sb(9, 0.0);
        std::vector<double> sc(9, 0.0);
        auto nested = compiled.make_nested_buffers(3, 3);

        const double eps = 1e-6;
        auto eval_at = [&](const std::vector<double>& p, std::vector<double>& out)
        {
            ASSERT_TRUE(compiled.set_params(p).has_value());
            ASSERT_TRUE(compiled
                    .apply_points(
                        x_ref, cxxgpr::MatrixMut<double>(out.data(), 3, 3), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), nested
                    )
                    .has_value());
        };

        // Check analytical first-order log-parameter gradients against central differences.
        std::vector<double> kp(9, 0.0);
        std::vector<double> km(9, 0.0);
        for (std::size_t i = 0; i < np; ++i)
        {
            ASSERT_TRUE(compiled.set_params(theta).has_value());
            ASSERT_TRUE(compiled
                    .grad_points(
                        x_ref, cxxgpr::MatrixMut<double>(dk.data(), 3, 3), i, cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3), nested
                    )
                    .has_value());
            // Zero-distance points (row 0 vs row 1) must have continuous limit 0 for Matern/Periodic.
            auto tp = theta;
            auto tm = theta;
            tp[i] += eps;
            tm[i] -= eps;
            eval_at(tp, kp);
            eval_at(tm, km);
            for (std::size_t idx = 0; idx < 9U; ++idx)
            {
                const double fd = (kp[idx] - km[idx]) / (2.0 * eps);
                EXPECT_NEAR(dk[idx], fd, 1e-7) << "grad mismatch at param " << i << ", entry " << idx;
            }
        }

        auto eval_grad_at = [&](const std::vector<double>& p, std::size_t idx_p, std::vector<double>& out)
        {
            ASSERT_TRUE(compiled.set_params(p).has_value());
            ASSERT_TRUE(compiled
                    .grad_points(
                        x_ref, cxxgpr::MatrixMut<double>(out.data(), 3, 3), idx_p, cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3), nested
                    )
                    .has_value());
        };

        // Check analytical second-order log-parameter Hessians against central differences of gradients.
        for (std::size_t i = 0; i < np; ++i)
        {
            for (std::size_t j = 0; j < np; ++j)
            {
                ASSERT_TRUE(compiled.set_params(theta).has_value());
                ASSERT_TRUE(compiled
                        .hess_points(
                            x_ref, cxxgpr::MatrixMut<double>(d2k.data(), 3, 3), i, j, cxxgpr::Triangle::Full,
                            cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3),
                            cxxgpr::MatrixMut<double>(sc.data(), 3, 3), nested
                        )
                        .has_value());
                auto tp = theta;
                auto tm = theta;
                tp[j] += eps;
                tm[j] -= eps;
                eval_grad_at(tp, i, kp);
                eval_grad_at(tm, i, km);
                for (std::size_t idx = 0; idx < 9U; ++idx)
                {
                    const double fd = (kp[idx] - km[idx]) / (2.0 * eps);
                    EXPECT_NEAR(d2k[idx], fd, 1e-6) << "hess mismatch at (" << i << ", " << j << "), entry " << idx;
                }
            }
        }
        (void)k_mat;
    }

    /// @brief Verifies single-precision kernel evaluation and zero-allocation hot-path compliance.
    /// @details Compiles a composite Constant * RBF + White kernel to float and asserts zero heap
    /// allocations inside NoAllocationScope across apply(), grad(), and hess().
    TEST(CompiledKernelTest, SinglePrecisionAndZeroAllocationHotPathWork)
    {
        const auto c = cxxgpr::ConstantKernel::create(2.0);
        const auto rbf = cxxgpr::RbfKernel::create(1.5);
        const auto white = cxxgpr::WhiteKernel::create(0.1);
        ASSERT_TRUE(c && rbf && white);

        const cxxgpr::KernelSpec spec =
            (cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf)) + cxxgpr::KernelSpec(*white);
        const auto compiled_f = spec.compile<float>();

        const std::array<float, 4> d2{0.0F, 1.0F, 1.0F, 0.0F};
        const cxxgpr::MatrixRef<float> d2_ref(d2.data(), 2, 2);
        std::array<float, 4> out{};
        std::array<float, 4> sa{};
        std::array<float, 4> sb{};
        std::array<float, 4> sc{};
        auto nested = compiled_f.make_nested_buffers(2, 2);

        // Warm up and verify zero heap allocations during apply/grad/hess.
        ASSERT_TRUE(compiled_f
                .apply(
                    d2_ref, cxxgpr::MatrixMut<float>(out.data(), 2, 2), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<float>(sa.data(), 2, 2), nested
                )
                .has_value());
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(compiled_f
                    .apply(
                        d2_ref, cxxgpr::MatrixMut<float>(out.data(), 2, 2), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<float>(sa.data(), 2, 2), nested
                    )
                    .has_value());
            ASSERT_TRUE(compiled_f
                    .grad(
                        d2_ref, cxxgpr::MatrixMut<float>(out.data(), 2, 2), 1U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<float>(sa.data(), 2, 2), cxxgpr::MatrixMut<float>(sb.data(), 2, 2), nested
                    )
                    .has_value());
            ASSERT_TRUE(compiled_f
                    .hess(
                        d2_ref, cxxgpr::MatrixMut<float>(out.data(), 2, 2), 0U, 1U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<float>(sa.data(), 2, 2), cxxgpr::MatrixMut<float>(sb.data(), 2, 2),
                        cxxgpr::MatrixMut<float>(sc.data(), 2, 2), nested
                    )
                    .has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }
    }

    /// @brief Negative controls verifying invalid inputs, aliased buffers, and DES-02 error ordering.
    /// @details Checks UnsupportedKernelOperation for coordinate-only kernels or invalid cross-Hessians,
    /// InvalidConfig on aliased scratch/output buffers, NonFiniteInput on NaN distances, ARD dimension
    /// mismatches, nested fill_diag() accuracy, and EmptyInput -> LengthMismatch -> DimensionMismatch order.
    TEST(CompiledKernelTest, RejectsInvalidInputsAndAliasedBuffers)
    {
        const auto rbf = cxxgpr::RbfKernel::create(1.0);
        const auto lin = cxxgpr::LinearKernel::create(1.0);
        ASSERT_TRUE(rbf && lin);

        // Coordinate-only LinearKernel rejects distance-only apply().
        const auto lin_c = cxxgpr::KernelSpec(*lin).compile<double>();
        std::array<double, 4> d2{0.0, 1.0, 1.0, 0.0};
        std::array<double, 4> out{};
        std::array<double, 4> scratch{};
        EXPECT_EQ(
            lin_c
                .apply(
                    cxxgpr::MatrixRef<double>(d2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2),
                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(scratch.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::UnsupportedKernelOperation
        );

        const auto rbf_c = cxxgpr::KernelSpec(*rbf).compile<double>();
        EXPECT_EQ(
            rbf_c
                .apply(
                    cxxgpr::MatrixRef<double>(d2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2),
                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(out.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::InvalidConfig
        );

        std::array<double, 4> bad_d2{0.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0};
        EXPECT_EQ(
            rbf_c
                .apply(
                    cxxgpr::MatrixRef<double>(bad_d2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2),
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::NonFiniteInput
        );

        // SumTerm cross-Hessian and gradient must still validate CoordMode and all ARD dimensions.
        const auto c = cxxgpr::ConstantKernel::create(2.0);
        const auto rbf_ard3 = cxxgpr::RbfArdKernel::create(std::array<double, 3>{1.0, 2.0, 3.0});
        ASSERT_TRUE(c && rbf_ard3);

        const auto ard_plus_c = (cxxgpr::KernelSpec(*rbf_ard3) + cxxgpr::KernelSpec(*c)).compile<double>();
        std::array<double, 4> sa{};
        std::array<double, 4> sb{};
        std::array<double, 4> sc{};
        EXPECT_EQ(
            ard_plus_c
                .hess(
                    cxxgpr::MatrixRef<double>(d2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2), 0U, 3U,
                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), 2, 2),
                    cxxgpr::MatrixMut<double>(sb.data(), 2, 2), cxxgpr::MatrixMut<double>(sc.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::UnsupportedKernelOperation
        );

        const auto iso_plus_ard3 = (cxxgpr::KernelSpec(*rbf) + cxxgpr::KernelSpec(*rbf_ard3)).compile<double>();
        const std::array<double, 4> x_dim2{0.0, 1.0, 0.5, 1.5};
        EXPECT_EQ(
            iso_plus_ard3
                .hess_points(
                    cxxgpr::MatrixRef<double>(x_dim2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2), 0U, 1U,
                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), 2, 2),
                    cxxgpr::MatrixMut<double>(sb.data(), 2, 2), cxxgpr::MatrixMut<double>(sc.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::DimensionMismatch
        );
        EXPECT_EQ(
            iso_plus_ard3
                .grad_points(
                    cxxgpr::MatrixRef<double>(x_dim2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2), 0U,
                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), 2, 2),
                    cxxgpr::MatrixMut<double>(sb.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::DimensionMismatch
        );

        // Nested fill_diag evaluates compound sums/products accurately using preallocated scratch.
        const auto nested_spec =
            (cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf)) + (cxxgpr::KernelSpec(*c) + cxxgpr::KernelSpec(*lin));
        const auto nested_c = nested_spec.compile<double>();
        std::array<double, 2> diag{};
        ASSERT_TRUE(nested_c.fill_diag(cxxgpr::MatrixRef<double>(x_dim2.data(), 2, 2), diag).has_value());
        EXPECT_DOUBLE_EQ(diag[0], 2.0 + 2.0 + ((0.0 * 0.0) + (0.5 * 0.5)));
        EXPECT_DOUBLE_EQ(diag[1], 2.0 + 2.0 + ((1.0 * 1.0) + (1.5 * 1.5)));

        // Compound violations follow DES-02 precedence: EmptyInput -> LengthMismatch ->
        // DimensionMismatch -> NonFiniteInput.
        const std::array<double, 4> nan_dim2{std::numeric_limits<double>::quiet_NaN(), 1.0, 0.5, 1.5};
        std::array<double, 1> bad_shape_out{};
        std::array<double, 1> bad_shape_scratch{};
        EXPECT_EQ(
            iso_plus_ard3
                .apply_points(
                    cxxgpr::MatrixRef<double>(nan_dim2.data(), 2, 2),
                    cxxgpr::MatrixMut<double>(bad_shape_out.data(), 1, 1), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(bad_shape_scratch.data(), 1, 1)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );
        EXPECT_EQ(
            iso_plus_ard3
                .apply_points(
                    cxxgpr::MatrixRef<double>(nan_dim2.data(), 2, 2), cxxgpr::MatrixMut<double>(out.data(), 2, 2),
                    cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), 2, 2)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::DimensionMismatch
        );

        const cxxgpr::MatrixRef<double> empty_x(nullptr, 0, 0);
        EXPECT_EQ(
            iso_plus_ard3.fill_diag(empty_x, std::span<double>{}).error().code(), cxxgpr::GprErrorCode::EmptyInput
        );
        EXPECT_EQ(
            iso_plus_ard3.fill_diag(cxxgpr::MatrixRef<double>(nan_dim2.data(), 2, 2), std::span<double>{bad_shape_out})
                .error()
                .code(),
            cxxgpr::GprErrorCode::LengthMismatch
        );
        EXPECT_EQ(
            iso_plus_ard3.fill_diag(cxxgpr::MatrixRef<double>(nan_dim2.data(), 2, 2), std::span<double>{diag})
                .error()
                .code(),
            cxxgpr::GprErrorCode::DimensionMismatch
        );
    }

}  // namespace
