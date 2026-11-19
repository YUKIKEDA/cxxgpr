// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, REQ-11, DES-08
/// @file exp_test.cpp
/// @brief Unit and property tests for fast exponential approximation and kernel exp jet derivatives.
/// @details Verifies KernelExp::FastApprox scalar accuracy against Accurate in double and float
/// across [-50, 0] and extreme values, jet first/second derivative consistency, compiled kernel
/// FastApprox self-consistency against finite differences, and DistanceKernel support.

#include "cxxgpr/kernel/kernel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace
{

    /// @brief Computes a column-major pairwise squared Euclidean distance matrix.
    /// @details Evaluates ||x_i - x_j||_2^2 for validating DistanceKernel under FastApprox.
    /// @param[in] x Column-major n x d coordinate matrix.
    /// @return Column-major n x n vector of squared Euclidean distances.
    std::vector<double> make_sq_dist(cxxgpr::MatrixRef<double> x)
    {
        const std::size_t n = x.rows();
        std::vector<double> d2(n * n, 0.0);
        // Accumulate squared differences across all dimensions.
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

    /// @brief Computes a column-major matrix of per-dimension squared coordinate differences.
    /// @details Evaluates (x_{i,dim} - x_{j,dim})^2 for binding ArdDistance slots in exp tests.
    /// @param[in] x Column-major n x d coordinate matrix.
    /// @param[in] dim Zero-based coordinate dimension index.
    /// @return Column-major n x n vector of single-dimension squared differences.
    std::vector<double> make_dim_sq_diff(cxxgpr::MatrixRef<double> x, std::size_t dim)
    {
        const std::size_t n = x.rows();
        std::vector<double> out(n * n, 0.0);
        // Store squared difference along the selected feature column.
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                const std::size_t r1 = row;
                const std::size_t r2 = col;
                const double diff = x(r1, dim) - x(r2, dim);
                out[(col * n) + row] = diff * diff;
            }
        }
        return out;
    }

    /// @brief Verifies double-precision FastApprox exponential accuracy across [-50, 0] and extremes.
    /// @details Checks relative error < 1.2e-7 against KernelExp::Accurate and probes exact 0.0,
    /// underflow (-800), overflow (+800), and quiet NaN IEEE-754 edge inputs.
    TEST(KernelExpTest, FastApproxDoubleAccuracyAndExtremes)
    {
        // Sweep negative kernel arguments [-50, 0] and check relative error against std::exp oracle.
        for (int i = 0; i <= 500; ++i)
        {
            const double x = -0.1 * static_cast<double>(i);
            const double exact = cxxgpr::eval_kernel_exp(x, cxxgpr::KernelExp::Accurate);
            const double approx = cxxgpr::eval_kernel_exp(x, cxxgpr::KernelExp::FastApprox);
            ASSERT_GT(exact, 0.0);
            const double rel = std::abs(approx - exact) / exact;
            EXPECT_LT(rel, 1.2e-7) << "x=" << x;
        }

        // Probe IEEE-754 boundary cases: identity at 0, underflow to 0, overflow to +inf, NaN pass-through.
        EXPECT_DOUBLE_EQ(cxxgpr::eval_kernel_exp(0.0, cxxgpr::KernelExp::FastApprox), 1.0);
        EXPECT_DOUBLE_EQ(cxxgpr::eval_kernel_exp(-800.0, cxxgpr::KernelExp::FastApprox), 0.0);
        EXPECT_TRUE(std::isinf(cxxgpr::eval_kernel_exp(800.0, cxxgpr::KernelExp::FastApprox)));
        EXPECT_TRUE(
            std::isnan(cxxgpr::eval_kernel_exp(std::numeric_limits<double>::quiet_NaN(), cxxgpr::KernelExp::FastApprox))
        );
    }

    /// @brief Verifies single-precision FastApprox exponential accuracy across [-50, 0] and extremes.
    /// @details Checks relative error < 1.0e-6F above subnormal range and probes 0, -120, +120, and NaN.
    TEST(KernelExpTest, FastApproxFloatAccuracyAndExtremes)
    {
        // Sweep [-50, 0] in single precision excluding subnormal underflow regime.
        for (int i = 0; i <= 500; ++i)
        {
            const float x = -0.1F * static_cast<float>(i);
            const float exact = cxxgpr::eval_kernel_exp(x, cxxgpr::KernelExp::Accurate);
            const float approx = cxxgpr::eval_kernel_exp(x, cxxgpr::KernelExp::FastApprox);
            if (exact > 1e-35F)
            {
                const float rel = std::abs(approx - exact) / exact;
                EXPECT_LT(rel, 1.0e-6F) << "x=" << x;
            }
        }

        // Probe single-precision edge arguments: 0.0F, underflow, overflow, and quiet NaN.
        EXPECT_FLOAT_EQ(cxxgpr::eval_kernel_exp(0.0F, cxxgpr::KernelExp::FastApprox), 1.0F);
        EXPECT_FLOAT_EQ(cxxgpr::eval_kernel_exp(-120.0F, cxxgpr::KernelExp::FastApprox), 0.0F);
        EXPECT_TRUE(std::isinf(cxxgpr::eval_kernel_exp(120.0F, cxxgpr::KernelExp::FastApprox)));
        EXPECT_TRUE(
            std::isnan(cxxgpr::eval_kernel_exp(std::numeric_limits<float>::quiet_NaN(), cxxgpr::KernelExp::FastApprox))
        );
    }

    /// @brief Verifies FastApprox exponential jet first and second derivatives match finite differences.
    /// @details Checks eval_kernel_exp_jet() d1 and d2 values against central differences of v and d1.
    TEST(KernelExpTest, ExpJetDerivativesMatchFiniteDifferences)
    {
        // Compare analytical jet derivatives d1 and d2 against central finite differences.
        constexpr double h = 1e-5;
        for (double x : {-12.3, -2.5, -0.37, 0.0})
        {
            const auto jet = cxxgpr::eval_kernel_exp_jet(x, cxxgpr::KernelExp::FastApprox);
            const auto plus = cxxgpr::eval_kernel_exp_jet(x + h, cxxgpr::KernelExp::FastApprox);
            const auto minus = cxxgpr::eval_kernel_exp_jet(x - h, cxxgpr::KernelExp::FastApprox);
            const double fd1 = (plus.v - minus.v) / (2.0 * h);
            const double fd2 = (plus.d1 - minus.d1) / (2.0 * h);
            EXPECT_NEAR(jet.d1, fd1, 1e-8 * std::max(1.0, std::abs(jet.d1))) << "x=" << x;
            EXPECT_NEAR(jet.d2, fd2, 1e-7 * std::max(1.0, std::abs(jet.d2))) << "x=" << x;
        }
    }

    /// @brief Verifies CompiledKernel FastApprox value parity and analytical derivative consistency.
    /// @details Evaluates nine built-in exponential kernel families under Accurate and FastApprox,
    /// checks parameter roundtrip, and compares FastApprox gradients and Hessians to finite differences.
    TEST(KernelExpTest, CompiledKernelFastApproxGradAndHessSelfConsistent)
    {
        const std::vector<double> x_data{0.0, 0.4, 1.2, -0.7, 0.2, 0.9, -0.4, 1.5};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 4, 2, 4);

        // Cover every isotropic and ARD kernel family whose evaluation invokes eval_kernel_exp.
        const std::vector<cxxgpr::KernelSpec> specs{
            cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.1).value()),
            cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.9, 1.4}).value()),
            cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.2, cxxgpr::MaternNu::OneHalf).value()),
            cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.2, cxxgpr::MaternNu::ThreeHalves).value()),
            cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(1.2, cxxgpr::MaternNu::FiveHalves).value()),
            cxxgpr::KernelSpec(
                cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.8, 1.5}, cxxgpr::MaternNu::ThreeHalves).value()
            ),
            cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(0.8, 2.3).value()),
            cxxgpr::KernelSpec(cxxgpr::RationalQuadraticKernel::create(1.1, 1.7).value()),
            cxxgpr::KernelSpec(
                cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{0.9, 1.3}, 1.8).value()
            ),
        };

        constexpr std::size_t n = 4;
        constexpr double h = 1e-5;
        std::vector<double> k_acc(n * n);
        std::vector<double> k_fast(n * n);
        std::vector<double> k_plus(n * n);
        std::vector<double> k_minus(n * n);
        std::vector<double> g_mat(n * n);
        std::vector<double> g_plus(n * n);
        std::vector<double> g_minus(n * n);
        std::vector<double> h_mat(n * n);
        std::vector<double> sc_a(n * n);
        std::vector<double> sc_b(n * n);
        std::vector<double> sc_c(n * n);

        for (const auto& spec : specs)
        {
            auto ck = cxxgpr::CompiledKernel<double>::from_spec(spec);
            EXPECT_EQ(ck.kernel_exp(), cxxgpr::KernelExp::Accurate);
            ASSERT_TRUE(ck
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(k_acc.data(), n, n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                    )
                    .has_value());

            ck.set_kernel_exp(cxxgpr::KernelExp::FastApprox);
            EXPECT_EQ(ck.kernel_exp(), cxxgpr::KernelExp::FastApprox);
            ASSERT_TRUE(ck
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(k_fast.data(), n, n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                    )
                    .has_value());

            for (std::size_t idx = 0; idx < k_acc.size(); ++idx)
            {
                EXPECT_NEAR(k_fast[idx], k_acc[idx], 5e-7 * std::max(1.0, std::abs(k_acc[idx])));
            }

            const std::size_t np = ck.num_params();
            std::vector<double> theta(np);
            ASSERT_TRUE(ck.get_params(theta).has_value());
            std::vector<double> theta_copy = theta;
            ASSERT_TRUE(ck.set_params(theta_copy).has_value());
            std::vector<double> theta_rt(np);
            ASSERT_TRUE(ck.get_params(theta_rt).has_value());
            // Verify parameter roundtrip stays bit-exact after setting FastApprox.
            for (std::size_t p = 0; p < np; ++p)
            {
                EXPECT_DOUBLE_EQ(theta_rt[p], theta[p]);
            }

            // Verify FastApprox gradients and Hessians match central finite differences.
            for (std::size_t p = 0; p < np; ++p)
            {
                ASSERT_TRUE(ck
                        .grad_points(
                            x, cxxgpr::MatrixMut<double>(g_mat.data(), n, n, n), p, cxxgpr::Triangle::Full,
                            cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                            cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                        )
                        .has_value());

                auto tp = theta;
                tp[p] = theta[p] + h;
                ASSERT_TRUE(ck.set_params(tp).has_value());
                ASSERT_TRUE(ck
                        .apply_points(
                            x, cxxgpr::MatrixMut<double>(k_plus.data(), n, n, n), cxxgpr::Triangle::Full,
                            cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                        )
                        .has_value());

                auto tm = theta;
                tm[p] = theta[p] - h;
                ASSERT_TRUE(ck.set_params(tm).has_value());
                ASSERT_TRUE(ck
                        .apply_points(
                            x, cxxgpr::MatrixMut<double>(k_minus.data(), n, n, n), cxxgpr::Triangle::Full,
                            cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                        )
                        .has_value());
                ASSERT_TRUE(ck.set_params(theta).has_value());

                for (std::size_t idx = 0; idx < g_mat.size(); ++idx)
                {
                    const double fd = (k_plus[idx] - k_minus[idx]) / (2.0 * h);
                    EXPECT_NEAR(g_mat[idx], fd, 5e-6);
                }
            }

            for (std::size_t i = 0; i < np; ++i)
            {
                for (std::size_t j = 0; j < np; ++j)
                {
                    ASSERT_TRUE(ck
                            .hess_points(
                                x, cxxgpr::MatrixMut<double>(h_mat.data(), n, n, n), i, j, cxxgpr::Triangle::Full,
                                cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_c.data(), n, n, n)
                            )
                            .has_value());

                    auto tp = theta;
                    tp[j] = theta[j] + h;
                    ASSERT_TRUE(ck.set_params(tp).has_value());
                    ASSERT_TRUE(ck
                            .grad_points(
                                x, cxxgpr::MatrixMut<double>(g_plus.data(), n, n, n), i, cxxgpr::Triangle::Full,
                                cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                            )
                            .has_value());

                    auto tm = theta;
                    tm[j] = theta[j] - h;
                    ASSERT_TRUE(ck.set_params(tm).has_value());
                    ASSERT_TRUE(ck
                            .grad_points(
                                x, cxxgpr::MatrixMut<double>(g_minus.data(), n, n, n), i, cxxgpr::Triangle::Full,
                                cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                                cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                            )
                            .has_value());
                    ASSERT_TRUE(ck.set_params(theta).has_value());

                    for (std::size_t idx = 0; idx < h_mat.size(); ++idx)
                    {
                        const double fd = (g_plus[idx] - g_minus[idx]) / (2.0 * h);
                        EXPECT_NEAR(h_mat[idx], fd, 5e-5);
                    }
                }
            }
        }
    }

    /// @brief Verifies DistanceKernel supports FastApprox across scalar and ARD distance slots.
    /// @details Evaluates a ScalarDistance RBF * ArdDistance RationalQuadratic product under both
    /// FastApprox and Accurate modes and checks gradient consistency via central finite differences.
    TEST(KernelExpTest, CompiledDistanceKernelFastApproxArdAndScalar)
    {
        const std::vector<double> x_data{0.1, 0.5, 1.3, -0.3, 0.4, 1.1};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 3, 2, 3);
        const std::vector<double> d2 = make_sq_dist(x);
        const std::vector<double> ard0 = make_dim_sq_diff(x, 0U);
        const std::vector<double> ard1 = make_dim_sq_diff(x, 1U);
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_refs{
            cxxgpr::MatrixRef<double>(ard0.data(), 3, 3, 3),
            cxxgpr::MatrixRef<double>(ard1.data(), 3, 3, 3),
        };

        // Construct a supplied-distance product tree and switch it to FastApprox mode.
        cxxgpr::ScalarDistance sdist;
        auto ard_dist = cxxgpr::ArdDistance::create(2).value();
        auto dk = sdist.rbf(cxxgpr::RbfKernel::create(1.0).value())
            * ard_dist
                  .rational_quadratic_ard(
                      cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{0.8, 1.3}, 1.5).value()
                  )
                  .value();

        auto compiled = dk.compile<double>();
        EXPECT_EQ(compiled.kernel_exp(), cxxgpr::KernelExp::Accurate);
        compiled.set_kernel_exp(cxxgpr::KernelExp::FastApprox);
        EXPECT_EQ(compiled.kernel_exp(), cxxgpr::KernelExp::FastApprox);

        cxxgpr::DistanceSource src;
        src.bind_scalar(sdist.slot(), cxxgpr::MatrixRef<double>(d2.data(), 3, 3, 3));
        src.bind_ard(ard_dist.slot(), ard_refs);
        auto checked = compiled.prepare_square(src);
        ASSERT_TRUE(checked.has_value());
        if (!checked.has_value())
        {
            return;
        }

        constexpr std::size_t n = 3;
        std::vector<double> k_fast(n * n);
        std::vector<double> k_acc(n * n);
        std::vector<double> g_mat(n * n);
        std::vector<double> h_mat(n * n);
        std::vector<double> sc_a(n * n);
        std::vector<double> sc_b(n * n);
        std::vector<double> sc_c(n * n);

        ASSERT_TRUE(compiled
                .apply(
                    *checked, std::nullopt, cxxgpr::MatrixMut<double>(k_fast.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                )
                .has_value());

        compiled.set_kernel_exp(cxxgpr::KernelExp::Accurate);
        ASSERT_TRUE(compiled
                .apply(
                    *checked, std::nullopt, cxxgpr::MatrixMut<double>(k_acc.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                )
                .has_value());

        for (std::size_t idx = 0; idx < k_fast.size(); ++idx)
        {
            EXPECT_NEAR(k_fast[idx], k_acc[idx], 5e-7 * std::max(1.0, std::abs(k_acc[idx])));
        }

        compiled.set_kernel_exp(cxxgpr::KernelExp::FastApprox);
        ASSERT_TRUE(compiled
                .grad(
                    *checked, std::nullopt, cxxgpr::MatrixMut<double>(g_mat.data(), n, n, n), 1, cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                )
                .has_value());
        ASSERT_TRUE(compiled
                .hess(
                    *checked, std::nullopt, cxxgpr::MatrixMut<double>(h_mat.data(), n, n, n), 1, 3,
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                    cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), cxxgpr::MatrixMut<double>(sc_c.data(), n, n, n)
                )
                .has_value());

        constexpr double h = 1e-5;
        std::vector<double> theta(compiled.num_params());
        ASSERT_TRUE(compiled.get_params(theta).has_value());
        std::vector<double> k_plus(n * n);
        std::vector<double> k_minus(n * n);

        auto tp = theta;
        tp[1] = theta[1] + h;
        ASSERT_TRUE(compiled.set_params(tp).has_value());
        ASSERT_TRUE(compiled
                .apply(
                    *checked, std::nullopt, cxxgpr::MatrixMut<double>(k_plus.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                )
                .has_value());
        auto tm = theta;
        tm[1] = theta[1] - h;
        ASSERT_TRUE(compiled.set_params(tm).has_value());
        ASSERT_TRUE(compiled
                .apply(
                    *checked, std::nullopt, cxxgpr::MatrixMut<double>(k_minus.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                )
                .has_value());
        ASSERT_TRUE(compiled.set_params(theta).has_value());

        for (std::size_t idx = 0; idx < g_mat.size(); ++idx)
        {
            EXPECT_NEAR(g_mat[idx], (k_plus[idx] - k_minus[idx]) / (2.0 * h), 5e-6);
        }
    }

}  // namespace
