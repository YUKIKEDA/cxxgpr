// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-10, DES-06, DES-07, DES-10

#include "cxxgpr/kernel/kernel.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    // Inverse-quadratic kernel: k(d2) = (1 + d2 / ell^2)^(-1)
    template <cxxgpr::KernelScalarConcept T> class InvQuadTerm final : public cxxgpr::KernelTerm<T>
    {
    public:
        explicit InvQuadTerm(double ell = 1.0) : ell_(ell) {}

        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<T>> clone() const override
        {
            return std::make_unique<InvQuadTerm<T>>(ell_);
        }

        [[nodiscard]] std::size_t num_params() const noexcept override { return 1U; }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            if (out.size() != 1U)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("out"));
            }
            out[0] = std::log(ell_);
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            if (params.size() != 1U)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("params"));
            }
            if (!std::isfinite(params[0]))
            {
                return std::unexpected(cxxgpr::GprError::non_finite_input());
            }
            const double ell = std::exp(params[0]);
            if (!std::isfinite(ell) || ell < 0.05 || ell > 50.0)
            {
                return std::unexpected(cxxgpr::GprError::invalid_config("ell out of bounds"));
            }
            ell_ = ell;
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            if (out.size() != 1U)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("out"));
            }
            auto b = cxxgpr::Interval::create(0.05, 50.0);
            if (!b.has_value())
            {
                return std::unexpected(std::move(b).error());
            }
            out[0] = *b;
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo
        ) const override
        {
            const T inv_l2 = T{1} / static_cast<T>(ell_ * ell_);
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if (uplo == cxxgpr::Triangle::Lower && row < col)
                    {
                        continue;
                    }
                    if (uplo == cxxgpr::Triangle::Upper && row > col)
                    {
                        continue;
                    }
                    const T d2 = dist(row, col) < T{0} ? T{0} : dist(row, col);
                    out(row, col) = T{1} / (T{1} + (d2 * inv_l2));
                }
            }
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out
        ) const override
        {
            return apply(dist, out, cxxgpr::Triangle::Full);
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<T> out) const override
        {
            std::fill(out.begin(), out.end(), T{1});
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d_k, std::size_t param_idx, cxxgpr::Triangle uplo
        ) const override
        {
            if (param_idx != 0U)
            {
                return std::unexpected(cxxgpr::GprError::index_out_of_range("param_idx"));
            }
            const T inv_l2 = T{1} / static_cast<T>(ell_ * ell_);
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if (uplo == cxxgpr::Triangle::Lower && row < col)
                    {
                        continue;
                    }
                    if (uplo == cxxgpr::Triangle::Upper && row > col)
                    {
                        continue;
                    }
                    const T d2 = dist(row, col) < T{0} ? T{0} : dist(row, col);
                    const T w = d2 * inv_l2;
                    const T b = T{1} + w;
                    // d/d(log ell) (1 + w)^(-1) = 2 * w / (1 + w)^2
                    d_k(row, col) = (T{2} * w) / (b * b);
                }
            }
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d2_k, std::size_t i, std::size_t j, cxxgpr::Triangle uplo
        ) const override
        {
            if (i != 0U || j != 0U)
            {
                return std::unexpected(cxxgpr::GprError::index_out_of_range("param_idx"));
            }
            const T inv_l2 = T{1} / static_cast<T>(ell_ * ell_);
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if (uplo == cxxgpr::Triangle::Lower && row < col)
                    {
                        continue;
                    }
                    if (uplo == cxxgpr::Triangle::Upper && row > col)
                    {
                        continue;
                    }
                    const T d2 = dist(row, col) < T{0} ? T{0} : dist(row, col);
                    const T w = d2 * inv_l2;
                    const T b = T{1} + w;
                    // d2/d(log ell)^2 (1 + w)^(-1) = 4 * w * (w - 1) / (1 + w)^3
                    d2_k(row, col) = (T{4} * w * (w - T{1})) / (b * b * b);
                }
            }
            return {};
        }

        [[nodiscard]] std::string_view persist_id() const noexcept override { return "inv_quad"; }

    private:
        double ell_;
    };

    std::vector<double> make_sq_dist(cxxgpr::MatrixRef<double> x)
    {
        const std::size_t n = x.rows();
        std::vector<double> d2(n * n, 0.0);
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

    TEST(CustomKernelTest, ValidationAndValueSemantics)
    {
        auto bad_null = cxxgpr::CustomKernel::create(nullptr, std::make_unique<InvQuadTerm<float>>());
        EXPECT_FALSE(bad_null.has_value());
        EXPECT_EQ(bad_null.error().code(), cxxgpr::GprErrorCode::InvalidConfig);

        auto ck_res = cxxgpr::make_custom_kernel<InvQuadTerm>(1.5);
        ASSERT_TRUE(ck_res.has_value());
        if (!ck_res.has_value())
        {
            return;
        }

        const cxxgpr::CustomKernel& a = *ck_res;
        cxxgpr::CustomKernel b = a;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a.persist_id(), "inv_quad");
        EXPECT_EQ(a.num_params(), 1U);

        // Atomic set_params rejection leaves state unchanged.
        const std::vector<double> bad_params{std::log(1000.0)};
        EXPECT_FALSE(b.set_params(bad_params).has_value());
        EXPECT_EQ(a, b);

        const std::vector<double> new_params{std::log(2.5)};
        ASSERT_TRUE(b.set_params(new_params).has_value());
        EXPECT_NE(a, b);

        std::vector<double> a_params(1);
        std::vector<double> b_params(1);
        ASSERT_TRUE(a.get_params(a_params).has_value());
        ASSERT_TRUE(b.get_params(b_params).has_value());
        EXPECT_DOUBLE_EQ(a_params[0], std::log(1.5));
        EXPECT_DOUBLE_EQ(b_params[0], std::log(2.5));
    }

    TEST(CustomKernelTest, SpecCompositionAndFiniteDifferenceChecks)
    {
        auto ck_res = cxxgpr::make_custom_kernel<InvQuadTerm>(1.3);
        ASSERT_TRUE(ck_res.has_value());
        if (!ck_res.has_value())
        {
            return;
        }

        auto spec =
            cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(2.0).value()) * cxxgpr::KernelSpec::custom(*ck_res)
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.1).value());

        auto compiled = cxxgpr::CompiledKernel<double>::from_spec(spec);
        EXPECT_EQ(compiled.num_params(), 3U);
        EXPECT_EQ(compiled.coord_mode(), cxxgpr::CoordMode::Dist);

        const std::vector<double> x_data{0.0, 0.5, 1.4, -0.3, 0.8, 1.1};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 3, 2, 3);
        const std::vector<double> d2 = make_sq_dist(x);
        const cxxgpr::MatrixRef<double> dist(d2.data(), 3, 3, 3);

        constexpr std::size_t n = 3;
        std::vector<double> k_pts(n * n);
        std::vector<double> k_dist(n * n);
        std::vector<double> g_pts(n * n);
        std::vector<double> h_pts(n * n);
        std::vector<double> sc_a(n * n);
        std::vector<double> sc_b(n * n);
        std::vector<double> sc_c(n * n);

        ASSERT_TRUE(compiled
                .apply_points(
                    x, cxxgpr::MatrixMut<double>(k_pts.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                )
                .has_value());
        ASSERT_TRUE(compiled
                .apply(
                    dist, cxxgpr::MatrixMut<double>(k_dist.data(), n, n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                )
                .has_value());
        for (std::size_t idx = 0; idx < k_pts.size(); ++idx)
        {
            EXPECT_NEAR(k_pts[idx], k_dist[idx], 1e-14);
        }

        constexpr double h = 1e-5;
        std::vector<double> theta(3);
        ASSERT_TRUE(compiled.get_params(theta).has_value());
        std::vector<double> k_plus(n * n);
        std::vector<double> k_minus(n * n);
        for (std::size_t p = 0; p < 3; ++p)
        {
            ASSERT_TRUE(compiled
                    .grad_points(
                        x, cxxgpr::MatrixMut<double>(g_pts.data(), n, n, n), p, cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                    )
                    .has_value());
            auto tp = theta;
            tp[p] = theta[p] + h;
            ASSERT_TRUE(compiled.set_params(tp).has_value());
            ASSERT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(k_plus.data(), n, n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                    )
                    .has_value());
            auto tm = theta;
            tm[p] = theta[p] - h;
            ASSERT_TRUE(compiled.set_params(tm).has_value());
            ASSERT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(k_minus.data(), n, n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n)
                    )
                    .has_value());
            ASSERT_TRUE(compiled.set_params(theta).has_value());
            for (std::size_t idx = 0; idx < g_pts.size(); ++idx)
            {
                EXPECT_NEAR(g_pts[idx], (k_plus[idx] - k_minus[idx]) / (2.0 * h), 1e-8);
            }
        }

        // Check Hessian for (0, 1) and (1, 1)
        std::vector<double> g_plus(n * n);
        std::vector<double> g_minus(n * n);
        for (std::size_t i : {std::size_t{0}, std::size_t{1}})
        {
            ASSERT_TRUE(compiled
                    .hess_points(
                        x, cxxgpr::MatrixMut<double>(h_pts.data(), n, n, n), i, 1U, cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), cxxgpr::MatrixMut<double>(sc_c.data(), n, n, n)
                    )
                    .has_value());
            auto tp = theta;
            tp[1] = theta[1] + h;
            ASSERT_TRUE(compiled.set_params(tp).has_value());
            ASSERT_TRUE(compiled
                    .grad_points(
                        x, cxxgpr::MatrixMut<double>(g_plus.data(), n, n, n), i, cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                    )
                    .has_value());
            auto tm = theta;
            tm[1] = theta[1] - h;
            ASSERT_TRUE(compiled.set_params(tm).has_value());
            ASSERT_TRUE(compiled
                    .grad_points(
                        x, cxxgpr::MatrixMut<double>(g_minus.data(), n, n, n), i, cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n)
                    )
                    .has_value());
            ASSERT_TRUE(compiled.set_params(theta).has_value());
            for (std::size_t idx = 0; idx < h_pts.size(); ++idx)
            {
                EXPECT_NEAR(h_pts[idx], (g_plus[idx] - g_minus[idx]) / (2.0 * h), 1e-7);
            }
        }
    }

    TEST(CustomKernelTest, DistanceKernelAndZeroAllocationHotPath)
    {
        auto ck_res = cxxgpr::make_custom_kernel<InvQuadTerm>(1.2);
        ASSERT_TRUE(ck_res.has_value());
        if (!ck_res.has_value())
        {
            return;
        }

        cxxgpr::ScalarDistance sdist;
        auto dk = sdist.custom(*ck_res)
            + cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::WhiteKernel::create(0.05).value());
        auto d_compiled = dk.compile<double>();

        const std::vector<double> x_data{0.0, 0.6, 1.5, -0.2, 0.7, 1.3};
        const cxxgpr::MatrixRef<double> x(x_data.data(), 3, 2, 3);
        const std::vector<double> d2 = make_sq_dist(x);

        cxxgpr::DistanceSource src;
        src.bind_scalar(sdist.slot(), cxxgpr::MatrixRef<double>(d2.data(), 3, 3, 3));
        auto checked = d_compiled.prepare_square(src);
        ASSERT_TRUE(checked.has_value());
        if (!checked.has_value())
        {
            return;
        }

        auto compiled = cxxgpr::CompiledKernel<double>::from_spec(
            cxxgpr::KernelSpec::custom(*ck_res) + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.05).value())
        );

        constexpr std::size_t n = 3;
        std::vector<double> out(n * n);
        std::vector<double> sc_a(n * n);
        std::vector<double> sc_b(n * n);
        std::vector<double> sc_c(n * n);
        auto nested = compiled.make_nested_buffers(n, n);
        auto d_nested = d_compiled.make_nested_buffers(n, n);

        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(compiled
                    .apply_points(
                        x, cxxgpr::MatrixMut<double>(out.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), nested
                    )
                    .has_value());
            ASSERT_TRUE(compiled
                    .grad_points(
                        x, cxxgpr::MatrixMut<double>(out.data(), n, n, n), 0U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), nested
                    )
                    .has_value());
            ASSERT_TRUE(compiled
                    .hess_points(
                        x, cxxgpr::MatrixMut<double>(out.data(), n, n, n), 0U, 0U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_c.data(), n, n, n), nested
                    )
                    .has_value());
            ASSERT_TRUE(d_compiled
                    .apply(
                        *checked, std::nullopt, cxxgpr::MatrixMut<double>(out.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n), d_nested
                    )
                    .has_value());
            ASSERT_TRUE(d_compiled
                    .grad(
                        *checked, std::nullopt, cxxgpr::MatrixMut<double>(out.data(), n, n, n), 0U,
                        cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sc_a.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc_b.data(), n, n, n), d_nested
                    )
                    .has_value());
        }
    }

}  // namespace
