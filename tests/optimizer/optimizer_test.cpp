// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-09, DES-11, DES-19, DES-20

#include "cxxgpr/optimizer/optimizer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

    static_assert(!cxxgpr::optimizer::OptimizerTraits<cxxgpr::optimizer::Lbfgs>::uses_change_indices);
    static_assert(!cxxgpr::optimizer::OptimizerTraits<cxxgpr::optimizer::TrustRegion>::uses_change_indices);
    static_assert(!cxxgpr::optimizer::OptimizerTraits<cxxgpr::optimizer::NelderMead>::uses_change_indices);
    static_assert(cxxgpr::optimizer::OptimizerTraits<cxxgpr::optimizer::FastSimulatedAnnealing>::uses_change_indices);
    static_assert(!cxxgpr::optimizer::OptimizerTraits<cxxgpr::optimizer::Fixed>::uses_change_indices);

    // Convex anisotropic quadratic in log-theta space: f(theta) = 0.5 * sum_i w_i * (theta_i -
    // target_i)^2
    class QuadraticObjective final : public cxxgpr::optimizer::TwiceDifferentiable
    {
    public:
        QuadraticObjective(
            std::vector<double> target, std::vector<double> weights, std::vector<cxxgpr::Interval> bounds
        ) :
            target_(std::move(target)),
            weights_(std::move(weights)),
            bounds_(std::move(bounds))
        {
        }

        [[nodiscard]] std::size_t dim() const noexcept override { return target_.size(); }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value(std::span<const double> params) override
        {
            ++eval_count;
            last_params_.assign(params.begin(), params.end());
            double acc = 0.0;
            for (std::size_t i = 0; i < params.size(); ++i)
            {
                const double diff = params[i] - target_[i];
                acc += 0.5 * weights_[i] * diff * diff;
            }
            return acc;
        }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value_at_changes(
            std::span<const double> params, std::span<const std::size_t> changed_indices
        ) override
        {
            ++change_eval_count;
            if (!last_params_.empty())
            {
                for (std::size_t i = 0; i < params.size(); ++i)
                {
                    const bool listed = std::ranges::find(changed_indices, i) != changed_indices.end();
                    if (!listed)
                    {
                        EXPECT_EQ(
                            std::bit_cast<std::uint64_t>(params[i]), std::bit_cast<std::uint64_t>(last_params_[i])
                        );
                    }
                }
            }
            return value(params);
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_intervals(
            std::span<cxxgpr::Interval> out
        ) const override
        {
            std::ranges::copy(bounds_, out.begin());
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> gradient_into(
            std::span<const double> params, std::span<double> out_grad
        ) override
        {
            for (std::size_t i = 0; i < params.size(); ++i)
            {
                out_grad[i] = weights_[i] * (params[i] - target_[i]);
            }
            return {};
        }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value_and_gradient_into(
            std::span<const double> params, std::span<double> out_grad
        ) override
        {
            auto v = value(params);
            if (!v.has_value())
            {
                return v;
            }
            if (auto g = gradient_into(params, out_grad); !g.has_value())
            {
                return std::unexpected(std::move(g).error());
            }
            return v;
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> hessian_into(
            [[maybe_unused]] std::span<const double> params, std::span<double> out_hess
        ) override
        {
            const std::size_t d = dim();
            std::ranges::fill(out_hess, 0.0);
            for (std::size_t i = 0; i < d; ++i)
            {
                out_hess[(i * d) + i] = weights_[i];
            }
            return {};
        }

        std::size_t eval_count{0};
        std::size_t change_eval_count{0};

    private:
        std::vector<double> target_;
        std::vector<double> weights_;
        std::vector<cxxgpr::Interval> bounds_;
        std::vector<double> last_params_;
    };

    // 2D Rosenbrock function in log-theta space: minimum at (1.0, 1.0), value 0.
    class RosenbrockObjective final : public cxxgpr::optimizer::TwiceDifferentiable
    {
    public:
        explicit RosenbrockObjective(cxxgpr::Interval iv) : iv_(iv) {}

        [[nodiscard]] std::size_t dim() const noexcept override { return 2U; }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value(std::span<const double> p) override
        {
            const double x = p[0];
            const double y = p[1];
            const double a = 1.0 - x;
            const double b = y - (x * x);
            return (a * a) + (100.0 * b * b);
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_intervals(
            std::span<cxxgpr::Interval> out
        ) const override
        {
            out[0] = iv_;
            out[1] = iv_;
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> gradient_into(
            std::span<const double> p, std::span<double> g
        ) override
        {
            const double x = p[0];
            const double y = p[1];
            g[0] = (-2.0 * (1.0 - x)) - (400.0 * x * (y - (x * x)));
            g[1] = 200.0 * (y - (x * x));
            return {};
        }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value_and_gradient_into(
            std::span<const double> p, std::span<double> g
        ) override
        {
            auto v = value(p);
            if (auto gr = gradient_into(p, g); !gr.has_value())
            {
                return std::unexpected(std::move(gr).error());
            }
            return v;
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> hessian_into(
            std::span<const double> p, std::span<double> h
        ) override
        {
            const double x = p[0];
            const double y = p[1];
            h[0] = 2.0 - (400.0 * (y - (3.0 * x * x)));
            h[1] = -400.0 * x;
            h[2] = -400.0 * x;
            h[3] = 200.0;
            return {};
        }

    private:
        cxxgpr::Interval iv_;
    };

    // Multi-modal 1D function with local minimum near -2.5 and global minimum at +1.5.
    class MultiModalObjective final : public cxxgpr::optimizer::Differentiable
    {
    public:
        explicit MultiModalObjective(cxxgpr::Interval iv) : iv_(iv) {}

        [[nodiscard]] std::size_t dim() const noexcept override { return 1U; }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value(std::span<const double> p) override
        {
            const double x = p[0];
            const double x2 = x * x;
            return ((x2 - 4.0) * (x2 - 4.0)) - (0.6 * x) + 2.0;
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_intervals(
            std::span<cxxgpr::Interval> out
        ) const override
        {
            out[0] = iv_;
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> gradient_into(
            std::span<const double> p, std::span<double> g
        ) override
        {
            const double x = p[0];
            g[0] = (4.0 * x * ((x * x) - 4.0)) - 0.6;
            return {};
        }

        [[nodiscard]] std::expected<double, cxxgpr::GprError> value_and_gradient_into(
            std::span<const double> p, std::span<double> g
        ) override
        {
            auto v = value(p);
            if (auto gr = gradient_into(p, g); !gr.has_value())
            {
                return std::unexpected(std::move(gr).error());
            }
            return v;
        }

    private:
        cxxgpr::Interval iv_;
    };

    TEST(OptimizerTest, LogBoundsAndChainRuleMatchFiniteDifferences)
    {
        // Spec: DES-19
        const auto iv0 = cxxgpr::Interval::positive(1e-3, 1e3).value();
        const auto iv1 = cxxgpr::Interval::positive(1e-2, 1e2).value();
        const std::array<cxxgpr::Interval, 2> intervals{iv0, iv1};
        auto bounds = cxxgpr::optimizer::LogBounds::from_intervals(intervals).value();

        // At midpoint (z = 0), Jacobian d(theta)/dz == 1 and d^2(theta)/dz^2 == 0.
        const std::array<double, 2> z_mid{0.0, 0.0};
        std::array<double, 2> theta{};
        std::array<double, 2> d1{};
        std::array<double, 2> d2{};
        bounds.to_constrained_with_derivatives(z_mid, theta, d1, d2);
        EXPECT_DOUBLE_EQ(d1[0], 1.0);
        EXPECT_DOUBLE_EQ(d1[1], 1.0);
        EXPECT_DOUBLE_EQ(d2[0], 0.0);
        EXPECT_DOUBLE_EQ(d2[1], 0.0);

        // Roundtrip constrained -> unconstrained -> constrained.
        const std::array<double, 2> theta_in{1.2, -0.7};
        std::array<double, 2> z_out{};
        std::array<double, 2> theta_rt{};
        ASSERT_TRUE(bounds.to_unconstrained(theta_in, z_out).has_value());
        bounds.to_constrained(z_out, theta_rt);
        EXPECT_NEAR(theta_rt[0], theta_in[0], 1e-13);
        EXPECT_NEAR(theta_rt[1], theta_in[1], 1e-13);

        // Chain rule gradient and Hessian match finite differences wrt z.
        QuadraticObjective obj({0.4, -0.5}, {2.5, 3.0}, {iv0, iv1});
        const std::array<double, 2> z_test{0.65, -0.9};
        bounds.to_constrained_with_derivatives(z_test, theta, d1, d2);
        std::array<double, 2> g_theta{};
        std::array<double, 4> h_theta{};
        ASSERT_TRUE(obj.gradient_into(theta, g_theta).has_value());
        ASSERT_TRUE(obj.hessian_into(theta, h_theta).has_value());

        std::array<double, 2> g_z{};
        std::array<double, 4> h_z{};
        cxxgpr::optimizer::LogBounds::chain_gradient(g_theta, d1, g_z);
        cxxgpr::optimizer::LogBounds::chain_hessian(h_theta, g_theta, d1, d2, h_z);

        const double eps = 1e-6;
        for (std::size_t i = 0; i < 2U; ++i)
        {
            std::array<double, 2> zp = z_test;
            std::array<double, 2> zm = z_test;
            zp[i] += eps;
            zm[i] -= eps;
            std::array<double, 2> tp{};
            std::array<double, 2> tm{};
            bounds.to_constrained(zp, tp);
            bounds.to_constrained(zm, tm);
            const double fd_g = (*obj.value(tp) - *obj.value(tm)) / (2.0 * eps);
            EXPECT_NEAR(g_z[i], fd_g, 1e-8);

            std::array<double, 2> d1p{};
            std::array<double, 2> d1m{};
            std::array<double, 2> gtp{};
            std::array<double, 2> gtm{};
            std::array<double, 2> gzp{};
            std::array<double, 2> gzm{};
            bounds.to_constrained_with_jacobian(zp, tp, d1p);
            bounds.to_constrained_with_jacobian(zm, tm, d1m);
            ASSERT_TRUE(obj.gradient_into(tp, gtp).has_value());
            ASSERT_TRUE(obj.gradient_into(tm, gtm).has_value());
            cxxgpr::optimizer::LogBounds::chain_gradient(gtp, d1p, gzp);
            cxxgpr::optimizer::LogBounds::chain_gradient(gtm, d1m, gzm);
            for (std::size_t row = 0; row < 2U; ++row)
            {
                const double fd_h = (gzp[row] - gzm[row]) / (2.0 * eps);
                EXPECT_NEAR(h_z[(i * 2U) + row], fd_h, 1e-6);
            }
        }

        // EvalCache avoids duplicate evaluations at identical z.
        cxxgpr::optimizer::EvalCache cache;
        const std::size_t before = obj.eval_count;
        ASSERT_TRUE(cache.eval_unconstrained(obj, bounds, z_test, theta, d1, g_theta, g_z).has_value());
        ASSERT_TRUE(cache.eval_unconstrained(obj, bounds, z_test, theta, d1, g_theta, g_z).has_value());
        EXPECT_EQ(obj.eval_count, before + 1U);
    }

    TEST(OptimizerTest, LbfgsAndTrustRegionConvergeOnRosenbrock)
    {
        // Spec: REQ-09, DES-19
        const auto iv = cxxgpr::Interval::positive(1e-4, 1e4).value();
        RosenbrockObjective rosen(iv);
        const std::array<double, 2> start{-1.2, 1.0};

        const auto lbfgs = cxxgpr::optimizer::Lbfgs().with_max_iterations(200U).with_tolerance(1e-8);
        auto r_lbfgs = lbfgs.optimize(rosen, start);
        ASSERT_TRUE(r_lbfgs.has_value());
        EXPECT_TRUE(r_lbfgs->converged);
        EXPECT_NEAR(r_lbfgs->params[0], 1.0, 1e-4);
        EXPECT_NEAR(r_lbfgs->params[1], 1.0, 1e-4);
        EXPECT_LT(r_lbfgs->value, 1e-8);

        const auto tr = cxxgpr::optimizer::TrustRegion().with_max_iterations(200U).with_tolerance(1e-8);
        auto r_tr = tr.optimize(rosen, start);
        ASSERT_TRUE(r_tr.has_value());
        EXPECT_TRUE(r_tr->converged);
        EXPECT_NEAR(r_tr->params[0], 1.0, 1e-5);
        EXPECT_NEAR(r_tr->params[1], 1.0, 1e-5);
        EXPECT_LT(r_tr->value, 1e-10);
    }

    TEST(OptimizerTest, NelderMeadFsaAndFixedBehaveAsSpecified)
    {
        // Spec: REQ-09, DES-11, DES-19, DES-20
        const auto iv = cxxgpr::Interval::positive(1e-3, 1e3).value();
        QuadraticObjective quad({0.7, -1.1}, {3.0, 1.5}, {iv, iv});
        const std::array<double, 2> start{-1.5, 2.0};

        const auto nm = cxxgpr::optimizer::NelderMead().with_max_iterations(300U).with_tolerance(1e-6);
        auto r_nm = nm.optimize(quad, start);
        ASSERT_TRUE(r_nm.has_value());
        EXPECT_TRUE(r_nm->converged);
        EXPECT_NEAR(r_nm->params[0], 0.7, 1e-4);
        EXPECT_NEAR(r_nm->params[1], -1.1, 1e-4);
        EXPECT_LT(r_nm->value, 1e-7);

        // Fixed performs zero iterations and evaluates once.
        const std::size_t before_fixed = quad.eval_count;
        const cxxgpr::optimizer::Fixed fixed;
        auto r_fixed = fixed.optimize(quad, start);
        ASSERT_TRUE(r_fixed.has_value());
        EXPECT_EQ(r_fixed->iterations, 0U);
        EXPECT_TRUE(r_fixed->converged);
        EXPECT_EQ(quad.eval_count, before_fixed + 1U);
        EXPECT_DOUBLE_EQ(r_fixed->params[0], start[0]);
        EXPECT_DOUBLE_EQ(r_fixed->params[1], start[1]);

        // FastSimulatedAnnealing: deterministic with identical seed, uses value_at_changes, stays in
        // open bounds for both Clamp and Reflect.
        const auto fsa = cxxgpr::optimizer::FastSimulatedAnnealing()
                             .with_max_iterations(400U)
                             .with_initial_temperature(2.0)
                             .with_cooling_rate(3.0)
                             .with_seed(1234ULL)
                             .with_boundary_policy(cxxgpr::optimizer::BoundaryPolicy::Reflect);
        const std::size_t before_chg = quad.change_eval_count;
        auto r_fsa1 = fsa.optimize(quad, start);
        EXPECT_GT(quad.change_eval_count, before_chg);
        auto r_fsa2 = fsa.optimize(quad, start);
        ASSERT_TRUE(r_fsa1.has_value());
        ASSERT_TRUE(r_fsa2.has_value());
        EXPECT_EQ(std::bit_cast<std::uint64_t>(r_fsa1->value), std::bit_cast<std::uint64_t>(r_fsa2->value));
        EXPECT_EQ(r_fsa1->params, r_fsa2->params);
        EXPECT_LT(r_fsa1->value, 0.05);

        const auto fsa_clamp = fsa.with_boundary_policy(cxxgpr::optimizer::BoundaryPolicy::Clamp);
        auto r_clamp = fsa_clamp.optimize(quad, start);
        ASSERT_TRUE(r_clamp.has_value());
        EXPECT_GT(r_clamp->params[0], std::log(iv.lo()));
        EXPECT_LT(r_clamp->params[0], std::log(iv.hi()));
    }

    TEST(OptimizerTest, RestartsFindGlobalOptimumDeterministically)
    {
        // Spec: DES-19, DES-20
        const auto iv = cxxgpr::Interval::positive(std::exp(-4.0), std::exp(4.0)).value();
        MultiModalObjective obj(iv);
        // Start trapped near the inferior local minimum (-2.0).
        const std::array<double, 1> local_start{-2.0};

        const auto lbfgs_single = cxxgpr::optimizer::Lbfgs().with_max_iterations(50U);
        auto r_single = lbfgs_single.optimize(obj, local_start);
        ASSERT_TRUE(r_single.has_value());
        EXPECT_LT(r_single->params[0], 0.0);

        const auto lbfgs_multi = lbfgs_single.with_restarts(8U, 42ULL);
        auto r_m1 = lbfgs_multi.optimize(obj, local_start);
        auto r_m2 = lbfgs_multi.optimize(obj, local_start);
        ASSERT_TRUE(r_m1.has_value());
        ASSERT_TRUE(r_m2.has_value());
        EXPECT_GT(r_m1->params[0], 1.5);
        EXPECT_LT(r_m1->value, r_single->value);
        EXPECT_EQ(r_m1->params, r_m2->params);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(r_m1->value), std::bit_cast<std::uint64_t>(r_m2->value));
    }

    TEST(OptimizerTest, NegativeControlsRejectInvalidInputsAndConfigs)
    {
        // Spec: DES-02, DES-19
        const auto iv = cxxgpr::Interval::positive(1e-2, 1e2).value();
        QuadraticObjective quad({0.0, 0.0}, {1.0, 1.0}, {iv, iv});
        const cxxgpr::optimizer::Lbfgs lbfgs;

        const std::vector<double> empty;
        auto err_empty = lbfgs.optimize(quad, empty);
        ASSERT_FALSE(err_empty.has_value());
        EXPECT_EQ(err_empty.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const std::array<double, 1> wrong_len{0.0};
        auto err_len = lbfgs.optimize(quad, wrong_len);
        ASSERT_FALSE(err_len.has_value());
        EXPECT_EQ(err_len.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        const std::array<double, 2> non_finite{std::numeric_limits<double>::quiet_NaN(), 0.0};
        auto err_nan = lbfgs.optimize(quad, non_finite);
        ASSERT_FALSE(err_nan.has_value());
        EXPECT_EQ(err_nan.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);

        const std::array<double, 2> out_of_range{std::log(iv.hi()), 0.0};
        auto err_oor = lbfgs.optimize(quad, out_of_range);
        ASSERT_FALSE(err_oor.has_value());
        EXPECT_EQ(err_oor.error().code(), cxxgpr::GprErrorCode::InvalidInterval);

        const std::array<double, 2> valid_start{0.5, -0.5};
        EXPECT_EQ(
            lbfgs.with_history_size(0U).optimize(quad, valid_start).error().code(), cxxgpr::GprErrorCode::InvalidConfig
        );
        EXPECT_EQ(
            cxxgpr::optimizer::TrustRegion().with_initial_radius(-1.0).optimize(quad, valid_start).error().code(),
            cxxgpr::GprErrorCode::InvalidConfig
        );
        EXPECT_EQ(
            cxxgpr::optimizer::FastSimulatedAnnealing()
                .with_initial_temperature(0.0)
                .optimize(quad, valid_start)
                .error()
                .code(),
            cxxgpr::GprErrorCode::InvalidConfig
        );
    }

}  // namespace
