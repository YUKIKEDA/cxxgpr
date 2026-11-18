// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-09, DES-11, DES-19, DES-20
/// @file optimizer_test.cpp
/// @brief Unit and property tests for hyperparameter optimizers, LogBounds, restarts, and validation.
/// @details Verifies LogBounds unconstrained-to-constrained transformations, chain-rule gradients
/// and Hessians against finite differences, EvalCache deduplication, convergence of L-BFGS,
/// TrustRegion, NelderMead, FastSimulatedAnnealing, and Fixed, deterministic multi-start restarts,
/// and negative validation controls.

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

    /// @brief Convex anisotropic quadratic objective in log-theta space for optimizer tests.
    /// @details Evaluates f(theta) = 0.5 * sum_i w_i * (theta_i - target_i)^2 with analytical
    /// gradients, diagonal Hessians, and bitwise unchanged-parameter verification in value_at_changes().
    class QuadraticObjective final : public cxxgpr::optimizer::TwiceDifferentiable
    {
    public:
        /// @brief Constructs a quadratic objective with target minimum, diagonal weights, and bounds.
        /// @param target Target minimizer vector in log-parameter space.
        /// @param weights Positive per-coordinate curvature weights.
        /// @param bounds Parameter Interval constraints matching target.size().
        QuadraticObjective(
            std::vector<double> target, std::vector<double> weights, std::vector<cxxgpr::Interval> bounds
        ) :
            target_(std::move(target)),
            weights_(std::move(weights)),
            bounds_(std::move(bounds))
        {
        }

        /// @brief Returns the number of optimization variables.
        /// @return Objective dimension matching target_.size().
        [[nodiscard]] std::size_t dim() const noexcept override { return target_.size(); }

        /// @brief Evaluates the quadratic objective value and records evaluation count and last iterate.
        /// @param params Current log-parameter vector of length dim().
        /// @return Scalar objective value 0.5 * sum_i w_i * (params_i - target_i)^2.
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

        /// @brief Evaluates the objective while verifying unlisted indices stay bitwise unchanged.
        /// @param params Candidate log-parameter vector of length dim().
        /// @param changed_indices Sorted parameter indices declared as mutated since the prior call.
        /// @return Scalar objective value from value(params).
        [[nodiscard]] std::expected<double, cxxgpr::GprError> value_at_changes(
            std::span<const double> params, std::span<const std::size_t> changed_indices
        ) override
        {
            ++change_eval_count;
            // Assert every parameter outside changed_indices matches the prior iterate bit-for-bit.
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

        /// @brief Copies parameter Interval bounds into out.
        /// @param out Destination span of length dim().
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_intervals(
            std::span<cxxgpr::Interval> out
        ) const override
        {
            std::ranges::copy(bounds_, out.begin());
            return {};
        }

        /// @brief Writes analytical objective gradient into out_grad.
        /// @param params Log-parameter vector of length dim().
        /// @param out_grad Output gradient vector of length dim().
        /// @return Empty expected on success.
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

        /// @brief Evaluates both quadratic objective value and gradient into out_grad.
        /// @param params Log-parameter vector of length dim().
        /// @param out_grad Output gradient vector of length dim().
        /// @return Scalar objective value on success, or propagated GprError on failure.
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

        /// @brief Writes the constant diagonal Hessian matrix into row/column-major out_hess.
        /// @param params Log-parameter vector (unused because curvature is constant).
        /// @param out_hess Output dim x dim Hessian matrix buffer.
        /// @return Empty expected on success.
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

        /// @brief Cumulative count of value() evaluations performed.
        std::size_t eval_count{0};
        /// @brief Cumulative count of value_at_changes() evaluations performed.
        std::size_t change_eval_count{0};

    private:
        /// @brief Target minimizer coordinates in log-parameter space.
        std::vector<double> target_;
        /// @brief Positive diagonal quadratic curvature weights.
        std::vector<double> weights_;
        /// @brief Valid parameter bounds for each dimension.
        std::vector<cxxgpr::Interval> bounds_;
        /// @brief Snapshot of the previous params vector for change-index bit checks.
        std::vector<double> last_params_;
    };

    /// @brief Two-dimensional Rosenbrock test objective with global minimum 0 at (1.0, 1.0).
    /// @details Implements TwiceDifferentiable with exact analytical value, gradient, and Hessian
    /// to benchmark L-BFGS and TrustRegion convergence through curved valleys.
    class RosenbrockObjective final : public cxxgpr::optimizer::TwiceDifferentiable
    {
    public:
        /// @brief Constructs a Rosenbrock objective sharing interval iv on both coordinates.
        /// @param iv Positive hyperparameter Interval applied to both coordinates.
        explicit RosenbrockObjective(cxxgpr::Interval iv) : iv_(iv) {}

        /// @brief Returns dimensionality (always 2).
        /// @return Problem dimension (2).
        [[nodiscard]] std::size_t dim() const noexcept override { return 2U; }

        /// @brief Evaluates (1 - x)^2 + 100 * (y - x^2)^2 at p.
        /// @param p Two-element coordinate vector [x, y].
        /// @return Scalar Rosenbrock objective value.
        [[nodiscard]] std::expected<double, cxxgpr::GprError> value(std::span<const double> p) override
        {
            const double x = p[0];
            const double y = p[1];
            const double a = 1.0 - x;
            const double b = y - (x * x);
            return (a * a) + (100.0 * b * b);
        }

        /// @brief Copies interval iv_ into both elements of out.
        /// @param out Destination interval span of length 2.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_intervals(
            std::span<cxxgpr::Interval> out
        ) const override
        {
            out[0] = iv_;
            out[1] = iv_;
            return {};
        }

        /// @brief Computes analytical Rosenbrock gradient at p into g.
        /// @param p Two-element coordinate vector [x, y].
        /// @param g Output gradient vector of length 2.
        /// @return Empty expected on success.
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

        /// @brief Evaluates Rosenbrock value and analytical gradient together.
        /// @param p Two-element coordinate vector [x, y].
        /// @param g Output gradient vector of length 2.
        /// @return Scalar objective value on success, or propagated GprError on failure.
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

        /// @brief Computes exact 2x2 analytical Rosenbrock Hessian at p into h.
        /// @param p Two-element coordinate vector [x, y].
        /// @param h Output 2x2 Hessian buffer of length 4.
        /// @return Empty expected on success.
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
        /// @brief Shared coordinate Interval constraint.
        cxxgpr::Interval iv_;
    };

    /// @brief Multi-modal 1D quartic test objective for multi-start restart verification.
    /// @details Evaluates (x^2 - 4)^2 - 0.6*x + 2, which traps local searches started near -2.0
    /// in an inferior local basin while the global minimum lies above +1.5.
    class MultiModalObjective final : public cxxgpr::optimizer::Differentiable
    {
    public:
        /// @brief Constructs the 1D multi-modal objective with bound interval iv.
        /// @param iv Positive hyperparameter Interval bound.
        explicit MultiModalObjective(cxxgpr::Interval iv) : iv_(iv) {}

        /// @brief Returns problem dimensionality (always 1).
        /// @return Problem dimension (1).
        [[nodiscard]] std::size_t dim() const noexcept override { return 1U; }

        /// @brief Evaluates (x^2 - 4)^2 - 0.6*x + 2 at p[0].
        /// @param p Single-element coordinate vector [x].
        /// @return Scalar objective value.
        [[nodiscard]] std::expected<double, cxxgpr::GprError> value(std::span<const double> p) override
        {
            const double x = p[0];
            const double x2 = x * x;
            return ((x2 - 4.0) * (x2 - 4.0)) - (0.6 * x) + 2.0;
        }

        /// @brief Copies bound interval iv_ into out[0].
        /// @param out Destination interval span of length 1.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_intervals(
            std::span<cxxgpr::Interval> out
        ) const override
        {
            out[0] = iv_;
            return {};
        }

        /// @brief Computes analytical derivative 4*x*(x^2 - 4) - 0.6 at p[0].
        /// @param p Single-element coordinate vector [x].
        /// @param g Output gradient span of length 1.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> gradient_into(
            std::span<const double> p, std::span<double> g
        ) override
        {
            const double x = p[0];
            g[0] = (4.0 * x * ((x * x) - 4.0)) - 0.6;
            return {};
        }

        /// @brief Evaluates multi-modal value and analytical derivative together.
        /// @param p Single-element coordinate vector [x].
        /// @param g Output gradient span of length 1.
        /// @return Scalar objective value on success, or propagated GprError on failure.
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
        /// @brief Coordinate Interval constraint.
        cxxgpr::Interval iv_;
    };

    /// @brief Verifies LogBounds mappings, chain-rule gradients/Hessians, and EvalCache deduplication.
    /// @details Confirms z = 0 midpoint Jacobian identity, constrained/unconstrained roundtrip,
    /// chain-rule gradient/Hessian match against central finite differences, and EvalCache hit reuse.
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

    /// @brief Verifies L-BFGS and TrustRegion converge to the Rosenbrock global minimum.
    /// @details Starts both first- and second-order optimizers at (-1.2, 1.0) and confirms convergence
    /// to (1.0, 1.0) with objective value below tolerance.
    TEST(OptimizerTest, LbfgsAndTrustRegionConvergeOnRosenbrock)
    {
        // Spec: REQ-09, DES-19
        const auto iv = cxxgpr::Interval::positive(1e-4, 1e4).value();
        RosenbrockObjective rosen(iv);
        const std::array<double, 2> start{-1.2, 1.0};

        // Run first-order L-BFGS optimization.
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

    /// @brief Verifies NelderMead, FastSimulatedAnnealing, and Fixed optimizer behaviors.
    /// @details Checks NelderMead convergence on anisotropic quadratic, Fixed single-evaluation
    /// behavior, and FastSimulatedAnnealing seeded bitwise determinism, change-index calls, and
    /// open-interval boundary policies (Reflect and Clamp).
    TEST(OptimizerTest, NelderMeadFsaAndFixedBehaveAsSpecified)
    {
        // Spec: REQ-09, DES-11, DES-19, DES-20
        const auto iv = cxxgpr::Interval::positive(1e-3, 1e3).value();
        QuadraticObjective quad({0.7, -1.1}, {3.0, 1.5}, {iv, iv});
        const std::array<double, 2> start{-1.5, 2.0};

        // Derivative-free Nelder-Mead simplex search on convex anisotropic quadratic.
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

    /// @brief Verifies multi-start restarts escape local traps and produce bit-identical results.
    /// @details Confirms single-start L-BFGS remains in the x < 0 local minimum while 8-restart
    /// seeded optimization locates x > 1.5 deterministically with bitwise identical value and params.
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

    /// @brief Negative controls verifying optimizer input and configuration validation errors.
    /// @details Checks EmptyInput, LengthMismatch, NonFiniteInput, InvalidInterval on boundary points,
    /// and InvalidConfig for invalid history sizes, trust-region radii, and cooling temperatures.
    TEST(OptimizerTest, NegativeControlsRejectInvalidInputsAndConfigs)
    {
        // Spec: DES-02, DES-19
        const auto iv = cxxgpr::Interval::positive(1e-2, 1e2).value();
        QuadraticObjective quad({0.0, 0.0}, {1.0, 1.0}, {iv, iv});
        const cxxgpr::optimizer::Lbfgs lbfgs;

        // Validate initial parameter vector errors before running optimizer iterations.
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
