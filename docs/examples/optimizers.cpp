// SPDX-License-Identifier: Apache-2.0
// How-to: compare the built-in optimizers, drive one directly on your own objective, and write a custom optimizer.
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/optimizer/optimizer.hpp>
#include <cxxgpr/transform/transform.hpp>
#include <expected>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace
{
    constexpr std::size_t n_train = 30;

    double truth(double x)
    {
        return std::sin(6.0 * x) + 0.5 * x;
    }

    // [custom-optimizer]
    // A custom optimizer is any copyable type with this static flag and this optimize() signature.
    // It receives the same objective that the built-in optimizers see: dim() log-space parameters, the
    // negative marginal log-likelihood as value(), and gradient and Hessian that this optimizer ignores.
    // This one is a (1+1) evolution strategy: perturb the best point, keep the candidate if it is better.
    struct RandomSearch
    {
        static constexpr bool uses_change_indices = false;

        std::size_t steps = 300;
        double step_size = 0.3;  // standard deviation of the perturbation in log space
        std::uint64_t seed = 1;

        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable& obj, std::span<const double> initial_params
        ) const
        {
            auto bounds = cxxgpr::optimizer::LogBounds::from_objective(obj);
            if (!bounds)
            {
                return std::unexpected(std::move(bounds).error());
            }
            auto start_value = obj.value(initial_params);
            if (!start_value)
            {
                return std::unexpected(std::move(start_value).error());
            }
            cxxgpr::optimizer::OptimizerResult best;
            best.params.assign(initial_params.begin(), initial_params.end());
            best.value = *start_value;

            cxxgpr::Xoshiro256PlusPlus rng(seed);
            std::vector<double> candidate(obj.dim());
            for (std::size_t i = 0; i < steps; ++i)
            {
                for (std::size_t k = 0; k < candidate.size(); ++k)
                {
                    candidate[k] = best.params[k] + step_size * rng.unit_normal();
                }
                if (!bounds->validate_log_params(candidate))
                {
                    continue;  // outside the open bounds: skip the proposal
                }
                auto value = obj.value(candidate);
                if (value && *value < best.value)
                {
                    best.value = *value;
                    best.params = candidate;
                }
                ++best.iterations;
            }
            best.converged = true;  // a fixed budget always "finishes"
            return best;
        }
    };

    // [end]

    // [objective]
    // f(theta) = 0.5 * sum_i w_i * (theta_i - t_i)^2 over theta = log(parameters), parameters in (0.01, 100).
    class Quadratic final : public cxxgpr::optimizer::TwiceDifferentiable
    {
    public:
        Quadratic(std::vector<double> target, std::vector<double> weights) :
            target_(std::move(target)),
            weights_(std::move(weights))
        {
        }

        std::size_t dim() const noexcept override { return target_.size(); }

        std::expected<double, cxxgpr::GprError> value(std::span<const double> p) override
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < p.size(); ++i)
            {
                acc += 0.5 * weights_[i] * (p[i] - target_[i]) * (p[i] - target_[i]);
            }
            return acc;
        }

        std::expected<void, cxxgpr::GprError> fill_intervals(std::span<cxxgpr::Interval> out) const override
        {
            for (auto& interval : out)
            {
                interval = cxxgpr::Interval::create(0.01, 100.0).value();
            }
            return {};
        }

        std::expected<void, cxxgpr::GprError> gradient_into(std::span<const double> p, std::span<double> g) override
        {
            for (std::size_t i = 0; i < p.size(); ++i)
            {
                g[i] = weights_[i] * (p[i] - target_[i]);
            }
            return {};
        }

        std::expected<double, cxxgpr::GprError> value_and_gradient_into(
            std::span<const double> p, std::span<double> g
        ) override
        {
            ++gradient_evaluations;
            if (auto r = gradient_into(p, g); !r)
            {
                return std::unexpected(std::move(r).error());
            }
            return value(p);
        }

        std::expected<void, cxxgpr::GprError> hessian_into(std::span<const double> p, std::span<double> h) override
        {
            const std::size_t d = p.size();
            for (std::size_t i = 0; i < d; ++i)
            {
                for (std::size_t j = 0; j < d; ++j)
                {
                    h[i * d + j] = (i == j) ? weights_[i] : 0.0;
                }
            }
            return {};
        }

        std::size_t gradient_evaluations = 0;  // counts value-and-gradient calls

    private:
        std::vector<double> target_;
        std::vector<double> weights_;
    };

    // [end]

    template <typename Optimizer>
    bool run_on_gp(
        const std::string& name, const Optimizer& optimizer, std::span<const double> x, std::span<const double> y
    )
    {
        const std::array<double, 1> length_scale{0.3};
        const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scale).value());
        auto trainer = cxxgpr::Gpr<Optimizer>(kernel, optimizer).with_noise_variance(1e-2);
        if (!trainer)
        {
            return false;
        }
        const auto configured = trainer->with_target_transform(cxxgpr::transform::StandardizeTarget{});
        auto model = configured.fit(x, x.size(), 1U, y);
        if (!model)
        {
            std::cerr << name << ": " << model.error().message() << '\n';
            return false;
        }
        const std::array<double, 1> query{0.35};
        auto pred = model->predict(query, 1U, 1U);
        if (!pred)
        {
            return false;
        }
        const auto& result = model->optimizer_result();
        std::cout << std::left << std::setw(22) << name << std::right << std::setw(7) << result.iterations
                  << std::setw(11) << (result.converged ? "yes" : "no") << std::setw(12)
                  << model->neg_log_marginal_likelihood() << std::setw(10) << pred->mean[0] << '\n';
        return std::abs(pred->mean[0] - truth(0.35)) < 0.15;
    }
}  // namespace

int main()
{
    bool ok = true;
    std::cout << std::fixed << std::setprecision(4);

    // ---- Part 1: the five optimizers on the same regression problem -------------------------------------------
    std::vector<double> x(n_train);
    std::vector<double> y(n_train);
    cxxgpr::Xoshiro256PlusPlus rng(42);
    for (std::size_t i = 0; i < n_train; ++i)
    {
        x[i] = static_cast<double>(i) / static_cast<double>(n_train - 1);
        y[i] = truth(x[i]) + 0.05 * rng.unit_normal();
    }
    std::cout << "truth at 0.35 = " << truth(0.35) << '\n';
    std::cout << std::left << std::setw(22) << "optimizer" << std::right << std::setw(7) << "iters" << std::setw(11)
              << "converged" << std::setw(12) << "neg-log-ML" << std::setw(10) << "mean@0.35" << '\n';

    // [compare]
    using namespace cxxgpr::optimizer;
    ok = run_on_gp("Lbfgs", Lbfgs{}, x, y) && ok;
    ok = run_on_gp("TrustRegion", TrustRegion{}, x, y) && ok;
    ok = run_on_gp("NelderMead", NelderMead{}.with_max_iterations(400), x, y) && ok;
    ok =
        run_on_gp("FastSimulatedAnnealing", FastSimulatedAnnealing{}.with_max_iterations(400).with_seed(7), x, y) && ok;
    ok = run_on_gp("Lbfgs + 3 restarts", Lbfgs{}.with_restarts(3, 11), x, y) && ok;
    ok = run_on_gp("Fixed", Fixed{}, x, y) && ok;
    ok = run_on_gp("RandomSearch (custom)", RandomSearch{}, x, y) && ok;
    // [end]

    // ---- Part 2: driving an optimizer directly ---------------------------------------------------------------
    std::cout << "\nDirect use on a quadratic objective (optimum at exp(t) = 2, 0.5, 10):\n";
    const std::vector<double> target{std::log(2.0), std::log(0.5), std::log(10.0)};
    const std::vector<double> weights{1.0, 4.0, 0.25};
    const std::vector<double> start{0.0, 0.0, 0.0};  // log(1) for each parameter

    // [direct]
    Quadratic objective(target, weights);
    auto result = Lbfgs{}.optimize(objective, start);
    if (!result)
    {
        std::cerr << result.error().message() << '\n';
        return 1;
    }
    std::cout << "Lbfgs: converged = " << std::boolalpha << result->converged << ", iterations = " << result->iterations
              << ", value = " << std::scientific << std::setprecision(2) << result->value << std::fixed
              << std::setprecision(4) << '\n';
    std::cout << "  parameters = ";
    for (const double theta : result->params)
    {
        std::cout << std::exp(theta) << ' ';  // params are in log space
    }
    std::cout << '\n';
    // [end]
    ok = ok && result->converged && result->value < 1e-8;

    auto tr = TrustRegion{}.optimize(objective, start);
    auto nm = NelderMead{}.with_max_iterations(500).optimize(objective, start);
    auto fsa = FastSimulatedAnnealing{}.with_max_iterations(3000).with_seed(3).optimize(objective, start);
    auto fixed = Fixed{}.optimize(objective, start);
    if (!tr || !nm || !fsa || !fixed)
    {
        return 1;
    }
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "TrustRegion value = " << tr->value << " (" << tr->iterations << " iterations)\n";
    std::cout << "NelderMead  value = " << nm->value << " (" << nm->iterations << " iterations)\n";
    std::cout << "FSA         value = " << fsa->value << " (" << fsa->iterations << " iterations)\n";
    std::cout << "Fixed       value = " << fixed->value << " (" << fixed->iterations << " iterations)\n";
    std::cout << std::fixed << std::setprecision(4);
    ok = ok && tr->value < 1e-8 && nm->value < 1e-4 && fsa->value < 0.5 && fixed->iterations == 0;

    // ---- Part 3: restarts are deterministic -------------------------------------------------------------------
    // [restarts]
    const auto multi = Lbfgs{}.with_restarts(5, 2024);
    auto first = multi.optimize(objective, start);
    auto second = multi.optimize(objective, start);
    if (!first || !second)
    {
        return 1;
    }
    bool identical = first->params.size() == second->params.size();
    for (std::size_t i = 0; identical && i < first->params.size(); ++i)
    {
        identical = std::bit_cast<std::uint64_t>(first->params[i]) == std::bit_cast<std::uint64_t>(second->params[i]);
    }
    std::cout << "restarts with the same seed give bit-identical results: " << (identical ? "yes" : "no") << '\n';
    // [end]
    ok = ok && identical && first->params.size() == 3;

    // ---- Part 4: LogBounds -----------------------------------------------------------------------------------
    // [log-bounds]
    auto bounds = LogBounds::from_objective(objective);
    if (!bounds)
    {
        return 1;
    }
    std::array<double, 3> theta{std::log(2.0), std::log(0.5), 0.0};  // log-space values inside (log 0.01, log 100)
    std::array<double, 3> z{};
    std::array<double, 3> theta_back{};
    if (!bounds->to_unconstrained(theta, z))
    {
        return 1;
    }
    bounds->to_constrained(z, theta_back);
    std::cout << "z = (" << z[0] << ", " << z[1] << ", " << z[2]
              << "); round trip error = " << std::abs(theta_back[0] - theta[0]) << '\n';
    // [end]
    ok = ok && std::abs(theta_back[0] - theta[0]) < 1e-12 && std::abs(z[2]) < 1e-12;

    // ---- Part 5: EvalCache -----------------------------------------------------------------------------------
    // [eval-cache]
    EvalCache cache;
    std::array<double, 3> theta_buf{};
    std::array<double, 3> d1_buf{};
    std::array<double, 3> grad_theta_buf{};
    std::array<double, 3> grad_z{};
    const std::array<double, 3> z0{0.1, -0.2, 0.3};
    const std::size_t before = objective.gradient_evaluations;
    auto v1 = cache.eval_unconstrained(objective, *bounds, z0, theta_buf, d1_buf, grad_theta_buf, grad_z);
    auto v2 = cache.eval_unconstrained(objective, *bounds, z0, theta_buf, d1_buf, grad_theta_buf, grad_z);
    if (!v1 || !v2)
    {
        return 1;
    }
    std::cout << "two evaluations at the same point cost " << objective.gradient_evaluations - before
              << " objective call(s)\n";
    cache.clear();  // forget the cached point
    // [end]
    ok = ok && objective.gradient_evaluations - before == 1 && *v1 == *v2;

    return ok ? 0 : 1;
}
