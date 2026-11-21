// SPDX-License-Identifier: Apache-2.0
// How-to: read, set, bound and re-tune hyperparameters; evaluate the marginal-likelihood objective.
#include <cxxgpr/core/interval.hpp>
#include <cxxgpr/core/likelihood.hpp>
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/optimizer/optimizer.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace
{
    using namespace cxxgpr;

    // [probe]
    // A custom optimizer that only inspects the objective at the starting point: it checks the analytic
    // gradient and Hessian against finite differences and returns the starting point unchanged.
    struct ProbeStats
    {
        double max_gradient_error = 0.0;
        double max_hessian_error = 0.0;
    };

    struct ProbeOptimizer
    {
        static constexpr bool uses_change_indices = false;

        ProbeStats* stats = nullptr;

        [[nodiscard]] std::expected<optimizer::OptimizerResult, GprError> optimize(
            optimizer::TwiceDifferentiable& obj, std::span<const double> initial
        ) const
        {
            const std::size_t p = obj.dim();  // kernel parameters followed by log noise variance
            std::vector<double> theta(initial.begin(), initial.end());
            std::vector<double> grad(p);
            std::vector<double> hess(p * p);
            const auto value = obj.value_and_gradient_into(theta, grad);
            if (!value)
            {
                return std::unexpected(value.error());
            }
            if (auto h = obj.hessian_into(theta, hess); !h)
            {
                return std::unexpected(h.error());
            }
            constexpr double step = 1e-5;
            std::vector<double> up_grad(p);
            std::vector<double> down_grad(p);
            for (std::size_t i = 0; i < p; ++i)
            {
                auto plus = theta;
                auto minus = theta;
                plus[i] += step;
                minus[i] -= step;
                const auto f_plus = obj.value(plus);
                const auto f_minus = obj.value(minus);
                if (!f_plus || !f_minus)
                {
                    return std::unexpected(f_plus ? f_minus.error() : f_plus.error());
                }
                stats->max_gradient_error = std::max(stats->max_gradient_error, std::abs(((*f_plus - *f_minus) / (2.0 * step)) - grad[i]));
                (void)obj.value_and_gradient_into(plus, up_grad);
                (void)obj.value_and_gradient_into(minus, down_grad);
                for (std::size_t j = 0; j < p; ++j)
                {
                    const double fd = (up_grad[j] - down_grad[j]) / (2.0 * step);
                    stats->max_hessian_error = std::max(stats->max_hessian_error, std::abs(fd - hess[(i * p) + j]));
                }
            }
            return optimizer::OptimizerResult{.params = theta, .value = *value, .iterations = 0U, .converged = true};
        }
    };
    // [end]
}  // namespace

int main()
{
    std::cout << std::fixed << std::setprecision(4);
    bool ok = true;

    Xoshiro256PlusPlus rng(11U);
    std::vector<double> x;
    std::vector<double> y;
    for (std::size_t i = 0; i < 30U; ++i)
    {
        const double xi = 6.0 * static_cast<double>(i) / 29.0;
        x.push_back(xi);
        y.push_back(std::sin(xi) + 0.1 * rng.unit_normal());
    }

    // [bounds]
    // Bounds are open intervals in user units (not log units). Leaf factories accept them directly.
    const auto length_bounds = Interval::positive(0.1, 10.0).value();
    const auto variance_bounds = Interval::positive(0.01, 100.0).value();
    const auto kernel = KernelSpec(ConstantKernel::create(1.0, variance_bounds).value())
        * KernelSpec(RbfKernel::create(2.0, length_bounds).value());
    // [end]

    // [params]
    std::vector<double> theta(kernel.num_params());
    if (auto r = kernel.get_params(theta); !r)
    {
        return 1;
    }
    std::cout << "kernel parameters (log): " << theta[0] << ", " << theta[1] << "  -> constant " << std::exp(theta[0])
              << ", lengthscale " << std::exp(theta[1]) << '\n';
    std::vector<Interval> intervals(kernel.num_params(), Interval::default_positive());
    (void)kernel.bounds_into(intervals);
    std::cout << "lengthscale bounds: (" << intervals[1].lo() << ", " << intervals[1].hi() << ")\n";
    for (const auto& b : kernel.parameter_bindings())
    {
        std::cout << "parameter " << b.index << " is leaf " << b.leaf_id << ", local slot " << b.local_index << '\n';
    }
    // [end]
    ok = ok && std::abs(std::exp(theta[1]) - 2.0) < 1e-12;

    // [set-params]
    auto edited = kernel;  // KernelSpec has value semantics; set_params mutates only the copy
    const std::array<double, 2> proposal{std::log(3.0), std::log(0.5)};
    if (auto r = edited.set_params(proposal); !r)
    {
        return 1;
    }
    const std::array<double, 2> outside{std::log(3.0), std::log(50.0)};  // lengthscale 50 > 10
    const auto rejected = edited.set_params(outside);
    std::cout << "out-of-bounds update rejected: " << (!rejected && rejected.error().code() == GprErrorCode::InvalidInterval)
              << ", spec unchanged: " << (edited.get_params(std::span<double>(theta)).has_value() && std::abs(std::exp(theta[1]) - 0.5) < 1e-12)
              << '\n';
    // [end]
    ok = ok && !rejected;

    // [likelihood]
    const auto noise = GaussianLikelihood::from_log_noise_variance(std::log(0.04)).value();
    std::array<double, 1> log_noise{};
    (void)noise.get_params(log_noise);
    std::cout << "noise variance " << noise.noise_variance() << " = exp(" << noise.log_noise_variance()
              << "), num_params " << noise.num_params() << ", default bounds (" << std::scientific << noise.bounds().lo() << ", "
              << noise.bounds().hi() << std::fixed << ")\n";
    std::array<double, 3> diag{1.0, 1.0, 1.0};
    noise.add_noise_diag(diag);  // adds sigma_n^2 to each diagonal entry
    std::array<double, 3> dk{};
    (void)noise.noise_grad_diag(dk, 0U);  // d(sigma_n^2)/d(log sigma_n^2) = sigma_n^2
    std::cout << "diagonal after adding noise: " << diag[0] << ", gradient diagonal " << dk[0] << '\n';
    // [end]
    ok = ok && std::abs(diag[0] - 1.04) < 1e-12 && std::abs(dk[0] - 0.04) < 1e-12;

    // [bounded-param]
    const auto param = BoundedParam::create(2.0, Interval::create(0.5, 8.0).value()).value();
    const auto moved = param.with_value(4.0);
    const auto widened = param.with_interval(Interval::create(1.0, 3.0).value());
    const auto bad = param.with_value(9.0);
    std::cout << "ln(2) = " << param.ln() << ", with_value(4).value() = " << moved->value()
              << ", with_value(9) fails: " << !bad << ", interval width " << param.interval().width() << '\n';
    // [end]
    ok = ok && !bad && widened.has_value() && std::abs(param.ln() - std::log(2.0)) < 1e-15;
    ok = ok && !param.interval().contains(8.0) && param.interval().contains(2.0);
    ok = ok && !BoundedParam::default_positive(0.0).has_value() && !Interval::create(1.0, 1.0).has_value();

    // [fit]
    const auto trainer = Gpr<optimizer::Lbfgs>(kernel).with_noise_variance(0.5).value();
    const auto fitted = trainer.fit(x, x.size(), 1U, y);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }
    std::vector<double> learned(fitted->kernel_spec().num_params());
    (void)fitted->kernel_spec().get_params(learned);
    std::cout << "learned constant " << std::exp(learned[0]) << ", lengthscale " << std::exp(learned[1]) << ", noise "
              << fitted->likelihood().noise_variance() << ", neg-MLL " << fitted->neg_log_marginal_likelihood() << '\n';
    // [end]
    ok = ok && std::exp(learned[1]) > 0.1 && std::exp(learned[1]) < 10.0;
    ok = ok && fitted->likelihood().noise_variance() > 0.003 && fitted->likelihood().noise_variance() < 0.03;

    // [refit]
    // New observations arrive: warm-start from the fitted hyperparameters.
    std::vector<double> x2 = x;
    std::vector<double> y2 = y;
    for (std::size_t i = 0; i < 10U; ++i)
    {
        const double xi = 6.0 + 0.2 * static_cast<double>(i + 1U);
        x2.push_back(xi);
        y2.push_back(std::sin(xi) + 0.1 * rng.unit_normal());
    }
    const auto refitted = fitted->refit(x2, x2.size(), 1U, y2);
    const auto cold = trainer.fit(x2, x2.size(), 1U, y2);
    std::cout << "refit: " << refitted->optimizer_result().iterations << " iterations, cold fit: "
              << cold->optimizer_result().iterations << " iterations\n";
    // [end]
    ok = ok && refitted.has_value() && cold.has_value() && fitted->n_train() == 30U && refitted->n_train() == 40U;
    ok = ok && std::abs(refitted->neg_log_marginal_likelihood() - cold->neg_log_marginal_likelihood()) < 0.05;

    // [into-trainer]
    const auto warm = fitted->into_trainer();  // Gpr starting at the learned hyperparameters
    const auto frozen = Gpr<optimizer::Fixed>(fitted->kernel_spec()).with_likelihood(fitted->likelihood());
    const auto same = frozen.factor(x, x.size(), 1U, y);
    std::cout << "factor at learned values reproduces neg-MLL: " << same->neg_log_marginal_likelihood() << " vs "
              << fitted->neg_log_marginal_likelihood() << '\n';
    // [end]
    ok = ok && std::abs(same->neg_log_marginal_likelihood() - fitted->neg_log_marginal_likelihood()) < 1e-6;
    ok = ok && warm.likelihood() == fitted->likelihood();

    // [objective]
    // [objective]
    ProbeStats stats;
    const auto probe = Gpr<ProbeOptimizer>(kernel, ProbeOptimizer{&stats}).with_noise_variance(0.05).value();
    const auto probed = probe.fit(x, x.size(), 1U, y);
    if (!probed)
    {
        std::cerr << probed.error().message() << '\n';
        return 1;
    }
    std::cout << "neg-MLL at the start " << probed->optimizer_result().value << ", max gradient error "
              << std::scientific << stats.max_gradient_error << ", max Hessian error " << stats.max_hessian_error
              << std::fixed << '\n';
    // [end]
    ok = ok && probed->optimizer_result().iterations == 0U && stats.max_gradient_error < 1e-5
        && stats.max_hessian_error < 1e-4;

    std::cout << (ok ? "OK" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
