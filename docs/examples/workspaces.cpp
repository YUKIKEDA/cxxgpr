// SPDX-License-Identifier: Apache-2.0
// Workspace-level building blocks: factor, marginal likelihood, gradient, Hessian, incremental
// Gram updates and posterior inference, all on caller-owned buffers.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/linalg/cholesky.hpp>
#include <cxxgpr/linalg/dense.hpp>
#include <cxxgpr/workspace/workspace.hpp>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <span>
#include <vector>

namespace
{

    constexpr std::size_t n = 30;  // training points
    constexpr std::size_t m = 5;   // query points
    constexpr std::size_t p = 3;   // hyperparameters: log c, log length-scale, log noise variance

    int fail(const char* what, const cxxgpr::GprError& error)
    {
        std::cerr << what << ": " << error.message() << '\n';
        return 1;
    }

}  // namespace

int main()
{
    using namespace cxxgpr;
    std::cout << std::setprecision(6);

    // Noisy samples of sin(2x) on [0, 3]; the generator is deterministic on every platform.
    Xoshiro256PlusPlus rng(7);
    std::vector<double> x(n);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = 3.0 * static_cast<double>(i) / static_cast<double>(n - 1);
        y[i] = std::sin(2.0 * x[i]) + 0.1 * rng.unit_normal();
    }
    const MatrixRef<double> x_ref(x.data(), n, 1, n);

    const auto spec = KernelSpec(ConstantKernel::create(1.0).value()) * KernelSpec(RbfKernel::create(0.8).value());
    auto kernel = spec.compile<double>();
    const auto likelihood = GaussianLikelihood::create(0.01).value();
    const auto jitter = JitterPolicy::adaptive(1e-10, 10.0, 8, 1e-4).value();

    // [factor]
    auto buffers = FitBuffers<DoublePrecision>::create(n, FitPolicy::prefer_speed());
    if (!buffers)
    {
        return fail("FitBuffers::create", buffers.error());
    }
    auto nll = factor_and_mll(kernel, likelihood, x_ref, y, jitter, *buffers);
    if (!nll)
    {
        return fail("factor_and_mll", nll.error());
    }
    std::cout << "negative log marginal likelihood = " << *nll << '\n';
    std::cout << "adopted jitter                   = " << buffers->core.factor_jitter << '\n';

    // Recompute the same number from the factor and weights the call left in the buffers.
    // factor_and_mll() stores L in core.k_matrix and alpha = A^-1 y in core.rhs.
    const std::span<const double> alpha = std::span<const double>(buffers->core.rhs).first(n);
    const double quad = linalg::dot(std::span<const double>(y), alpha).value();
    const double half_logdet = linalg::half_log_det_from_l(buffers->core.k_ref()).value();
    const double nll_check = 0.5 * quad + half_logdet + 0.5 * static_cast<double>(n) * std::log(2.0 * std::numbers::pi);
    std::cout << "recomputed from L and alpha      = " << nll_check << '\n';
    if (std::abs(nll_check - *nll) > 1e-10 * (1.0 + std::abs(*nll)))
    {
        return 1;
    }
    // [end]

    // [gradient]
    std::array<double, p> theta{};
    if (!kernel.get_params(std::span<double>(theta).first(2)))
    {
        return 1;
    }
    theta[2] = likelihood.log_noise_variance();

    std::array<double, p> grad{};
    auto nll_g = mll_and_gradient_into(kernel, likelihood, x_ref, y, jitter, *buffers, grad);
    if (!nll_g)
    {
        return fail("mll_and_gradient_into", nll_g.error());
    }

    // Central finite differences of factor_and_mll() with the parameters moved by hand.
    auto value_at = [&](std::span<const double> t) -> double
    {
        auto k = spec.compile<double>();
        if (!k.set_params(t.first(2)))
        {
            return std::nan("");
        }
        const auto lik = GaussianLikelihood::from_log_noise_variance(t[2]).value();
        auto work = FitBuffers<DoublePrecision>::create(n, FitPolicy::prefer_speed());
        return factor_and_mll(k, lik, x_ref, y, jitter, *work).value();
    };
    constexpr double h = 1e-5;
    std::cout << "gradient (analytic | finite difference):\n";
    for (std::size_t i = 0; i < p; ++i)
    {
        auto up = theta;
        auto down = theta;
        up[i] += h;
        down[i] -= h;
        const double fd = (value_at(up) - value_at(down)) / (2.0 * h);
        std::cout << "  d/dtheta" << i << ": " << std::setw(12) << grad[i] << " | " << std::setw(12) << fd << '\n';
        if (std::abs(grad[i] - fd) > 1e-5 * (1.0 + std::abs(fd)))
        {
            return 1;
        }
    }
    // [end]

    // [hessian]
    const bool had_hessian_scratch = buffers->core.hessian.is_allocated();
    std::array<double, p * p> hess{};
    if (auto r = hessian_into(kernel, likelihood, x_ref, y, jitter, *buffers, hess); !r)
    {
        return fail("hessian_into", r.error());
    }
    std::cout << "Hessian scratch allocated before/after the first call: " << had_hessian_scratch << " / "
              << buffers->core.hessian.is_allocated() << '\n';

    // The Hessian is symmetric and equals the finite-difference Jacobian of the gradient.
    auto grad_at = [&](std::span<const double> t)
    {
        auto k = spec.compile<double>();
        (void)k.set_params(t.first(2));
        const auto lik = GaussianLikelihood::from_log_noise_variance(t[2]).value();
        auto work = FitBuffers<DoublePrecision>::create(n, FitPolicy::prefer_speed());
        std::array<double, p> g{};
        (void)mll_and_gradient_into(k, lik, x_ref, y, jitter, *work, g);
        return g;
    };
    double worst = 0.0;
    for (std::size_t j = 0; j < p; ++j)
    {
        auto up = theta;
        auto down = theta;
        up[j] += h;
        down[j] -= h;
        const auto g_up = grad_at(up);
        const auto g_down = grad_at(down);
        for (std::size_t i = 0; i < p; ++i)
        {
            const double fd = (g_up[i] - g_down[i]) / (2.0 * h);
            worst = std::max(worst, std::abs(hess[(j * p) + i] - fd) / (1.0 + std::abs(fd)));
        }
    }
    std::cout << "Hessian vs finite differences of the gradient, worst relative error = " << worst << '\n';
    if (worst > 1e-5)
    {
        return 1;
    }

    // One call that returns value, gradient and Hessian from a single factorization.
    std::array<double, p> grad2{};
    std::array<double, p * p> hess2{};
    auto nll_h = value_gradient_hessian_into(kernel, likelihood, x_ref, y, jitter, *buffers, grad2, hess2);
    if (!nll_h || std::abs(*nll_h - *nll) > 1e-10 * (1.0 + std::abs(*nll)))
    {
        return 1;
    }
    // [end]

    // [policies]
    auto lean = FitBuffers<DoublePrecision>::create(n, FitPolicy::prefer_memory());
    if (!lean)
    {
        return 1;
    }
    std::array<double, p> grad_lean{};
    auto nll_lean = mll_and_gradient_into(kernel, likelihood, x_ref, y, jitter, *lean, grad_lean);
    if (!nll_lean)
    {
        return 1;
    }
    std::cout << "prefer_speed : W buffer " << buffers->w_matrix.size() << " doubles, distance cache "
              << buffers->dist.has_value() << ", overwrites L " << buffers->overwrites_cholesky() << '\n';
    std::cout << "prefer_memory: W buffer " << lean->w_matrix.size() << " doubles, distance cache "
              << lean->dist.has_value() << ", overwrites L " << lean->overwrites_cholesky() << '\n';
    std::cout << "same value and gradient under both policies: " << (*nll_lean == *nll_g && grad_lean == grad) << '\n';
    if (*nll_lean != *nll_g || grad_lean != grad)
    {
        return 1;
    }
    // [end]

    // [leaf-cache]
    static_assert(optimizer_uses_change_indices_v<optimizer::FastSimulatedAnnealing>);
    static_assert(!optimizer_uses_change_indices_v<optimizer::Lbfgs>);

    LeafCache<double> cache;
    auto nested = kernel.make_nested_buffers(n, n);
    std::vector<double> k_inc(n * n);
    std::vector<double> k_ref(n * n);
    std::vector<double> scratch(n * n);
    const MatrixMut<double> out(k_inc.data(), n, n, n);
    const MatrixMut<double> scr(scratch.data(), n, n, n);

    const std::array<std::size_t, p> everything{0, 1, 2};
    if (auto r =
            cache.evaluate_incremental(kernel, x_ref, nullptr, theta, everything, p, out, Triangle::Lower, scr, nested);
        !r)
    {
        return fail("evaluate_incremental", r.error());
    }
    std::cout << "leaves cached: " << cache.num_leaves() << ", primed: " << cache.is_primed() << '\n';

    // Move only the length-scale (parameter index 1). Only the leaf that owns it is recomputed.
    theta[1] = std::log(1.2);
    if (!kernel.set_params(std::span<const double>(theta).first(2)))
    {
        return 1;
    }
    const std::array<std::size_t, 1> only_length_scale{1};
    if (auto r = cache.evaluate_incremental(
            kernel, x_ref, nullptr, theta, only_length_scale, p, out, Triangle::Lower, scr, nested
        );
        !r)
    {
        return fail("evaluate_incremental", r.error());
    }
    if (!kernel.apply_points(x_ref, MatrixMut<double>(k_ref.data(), n, n, n), Triangle::Lower, scr, nested))
    {
        return 1;
    }
    double gram_diff = 0.0;
    for (std::size_t col = 0; col < n; ++col)
    {
        for (std::size_t row = col; row < n; ++row)
        {
            gram_diff = std::max(gram_diff, std::abs(k_inc[(col * n) + row] - k_ref[(col * n) + row]));
        }
    }
    std::cout << "incremental Gram vs full evaluation, max difference = " << gram_diff << '\n';

    // An undeclared change is rejected: parameter 0 moves although only index 1 is announced.
    auto cheat = theta;
    cheat[0] += 0.5;
    const auto rejected = cache.prepare_step(cheat, only_length_scale, kernel, p);
    std::cout << "undeclared change rejected: "
              << (!rejected && rejected.error().code() == GprErrorCode::IndexOutOfRange) << '\n';
    if (gram_diff > 1e-13 || rejected)
    {
        return 1;
    }
    // Restore the original hyperparameters for the prediction below.
    theta[1] = std::log(0.8);
    if (!kernel.set_params(std::span<const double>(theta).first(2)))
    {
        return 1;
    }
    // [end]

    // [predict]
    auto refactor = factor_and_mll(kernel, likelihood, x_ref, y, jitter, *buffers);
    if (!refactor)
    {
        return 1;
    }
    const std::vector<double> xq{0.25, 0.75, 1.5, 2.25, 2.75};
    const MatrixRef<double> xq_ref(xq.data(), m, 1, m);

    QueryWorkspace<DoublePrecision> query;
    Prediction<double> pred;
    if (auto r = predict_into(
            kernel, likelihood, x_ref, buffers->core.k_ref(), alpha, xq_ref, PredictOptions{}, query, pred
        );
        !r)
    {
        return fail("predict_into", r.error());
    }
    std::cout << "query workspace capacity: n = " << query.train_n << ", m = " << query.query_m
              << ", d = " << query.dims << '\n';

    // The same numbers come out of the high-level model with identical hyperparameters.
    auto model = Gpr<optimizer::Fixed>(spec).with_noise_variance(0.01);
    if (!model)
    {
        return 1;
    }
    auto fitted = model->factor(std::span<const double>(x), n, 1, std::span<const double>(y));
    if (!fitted)
    {
        return fail("Gpr::factor", fitted.error());
    }
    auto reference = fitted->predict(std::span<const double>(xq), m, 1);
    if (!reference)
    {
        return 1;
    }
    double mean_diff = 0.0;
    for (std::size_t j = 0; j < m; ++j)
    {
        mean_diff = std::max(mean_diff, std::abs(pred.mean[j] - reference->mean[j]));
        std::cout << "  x* = " << std::setw(5) << xq[j] << "  mean = " << std::setw(10) << pred.mean[j]
                  << "  sd = " << std::setw(9) << std::sqrt(pred.variance[j]) << "  sin(2x*) = " << std::setw(9)
                  << std::sin(2.0 * xq[j]) << '\n';
    }
    std::cout << "low-level vs FittedGpr::predict, max mean difference = " << mean_diff << '\n';
    if (mean_diff > 1e-10)
    {
        return 1;
    }

    // Leave-one-out predictions need only the factor, the weights and the targets.
    const std::span<const double> y_span(y);
    auto loo = loo_predict<DoublePrecision>(likelihood, buffers->core.k_ref(), alpha, y_span, PredictOptions{});
    if (!loo)
    {
        return fail("loo_predict", loo.error());
    }
    double loo_rmse = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        loo_rmse += (loo->mean[i] - y[i]) * (loo->mean[i] - y[i]);
    }
    std::cout << "leave-one-out RMSE against the noisy targets = " << std::sqrt(loo_rmse / static_cast<double>(n))
              << '\n';

    // The full posterior covariance and seeded samples reuse the same workspace.
    auto cov = predict_covariance(
        kernel, likelihood, x_ref, buffers->core.k_ref(), alpha, xq_ref, PredictOptions{VarianceKind::Latent}, query
    );
    auto draws_a = sample(
        kernel, likelihood, x_ref, buffers->core.k_ref(), alpha, xq_ref, 4, 99, PredictOptions{VarianceKind::Latent},
        query
    );
    auto draws_b = sample(
        kernel, likelihood, x_ref, buffers->core.k_ref(), alpha, xq_ref, 4, 99, PredictOptions{VarianceKind::Latent},
        query
    );
    if (!cov || !draws_a || !draws_b)
    {
        return 1;
    }
    std::cout << "covariance is " << cov->n_points << " x " << cov->n_points << ", " << draws_a->size()
              << " sample values, same seed gives identical draws: " << (*draws_a == *draws_b) << '\n';
    if (*draws_a != *draws_b)
    {
        return 1;
    }
    // [end]
    return 0;
}
