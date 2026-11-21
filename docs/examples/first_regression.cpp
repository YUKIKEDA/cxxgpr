// SPDX-License-Identifier: Apache-2.0
// Tutorial: fit a one-dimensional Gaussian process to noisy samples of sin(x)
// and predict the mean and standard deviation at new inputs.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>

int main()
{
    // [data]
    // Eight training inputs: one column, so the column-major buffer is just the values.
    const std::array<double, 8> x_train{0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3};
    // y = sin(x) plus small hand-picked noise (standard deviation about 0.1).
    const std::array<double, 8> y_train{0.08, 0.83, 1.01, 0.33, -0.52, -1.07, -0.67, 0.12};
    // Three query inputs, one of them (3.15) between two training points.
    const std::array<double, 3> x_query{1.35, 3.15, 5.85};
    // [end]

    // [model]
    // k(x, x') = c * exp(-(x - x')^2 / (2 l^2)): signal variance c times an RBF kernel.
    auto signal = cxxgpr::ConstantKernel::create(1.0);
    auto shape = cxxgpr::RbfKernel::create(1.0);
    if (!signal || !shape)
    {
        std::cerr << "kernel construction failed\n";
        return 1;
    }
    const cxxgpr::KernelSpec spec = cxxgpr::KernelSpec(*signal) * cxxgpr::KernelSpec(*shape);

    // Initial observation noise variance 0.01 (it is optimised too); targets are standardised before fitting.
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec).with_noise_variance(0.01);
    if (!trainer)
    {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }
    const auto configured = trainer->with_target_transform(cxxgpr::transform::StandardizeTarget{});
    // [end]

    // [fit]
    auto fitted = configured.fit(x_train, 8U, 1U, y_train);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }
    // [end]

    // [predict]
    auto pred = fitted->predict(x_query, 3U, 1U);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    // [end]

    // [report]
    // Learned hyperparameters are stored as logarithms: signal variance c and lengthscale l.
    std::array<double, 2> log_params{};
    if (auto got = fitted->kernel_spec().get_params(log_params); !got)
    {
        std::cerr << got.error().message() << '\n';
        return 1;
    }
    std::cout << "signal variance = " << std::exp(log_params[0]) << ", lengthscale = " << std::exp(log_params[1])
              << '\n';
    std::cout << "noise variance = " << fitted->likelihood().noise_variance() << " (standardised scale)\n";
    std::cout << "negative log marginal likelihood = " << fitted->neg_log_marginal_likelihood() << '\n';
    // [end]
    bool ok = true;
    for (std::size_t i = 0; i < x_query.size(); ++i)
    {
        const double sd = std::sqrt(pred->variance[i]);
        const double truth = std::sin(x_query[i]);
        std::cout << "x = " << x_query[i] << "  mean = " << pred->mean[i] << "  std = " << sd
                  << "  sin(x) = " << truth << '\n';
        // The posterior mean must be within three standard deviations of the noise-free function.
        ok = ok && std::abs(pred->mean[i] - truth) < 3.0 * sd;
    }
    return ok ? 0 : 1;
}
