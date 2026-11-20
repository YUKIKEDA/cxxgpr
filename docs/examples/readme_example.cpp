// SPDX-License-Identifier: Apache-2.0
// The program shown in README.md and README.ja.md; kept here so it is compiled and run by CTest.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <iostream>

int main()
{
    // Eight training inputs: one column, so the column-major buffer is just the values.
    const std::array<double, 8> x_train{0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3};
    // y = sin(x) plus small hand-picked noise (standard deviation about 0.1).
    const std::array<double, 8> y_train{0.08, 0.83, 1.01, 0.33, -0.52, -1.07, -0.67, 0.12};
    // Three query inputs, one of them (3.15) between two training points.
    const std::array<double, 3> x_query{1.35, 3.15, 5.85};

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

    auto fitted = configured.fit(x_train, 8U, 1U, y_train);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }

    auto pred = fitted->predict(x_query, 3U, 1U);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }

    std::cout << "mean = " << pred->mean[0] << ", std = " << std::sqrt(pred->variance[0]) << '\n';
    return 0;
}