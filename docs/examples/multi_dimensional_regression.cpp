// SPDX-License-Identifier: Apache-2.0
// Tutorial: Gaussian process regression with two inputs on very different scales,
// an ARD kernel, standardising transforms, and a leave-one-out check.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace
{
    // The function we pretend not to know: oscillates in x1, grows slowly and linearly in x2.
    double truth(double x1, double x2) { return std::sin(x1) + x2 / 500.0; }
}  // namespace

int main()
{
    // [data]
    constexpr std::size_t n = 30;
    // Column-major 30 x 2 matrix: column 0 (x1 in [0, 6]) first, then column 1 (x2 in [0, 500]).
    std::vector<double> x(n * 2);
    std::vector<double> y(n);
    std::mt19937 gen(7U);  // mt19937 output is specified by the standard, so the data is portable.
    const auto unit = [&gen] { return static_cast<double>(gen()) / 4294967296.0; };
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = 6.0 * unit();
        x[n + i] = 500.0 * unit();
        y[i] = truth(x[i], x[n + i]) + 0.2 * (unit() - 0.5);  // uniform noise in [-0.1, 0.1]
    }
    // [end]

    // [model]
    // One lengthscale per input column, started at 1 in standardised units.
    const std::array<double, 2> lengthscales{1.0, 1.0};
    auto ard = cxxgpr::RbfArdKernel::create(lengthscales);
    auto signal = cxxgpr::ConstantKernel::create(1.0);
    if (!ard || !signal)
    {
        std::cerr << "kernel construction failed\n";
        return 1;
    }
    const cxxgpr::KernelSpec spec = cxxgpr::KernelSpec(*signal) * cxxgpr::KernelSpec(*ard);

    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec).with_noise_variance(0.01);
    if (!trainer)
    {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }
    const auto configured = trainer->with_input_transform(cxxgpr::transform::StandardizeInput{})
                                .with_target_transform(cxxgpr::transform::StandardizeTarget{});
    // [end]

    // [fit]
    auto fitted = configured.fit(x, n, 2U, y);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }
    std::array<double, 3> log_params{};  // [log c, log l1, log l2]
    if (auto got = fitted->kernel_spec().get_params(log_params); !got)
    {
        std::cerr << got.error().message() << '\n';
        return 1;
    }
    const double l1 = std::exp(log_params[1]);
    const double l2 = std::exp(log_params[2]);
    std::cout << "noise variance = " << fitted->likelihood().noise_variance() << " (standardised scale)\n";
    std::cout << "lengthscale for x1 = " << l1 << ", for x2 = " << l2 << " (standardised units)\n";
    // [end]

    // [predict]
    // Three queries (column-major 3 x 2): inside the data, inside the data, and far outside it.
    const std::array<double, 6> xq{3.0, 1.0, 12.0,    // x1
                                   250.0, 100.0, 250.0};  // x2
    auto pred = fitted->predict(xq, 3U, 2U);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    bool ok = l2 > l1;  // x2 matters less, so its lengthscale must be the longer one.
    for (std::size_t i = 0; i < 3; ++i)
    {
        const double sd = std::sqrt(pred->variance[i]);
        std::cout << "query (" << xq[i] << ", " << xq[3 + i] << "): mean = " << pred->mean[i] << ", std = " << sd
                  << ", truth = " << truth(xq[i], xq[3 + i]) << '\n';
    }
    // The extrapolated query must be much more uncertain than the interpolated ones.
    ok = ok && std::sqrt(pred->variance[2]) > 5.0 * std::sqrt(pred->variance[0]);
    // [end]

    // [loo]
    auto loo = fitted->loo_predict();
    if (!loo)
    {
        std::cerr << loo.error().message() << '\n';
        return 1;
    }
    double sq = 0.0;
    double z2 = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double err = loo->mean[i] - y[i];
        sq += err * err;
        z2 += err * err / loo->variance[i];
    }
    const double rmse = std::sqrt(sq / static_cast<double>(n));
    const double mean_z2 = z2 / static_cast<double>(n);
    std::cout << "leave-one-out RMSE = " << rmse << ", mean squared z-score = " << mean_z2 << '\n';
    // [end]

    // The noise has standard deviation 0.2 / sqrt(12) = 0.058; LOO error cannot be far below it.
    ok = ok && rmse < 0.15 && mean_z2 > 0.3 && mean_z2 < 3.0;
    return ok ? 0 : 1;
}
