// SPDX-License-Identifier: Apache-2.0
// How-to: fit and apply input/target transforms, compose them, and use them inside a Gpr.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
    constexpr double pi = 3.14159265358979323846;
    constexpr std::size_t n_rows = 20;
    constexpr std::size_t n_cols = 2;

    // Ground truth: a sine in the first feature plus a gentle trend in the second, on a large offset.
    double truth(double x0, double x1)
    {
        return 50.0 + 20.0 * std::sin(2.0 * pi * x0) + 0.005 * (x1 - 3000.0);
    }

    bool close(double a, double b, double tol)
    {
        return std::abs(a - b) <= tol;
    }
}  // namespace

int main()
{
    // [data]
    // Column-major training matrix: column 0 lives in [0, 1], column 1 in [1000, 5000].
    std::vector<double> x(n_rows * n_cols);
    std::vector<double> y(n_rows);
    for (std::size_t i = 0; i < n_rows; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(n_rows - 1);
        x[0 * n_rows + i] = t;
        x[1 * n_rows + i] = 1000.0 + 4000.0 * std::fmod(3.7 * t + 0.1, 1.0);
        y[i] = truth(x[0 * n_rows + i], x[1 * n_rows + i]);
    }
    // [end]

    // [standardize-input]
    const cxxgpr::transform::StandardizeInput standardize;
    auto fitted = standardize.fit_typed(x, n_rows, n_cols);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "column means: " << fitted->mean()[0] << ", " << fitted->mean()[1] << '\n';
    std::cout << "column stds:  " << fitted->std()[0] << ", " << fitted->std()[1] << '\n';

    std::vector<double> z(x.size());
    if (auto r = fitted->apply_into(x, n_rows, n_cols, z); !r)  // out of place, no allocation
    {
        std::cerr << r.error().message() << '\n';
        return 1;
    }
    // [end]

    // [round-trip]
    std::vector<double> back = z;
    if (auto r = fitted->inverse_apply(back, n_rows, n_cols); !r)  // in place
    {
        std::cerr << r.error().message() << '\n';
        return 1;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        worst = std::max(worst, std::abs(back[i] - x[i]));
    }
    std::cout << "round-trip max |error|: " << std::scientific << std::setprecision(2) << worst << '\n';
    std::cout << std::fixed << std::setprecision(4);
    if (worst > 1e-9)
    {
        return 1;
    }
    // [end]

    // [min-max-and-columnwise]
    auto unit = cxxgpr::transform::MinMaxInput::create(-1.0, 1.0);
    if (!unit)
    {
        return 1;
    }
    // Column 0 is left alone, column 1 is mapped onto [-1, 1].
    cxxgpr::transform::ColumnwiseInput per_column;
    per_column.then(cxxgpr::transform::IdentityInput{}).then(*unit);
    auto per_column_fit = per_column.fit_typed(x, n_rows, n_cols);
    if (!per_column_fit)
    {
        std::cerr << per_column_fit.error().message() << '\n';
        return 1;
    }
    std::vector<double> mixed = x;
    if (auto r = per_column_fit->apply(mixed, n_rows, n_cols); !r)
    {
        return 1;
    }
    std::cout << "columnwise: row 0 -> (" << mixed[0] << ", " << mixed[n_rows] << "), row " << n_rows - 1 << " -> ("
              << mixed[n_rows - 1] << ", " << mixed[2 * n_rows - 1] << ")\n";
    // [end]

    // [pipeline]
    // Steps run in order when fitting and applying, and in reverse order when inverting.
    cxxgpr::transform::Pipeline pipeline;
    pipeline.then(cxxgpr::transform::MinMaxInput{}).then(cxxgpr::transform::StandardizeInput{});
    auto pipeline_fit = pipeline.fit_typed(x, n_rows, n_cols);
    if (!pipeline_fit)
    {
        return 1;
    }
    std::cout << "pipeline steps: " << pipeline_fit->size() << '\n';
    // [end]

    // [target-transform]
    const cxxgpr::transform::StandardizeTarget standardize_target;
    auto target = standardize_target.fit_typed(y);
    if (!target)
    {
        return 1;
    }
    std::cout << "target mean = " << target->mean() << ", std = " << target->std() << '\n';

    // A model-space mean of 0.5 and variance of 0.04 map back to original units.
    std::array<double, 1> mean{0.5};
    std::array<double, 1> var{0.04};
    if (!target->inverse_transform_mean(mean) || !target->inverse_transform_variance(var))
    {
        return 1;
    }
    std::cout << "original-scale mean = " << mean[0] << ", variance = " << var[0] << '\n';
    // The mean is un-shifted and un-scaled; the variance is scaled by std^2 only.
    if (!close(mean[0], 0.5 * target->std() + target->mean(), 1e-12)
        || !close(var[0], 0.04 * target->std() * target->std(), 1e-9))
    {
        return 1;
    }
    // [end]

    // [use-in-gpr]
    const std::array<double, 2> length_scales{1.0, 1.0};
    const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
        * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scales).value());
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(1e-4);
    if (!trainer)
    {
        return 1;
    }
    const auto configured = trainer->with_input_transform(cxxgpr::transform::StandardizeInput{})
                                .with_target_transform(cxxgpr::transform::StandardizeTarget{});
    auto model = configured.fit(x, n_rows, n_cols, y);
    if (!model)
    {
        std::cerr << model.error().message() << '\n';
        return 1;
    }

    // Queries are given in original units; the model standardizes them and un-standardizes the answer.
    const std::array<double, 2> query{0.30, 3000.0};
    auto pred = model->predict(query, 1U, 2U);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    std::cout << "prediction at (0.30, 3000): mean = " << pred->mean[0] << ", sd = " << std::sqrt(pred->variance[0])
              << ", truth = " << truth(0.30, 3000.0) << '\n';
    // [end]
    if (!close(pred->mean[0], truth(0.30, 3000.0), 5.0))
    {
        return 1;
    }
    return 0;
}
