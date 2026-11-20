// SPDX-License-Identifier: Apache-2.0
// Tutorial: save a fitted model to disk, load it back, and check that the predictions agree.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/persist/persist.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>

int main()
{
    // [fit]
    // Fit the same model as in the first tutorial.
    const std::array<double, 8> x_train{0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3};
    const std::array<double, 8> y_train{0.08, 0.83, 1.01, 0.33, -0.52, -1.07, -0.67, 0.12};
    const std::array<double, 3> x_query{1.35, 3.15, 5.85};

    const cxxgpr::KernelSpec spec =
        cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value()) *
        cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
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
    auto before = fitted->predict(x_query, 3U, 1U);
    if (!before)
    {
        std::cerr << before.error().message() << '\n';
        return 1;
    }

    // [end]

    // [save]
    // A checkpoint is a directory holding config.json and model.safetensors.
    const std::filesystem::path dir = "sin_model";
    std::filesystem::remove_all(dir);
    if (auto saved = cxxgpr::persist::save_with_factor(*fitted, dir); !saved)
    {
        std::cerr << saved.error().message() << '\n';
        return 1;
    }
    std::cout << "saved to " << dir.string() << '\n';
    // [end]

    // [load]
    auto loaded = cxxgpr::persist::LoadedGpr::load(dir);
    if (!loaded)
    {
        std::cerr << loaded.error().message() << '\n';
        return 1;
    }
    std::cout << "loaded: n = " << loaded->n() << ", d = " << loaded->d() << ", has_factor = " << std::boolalpha
              << loaded->has_factor() << '\n';
    // [end]

    // [compare]
    auto after = loaded->predict(x_query, 3U, 1U);
    if (!after)
    {
        std::cerr << after.error().message() << '\n';
        return 1;
    }
    bool identical = true;
    for (std::size_t i = 0; i < 3; ++i)
    {
        std::cout << "x = " << x_query[i] << "  before: " << before->mean[i] << " / " << before->variance[i]
                  << "  after: " << after->mean[i] << " / " << after->variance[i] << '\n';
        identical = identical && before->mean[i] == after->mean[i] && before->variance[i] == after->variance[i];
    }
    std::cout << "bit-identical = " << identical << '\n';
    // [end]
    return identical ? 0 : 1;
}
