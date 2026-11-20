// SPDX-License-Identifier: Apache-2.0
// How-to: save a fitted model, load it back, predict from the loaded model, and handle load errors.
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/core/views.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/persist/persist.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace
{
    namespace fs = std::filesystem;

    // Removes the example's scratch directory on every exit path.
    struct ScratchDir
    {
        fs::path path;
        explicit ScratchDir(std::string name) : path(fs::temp_directory_path() / std::move(name))
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
        ~ScratchDir()
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    };

    bool same_bits(const std::vector<double>& a, const std::vector<double>& b)
    {
        if (a.size() != b.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
            {
                return false;
            }
        }
        return true;
    }
}  // namespace

int main()
{
    // ---- Fit a model --------------------------------------------------------------------------------------------
    constexpr std::size_t n = 24;
    std::vector<double> x(n);
    std::vector<double> y(n);
    cxxgpr::Xoshiro256PlusPlus rng(5);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = static_cast<double>(i) / static_cast<double>(n - 1);
        y[i] = 10.0 + 3.0 * std::sin(5.0 * x[i]) + 0.1 * rng.unit_normal();
    }
    const std::array<double, 1> length_scale{0.3};
    const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
        * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(length_scale).value());
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(1e-3);
    if (!trainer)
    {
        return 1;
    }
    auto fitted = trainer->with_input_transform(cxxgpr::transform::StandardizeInput{})
                      .with_target_transform(cxxgpr::transform::StandardizeTarget{})
                      .fit(x, n, 1U, y);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }
    const std::vector<double> query{0.05, 0.37, 0.5, 0.93};
    auto reference = fitted->predict(query, query.size(), 1U);
    if (!reference)
    {
        return 1;
    }

    // ---- Save ---------------------------------------------------------------------------------------------------
    // [save]
    const ScratchDir with_factor_dir("cxxgpr_example_with_factor");
    const ScratchDir light_dir("cxxgpr_example_light");

    // Stores x, y, the Cholesky factor L and alpha: loading skips the O(n^3) refactorization.
    if (auto r = cxxgpr::persist::save_with_factor(*fitted, with_factor_dir.path); !r)
    {
        std::cerr << r.error().message() << '\n';
        return 1;
    }
    // Stores only x and y: loading refactors the kernel matrix from the saved hyperparameters.
    if (auto r = cxxgpr::persist::save(*fitted, light_dir.path); !r)
    {
        std::cerr << r.error().message() << '\n';
        return 1;
    }
    for (const char* name : {"config.json", "model.safetensors"})
    {
        std::cout << name << ": " << fs::file_size(with_factor_dir.path / name) << " bytes\n";
    }
    // [end]

    // ---- Load ---------------------------------------------------------------------------------------------------
    // [load]
    auto loaded = cxxgpr::persist::LoadedGpr::load(with_factor_dir.path);
    if (!loaded)
    {
        std::cerr << loaded.error().message() << '\n';
        return 1;
    }
    std::cout << "n = " << loaded->n() << ", d = " << loaded->d() << ", has_factor = " << std::boolalpha
              << loaded->has_factor() << ", mmap factor = " << loaded->has_mmap_factor() << '\n';

    auto restored = loaded->predict(query, query.size(), 1U);  // double precision, original target scale
    if (!restored)
    {
        std::cerr << restored.error().message() << '\n';
        return 1;
    }
    // [end]

    const bool bitwise_factor = same_bits(restored->mean, reference->mean)
        && same_bits(restored->variance, reference->variance);
    std::cout << "save_with_factor round trip is bit-identical: " << bitwise_factor << '\n';

    auto light = cxxgpr::persist::LoadedGpr::load(light_dir.path);
    if (!light)
    {
        std::cerr << light.error().message() << '\n';
        return 1;
    }
    auto light_pred = light->predict(query, query.size(), 1U);
    if (!light_pred)
    {
        return 1;
    }
    double light_diff = 0.0;
    for (std::size_t i = 0; i < query.size(); ++i)
    {
        light_diff = std::max(light_diff, std::abs(light_pred->mean[i] - reference->mean[i]));
    }
    std::cout << "save() (refactorized on load): has_factor = " << light->has_factor()
              << ", max |mean difference| = " << light_diff << '\n';

    // ---- Everything a loaded model offers ---------------------------------------------------------------------
    // [loaded-api]
    const cxxgpr::MatrixRef<double> xs(query.data(), query.size(), 1U, query.size());
    auto cov = loaded->predict_covariance(xs);
    auto draws = loaded->sample(xs, 3U, /*seed=*/7U);
    auto loo = loaded->loo_predict();
    cxxgpr::Prediction<double> reused;
    auto into = loaded->predict_into(xs, reused);
    if (!cov || !draws || !loo || !into)
    {
        return 1;
    }
    std::cout << "covariance entries = " << cov->covariance.size() << ", samples = " << draws->size()
              << ", loo points = " << loo->mean.size() << '\n';
    // [end]

    // ---- Turn the loaded model back into a typed FittedGpr ------------------------------------------------------
    // [into-fitted]
    auto typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
    if (!typed)
    {
        std::cerr << typed.error().message() << '\n';
        return 1;
    }
    auto typed_pred = typed->predict(query, query.size(), 1U);
    // [end]
    if (!typed_pred)
    {
        return 1;
    }
    const bool bitwise_typed = same_bits(typed_pred->mean, reference->mean);
    std::cout << "into_fitted prediction is bit-identical: " << bitwise_typed << '\n';

    // Asking for the wrong precision policy is an error, not a silent conversion.
    auto wrong = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::SinglePrecision>();
    std::cout << "into_fitted<SinglePrecision> on a double model: "
              << (wrong ? "ok" : std::string(wrong.error().message())) << '\n';

    // ---- Error handling -----------------------------------------------------------------------------------------
    // [errors]
    auto missing = cxxgpr::persist::LoadedGpr::load(fs::temp_directory_path() / "cxxgpr_example_no_such_model");
    if (!missing)
    {
        const auto kind = missing.error().persist_error_kind();
        std::cout << "missing directory -> PersistErrorKind::Io: " << (kind && *kind == cxxgpr::PersistErrorKind::Io)
                  << '\n';
    }
    // [end]

    const bool ok = bitwise_factor && bitwise_typed && light_diff < 1e-9 && loaded->has_mmap_factor() && !wrong
        && !missing;
    return ok ? 0 : 1;
}
