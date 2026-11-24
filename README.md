English | [日本語](README.ja.md)

# cxxgpr

**Exact Gaussian process regression for C++23.** Fit a model to your data, get predictions with calibrated uncertainty, and keep full control over numerics, memory and threads. No exceptions, no hidden allocation in the hot paths, and bit-identical results regardless of thread count.

[![CI](https://github.com/YUKIKEDA/cxxgpr/actions/workflows/ci.yml/badge.svg)](https://github.com/YUKIKEDA/cxxgpr/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **Status: `0.1.0`.** The library is feature-complete and verified; the API may still change before 1.0.

## Why cxxgpr

- **Errors are values.** Every fallible call returns `std::expected<T, GprError>`; nothing throws across the API.
- **Reproducible.** Parallel results are bit-identical for any thread count, and random sampling is seeded and portable.
- **Predictable cost.** Repeated predictions allocate nothing after the first call, and the memory/speed trade-off is yours to choose.
- **Clean headers.** Eigen, OpenBLAS/oneMKL, `herumi/fmath`, nlohmann/json and OpenMP never appear in the public headers, so they never leak into your build.
- **Safe by construction.** Failed calls leave your objects untouched, bounded hyperparameters cannot be set out of range, and trained and untrained models are different types.

## What you get

| Area | Included |
|---|---|
| **Models** | `Gpr` (trainer) and `FittedGpr` (trained model): fit, predict mean and variance, joint covariance, posterior samples, leave-one-out, marginal likelihood with analytic gradient and Hessian. `DistanceGpr` and `FittedDistanceGpr` do the same on distances you supply |
| **Kernels** | 10 built-in kernels (RBF, Matérn 1/2, 3/2 and 5/2, periodic, rational quadratic, constant, linear, white, plus ARD variants), sums and products, your own kernels, and distances you supply yourself |
| **Optimizers** | L-BFGS, trust region, Nelder–Mead, fast simulated annealing, or fixed hyperparameters; deterministic restarts |
| **Data handling** | Standardize, min–max and column-wise transforms for inputs and targets, chained in pipelines |
| **Numerics** | Double, single and mixed precision with iterative refinement; adaptive jitter; fast approximate exponential on request |
| **Persistence** | Save and load a trained model as a portable directory (`config.json` and `model.safetensors`), with optional memory-mapped loading |

Sparse approximations and online updates are intentionally not part of the library.

## Quick start

```cpp
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
```

This fits a noisy sine curve and prints the predicted mean and standard deviation. The [first-regression tutorial](docs/en/tutorials/first-regression.md) walks through it line by line.

## Install

**Prebuilt packages.** Each [GitHub Release](https://github.com/YUKIKEDA/cxxgpr/releases) provides ready-to-use packages for Windows (MSVC) and Linux (GCC 14), the documentation, and SHA-256 checksums. Unpack one and point CMake at it:

```cmake
find_package(cxxgpr 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

The package locates Eigen, OpenBLAS (or Intel oneMKL), nlohmann/json and (optionally) OpenMP when you configure, so they must be installed, for example through vcpkg.

**From source with `FetchContent`.**

```cmake
include(FetchContent)
FetchContent_Declare(
    cxxgpr
    GIT_REPOSITORY https://github.com/YUKIKEDA/cxxgpr.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(cxxgpr)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

Requirements: a C++23 compiler with `std::expected` (MSVC 19.44 or newer, or GCC 14 or newer), CMake 3.25 or newer, Ninja, and vcpkg with `VCPKG_ROOT` set. Build options and presets are described in the [installation guide](docs/en/how-to/install-and-link.md).

## Documentation

| | |
|---|---|
| [Tutorials](docs/en/tutorials/index.md) | Learn by building: first regression, a multi-dimensional model, save and reload |
| [How-to guides](docs/en/how-to/index.md) | Recipes for specific tasks; together they cover every public API |
| [Reference](docs/en/reference/index.md) | API reference, error codes, build options, file format |
| [Explanation](docs/en/explanation/index.md) | Background on Gaussian processes, kernels, numerical stability and design |

Build the documentation sites yourself with `just docs`. Contributors: see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the developer documentation in [`docs/internal/`](docs/internal/roadmap.md).

## License

Apache-2.0, see [`LICENSE`](LICENSE). Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
