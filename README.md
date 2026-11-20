English | [日本語](README.ja.md)

# cxxgpr

Exact Gaussian Process Regression (GPR) library in C++23, ported from the Exact GPR model of [`gprx`](https://github.com/YUKIKEDA/gprx).

> **Status (`0.1.0`)**: Exact GPR implementation, persistence, CMake install/package support, and verification suites are complete. Public APIs follow pre-1.0 (`0.x`) stability expectations.

## Scope

`cxxgpr` focuses exclusively on **Exact GPR** (`Gpr` / `FittedGpr`):

- **Model & inference**: `Gpr::fit` (`const`, non-mutating on failure), `Gpr::factor`, `predict`, `predict_with` (`VarianceKind::Observation` / `Latent`), `predict_into`, `predict_with_into` (zero allocation after warmup), `predict_covariance`, `sample` (deterministic `Xoshiro256++`), `loo_predict`, `neg_log_marginal_likelihood`, `value_and_gradient_into`, `hessian_into`, `get_params`, `set_params`, `refit`, `into_trainer`.
- **Kernels**: All 10 built-in leaves (`RbfKernel`, `RbfArdKernel`, `MaternKernel` with $\nu \in \{1/2, 3/2, 5/2\}$, `MaternArdKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, `RationalQuadraticArdKernel`, `ConstantKernel`, `LinearKernel`, `WhiteKernel`), `+`/`*` composition via `KernelSpec` -> `CompiledKernel<T>`, user `Custom` leaves (`KernelTerm<T>`), `LeafCache` incremental updates, and supplied squared distances (`ScalarDistance`, `ArdDistance`, `DistanceKernel`).
- **Optimizers**: Self-contained implementations of `Lbfgs` (Moré–Thuente), `NelderMead`, `TrustRegion` (analytic Hessian + Steihaug–Toint CG), `FastSimulatedAnnealing` (Cauchy/Metropolis/Ingber + incremental eval), and `Fixed`, with logit interval transforms and deterministic restarts.
- **Transforms & policies**: Input/target transforms (`Identity*`, `Standardize*`, `MinMax*`, `Pipeline`, `TargetPipeline`, `ColumnwiseInput`, custom interfaces), `DoublePrecision` / `SinglePrecision` / `MixedPrecision<PromoteStorage | ReevaluateKernel>`, `KernelExp::{Accurate, FastApprox}`, `DistanceCachePolicy`, `CholeskyBuffer`, and `JitterPolicy::{Fixed, Adaptive}`.
- **Persistence**: Directory format (`config.json` + `model.safetensors`) with `save`, `save_with_factor`, `LoadedGpr::load`, `LoadedDistanceGpr::load`, and `PersistRegistry`.
- **Exclusions**: Sparse models (`Sgpr`, `Svgp`) and online updates (`OnlineGpr`) are intentionally outside scope.

## Design Highlights

- **C++23 & error handling**: Static library (`namespace cxxgpr`), no exceptions across the API boundary (`[[nodiscard]] std::expected<T, GprError>`).
- **Column-major input**: Data matrices use column-major storage via `MatrixRef<double>` or `std::span<const double>` (`x`, `n_rows`, `n_cols`).
- **Clean public headers**: Eigen, `nlohmann::json`, and OpenMP headers/types never leak into `include/cxxgpr/`.
- **License**: Apache-2.0 (`LICENSE`). Third-party notices are tracked in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Building & Integration

### Requirements

- C++23 compiler with `std::expected` support (MSVC 19.44+ / Visual Studio 17.14+, or GCC 14+)
- CMake 3.25+, Ninja, and `vcpkg` (`VCPKG_ROOT` set)

### Build & Test with Presets

```sh
# Linux (GCC 14)
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release

# Windows (MSVC, from Developer PowerShell or with VCPKG_ROOT)
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
```

### Install & `find_package(cxxgpr)`

```sh
cmake --install out/build/linux-release --prefix /path/to/install
```

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES CXX)

find_package(cxxgpr 0.1.0 CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

### `FetchContent`

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

### Minimal Example

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

See the [first-regression tutorial](docs/en/tutorials/first-regression.md) for a line-by-line walkthrough.

## Documentation

The user documentation follows the [Diataxis](https://diataxis.fr/) structure and is available in English and Japanese. Build it with `just docs`; the source is under `docs/en/` and `docs/ja/`.

| Section | Contents |
|---|---|
| [Tutorials](docs/en/tutorials/index.md) | Learn by building: first regression, multi-dimensional models, save and reload |
| [How-to guides](docs/en/how-to/index.md) | Task recipes that together cover every public API |
| [Reference](docs/en/reference/index.md) | API reference generated from the headers, error codes, build options, file format |
| [Explanation](docs/en/explanation/index.md) | Gaussian-process background, kernels, numerical stability, design principles |
| [`docs/internal/`](docs/internal/roadmap.md) | Developer documentation (English): specifications, ADRs, quality plan, roadmap |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and gates |
