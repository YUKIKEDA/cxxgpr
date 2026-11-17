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

#include <array>
#include <iostream>

int main() {
    const std::array<double, 4> x_train{0.0, 1.0, 2.0, 3.0};
    const std::array<double, 4> y_train{0.0, 1.0, 0.5, 1.2};
    const std::array<double, 2> x_query{0.5, 1.5};

    const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec).with_noise_variance(1e-4);
    if (!trainer) {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }

    auto fitted = trainer->fit(x_train, 4U, 1U, y_train);
    if (!fitted) {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }

    auto pred = fitted->predict(x_query, 2U, 1U);
    if (!pred) {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    std::cout << pred->mean[0] << ", " << pred->mean[1] << '\n';
    return 0;
}
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | Milestones, dependencies, and status |
| [`docs/design.md`](docs/design.md) | Requirements, C++23 API contracts, math, and workspace design |
| [`docs/persist-format.md`](docs/persist-format.md) | On-disk `config.json` and `model.safetensors` format specification |
| [`docs/architecture.md`](docs/architecture.md) | Module layers and dependency direction |
| [`docs/conventions.md`](docs/conventions.md) | Directory structure and naming conventions |
| [`docs/quality/quality-plan.md`](docs/quality/quality-plan.md) | Traceability, baselines, provenance, and platform qualification |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and gates |
