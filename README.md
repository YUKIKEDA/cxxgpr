English | [日本語](README.ja.md)

# cxxgpr

Exact Gaussian Process Regression (GPR) library in C++23, faithfully ported from the Exact GPR model of [`gprx`](https://github.com/YUKIKEDA/gprx).

> **Project status**: Initial repository foundation (specifications, roadmap, and development conventions) is being established per [`docs/roadmap.md`](docs/roadmap.md). CMake/vcpkg build presets, QA gates (M0-2), and the library implementation (M1+) are in progress.

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
- **Column-major input**: Inputs are passed as column-major `std::span<const double>` (`x`, `n_rows`, `n_cols`).
- **Clean public headers**: Eigen, `nlohmann::json`, and OpenMP headers/types never leak into `include/cxxgpr/`.
- **License**: Apache-2.0 (`LICENSE`). Third-party notices are tracked in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

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
