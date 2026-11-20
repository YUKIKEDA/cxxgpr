# 0001: Exact GPR scope and C++23 design and dependency policy

- Status: Accepted
- Date: 2026-11-15
- Issue: #1

## Context

The Rust library `gprx` contains Exact GPR (`Gpr` / `FittedGpr`), an online-updating Exact GPR (`OnlineGpr`), and approximate sparse models (`Sgpr` / `Svgp` / `OnlineSgpr`). The goal of this project, `cxxgpr`, is to rebuild the full functionality of **Exact GPR** faithfully and safely in C++23, so that it can be used directly from production, embedded, and C++ ecosystem code.

At the same time, the dependency management approach (Eigen / JSON / optimizers / testing / benchmarking), the API error contract, and the public-header boundary must be fixed at the initial stage to prevent large rework after implementation starts.

## Decision

1. **Explicit port scope**:
   - Port Exact GPR of `gprx` completely (`Gpr` / `FittedGpr`, all 10 built-in kernel leaves, composition, `Custom`, supplied squared distances `DistanceKernel`, preprocessing transforms, `Double`/`Single`/`MixedPrecision`, `KernelExp`, the 5 optimizers, and `persist`).
   - Exclude `Sgpr` / `Svgp` / `OnlineGpr` (`into_online` / `LdltStore` / `PointId` / `ldlt` persist) and sparse inducing-point coordinate derivatives from the port.
2. **C++23 public API contract**:
   - The namespace is `cxxgpr` (`kernel`, `transform`, `persist`). Input matrices are column-major `std::span<const double>`.
   - Exceptions are not thrown across the API boundary; errors are returned as `[[nodiscard]] std::expected<T, GprError>`.
   - `Gpr::fit(...)` and `Gpr::factor(...)` are `const` member functions and leave `*this` unchanged even on failure.
   - Types that cannot fail use constructors; types that can fail use `static std::expected<T, GprError> create(...)`.
3. **Dependency selection and non-leakage into public headers**:
   - Dependencies are pinned by the vcpkg manifest (`builtin-baseline` + `overrides` in `vcpkg.json`).
   - Internal linear algebra uses `eigen3` (5.0.1, `EIGEN_MPL2_ONLY` + `EIGEN_DONT_PARALLELIZE`), persistence JSON uses `nlohmann-json` (3.12.0), and parallelization uses OpenMP. All are confined to `PRIVATE` dependencies of the static library `cxxgpr` and are not exposed in `include/cxxgpr/`.
   - The optimizers (`Lbfgs`, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing`, `Fixed`) are in-house implementations faithful to the `gprx` / argmin algorithm specifications; no external solver library is introduced.
   - Tests use `gtest` (1.18.0, `tests` feature) and measurements use `benchmark` (1.9.5, `benchmarks` feature), installed as separate features.

## Consequences

- Users of `include/cxxgpr/` can use the static library `cxxgpr` without bringing in Eigen, `nlohmann/json`, or OpenMP headers (minimizing ABI, build-dependency, and license exposure).
- The implementation cost of in-house optimizers increases, but the numerical behavior and determinism of L-BFGS Moré–Thuente, TrustRegion Steihaug–Toint CG, and FSA incremental evaluation are fully under our control.
- Compatibility is preserved with Apache-2.0 (`cxxgpr` itself), MPL-2.0 (Eigen, `EIGEN_MPL2_ONLY`), and MIT (`nlohmann-json`).
