# Layout and modules

The canonical source for repository layout and naming conventions. A PR that adds a new module or directory updates this document in the same PR.

## Directories

| Path | Contents |
|---|---|
| `include/cxxgpr/<module>/` | Public C++23 headers. Must not expose Eigen, `nlohmann/json`, or OpenMP |
| `src/<module>/` | Implementation of the static library `cxxgpr` and non-public `detail/` headers |
| `tests/` | Product tests with GoogleTest / CTest (split into `tests/<module>/`) |
| `tests/support/` | Shared test scaffolding such as allocation audits (`EIGEN_RUNTIME_NO_MALLOC` etc.) and comparison helpers |
| `benchmarks/` | Google Benchmark kernel / integrated measurements (not linked into the product or CTest) |
| `cmake/` | Compiler policy, warning settings, vcpkg dependency and third-party notice checks, build-information generation |
| `docs/en/`, `docs/ja/` | User documentation sources (Diataxis: tutorials / how-to / reference / explanation), English and Japanese |
| `docs/internal/` | Developer documentation (English only): specifications, ADRs, quality assurance, roadmap |
| `docs/internal/adr/` | Records of design decisions. One decision per file |
| `docs/internal/quality/` | Quality assurance plan, traceability matrix (generated), result-change records |
| `docs/internal/verification/` | Verification records (numerical qualification and independent comparisons) |
| `docs/Doxyfile`, `docs/conf.py` | Doxygen configuration and the Sphinx configuration that builds the user documentation |
| `docs/examples/` | Compiled and tested example programs that the user documentation includes |
| `qa/numeric_canary/` | Floating-point environment canary executable (built in M0-2) |
| `qa/baselines/` | Per-platform hexfloat baseline results (written only by `tools/qa/results.py`) |
| `tools/dev/` | Development helpers (MSVC environment launcher, Skill sync, CI scope detection, etc.; built in M0-2) |
| `tools/qa/` | Quality gates (`trace.py`, `arch_check.py`, `doc_check.py`, `results.py`; built in M0-2) |
| `tools/docs/` | Documentation generators (API reference pages, Doxygen input filter) |
| `tools/release/` | Release helpers (version check, archive, checksums); see [release.md](release.md) |
| `.agents/skills/`, `.claude/skills/` | The canonical Skills and their exact copies |
| `.cursor/rules/` | Development, design, numerical, and C++23 rules |
| `out/` | Build artifacts and the vcpkg installation area (not tracked by git) |
| `build/` | Temporary working area (not tracked by git) |
| `.dev/` | Local notes (not tracked by git) |

## Documentation languages

- User documentation (`docs/en/`, `docs/ja/`, `README.md`, `README.ja.md`) is bilingual: English and Japanese.
- Developer documentation (`docs/internal/`, `CONTRIBUTING.md`) is English only.
- Agent configuration (`.cursor/rules/`, `AGENTS.md`, `CLAUDE.md`, Skills) is written in Japanese.
- Code, Doxygen comments, and commit subjects are English.

## Product modules

Dependency directions follow [docs/internal/architecture.md](architecture.md).

| Module | Responsibility | Introduced in |
|---|---|---|
| `core` | `BuildInfo`, `GprError`, `Interval`, `BoundedParam`, `GaussianLikelihood`, checked dimensions, `Xoshiro256++`, `MatrixRef`/`MatrixMut`, `std::expected` helpers | M0-2 (`BuildInfo`), M1-1 |
| `precision` | `DoublePrecision`, `SinglePrecision`, `MixedPrecision`, iterative-refinement control constants and engine | M1-2 |
| `linalg` | Eigen LLT (`uplo=Lower`), triangular solves, the `JitterPolicy` retry loop, deterministic OpenMP helpers | M1-3 |
| `transform` | `Identity`/`Standardize`/`MinMax`/`Pipeline`/`ColumnwiseInput`, target inverse transforms, custom interface | M1-4 |
| `kernel` | The 10 built-in kernel leaves, `KernelSpec` to `CompiledKernel`, `KernelTerm<T>`, `DistanceKernel`, `KernelExp` | M2-A to M2-E |
| `workspace` | `WorkspaceCore`, `FitBuffers`, `QueryWorkspace`, `HessianScratch`, `LeafCache`, `CholeskyBuffer` | M2-E to M3-B |
| `optimizer` | The `Objective` hierarchy, `Lbfgs`, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing`, `Fixed`, logit transform, restarts | M4-A |
| `gpr` | `Gpr`, `FittedGpr`, `GprObjective`, LOO, covariance, sampling, `predict_into` | M3-C, M4-B |
| `persist` | `save`/`save_with_factor`, `LoadedGpr`/`LoadedDistanceGpr`, `config.json`/`model.safetensors`, `mmap`, `PersistRegistry` | M4-C |

M0 builds the repository foundation and the build and QA infrastructure; M1 onward implements the modules in order.

## Naming

- **CMake targets**: `cxxgpr` (static library); the test helper `cxxgpr_add_test(<module>.<topic> ...)`.
- **Namespaces**: `cxxgpr` (`cxxgpr::kernel`, `cxxgpr::transform`, `cxxgpr::persist`). Internal details live in `cxxgpr::detail`.
- **C++ identifiers**: types in `PascalCase`; functions, methods, local variables, and constants in `snake_case`; a trailing `_` on private members; the macro prefix `CXXGPR_`.
- **Doxygen comment convention**: every C++ file uses English `///` Doxygen with `@file`, `@defgroup`/`@ingroup`, `@brief`, `@details`, `@tparam`, `@param[in]`/`@param[out]`/`@param[in,out]`, `@return`, `@retval`, `@pre`, `@post`, `@note`, `@warning`, `@attention`, `@code`/`@endcode`, and `@see` (the canonical source for details is the [dev-docs rule](../../.cursor/rules/dev-docs.mdc); every `@param` must carry `[in]`/`[out]`/`[in,out]`).
- **Roadmap row IDs**: `M0-1` to `M5-4`, `REL-1` to `REL-11`.
- **Specification and quality IDs**: requirements `REQ-01` to `REQ-12`, design decisions `DES-01` to `DES-21`, persistence `FMT-01` to `FMT-09`, quality `QA-01` to `QA-10`.
