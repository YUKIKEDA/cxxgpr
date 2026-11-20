# Relationship to gprx

cxxgpr is a C++23 port of the **Exact GPR** model of the Rust library [gprx](https://github.com/YUKIKEDA/gprx). This page explains what that means: what has been carried over, what was deliberately left out and why, and which parts changed because C++ is not Rust. It is written for readers who know `gprx`, or who wonder whether cxxgpr can replace a model they already use.

:::{note}
Statements about `gprx` on this page are limited to what the cxxgpr project records about its own porting scope and its verification against `gprx` output. We have not independently re-verified `gprx` behaviour for this page; consult the `gprx` documentation for anything else.
:::

## What is ported

The port covers the Exact GPR part of `gprx`: the model in which the full $n \times n$ covariance matrix is factorised, without approximation. In cxxgpr terms:

| Area | Ported content |
|---|---|
| Model | `Gpr` (trainer) and `FittedGpr` (trained model): `fit`, `factor`, `predict`, `predict_covariance`, `sample`, `loo_predict`, marginal likelihood with gradient and Hessian, refitting, and parameter access |
| Kernels | All ten built-in leaf kernels, sums and products of kernels, user-defined `Custom` leaves, and training from *supplied* squared distances (`DistanceKernel`) |
| Preprocessing | Input and target transforms: identity, standardisation, min-max scaling, pipelines and column-wise transforms |
| Precision | `DoublePrecision`, `SinglePrecision` and `MixedPrecision` with iterative refinement |
| Numerics | The `KernelExp` choice between accurate and fast exponentials, and the separation of observation noise from numerical jitter |
| Optimization | Five optimizers: `Lbfgs`, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing` and `Fixed` |
| Persistence | `config.json` plus `model.safetensors` directories, `save`, `save_with_factor`, `LoadedGpr` and `PersistRegistry` |

The optimizers are implemented inside cxxgpr and follow the algorithm specifications used by `gprx` (and the argmin library it relies on); no external solver library is linked. The random number generator `Xoshiro256++`, used for posterior samples and optimizer restarts, is specified to produce the same sequence as the `gprx` generator for the same seed, on every operating system and build configuration.

## What is not ported

Three models of `gprx` are outside the scope of cxxgpr, intentionally:

- **`Sgpr`** and **`Svgp`**, the sparse approximations that summarise the data with a small set of inducing points to avoid the cubic cost of exact inference.
- **`OnlineGpr`**, an Exact GPR that is updated one observation at a time. It depends on a different factorisation (LDLT) and on stable identifiers for data points; neither is part of cxxgpr.
- Consequently, derivatives with respect to inducing-point coordinates, which only exist for sparse models.

The reasons are practical. cxxgpr aims to reproduce one model completely, with verified numerics, a stable API, deterministic results and a documented file format, instead of reproducing every model partially. The exact model is also the reference against which sparse methods are judged, so a faithful exact core is the useful first foundation. If you need to handle many thousands of points, subsample or partition the problem, or use a sparse method from another library.

Persistence follows from the same boundary. The directory format is essentially compatible with the Exact GPR persistence format of `gprx`, but checkpoints that use the LDLT factor or point identifiers of online models, and checkpoints of sparse models, are rejected on load with an error.

## What is shaped by C++

| Topic | How cxxgpr handles it |
|---|---|
| Errors | `[[nodiscard]] std::expected<T, GprError>`; no exception ever crosses the API boundary |
| Construction that can fail | `static std::expected<T, GprError> create(...)` factories; types that cannot fail use plain constructors |
| Immutability | `Gpr::fit` and `Gpr::factor` are `const`; the trainer is unchanged by a fit, even a failed one |
| Matrices | Column-major `std::span<const double>` plus shape, or the non-owning `MatrixRef`; see {doc}`../how-to/prepare-data` |
| Linear algebra and parallelism | Eigen and OpenMP, both private, with results that do not depend on the thread count |
| Dependencies visible to users | None: Eigen, nlohmann/json and OpenMP never appear in public headers |

Two consequences are worth stating plainly. First, the API is shaped by C++: you compose kernels with `KernelSpec` and the builder methods `with_*`, not with Rust traits. Second, cxxgpr treats reproducibility as a feature: the same inputs give the same bits regardless of how many threads run, and the zero-allocation prediction path (`predict_into`) does not allocate after warm-up. See {doc}`determinism-and-performance`.

## How closeness to gprx is verified

Closeness is checked, not assumed. The cxxgpr test suite compares fitted and predicted values against output recorded from `gprx`, against the `scikit-learn` library, and against analytic solutions, for fixed hyperparameters, for L-BFGS optimisation, and for leave-one-out predictions. Differences between independent implementations in floating-point arithmetic are expected to be small but are not zero, and optimisers started from the same point can follow slightly different paths; compare results with a tolerance, not for bitwise equality.

## Moving from gprx

The model names carry over: `Gpr` is the trainer and `FittedGpr` the trained model, as in `gprx`. Start with {doc}`../tutorials/first-regression`, then use {doc}`../how-to/build-kernels` and {doc}`../how-to/choose-optimizer`. For the design principles that shaped the C++ API, see {doc}`design-principles`.
