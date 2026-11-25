# Design principles

This page explains the decisions that shape the cxxgpr API and why they were made: errors as values, types that cannot be in an invalid state, a strict separation of training and inference, clear ownership, column-major data, and a public interface free of third-party types. Knowing them lets you predict how the library behaves and why a call has the shape it has. Each principle also has a cost, which is stated.

## Errors are values

Every operation that can fail returns `std::expected<T, GprError>` and is marked `[[nodiscard]]`. The library never throws an exception across its API boundary. A failure is an ordinary return value that carries a `GprErrorCode`, a message and, depending on the code, structured details: the offending dimensions, the last jitter tried by a failed factorization, the optimizer iteration count, the position of an invalid distance, and so on.

```cpp
auto fitted = trainer.fit(x, n_rows, n_cols, y);
if (!fitted) {
    // fitted.error().code(), fitted.error().message(), ...
    return;
}
auto prediction = fitted->predict(xs, m, n_cols);
```

The reasons are practical. Numerical code fails for data-dependent reasons (a matrix that is not positive definite, a kernel value that overflows to infinity), so failure is expected control flow and not an exceptional event. A value-returning error can be handled in real-time and embedded code that disables exceptions, it is visible in the signature, and `[[nodiscard]]` makes ignoring it a compiler warning. It also composes: with C++23, `and_then`, `transform` and `or_else` chain calls without nested `if`.

The cost is verbosity. Every call site needs a check, and constructing a kernel or a likelihood is a two-step affair (`create(...)` then `.value()` or a test). {doc}`../how-to/handle-errors` shows patterns that keep this readable.

A consequence that is easy to rely on: when `Gpr::fit` or `Gpr::factor` fails, the trainer is exactly as it was before, and no output argument is half written. Both functions are `const` members, so they cannot modify the trainer; they return a new `FittedGpr` or nothing.

Validation is deterministic. When several things are wrong at once, the first error in a fixed order is reported: empty input, size overflow, length mismatch, dimension mismatch, non-finite values, then insufficient data. The same bad call always produces the same error.

## Invalid states cannot be constructed

A type that can be created in an invalid state forces every function that takes it to re-check it. cxxgpr instead moves the check to construction and makes the validated object the only way to hold a value.

- **Fallible construction is a named factory.** `RbfKernel::create(length_scale)`, `GaussianLikelihood::create(noise_variance)`, `Interval::create(lo, hi)`, `JitterPolicy::fixed(j)` and `JitterPolicy::adaptive(...)` return `std::expected`; their constructors are private. A `RbfKernel` that exists therefore has a positive finite length scale, and a `JitterPolicy` that exists has a sensible schedule.
- **Infallible construction is an ordinary constructor.** Types with defaults that are always valid, such as `Gpr`, `Lbfgs`, `IdentityInput` or `StandardizeTarget`, use constructors.
- **Bounded parameters carry their interval.** A hyperparameter and its allowed range travel together (`BoundedParam`, `Interval`), so the optimizer cannot step outside the range and a stored value is always inside it.
- **Closed sets are enumerations and concepts.** `MaternNu` has three values, `VarianceKind` two, `CholeskyBuffer` two, and precision is a template parameter constrained by `PrecisionPolicyConcept` to exactly the four supported policies (`DoublePrecision`, `SinglePrecision`, `MixedPrecision<PromoteStorage>`, `MixedPrecision<ReevaluateKernel>`). `KernelScalarConcept` admits only `float` and `double`. A typo or an unsupported combination is a compile error, not a run-time surprise.
- **Builders return copies.** The `with_...` functions of `Gpr` leave the original untouched and return a modified copy, so a configured trainer is never in an intermediate state.

## Training and inference are different types

A model that is not fitted cannot predict; instead of making `predict` return a "not fitted" error, cxxgpr makes the question impossible to ask.

```{mermaid}
stateDiagram-v2
  [*] --> Gpr : construct (kernel, optimizer)
  Gpr --> Gpr : with_... (returns a modified copy)
  Gpr --> FittedGpr : fit / factor
  FittedGpr --> FittedGpr : predict, predict_into, sample, loo_predict
  FittedGpr --> FittedGpr : refit
  FittedGpr --> Gpr : into_trainer
  FittedGpr --> Files : save, save_with_factor
  Files --> LoadedGpr : load
  LoadedGpr --> FittedGpr : into_fitted
```

`Gpr` has no `predict`, no `save` and no accessor for a factor. `FittedGpr` has no `fit`; it has `refit`, which starts a new optimization from its hyperparameters, and `into_trainer`, which returns to an unfitted `Gpr` with the fitted hyperparameters as the starting point. The same split applies to preprocessing: `UnfittedInputTransform` and `UnfittedTargetTransform` can only `fit`, producing `FittedInputTransform` and `FittedTargetTransform`, which can `apply` and `inverse_apply`.

The benefit is that a whole class of run-time checks disappears, and the state of an object is visible in its type. The cost is that you pass a `FittedGpr` rather than a flag around, and that a persisted model must be turned back into a `FittedGpr` explicitly.

## Ownership is explicit

Three kinds of objects exist, and the rule for each is simple.

- **Results are owned values.** `Prediction`, `PredictiveCovariance`, sample vectors and `FittedGpr` own their memory (`std::vector`, `std::unique_ptr`). They outlive the buffers you passed in.
- **Inputs are non-owning views.** `MatrixRef<T>`, `MatrixMut<T>` and `std::span` describe memory that you own. The library reads it during the call and does not keep a pointer. `Gpr::fit` copies what it needs, including the raw training data, into the `FittedGpr`; you may free your arrays right after fitting.
- **Workspaces are owned by whoever created them.** A `QueryWorkspace` that you construct is yours; the one inside a `FittedGpr` belongs to that object, which is why `predict_into` is not thread-safe on a shared model but a copy of the model starts with its own workspace.

Output parameters ("`_into`" functions) let you control allocation: you provide the destination and the library writes into it.

## Column-major data

Matrices are column-major, with element $(i, j)$ of an $n \times d$ matrix at index $j\,n + i$ (more generally $j \cdot \text{col\_stride} + i$). A feature column is therefore a contiguous run of memory.

This matches Eigen, BLAS, LAPACK, Fortran, MATLAB and R, so data from those worlds can be passed without copying, and it matches how kernels are evaluated: distances and Gram matrices are built column by column, and a lower-triangular factorization traverses columns. The view types carry `col_stride`, so a sub-block of a bigger matrix is a view with no copy.

The cost falls on code that stores samples as rows (a `std::vector<std::array<double, d>>`, a C array of structs, NumPy default order). Such data must be transposed once. The raw-span overloads `fit(x, n_rows, n_cols, y)` take a flat column-major buffer; a row-major $n \times d$ buffer read this way would silently be a different, $d \times n$ matrix. {doc}`../how-to/prepare-data` shows how to lay data out.

## No third-party types in public headers

The headers under `include/cxxgpr/` include only the C++ standard library and each other. Eigen, external BLAS/LAPACK (OpenBLAS or Intel oneMKL), nlohmann/json (persistence), and OpenMP (parallelism) are implementation details of the static library and appear in no signature, type or macro.

```{mermaid}
flowchart LR
  subgraph yours ["Your program"]
    U["cxxgpr headers<br/>(standard library types only)"]
  end
  subgraph lib ["libcxxgpr (static)"]
    API["Public API"] --> IMPL["Implementation"]
    IMPL --> E["Eigen + OpenBLAS/MKL"]
    IMPL --> J["nlohmann/json"]
    IMPL --> O["OpenMP"]
  end
  U --> API
```

This keeps your compile times short, your include path free of Eigen and JSON headers, and the library's internal choices replaceable without changing the API. Kernel trees hide behind `CompiledKernel<T>` (a pointer-to-implementation), and persistence takes and returns plain strings and paths.

The boundary is a compile-time boundary only. Linking still needs the OpenMP runtime if the library was built with OpenMP, which the installed CMake package takes care of through `find_dependency`, and the third-party licenses still apply to the binary you ship (see `THIRD_PARTY_NOTICES.md`). The bundled build recipe is in {doc}`../reference/build-options`.

## Modules and their dependencies

The library is organized in nine modules. Dependencies run in one direction: a module may use the modules to its left in the diagram, never the other way round.

```{mermaid}
flowchart LR
  core["core<br/>errors, views, intervals,<br/>likelihood, rng, build info"]
  precision["precision<br/>policies, refinement"]
  linalg["linalg<br/>Cholesky, jitter,<br/>parallel helpers"]
  kernel["kernel<br/>leaves, KernelSpec,<br/>CompiledKernel, distances"]
  transform["transform<br/>input and target maps"]
  workspace["workspace<br/>buffers, MLL, inference,<br/>LeafCache"]
  optimizer["optimizer<br/>Lbfgs, NelderMead, ..."]
  gpr["gpr<br/>Gpr, FittedGpr"]
  persist["persist<br/>save and load"]

  core --> precision --> linalg --> kernel --> workspace --> gpr --> persist
  core --> transform --> gpr
  core --> optimizer --> gpr
```

| Module | Namespace | You use it to | Guide |
|---|---|---|---|
| `core` | `cxxgpr` | Handle errors, pass matrices, set intervals, seed random numbers | {doc}`../how-to/handle-errors`, {doc}`../how-to/prepare-data` |
| `precision` | `cxxgpr` | Choose `DoublePrecision`, `SinglePrecision`, `MixedPrecision`; refine solutions | {doc}`../how-to/optimize-performance` |
| `linalg` | `cxxgpr::linalg` | Factor, solve, jitter, deterministic parallel loops | {doc}`../how-to/use-low-level-building-blocks` |
| `kernel` | `cxxgpr` | Build, compile and evaluate kernels; supply distances; write custom leaves | {doc}`../how-to/build-kernels` |
| `transform` | `cxxgpr::transform` | Standardize, scale or otherwise map inputs and targets | {doc}`../how-to/transform-data` |
| `workspace` | `cxxgpr` | Drive marginal likelihood, gradient, Hessian and inference on your own buffers | {doc}`../how-to/use-low-level-building-blocks` |
| `optimizer` | `cxxgpr::optimizer` | Choose and configure the hyperparameter optimizer | {doc}`../how-to/choose-optimizer` |
| `gpr` | `cxxgpr` | Fit and predict with `Gpr` and `FittedGpr` | {doc}`../how-to/configure-fit` |
| `persist` | `cxxgpr::persist` | Save and load models | {doc}`../how-to/persist-models` |

`cxxgpr/gpr/gpr.hpp` includes the headers of every module it needs, so most programs include only that header and `cxxgpr/kernel/kernel.hpp`. The lower modules can be used on their own: `linalg` and `precision` do not need a model at all.

## Determinism and performance are design constraints

Reproducible numerics and allocation-free hot paths were requirements from the start and shape the API: `_into` functions, explicit workspaces, seeds as arguments, a fixed reduction order. The reasoning is in {doc}`determinism-and-performance` and {doc}`numerical-stability`.

## What these choices cost

| Choice | Cost to you |
|---|---|
| Errors as values | A check at every call |
| Validating factories | `create(...)` plus `.value()` or a test for every kernel and likelihood |
| Separate fitted type | Persisted models must be converted with `into_fitted`; `refit` for new data |
| Column-major views | Row-major data must be transposed once |
| Private third-party types | No access to the underlying Eigen matrices; the supported way to reach the numerics is the low-level API |
| Static library only | The library is linked into each program; a shared library is not offered |
| Pre-1.0 versions | Defaults and auxiliary fields may still change; see {doc}`../reference/compatibility` |
