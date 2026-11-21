# How to configure a fit

This guide shows how to set up a `Gpr` trainer before training: observation noise, jitter, precision, optimizer, exponential backend and workspace memory. It ends with a table of every public symbol involved. The snippets come from one complete program, listed at the end of the page.

Every snippet assumes `using namespace cxxgpr;` and the headers `cxxgpr/gpr/gpr.hpp`, `cxxgpr/kernel/kernel.hpp`, `cxxgpr/optimizer/optimizer.hpp` and `cxxgpr/transform/transform.hpp`.

## Start from a trainer

`Gpr<O, P, K>` is the *unfitted* trainer. Its three template parameters are the optimizer `O` (default `optimizer::Lbfgs`), the precision policy `P` (default `DoublePrecision`) and the kernel specification `K` (always `KernelSpec`). A trainer offers no `predict`; training turns it into a `FittedGpr` (see {doc}`predict-and-sample`).

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
```

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [builder]
:end-before: // [end]
```

The constructor `Gpr(KernelSpec, O)` takes the kernel (see {doc}`build-kernels`) and an optional optimizer instance. Every `with_*` method is `const` and returns a modified **copy**; the original trainer never changes. That makes it cheap and safe to derive several variants from one base.

A new trainer starts with these settings:

| Setting | Default | Change with |
|---|---|---|
| Observation noise variance | `1e-2`, bounds $(10^{-5}, 10^{5})$ | `with_noise_variance`, `with_likelihood` |
| Input / target transform | identity | `with_input_transform`, `with_target_transform` |
| Jitter policy | fixed, offset `0` (no retry) | `with_jitter` |
| Exponential backend | `KernelExp::Accurate` | `with_kernel_exp` |
| Workspace policy | `FitPolicy::prefer_speed()` | `with_prefer_speed`, `with_prefer_memory`, `with_distance_cache_policy`, `with_cholesky_buffer` |
| Optimizer | `O{}` | `with_optimizer` |

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [defaults]
:end-before: // [end]
```

```{code-block} text
default jitter kind is fixed, kernel_exp is Accurate, policy is prefer_speed
```

:::{note}
The matching getters are `kernel_spec()`, `optimizer()`, `likelihood()`, `jitter()`, `kernel_exp()` and `fit_policy()`. A `FittedGpr` offers the same getters (it retains the configuration for `refit` and `into_trainer`).
:::

Copying a `Gpr` deep-copies its transforms; moving it is `noexcept`. Copy assignment and move assignment are available, so a trainer can be stored in a container or reassigned.

## Set the observation noise

The noise variance $\sigma_n^2$ is a hyperparameter: unless you use `optimizer::Fixed`, it is the **starting value** the optimizer improves on.

`with_noise_variance` validates its argument and therefore returns `std::expected<Gpr, GprError>`:

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [noise]
:end-before: // [end]
```

```{code-block} text
noise variance = 0.0500 in (0.0001, 1.0000)
```

To change the search bounds, build a `GaussianLikelihood`, narrow it with `GaussianLikelihood::with_bounds`, and pass it to `with_likelihood` (which cannot fail, because the likelihood is already validated). `GaussianLikelihood::from_log_noise_variance` builds one from $\log\sigma_n^2$ instead.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [reject]
:end-before: // [end]
```

```{code-block} text
negative noise rejected: 1
```

:::{warning}
Bounds are **open** intervals. The default interval is $(10^{-5}, 10^{5})$, so a noise variance of exactly `1e-5` is rejected with `GprErrorCode::InvalidInterval`; use `2e-5` or widen the bounds. A zero or negative variance gives `GprErrorCode::InvalidNoiseVariance`.
:::

:::{tip}
Noise is separate from jitter. Noise is statistical (the sensors are imprecise); jitter is numerical (a rescue when the Cholesky factorization fails). Never raise the noise to fix a numerical failure unless the data really are that noisy.
:::

## Choose between `fit` and `factor`

| | `fit` | `factor` |
|---|---|---|
| Optimizes hyperparameters | yes (unless `O` is `optimizer::Fixed`) | no |
| Hyperparameters used | optimizer result | exactly those on the trainer |
| Workspace policy | trainer's `FitPolicy` | always `FitPolicy::prefer_memory()` |
| `optimizer_result().iterations` | optimizer iterations | `0` |
| Typical use | learn hyperparameters from data | condition on data at known hyperparameters |

Both exist in two overloads: one taking a `MatrixRef<double>` plus `std::span<const double>` targets, and one taking a raw column-major `std::span<const double>` with `n_rows` and `n_cols`. Input is **column-major**: for $n$ rows and $d$ columns, element $(i, j)$ is at index `j * n + i`.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [fit-factor]
:end-before: // [end]
```

```{code-block} text
factor: neg-MLL 4.7716, noise 0.0500, iterations 0
fit:    neg-MLL 0.7080, noise 0.0227, iterations 17, converged 1
```

`neg_log_marginal_likelihood()` returns the **negative** log marginal likelihood (lower is better). The optimized fit lowers it from 4.77 to 0.71 and finds a noise variance of 0.023 on standardized targets. That matches the data: 0.01 of noise was added to a signal whose variance is about 0.45, and 0.01 / 0.45 is about 0.022.

:::{note}
A fit that does not converge is **not** an error. Check `optimizer_result().converged` (and `iterations`) on the fitted model. See {doc}`choose-optimizer`.
:::

## Pick a jitter policy

When the Cholesky factorization of $K + \sigma_n^2 I$ fails, the trainer retries with a small diagonal offset ("jitter"). `JitterPolicy` selects the strategy:

- `JitterPolicy::fixed(j)` tries offset `0` first and, if that fails, retries once with `j`. `fixed(0.0)` never retries. The default trainer uses this with `j = 0`.
- `JitterPolicy::adaptive(initial, multiplier, max_retries, max_jitter)` retries with `initial`, then `initial * multiplier`, and so on until the factorization succeeds, `max_retries` is used up, or the next offset would exceed `max_jitter`.

Both factories return `std::expected<JitterPolicy, GprError>` and reject non-finite or out-of-range arguments with `GprErrorCode::InvalidConfig`.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [jitter]
:end-before: // [end]
```

```{code-block} text
accepted factor jitter: 0.0000e+00
```

The accepted offset is reported by `FittedGpr::factor_jitter()`. Here it is zero: the factorization succeeded without help. If it fails with every offset, the call returns `GprErrorCode::CholeskyFailed` (see {doc}`handle-errors`).

Inspect a policy with `kind()` (`JitterPolicy::Kind::Fixed` or `JitterPolicy::Kind::Adaptive`), `is_fixed()`, `is_adaptive()`, `as_fixed()` and `as_adaptive()`. The last two return `std::optional<FixedJitter>` and `std::optional<AdaptiveJitter>`; `FixedJitter::jitter()` and `AdaptiveJitter::initial()`, `multiplier()`, `max_retries()`, `max_jitter()` read the stored values. `retry_schedule()` returns a `JitterSchedule` whose `next()` yields each offset in turn and then `std::nullopt`:

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [jitter-schedule]
:end-before: // [end]
```

```{code-block} text
adaptive retry ladder: 1.0000e-10 1.0000e-09 1.0000e-08 1.0000e-07 1.0000e-06 1.0000e-05 1.0000e-04
max_retries 8, multiplier 10.0000, fixed retry 1.0000e-08, fixed has no adaptive part: 1
```

The ladder stops at seven rungs although `max_retries` is 8, because the next value (`1e-3`) would exceed `max_jitter = 1e-4`.

`FixedJitter`, `AdaptiveJitter` and `JitterSchedule` have public constructors, but they do not validate. Prefer the `JitterPolicy` factories.

Posterior sampling has its own jitter, passed to `FittedGpr::sample` (see {doc}`predict-and-sample`).

## Choose a precision policy

The precision policy `P` fixes the scalar type used to store the kernel matrix and Cholesky factor (`P::Storage`) and the type of weights and outputs (`P::Refine`):

| Policy | `Storage` | `Refine` | `refines_in_f64` | Notes |
|---|---|---|---|---|
| `DoublePrecision` | `double` | `double` | `false` | Default. |
| `SinglePrecision` | `float` | `float` | `false` | Half the memory; results in `float`. |
| `MixedPrecision<PromoteStorage>` | `float` | `double` | `true` | `float` factor, `double` weights by iterative refinement; residuals promote the stored `float` matrix. |
| `MixedPrecision<ReevaluateKernel>` | `float` | `double` | `true` | As above, but residuals re-evaluate the kernel entries in `double` instead of promoting the stored `float` values. |

`MixedPrecision` takes a `ResidualFormulaConcept` type, either the tag `PromoteStorage` or `ReevaluateKernel`; its default is `PromoteStorage`. `KernelScalarConcept` accepts `float` or `double`. `PrecisionPolicyConcept` (alias `GpScalarConcept`) accepts exactly the four policies above; any other type is rejected at compile time.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [precision]
:end-before: // [end]
```

```{code-block} text
single neg-MLL 0.7079, mixed neg-MLL 0.7079, refinement iterations 2
```

For mixed policies, `FittedGpr::refinement_summary()` holds a `RefinementSummary` (`status`, `iterations`, `final_relative_residual`, `used_fallback()`); for the other two it is `std::nullopt`. `FittedGpr::Storage` and `FittedGpr::Refine` expose the policy's types.

Each policy also reports how it is stored on disk: `persist_kind()` returns a `PersistPrecisionKind` (`Double`, `Single`, `MixedPromote`, `MixedReevaluate`), and `to_precision_string` / `to_residual_string` turn it into the strings used in the saved configuration. The tags `PromoteStorage` and `ReevaluateKernel` have `name()` and `reads_storage`.

Two small `inline` helpers apply a precision policy to raw buffers. `copy_storage_to_refine<P>` widens a `P::Storage` span into a `P::Refine` span; `compute_storage_means<P>` computes $K_*^\top\alpha$ with `double` accumulation. Both return `GprErrorCode::EmptyInput` or `GprErrorCode::LengthMismatch` on bad spans.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [precision-helpers]
:end-before: // [end]
```

```{code-block} text
mixed / promote_storage, reevaluate variant: reevaluate_kernel, tag name reevaluate_kernel
k^T alpha = -1.0000, -1.0000
```

:::{tip}
Use `DoublePrecision` unless memory or speed forces otherwise. `MixedPrecision` gives near-double accuracy with a `float` factorization; `SinglePrecision` can fail to factor matrices that `DoublePrecision` handles, so pair it with an adaptive jitter policy. For persistence of a trained model by precision, see {doc}`persist-models`.
:::

## Set the optimizer, restarts and iteration limits

`with_optimizer` swaps the optimizer. Each built-in optimizer is configured by its own `with_*` methods; for example `optimizer::Lbfgs::with_restarts(count, seed)` adds `count` deterministic random restarts (log-uniform starting points drawn from a `Xoshiro256PlusPlus` seeded by `seed`) and keeps the best result.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [optimizer]
:end-before: // [end]
```

```{code-block} text
with 3 restarts: neg-MLL 0.7080
```

Restarts guard against poor local minima: the best result over all starts is kept. `optimizer::Fixed` skips the search and evaluates the objective once at the trainer's hyperparameters. The other optimizers and their options are described in {doc}`choose-optimizer`.

## Choose the exponential backend

`with_kernel_exp(KernelExp::FastApprox)` replaces `std::exp` inside the built-in kernels with a range-reduced degree-7 polynomial. It speeds up kernel evaluation at a small cost in accuracy. `KernelExp::Accurate` (the default) uses `std::exp`. The setting does not apply to parameter transforms or to `CustomKernel` leaves. The line using it is part of the next example.

## Trade memory against speed

`FitPolicy` combines two independent choices:

| Field | Values | Effect |
|---|---|---|
| `distance_cache` | `DistanceCachePolicy::Cached` / `Uncached` | Keep $O(n^2)$ squared distances between optimizer steps, or recompute them from the coordinates. |
| `cholesky_buffer` | `CholeskyBuffer::Retain` / `Reuse` | Keep a separate $n \times n$ buffer for the gradient weight matrix, or overwrite `K` and refactorize afterwards. |

Presets: `FitPolicy::prefer_speed()` is `Cached` + `Retain` (the default); `FitPolicy::prefer_memory()` is `Uncached` + `Reuse`. The trainer offers `with_prefer_speed`, `with_prefer_memory`, and the single-field setters `with_distance_cache_policy` and `with_cholesky_buffer`.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [policy]
:end-before: // [end]
```

```{code-block} text
prefer_memory neg-MLL 0.7080, custom policy neg-MLL 0.7080
```

The policy changes memory use and run time, not the result: both models reach the same optimum to the printed digits. `FitPolicy` compares with `==`.

## Add preprocessing

`with_input_transform` and `with_target_transform` clone the transform you pass in, so the same prototype can be reused. They are described in {doc}`transform-data`; the fitted forms are available as `FittedGpr::input_transform()` and `FittedGpr::target_transform()`, and the unfitted prototypes as `unfitted_input_transform()` and `unfitted_target_transform()`.

## Rely on the failure guarantee

`fit` and `factor` are `const`. If they fail, no state changes: the trainer is exactly as before and can be used again.

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [failure]
:end-before: // [end]
```

```{code-block} text
bad targets: input contains a non-finite value (NaN/Inf)
trainer still usable: 1, noise still 0.0500
```

On success, the returned `FittedGpr` is an independent object that owns copies of the training data (`x_raw()`, `y_raw()`), the Cholesky factor (`l_factor()`), the weights (`alpha()`) and its own configuration. Other inspectors are `n_train()`, `n_features()`, `kernel_spec()` and `likelihood()` (both with the *fitted* hyperparameters), `optimizer_result()`, `jitter()`, `kernel_exp()` and `fit_policy()`. Copying a `FittedGpr` gives the copy independent scratch space; moving is `noexcept`.

Errors and their meaning are in {doc}`handle-errors`.

## Complete program

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:linenos:
```

## API summary

| Symbol | Purpose |
|---|---|
| `Gpr`, `Gpr::Gpr` | Unfitted trainer; constructor takes `KernelSpec` and optimizer. |
| `with_optimizer`, `with_likelihood`, `with_noise_variance`, `with_jitter` | Copy with changed optimizer, noise, jitter. |
| `with_input_transform`, `with_target_transform`, `with_kernel_exp` | Copy with changed transforms, exponential backend. |
| `with_distance_cache_policy`, `with_cholesky_buffer`, `with_prefer_memory`, `with_prefer_speed` | Copy with changed workspace policy. |
| `kernel_spec`, `optimizer`, `likelihood`, `jitter`, `kernel_exp`, `fit_policy` | Trainer getters (also on `FittedGpr`). |
| `fit`, `factor` | Train with / without optimization; `const`, strong guarantee. |
| `FittedGpr`, `Storage`, `Refine` | Trained model and its scalar types. |
| `n_train`, `n_features`, `factor_jitter`, `neg_log_marginal_likelihood`, `alpha`, `l_factor`, `x_raw`, `y_raw` | Fitted-model inspectors. |
| `input_transform`, `target_transform`, `unfitted_input_transform`, `unfitted_target_transform` | Fitted and prototype transforms. |
| `refinement_summary`, `optimizer_result` | Diagnostics of refinement and optimization. |
| `GaussianLikelihood`, `create`, `from_log_noise_variance`, `with_bounds`, `noise_variance`, `bounds` | Observation noise (more in {doc}`tune-hyperparameters`). |
| `JitterPolicy`, `fixed`, `adaptive`, `Kind`, `kind`, `is_fixed`, `is_adaptive`, `as_fixed`, `as_adaptive`, `retry_schedule` | Cholesky stabilization strategy. |
| `FixedJitter`, `AdaptiveJitter`, `JitterSchedule`, `next`, `initial`, `multiplier`, `max_retries`, `max_jitter`, `jitter` | Jitter parameter and schedule types. |
| `DoublePrecision`, `SinglePrecision`, `MixedPrecision`, `PromoteStorage`, `ReevaluateKernel` | Precision policies and residual tags. |
| `PrecisionPolicyConcept`, `GpScalarConcept`, `KernelScalarConcept`, `ResidualFormulaConcept` | Compile-time constraints. |
| `refines_in_f64`, `reads_storage`, `name`, `persist_kind`, `PersistPrecisionKind`, `to_precision_string`, `to_residual_string` | Policy properties and persistence names. |
| `copy_storage_to_refine`, `compute_storage_means` | Precision conversion and mean helpers. |
| `FitPolicy`, `prefer_speed`, `prefer_memory`, `DistanceCachePolicy`, `CholeskyBuffer` | Workspace trade-off. |
| `KernelExp` | `Accurate` or `FastApprox` exponential. |
| `RefinementSummary` | Result of mixed-precision refinement. |

Generated reference: [`gpr`](../reference/api/gpr.md), [`core`](../reference/api/core.md), [`precision`](../reference/api/precision.md), [`linalg`](../reference/api/linalg.md).
