# How to tune hyperparameters

This guide shows how to read and write the hyperparameters of a kernel and of the noise likelihood, bound them, warm-start a new fit from a trained model, and evaluate the marginal-likelihood objective (value, gradient and Hessian) yourself. For choosing an optimizer, see {doc}`choose-optimizer`.

The snippets come from one complete program, listed at the end of the page. They assume `using namespace cxxgpr;`.

## Understand the parameter vector

Every hyperparameter has two representations:

- **User units**: the lengthscale, the constant or the noise variance as you would write it down, always strictly positive. Bounds are given here, as open intervals.
- **Log space**: $\theta = \log(\text{value})$. All `get_params` / `set_params` functions, gradients and Hessians use this representation, so the optimizer works on an unconstrained scale.

The full vector the optimizer sees has $p$ entries: the kernel parameters in depth-first, left-to-right leaf order, followed by one entry for the noise, $\log\sigma_n^2$. Gradients and Hessians are with respect to this log-space vector.

## Bound a parameter

`Interval` is a validated **open** interval $(lo, hi)$ with `lo < hi` and finite ends. Build one with `Interval::create(lo, hi)` (any finite ends) or `Interval::positive(lo, hi)` (requires `0 < lo`). `Interval::default_positive()` is the canonical $(10^{-5}, 10^{5})$ used when you give no bounds. Both factories return `std::expected<Interval, GprError>`; invalid bounds give `GprErrorCode::InvalidInterval`. Query an interval with `lo()`, `hi()`, `width()` and `contains(value)`; two intervals compare with `==`.

Kernel leaves accept the interval at creation:

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [bounds]
:end-before: // [end]
```

The optimizer can then never leave $(0.1, 10)$ for the lengthscale. The starting value must lie strictly inside its interval.

`BoundedParam` is the building block: a value in user units that is guaranteed to lie strictly inside its `Interval`. Leaves and `GaussianLikelihood` use it internally; you can use it for your own parameters.

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [bounded-param]
:end-before: // [end]
```

```{code-block} text
ln(2) = 0.6931, with_value(4).value() = 4.0000, with_value(9) fails: 1, interval width 7.5000
```

| Function | Result |
|---|---|
| `BoundedParam::create(value, interval)` | validated parameter, or `GprErrorCode::InvalidInterval` |
| `BoundedParam::default_positive(value)` | same, with the interval $(10^{-5}, 10^{5})$ |
| `value()`, `interval()` | read the value (user units) and the bounds |
| `ln()` | $\log(\text{value})$, the optimizer's coordinate |
| `with_value(v)`, `with_interval(i)` | new parameter; error if the result would be outside |

All of them leave the original unchanged.

## Read the hyperparameters of a kernel

`KernelSpec` is a value type. `num_params()` is the number of log-space parameters; `get_params` fills a span of exactly that length; `bounds_into` fills the matching `Interval`s; `parameter_bindings` maps each flat index to its leaf and the slot inside it (`ParameterBinding`).

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [params]
:end-before: // [end]
```

```{code-block} text
kernel parameters (log): 0.0000, 0.6931  -> constant 1.0000, lengthscale 2.0000
lengthscale bounds: (0.1000, 10.0000)
parameter 0 is leaf 0, local slot 0
parameter 1 is leaf 1, local slot 0
```

A length mismatch returns `GprErrorCode::LengthMismatch`. Apply `std::exp` to turn a log-space entry into user units.

## Set hyperparameters

`set_params` replaces all parameters from a log-space span. It is **atomic**: if any entry is non-finite or lands outside its bounds, the call fails and the specification is left as it was.

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [set-params]
:end-before: // [end]
```

```{code-block} text
out-of-bounds update rejected: 1, spec unchanged: 1
```

Failures are `GprErrorCode::LengthMismatch`, `GprErrorCode::NonFiniteInput` or `GprErrorCode::InvalidInterval`. Since `KernelSpec` has value semantics, copy first if you need the old values. Leaf types (`RbfKernel` and the others) have the same `get_params` / `set_params` pair for their own parameters; see {doc}`build-kernels`.

## Set the noise hyperparameter

The noise is a one-parameter likelihood, `GaussianLikelihood`, parameterized by $\theta = \log\sigma_n^2$ (`num_params()` is always 1).

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [likelihood]
:end-before: // [end]
```

```{code-block} text
noise variance 0.0400 = exp(-3.2189), num_params 1, default bounds (1.0000e-05, 1.0000e+05)
diagonal after adding noise: 1.0400, gradient diagonal 0.0400
```

| Function | Purpose |
|---|---|
| `create(variance)`, `from_log_noise_variance(theta)` | build from user units or log units |
| `noise_variance()`, `log_noise_variance()`, `bounds()` | read |
| `with_bounds(interval)` | new likelihood with other bounds; the current value must stay inside |
| `get_params(out)`, `set_params(params)`, `bounds_into(out)` | length-1 log-space interface (`set_params` is atomic) |
| `add_noise_diag(k_diag)` | add $\sigma_n^2$ to each entry of a diagonal |
| `noise_grad_diag(dk_diag, idx)` | write $\partial K / \partial\theta$ diagonal, $\sigma_n^2$; `idx` must be 0 (else `GprErrorCode::IndexOutOfRange`) |

`create` returns `GprErrorCode::InvalidNoiseVariance` for non-positive or non-finite input and `GprErrorCode::InvalidInterval` if the value lies outside the bounds.

## Read the learned hyperparameters

After `fit`, the model carries the optimized values:

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
```

```{code-block} text
learned constant 1.1308, lengthscale 1.8844, noise 0.0134, neg-MLL -8.8970
```

`FittedGpr::kernel_spec()` and `likelihood()` return the fitted `KernelSpec` and `GaussianLikelihood`. `log_marginal_likelihood()` returns the final **negative** log marginal likelihood: smaller is a better fit. The values here (constant 1.13, lengthscale 1.88, noise variance 0.013) are close to what generated the data (unit signal amplitude, noise variance 0.01).

:::{note}
The method is named `log_marginal_likelihood`, but its value is the *negative* log marginal likelihood, the quantity the optimizer minimizes. Negate it to get the log marginal likelihood.
:::

## Re-tune after new data

`refit` trains a new model on new data, starting the search from this model's hyperparameters and keeping its configuration. `into_trainer` returns the same warm-started configuration as an unfitted `Gpr` so you can still change settings before `fit`. Neither changes the original model.

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [refit]
:end-before: // [end]
```

```{code-block} text
refit: 12 iterations, cold fit: 35 iterations
```

The warm start converges in a third of the iterations and reaches the same optimum (the program checks that the two negative likelihoods agree within 0.05). `refit` has the same `MatrixRef` and raw-span overloads as `fit`, with the same errors.

To evaluate a model at *given* hyperparameters without searching, build a trainer with `optimizer::Fixed` (or call `factor`) and the kernel and likelihood you want:

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [into-trainer]
:end-before: // [end]
```

```{code-block} text
factor at learned values reproduces neg-MLL: -8.8970 vs -8.8970
```

## Evaluate the objective, its gradient and its Hessian

During `fit`, the optimizer receives an `optimizer::TwiceDifferentiable` objective: the negative log marginal likelihood as a function of the log-space vector. You can reach it by writing an optimizer type. The requirements are small:

- `static constexpr bool uses_change_indices` (use `false`),
- `optimize(optimizer::TwiceDifferentiable&, std::span<const double>) const` returning `std::expected<optimizer::OptimizerResult, GprError>`.

The example below evaluates the analytic gradient and Hessian at the starting point and compares them with finite differences.

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [probe]
:end-before: // [end]
```

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [objective]
:end-before: // [end]
```

```{code-block} text
neg-MLL at the start -1.5600, max gradient error 3.6651e-09, max Hessian error 9.3551e-09
```

The objective interface:

| Method | Meaning |
|---|---|
| `dim()` | $p$, the number of log-space parameters (kernel parameters + 1) |
| `value(params)` | negative log marginal likelihood |
| `value_and_gradient_into(params, grad)` | value and gradient in one factorization |
| `hessian_into(params, hess)` | $p \times p$ Hessian, flattened so that entry $(i, j)$ is `hess[i * p + j]` |
| `value_gradient_hessian_into(params, grad, hess)` | all three with one Cholesky factor |

Gradient and Hessian are exact analytic derivatives with respect to $\theta$ (not finite differences); the example shows agreement to about $10^{-8}$. All methods return `std::expected` and propagate errors such as `GprErrorCode::CholeskyFailed`.

The same computation is available without a `Gpr` as the workspace functions `mll_and_gradient_into`, `hessian_into` and `value_gradient_hessian_into`; see {doc}`use-low-level-building-blocks`.

## Complete program

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:linenos:
```

## API summary

| Symbol | Purpose |
|---|---|
| `Interval`, `create`, `positive`, `default_positive`, `lo`, `hi`, `width`, `contains` | Validated open interval in user units. |
| `BoundedParam`, `create`, `default_positive`, `value`, `interval`, `ln`, `with_value`, `with_interval` | Parameter guaranteed inside its interval. |
| `KernelSpec::num_params`, `get_params`, `set_params`, `bounds_into`, `parameter_bindings` | Kernel hyperparameters (log space). |
| `ParameterBinding` | Flat index, leaf id, local slot. |
| `GaussianLikelihood`, `create`, `from_log_noise_variance`, `noise_variance`, `log_noise_variance`, `bounds`, `with_bounds`, `num_params`, `get_params`, `set_params`, `bounds_into`, `add_noise_diag`, `noise_grad_diag` | Noise likelihood. |
| `FittedGpr::kernel_spec`, `likelihood`, `log_marginal_likelihood`, `optimizer_result` | Learned values and objective at the optimum. |
| `refit`, `into_trainer` | Warm start from a fitted model. |
| `optimizer::TwiceDifferentiable`, `dim`, `value`, `value_and_gradient_into`, `hessian_into`, `value_gradient_hessian_into` | The objective seen by an optimizer. |
| `optimizer::OptimizerResult` | Parameters, value, iterations and convergence flag. |

Generated reference: [`core`](../reference/api/core.md), [`gpr`](../reference/api/gpr.md), [`optimizer`](../reference/api/optimizer.md), [`kernel`](../reference/api/kernel.md).
