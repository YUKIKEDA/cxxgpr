# How to choose and configure an optimizer

A `Gpr` trainer chooses its kernel hyperparameters and noise variance by minimizing the negative log marginal likelihood. This guide shows how to pick one of the five built-in optimizers, configure it, read its result, make multi-start runs reproducible, drive an optimizer on your own objective, and plug in an optimizer you wrote. For why there are five and how each one works, see {doc}`../explanation/optimizers`.

The programs behind this page are `optimizers.cpp` (everything except the first table) and `transforms.cpp`. Include `cxxgpr/optimizer/optimizer.hpp` to get every class named here.

## Know what is being optimized

The optimizer minimizes one scalar function of $\theta$, the vector of **natural logarithms** of all hyperparameters: the kernel parameters in the order of `KernelSpec::get_params`, then the noise variance last. The function is the negative log marginal likelihood (NLML), evaluated on the transformed data. Each parameter has an open interval in user units, for example `(1e-5, 1e5)` by default; the optimizer works strictly inside it.

Three consequences matter in practice:

- **The starting point is your kernel.** The values you passed to `RbfKernel::create(1.0)` or `with_noise_variance(1e-2)` are where the optimizer starts. A start far from a sensible scale can end in a poor local optimum (see the warning below).
- **Bounds decide what is reachable.** Pass an `Interval` to the kernel `create` overloads to keep a length scale out of regions you know are wrong.
- **Lower is better.** `FittedGpr::log_marginal_likelihood()` returns the final value of the *negative* log marginal likelihood, and so does `OptimizerResult::value`.

## Choose an optimizer

| Optimizer | Needs | Uses | Pick it when | Cost per iteration |
|---|---|---|---|---|
| `Lbfgs` | gradient | quasi-Newton steps, Moré–Thuente line search | the default; smooth problems with few to moderate parameters | one value-and-gradient evaluation (a few per line search) |
| `TrustRegion` | gradient and Hessian | Newton steps in a trust region | few parameters and you want the fewest iterations, or the surface is ill-conditioned | one value, gradient and Hessian evaluation |
| `NelderMead` | values only | simplex search | a quick derivative-free check, or a surface where gradients are unreliable | one value evaluation, a few per iteration |
| `FastSimulatedAnnealing` | values only | stochastic coordinate search | rough, multimodal surfaces where local methods get trapped | one value evaluation, cheap incremental update for sum kernels |
| `Fixed` | nothing | one evaluation | you already know the hyperparameters; `fit` then only factors the kernel matrix | none (zero iterations) |

The `Gpr` objective provides values, gradients and Hessians, so every optimizer works with a model. Start with `Lbfgs`. Switch to `TrustRegion` when the optimization takes many iterations or stalls, and add restarts (below) when different starts give different answers.

## Compare them on one problem

This program fits the same noisy one-dimensional data with every optimizer, then adds a custom one (see the end of this guide):

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [compare]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
truth at 0.35 = 1.0382
optimizer               iters  converged  neg-log-ML mean@0.35
Lbfgs                      18        yes    -14.5571    1.0564
TrustRegion                16        yes    -14.5571    1.0564
NelderMead                101        yes    -14.5571    1.0564
FastSimulatedAnnealing    400        yes    -14.5462    1.0561
Lbfgs + 3 restarts         18        yes    -14.5571    1.0564
Fixed                       0        yes    -10.4670    1.0605
RandomSearch (custom)     300        yes    -14.4880    1.0560
```

Reading the table:

- The three local optimizers find the same minimum to four decimals. `TrustRegion` and `Lbfgs` need 16 to 18 iterations; `NelderMead` needs six times as many because it uses no derivatives.
- `FastSimulatedAnnealing` spends its full iteration budget (400 here) and lands near, not at, the minimum.
- `Fixed` does not optimize: it evaluates the initial hyperparameters once. Its NLML of $-10.47$ is worse than the optimized $-14.56$, though the prediction at 0.35 is still close to the truth.
- A lower NLML means the optimizer found better hyperparameters, so that column is the one to compare. All predictions are close here because the data are easy.

:::{note}
On this problem `Lbfgs` reaches the same minimum (NLML $-14.56$) from every starting length scale between $0.05$ and $5$. That is not a guarantee in general: a local optimizer ends in whichever minimum its start leads to, so for multimodal objectives add `with_restarts` or use `FastSimulatedAnnealing`, and compare `neg_log_marginal_likelihood()` across candidate runs.
:::

## Configure an optimizer

Each optimizer is a small value type. Its `with_*` methods return a modified copy; the original is untouched and none of them can fail. Validation happens in `optimize` (and so inside `fit`), which returns `GprErrorCode::InvalidConfig` for a bad setting.

### `Lbfgs`

| Setting | Method | Accessor | Default | Valid |
|---|---|---|---|---|
| iterations per start | `with_max_iterations` | `max_iterations` | 100 | any |
| convergence tolerance | `with_tolerance` | `tolerance` | $\sqrt{\varepsilon}\approx 1.5\times 10^{-8}$ | finite, $\ge 0$ |
| stored curvature pairs | `with_history_size` | `history_size` | 10 | $\ge 1$ |
| restarts | `with_restarts` | `restarts` | none | — |

### `TrustRegion`

| Setting | Method | Accessor | Default | Valid |
|---|---|---|---|---|
| iterations per start | `with_max_iterations` | `max_iterations` | 100 | any |
| convergence tolerance | `with_tolerance` | `tolerance` | $\sqrt{\varepsilon}$ | finite, $\ge 0$ |
| initial radius | `with_initial_radius` | `initial_radius` | 1.0 | finite, $> 0$, $\le$ max radius |
| maximum radius | `with_max_radius` | `max_radius` | 100.0 | finite, $\ge$ initial radius |
| restarts | `with_restarts` | `restarts` | none | — |

Radii are measured in the unconstrained coordinates described in {doc}`../explanation/optimizers`. The defaults rarely need changing.

### `NelderMead`

| Setting | Method | Accessor | Default | Valid |
|---|---|---|---|---|
| iterations per start | `with_max_iterations` | `max_iterations` | 100 | any |
| convergence tolerance (value spread and simplex size) | `with_tolerance` | `tolerance` | $\sqrt{\varepsilon}$ | finite, $\ge 0$ |
| restarts | `with_restarts` | `restarts` | none | — |

### `FastSimulatedAnnealing`

| Setting | Method | Accessor | Default | Valid |
|---|---|---|---|---|
| iterations per start (also the cooling horizon) | `with_max_iterations` | `max_iterations` | 100 | any |
| initial temperature $T_0$ | `with_initial_temperature` | `initial_temperature` | 1.0 | finite, $> 0$ |
| cooling rate $c$ | `with_cooling_rate` | `cooling_rate` | 3.0 | finite, $\ge 0$ |
| random seed | `with_seed` | `seed` | 0 | any |
| out-of-bounds proposals | `with_boundary_policy` | `boundary_policy` | `BoundaryPolicy::Clamp` | `Clamp` or `Reflect` |
| restarts | `with_restarts` | `restarts` | none | — |

`BoundaryPolicy::Clamp` places an out-of-bounds proposal at the nearest point strictly inside the interval; `BoundaryPolicy::Reflect` folds it back across the boundary. The iteration count is also the length of the cooling schedule, so more iterations means slower, gentler cooling rather than just a longer tail. Unlike the others, `FastSimulatedAnnealing` has its own `with_seed`, and its restarts add the run index to that seed.

### `Fixed`

`Fixed` has no settings. Use it as the optimizer type, `Gpr<optimizer::Fixed>`, when the hyperparameters are known, or call `Gpr::factor` on any trainer to factor at the current hyperparameters without optimizing.

### Tolerance and iteration budget

The iteration limit applies per start. When an optimizer stops at the limit without meeting its tolerance, `OptimizerResult::converged` is `false`, the best point found is still returned, and `fit` succeeds. It does not fail with `GprErrorCode::OptimizationNotConverged`. Check `FittedGpr::optimizer_result()` if you care:

```cpp
const auto& r = model->optimizer_result();   // OptimizerResult
// r.params: best log-space parameters, r.value: NLML there,
// r.iterations: iterations of the best run, r.converged: stopping test met
```

## Make restarts reproducible

Local optimizers can end in different local minima from different starts. `with_restarts(count, seed)` runs the optimizer from the initial point and from `count` extra points drawn uniformly (in log space) inside the bounds, and returns the best result. The extra points come from a seeded `Xoshiro256PlusPlus` generator, so a given `(count, seed)` always produces the same sequence:

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [restarts]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
restarts with the same seed give bit-identical results: yes
```

`RestartConfig` is the plain struct that stores the pair (`count` and `seed`); `restarts()` returns it, and it has `operator==`. Restarts cost `count + 1` times as much, so use a small `count`. Changing the seed gives a different, equally reproducible set of extra starting points.

## Use an optimizer on your own objective

The optimizers are not tied to Gaussian processes. They minimize any function that implements the objective interface. Derive from `Objective` for value-only optimizers, from `Differentiable` when you can provide a gradient (`Lbfgs`), and from `TwiceDifferentiable` when you can also provide a Hessian (`TrustRegion`):

| Class | You implement | Optimizers that accept it |
|---|---|---|
| `Objective` | `dim`, `value`, `fill_intervals`; optionally `value_at_changes` | `NelderMead`, `FastSimulatedAnnealing`, `Fixed` |
| `Differentiable` | also `gradient_into`, `value_and_gradient_into` | the above and `Lbfgs` |
| `TwiceDifferentiable` | also `hessian_into`; optionally `value_gradient_hessian_into` | all of the above and `TrustRegion` |

The parameters you receive are logarithms. `fill_intervals` states the open interval of each *user-space* parameter; the optimizer takes its logarithm, so every `lo` must be positive. Here is a complete quadratic objective with analytic derivatives:

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [objective]
:end-before: // [end]
:dedent: 4
```

Hessians are stored as a flattened `dim * dim` row-major array, entry $(i, j)$ at `i * dim + j`. `value_at_changes` defaults to a call of `value`; override it only to exploit the single-coordinate updates that `FastSimulatedAnnealing` makes. `value_gradient_hessian_into` defaults to calling `value_and_gradient_into` and then `hessian_into`; override it to share work between the three.

Run any optimizer with `optimize(objective, initial_params)`; `initial_params` must lie strictly inside the log-space bounds:

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [direct]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
Lbfgs: converged = true, iterations = 13, value = 1.83e-21
  parameters = 2.0000 0.5000 10.0000 
TrustRegion value = 2.82e-15 (7 iterations)
NelderMead  value = 3.62e-18 (128 iterations)
FSA         value = 1.60e-03 (3000 iterations)
Fixed       value = 1.86e+00 (0 iterations)
```

On this smooth problem the derivative-based optimizers reach the exact minimum in 7 to 13 iterations, `NelderMead` needs 128, and `FastSimulatedAnnealing` gets close after 3000. Use these as typical orders of magnitude, not as a benchmark: the best choice depends on your problem.

`OptimizerResult` holds `params` (the best log-space vector), `value`, `iterations` and `converged`. `Fixed::optimize` returns `initial_params` with `iterations = 0` and `converged = true`.

### Errors from `optimize`

| Code | Cause |
|---|---|
| `GprErrorCode::InvalidConfig` | negative or non-finite tolerance, `history_size == 0`, bad radii, non-positive temperature, negative cooling rate |
| `GprErrorCode::EmptyInput` | `obj.dim() == 0` or empty `initial_params` |
| `GprErrorCode::LengthMismatch` | `initial_params.size() != obj.dim()` |
| `GprErrorCode::NonFiniteInput` | non-finite start, value, gradient or Hessian |
| `GprErrorCode::InvalidInterval` | an interval is not `0 < lo < hi`, or the start lies outside it |

Any error returned by your own `value` or derivative functions is passed through unchanged.

## Work with bounds directly

`LogBounds` is the coordinate map shared by the optimizers. It converts the open intervals into log space, and maps between the bounded log parameters $\theta$ and an unconstrained vector $z$ in which the optimizers move.

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [log-bounds]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
z = (0.6985, -0.6985, -0.0000); round trip error = 0.0000
```

The midpoint of an interval maps to $z = 0$, which is why the last component above is zero (the start $\log 1 = 0$ is the midpoint of $(\log 0.01, \log 100)$). `LogBounds` is created with `LogBounds::from_objective(obj)` or `LogBounds::from_intervals(intervals)`, which fail with `GprErrorCode::EmptyInput` or `GprErrorCode::InvalidInterval`. Its members are:

| Member | Purpose |
|---|---|
| `dim` | number of parameters |
| `lo_log`, `hi_log` | the log-space bounds |
| `validate_log_params` | check that $\theta$ is finite and strictly inside the bounds |
| `to_unconstrained` | $\theta \to z$ (fails if $\theta$ is outside the bounds) |
| `to_constrained` | $z \to \theta$ (never fails, result stays strictly inside) |
| `to_constrained_with_jacobian` | also returns $d\theta/dz$ |
| `to_constrained_with_derivatives` | also returns $d^2\theta/dz^2$ |
| `chain_gradient`, `chain_hessian` | convert a gradient or Hessian from $\theta$ to $z$ |
| `sample_log_uniform` | draw a deterministic random point strictly inside the bounds |

`EvalCache` remembers the most recent evaluation so a line search does not pay twice for the same point. Pass it the objective, the bounds, the point $z$ and scratch buffers:

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [eval-cache]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
two evaluations at the same point cost 1 objective call(s)
```

`EvalCache::eval_unconstrained` returns `GprErrorCode::NonFiniteInput` for a non-finite point, value or gradient. Call `clear` to drop the cached point. You need `EvalCache` only when writing an optimizer that evaluates points repeatedly.

## Write your own optimizer

`Gpr<O>` accepts any copyable type `O` that provides

- `static constexpr bool uses_change_indices` (read through `OptimizerTraits<O>::uses_change_indices`); set it to `true` only if your optimizer calls `value_at_changes` with the indices of the coordinates it changed, which lets the model reuse cached kernel matrices, and
- `std::expected<OptimizerResult, GprError> optimize(TwiceDifferentiable& obj, std::span<const double> initial_params) const`.

The model hands you the same objective the built-in optimizers see. The one below perturbs the best point and keeps improvements, using `LogBounds`, `Xoshiro256PlusPlus` and `OptimizerResult`:

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [custom-optimizer]
:end-before: // [end]
:dedent: 4
```

It reaches an NLML of $-14.49$ in the comparison table above, close to the $-14.56$ of the gradient-based optimizers. Use it like a built-in: `Gpr<RandomSearch>(kernel, RandomSearch{})`.

:::{note}
The model passes your optimizer a `TwiceDifferentiable` objective, so you may call `gradient_into` and `hessian_into` if you want them. Only `NelderMead` and `FastSimulatedAnnealing` are treated by the model as value-only; a custom optimizer gets no other special treatment.
:::

Check the points your optimizer proposes with `LogBounds::validate_log_params` (or draw them with `sample_log_uniform`) so they stay strictly inside the open bounds; a point outside them is rejected by the objective with an error such as `GprErrorCode::InvalidInterval`.

## API summary

| Symbol | Purpose |
|---|---|
| `Lbfgs` | limited-memory BFGS; `with_max_iterations`, `with_tolerance`, `with_history_size`, `with_restarts`, the matching accessors, `optimize`, `operator==` |
| `TrustRegion` | trust-region Newton; `with_max_iterations`, `with_tolerance`, `with_initial_radius`, `with_max_radius`, `with_restarts`, accessors, `optimize`, `operator==` |
| `NelderMead` | simplex search; `with_max_iterations`, `with_tolerance`, `with_restarts`, accessors, `optimize`, `operator==` |
| `FastSimulatedAnnealing` | annealing; `with_max_iterations`, `with_initial_temperature`, `with_cooling_rate`, `with_seed`, `with_boundary_policy`, `with_restarts`, accessors, `optimize`, `operator==` |
| `BoundaryPolicy` | `Clamp` or `Reflect` for out-of-bounds annealing proposals |
| `Fixed` | evaluates once; `optimize`, `operator==` |
| `Objective`, `Differentiable`, `TwiceDifferentiable` | objective interfaces: `dim`, `value`, `value_at_changes`, `fill_intervals`, `gradient_into`, `value_and_gradient_into`, `hessian_into`, `value_gradient_hessian_into` |
| `OptimizerResult` | `params`, `value`, `iterations`, `converged`, `operator==` |
| `RestartConfig` | `count`, `seed`, `operator==` |
| `OptimizerTraits` | `uses_change_indices` of an optimizer type |
| `LogBounds` | log-space bounds and the $\theta \leftrightarrow z$ map (members in the table above) |
| `EvalCache` | single-entry cache; `eval_unconstrained`, `clear` |

The generated reference is in [the optimizer API reference](../reference/api/optimizer.md).
