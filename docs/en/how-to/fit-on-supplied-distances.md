# How to fit a model on supplied distances

This guide shows how to train a Gaussian process whose kernel reads squared distances that you computed yourself (arc lengths, graph distances, edit distances), then predict, sample and cross-validate with it. The trainer is `DistanceGpr` and the fitted model is `FittedDistanceGpr`; both are declared in `cxxgpr/gpr/distance_gpr.hpp`.

Read {doc}`use-supplied-distances` first: it explains distance tables, slots and `DistanceSource`, which this guide reuses without repeating. Signatures are in the [gpr API reference](../reference/api/gpr.md).

## DistanceGpr and Gpr side by side

`DistanceGpr` is the counterpart of `Gpr` for kernels built from `DistanceKernel`. The fitting procedure, the optimizers, the precision policies and the numerical safeguards are the same; only the inputs differ.

| | `Gpr` / `FittedGpr` | `DistanceGpr` / `FittedDistanceGpr` |
|---|---|---|
| Kernel | `KernelSpec` | `DistanceKernel<DistanceOnly>` or `DistanceKernel<WithPoints>` |
| Training data | coordinates `x` (n × d) and `y` | a `DistanceSource` holding squared-distance tables, and `y` (plus `x` for `WithPoints`) |
| Query data | coordinates `xq` | a `DistanceSource` of training-to-query tables (plus `xq` for `WithPoints`) |

A `DistanceOnly` kernel never sees coordinates, so its model needs none. A `WithPoints` kernel mixes distance leaves with ordinary coordinate leaves; its model takes coordinates in addition to the tables.

## Step 1: Describe the data and the kernel

The data are noisy samples of a periodic function on a circle, and the metric is the arc length. Squared arc lengths are what the kernel reads.

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [data]
:end-before: // [end]
```

The kernel is an exponential kernel of the arc length, which is valid on the circle ({doc}`use-supplied-distances` explains why). `DistanceGpr` takes it directly:

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
```

## Step 2: Fit

Bind one table per slot to a `DistanceSource`, then call `fit`. The tables are validated once: they must be finite, non-negative, symmetric and zero on the diagonal, unless you call `DistanceSource::tidy` to allow a documented repair.

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
```

`with_noise_variance` validates its argument and returns `std::expected`, exactly as for `Gpr`. The other configuration methods are `with_jitter`, `with_kernel_exp`, `with_target_transform`, `with_optimizer` and, for `WithPoints` kernels only, `with_input_transform`. The getters are `kernel`, `likelihood`, `jitter` and `optimizer`.

`fit` optimizes the kernel hyperparameters and the noise variance by maximizing the marginal likelihood. To factorize at the current hyperparameters without optimizing, call `factor`. For a `WithPoints` kernel both take the coordinates as an extra argument: `fit(train, x, y)` and `factor(train, x, y)`.

:::{note}
The default precision policy is double precision. `DistanceGpr<Lbfgs, MixedPrecision<ReevaluateKernel>>` and the other policies of {doc}`optimize-performance` work the same way and report `refinement_summary`.
:::

## Step 3: Predict

Build one table of squared distances from every training point to every query point (n × m, with the training points along the rows) and bind it to the same slot. A kernel with several slots needs a table for each.

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
```

`predict` returns the mean and the marginal variance on the original target scale. `predict_into` writes into a `Prediction` you own, so repeated calls reuse its storage. For a `WithPoints` model, pass the query coordinates after the table: `predict(cross, xq)`.

## Step 4: Covariance, samples and cross-validation

The joint predictive covariance and posterior samples also need the table of distances between the query points themselves (m × m):

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [covariance]
:end-before: // [end]
```

- `predict_covariance(cross, query)` returns the full m × m posterior covariance of the queries.
- `sample(cross, query, n_draws, seed)` draws `n_draws` joint samples. The result is deterministic for a given seed, independent of the number of threads.
- `loo_predict` returns leave-one-out predictions at the training points without refitting.

## Step 5: Inspect, update and refit

The inspectors of a `FittedDistanceGpr` are `kernel`, `likelihood` (both with the fitted values), `n_train`, `n_features`, `factor_jitter`, `neg_log_marginal_likelihood`, `alpha`, `l_factor`, `refinement_summary` and `optimizer_result`.

`into_trainer` returns a `DistanceGpr` that starts at the fitted hyperparameters. `refit(train, y)` fits again on new tables and targets with the same configuration, warm-started from the fitted values, and leaves the original model unchanged. Copying a `FittedDistanceGpr` is cheap: copies share the fitted data and keep independent scratch space, so each thread should use its own copy.

## Invalid tables and errors

Every failure is returned as a `GprError`. The ones specific to distances:

| `GprErrorCode` | Cause |
|---|---|
| `InvalidDistance` | a table is non-finite, negative, asymmetric or nonzero on the diagonal |
| `DistanceSlot` | a slot is missing, bound twice, or the table has the wrong shape |
| `UnsupportedKernelOperation` | `predict(cross)` was called on a `WithPoints` model (pass the coordinates) |

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
```

Because the tables are checked at the API boundary, a bad table is reported instead of producing a silently wrong model. See {doc}`handle-errors` for the other codes.

## Summary

| Symbol | Purpose |
|---|---|
| `DistanceGpr` | trainer for a `DistanceKernel`; `fit`, `factor`, `with_*` configuration |
| `FittedDistanceGpr` | fitted model; `predict`, `predict_into`, `predict_covariance`, `sample`, `loo_predict`, `into_trainer`, `refit` |

Saving a distance model to disk is not part of this release; see {doc}`persist-models`.
