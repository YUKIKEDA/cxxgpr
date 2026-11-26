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

Bind one table per slot to a `DistanceSource` (or pre-validate with `prepare_square` into a `CheckedDistanceStore`), then call `fit`. Tables passed through `DistanceSource` are validated once (`finite`, non-negative, symmetric and zero on the diagonal) unless you call `DistanceSource::tidy` to allow a documented repair, or `DistanceSource::assume_valid(true)` when tables are already trusted. Both `MatrixRef<double>` and `MatrixRef<float>` tables, `std::span<const double>` and `std::span<const float>` targets, and (for `WithPoints`) `MatrixRef<float>` coordinates are accepted.

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
```

`with_noise_variance` validates its argument and returns `std::expected`, exactly as for `Gpr`, while `with_likelihood(GaussianLikelihood)` installs custom open-interval bounds on the observation noise. The other configuration methods are `with_jitter`, `with_kernel_exp`, `with_retain_distances`, `with_target_transform`, `with_optimizer` and, for `WithPoints` kernels only, `with_input_transform`. The getters are `kernel`, `likelihood`, `jitter`, `optimizer` and `retains_distances`.

When training models that will **not** be saved to disk, calling `with_retain_distances(false)` lets `fit` and `factor` borrow compatible caller tables without copying or retaining an internal training store (and drops retained copies from fitted models). Persistence refuses non-retained models (`retains_distances() == false`) with `PersistFailed(PersistErrorKind::NotPersistable)`.

`fit` optimizes the kernel hyperparameters and the noise variance by maximizing the marginal likelihood. To factorize at the current hyperparameters without optimizing, call `factor`. For a `WithPoints` kernel both take the coordinates as an extra argument: `fit(train, x, y)` and `factor(train, x, y)`. Both `DistanceSource` and a pre-validated `CheckedDistanceStore<Storage>` (matching the model's `DistanceInputConcept`) are accepted wherever training, cross or query tables are taken (`fit`, `factor`, `refit`, `predict`, `predict_into`, `predict_covariance`, `sample`).

:::{note}
The default precision policy is double precision. `DistanceGpr<Lbfgs, MixedPrecision<ReevaluateKernel>>` and the other policies of {doc}`optimize-performance` work the same way and report `refinement_summary`.
:::

## Step 3: Predict

Build one table of squared distances from every training point to every query point (n × m, with the training points along the rows) and bind it to the same slot (or pass a `CheckedDistanceStore<Storage>` built by `prepare_cross`). A kernel with several slots needs a table for each.

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
```

`predict` returns the mean and the marginal variance on the original target scale. `predict_into` writes into a `Prediction` you own and reuses the model's query workspace: after one call with m queries, later calls with the same slots and at most m queries allocate no memory (including when passing `MatrixRef<float>` query coordinates or a `CheckedDistanceStore`). For a `WithPoints` model, pass the query coordinates after the table: `predict(cross, xq)`.

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
| `DistanceSlot` | a slot is missing, bound twice, or bound with the other kind of table (scalar instead of ARD, or the reverse) |
| `LengthMismatch` | a training table is not square, the tables of one source differ in shape, or the table order does not match the targets, the training set or the query coordinates |
| `DimensionMismatch` | an ARD slot is bound with a different number of per-dimension tables than it declares |
| `UnsupportedKernelOperation` | a source binds a slot of another kernel, such as a slot object that belongs to a different model |

All shape errors are reported before any table value is read, so a source with both a wrong shape and a bad entry reports the shape error.

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
```

Because the tables are checked at the API boundary, a bad table is reported instead of producing a silently wrong model. See {doc}`handle-errors` for the other codes.

## Summary

| Symbol | Purpose |
|---|---|
| `DistanceInputConcept` | concept accepting `DistanceSource` or `CheckedDistanceStore<Storage>` |
| `DistanceGpr` | trainer for a `DistanceKernel`; `fit`, `factor`, `with_*` configuration (`with_likelihood`, `with_retain_distances`) |
| `FittedDistanceGpr` | fitted model; `predict`, `predict_into`, `predict_covariance`, `sample`, `loo_predict`, `retains_distances`, `into_trainer`, `refit` |

To save a distance model to disk and load it back, see {doc}`persist-models`.
