# How to predict, sample and cross-validate

This guide shows how to query a trained `FittedGpr`: pointwise means and variances, joint covariances, posterior samples and closed-form leave-one-out predictions. It also introduces the deterministic random number generator used for sampling. All results are on the **original target scale**: input and target transforms are undone for you.

The snippets come from one complete program, listed at the end of the page. They assume `using namespace cxxgpr;` and `fitted` is a `FittedGpr<optimizer::Lbfgs>` fitted to 25 noisy samples of $\sin x$ on $[0, 6]$ (see {doc}`configure-fit` for how to train one).

## Predict means and variances

`predict` returns a `Prediction<Refine>`, a struct with the posterior `mean`, the `variance` (one entry per query row) and the `variance_kind`. Pass queries either as raw column-major data plus the number of rows and columns, or as a `MatrixRef<double>`.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
```

```{code-block} text
x = 0.5000  mean = 0.4481  sd = 0.1006  sin(x) = 0.4794
x = 1.5000  mean = 1.0660  sd = 0.0993  sin(x) = 0.9975
x = 3.0000  mean = 0.1754  sd = 0.0991  sin(x) = 0.1411
x = 5.0000  mean = -1.0728  sd = 0.0999  sin(x) = -0.9589
```

The means follow $\sin x$, and the true value lies within about 1.2 standard deviations of each prediction. The reported deviation (about 0.1) is dominated by the fitted observation noise: the default variance is the variance of a new **observation**.

`xs` must have exactly `n_features()` columns and only finite values. Otherwise the call fails with `GprErrorCode::DimensionMismatch`, `GprErrorCode::NonFiniteInput` or `GprErrorCode::EmptyInput`.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
```

```{code-block} text
two-column query: input dimension mismatch: X.ncols()=2, expected 1 (x_dim 2, expected 1)
```

The predictive variance is never negative. Internally the latent part is clamped at zero before the noise variance is added, so rounding cannot produce a negative value.

### Latent or observation variance

`PredictOptions` has one field, `variance_kind`, of type `VarianceKind`:

| Value | Meaning | Use for |
|---|---|---|
| `VarianceKind::Observation` (default) | Variance of a new noisy measurement $y_*$: latent variance **plus** $\sigma_n^2$. | Prediction intervals for data you expect to measure. |
| `VarianceKind::Latent` | Variance of the noise-free function $f_*$. | Uncertainty of the underlying function; posterior samples of the function. |

The `variance_kind` of the returned `Prediction` records which one you got.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [variance-kind]
:end-before: // [end]
```

```{code-block} text
observation - latent variance = 0.0080 (noise variance 0.0080)
```

With identity target transforms the two differ by exactly the noise variance. If you use a target transform that rescales, the difference is the noise variance multiplied by the squared scale.

`PredictOptions` compares with `==` and is an aggregate, so designated initializers work: `PredictOptions{.variance_kind = VarianceKind::Latent}`.

## Predict without allocating

`predict` allocates its result. In a loop, reuse one `Prediction` with `predict_into`. After the first call has sized the model's internal scratch space and the output vectors, repeated calls with the same or a smaller number of query rows perform no heap allocation.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [predict-into]
:end-before: // [end]
```

The four mean/variance entry points are siblings:

| Function | Result | Allocates | Options |
|---|---|---|---|
| `predict(xs, options = {})` | returns `Prediction` | yes | optional |
| `predict_with(xs, options)` | returns `Prediction` | yes | required |
| `predict_into(xs, out, options = {})` | fills `out` | no (after warm-up) | optional |
| `predict_with_into(xs, options, out)` | fills `out` | no (after warm-up) | required |

`predict` and `predict_into` forward to `predict_with` and `predict_with_into`; they exist so the common call needs no `PredictOptions`. Every function has an overload taking a raw `std::span<const double>` with `n_rows` and `n_cols`.

:::{warning}
The prediction functions are `const`, but they write to a scratch workspace stored inside the `FittedGpr`. Do **not** call them concurrently on the same object from several threads. Give each thread its own copy: copying a `FittedGpr` creates an independent workspace.
:::

## Predict a joint covariance

`predict_covariance` returns a `PredictiveCovariance<Refine>`: the `mean`, the full $m \times m$ `covariance` matrix over the query points in column-major order, the number of points `n_points` and the `variance_kind`. Element $(i, j)$ is `covariance[j * n_points + i]`. For `VarianceKind::Observation`, $\sigma_n^2$ is added to the diagonal.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [covariance]
:end-before: // [end]
```

```{code-block} text
cov(0, 1) = -0.0000, var(0) = 0.0022
```

The diagonal equals the pointwise latent variance and the matrix is symmetric (the example checks both). The data pin the function down at both query points, so their posterior errors are almost uncorrelated and the off-diagonal entry is close to zero.

`predict_covariance_with(xs, options)` is the explicit-options form, and both have raw-span overloads. They allocate the result: a covariance for $m$ points needs $O(m^2)$ memory, so keep $m$ moderate. The preallocated variant `predict_covariance_into` is part of the low-level workspace API ({doc}`use-low-level-building-blocks`).

## Draw posterior samples

`sample` draws functions from the joint posterior at the query points. It returns a `std::vector<Refine>` holding an $m \times n_\text{draws}$ column-major matrix: draw `j` occupies indices `j * m` to `j * m + m - 1`.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [sample]
:end-before: // [end]
```

```{code-block} text
sample mean 0.4484 (posterior 0.4481), sample variance 0.0021 (posterior 0.0022)
```

With 4000 draws the empirical mean and variance reproduce the posterior moments. The covariance used follows `options.variance_kind`. Use `VarianceKind::Latent` to draw the noise-free function; the default `VarianceKind::Observation` also adds the observation noise to every sampled value.

### Seeds and reproducibility

The `seed` argument initializes a `Xoshiro256PlusPlus` generator inside the call. The same model, queries, seed and options always give bit-identical samples; a different seed gives different samples.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [seed]
:end-before: // [end]
```

```{code-block} text
same seed identical: 1, different seed identical: 0
```

### Sampling jitter

To sample, the library factors the query covariance. This matrix is only positive semi-definite and becomes numerically singular for nearby query points, so the factorization is stabilized by jitter. If you pass nothing, an adaptive policy is used (first offset `1e-12`, factor 10, up to 10 retries, cap `1e-2`). Override it with the last argument:

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [jitter]
:end-before: // [end]
```

If the factorization still fails, `sample` returns `GprErrorCode::CholeskyFailed`. Passing `n_draws = 0` returns an empty vector. `sample_with(xs, n_draws, seed, options, jitter)` is the explicit-options form; `sample` has the raw-span overload as well. See {doc}`configure-fit` for `JitterPolicy`.

## Leave-one-out predictions

`loo_predict` returns, for every training point, the prediction of a model trained on all the *other* points. It uses the closed-form identity based on the diagonal of $A^{-1}$ with $A = K + \sigma_n^2 I$ (Rasmussen and Williams, section 5.4.2), so it needs no refitting and takes essentially the cost of one inverse diagonal.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [loo]
:end-before: // [end]
```

```{code-block} text
LOO for training point 10: mean 0.6425, sd 0.1018, observed 0.5218
explicit refit without point 10: mean 0.6425, sd 0.1018
```

The second line comes from the program's cross-check: it removes point 10, factors a model at the same hyperparameters and predicts that point. The two agree to better than $10^{-6}$. Note that LOO keeps the **hyperparameters** fixed; it does not re-optimize them without the held-out point.

`loo_predict(options = {})` and `loo_predict_with(options)` return a `Prediction` aligned with the training rows. The default variance is that of the noisy observation; use `VarianceKind::Latent` to subtract the noise. They fail only with `GprErrorCode::NonFiniteKernelValue` if inverse target scaling produces a non-finite value.

## Generate random numbers yourself

`Xoshiro256PlusPlus` (alias `SeededRng`) is the same deterministic generator the library uses for sampling and optimizer restarts. It is seeded through SplitMix64, is portable across platforms, and is handy for reproducible synthetic data. It is `constexpr`-constructible and compares with `==`.

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [rng]
:end-before: // [end]
```

```{code-block} text
u = 0.8143, normal = 1.5043, pick = 7, open = 0.7935, word = 10848501901068131965, fresh generator equals advanced one: 0
```

| Method | Returns |
|---|---|
| `next_u64()` | next raw 64-bit word |
| `unit()` | uniform double in $[0, 1)$ |
| `open_unit()` | uniform double strictly inside $(0, 1)$ (safe for `std::log`) |
| `up_to(max)` | uniform integer in $[0, \text{max}]$ (inclusive; unbiased) |
| `unit_normal()` | standard normal via Box-Muller |

:::{note}
The generator is not thread-safe. Use one instance per thread, with different seeds.
:::

## Complete program

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:linenos:
```

## API summary

| Symbol | Purpose |
|---|---|
| `FittedGpr::predict`, `predict_with` | Posterior mean and variance; allocate the result. |
| `predict_into`, `predict_with_into` | Same, into a caller-owned `Prediction`; no allocation after warm-up. |
| `predict_covariance`, `predict_covariance_with` | Mean and full covariance as `PredictiveCovariance`. |
| `sample`, `sample_with` | Deterministic posterior draws; $m \times n_\text{draws}$ column-major. |
| `loo_predict`, `loo_predict_with` | Closed-form leave-one-out mean and variance. |
| `Prediction` (`mean`, `variance`, `variance_kind`) | Pointwise result. |
| `PredictiveCovariance` (`mean`, `covariance`, `n_points`, `variance_kind`) | Joint result. |
| `PredictOptions` (`variance_kind`) | Per-call options. |
| `VarianceKind` (`Latent`, `Observation`) | Which variance is reported. |
| `Xoshiro256PlusPlus`, `SeededRng` | Deterministic generator; `next_u64`, `unit`, `open_unit`, `up_to`, `unit_normal`. |
| `MatrixRef` | Non-owning column-major query view. |

Generated reference: [`gpr`](../reference/api/gpr.md), [`core`](../reference/api/core.md).
