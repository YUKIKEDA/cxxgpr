# How to transform data

Gaussian process hyperparameters are easiest to fit when inputs and targets are on comparable scales. This guide shows how to choose, fit, combine, apply and write transforms with the classes in `cxxgpr/transform/`, and how a model maps its predictions back to your original units. It covers `IdentityInput`, `StandardizeInput`, `MinMaxInput`, their `Fitted*` counterparts, the target transforms, `ColumnwiseInput`, `Pipeline` and `TargetPipeline`, and the two base interfaces you implement for a custom transform.

All code on this page comes from two programs that are built and run with the documentation: `transforms.cpp` and `custom_transform.cpp`. Include `cxxgpr/transform/transform.hpp` to get every class described here.

## Know the two halves of a transform

Every transform exists in two forms, and the split is the same for inputs and for targets.

| Form | Input side | Target side | Holds | Job |
|---|---|---|---|---|
| Unfitted (a recipe) | `UnfittedInputTransform` | `UnfittedTargetTransform` | configuration only | `fit` reads the training data and returns a fitted transform |
| Fitted (a frozen map) | `FittedInputTransform` | `FittedTargetTransform` | statistics from the training data | maps data forward and back |

You hand the model an unfitted transform. The model fits it on the training data and keeps both the recipe and the fitted result: the recipe makes `refit` possible on new data, the fitted map is what `predict` uses. You only touch fitted transforms directly when you want to preprocess data yourself or inspect the learned statistics.

## Pick the right transform

| Transform | Maps | Use it when | Avoid it when |
|---|---|---|---|
| `IdentityInput`, `IdentityTarget` | nothing (still validates) | the data are already well scaled; this is the default | the features differ by orders of magnitude |
| `StandardizeInput`, `StandardizeTarget` | $z = (x - \mu)/\sigma$ | the default choice for almost everything: features on different scales, targets with a large offset | data with outliers that dominate $\mu$ and $\sigma$ |
| `MinMaxInput`, `MinMaxTarget` | affine map onto $[lo, hi]$, default $[0, 1]$ | features have meaningful bounds, or you want the training range to map exactly onto a known interval | new data may fall far outside the training range |
| `ColumnwiseInput` | one transform per feature column | columns need different treatment, for example one is already in $[0,1]$ and another needs a custom map | all columns use the same transform (use the plain transform) |
| `Pipeline`, `TargetPipeline` | several transforms in sequence | you need a nonlinear map followed by a rescaling, such as a logarithm then standardization | a single transform is enough |

Statistics are always computed on the training data only. `StandardizeInput` and `StandardizeTarget` use the *population* standard deviation (divide by $n$), and a constant column gets scale `1.0` instead of $0$. `MinMaxInput` treats a constant column the same way: its span becomes `1.0`, so every value maps to the lower end of the range.

## Use a transform inside a model

Attach transforms to the trainer with `with_input_transform` and `with_target_transform`. Both take an unfitted transform by reference and store a copy.

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [use-in-gpr]
:end-before: // [end]
:dedent: 4
```

The data for this program has a first feature in $[0, 1]$, a second in $[1000, 5000]$ and targets around $50$. The model standardizes the inputs and targets before training, and you pass queries in the original units:

```{code-block} text
prediction at (0.30, 3000): mean = 69.0443, sd = 0.0701, truth = 69.0211
```

### How predictions are mapped back

A fitted model computes everything in transformed space and converts the result before it returns it. The conversion uses the inverse target transform $y = g(z)$:

| Quantity | Conversion | Method on `FittedTargetTransform` |
|---|---|---|
| mean | $g(\mu_z)$ | `inverse_transform_mean` |
| variance | $\sigma_z^2 \cdot \bigl(g'\bigr)^2$ | `inverse_transform_variance` |
| covariance matrix | every entry scaled by $\bigl(g'\bigr)^2$ | `inverse_transform_covariance` |

For `StandardizeTarget`, $g(z) = \sigma z + \mu$, so the variance is multiplied by $\sigma^2$. For `MinMaxTarget` the derivative is the ratio of the data span to the output span. Both are affine, so the mapped distribution is again exactly Gaussian and the covariance conversion is exact. Observation variance (the default) and latent variance are converted in the same way. Input transforms are applied to the query points before the kernel is evaluated; you never invert them for a prediction.

:::{note}
The base-class `inverse_transform_covariance` simply calls `inverse_transform_variance`. That is exact only for affine target transforms. A nonlinear custom target transform must override `inverse_transform_covariance`, because a nonlinear map of a Gaussian is not Gaussian and no single scale factor describes the covariance.
:::

## Fit and apply a transform yourself

Use `fit_typed` when you want the concrete fitted type with its accessors, and `fit` when you want a `std::unique_ptr<FittedInputTransform>` that you can store polymorphically. Every call returns `std::expected` and never throws.

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [standardize-input]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
column means: 0.5000, 2800.0000
column stds:  0.3035, 1107.8666
```

`FittedStandardizeInput` exposes the learned statistics through `mean()` and `std()`, and `FittedMinMaxInput` through `min()`, `max()` and `feature_range()`. For a `MinMaxInput` created with `MinMaxInput::create(lo, hi)`, `feature_range()` also tells you the configured interval; a default-constructed `MinMaxInput` maps onto $[0, 1]$.

### In place or out of place

Every input transform has four entry points. The matrix is column-major with `n_rows` samples and `n_cols` features.

| Method | Effect |
|---|---|
| `apply(x, n_rows, n_cols)` | transforms `x` in place |
| `apply_into(x, n_rows, n_cols, out)` | writes the transformed copy into the preallocated `out`, without allocating |
| `inverse_apply(x, n_rows, n_cols)` | maps transformed data back to the original coordinates in place |
| `inverse_apply_into(x, n_rows, n_cols, out)` | the same, into `out` |

The target transforms mirror this for a vector: `transform` and `transform_into` map forward, `inverse_transform_mean` and `inverse_transform_mean_into` map predictive means back, `inverse_transform_variance` and `inverse_transform_variance_into` map variances back.

The inverse is exact up to floating-point rounding:

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [round-trip]
:end-before: // [end]
:dedent: 4
```

All of these validate their input first. They return `GprErrorCode::EmptyInput` for an empty buffer or a zero extent, `GprErrorCode::SizeOverflow` if `n_rows * n_cols` overflows, `GprErrorCode::LengthMismatch` if the buffer size is not `n_rows * n_cols`, `GprErrorCode::DimensionMismatch` if `n_cols` differs from the fitted width, and `GprErrorCode::NonFiniteInput` if a value is `NaN` or infinite. `IdentityInput` validates too; that makes errors behave the same with and without a transform.

### Build a fitted transform from stored statistics

If you already know the statistics, for example from an earlier run, skip fitting and construct the fitted transform with `FittedStandardizeInput::from_parts(mean, std)`, `FittedMinMaxInput::from_parts(data_min, data_max, range_lo, range_hi)`, `FittedStandardizeTarget::from_parts(mean, std)` or `FittedMinMaxTarget::from_parts(data_min, data_max, range_lo, range_hi)`. These validate their arguments: vectors must be non-empty and of equal length, values finite, scales strictly positive (`GprErrorCode::InvalidHyperparameter`), minima not above maxima, and the range endpoints finite with `range_lo < range_hi` (`GprErrorCode::InvalidConfig`). `FittedIdentityInput` and `FittedIdentityTarget` carry no state and are default-constructible. All built-in transform classes provide `operator==`, which compares configuration or statistics value by value.

## Scale each column differently

`ColumnwiseInput` holds one unfitted transform per feature column and applies them independently. Add them in column order with `then`; `fit` (or `fit_typed`) needs exactly as many maps as the matrix has columns.

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [min-max-and-columnwise]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
columnwise: row 0 -> (0.0000, -0.8160), row 19 -> (1.0000, 0.8160)
```

Column 0 passed through `IdentityInput`; column 1 was mapped onto $[-1, 1]$. `then` accepts either a `const UnfittedInputTransform&` (copied) or a `std::unique_ptr<UnfittedInputTransform>` (moved), and it returns the object itself so calls chain. `size()`, `empty()` and `maps()` report and expose the stored maps. The fitted result `FittedColumnwiseInput` offers the same `size()`, `empty()` and `maps()`, plus the four apply methods; a width mismatch gives `GprErrorCode::DimensionMismatch`, and a null map gives `GprErrorCode::InvalidConfig`. Both classes copy deeply through `clone()`.

## Chain transforms

`Pipeline` applies unfitted input transforms in sequence. While fitting, each step is fitted on the *output* of the previous steps. Applying runs the steps in order; inverting runs them in reverse order.

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [pipeline]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
pipeline steps: 2
```

`TargetPipeline` does the same for targets, including the variance, mean and covariance inverses, which are applied step by step in reverse. Both pipelines accept `then(const T&)` and `then(std::unique_ptr<T>)` and provide `size()`, `empty()` and `steps()`. An empty pipeline behaves exactly like the identity. `FittedPipeline` and `FittedTargetPipeline` are the fitted results; `fit_typed` returns them directly and `fit` boxes them.

## Transform targets

The target transforms work on a plain vector. `fit_typed` returns a fitted transform whose accessors show what it learned:

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [target-transform]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
target mean = 49.0000, std = 15.4347
original-scale mean = 56.7173, variance = 9.5292
```

A model-space mean of $0.5$ becomes $0.5 \sigma + \mu$, and a model-space variance of $0.04$ becomes $0.04\,\sigma^2$. `FittedStandardizeTarget` has `mean()` and `std()`; `FittedMinMaxTarget` has `min()`, `max()` and `feature_range()`; `MinMaxTarget::create(lo, hi)` and `MinMaxTarget::feature_range()` mirror the input side. `StandardizeTarget` and `MinMaxTarget` fit from the training targets only.

## Write a custom transform

Implement one unfitted and one fitted class. The example below is a logarithmic input map, useful when a feature spans several decades and the target varies smoothly in $\log x$. The fitted half remembers the column count and maps $z = \log x$ and back:

```{literalinclude} /../examples/custom_transform.cpp
:language: cpp
:start-after: // [fitted]
:end-before: // [end]
:dedent: 4
```

The unfitted half validates the training matrix and produces the fitted half:

```{literalinclude} /../examples/custom_transform.cpp
:language: cpp
:start-after: // [unfitted]
:end-before: // [end]
:dedent: 4
```

Use it like any built-in transform, here followed by standardization in a `Pipeline`:

```{literalinclude} /../examples/custom_transform.cpp
:language: cpp
:start-after: // [use]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
x = 3: mean = 2.96786, truth = 2.9683, |error| = 0.00043564
x = 30: mean = 1.19237, truth = 1.19253, |error| = 0.000163227
x = 300: mean = 2.53224, truth = 2.53233, |error| = 9.33202e-05
negative query rejected: yes
```

### What you must implement

| Interface | Required | Optional (has a default) |
|---|---|---|
| `FittedInputTransform` | `apply`, `inverse_apply`, `clone` | `apply_into`, `inverse_apply_into` (copy, then call the in-place version), `persist_id`, `persist_state_json` |
| `UnfittedInputTransform` | `fit`, `clone` | `persist_id`, `persist_state_json` |
| `FittedTargetTransform` | `transform`, `inverse_transform_mean`, `inverse_transform_variance`, `clone` | `transform_into`, `inverse_transform_mean_into`, `inverse_transform_variance_into`, `inverse_transform_covariance`, `inverse_transform_covariance_into`, `persist_id`, `persist_state_json` |
| `UnfittedTargetTransform` | `fit`, `clone` | `persist_id`, `persist_state_json` |

Follow these rules so your transform behaves like the built-in ones:

- Validate every call: non-empty input, the expected column count (`GprError::dimension_mismatch`), a tightly packed buffer (`GprError::length_mismatch`), finite values (`GprError::non_finite_input`). The base class cannot do this for you in `apply`. The default `apply_into` and `inverse_apply_into` copy `x` into `out` unless they alias, then call your in-place method.
- Make `inverse_apply` the exact inverse of `apply` on the range the model produces.
- Make `clone` return a deep copy; the library clones transforms when it copies models, pipelines and trainers.
- Return a domain error for inputs you cannot map, as `LogInput` does for non-positive values. The model passes the error back as a value from `fit` or `predict`.
- For a target transform, `inverse_transform_variance` must multiply by the squared derivative of the inverse map. If the map is nonlinear, override `inverse_transform_covariance` as well, as the note above explains.

The default `persist_id()` returns an empty string, which marks the transform as not persistable: `save` then fails with `PersistErrorKind::NotPersistable`. To save models that use your transform, override `persist_id` and `persist_state_json` and register a factory; see {doc}`persist-models`.

## Handle errors

Every fallible call in this module returns `std::expected<..., GprError>`. The codes you will meet most often:

| Code | Cause |
|---|---|
| `GprErrorCode::EmptyInput` | empty buffer, or `n_rows` or `n_cols` is zero |
| `GprErrorCode::LengthMismatch` | `x.size() != n_rows * n_cols`, or `out` has the wrong size |
| `GprErrorCode::DimensionMismatch` | `n_cols` differs from the number of fitted columns |
| `GprErrorCode::NonFiniteInput` | a `NaN` or an infinity in the data |
| `GprErrorCode::InvalidConfig` | bad range for `MinMaxInput::create`, or a null step or map |
| `GprErrorCode::InvalidHyperparameter` | a non-positive scale or `data_min > data_max` in `from_parts` |

See {doc}`handle-errors` for the full list and for how to inspect a `GprError`.

## API summary

### Input transforms (`input.hpp`)

| Symbol | Purpose |
|---|---|
| `FittedInputTransform` | interface of a fitted input map: `apply`, `apply_into`, `inverse_apply`, `inverse_apply_into`, `clone`, `persist_id`, `persist_state_json` |
| `UnfittedInputTransform` | interface of an input recipe: `fit`, `clone`, `persist_id`, `persist_state_json` |
| `IdentityInput`, `FittedIdentityInput` | no change; validates the matrix; `fit_typed`, `fit`, `clone`, `operator==` |
| `StandardizeInput` | per-column mean and population std; `fit_typed`, `fit`, `clone`, `operator==` |
| `FittedStandardizeInput` | `from_parts`, `mean`, `std`, the four apply methods, `clone`, `operator==` |
| `MinMaxInput` | affine map onto $[lo, hi]$; `create`, `feature_range`, `fit_typed`, `fit`, `clone`, `operator==` |
| `FittedMinMaxInput` | `from_parts`, `feature_range`, `min`, `max`, the four apply methods, `clone`, `operator==` |

### Target transforms (`target.hpp`)

| Symbol | Purpose |
|---|---|
| `FittedTargetTransform` | `transform`, `transform_into`, `inverse_transform_mean`, `inverse_transform_mean_into`, `inverse_transform_variance`, `inverse_transform_variance_into`, `inverse_transform_covariance`, `inverse_transform_covariance_into`, `clone`, `persist_id`, `persist_state_json` |
| `UnfittedTargetTransform` | `fit`, `clone`, `persist_id`, `persist_state_json` |
| `IdentityTarget`, `FittedIdentityTarget` | no change; `fit_typed`, `fit`, `clone`, `operator==` |
| `StandardizeTarget` | mean and population std of the targets |
| `FittedStandardizeTarget` | `from_parts`, `mean`, `std` |
| `MinMaxTarget` | `create`, `feature_range`, `fit_typed`, `fit` |
| `FittedMinMaxTarget` | `from_parts`, `feature_range`, `min`, `max` |

### Composition (`columnwise.hpp`, `pipeline.hpp`)

| Symbol | Purpose |
|---|---|
| `ColumnwiseInput`, `FittedColumnwiseInput` | one map per feature column; `then`, `size`, `empty`, `maps`, `fit_typed`, `fit`, `clone` |
| `Pipeline`, `FittedPipeline` | input maps in sequence; `then`, `size`, `empty`, `steps`, `fit_typed`, `fit`, `clone` |
| `TargetPipeline`, `FittedTargetPipeline` | target maps in sequence; same members, with the target inverse methods |

The generated reference for these headers is in [the transform API reference](../reference/api/transform.md).
