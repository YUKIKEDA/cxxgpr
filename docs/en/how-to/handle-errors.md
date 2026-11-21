# How to handle errors

This guide shows how to detect, inspect and react to failures. cxxgpr never throws across its API: every operation that can fail returns `std::expected<T, GprError>` (or `std::expected<void, GprError>`), and a failed operation leaves its inputs and the object it was called on unchanged. For a terse list of codes, see [error codes](../reference/error-codes.md).

The snippets come from one complete program, listed at the end of the page. They assume `using namespace cxxgpr;` and `#include <cxxgpr/core/error.hpp>`.

## Check every result

Test the returned `std::expected` before using the value. On success `*result` or `result.value()` is the value; on failure `result.error()` is a `GprError`.

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [check]
:end-before: // [end]
```

Most functions are marked `[[nodiscard]]`, so ignoring a result is a compiler warning.

:::{warning}
`result.value()` on a failed result **throws** `std::bad_expected_access<GprError>`. The library itself never throws, but this standard-library call does. Use it only in code that deliberately turns failures into exceptions, or where failure is impossible. Calling `result.error()` on a *successful* result is undefined behavior.
:::

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [value-throws]
:end-before: // [end]
```

```{code-block} text
value() threw; carried code InvalidInterval
```

## Read the code and the message

`GprError::code()` returns a `GprErrorCode`. Branch on the code, never on the message text. `GprError::message()` returns a `std::string_view` for logs and users.

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [report]
:end-before: // [end]
```

:::{warning}
`message()` is a view into the error object. It dangles once the `GprError` is destroyed. Keep the error alive while you use the view, or copy it into a `std::string` first. `trainer.fit(...).error().message()` on a temporary is a dangling view.
:::

The library does not provide a function that turns a `GprErrorCode` into a string. The example defines `code_name` with a `switch` that lists every enumerator and has no `default`, so the compiler warns you when a new code is added:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [name]
:end-before: // [end]
```

## Trigger and read common input errors

These failures all come from violated preconditions on the caller's side.

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [bad-inputs]
:end-before: // [end]
```

```{code-block} text
empty fit -> EmptyInput: input is empty
short targets -> LengthMismatch: length mismatch: target length must match training row count
infinite target -> NonFiniteInput: input contains a non-finite value (NaN/Inf)
2-column query -> DimensionMismatch: input dimension mismatch: X.ncols()=2, expected 1
```

Some errors carry structured data in addition to the message. Each accessor returns a `std::optional` that is empty unless the error is of the matching kind:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [accessors]
:end-before: // [end]
```

```{code-block} text
got 2 columns, expected 1
```

Validators for configuration values return their own errors:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [validators]
:end-before: // [end]
```

```{code-block} text
noise 0 -> InvalidNoiseVariance: invalid observation noise variance: noise variance must be positive
negative jitter -> InvalidConfig: invalid configuration: jitter must be finite and non-negative, got -1
negative lengthscale -> InvalidInterval: value -1 is not strictly inside (1e-05, 1e+05)
reversed interval -> InvalidInterval: interval bounds must be finite and satisfy lo < hi (got lo=2, hi=1)
value outside interval -> InvalidInterval: value 5 is not strictly inside (0, 1)
```

Note that a non-positive kernel lengthscale is reported as `GprErrorCode::InvalidInterval` (it is outside the default positive interval), not as `GprErrorCode::InvalidHyperparameter`. For `InvalidInterval`, `interval_error_kind()` tells the two cases apart: `IntervalErrorKind::InvalidBounds` (bad endpoints) or `IntervalErrorKind::OutOfRange` (value outside; read it with `interval_value()`, `interval_lo()`, `interval_hi()`).

## Decide how to react

Group the codes by what the caller can do about them:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [classify]
:end-before: // [end]
```

| Reaction | Codes | What to do |
|---|---|---|
| Fix the call | `EmptyInput`, `SizeOverflow`, `LengthMismatch`, `DimensionMismatch`, `IndexOutOfRange`, `InsufficientData`, `NonFiniteInput`, `InvalidHyperparameter`, `InvalidNoiseVariance`, `InvalidInterval`, `InvalidConfig`, `InvalidDistance`, `DistanceSlot`, `UnsupportedKernelOperation`, `WorkspaceTooSmall` | A precondition is violated. Retrying the same call fails again; correct the data or configuration. |
| Adjust numerics and retry | `CholeskyFailed`, `NonPositiveDefiniteMatrix`, `NonFiniteKernelValue`, `OptimizationNotConverged` | The input is valid but the arithmetic broke down. Change jitter, noise, kernel hyperparameters, precision or optimizer. |
| Report to the user | `PersistFailed`, `UnsupportedPersistVersion` | The file system or the saved model is at fault. |

A full table with the circumstances of each code follows. The one-line version is in [error codes](../reference/error-codes.md).

### Code by code

**`EmptyInput`**: an input matrix, vector or span has zero extent or a null pointer. It comes from `fit`, `factor`, `predict*`, `sample*` and most low-level routines. Check that the data were loaded.

**`SizeOverflow`**: a size such as $n^2$ or $p\,n^2$ overflows `std::size_t`. It signals an absurdly large problem or corrupt dimensions.

**`LengthMismatch`**: a vector or span has the wrong length; for example `y.size() != x.rows()` in `fit`, or a parameter span of the wrong length in `get_params` / `set_params`.

**`DimensionMismatch`**: the number of feature columns differs from what the model or an ARD kernel expects. `x_dim()` is what you supplied, `expected_dim()` what is needed.

**`IndexOutOfRange`**: a parameter index, distance slot or leaf index is out of range, for example `GaussianLikelihood::noise_grad_diag` with an index other than 0.

**`InsufficientData`**: fewer points than a validation helper requires; `n_points()` and `min_points()` give the numbers. `Gpr::fit` accepts any non-empty data set and does not return this code, but the validation functions in `cxxgpr/core/checked.hpp` do.

**`NonFiniteInput`**: NaN or infinity in training inputs, targets, query points or parameters. Clean the data first.

**`NonFiniteKernelValue`**: a kernel evaluation, the marginal likelihood, a latent predictive variance or a back-transformed value is NaN or infinite. Reduce extreme hyperparameters or tighten their bounds.

**`InvalidHyperparameter`**: a fitted transform is built from parts (`from_parts`, or while loading a saved model) with an invalid parameter, such as a standardization scale that is not positive or a minimum above the maximum. Ordinary `fit` does not produce it; constant columns and targets are handled by the transforms.

**`InvalidNoiseVariance`**: the noise variance is not finite or not strictly positive, or the log-noise is not finite or overflows `exp`.

**`InvalidInterval`**: bounds are not finite or not ordered, or a value lies outside its open interval (see above).

**`InvalidConfig`**: a configuration object is invalid: a negative jitter, a retry count of zero, a negative optimizer tolerance, an inconsistent transform pipeline and similar. The message names the field.

**`InvalidDistance`**: a supplied squared-distance matrix is not symmetric, has a non-zero diagonal, a negative entry or the wrong shape. `distance_slot()`, `distance_dim()`, `distance_pair()` and `distance_reason()` locate the problem. See {doc}`use-supplied-distances`.

**`DistanceSlot`**: the supplied distance sources do not match the kernel's slots; `slot_error_kind()` is `SlotErrorKind::NotRead` (source for a slot the kernel never reads), `Duplicate` or `Missing`. `distance_slot()` gives the slot.

**`UnsupportedKernelOperation`**: the kernel cannot do what was asked, such as evaluating coordinates for a distance-only kernel, or a composite kernel with no terms.

**`CholeskyFailed`**: factorization failed even after the whole jitter schedule. The next section explains how to recover. `jitter()` is the last offset tried, `matrix_size()` the order, `cholesky_stage()` is `CholeskyStage::Fit` (training) or `CholeskyStage::Predict` (prediction and sampling).

**`NonPositiveDefiniteMatrix`**: a strict definiteness check failed without any jitter retry (low-level factorization and leave-one-out evaluation of a degenerate factor).

**`OptimizationNotConverged`**: an optimizer gave up. The built-in optimizers do **not** return it: they report non-convergence through `optimizer_result().converged` and still deliver the best parameters found. A custom optimizer may return it with `iterations()` set (see below).

**`WorkspaceTooSmall`**: a caller-supplied scratch buffer is smaller than required (low-level API only).

**`PersistFailed`**: saving or loading failed; `persist_error_kind()` gives the `PersistErrorKind`: `Io`, `Config`, `Tensor`, `InvalidPersistId`, `NotPersistable`, `UnregisteredId` or `WrongModel`. See {doc}`persist-models`.

**`UnsupportedPersistVersion`**: the saved format version is not supported; `found_version()` and `supported_version()` give both.

## Recover from `CholeskyFailed`

The Gram matrix $K + \sigma_n^2 I$ should be positive definite, but floating-point rounding can break that, in particular with `SinglePrecision`, very large kernel amplitudes or near-duplicate points. The error records what happened:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [cholesky]
:end-before: // [end]
```

```{code-block} text
CholeskyFailed: last jitter 0, matrix order 300, stage Fit
adaptive jitter recovered, accepted jitter 0.01
```

Try these remedies in order:

1. Use an adaptive jitter policy (`JitterPolicy::adaptive`); `FittedGpr::factor_jitter()` reports the offset that was needed.
2. Increase the noise variance or its lower bound, if the data justify it.
3. Switch to a higher precision (`MixedPrecision` or `DoublePrecision`).
4. Remove duplicate points, shorten very long lengthscales or reduce a huge amplitude.

The default trainer has jitter `0`, so a trainer that must be robust should set a policy explicitly. See {doc}`configure-fit`.

## Chain operations without branching

`std::expected` has monadic members. Use them to write a pipeline that stops at the first failure and carries the original `GprError` to the end:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [and-then]
:end-before: // [end]
```

```{code-block} text
chained mean at 0.5: 0.141139, failing chain stops at: InvalidNoiseVariance
```

| Member | Use |
|---|---|
| `and_then(f)` | run `f` on the value; `f` itself returns an `expected`, so errors propagate |
| `transform(f)` | map the value with a function that cannot fail |
| `or_else(f)` | run `f` on the error to recover or replace it |
| `value_or(x)` | the value, or `x` on error (the error is discarded) |

`or_else` turns a failure into a fallback:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [or-else]
:end-before: // [end]
```

## Return your own errors

Custom kernels, optimizers and transforms return `GprError` through the same static factories the library uses; there is one per code, such as `GprError::invalid_config(reason)`, `GprError::optimization_not_converged(iterations)` and `GprError::cholesky_failed(jitter, matrix_size, stage)`. The library propagates your error unchanged and leaves the trainer as it was:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [give-up]
:end-before: // [end]
```

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [custom-error]
:end-before: // [end]
```

```{code-block} text
custom optimizer error passes through: OptimizationNotConverged after 3 iterations
```

`GprError` is a value type. It is copyable, and `operator==` compares the code, the message and every payload. `with_slot_if_unset` and `with_dim_if_unset` return a copy of an `InvalidDistance` error with the slot or dimension filled in, which lets an outer layer add context the inner layer did not know:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [factories]
:end-before: // [end]
```

```{code-block} text
invalid_distance: invalid squared distance in slot 2, dimension 0 at (1, 2): negative entry | slot 2, dim 0, pair (1, 2), reason "negative entry"
distance_slot: distance slot mismatch in slot 1: a kernel slot has no supplied distance source
persist_failed: persist failed (config): bad json
cholesky_failed: Cholesky factorization failed (stage=Predict, size=10, jitter=1e-06 already applied)
```

The program prints the message produced by every factory; the full list is in [error codes](../reference/error-codes.md).

## Loading from disk

A missing model directory is an ordinary error:

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [persist]
:end-before: // [end]
```

```{code-block} text
load -> PersistFailed, kind Io
```

## Complete program

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:linenos:
```

## API summary

| Symbol | Purpose |
|---|---|
| `GprError`, `code`, `message` | The error value; category and text. |
| `GprErrorCode` | The 21 error categories. |
| `CholeskyStage` (`Fit`, `Predict`), `SlotErrorKind` (`NotRead`, `Duplicate`, `Missing`), `PersistErrorKind`, `IntervalErrorKind` (`InvalidBounds`, `OutOfRange`) | Sub-categories of some codes. |
| `x_dim`, `expected_dim`, `n_points`, `min_points`, `jitter`, `matrix_size`, `cholesky_stage`, `iterations` | Payload accessors for dimension, data, Cholesky and optimizer errors. |
| `interval_error_kind`, `interval_value`, `interval_lo`, `interval_hi` | Payload of `InvalidInterval`. |
| `distance_slot`, `distance_dim`, `distance_pair`, `distance_reason`, `slot_error_kind` | Payload of distance errors. |
| `persist_error_kind`, `found_version`, `supported_version` | Payload of persistence errors. |
| `with_slot_if_unset`, `with_dim_if_unset`, `operator==` | Add context; compare. |
| `empty_input`, `size_overflow`, `length_mismatch`, `dimension_mismatch`, `index_out_of_range`, `insufficient_data`, `non_finite_input`, `non_finite_kernel_value`, `invalid_hyperparameter`, `invalid_noise_variance`, `invalid_interval_bounds`, `invalid_interval_out_of_range`, `invalid_config`, `invalid_distance`, `distance_slot`, `unsupported_kernel_operation`, `cholesky_failed`, `non_positive_definite_matrix`, `optimization_not_converged`, `workspace_too_small`, `persist_failed`, `unsupported_persist_version` | Static factories, one per code (two for `InvalidInterval`). |

Generated reference: [`core`](../reference/api/core.md).
