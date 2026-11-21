# Error codes

Every fallible cxxgpr function returns `std::expected<T, GprError>`. This page lists each `GprErrorCode`, its sub-categories and the `GprError` interface. Header: `cxxgpr/core/error.hpp`. For usage, see [How to handle errors](../how-to/handle-errors.md).

## GprErrorCode

The enumeration has underlying type `std::uint8_t`. Values are assigned in the order shown (`EmptyInput` is 0). Do not persist the numeric values.

| Code | Raised when | Payload accessors | Factory |
|---|---|---|---|
| `EmptyInput` | An input matrix, vector or span is empty, has a zero extent or a null pointer. | none | `empty_input()` |
| `SizeOverflow` | A dimension or packed-buffer size overflows `std::size_t`. | none | `size_overflow()` |
| `LengthMismatch` | A vector or span length differs from the required length. | none | `length_mismatch(reason)` |
| `DimensionMismatch` | The feature column count differs from the model's or an ARD kernel's. | `x_dim()`, `expected_dim()` | `dimension_mismatch(x_dim, expected_dim)` |
| `IndexOutOfRange` | A parameter, distance slot, leaf or element index is out of range. | none | `index_out_of_range(reason)` |
| `InsufficientData` | Fewer points than an operation or validation helper requires. | `n_points()`, `min_points()` | `insufficient_data(n_points, min_points)` |
| `NonFiniteInput` | An input coordinate, target or scalar is NaN or infinite. | none | `non_finite_input()` |
| `NonFiniteKernelValue` | A kernel matrix entry, the marginal likelihood, a latent predictive variance or a back-transformed value is NaN or infinite. | none | `non_finite_kernel_value()` |
| `InvalidHyperparameter` | A component parameter failed validation (for example a non-positive transform scale). | none | `invalid_hyperparameter(reason)` |
| `InvalidNoiseVariance` | The noise variance or log noise variance is invalid (non-finite, not positive, overflow). | none | `invalid_noise_variance(reason)` |
| `InvalidInterval` | Interval bounds are invalid, or a value lies outside its open interval. | `interval_error_kind()`, `interval_value()`, `interval_lo()`, `interval_hi()` | `invalid_interval_bounds(lo, hi)`, `invalid_interval_out_of_range(value, lo, hi)` |
| `InvalidConfig` | A model, optimizer, jitter, transform or kernel configuration is invalid. | none | `invalid_config(reason)` |
| `InvalidDistance` | A supplied squared-distance matrix violates symmetry, diagonal, non-negativity or shape rules. | `distance_slot()`, `distance_dim()`, `distance_pair()`, `distance_reason()` | `invalid_distance(slot, dim, pair, reason)` |
| `DistanceSlot` | Supplied distance sources do not match the kernel's distance slots. | `slot_error_kind()`, `distance_slot()` | `distance_slot(kind, slot)` |
| `UnsupportedKernelOperation` | The kernel does not support the requested operation. | none | `unsupported_kernel_operation(reason)` |
| `CholeskyFailed` | Cholesky factorization failed after the whole jitter schedule. | `jitter()`, `matrix_size()`, `cholesky_stage()` | `cholesky_failed(jitter, matrix_size, stage)` |
| `NonPositiveDefiniteMatrix` | A strict positive-definiteness check failed without jitter recovery. | none | `non_positive_definite_matrix()` |
| `OptimizationNotConverged` | An optimizer reported that it did not converge. Built-in optimizers do not return it; they report `OptimizerResult::converged`. | `iterations()` | `optimization_not_converged(iterations)` |
| `WorkspaceTooSmall` | A caller-provided workspace or scratch buffer is too small. | none | `workspace_too_small()` |
| `PersistFailed` | Saving or loading a model failed (I/O, JSON, tensors, registry). | `persist_error_kind()` | `persist_failed(kind, reason)` |
| `UnsupportedPersistVersion` | The persisted format version is not supported by this build. | `found_version()`, `supported_version()` | `unsupported_persist_version(found, supported)` |

## Sub-categories

| Enumeration | Enumerators | Used with |
|---|---|---|
| `CholeskyStage` | `Fit`, `Predict` | `CholeskyFailed`: training versus prediction or sampling. |
| `SlotErrorKind` | `NotRead`, `Duplicate`, `Missing` | `DistanceSlot`: slot not read by the kernel, supplied twice, or required but absent. |
| `PersistErrorKind` | `Io`, `Config`, `Tensor`, `InvalidPersistId`, `NotPersistable`, `UnregisteredId`, `WrongModel` | `PersistFailed`. |
| `IntervalErrorKind` | `InvalidBounds`, `OutOfRange` | `InvalidInterval`. |

All four have underlying type `std::uint8_t`.

## GprError

`GprError` is a copyable value type. Its constructors are private; create instances through the static factories above.

| Member | Returns | Notes |
|---|---|---|
| `code()` | `GprErrorCode` | `noexcept`. |
| `message()` | `std::string_view` | View into the error object; valid while the error lives. `noexcept`. |
| `x_dim()`, `expected_dim()` | `std::optional<std::size_t>` | Set by `dimension_mismatch`. |
| `n_points()`, `min_points()` | `std::optional<std::size_t>` | Set by `insufficient_data`. |
| `jitter()` | `std::optional<double>` | Last jitter attempted; set by `cholesky_failed`. |
| `matrix_size()` | `std::optional<std::size_t>` | Order of the failed matrix. |
| `cholesky_stage()` | `std::optional<CholeskyStage>` | |
| `iterations()` | `std::optional<std::size_t>` | Set by `optimization_not_converged`. |
| `interval_error_kind()` | `std::optional<IntervalErrorKind>` | |
| `interval_value()` | `std::optional<double>` | Only for `IntervalErrorKind::OutOfRange`. |
| `interval_lo()`, `interval_hi()` | `std::optional<double>` | |
| `distance_slot()` | `std::optional<std::size_t>` | `InvalidDistance` or `DistanceSlot`, when known. |
| `distance_dim()` | `std::optional<std::size_t>` | ARD dimension, when known. |
| `distance_pair()` | `std::optional<std::pair<std::size_t, std::size_t>>` | Offending matrix entry (row, column). |
| `distance_reason()` | `std::optional<std::string_view>` | Empty unless `code() == InvalidDistance`. |
| `slot_error_kind()` | `std::optional<SlotErrorKind>` | |
| `persist_error_kind()` | `std::optional<PersistErrorKind>` | |
| `found_version()`, `supported_version()` | `std::optional<std::uint32_t>` | |
| `with_slot_if_unset(slot)` | `GprError` | Copy with the slot set if it was unset; message rebuilt. |
| `with_dim_if_unset(dim)` | `GprError` | Same for the ARD dimension. |
| `operator==` | `bool` | Compares code, message and all payload fields. |

Accessors return `std::nullopt` when the error was not constructed by the matching factory.

## Messages

The text of `message()` is for people and may change between releases. Current texts, produced by the factories:

| Factory | Message |
|---|---|
| `empty_input()` | `input is empty` |
| `size_overflow()` | `size overflows std::size_t` |
| `length_mismatch("expected 3 values")` | `length mismatch: expected 3 values` |
| `dimension_mismatch(2, 3)` | `input dimension mismatch: X.ncols()=2, expected 3` |
| `index_out_of_range("leaf 4 of 3")` | `index out of range: leaf 4 of 3` |
| `insufficient_data(2, 5)` | `insufficient data: n=2, at least 5 points required` |
| `non_finite_input()` | `input contains a non-finite value (NaN/Inf)` |
| `non_finite_kernel_value()` | `kernel evaluation produced a non-finite value` |
| `invalid_hyperparameter("...")` | `invalid hyperparameter: ...` |
| `invalid_noise_variance("...")` | `invalid observation noise variance: ...` |
| `invalid_interval_bounds(2, 1)` | `interval bounds must be finite and satisfy lo < hi (got lo=2, hi=1)` |
| `invalid_interval_out_of_range(5, 0, 1)` | `value 5 is not strictly inside (0, 1)` |
| `invalid_config("...")` | `invalid configuration: ...` |
| `invalid_distance(2, 0, (1, 2), "negative entry")` | `invalid squared distance in slot 2, dimension 0 at (1, 2): negative entry` |
| `distance_slot(Missing, 1)` | `distance slot mismatch in slot 1: a kernel slot has no supplied distance source` |
| `unsupported_kernel_operation("...")` | `unsupported kernel operation: ...` |
| `cholesky_failed(1e-6, 10, Predict)` | `Cholesky factorization failed (stage=Predict, size=10, jitter=1e-06 already applied)` |
| `non_positive_definite_matrix()` | `matrix is not positive semidefinite` |
| `optimization_not_converged(100)` | `optimizer did not converge after 100 iterations` |
| `workspace_too_small()` | `workspace capacity is insufficient` |
| `persist_failed(Config, "bad json")` | `persist failed (config): bad json` |
| `unsupported_persist_version(2, 1)` | `unsupported persist format version 2; this library reads version 1` |

Factories that take a `reason` are `[[nodiscard]]` and take `std::string`. The factories without arguments do not allocate.
