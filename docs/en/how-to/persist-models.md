# How to save and load models

A fitted model can be written to a directory and read back later, in another process or on another machine, and give the same predictions. This guide shows how to save with `save` or `save_with_factor`, load with `LoadedGpr`, predict from a loaded model or turn it back into a typed `FittedGpr`, persist models that contain your own kernels and transforms with `PersistRegistry`, and handle the errors that come with files. The directory layout itself is described in {doc}`../reference/file-format`.

The programs behind this page are `persist_roundtrip.cpp` and `custom_persist_registry.cpp`. Both write to a temporary directory and delete it before they exit. Include `cxxgpr/persist/persist.hpp`.

## Choose how to save

Two free functions write a model. They take the fitted model, the target directory and an optional `PersistRegistry`, and return `std::expected<void, GprError>`:

| Function | Writes | Load time | File size | Use it when |
|---|---|---|---|---|
| `save` | training data, hyperparameters, transforms | refactors the kernel matrix, $O(n^3)$ | small | you ship the model and can afford to refactor on load |
| `save_with_factor` | the same, plus the Cholesky factor $L$ and the weights $\alpha$ | no refactorization; the double-precision factor is memory-mapped | about $n^2$ values more | you load often, or $n$ is large |

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [save]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
config.json: 1176 bytes
model.safetensors: 5440 bytes
```

The directory is created if it does not exist. After a successful save it contains exactly `config.json` and `model.safetensors`. The model in the example has 24 training points; the factor adds a $24 \times 24$ matrix of doubles, which is most of the 5440 bytes.

Both functions work for every precision policy and every built-in kernel and transform. A model that contains a custom kernel or transform needs the registry argument; see {ref}`persist-models-custom`.

## Understand what a save guarantees

Saving does not modify the model. It writes two temporary files, `model.safetensors.tmp` and `config.json.tmp`, inside the target directory, flushes them, and then renames each onto its final name. Consequences:

- A reader never sees a half-written file: each file is replaced as a whole, by a rename.
- If writing or renaming fails, `save` removes the temporary files and returns `PersistErrorKind::Io`.
- Models that cannot be saved are detected before anything is written, and the target directory is left untouched: a custom component without a valid `persist_id` gives `PersistErrorKind::NotPersistable` or `PersistErrorKind::InvalidPersistId`, custom state that cannot be written into UTF-8 JSON gives `PersistErrorKind::NotPersistable` (see {ref}`persist-models-custom`), and a `config.json` that would be nested deeper than 128 levels, which `load` would reject, gives `PersistErrorKind::Config`.

:::{warning}
The two files are renamed one after the other, not as one transaction. If the process is killed between the two renames, or the second rename fails (`save` then returns `PersistErrorKind::Io`), the directory holds the new `model.safetensors` and the old `config.json`; `save` does not put the old `model.safetensors` back. The loader rejects the pair when a tensor's shape or `dtype` does not match `config.json`, but it cannot know that two matching files come from different saves, and then loads a model that mixes the new data with the old hyperparameters. When correctness matters, save into a fresh directory and move the whole directory into place.
:::

:::{warning}
On Windows a file that is memory-mapped cannot be replaced. Saving into a directory whose `model.safetensors` is mapped by a live `LoadedGpr` (see below) fails with `PersistErrorKind::Io` and leaves the existing files as they were. Destroy every `LoadedGpr` that maps the old file first, or save to a new directory. On Linux, replacing a mapped file works: the live `LoadedGpr` keeps predicting from the old data, and a new `load` reads the new files.
:::

## Load a model

`LoadedGpr::load` reads a directory and returns a `LoadedGpr`: a type-erased model that predicts in double precision whatever precision it was trained with.

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [load]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
n = 24, d = 1, has_factor = true, mmap factor = true
save_with_factor round trip is bit-identical: true
save() (refactorized on load): has_factor = false, max |mean difference| = 0
```

For a model saved with `save_with_factor` in `DoublePrecision`, the loaded model predicts **bit-for-bit** the same mean and variance as the original: the factor and the weights are read back unchanged, and the hyperparameters and transform statistics are stored with all digits needed to reproduce them exactly. In this example the model saved with `save` also reproduced the means exactly, because refactoring the same matrix with the same code is deterministic. Treat that as likely, not guaranteed: the format guarantees bit-identical results only when the factor is stored, so compare with a tolerance when you use `save`.

`load` takes the registry as its second argument, `load(dir, registry)`; omit it for models that use only built-in components.

### Inspect a loaded model

A `LoadedGpr` reports what the file contained:

| Member | Returns |
|---|---|
| `n`, `d` | number of training samples and features |
| `precision_kind` | the `PersistPrecisionKind` the model was trained with: `Double`, `Single`, `MixedPromote` or `MixedReevaluate` |
| `has_factor` | whether the file stored $L$ and $\alpha$ |
| `has_mmap_factor` | whether $L$ is read directly from the memory-mapped file |
| `factor_jitter` | the diagonal jitter that was accepted when the factorization succeeded |
| `kernel_spec` | the restored `KernelSpec`, with the fitted hyperparameters |
| `likelihood` | the restored `GaussianLikelihood` (noise variance and its interval) |
| `jitter` | the restored `JitterPolicy` |
| `kernel_exp` | the `KernelExp` mode |
| `distance_cache_policy` | the `DistanceCachePolicy` |

### Predict from a loaded model

`LoadedGpr` offers the prediction methods of `FittedGpr`, always in `double` and on the original target scale. Input and target transforms are applied for you.

| Method | Result |
|---|---|
| `predict(xs, options)` and `predict(span, n_rows, n_cols, options)` | `Prediction<double>` with the mean and variance |
| `predict_into(xs, out, options)` | the same into a caller-supplied `Prediction<double>` |
| `predict_covariance(xs, options)` | `PredictiveCovariance<double>`, the full $m \times m$ covariance |
| `sample(xs, n_draws, seed, options, jitter)` | `std::vector<double>`, an $m \times n_\text{draws}$ column-major matrix of posterior samples, deterministic in `seed` |
| `loo_predict(options)` | closed-form leave-one-out predictions for the training points |

The `options` argument is a `PredictOptions` that selects latent or observation variance, as it does for `FittedGpr`.

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [loaded-api]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
covariance entries = 16, samples = 12, loo points = 24
```

All of these return `GprErrorCode::EmptyInput` for an empty query, `GprErrorCode::DimensionMismatch` if the query has a different number of columns than `d()`, and `GprErrorCode::NonFiniteInput` for `NaN` or infinity; `sample` can also return `GprErrorCode::CholeskyFailed`.

## Turn a loaded model back into a `FittedGpr`

`into_fitted` extracts a typed model that owns its data and no longer depends on the file or the memory map. Pass the optimizer type and the precision policy you want; the policy must match the one stored in the file.

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [into-fitted]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
into_fitted prediction is bit-identical: true
into_fitted<SinglePrecision> on a double model: persist failed (wrong_model): requested precision policy does not match persisted model precision
```

`into_fitted<O, P>(optimizer)` returns `PersistErrorKind::WrongModel` if `P` is not the stored policy: you cannot reinterpret a double model as single precision. The optimizer is not stored in the file, so you choose it here; it only matters if you later call `refit` or `into_trainer` on the result. The extracted model keeps its Cholesky factor, so it behaves like a freshly fitted one.

| Stored `precision_kind()` | Policy to ask for |
|---|---|
| `PersistPrecisionKind::Double` | `DoublePrecision` |
| `PersistPrecisionKind::Single` | `SinglePrecision` |
| `PersistPrecisionKind::MixedPromote` | `MixedPrecision<PromoteStorage>` |
| `PersistPrecisionKind::MixedReevaluate` | `MixedPrecision<ReevaluateKernel>` |

## Know when the factor is memory-mapped

When a `DoublePrecision` model with `has_factor == true` is loaded, the factor $L$ is not copied: `has_mmap_factor()` is `true` and the operating system pages the data in on demand. That makes loading a large model almost instant and lets several processes share one copy. Single-precision and mixed-precision factors are stored as 32-bit floats and are copied into memory, so `has_mmap_factor()` is `false` for them, and so it is for models saved with `save`.

The mapping lives as long as any copy of the `LoadedGpr` does: copies share the underlying data, and the file stays mapped until the last one is destroyed.

:::{warning}
Never modify `model.safetensors` in place while a `LoadedGpr` that maps it exists; the loaded model would then read changed bytes. Predictions do not check the factor again after `load`, so the results are then unspecified and can be wrong without an error. Replace the file instead, which is what `save` does (subject to the Windows restriction above).
:::

(persist-models-custom)=
## Persist custom kernels and transforms

A saved model describes its kernel and transforms by name. Built-in ones are recognised by the loader. A kernel built from `KernelTerm` through `CustomKernel` and a transform you derived from the transform interfaces are not, so you must give each a stable identifier and tell the loader how to rebuild it:

1. In the class, override `persist_id()` to return a unique identifier and `persist_state_json()` to return the state needed to rebuild the object. For a transform this is the same pair of virtual functions described in {doc}`transform-data`; for a kernel leaf they are on `KernelTerm`.
2. Register one factory per identifier in a `PersistRegistry`, with `register_kernel`, `register_unfitted_input`, `register_fitted_input`, `register_unfitted_target` or `register_fitted_target`.
3. Pass the same registry to `save`/`save_with_factor` and to `LoadedGpr::load`.

The example persists a model with a custom kernel and a custom input transform (a logarithm followed by standardization). The classes `GaussTerm`, `FittedLog` and `UnfittedLog` implement the interfaces and the persistence pair, for instance:

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [transform]
:end-before: // [end]
:dedent: 4
```

One registry knows how to rebuild every custom piece. A factory receives the state text and returns the object or an error:

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [registry]
:end-before: // [end]
:dedent: 4
```

Saving and loading are then ordinary calls with the registry attached:

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [save-load]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
load without registry fails with UnregisteredId: true
predictions after load are bit-identical: true
```

### Rules for identifiers and state

- An identifier must be non-empty, must not start with `cxxgpr.` (the constant `RESERVED_PREFIX`), and must be valid UTF-8. `validate_persist_id` applies this rule, and registration and saving both use it. A violation gives `PersistErrorKind::InvalidPersistId`.
- An identifier can be registered once per category. A second registration fails with `PersistErrorKind::InvalidPersistId`. The five registration categories (kernel, unfitted and fitted input transform, unfitted and fitted target transform) are separate. A transform needs an unfitted and a fitted registration, and the example gives them different identifiers (`example.log.unfitted`, `example.log.fitted`).
- The state is a string that should be valid JSON. If it parses as a JSON object, array, number, `true`, `false` or `null`, the library stores it as structured JSON inside `config.json` and hands your factory the re-serialized text, which need not equal what you returned: numbers are rewritten (`2.500000` becomes `2.5`, `1e2` becomes `100.0`), whitespace is dropped and object keys are sorted. Any other state, including one that is itself a JSON string literal such as `"abc"`, is stored as a plain string, and your factory receives exactly the text you returned, quotes included. A state stored as a plain string must be valid UTF-8; otherwise saving fails with `PersistErrorKind::NotPersistable`. Prefer a JSON object or number, and make it complete: it must contain everything the factory needs, including the *fitted* hyperparameters of a kernel.
- To keep results bit-identical, write floating-point state with all digits. `std::format("{}", x)` produces the shortest text that reads back exactly. Do not use `std::to_string`, which keeps six digits.
- Without a `persist_id` (the default), a custom component is not persistable and saving fails with `PersistErrorKind::NotPersistable`.

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [registration-errors]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
reserved prefix rejected: true, duplicate rejected: true
```

The factory aliases `CustomKernelFactory`, `UnfittedInputFactory`, `FittedInputFactory`, `UnfittedTargetFactory` and `FittedTargetFactory` name the function types that `PersistRegistry` accepts. The registry also exposes the lookups it uses internally: `create_kernel`, `create_unfitted_input`, `create_fitted_input`, `create_unfitted_target` and `create_fitted_target`. They return the rebuilt object, or `PersistErrorKind::UnregisteredId` for an unknown identifier. You do not normally call them yourself, but they are handy for testing a factory in isolation.

## Handle errors

Failures are reported through `GprError`. For persistence errors, `GprError::code()` is `GprErrorCode::PersistFailed`, and `persist_error_kind()` returns the `PersistErrorKind` that tells you what went wrong:

| `PersistErrorKind` | Meaning | Typical cause |
|---|---|---|
| `Io` | the file system failed | missing directory, no permission, disk full, mapping failed |
| `Config` | `config.json` is malformed | invalid JSON, a missing key, nesting deeper than 128 levels |
| `Tensor` | `model.safetensors` is invalid | truncated file, wrong shape or `dtype`, a non-finite value, misaligned data |
| `InvalidPersistId` | a custom identifier is not allowed | empty, reserved prefix, not valid UTF-8, or already registered |
| `NotPersistable` | the model has a component that cannot be saved | a custom kernel or transform without `persist_id`, or with state that is not JSON and not valid UTF-8 |
| `UnregisteredId` | the file names a custom component that is not registered | `load` without the registry that `save` used |
| `WrongModel` | the file is not what the caller asked for | precision mismatch in `into_fitted`, or a file written for another model family |

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
missing directory -> PersistErrorKind::Io: true
```

Loading never trusts the file. It rejects non-finite values anywhere in the tensors, checks every shape, `dtype` and alignment, and bounds the nesting depth of the JSON. A load can also fail with `GprErrorCode::CholeskyFailed` for a model saved with `save` whose kernel matrix can no longer be factored.

## Handle format versions

The constant `FORMAT_VERSION` is the on-disk format version that this release writes and reads; it is currently `1`. A file with any other `format_version` makes `load` return `GprErrorCode::UnsupportedPersistVersion`. The error carries both numbers:

```cpp
auto loaded = cxxgpr::persist::LoadedGpr::load(dir);
if (!loaded && loaded.error().code() == cxxgpr::GprErrorCode::UnsupportedPersistVersion)
{
    // loaded.error().found_version()     -> version in the file (std::optional<std::uint32_t>)
    // loaded.error().supported_version() -> version this library supports
}
```

There is no automatic migration. Keep the library version that wrote an old model if you need to read it, or refit from the training data, which every file contains in `x` and `y`.

## Persist a supplied-distance model

A `FittedDistanceGpr` is saved with the same `save` and `save_with_factor` functions, and read back with `LoadedDistanceGpr<C>`, where `C` is `DistanceOnly` or `WithPoints`, the mode of the model that was saved. The checkpoint adds one `d2.<k>` tensor per distance slot (the lower triangle of the validated training distances) and a `distance` section in `config.json`; see {doc}`../reference/file-format`.

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [save]
:end-before: // [end]
```

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [load]
:end-before: // [end]
```

Loading issues **fresh slots**. A `ScalarDistance` or `ArdDistance` that you used before saving is not valid for the loaded model, so take the slots from `kernel().slots()` of the loaded model, in the order of the kernel, and bind the query tables to them:

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [bind]
:end-before: // [end]
```

`LoadedDistanceGpr` offers `load`, `n`, `d`, `precision_kind`, `has_factor`, `has_mmap_factor`, `factor_jitter`, `kernel`, `likelihood`, `jitter`, `kernel_exp`, `predict`, `predict_covariance`, `sample`, `loo_predict` and `into_fitted`. The query methods take the same tables as `FittedDistanceGpr` (and the query coordinates for `WithPoints`) and return `double` results. `into_fitted` gives back a typed `FittedDistanceGpr` for the precision policy that was saved.

- With `save_with_factor`, the factor and weights are restored as stored, and every prediction is bit-identical to the model before saving. A double-precision factor is read through a read-only memory map, as for `LoadedGpr` (`has_mmap_factor()` is `true`), so the same rule holds: do not rewrite the file in place while the model or a copy of it is alive. `into_fitted` copies the factor, so the model it returns does not depend on the file. With `save`, the factor is computed again at load, and predictions agree to rounding.
- A loaded `MixedPrecision` model has no `refinement_summary`, and its `neg_log_marginal_likelihood` is recomputed from the stored factor and the refined weights, so it can differ slightly from the value reported at training, which comes from the single-precision solve.
- The tables are validated again on load. A negative entry or a nonzero diagonal gives `InvalidDistance`; a non-finite value is rejected as a corrupt tensor (`PersistErrorKind::Tensor`).
- A checkpoint of the other mode, or of a coordinate model, is refused with `PersistErrorKind::WrongModel`:

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [wrong-loader]
:end-before: // [end]
```

A `MixedPrecision` model stores its tables in single precision, which is the precision it predicts with.

## API summary

| Symbol | Purpose |
|---|---|
| `FORMAT_VERSION` | the format version this release reads and writes (`1`) |
| `RESERVED_PREFIX` | the `cxxgpr.` prefix that custom identifiers must not use |
| `save` | write a model without the factor; atomic per file |
| `save_with_factor` | write a model with $L$ and $\alpha$ |
| `LoadedGpr` | the loaded, type-erased model: `load`, `n`, `d`, `precision_kind`, `has_factor`, `has_mmap_factor`, `factor_jitter`, `kernel_spec`, `likelihood`, `jitter`, `kernel_exp`, `distance_cache_policy`, `predict`, `predict_into`, `predict_covariance`, `sample`, `loo_predict`, `into_fitted` |
| `LoadedDistanceGpr` | the loaded supplied-distance model of mode `DistanceOnly` or `WithPoints`: `load`, `n`, `d`, `precision_kind`, `has_factor`, `has_mmap_factor`, `factor_jitter`, `kernel`, `likelihood`, `jitter`, `kernel_exp`, `predict`, `predict_covariance`, `sample`, `loo_predict`, `into_fitted` |
| `PersistRegistry` | factories for custom components: `register_kernel`, `register_unfitted_input`, `register_fitted_input`, `register_unfitted_target`, `register_fitted_target`, and `create_kernel`, `create_unfitted_input`, `create_fitted_input`, `create_unfitted_target`, `create_fitted_target` |
| `validate_persist_id` | checks an identifier against the empty-string, reserved-prefix and UTF-8 rules |
| `CustomKernelFactory`, `UnfittedInputFactory`, `FittedInputFactory`, `UnfittedTargetFactory`, `FittedTargetFactory` | factory function types |

The generated reference is in [the persist API reference](../reference/api/persist.md).
