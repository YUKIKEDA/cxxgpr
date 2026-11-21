# Persistence format specification

This specification defines the format (`FMT-01` to `FMT-08`) in which `cxxgpr` (Exact GPR) saves to and loads from a single directory on disk. The format version is `FORMAT_VERSION = 1`. The files are `config.json` + `model.safetensors`. Note that `factor_kind: "ldlt"` and `point_ids` for online models, and Sparse models, are out of scope and are rejected on load.

---

## FMT-01 Directory layout and version contract

A model is saved to one directory, which contains exactly the following two files directly under it.

- `config.json`: UTF-8 JSON describing the model structure, precision, policies, hyperparameter values and intervals, preprocessing state, and the distance-slot configuration.
- `model.safetensors`: a safetensors binary storing the training data, the computed Cholesky factor, and so on.

The constant `cxxgpr::persist::FORMAT_VERSION = 1` (`std::uint32_t`) is defined. On load, if `format_version` in `config.json` is not the integer `1`, `GprError::UnsupportedPersistVersion { found, supported }` is returned (there is no automatic migration mechanism for other versions).

---

## FMT-02 Atomic writes and the input trust boundary

- **Atomic save**: `save` / `save_with_factor` write temporary files inside the output directory (created if it does not exist) and, after flushing completes, rename-replace them to the final names (`config.json`, `model.safetensors`) in the same directory. Even when re-saving over an existing directory, files that are currently mapped are never destructively rewritten in place.
- **Trust boundary and validation**: the directory to load is verified as untrusted external input.
  - If the JSON nesting depth (`sum`/`product`/`pipeline`/`columnwise`) exceeds 128 levels, it is rejected with `PersistFailed(Config)`.
  - On load, the shape, dtype, and 8-byte alignment of every tensor (`x`, `y`, `l`, `alpha`, `d2.*`) are verified, as is that **every element is finite (not `NaN` and not `±∞`)**.
  - If a non-finite value arises in a transform application result or a reconstructed value, that is also an error, and an inconsistent `LoadedGpr` is never returned.

---

## FMT-03 `config.json` schema

| Key | Type / allowed values | Default when omitted (load) | Description |
|---|---|---|---|
| `format_version` | integer `1` | (required) | Format version |
| `n` | integer ($> 0$) | (required) | Number of training data points |
| `d` | integer ($0$ only for `distance_only`) | (required) | Feature dimension of the training data |
| `has_factor` | bool | (required) | Whether `model.safetensors` contains `l` and `alpha` (`false` for `save`, `true` for `save_with_factor`) |
| `factor_kind` | string `"llt"` | (required) | Exact GPR factor kind. Anything other than `"llt"` gives `PersistFailed(WrongModel)` |
| `factor_jitter` | number ($\ge 0$) | `0.0` | The numerical-stabilization jitter adopted when the factorization succeeded |
| `precision` | `"double"` / `"single"` / `"mixed"` | `"double"` | Precision policy kind |
| `residual` | `"promote_storage"` / `"reevaluate_kernel"` | `"promote_storage"` | Residual scheme for `mixed` |
| `math` | `"accurate"` / `"fast_approx"` | `"accurate"` | `KernelExp` mode |
| `distance_cache` | `"always"` / `"never"` | `"always"` | `DistanceCachePolicy` (`Cached` / `Uncached`) |
| `kernel` | JSON object | (required) | Kernel tree (see below) |
| `likelihood` | `{"noise_variance": <with interval>}` | (required) | The actual value of $\sigma_n^2$ (not its logarithm) and its interval |
| `jitter` | `{"fixed": {"jitter": j}}` or `{"adaptive": {"initial", "multiplier", "max_retries", "max_jitter"}}` | (required) | `JitterPolicy` (passed through the constructor checks at decode time) |
| `x_unfitted`, `x_transform` | transform representation | (required) | Input transform (pre-fit configuration and fitted statistics) |
| `y_unfitted`, `y_transform` | transform representation | (required) | Target transform (pre-fit configuration and fitted statistics) |
| `distance` | object | optional (coordinate model) | Supplied squared-distance metadata (`FMT-07`) |
| `model` | must be absent | none | If `"sgpr"`, `"online_sgpr"`, `"svgp"`, etc. is present, `PersistFailed(WrongModel)` |

Floating-point JSON values are written in a high-precision round-trip notation with no rounding loss (17 digits by default / shortest round-trip). Unknown arbitrary top-level keys are ignored on load.

- **Kernel representation**: a number with an interval is `{"value": v, "lo": a, "hi": b}`. The representation covers each built-in leaf (`rbf`, `rbf_ard`, `matern`, `matern_ard`, `periodic`, `rq`, `rq_ard`, `constant`, `linear`, `white`), composite nodes (`{"sum": {"left": ..., "right": ...}}`, `{"product": {"left": ..., "right": ...}}`), distance nodes (`{"distance": {"slot": k, "leaf": ...}}`), and custom leaves (`{"custom": {"persist_id": "...", "state": ...}}`).
- **Transform representation**: `"identity"`, `"standardize"` (`mean`, `std` once fitted), `min_max` (`range_lo`, `range_hi` before fitting; `data_min`, `data_max`, `range_lo`, `range_hi` after fitting), `pipeline` (`steps`), `columnwise` (`maps`, `x_*` only), and `custom` (`persist_id`, `state`).

---

## FMT-04 `model.safetensors` layout and the column-major contract

The file conforms to the [safetensors](https://github.com/huggingface/safetensors) specification (the first 8 bytes are a little-endian `uint64` header length, followed by the JSON header and the raw tensor data; `__metadata__` is not emitted). **Two-dimensional matrices (`x`, `l`) are stored column-major (Column-major / Fortran order)**, and the position of element `(i, j)` is `j * n + i`. On the other hand, the ARD slots `d2.<k>` are stored as concatenated per-dimension blocks (the `dims` axis outermost, the `n(n+1)/2` packed axis innermost with stride 1). The data section must be aligned to an 8-byte boundary.

| Tensor name | `shape` | `dtype` | Stored when | Contents |
|---|---|---|---|---|
| `x` | `[n, d]` | `F64` | always (omitted for a distance-only model, where `d = 0`) | Training `X` before the input transform |
| `y` | `[n]` | `F64` | always | Training `y` before the target transform |
| `l` | `[n, n]` | `double`→`F64`, `single`/`mixed`→`F32` | `has_factor == true` | The lower-triangular Cholesky factor $L$ of $A + (\sigma_n^2 + j)I$ (zero-filled above the diagonal) |
| `alpha` | `[n]` | `single`→`F32`, `double`/`mixed`→`F64` | `has_factor == true` | The solution weights $\alpha$ (values in the transformed `y` space) |
| `d2.<k>` | scalar `[n(n+1)/2]` / ARD `[dims, n(n+1)/2]` | `single`/`mixed`→`F32`, `double`→`F64` | distance models only | The repaired lower-triangular squared distances of slot `k`. A scalar uses the packed position $p = jn - j(j-1)/2 + (i-j)$; an ARD slot concatenates, for each dimension $m \in [0, \text{dims})$, a packed array of length $\text{packed\_len} = n(n+1)/2$ (flat index `m * packed_len + p`) (`FMT-07`) |

When verifying or using the file directly from NumPy, column-major matrices such as `x` and `l` are restored bit-exactly to their original shape and meaning with `flat.reshape(meta["shape"], order="F")`, and the ARD distance tensor `d2.<k>` with `flat.reshape(meta["shape"], order="C")` (the one-dimensional tensors `y`, `alpha`, and a scalar `d2.<k>` are identical either way).

---

## FMT-05 `save` / `save_with_factor` and `mmap` loading of the `f64` factor

| Method | `has_factor` | `model.safetensors` contents | Load behavior (`LoadedGpr::load` / `LoadedDistanceGpr<C>::load`) |
|---|---|---|---|
| `save(dir)` | `false` | `x`, `y` (+ `d2.*`) | Rebuilds the Gram matrix from the saved transforms, $\theta$, `jitter`, and `factor_jitter`, then reconstructs the Cholesky factorization and $\alpha$ in $O(n^3)$ |
| `save_with_factor(dir)` | `true` | `x`, `y`, `l`, `alpha` (+ `d2.*`) | Skips refactorization. The `F64` factor `l` of `DoublePrecision` is referenced directly through a read-only memory map (`mmap`) by both loaders (`has_mmap_factor()` is `true`); an `F32` factor is copied into memory. Typed extraction (`into_fitted`) copies the factor into the returned fitted model, which does not depend on the file |

- **`mmap` safety requirements**: the `mmap` of the `F64` factor `l` verifies that the data start offset is aligned to an 8-byte boundary before mapping (Windows `CreateFileMapping`/`MapViewOfFile`, Linux `mmap(PROT_READ, MAP_PRIVATE)`). The contract is that no external process rewrites the file in place during the model's lifetime; replacement by `save` is done by atomic rename and therefore does not break a map in progress.
- **Loader return types**: `LoadedGpr` (a tagged union / `std::variant` wrapper holding the four Exact LLT variants `double`, `single`, `mixed_promote`, and `mixed_reevaluate`) and `LoadedDistanceGpr<C>` (`C = DistanceOnly | WithPoints`). They commonly provide `predict` (widened to `double`), `n()`, and `d()` independent of the model kind, and can be converted to a `FittedGpr` by typed extraction. A `load`ed model uses the `Retain` policy and can `refit` after `with_optimizer`.

---

## FMT-06 Custom extension registration and `PersistRegistry`

`PersistRegistry` is provided to save and restore non-built-in `KernelTerm`s and custom input and target transforms.

- **`persist_id` constraints**: the empty string and IDs starting with the reserved prefix `"cxxgpr."` (`RESERVED_PREFIX`) are forbidden. On violation, `PersistFailed(InvalidPersistId)` is returned at `save` time and at registration time. Duplicate registration in the same category is also `InvalidPersistId`.
- **Registration functions** (4 categories): `register_kernel`, `register_unfitted_input`, `register_fitted_input`, `register_unfitted_target`, `register_fitted_target` (5 registration functions in total: 2 each for input and target before and after fitting, plus 1 for kernels).
- **Unsupported and unregistered errors**: saving a custom type that does not implement `persist_id()` or does not support persistence gives `PersistFailed(NotPersistable)`; an unregistered ID that appears on load gives `PersistFailed(UnregisteredId)`.

---

## FMT-07 Persistence of supplied squared-distance models (`LoadedDistanceGpr<C>`)

API: `persist::save(const FittedDistanceGpr<O, P, C>&, dir[, registry])`, `persist::save_with_factor(...)` and `persist::LoadedDistanceGpr<C>::load(dir[, registry])` (REL-7, DES-21). The loaded model exposes the queries `predict`, `predict_covariance`, `sample` and `loo_predict` (results widened to `double`, with the query supplies, validation order and errors of DES-21), the getters `n()`, `d()`, `precision_kind()`, `has_factor()`, `has_mmap_factor()`, `factor_jitter()`, `kernel()`, `likelihood()`, `jitter()` and `kernel_exp()`, and the typed extraction `into_fitted<O, P>(O optimizer = O{})` to `FittedDistanceGpr<O, P, C>` (`PersistFailed(WrongModel)` when `P` is not the saved precision). It has no `predict_into`: the allocation-free query path is `FittedDistanceGpr::predict_into` on the extracted model. The `F64` factor of a `DoublePrecision` checkpoint saved with `save_with_factor` is memory-mapped as in FMT-05, under the same `mmap` safety requirements.

For `DistanceKernel<C>` models, a `distance` section is added to `config.json`, and per-slot lower-triangular tensors `d2.<k>` are saved in `model.safetensors`.

```json
"distance": {
  "points": "distance_only",
  "slots": [ { "kind": "scalar" }, { "kind": "ard", "dims": 2 } ]
}
```

- `points` is `"distance_only"` or `"with_points"`. Loading a `DistanceKernel<WithPoints>` with a `DistanceOnly` loader (or the reverse, or with a coordinate `LoadedGpr`) always returns `PersistFailed(WrongModel)`.
- `slots` must match the order of `DistanceKernel::slots()` (depth-first, order of first appearance). A `distance_only` kernel that contains a coordinate-dependent leaf (`rbf_ard`, `linear`, etc.) is also rejected with `PersistFailed(Config)`.
- **`d2.<k>` storage rule**: for a scalar distance slot, the lower-triangular elements `j..n` of column $j$ (diagonal included) are packed in column order (packed position $p = jn - j(j-1)/2 + (i - j)$, length $n(n+1)/2$). For an ARD distance slot (header `shape` is `[dims, n(n+1)/2]`), a lower-triangular packed array of length $\text{packed\_len} = n(n+1)/2$ is concatenated for each dimension $m \in [0, \text{dims})$ in order (flat index `m * packed_len + p`, with the inner axis being `packed_len`). Therefore, when reading with NumPy or similar, interpret the two-dimensional column-major matrices (`x`, `l`) with `flat.reshape(meta["shape"], order="F")` and the distance tensor `d2.<k>` with `flat.reshape(meta["shape"], order="C")` (a one-dimensional scalar `d2.<k>` is identical under either `F` or `C`). For a `MixedPrecision` model, the `float` tables that the model predicts with are written (the double tables are not retained after training), so a reload reproduces the stored state exactly.

```python
# Minimal Python / NumPy read rule (x, l are column-major; ARD d2.<k> is concatenated per-dimension blocks)
def reshape_tensor(name: str, flat: "np.ndarray", shape: list[int]) -> "np.ndarray":
    order = "C" if name.startswith("d2.") else "F"
    return flat.reshape(shape, order=order)
```
- **Issuing new slots**: on load, a fresh sequence of slot handles dedicated to the model is generated, so a `ScalarDistance` / `ArdDistance` from before saving cannot be used with the loaded model. Users obtain the new slots from `slots()` / `to_kernel()` to build the prediction supply.
- On load, the tensor reader rejects non-finite `d2.<k>` elements and shape or dtype mismatches with `PersistFailed(Tensor)`; the stored tables then pass the same `prepare_square` validation as supplied tables (non-negativity, zero diagonal, symmetry), and `InvalidDistance` is returned on violation.
- For a distance-only model `d` is `0`, the `x` tensor is omitted, and `x_transform` is `"identity"`. A `with_points` model stores `x` and requires `d > 0`. The `distance` section is the only marker of a distance model, and `distance_cache` is written as `"always"` and ignored.
- `LoadedDistanceGpr<C>` exposes `predict`, `predict_covariance`, `sample`, `loo_predict`, `into_fitted<O, P>` and the getters `n`, `d`, `precision_kind`, `has_factor`, `factor_jitter`, `kernel`, `likelihood`, `jitter`, `kernel_exp`. Its results are `double`. Copies own independent query buffers. With `save` (no factor) the factor is recomputed at load without optimization.

---

## FMT-08 Error kinds and external-tool compatibility mapping

| Abnormal condition | `GprError` returned |
|---|---|
| File open/close, read, write, or `mmap` failure | `PersistFailed { PersistErrorKind::Io, reason }` |
| JSON syntax error, missing required key, nesting exceeded, `factor_kind != "llt"` | `PersistFailed { PersistErrorKind::Config, reason }` |
| Missing tensor, shape / dtype mismatch, not 8-byte aligned, non-finite tensor | `PersistFailed { PersistErrorKind::Tensor, reason }` |
| Coordinate / distance / marker mismatch, file specifying a Sparse/Online model | `PersistFailed { PersistErrorKind::WrongModel, reason }` |
| Unregistered custom `persist_id` | `PersistFailed { PersistErrorKind::UnregisteredId, reason }` |
| Empty or `"cxxgpr."`-prefixed `persist_id`, or duplicate registration | `PersistFailed { PersistErrorKind::InvalidPersistId, reason }` |
| `save` of a custom type that does not support `persist_id` | `PersistFailed { PersistErrorKind::NotPersistable, reason }` |
| `format_version != 1` | `UnsupportedPersistVersion { found, supported: 1 }` |
| `d2.<k>` negative or non-zero diagonal | `InvalidDistance` |
| `d2.<k>` non-finite element, wrong shape or dtype | `PersistFailed { PersistErrorKind::Tensor, reason }` |
| Invalid saved interval, Jitter, or kernel value | The `GprError` of the respective constructor / factory |
