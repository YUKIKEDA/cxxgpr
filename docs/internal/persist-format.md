# Persistence format specification

This specification defines the format (`FMT-01` to `FMT-09`) in which `cxxgpr` (Exact GPR) saves to and loads from a single directory on disk. The format version is `FORMAT_VERSION = 1`. The files are `config.json` + `model.safetensors`. Note that `factor_kind: "ldlt"` and `point_ids` for online models, and Sparse models, are out of scope and are rejected on load.

---

## FMT-01 Directory layout and version contract

A model is saved to one directory, which contains exactly the following two files directly under it.

- `config.json`: UTF-8 JSON describing the model structure, precision, policies, hyperparameter values and intervals, preprocessing state, and the distance-slot configuration.
- `model.safetensors`: a safetensors binary storing the training data, the computed Cholesky factor, and so on.

The constant `cxxgpr::persist::FORMAT_VERSION = 1` (`std::uint32_t`) is defined. On load, if `format_version` in `config.json` is not the integer `1`, `GprError::UnsupportedPersistVersion { found, supported }` is returned (there is no automatic migration mechanism for other versions).

---

## FMT-02 Atomic writes and the input trust boundary

- **Atomic save**: `save` / `save_with_factor` write temporary files inside the output directory (created if it does not exist) and, after flushing completes, rename-replace them to the final names (`config.json`, `model.safetensors`) in the same directory. Even when re-saving over an existing directory, files that are currently mapped are never destructively rewritten in place.
- **Platform limits of the atomic save**:
  - The two files are replaced one after the other (`model.safetensors` first, then `config.json`), so the pair is not replaced atomically. Two events leave the new `model.safetensors` beside the old `config.json`: the process stopping between the two renames, and the second rename failing. In the second case `save` returns `PersistFailed(Io)` ("failed to atomically replace config.json") and removes `config.json.tmp`, but does not restore the old `model.safetensors`. Loading checks `model.safetensors` against `config.json`, so such a pair is rejected with `PersistFailed(Tensor)` when a tensor's shape or dtype differs from what `config.json` implies (`n`, `d`, `has_factor`, `precision`) or a tensor it requires is missing. When they agree, the pair loads without error, and the model combines the new tensors with the old hyperparameters and transforms. Save into a fresh directory and rename the directory when the pair must change atomically. `FMT-09` specifies the identifier with which `load` will reject such a pair (not implemented until roadmap row `REL-11` is Done).
  - On Windows, a file that a live `LoadedGpr` has memory-mapped cannot be renamed over. Saving into that directory fails with `PersistFailed(Io)` ("failed to atomically replace model.safetensors") and leaves the existing files intact. On Linux the rename succeeds, the live `LoadedGpr` keeps predicting from the old contents it mapped, and a new `load` reads the new files. Release the `LoadedGpr` before saving over its directory when the save must succeed on every platform.
- **Trust boundary and validation**: the directory to load is verified as untrusted external input.
  - If the JSON nesting depth (`sum`/`product`/`pipeline`/`columnwise`) exceeds 128 levels, it is rejected with `PersistFailed(Config)`.
  - On load, the shape, dtype, and 8-byte alignment of every tensor (`x`, `y`, `l`, `alpha`, `d2.*`) are verified, as is that **every element is finite (not `NaN` and not `±∞`)**.
  - If a non-finite value arises in a transform application result or a reconstructed value, that is also an error. These checks reject every file that is invalid on its own, and every pair whose files disagree in shape, dtype, or the presence of a tensor. They cannot detect a `config.json` and a `model.safetensors` that are each valid and agree in all of these but come from different saves (see the platform limits above).
- **Save-side limits**: `save` / `save_with_factor` never write a directory that `load` would reject for its encoding or nesting. A model whose `config.json` would be nested more than 128 levels deep gives `PersistFailed(Config)`, and custom text that cannot be written into UTF-8 JSON gives the error stated in `FMT-06`. Both are detected before the output directory is created or any file in it is touched.

---

## FMT-03 `config.json` schema

| Key | Type / allowed values | Default when omitted (load) | Description |
|---|---|---|---|
| `format_version` | integer `1` | (required) | Format version |
| `save_id` | string of 32 lowercase hexadecimal digits | absent: no pairing check | Pairing identifier of this save; equal to `cxxgpr.save_id` in `model.safetensors` (`FMT-09`) |
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

The file conforms to the [safetensors](https://github.com/huggingface/safetensors) specification (the first 8 bytes are a little-endian `uint64` header length, followed by the JSON header and the raw tensor data; `__metadata__` holds only the pairing identifier `cxxgpr.save_id` of `FMT-09`). **Two-dimensional matrices (`x`, `l`) are stored column-major (Column-major / Fortran order)**, and the position of element `(i, j)` is `j * n + i`. On the other hand, the ARD slots `d2.<k>` are stored as concatenated per-dimension blocks (the `dims` axis outermost, the `n(n+1)/2` packed axis innermost with stride 1). The data section must be aligned to an 8-byte boundary.

| Tensor name | `shape` | `dtype` | Stored when | Contents |
|---|---|---|---|---|
| `x` | `[n, d]` | `F64` | always | Training `X` before the input transform |
| `y` | `[n]` | `F64` | always | Training `y` before the target transform |
| `l` | `[n, n]` | `double`→`F64`, `single`/`mixed`→`F32` | `has_factor == true` | The lower-triangular Cholesky factor $L$ of $A + (\sigma_n^2 + j)I$ (zero-filled above the diagonal) |
| `alpha` | `[n]` | `single`→`F32`, `double`/`mixed`→`F64` | `has_factor == true` | The solution weights $\alpha$ (values in the transformed `y` space) |
| `d2.<k>` | scalar `[n(n+1)/2]` / ARD `[dims, n(n+1)/2]` | `single`→`F32`, `double`/`mixed`→`F64` | distance models only | The repaired lower-triangular squared distances of slot `k`. A scalar uses the packed position $p = jn - j(j-1)/2 + (i-j)$; an ARD slot concatenates, for each dimension $m \in [0, \text{dims})$, a packed array of length $\text{packed\_len} = n(n+1)/2$ (flat index `m * packed_len + p`) (`FMT-07`) |

When verifying or using the file directly from NumPy, column-major matrices such as `x` and `l` are restored bit-exactly to their original shape and meaning with `flat.reshape(meta["shape"], order="F")`, and the ARD distance tensor `d2.<k>` with `flat.reshape(meta["shape"], order="C")` (the one-dimensional tensors `y`, `alpha`, and a scalar `d2.<k>` are identical either way).

---

## FMT-05 `save` / `save_with_factor` and `mmap` loading of the `f64` factor

| Method | `has_factor` | `model.safetensors` contents | Load behavior (`LoadedGpr::load` / `LoadedDistanceGpr<C>::load`) |
|---|---|---|---|
| `save(dir)` | `false` | `x`, `y` (+ `d2.*`) | Rebuilds the Gram matrix from the saved transforms, $\theta$, `jitter`, and `factor_jitter`, then reconstructs the Cholesky factorization and $\alpha$ in $O(n^3)$ |
| `save_with_factor(dir)` | `true` | `x`, `y`, `l`, `alpha` (+ `d2.*`) | Skips refactorization. The `F64` factor `l` of `DoublePrecision` is referenced directly through a read-only memory map (`mmap`) by both loaders (`has_mmap_factor()` is `true`); an `F32` factor is copied into memory. Typed extraction (`into_fitted`) copies the factor into the returned fitted model, which does not depend on the file |

- **`mmap` safety requirements**: the `mmap` of the `F64` factor `l` verifies that the data start offset is aligned to an 8-byte boundary before mapping (Windows `CreateFileMapping`/`MapViewOfFile`, Linux `mmap(PROT_READ, MAP_PRIVATE)`). The contract is that no external process rewrites the file in place during the model's lifetime. `save` never rewrites a file in place, so it does not break a map in progress: on Linux it renames the new file over the mapped one and the map keeps the old contents, and on Windows that rename fails and `save` returns `PersistFailed(Io)` with the existing files intact (`FMT-02`). Predictions trust the validation done by `load` (every element finite, positive diagonal) and do not re-check the factor on each call, so the result of violating this contract is unspecified: a rewritten value can yield a silently wrong prediction, and only a non-finite latent predictive variance is still reported (`NonFiniteKernelValue`, DES-16).
- **Loader return types**: `LoadedGpr` (a tagged union / `std::variant` wrapper holding the four Exact LLT variants `double`, `single`, `mixed_promote`, and `mixed_reevaluate`) and `LoadedDistanceGpr<C>` (`C = DistanceOnly | WithPoints`). They commonly provide `predict` (widened to `double`), `n()`, and `d()` independent of the model kind, and can be converted to a `FittedGpr` by typed extraction. A `load`ed model uses the `Retain` policy and can `refit` after `with_optimizer`.

---

## FMT-06 Custom extension registration and `PersistRegistry`

`PersistRegistry` is provided to save and restore non-built-in `KernelTerm`s and custom input and target transforms.

- **`persist_id` constraints**: the empty string, IDs starting with the reserved prefix `"cxxgpr."` (`RESERVED_PREFIX`), and IDs that are not valid UTF-8 (they could not be written into `config.json`) are forbidden. On violation, `PersistFailed(InvalidPersistId)` is returned at `save` time and at registration time. Duplicate registration in the same category is also `InvalidPersistId`.
- **Registration functions** (4 categories): `register_kernel`, `register_unfitted_input`, `register_fitted_input`, `register_unfitted_target`, `register_fitted_target` (5 registration functions in total: 2 each for input and target before and after fitting, plus 1 for kernels).
- **Custom state encoding**: the text returned by `persist_state_json()` is stored under `state` in one of two ways.
  - Text that parses as a JSON value other than a string is stored as that value, and the factory receives its compact re-serialization, which in general differs from the original text: numbers take the serializer's shortest round-trip form (`2.500000` → `2.5`, `1e2` → `100.0`, `-0` → `0`), an integer outside the 64-bit range becomes the nearest `double` (`12345678901234567890123` → `1.2345678901234568e+22`), whitespace outside strings is dropped, object keys are sorted, and of duplicate keys only the last is kept. If the value is nested so deeply that `config.json` would exceed 128 levels, `save` returns `PersistFailed(Config)` (`FMT-02`).
  - Every other text is stored as a JSON string holding the original text, and the factory receives it byte for byte. This covers text that does not parse and text that is itself a JSON string literal (for example `"abc"`, which is handed back with its quotes). Such text must be valid UTF-8, because `config.json` is UTF-8 JSON; otherwise `save` returns `PersistFailed(NotPersistable)`.
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

- **Pairing identifier**: distance models carry the `FMT-09` identifier exactly as coordinate models do; it covers the `distance` section and every `d2.<k>` tensor.
- `points` is `"distance_only"` or `"with_points"`. Loading a `DistanceKernel<WithPoints>` with a `DistanceOnly` loader (or the reverse, or with a coordinate `LoadedGpr`) always returns `PersistFailed(WrongModel)`.
- `slots` must match the order of `DistanceKernel::slots()` (depth-first, order of first appearance). A `distance_only` kernel that contains a coordinate-dependent leaf (`rbf_ard`, `linear`, etc.) is also rejected with `PersistFailed(Config)`.
- **`d2.<k>` storage rule**: for a scalar distance slot, the lower-triangular elements `j..n` of column $j$ (diagonal included) are packed in column order (packed position $p = jn - j(j-1)/2 + (i - j)$, length $n(n+1)/2$). For an ARD distance slot (header `shape` is `[dims, n(n+1)/2]`), a lower-triangular packed array of length $\text{packed\_len} = n(n+1)/2$ is concatenated for each dimension $m \in [0, \text{dims})$ in order (flat index `m * packed_len + p`, with the inner axis being `packed_len`). Therefore, when reading with NumPy or similar, interpret the two-dimensional column-major matrices (`x`, `l`) with `flat.reshape(meta["shape"], order="F")` and the distance tensor `d2.<k>` with `flat.reshape(meta["shape"], order="C")` (a one-dimensional scalar `d2.<k>` is identical under either `F` or `C`). For a `MixedPrecision` model, the retained high-precision `double` $d^2$ is written out and re-rounded to `float` on load.

```python
# Minimal Python / NumPy read rule (x, l are column-major; ARD d2.<k> is concatenated per-dimension blocks)
def reshape_tensor(name: str, flat: "np.ndarray", shape: list[int]) -> "np.ndarray":
    order = "C" if name.startswith("d2.") else "F"
    return flat.reshape(shape, order=order)
```
- **Issuing new slots**: on load, a fresh sequence of slot handles dedicated to the model is generated, so a `ScalarDistance` / `ArdDistance` from before saving cannot be used with the loaded model. Users obtain the new slots from `slots()` / `to_kernel()` to build the prediction supply.
- On load, the finiteness, non-negativity, zero diagonal, and shape of `d2.<k>` are all verified, and `InvalidDistance` is returned on violation.

---

## FMT-08 Error kinds and external-tool compatibility mapping

| Abnormal condition | `GprError` returned |
|---|---|
| File open/close, read, write, or `mmap` failure | `PersistFailed { PersistErrorKind::Io, reason }` |
| JSON syntax error, missing required key or a key of the wrong type, nesting exceeded (on load, or in the `config.json` that `save` would write) | `PersistFailed { PersistErrorKind::Config, reason }` |
| Missing tensor, shape / dtype mismatch, not 8-byte aligned, non-finite tensor | `PersistFailed { PersistErrorKind::Tensor, reason }` |
| Coordinate / distance / marker mismatch, `factor_kind != "llt"`, file specifying a Sparse/Online model | `PersistFailed { PersistErrorKind::WrongModel, reason }` |
| Unregistered custom `persist_id` | `PersistFailed { PersistErrorKind::UnregisteredId, reason }` |
| Empty, `"cxxgpr."`-prefixed, or non-UTF-8 `persist_id`, or duplicate registration | `PersistFailed { PersistErrorKind::InvalidPersistId, reason }` |
| `save` of a custom type that does not support `persist_id`, or whose state text must be stored as a JSON string and is not valid UTF-8 | `PersistFailed { PersistErrorKind::NotPersistable, reason }` |
| `format_version != 1` | `UnsupportedPersistVersion { found, supported: 1 }` |
| `d2.<k>` non-finite, negative, or non-zero diagonal | `InvalidDistance` |
| Invalid saved interval, Jitter, or kernel value | The `GprError` of the respective constructor / factory |
| `save_id` in `config.json` that is not 32 lowercase hexadecimal digits (`FMT-09`) | `PersistFailed { PersistErrorKind::Config, reason }` |
| Malformed `__metadata__` or `cxxgpr.save_id` in `model.safetensors`; pairing identifiers that differ, or an identifier in only one of the two files (`FMT-09`) | `PersistFailed { PersistErrorKind::Tensor, reason }` |

---

## FMT-09 Pairing identifier of `config.json` and `model.safetensors`

Status: specified; the implementation is roadmap row `REL-11`. Until that row is Done, `save` / `save_with_factor` write no identifier, `load` performs no pairing check, and the two files are paired only by the per-file checks of `FMT-02`.

`save` replaces the two files with two separate renames (`FMT-02`). When the process stops between them, or the second rename fails, the directory holds the new `model.safetensors` beside the old `config.json`. The per-file checks reject such a pair only when a tensor's shape or dtype, or the presence of a tensor, disagrees with `config.json`. Each save therefore writes one identifier into both files, and `load` rejects a pair whose identifiers differ.

- **Format**: the identifier is a string of exactly 32 lowercase hexadecimal digits (`[0-9a-f]{32}`). It encodes, first byte first, the first 16 bytes (128 bits) of the SHA-256 digest (FIPS 180-4) of the message $M$ defined below.
- **Content hash, not a random value**: the identifier is a function of the saved content only. It does not depend on the time, the process, the directory path, or a random source. Hence:
  - `save` / `save_with_factor` are deterministic: for two models whose saved state (every `config.json` value and every tensor) is bit-identical, the same version of `cxxgpr` writes byte-identical `config.json` and `model.safetensors`.
  - Two saves whose content differs get different identifiers, except with probability about $2^{-128}$ for a given pair of saves.
  - Two saves whose content is identical get the same identifier. A directory that mixes their files is a consistent pair of that content and loads.
- **Hashed message $M$**: the concatenation, in this order, of the parts below. `le64(v)` is `v` as an 8-byte little-endian unsigned integer.
  1. The 17 ASCII bytes `cxxgpr.save_id.v1` followed by one `0x00` byte.
  2. `le64(|C|)` and the bytes `C`, where `C` is `config.json` exactly as `save` writes it except that the `save_id` key is not serialized.
  3. `le64(k)`, where `k` is the number of tensors, then for each tensor in the order of its data offsets in `model.safetensors`: `le64` of the name's byte length and the name (UTF-8); `le64` of the dtype string's length and the dtype string (`F64` or `F32`); `le64` of the rank and `le64` of each dimension; `le64` of the tensor's byte length and its bytes exactly as stored in the data section.

  The safetensors header JSON, its padding, and `__metadata__` are not hashed: part 3 describes every tensor, and the padding carries no content. `config.json` is part of $M$ because a tensor-only hash cannot tell two saves apart that differ only in `config.json`: a `save` (no factor) after `refit` writes the same `x` and `y` with new hyperparameters.
- **Placement**:
  - `config.json`: the top-level key `save_id` (`FMT-03`).
  - `model.safetensors`: the header entry `"__metadata__": {"cxxgpr.save_id": "<identifier>"}` (safetensors defines `__metadata__` as a string-to-string map). The writer emits no other `__metadata__` entry.
- **Load check**: `LoadedGpr::load` and `LoadedDistanceGpr<C>::load` (`FMT-07`) check the pair in this order.
  1. `config.json` is parsed and validated (`FMT-01`, `FMT-03`). A `save_id` that is not a string of 32 lowercase hexadecimal digits gives `PersistFailed(Config)`.
  2. The `model.safetensors` header is parsed and validated (`FMT-04`). A `__metadata__` that is not a JSON object whose values are all strings, or a `cxxgpr.save_id` that is not a string of 32 lowercase hexadecimal digits, gives `PersistFailed(Tensor)`. Other `__metadata__` keys are ignored.
  3. If both files carry an identifier and the two are equal, loading continues. If neither carries one, loading continues without a pairing check. If they differ, or only one file carries an identifier, `load` returns `PersistFailed(Tensor)` with a reason stating that the two files come from different saves.
  4. Only then are the tensors read and checked (shape, dtype, finiteness, and `d2.<k>` by `FMT-07`). A pair from different saves is therefore reported by the pairing check even when its shapes also differ.

  `load` compares the two stored strings and does not recompute the hash. Recomputing it would require reproducing the `config.json` bytes exactly, which depends on the serializer's version and would reject pairs written by other versions or tools. The identifier detects files from different saves; it is not an integrity or authenticity check (`FMT-02` treats the directory as untrusted input, and a writer can always write matching identifiers).
- **Distance models**: `FittedDistanceGpr` saves carry the identifier exactly as coordinate models do. `C` includes the `distance` section, and part 3 includes every `d2.<k>` tensor (and no `x` for a distance-only model).
- **Format version and backward compatibility**: `FORMAT_VERSION` stays `1`; the identifier is optional in the schema.
  - A pair in which neither file carries an identifier (written before `REL-11`, or by an external tool) loads as before, without the pairing check.
  - A reader that predates the identifier ignores the `save_id` key (unknown top-level keys are ignored, `FMT-03`) and skips `__metadata__`, so it loads a pair with the identifier without checking it.
  - A pair in which only one file carries an identifier is rejected (step 3). An interrupted save leaves exactly this state when a writer that writes the identifier saves over a directory written without it, or the reverse.
- **External tools**: a tool that writes a pair writes the same identifier of the format above into both files, or writes it into neither. A tool that rewrites one file of a pair keeps the identifier only when the result is still a consistent pair.
- **Failure paths of `save`**: the identifier does not make the replacement of the pair atomic. After the second rename fails, `save` still returns `PersistFailed(Io)` (`FMT-02`), and the directory then fails to load by step 3 until the next successful save into it, unless the two saves wrote identical content.
