# Model file format

This page specifies the on-disk format that `cxxgpr::persist::save`, `cxxgpr::persist::save_with_factor` and `cxxgpr::persist::LoadedGpr::load` read and write for exact Gaussian process models. Use it to inspect a saved model with other tools, to write a reader in another language, or to judge whether two versions of the library can exchange files. For the tasks of saving and loading, see {doc}`../how-to/persist-models`.

## Directory layout

A model is one directory that contains exactly two files.

| File | Content |
|---|---|
| `config.json` | UTF-8 JSON: model structure, precision, policies, hyperparameters with their intervals, transform state |
| `model.safetensors` | binary tensors in the [safetensors](https://github.com/huggingface/safetensors) container: training data, and optionally the Cholesky factor and weights |

During a save, the files `config.json.tmp` and `model.safetensors.tmp` may exist briefly in the same directory. A completed save leaves only the two files above.

## Version and compatibility

| Item | Value |
|---|---|
| Format version | `1` (constant `cxxgpr::persist::FORMAT_VERSION`) |
| Version field | `format_version` in `config.json`, an integer |
| Reader behaviour | `format_version` other than `1`: `GprErrorCode::UnsupportedPersistVersion`, carrying the found and the supported version |
| Migration | none; there is no conversion between versions |
| Unknown top-level keys | ignored on load |
| Writer behaviour | a file is written by renaming a complete temporary file onto the final name |

Models of other families are rejected with `PersistErrorKind::WrongModel`: sparse and online models (a `model` key), and factorizations other than `"llt"` (`factor_kind`). A model trained on supplied squared distances carries a `distance` key and is read by `LoadedDistanceGpr`, not by `LoadedGpr`.

## `config.json`

The file is a single JSON object. The writer emits it with two-space indentation and sorted keys; a reader must accept any valid JSON layout.

### Top-level keys

| Key | Type | Required | Value on load when absent | Meaning |
|---|---|---|---|---|
| `format_version` | integer | yes | — | format version, `1` |
| `n` | integer, $> 0$ | yes | — | number of training points |
| `d` | integer, $> 0$ (0 for a distance-only model) | yes | — | number of features |
| `has_factor` | boolean | yes | — | `true` if `model.safetensors` contains `l` and `alpha` |
| `factor_kind` | string | yes | — | must be `"llt"` (Cholesky $LL^\top$) |
| `factor_jitter` | number, $\ge 0$ | no | `0.0` | diagonal jitter accepted when the model was factored |
| `precision` | string | no | `"double"` | `"double"`, `"single"` or `"mixed"` |
| `residual` | string | no | `"promote_storage"` | for `"mixed"` only: `"promote_storage"` or `"reevaluate_kernel"` |
| `math` | string | no | `"accurate"` | `"accurate"` or `"fast_approx"` (`KernelExp`) |
| `distance_cache` | string | no | `"always"` | `"always"` or `"never"` (`DistanceCachePolicy`) |
| `kernel` | object | yes | — | the kernel tree |
| `likelihood` | object | yes | — | `{"noise_variance": <bounded number>}` |
| `jitter` | object | yes | — | the Cholesky jitter policy |
| `x_unfitted` | transform | yes | — | input transform configuration before fitting |
| `x_transform` | transform | yes | — | fitted input transform |
| `y_unfitted` | transform | yes | — | target transform configuration before fitting |
| `y_transform` | transform | yes | — | fitted target transform |
| `distance` | object | only for a supplied-distance model | — | the mode and the distance slots, see below |
| `save_id` | string (32 lowercase hex digits) | no | absent (no pairing check) | content-hash pairing identifier matching `cxxgpr.save_id` in `model.safetensors` |

The writer omits `residual` unless `precision` is `"mixed"`. The `precision` and `residual` pair identifies the model's `PersistPrecisionKind`:

| `precision` | `residual` | `PersistPrecisionKind` | Policy |
|---|---|---|---|
| `"double"` | — | `Double` | `DoublePrecision` |
| `"single"` | — | `Single` | `SinglePrecision` |
| `"mixed"` | `"promote_storage"` | `MixedPromote` | `MixedPrecision<PromoteStorage>` |
| `"mixed"` | `"reevaluate_kernel"` | `MixedReevaluate` | `MixedPrecision<ReevaluateKernel>` |

Floating-point numbers are written with enough digits to read back the exact value. A number that is an integer value is still written as a JSON float (`0.0`, `100000.0`).

### Bounded numbers

Every hyperparameter is stored as its **value in user units** (not its logarithm) together with its open interval:

```json
{ "value": 1.1164913909984453, "lo": 1e-05, "hi": 100000.0 }
```

Constraint: $0 < lo < value < hi$.

### Kernel tree

A kernel is a JSON object with a single key naming the node type.

| Node | Body |
|---|---|
| `"sum"`, `"product"` | `{"left": <kernel>, "right": <kernel>}` |
| `"rbf"` | `{"length_scale": <bounded number>}` |
| `"rbf_ard"` | `{"length_scales": [<bounded number>, ...]}` |
| `"matern"` | `{"nu": "1/2" \| "3/2" \| "5/2", "length_scale": <bounded number>}` |
| `"matern_ard"` | `{"nu": ..., "length_scales": [<bounded number>, ...]}` |
| `"periodic"` | `{"length_scale": <bounded number>, "period": <bounded number>}` |
| `"rq"` | `{"length_scale": <bounded number>, "alpha": <bounded number>}` |
| `"rq_ard"` | `{"length_scales": [<bounded number>, ...], "alpha": <bounded number>}` |
| `"constant"` | `{"value": <bounded number>}` |
| `"linear"` | `{"variance": <bounded number>}` |
| `"white"` | `{"variance": <bounded number>}` |
| `"custom"` | `{"persist_id": <string>, "state": <JSON>}` |

The order of the parameters in the model's log-parameter vector is the depth-first, left-to-right order of the leaves, with each leaf's parameters in the order shown. The nesting depth of any structure (kernel, pipeline, columnwise) must not exceed 128 levels.

### Likelihood and jitter

`likelihood` holds the noise variance as a bounded number: `{"noise_variance": {"value": ..., "lo": ..., "hi": ...}}`.

`jitter` holds one of two policies:

| Policy | Body |
|---|---|
| Fixed offset | `{"fixed": {"jitter": <number>}}` |
| Adaptive | `{"adaptive": {"initial": <number>, "multiplier": <number>, "max_retries": <integer>, "max_jitter": <number>}}` |

The values pass the same validation as the `JitterPolicy` constructors.

### Transforms

A transform is the string `"identity"` or an object with one key.

| Node | Unfitted body (`x_unfitted`, `y_unfitted`) | Fitted body (`x_transform`, `y_transform`) |
|---|---|---|
| `"identity"` (a string, not an object) | — | — |
| `"standardize"` | `{}` | input: `{"mean": [...], "std": [...]}`; target: `{"mean": <number>, "std": <number>}` |
| `"min_max"` | `{"range_lo": <number>, "range_hi": <number>}` | input: `{"data_min": [...], "data_max": [...], "range_lo": ..., "range_hi": ...}`; target: the same with scalars |
| `"pipeline"` | `{"steps": [<transform>, ...]}` | `{"steps": [<transform>, ...]}` |
| `"columnwise"` (input only) | `{"maps": [<transform>, ...]}` | `{"maps": [<transform>, ...]}` |
| `"custom"` | `{"persist_id": <string>, "state": <JSON>}` | `{"persist_id": <string>, "state": <JSON>}` |

The unfitted and fitted transforms form two separate registrations for custom nodes: the unfitted `persist_id` goes to a factory registered with `register_unfitted_input` or `register_unfitted_target`, and the fitted one to `register_fitted_input` or `register_fitted_target`.

A custom node's `state` holds the text returned by the component's `persist_state_json`. Text that parses as a JSON object, array, number, `true`, `false` or `null` is stored as that JSON value, and the factory receives its compact re-serialization, which is not the original text in general: numbers are rewritten in the shortest form that reads back to the same `double` (`2.500000` → `2.5`, `1e2` → `100.0`, `-0` → `0`, `12345678901234567890123` → `1.2345678901234568e+22`), whitespace outside strings is dropped, object keys are sorted, and of duplicate keys only the last is kept. Every other text, including text that is itself a JSON string literal such as `"abc"`, is stored as a JSON string, and the factory receives the original text byte for byte, quotes included. Such text must be valid UTF-8; saving text that is not gives `PersistErrorKind::NotPersistable`.

### Supplied-distance models

A model trained on supplied squared distances (`FittedDistanceGpr`) has the same sections as above, with these differences.

```json
"distance": {
  "points": "distance_only",
  "slots": [ { "kind": "scalar" }, { "kind": "ard", "dims": 2 } ]
}
```

- `distance.points` is `"distance_only"` (then `d` is `0`) or `"with_points"` (then `d > 0` and `x` is stored). The loader of the other mode refuses the file with `PersistErrorKind::WrongModel`.
- `distance.slots` lists the distance slots in the depth-first order of first appearance in the kernel. A slot is `{"kind": "scalar"}` or `{"kind": "ard", "dims": <integer>}`.
- In the kernel tree, a leaf that reads a distance slot is `{"distance": {"slot": <index into slots>, "leaf": <kernel leaf>}}`. The slot numbers appear in the order of `distance.slots`: walking the tree depth first, left to right, the first new number is `0`, the next new one `1`, and so on (a number repeats when two leaves read the same slot). Another numbering is rejected with `PersistErrorKind::Config`, because `d2.<k>` belongs to slot `k`. A scalar slot takes `rbf`, `matern`, `periodic`, `rq` or `custom`; an ARD slot takes `rbf_ard`, `matern_ard` or `rq_ard` with `dims` length scales. Any other leaf is a coordinate leaf written as in the table above; a distance-only model may contain only `constant` and `white` coordinate leaves.
- `distance_cache` is written as `"always"` and is not used.
- For `"distance_only"`, both `x_unfitted` and `x_transform` are `"identity"`; anything else is rejected with `PersistErrorKind::Config`.

**Distance tensors.** For slot $k$, the tensor `d2.<k>` holds the lower triangle, diagonal included, of the repaired training distances. The packed position of the entry $(i, j)$ with $i \ge j$ is $p = j n - j(j-1)/2 + (i - j)$, so the packed length is $n(n+1)/2$. An ARD slot concatenates one packed array per dimension, in order: flat index `m * packed_len + p` for dimension `m`. Unlike `x` and `l`, `d2.<k>` is therefore read with `reshape(shape, order="C")` (for a scalar slot, which is one-dimensional, the order makes no difference). A model in `"mixed"` precision stores `F32` tables, the precision it predicts with.

On load, the stored tables go through the same validation as tables you supply: finite, non-negative and zero on the diagonal. A violation gives `InvalidDistance`; a non-finite element is rejected earlier as a corrupt tensor.

### Example

A model with kernel `Constant * RbfArd` (24 training points, one feature), `StandardizeInput` and `StandardizeTarget`, saved with `save_with_factor` in double precision:

```json
{
  "d": 1,
  "distance_cache": "always",
  "factor_jitter": 0.0,
  "factor_kind": "llt",
  "format_version": 1,
  "has_factor": true,
  "jitter": {
    "fixed": {
      "jitter": 0.0
    }
  },
  "kernel": {
    "product": {
      "left": {
        "constant": {
          "value": {
            "hi": 100000.0,
            "lo": 1e-05,
            "value": 1.9301390140085495
          }
        }
      },
      "right": {
        "rbf_ard": {
          "length_scales": [
            {
              "hi": 100000.0,
              "lo": 1e-05,
              "value": 1.1164913909984453
            }
          ]
        }
      }
    }
  },
  "likelihood": {
    "noise_variance": {
      "hi": 100000.0,
      "lo": 1e-05,
      "value": 0.0017863553180453662
    }
  },
  "math": "accurate",
  "n": 24,
  "precision": "double",
  "x_transform": {
    "standardize": {
      "mean": [
        0.5
      ],
      "std": [
        0.300964632714423
      ]
    }
  },
  "x_unfitted": {
    "standardize": {}
  },
  "y_transform": {
    "standardize": {
      "mean": 10.373560792920012,
      "std": 2.136314909314304
    }
  },
  "y_unfitted": {
    "standardize": {}
  }
}
```

## `model.safetensors`

The file follows the safetensors specification:

| Offset | Size | Content |
|---|---|---|
| 0 | 8 bytes | $H$, the length of the header in bytes, as a little-endian `uint64` |
| 8 | $H$ bytes | a JSON object mapping each tensor name to `{"dtype", "shape", "data_offsets"}` |
| $8 + H$ | rest | the tensor bytes; `data_offsets` are `[begin, end)` relative to this position |

The writer pads the header so that $8 + H$ is a multiple of 8, and starts every tensor at an offset that is a multiple of 8. `__metadata__` holds `{"cxxgpr.save_id": "<32 lowercase hex digits>"}` matching `save_id` in `config.json`. All numbers are little-endian IEEE 754.

### Tensors

| Name | `shape` | `dtype` | Present when | Content |
|---|---|---|---|---|
| `x` | `[n, d]` | `F64` | always (omitted for a distance-only model, where $d = 0$) | training inputs before the input transform |
| `y` | `[n]` | `F64` | always | training targets before the target transform |
| `l` | `[n, n]` | `double`: `F64`; `single`, `mixed`: `F32` | `has_factor` is `true` | lower-triangular Cholesky factor $L$ of $K + (\sigma_n^2 + j) I$, with the upper triangle zero-filled |
| `alpha` | `[n]` | `single`: `F32`; `double`, `mixed`: `F64` | `has_factor` is `true` | weights $\alpha = (K + (\sigma_n^2 + j) I)^{-1}\mathbf{y}'$ in transformed target space, where $j$ is `factor_jitter` |
| `d2.<k>` | scalar slot: `[n(n+1)/2]`; ARD slot: `[dims, n(n+1)/2]` | `double`: `F64`; `single`, `mixed`: `F32` | supplied-distance models, one per slot $k$ | the lower triangle of the validated training distances of slot $k$ (see below) |

Every element of every tensor must be finite.

### Layout and byte order

Matrices `x` and `l` are stored **column-major** (Fortran order): element $(i, j)$ is at flat index $j \cdot n + i$ (`x` has $n$ rows and $d$ columns; `l` is $n$ by $n$). A one-dimensional tensor is a plain array.

A model with six training points and one feature, saved with `save_with_factor` in double precision, has this header (shown formatted; the file stores it without extra whitespace):

```json
{
  "alpha": { "dtype": "F64", "shape": [6],    "data_offsets": [384, 432] },
  "l":     { "dtype": "F64", "shape": [6, 6], "data_offsets": [96, 384] },
  "x":     { "dtype": "F64", "shape": [6, 1], "data_offsets": [0, 48] },
  "y":     { "dtype": "F64", "shape": [6],    "data_offsets": [48, 96] }
}
```

In a `"mixed"` model the same header lists `l` as `F32` (`data_offsets` spanning $4 n^2$ bytes) and `alpha` as `F64`.

### Reading with Python

```python
import json, struct
import numpy as np

def read_safetensors(path):
    raw = open(path, "rb").read()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    data = memoryview(raw)[8 + header_len :]
    dtypes = {"F64": "<f8", "F32": "<f4"}
    tensors = {}
    for name, meta in header.items():
        lo, hi = meta["data_offsets"]
        flat = np.frombuffer(data[lo:hi], dtype=dtypes[meta["dtype"]])
        tensors[name] = flat.reshape(meta["shape"], order="F")  # column-major
    return tensors
```

`order="F"` restores `x` and `l` to their stored shape and meaning bit for bit; for the one-dimensional tensors `y` and `alpha` it makes no difference.

## Loading behaviour

`LoadedGpr::load` and `LoadedDistanceGpr::load` treat the directory as untrusted input and perform the following checks. `LoadedGpr::load` refuses a file with a `distance` key, and `LoadedDistanceGpr::load` refuses a file without one or with the other `distance.points`, both with `PersistErrorKind::WrongModel`.

1. Both files open and read. Failure: `PersistErrorKind::Io`.
2. The JSON depth does not exceed 128, the text parses, the root is an object, and no `model` key is present. A `model` key or a `distance` key: `PersistErrorKind::WrongModel`. Other syntax or schema errors: `PersistErrorKind::Config`.
3. `format_version` is present and an integer. A value other than `1`: `GprErrorCode::UnsupportedPersistVersion`.
4. `factor_kind` is `"llt"`. Otherwise: `PersistErrorKind::WrongModel`.
5. Each tensor header entry has a known `dtype` (`F64` or `F32`), the expected shape, an aligned start, and a byte length equal to the product of the shape and the element size. Failure: `PersistErrorKind::Tensor`.
6. Every tensor element is finite. Failure: `PersistErrorKind::Tensor`.
7. Hyperparameters, intervals, jitter and transform statistics pass the checks of their constructors. Failure: the error of that constructor, for example `GprErrorCode::InvalidHyperparameter` or `GprErrorCode::InvalidInterval`.
8. Custom nodes are looked up in the registry. An unknown `persist_id`: `PersistErrorKind::UnregisteredId`.
9. If `has_factor` is `false`, the factorization is recomputed from the stored data and hyperparameters. Failure: `GprErrorCode::CholeskyFailed`.

With `has_factor == true`, the loader uses `l` and `alpha` as stored and does not refactor. For `double` precision it binds `l` directly to a read-only memory map; every other `l` is copied into memory.

## Errors

| Condition | Error |
|---|---|
| file cannot be opened, read, written or mapped | `PersistFailed`, `PersistErrorKind::Io` |
| invalid JSON, missing or mistyped key, nesting deeper than 128 (on load, or in the `config.json` that saving would write) | `PersistFailed`, `PersistErrorKind::Config` |
| missing tensor, wrong shape or `dtype`, misaligned data, non-finite element | `PersistFailed`, `PersistErrorKind::Tensor` |
| `model` key, `distance` key in `LoadedGpr`, missing `distance` key or other `distance.points` in `LoadedDistanceGpr`, `factor_kind` other than `"llt"`, or a precision mismatch in `into_fitted` | `PersistFailed`, `PersistErrorKind::WrongModel` |
| `persist_id` in the file has no registered factory | `PersistFailed`, `PersistErrorKind::UnregisteredId` |
| `persist_id` empty, starting with `cxxgpr.` or not valid UTF-8, or registered twice | `PersistFailed`, `PersistErrorKind::InvalidPersistId` |
| saving a custom component without `persist_id`, or with state stored as a string that is not valid UTF-8 | `PersistFailed`, `PersistErrorKind::NotPersistable` |
| a stored distance table with a negative entry or a nonzero diagonal | `InvalidDistance` |
| `format_version` other than `1` | `UnsupportedPersistVersion` with `found_version` and `supported_version` |
