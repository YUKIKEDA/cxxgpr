# How to prepare data for cxxgpr

This guide shows how to lay out your training and query data the way cxxgpr expects it, how to use the non-owning matrix views, how to validate sizes and values yourself with the checked helpers, and how to read the result types and build metadata. It covers the whole core module: `cxxgpr/core/views.hpp`, `cxxgpr/core/checked.hpp` and `cxxgpr/core/build_info.hpp`.

## Lay out matrices in column-major order

Every matrix argument in the library is **column-major** (Fortran order): the elements of one column are adjacent in memory, and column $j$ starts at offset $j \cdot \text{col\_stride}$. For a training matrix with $n$ samples (rows) and $d$ features (columns), the buffer holds all $n$ values of feature 0, then all $n$ values of feature 1, and so on. Element $(i, j)$ is at index $j \cdot n + i$. Targets are a plain vector of $n$ values.

If your data are stored one sample per row, copy them into a column-major buffer:

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [layout]
:end-before: // [end]
:dedent: 4
```

Its output confirms that the three values of feature 0 come first:

```{code-block} text
x(1, 1) = 20, buffer = 1 2 3 | 10 20 30
```

:::{warning}
If you pass row-major data by mistake, nothing can detect it: the buffer has the right number of elements and the values are finite. You get a valid model of the wrong function. When in doubt, print `x(i, j)` for a corner you know.
:::

For one feature, the column-major buffer and the vector of values are the same thing.

## Pass data to the library

Every model entry point comes in two forms that differ only in how the matrix is passed:

| Form | Example | Use when |
|---|---|---|
| Span plus shape | `fit(x, n_rows, n_cols, y)` | You own a contiguous buffer such as `std::vector<double>` or `std::array<double, N>`. |
| `MatrixRef<double>` | `fit(MatrixRef<double>(...), y)` | The matrix is already a view over memory you own. |

The span form assumes a tightly packed matrix with column stride equal to `n_rows`. Both forms return `std::expected`; for example a target vector whose length differs from the row count gives `GprErrorCode::LengthMismatch`, empty data gives `GprErrorCode::EmptyInput`, and NaN or infinity gives `GprErrorCode::NonFiniteInput`.

:::{note}
The span form checks that `span.size() == n_rows * n_cols`. A span of any other length gives `GprErrorCode::LengthMismatch`, and dimensions whose product overflows give `GprErrorCode::SizeOverflow`, before any element is read. A `MatrixRef` is different: it only holds a pointer, so the library cannot check that the memory behind it really holds `n_rows * n_cols` elements. Getting the extents right for a `MatrixRef` is your responsibility; `validate_matrix` (below) checks a span against the shape you intend to pass.
:::

## Use `MatrixRef` and `MatrixMut`

`MatrixRef<T>` is a non-owning, read-only view of a column-major matrix. `MatrixMut<T>` is its mutable counterpart. Both are cheap to copy (a pointer and three sizes) and never own memory, so the buffer must outlive the view. They keep Eigen and the library's internal storage out of the public headers.

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [views]
:end-before: // [end]
:dedent: 4
```

Output:

```{code-block} text
a(2, 1) = 6, rows = 3, cols = 2, col_stride = 3, contiguous = true
column 1 has 3 elements, first = 4
block(0, 0) = 2, block(1, 1) = 6, contiguous = false
after write: a(0, 0) = 10, via MatrixMut::as_ref = 10, via conversion = 10
default view: is_empty = true, is_square = true
```

### Constructing views

Both classes have the same four constructors:

| Constructor | Meaning |
|---|---|
| `MatrixRef()` / `MatrixMut()` | Empty view: null pointer, zero extents. |
| `(data, rows, cols, col_stride)` | General strided view over a pointer; requires `col_stride >= rows`. Use it for a sub-block, as `block` does above. |
| `(data, rows, cols)` | Tightly packed view; `col_stride` equals `rows`. |
| `(span, rows, cols)` | Tightly packed view over a span; you ensure `span.size() >= rows * cols`. |

`MatrixRef<T>` takes `const T*` or `std::span<const T>`; `MatrixMut<T>` takes `T*` or `std::span<T>`. The constructors do no validation.

### Accessors

| Member | Returns |
|---|---|
| `data()` | Pointer to element $(0, 0)$; null for an empty view. |
| `rows()`, `cols()` | The extents. |
| `col_stride()` | Elements between the starts of adjacent columns; equals `rows()` when packed. |
| `is_empty()` | True when `rows() == 0` or `cols() == 0`. |
| `is_square()` | True when `rows() == cols()`. A $0 \times 0$ view counts as square. |
| `is_contiguous()` | True when `col_stride() == rows()`, i.e. no padding between columns. |
| `operator()(row, col)` | Reference to one element. **No bounds check**; you must keep `row < rows()` and `col < cols()`. |
| `col(j)` | Contiguous span over column $j$ (read-only for `MatrixRef`, writable for `MatrixMut`). Returns an empty span for a null view, zero rows, or `j >= cols()`. |
| `as_ref()` | `MatrixMut` only: the same region as a `MatrixRef`. |
| `operator MatrixRef<T>` | `MatrixMut` only: implicit conversion, so a mutable view can be passed wherever a read-only one is expected. |

`MatrixMut`'s `data()`, `operator()` and `col()` are `const` member functions that return mutable access: a `const MatrixMut` is a *handle* to writable memory, like a `std::span<T>`.

:::{warning}
`operator()` is unchecked on purpose, because it sits in inner loops. Use `col(j)`, which does guard the column index, when the index comes from outside.
:::

## Choose the result and option types

The prediction functions of `FittedGpr` and `LoadedGpr` use these types from `views.hpp`:

| Type | Purpose |
|---|---|
| `VarianceKind` | Enumeration: `VarianceKind::Latent` reports the variance of the noise-free function $f_*$; `VarianceKind::Observation` adds the observation noise $\sigma_n^2$ (the default). |
| `PredictOptions` | Struct with the single field `variance_kind` (default `Observation`). Comparable with `operator==`. Pass it as the last argument of `predict`, `loo_predict` and related functions. |
| `Prediction<T>` | Result of pointwise prediction: `mean` and `variance` vectors (one entry per query), plus the `variance_kind` they were computed for. Comparable with `operator==`. |
| `PredictiveCovariance<T>` | Result of joint prediction: `mean`, a column-major $m \times m$ `covariance` matrix (element $(r, c)$ at `c * n_points + r`), `n_points`, and `variance_kind`. Comparable with `operator==`. |
| `Triangle` | Enumeration `Lower`, `Upper`, `Full`: which part of a symmetric matrix a routine reads or writes. Used by kernel and distance routines (also visible as `kernel::Triangle`). The bounds include the diagonal. |

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [results]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
default options are Observation: true, options equal: false, prediction == itself: true, Cov(0, 1) = 0.01
```

`T` is `double` or `float`, depending on the precision policy of the model; see {doc}`predict-and-sample` for how to obtain these results and {doc}`../explanation/gaussian-processes` for the difference between latent and observation variance.

## Guard against overflow with the checked helpers

Index arithmetic on large matrices can wrap around silently in `std::size_t`. The checked helpers return `std::expected<std::size_t, GprError>` and report `GprErrorCode::SizeOverflow` instead.

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [checked]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
2 + 3 = 5, 1000 * 1000 = 1000000, 30 x 2 -> 60 elements, packed_len(4) = 10
checked_mul(max, 2): size overflows std::size_t
```

| Function | Result |
|---|---|
| `checked_add(a, b)` | $a + b$, or `SizeOverflow`. |
| `checked_mul(a, b)` | $a \cdot b$, or `SizeOverflow`. Returns 0 at once if either factor is 0. |
| `column_major_len(n_rows, n_cols)` | $n_\text{rows} \cdot n_\text{cols}$ for a non-empty matrix. `EmptyInput` if either extent is 0, `SizeOverflow` on overflow. |
| `packed_len(n)` | The packed lower-triangle length $n(n+1)/2$, computed without false intermediate overflow. |

## Validate inputs yourself

Call these validators to check data before you start a long job, for example to reject a bad data file with a precise message. All return `std::expected<void, GprError>` and are marked `[[nodiscard]]`.

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [validate]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
require_nonempty(0): input is empty
require_count(3, 4): length mismatch: expected 4 y, got 3
require_finite(bad): input contains a non-finite value (NaN/Inf)
validate_matrix(2 x 3 over 4 values): length mismatch: expected 6 X values, got 4
validate_training(y too short): length mismatch: expected 4 targets, got 3
validate_training(wrong d): input dimension mismatch: X.ncols()=1, expected 2
validate_training(min_points = 5): insufficient data: n=4, at least 5 points required
validate_distance_targets(n = 3): length mismatch: expected 3 targets, got 4
validate_query(d = 2 expected): input dimension mismatch: X.ncols()=1, expected 2
```

| Function | Checks | Error codes, in precedence order |
|---|---|---|
| `require_nonempty(n)` | `n > 0` | `EmptyInput` |
| `require_count(actual, expected, what)` | `actual == expected`; `what` names the quantity in the message | `LengthMismatch` |
| `require_finite(values)` | No NaN or infinity in a `std::span<const double>` or `std::span<const float>` (subnormals are allowed; an empty span passes) | `NonFiniteInput` |
| `validate_matrix(x, n_rows, n_cols, name)` | Non-empty, finite, `x.size() == n_rows * n_cols`; `name` defaults to `"X"` | `EmptyInput`, `SizeOverflow`, `LengthMismatch`, `NonFiniteInput` |
| `validate_training(x, n_rows, n_cols, y, expected_d, min_points)` | A training set: matrix and `y` sizes, optional required dimension `expected_d` (a `std::optional<std::size_t>`, default none), and at least `min_points` rows (default 1). Also checks that the $n \times n$ Gram matrix size does not overflow. | `EmptyInput`, `SizeOverflow`, `LengthMismatch`, `DimensionMismatch`, `NonFiniteInput`, `InsufficientData` |
| `validate_distance_targets(n, y, min_points)` | Targets for training from supplied distances, where no feature matrix exists | `EmptyInput`, `SizeOverflow`, `LengthMismatch`, `NonFiniteInput`, `InsufficientData` |
| `validate_query(xs, n_rows, n_cols, expected_d)` | Query matrix against the fitted model's feature count | `EmptyInput`, `SizeOverflow`, `LengthMismatch`, `DimensionMismatch`, `NonFiniteInput` |

When several conditions fail at once, the error listed first wins, so a given input always yields the same error. See {doc}`handle-errors` for how to react to each code.

## Read build information at runtime

`build_info.hpp` lets a program, a log, or a bug report identify the exact library build.

| Symbol | Meaning |
|---|---|
| `DependencyVersion` | A struct with `name` and `version` (`major.minor.patch`) of one linked third-party library. |
| `BuildInfo` | The provenance record described below. |
| `build_info()` | Returns a `BuildInfo` for the compiled library. |
| `build_info_json()` | Returns the same data as indented JSON with a trailing newline. |

The fields of `BuildInfo`:

| Field | Meaning |
|---|---|
| `schema` | Version of the record layout, such as `cxxgpr.build-info/1`. |
| `version` | Semantic version of the library. |
| `git_commit` | Commit hash recorded at build time. |
| `uncommitted_changes` | True if the working tree had local modifications when it was built. |
| `compiler` | Compiler identification string. |
| `platform` | Operating system and architecture. |
| `platform_key` | Normalised `os-arch-compiler` key used to name per-platform baselines. |
| `configuration` | Build configuration, such as `Release`. |
| `floating_point` | Floating-point mode, including contraction and the instruction set. |
| `dependencies` | A `std::vector<DependencyVersion>` of linked third-party libraries. |

```{literalinclude} /../examples/build_info.cpp
:language: cpp
:linenos:
```

See {doc}`install-and-link` for a sample of its output.

## API summary

| Symbol | Header | Section |
|---|---|---|
| `MatrixRef`, `MatrixMut` | `core/views.hpp` | Use `MatrixRef` and `MatrixMut` |
| `Triangle`, `VarianceKind`, `PredictOptions`, `Prediction`, `PredictiveCovariance` | `core/views.hpp` | Choose the result and option types |
| `checked_add`, `checked_mul`, `column_major_len`, `packed_len` | `core/checked.hpp` | Guard against overflow |
| `require_nonempty`, `require_count`, `require_finite`, `validate_matrix`, `validate_training`, `validate_distance_targets`, `validate_query` | `core/checked.hpp` | Validate inputs yourself |
| `DependencyVersion`, `BuildInfo`, `build_info`, `build_info_json` | `core/build_info.hpp` | Read build information at runtime |

The generated reference for these headers is in [the core API reference](../reference/api/core.md).
