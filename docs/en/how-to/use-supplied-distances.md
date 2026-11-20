# How to evaluate kernels from supplied distances

This guide shows how to evaluate kernels from squared distances that you compute yourself, for example arc lengths on a sphere, graph distances or edit distances, instead of Euclidean distances between coordinate vectors. It also covers `DistanceCache`, which stores distances computed from coordinates so an optimizer does not recompute them.

Read {doc}`build-kernels` first. The symbols come from `cxxgpr/kernel/distance.hpp` and `cxxgpr/kernel/distance_cache.hpp`; signatures are in the [kernel API reference](../reference/api/kernel.md).

:::{warning}
Supplied-distance kernels are a **kernel-evaluation** API. The `Gpr` trainer accepts only a `KernelSpec`, so you cannot pass a `DistanceKernel` to `Gpr::fit`. Use `CompiledDistanceKernel` to build Gram matrices, cross-covariances and derivatives, and feed them to your own solver or to the building blocks in {doc}`use-low-level-building-blocks`.
:::

## When to use supplied distances

Use them when

- the natural metric is not Euclidean (geodesic, graph, edit or domain-specific distance),
- the distances already exist as a table and recomputing them from coordinates would be wasteful or impossible,
- you want one kernel to read several distance tables, for example one per feature group.

:::{warning}
A kernel is only a valid covariance if its Gram matrix is positive semi-definite. The built-in kernels guarantee this for Euclidean distances, not for an arbitrary metric. On a circle with arc length as the distance, the exponential kernel $\exp(-d/\ell)$ (`MaternKernel` with `OneHalf`) is valid, but the squared-exponential kernel of the arc length is not. Check that your metric suits the leaf you choose.
:::

## The pipeline

```{mermaid}
flowchart LR
    T["distance tables<br/>(double, column-major)"] --> S["DistanceSource<br/>bind_scalar / bind_ard / tidy"]
    K["DistanceKernel (mode tag)<br/>ScalarDistance, ArdDistance leaves"] -->|compile| C["CompiledDistanceKernel"]
    S --> P["prepare_square / prepare_cross"]
    C --> P
    P --> V["CheckedDistanceStore"]
    V --> E["apply / apply_cross / grad / hess / fill_diag"]
    C --> E
```

## Step 1: build the distance table

The table holds **squared** distances. The leaves take the square root where they need it. A table for $n$ points is an $n \times n$ column-major matrix with a zero diagonal. The example uses the arc length on the unit circle:

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [distance-table]
:end-before: // [end]
:dedent: 4
```

## Step 2: create a slot and a kernel

A **slot** names one distance table inside a kernel. `ScalarDistance` is a slot for one isotropic table. Its factories bind an isotropic leaf to the slot and return a `DistanceKernel<DistanceOnly>`: `rbf`, `matern`, `periodic`, `rational_quadratic`, and `custom` for a `CustomKernel`. `ConstantKernel` and `WhiteKernel` need no table; `DistanceKernel` converts from them implicitly.

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
:dedent: 4
```

This is $k(a, b) = 2\exp(-\operatorname{arc}(a, b)/1) + 0.1\,\delta$. `DistanceKernel` has the same parameter interface as `KernelSpec`: `num_params`, `get_params`, `set_params` (atomic), `bounds_into` and `parameter_bindings`. It is a cheap value type; copies share an immutable tree.

Each slot gets a process-unique `SlotId` (`value` returns its number; `operator==` and `operator<=>` compare ids). `slots` lists the slots a kernel reads, in order of first appearance, as `SlotDescriptor` values with the `id`, `ard_dims` (empty for scalar slots) and `at`, the index among slots of the same shape. `num_scalar_slots` and `num_ard_slots` count them. Reusing one `ScalarDistance` in several leaves makes them read the same table:

```{code-block} text
num_params = 3, scalar slots = 1, ARD slots = 0
```

## Step 3: validate the tables

`DistanceKernel::compile<T>` gives a `CompiledDistanceKernel<T>`. A `DistanceSource` collects the tables for one call: `bind_scalar(slot, d2)` takes a `MatrixRef<double>` and returns the source itself so calls chain. `prepare_square` validates a square training set and returns a `CheckedDistanceStore<T>`; `prepare_cross` does the same for an $n_1 \times n_2$ query-by-training table.

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [prepare]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
store: 5 x 5, square = true, requires_points = false
```

Validation checks slot coverage, shapes, finiteness, non-negativity and, for square tables, a zero diagonal and symmetry, then converts the values to `T`. The store reports `rows`, `cols` and `is_square`. It is copyable and movable, and the same store can be reused for any number of evaluations.

### Strict by default, repairable with `tidy`

Distances computed from floating-point data often carry rounding noise. `DistanceSource::tidy(rel_tol)` permits small violations relative to the largest entry: negative values down to $-\mathrm{rel\_tol}\cdot\max|d^2|$ are clamped to 0, a diagonal within tolerance is zeroed, and asymmetric pairs within tolerance are averaged. `tidy_tolerance` returns the setting, or `std::nullopt` while validation is strict.

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [tidy]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
strict: invalid squared distance in slot 0 at (0, 0): diagonal must be zero
tidy(1e-9): accepted, tidy_tolerance = 1e-09
```

| `GprErrorCode` | Cause |
|---|---|
| `InvalidDistance` | Non-finite, negative, asymmetric or nonzero-diagonal values beyond tolerance; the error carries the slot and entry |
| `DistanceSlot` | A slot of the kernel is missing, bound twice, or bound with the wrong shape kind |
| `UnsupportedKernelOperation` | The source binds a slot the kernel does not use |
| `EmptyInput`, `LengthMismatch`, `DimensionMismatch` | Empty tables, non-square or unequal shapes, wrong ARD table count |
| `InvalidConfig` | Negative or non-finite `tidy` tolerance |

## Step 4: evaluate

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [gram]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
K(0,0) = 2.1  K(1,0) = 1.21306  K(3,0) = 0.0995741  K(4,0) = 0.913896
```

Check by hand: the angles $0$ and $0.5$ are $0.5$ apart, so $K_{10} = 2e^{-0.5} = 1.21306$. The angles $0$ and $5.5$ are $2\pi - 5.5 = 0.783$ apart the short way round, so $K_{40} = 2e^{-0.783} = 0.9139$. The example asserts all of these.

`CompiledDistanceKernel` offers the same evaluation family as `CompiledKernel`, with the store in place of the distance matrix and an `std::optional<MatrixRef<T>>` for the coordinates, which is `std::nullopt` for a pure distance kernel:

| Task | Member |
|---|---|
| Training Gram | `apply` |
| Cross-covariance | `apply_cross`, with a store from `prepare_cross` |
| Prior variance | `fill_diag(n, x, out)`; needs no store |
| $\partial K/\partial\log\theta_p$ | `grad` |
| $\partial^2 K/\partial\log\theta_i\partial\log\theta_j$ | `hess` |

Each has an overload that takes a `NestedBuffers` for allocation-free use, and one that allocates. `make_nested_buffers`, `nested_depth`, `num_params`, `num_leaves`, `get_params`, `set_params`, `bounds_into`, `parameter_bindings`, `slots`, `kernel_exp` and `set_kernel_exp` behave as on `CompiledKernel`. `requires_points` tells you whether the kernel also reads coordinates.

Cross-covariances between query and training points use a table with one row per query. `prepare_cross` builds the store, and white noise contributes nothing:

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [cross]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
k(1.0, 0.0) = 0.735759 (hand 0.735759)
```

Parameter 1 is the log length scale, so `grad` with `param_idx = 1` returns the derivative of $2e^{-r/\ell}$ with respect to $\log\ell$, which is $2e^{-r/\ell}\,r/\ell$. The example checks one entry against that value:

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [gradient]
:end-before: // [end]
:dedent: 4
```

## ARD: one table per feature

An `ArdDistance` slot expects **one squared-difference table per feature**. `ArdDistance::create(dims)` rejects `dims == 0` with `EmptyInput`; `dims` returns the count and `slot` the id. Its factories `rbf_ard`, `matern_ard` and `rational_quadratic_ard` return `std::expected<DistanceKernel<DistanceOnly>, GprError>`, because a leaf whose length-scale count differs from `dims` is rejected with `DimensionMismatch`. Bind the tables with `bind_ard(slot, span of MatrixRef<double>)`.

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [ard]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
ARD K(1,0) = 0.367879 (hand 0.367879), dims = 2
```

## Mix supplied distances with coordinates

`DistanceKernel` takes a mode tag. `DistanceOnly` (the default) needs no coordinates; `WithPoints` may also contain coordinate leaves such as `LinearKernel` or ARD leaves from a `KernelSpec`. `DistanceModeConcept` constrains the tag to those two. `with_points` promotes a `DistanceOnly` kernel, and `operator+` and `operator*` combine two kernels of the same mode, or a `DistanceKernel<DistanceOnly>` with a `KernelSpec` in either order, giving a `WithPoints` kernel.

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [with-points]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
mixed requires_points = true, K(2,1) = 1.36788
```

Here $K_{21} = e^{-1} + 0.5\cdot(1\cdot2) = 1.36788$. Calling `apply` without coordinates on such a kernel fails with `UnsupportedKernelOperation`.

## Reuse distances computed from coordinates

For ordinary coordinate kernels the library can compute distances once and reuse them across optimizer iterations. `DistanceCache<T>` holds the dense $n \times n$ squared-distance matrix for isotropic leaves and a packed per-feature table for ARD leaves.

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [cache]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
cache: points = 3, dims = 2, has_dist = true, has_ard = true, packed size = 12
```

| Member | Purpose |
|---|---|
| `DistanceCache::build(x, cache_dist, cache_ard, policy)` | Static factory returning a populated cache or an error (`EmptyInput`, `NonFiniteInput`, `SizeOverflow`) |
| `ensure(x, cache_dist, cache_ard, policy)` | Refills an existing cache in place; clears a buffer when its flag is false or the policy is `Uncached` |
| `reset` | Empties the cache, keeping capacity |
| `policy`, `num_points`, `num_dims` | What the last `build` or `ensure` recorded |
| `has_dist`, `has_ard` | Which buffers are populated |
| `dist_view`, `dist_mut` | $n \times n$ view of the squared distances (empty if not cached) |
| `ard_packed`, `ard_packed_mut` | Span of $d \cdot n(n+1)/2$ entries |

The packed ARD layout stores $d$ blocks, one per feature, each holding the lower triangle column by column; the entry for row $i \ge j$ is at `j * (2n - j + 1) / 2 + (i - j)` inside its block. The example checks the entry $(s_1 - s_0)^2$ at its computed offset.

`DistanceCachePolicy` chooses between memory and speed. `Cached` precomputes and keeps the tables. `Uncached` leaves the buffers empty and computes distances from the coordinates on every call. `Gpr` exposes it as `with_distance_cache_policy` and through `FitPolicy`; see {doc}`configure-fit`. A cache is consumed by `CompiledKernel::apply_cached`, `grad_cached`, `hess_cached` and the leaf-level variants described in {doc}`build-kernels`.

## API summary

| Symbol | What it is |
|---|---|
| `DistanceOnly`, `WithPoints`, `DistanceModeConcept` | Mode tags of `DistanceKernel` and the concept that admits them |
| `SlotId` | Process-unique slot identifier: `value`, `operator==`, `operator<=>` |
| `SlotDescriptor` | `id`, `ard_dims`, `at` |
| `ScalarDistance` | Isotropic slot: `slot`, `rbf`, `matern`, `periodic`, `rational_quadratic`, `custom` |
| `ArdDistance` | ARD slot: `create`, `slot`, `dims`, `rbf_ard`, `matern_ard`, `rational_quadratic_ard` |
| `DistanceKernel<Mode>` | Expression tree: `with_points`, `num_params`, `get_params`, `set_params`, `bounds_into`, `parameter_bindings`, `slots`, `num_scalar_slots`, `num_ard_slots`, `compile`, `operator+`, `operator*` |
| `DistanceSource` | `bind_scalar`, `bind_ard`, `tidy`, `tidy_tolerance` |
| `CheckedDistanceStore<T>` | Validated tables: `rows`, `cols`, `is_square` |
| `CompiledDistanceKernel<T>` | `num_params`, `num_leaves`, `requires_points`, `nested_depth`, `kernel_exp`, `set_kernel_exp`, `make_nested_buffers`, `get_params`, `set_params`, `bounds_into`, `parameter_bindings`, `slots`, `prepare_square`, `prepare_cross`, `apply`, `apply_cross`, `fill_diag`, `grad`, `hess` |
| `DistanceCachePolicy` | `Cached`, `Uncached` |
| `DistanceCache<T>` | `build`, `ensure`, `reset`, `policy`, `num_points`, `num_dims`, `has_dist`, `has_ard`, `dist_view`, `dist_mut`, `ard_packed`, `ard_packed_mut` |
