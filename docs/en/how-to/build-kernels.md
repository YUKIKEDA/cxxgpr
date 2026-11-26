# How to build and evaluate kernels

This guide shows how to create the ten built-in kernel leaves, combine them with `+` and `*`, compile the result, and evaluate Gram matrices, cross-covariances and derivatives. For what the kernels mean and how to pick one, see {doc}`../explanation/kernels`. To write your own leaf, see {doc}`write-custom-kernel`.

All symbols here come from `cxxgpr/kernel/kernel.hpp`. The generated reference is in the [kernel API reference](../reference/api/kernel.md).

## Choose a leaf

| Need | Leaf | Notes |
|---|---|---|
| Smooth functions, one length scale | `RbfKernel` | Infinitely differentiable; the usual first choice |
| Smooth functions, one length scale per feature | `RbfArdKernel` | Learns which features matter |
| Rougher functions | `MaternKernel` with `MaternNu::OneHalf` or `ThreeHalves` | `FiveHalves` is a common default |
| Rough functions, per-feature length scales | `MaternArdKernel` | |
| Repeating pattern | `PeriodicKernel` | Multiply with a decay kernel for drifting cycles |
| Several length scales at once | `RationalQuadraticKernel`, `RationalQuadraticArdKernel` | Scale mixture of `RbfKernel` |
| Output scale (signal variance) | `ConstantKernel` | Used as a factor: `ConstantKernel * RbfKernel` |
| Linear trend | `LinearKernel` | Added to a stationary kernel |
| Independent measurement noise inside the kernel | `WhiteKernel` | `Gpr` already models noise; see the warning below |

## Create the leaves

Every leaf has a static `create` that validates its arguments and returns `std::expected<Leaf, GprError>`. The snippet below uses a small helper `unwrap` (shown in the full example) that stops on an error. In your own code, test the result with `if (!result)`.

```{literalinclude} /../examples/kernels_tour.cpp
:language: cpp
:start-after: // [create-leaves]
:end-before: // [end]
:dedent: 4
```

The two ARD leaves take a `std::span<const double>` with one length scale per feature, so a `std::array` or `std::vector` of `double` converts directly. `MaternKernel` and `MaternArdKernel` take a `MaternNu` (`OneHalf`, `ThreeHalves` or `FiveHalves`). The smoothness order is fixed at construction and is not an optimizable parameter.

### Formulas and parameters

Let $r = \lVert x - x' \rVert$ and, for the ARD leaves, $s^2 = \sum_k \bigl((x_k - x'_k)/\ell_k\bigr)^2$.

| Leaf | $k(x, x')$ | Hyperparameters, in order |
|---|---|---|
| `RbfKernel` | $\exp\!\bigl(-r^2 / 2\ell^2\bigr)$ | $\ell$ |
| `RbfArdKernel` | $\exp\!\bigl(-s^2 / 2\bigr)$ | $\ell_1, \dots, \ell_d$ |
| `MaternKernel`, `OneHalf` | $\exp(-r/\ell)$ | $\ell$ |
| `MaternKernel`, `ThreeHalves` | $\bigl(1 + \sqrt3\, r/\ell\bigr)\exp\!\bigl(-\sqrt3\, r/\ell\bigr)$ | $\ell$ |
| `MaternKernel`, `FiveHalves` | $\bigl(1 + u + u^2/3\bigr)\exp(-u)$, $u = \sqrt5\, r/\ell$ | $\ell$ |
| `MaternArdKernel` | the Matern formula with $r/\ell$ replaced by $s$ | $\ell_1, \dots, \ell_d$ |
| `PeriodicKernel` | $\exp\!\bigl(-2\sin^2(\pi r / p)/\ell^2\bigr)$ | $\ell$, $p$ (period) |
| `RationalQuadraticKernel` | $\bigl(1 + r^2/(2\alpha\ell^2)\bigr)^{-\alpha}$ | $\ell$, $\alpha$ |
| `RationalQuadraticArdKernel` | $\bigl(1 + s^2/(2\alpha)\bigr)^{-\alpha}$ | $\ell_1, \dots, \ell_d$, $\alpha$ |
| `ConstantKernel` | $c$ | $c$ |
| `LinearKernel` | $v\,(x \cdot x')$ | $v$ |
| `WhiteKernel` | $v\,\delta_{ij}$ on the training Gram matrix, 0 otherwise | $v$ |

The example computes each value a second time with plain `std::` math and compares:

```{code-block} text
kernel                          k(p0,p0)    k(p0,p1)    expected
RbfKernel                       1.000000    0.329193    0.329193   ok
RbfArdKernel                    1.000000    0.367879    0.367879   ok
MaternKernel (1/2)              1.000000    0.225212    0.225212   ok
MaternKernel (3/2)              1.000000    0.270882    0.270882   ok
MaternKernel (5/2)              1.000000    0.286713    0.286713   ok
MaternArdKernel (5/2)           1.000000    0.317283    0.317283   ok
PeriodicKernel                  1.000000    0.200271    0.200271   ok
RationalQuadraticKernel         1.000000    0.413265    0.413265   ok
RationalQuadraticArdKernel      1.000000    0.444444    0.444444   ok
ConstantKernel                  2.000000    2.000000    2.000000   ok
LinearKernel                    0.625000    1.875000    1.875000   ok
WhiteKernel                     0.100000    0.000000    0.000000   ok
```

:::{warning}
`WhiteKernel` adds its variance to the diagonal of a *square training* Gram matrix. It identifies "the same point" by row index, not by coordinates, and contributes nothing to a cross-covariance between two point sets. `Gpr` already adds a separate observation noise variance, so adding a `WhiteKernel` on top of it makes the two noise terms compete. Use `WhiteKernel` only when you evaluate kernels yourself or want a noise term that lives inside the kernel's parameter vector.
:::

### Parameters, bounds and accessors

Hyperparameters are strictly positive and always stored and exchanged as **logarithms**. Every leaf offers the same small interface:

| Member | Purpose |
|---|---|
| `num_params()` | Number of hyperparameters: 1 for most leaves, 2 for `PeriodicKernel` and `RationalQuadraticKernel`, the dimension for the ARD Matern and RBF leaves, dimension plus one for `RationalQuadraticArdKernel` |
| `get_params(out)` | Writes $\log\theta$ into a span of length `num_params()` |
| `set_params(params)` | Replaces the values from $\log\theta$. Atomic: on error nothing changes |
| `bounds_into(out)` | Writes the open interval of each parameter (in user units) |
| `operator==` | Compares values and intervals |

Each leaf also has typed accessors: `lengthscale()` and `interval()` (`RbfKernel`, `MaternKernel`), `lengthscales()` and `intervals()` (`RbfArdKernel`, `MaternArdKernel`), `nu()` (both Matern leaves), `lengthscale()`, `period()`, `lengthscale_interval()` and `period_interval()` (`PeriodicKernel`), `lengthscale()`, `alpha()`, `lengthscale_interval()` and `alpha_interval()` (`RationalQuadraticKernel`), `lengthscales()`, `alpha()`, `lengthscale_intervals()` and `alpha_interval()` (`RationalQuadraticArdKernel`), `constant()` and `interval()` (`ConstantKernel`), and `variance()` and `interval()` (`LinearKernel`, `WhiteKernel`).

The default interval is `Interval::default_positive()`, the open range $(10^{-5}, 10^5)$. Pass your own `Interval` (one per parameter) to the `create` overload that accepts intervals to tighten it. A value outside its interval is rejected with `GprErrorCode::InvalidInterval`.

```{literalinclude} /../examples/kernels_tour.cpp
:language: cpp
:start-after: // [custom-bounds]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
bounded lengthscale in (0.100000, 10.000000); create(0.05, narrow) -> value 0.05 is not strictly inside (0.1, 10)
after set_params: lengthscale = 2.500000, second update refused, lengthscale still 2.500000
```

## Compose kernels

Wrap each leaf in a `KernelSpec` and combine with `+` and `*`. `KernelLeaf` is the `std::variant` over all ten leaves and `CustomKernel`; `KernelSpec` converts from it implicitly, but because C++ allows only one user-defined conversion, write `KernelSpec(leaf)` when an operator needs both operands converted.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [compose]
:end-before: // [end]
:dedent: 4
```

This is $k(x, x') = 2\,\exp(-r^2/2\ell^2) + 0.1\,\delta$ with $\ell = 1.5$. `*` binds tighter than `+`, as usual. `KernelSpec::sum` and `KernelSpec::product` are the named forms of the operators; `KernelSpec::custom` wraps a `CustomKernel` (see {doc}`write-custom-kernel`). A `KernelSpec` is an immutable tree of value types: copying it is cheap, and modifying a copy never changes the original.

### The active parameter vector

A tree exposes **one flat vector** of log-hyperparameters: leaves in depth-first, left-to-right order, each leaf's own parameters in the order of the table above. That vector is what an optimizer works on.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [parameters]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
num_params = 3
log-parameters = [0.693147, 0.405465, -2.30259]
  parameter 0 -> leaf 0, local slot 0
  parameter 1 -> leaf 1, local slot 0
  parameter 2 -> leaf 2, local slot 0
coord_mode = Dist
```

- `KernelSpec::num_params`, `get_params`, `set_params` and `bounds_into` work on the whole tree.
- `parameter_bindings` returns one `ParameterBinding` per flat index: `index`, the owning `leaf_id`, and the `local_index` inside that leaf.
- `set_params` is atomic across the tree. A value that leaves its interval makes the call fail with `InvalidInterval` and every leaf keeps its old value.

### Which input a kernel needs

`coord_mode` returns a `CoordMode` that says what a compiled kernel can be evaluated from:

| `CoordMode` | Meaning | Leaves |
|---|---|---|
| `Dist` | Squared-distance matrices suffice | `RbfKernel`, `MaternKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, `CustomKernel` |
| `Points` | Needs coordinates | ARD leaves, `LinearKernel` |
| `Either` | Either input works | `ConstantKernel`, `WhiteKernel` only |
| `Mixed` | Distance leaves and coordinate leaves together | any such sum or product |

## Compile and evaluate

`KernelSpec::compile<T>()` produces a `CompiledKernel<T>`, where `T` is `double` or `float` (the `KernelScalarConcept`). `CompiledKernel<T>::from_spec` is the underlying factory. Compilation flattens nested sums and products, caches ARD length scales and computes how many scratch buffers evaluation needs.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [compile]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
leaves = 3, nested_depth = 1, reads_distances = true, needs_ard_sq_diff = false
```

`num_leaves`, `nested_depth`, `reads_distances`, `needs_ard_sq_diff` and `coord_mode` describe the compiled tree. A `CompiledKernel` also has `num_params`, `get_params`, `set_params`, `bounds_into` and `parameter_bindings` with the same meaning as on `KernelSpec`. It is copyable (deep copy) and movable.

### Gram matrices

Matrices are column-major views: `MatrixRef<T>` (read-only) and `MatrixMut<T>` (writable). `Triangle` (`Lower`, `Upper` or `Full`) selects which part of a symmetric result is written. Every evaluation takes a scratch matrix of the same shape that must not alias the output. The overloads without a `NestedBuffers` argument allocate their own; to avoid allocation in a loop, create one with `make_nested_buffers` and pass it.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [gram]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
K(0,0) = 2.1  K(1,0) = 1.60147  K(2,0) = 0.822225
```

By hand: $K_{00} = 2 + 0.1 = 2.1$ and $K_{10} = 2\exp(-1/4.5) = 1.60147$. The example asserts these and that the distance and coordinate paths agree.

| Task | From squared distances | From coordinates |
|---|---|---|
| Training Gram $K(X, X)$ | `apply` | `apply_points` |
| Cross-covariance $K(X_1, X_2)$ | `apply_cross` | `apply_cross_points` |
| Prior variance $k(x_i, x_i)$ | | `fill_diag` |
| First derivative $\partial K / \partial \log\theta_p$ | `grad` | `grad_points` |
| Second derivative $\partial^2 K / \partial \log\theta_i \partial \log\theta_j$ | `hess` | `hess_points` |

`apply_cross_points` needs `x1` and `x2` to have the same number of columns; `WhiteKernel` contributes 0 there. `fill_diag` writes into a span of length $n$ and includes white-noise variance.

:::{note}
Derivatives are taken with respect to the **log** of each parameter, matching `get_params`. For $k = c\,e^{-r^2/2\ell^2}$ the derivative with respect to $\log\ell$ is $k\, r^2/\ell^2$. The example checks this by hand:

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [gradient]
:end-before: // [end]
:dedent: 4
```
:::

### Errors you will meet

Evaluation never throws; it returns `std::expected<void, GprError>`:

| Code | Typical cause |
|---|---|
| `UnsupportedKernelOperation` | `apply`, `apply_cross`, `grad` or `hess` on a `Points` or `Mixed` kernel, which needs coordinates |
| `DimensionMismatch` | `x.cols()` differs from an ARD leaf's length-scale count |
| `LengthMismatch`, `EmptyInput` | Output or input shapes disagree, or an input is empty |
| `WorkspaceTooSmall`, `InvalidConfig` | A scratch or nested buffer is too small, or scratch aliases the output |
| `NonFiniteInput` | NaN or infinity in the input |
| `IndexOutOfRange` | Parameter index $\ge$ `num_params()` |

### Single precision

The same `KernelSpec` compiles to `CompiledKernel<float>`. Hyperparameters stay in `double` inside the spec and are converted on compilation. Expect agreement with the `double` result to about $10^{-6}$.

## Choose the exponential backend

Built-in leaves with an exponential, such as the RBF and Matern leaves, can switch to vectorized Estrin minimax SIMD approximations instead of `std::exp`. `CompiledKernel::kernel_exp` returns the current `KernelExp` (default `Accurate`) and `set_kernel_exp` switches to `FastApprox`. On a trainer the same switch is `Gpr::with_kernel_exp`. The approximation maintains 1–2 ULP precision (relative error $< 10^{-14}$ in `double` and $< 5 \times 10^{-7}$ in `float` on $[-100, 0]$) while trading bit-for-bit `std::exp` parity for SIMD speed; see {doc}`../explanation/kernels` for details. The scalar functions are public:

```{literalinclude} /../examples/kernel_exp.cpp
:language: cpp
:start-after: // [scalar]
:end-before: // [end]
:dedent: 4
```

```{literalinclude} /../examples/kernel_exp.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
max relative error of FastApprox on [-100, 0]: double < 1.2e-07, float < 5.0e-06
```

`eval_kernel_exp(x, mode)` returns $e^x$ under the chosen backend. `eval_kernel_exp_jet(x, mode)` returns an `ExpJet` with the value `v` and the first two derivatives `d1` and `d2` (both backends set `d1 = d2 = v`):

```{literalinclude} /../examples/kernel_exp.cpp
:language: cpp
:start-after: // [jet]
:end-before: // [end]
:dedent: 4
```

Parameter transforms, `CustomKernel` leaves and everything outside kernel evaluation always use `std::exp`.

## Hand the kernel to a trainer

`Gpr` takes a `KernelSpec`. It adds its own observation noise, so leave the `WhiteKernel` out.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
predict(1.25) = 0.948457 (sin = 0.948985)
predict(3.75) = -0.571757 (sin = -0.571561)
```

Targets are standardised, and the kernel is `ConstantKernel * RbfKernel` so the optimizer can learn the signal variance. See {doc}`configure-fit` and {doc}`predict-and-sample` for the trainer and the prediction API.

## Work with single leaves and cached values

These members of `CompiledKernel` serve optimizers and custom solvers. Most users never call them directly; they are what `Gpr` uses internally.

### Reuse leaf Gram matrices

`apply_leaf_points` evaluates one leaf alone, `apply_leaf_cached` does the same from a `DistanceCache`, `apply_leaf_cached_incremental` updates a caller-owned packed ARD matrix incrementally when only some length scales changed, and `apply_leaf_incremental` + `commit_leaf_ard_s2` unify cached and uncached incremental ARD updates with non-destructive trial support. `ard_leaf_count` and `ard_dims_per_leaf` report per-leaf ARD sizing metadata, `leaf_index_for_param` maps a flat parameter index to its leaf, and `combine_from_leaf_grams` rebuilds the composite from the per-leaf matrices without recomputing any leaf.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [leaf-level]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
recombined == gram: true, parameter 2 belongs to leaf 2
```

### One-pass weighted gradients

The log marginal likelihood gradient needs $g_p = \langle W, \partial K/\partial\theta_p\rangle$ for every parameter $p$. `weighted_grads` computes all of them in one walk over the tree. `W` is passed as its lower triangle and is mirrored, so each off-diagonal entry counts twice and each diagonal entry once. For products, the factor Gram matrices can be kept from the forward pass: `kept_products` and `kept_buffers(products)` size those buffers, `weighted_buffers(products)` sizes the workspace, and `apply_keeping` and `apply_cached_keeping` evaluate the Gram matrix while storing them.

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [weighted]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
weighted gradients = [14.0503, 5.77053, 0.3]
```

The result for the white-noise parameter is $0.1 \times 3$ diagonal entries; the example also compares every entry with a hand contraction of `grad_points` output.

### Use a DistanceCache

`apply_cached`, `grad_cached` and `hess_cached` take a `DistanceCache` and skip recomputing distances; see {doc}`use-supplied-distances`.

## NestedBuffers

`NestedBuffers<T>` is a single contiguous block holding `depth` matrices of shape `rows` $\times$ `cols`, so a recursive tree evaluation needs no allocation. `make_nested_buffers(rows, cols)` creates one of the right depth; `depth`, `rows` and `cols` report its capacity, `resize` regrows it without shrinking, and `level(index, rows, cols)` returns the `MatrixMut<T>` view of one level. A default-constructed `NestedBuffers` has zero capacity.

## API summary

| Symbol | What it is |
|---|---|
| `MaternNu` | `OneHalf`, `ThreeHalves`, `FiveHalves` |
| `RbfKernel`, `RbfArdKernel`, `MaternKernel`, `MaternArdKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, `RationalQuadraticArdKernel`, `ConstantKernel`, `LinearKernel`, `WhiteKernel` | Built-in leaves with `create`, `num_params`, `get_params`, `set_params`, `bounds_into` and `operator==` |
| `KernelLeaf` | `std::variant` over the ten leaves and `CustomKernel` |
| `KernelSpec` | Immutable expression tree: `custom`, `sum`, `product`, `operator+`, `operator*`, `num_params`, `get_params`, `set_params`, `bounds_into`, `parameter_bindings`, `coord_mode`, `compile` |
| `ParameterBinding` | `index`, `leaf_id`, `local_index` |
| `CoordMode` | `Dist`, `Points`, `Either`, `Mixed` |
| `CompiledKernel<T>` | Execution tree: `from_spec`, `num_params`, `num_leaves`, `ard_leaf_count`, `ard_dims_per_leaf`, `coord_mode`, `nested_depth`, `reads_distances`, `needs_ard_sq_diff`, `kernel_exp`, `set_kernel_exp`, `make_nested_buffers`, `get_params`, `set_params`, `bounds_into`, `parameter_bindings` |
| `CompiledKernel<T>` evaluation | `apply`, `apply_points`, `apply_cross`, `apply_cross_points`, `fill_diag`, `grad`, `grad_points`, `hess`, `hess_points` |
| `CompiledKernel<T>` cached and leaf-level | `apply_cached`, `grad_cached`, `hess_cached`, `leaf_index_for_param`, `apply_leaf_points`, `apply_leaf_cached`, `apply_leaf_cached_incremental`, `apply_leaf_incremental`, `commit_leaf_ard_s2`, `combine_from_leaf_grams`, `kept_products`, `kept_buffers`, `weighted_buffers`, `apply_keeping`, `apply_cached_keeping`, `weighted_grads` |
| `NestedBuffers<T>` | `depth`, `rows`, `cols`, `resize`, `level` |
| `KernelExp` | `Accurate`, `FastApprox` |
| `ExpJet<T>` | `v`, `d1`, `d2` |
| `eval_kernel_exp`, `eval_kernel_exp_jet` | Scalar exponential under a `KernelExp` backend |
