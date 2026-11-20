# How to write a custom kernel

This guide shows how to add your own kernel leaf, plug it into `KernelSpec` composition, check its derivatives, and make it persistable. The worked example implements $k(d^2) = 1/(1 + d^2/2\ell^2)$ and verifies it against the built-in `RationalQuadraticKernel`, which equals it when $\alpha = 1$.

Read {doc}`build-kernels` first for `KernelSpec` and `CompiledKernel`. Everything here is declared in `cxxgpr/kernel/custom.hpp`; the [kernel API reference](../reference/api/kernel.md) lists the signatures.

## What a custom kernel is

A custom leaf is a function of the **squared Euclidean distance** $d^2 = \lVert x - x'\rVert^2$ with $p \ge 1$ hyperparameters. The library computes the squared distances and calls your code on matrices of them. You provide:

- the value of the kernel on a distance matrix,
- its first and second derivatives with respect to the **logarithm** of each of your parameters,
- the parameter vector in log space, with an open interval for each parameter.

Because the leaf sees only distances, it cannot be anisotropic or depend on the raw coordinates. For distances that you compute yourself, see {doc}`use-supplied-distances`.

## The `KernelTerm<T>` interface

Derive from `KernelTerm<T>`, where `T` is `double` or `float` (`KernelScalarConcept`). One class template covers both precisions. Every member below is pure virtual unless marked optional.

| Member | Contract |
|---|---|
| `clone` | Returns a deep copy as `std::unique_ptr<KernelTerm<T>>` |
| `num_params` | Number of hyperparameters; `noexcept` |
| `get_params(out)` | Writes the log-space parameters; `LengthMismatch` if `out.size() != num_params()` |
| `set_params(params)` | Validates, then assigns. On any error the object must be unchanged. Return `LengthMismatch`, `NonFiniteInput` or `InvalidInterval` |
| `bounds_into(out)` | Writes one open `Interval` per parameter, in user units, in parameter order |
| `apply(dist, out, uplo)` | Training Gram matrix from an $n \times n$ squared-distance matrix. Only the triangle `uplo` needs to be written |
| `apply_cross(dist, out)` | Cross-covariance from an $n_1 \times n_2$ matrix; write every entry |
| `fill_diag(out)` | $k(x_i, x_i)$ for all $i$: the kernel at distance 0 |
| `grad(dist, d_k, param_idx, uplo)` | $\partial k/\partial\log\theta_{p}$ |
| `hess(dist, d2_k, i, j, uplo)` | $\partial^2 k/\partial\log\theta_i\,\partial\log\theta_j$ |
| `persist_id` (optional) | Non-empty identifier enables saving; default is empty |
| `persist_state_json` (optional) | JSON string with the state; default returns a `PersistFailed` error of kind `NotPersistable` |

Squared distances can round to a tiny negative number; clamp them at zero before taking a square root or a power.

### Derivatives are with respect to log parameters

The optimizer works in $\vartheta = \log\theta$, so `grad` and `hess` must return derivatives with respect to $\vartheta$, not $\theta$. For the example, with $w = d^2/(2\ell^2)$ and $dw/d\vartheta = -2w$:

$$
k = \frac{1}{1 + w}, \qquad
\frac{\partial k}{\partial\vartheta} = \frac{2w}{(1 + w)^2}, \qquad
\frac{\partial^2 k}{\partial\vartheta^2} = \frac{4w\,(w - 1)}{(1 + w)^3}.
$$

## Implement the term

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [term]
:end-before: // [end]
:dedent: 4
```

The helper `fill` walks the requested triangle, applies a scalar function of $w$, and is shared by the value, gradient and Hessian. Three details matter:

- **Atomic `set_params`.** It validates the size, finiteness and bounds first and assigns last.
- **Distinct error codes.** Out-of-range values go through `GprError::invalid_interval_out_of_range`, which carries `GprErrorCode::InvalidInterval`, so callers can tell a bound violation from a shape error.
- **Everything is `T`.** The two instantiations `InverseQuadraticTerm<double>` and `InverseQuadraticTerm<float>` must behave identically up to precision.

## Wrap the term in a `CustomKernel`

`CustomKernel` is a value-semantic wrapper that owns one `KernelTerm<double>` and one `KernelTerm<float>`, so the same leaf works with `DoublePrecision`, `SinglePrecision` and mixed precision. `CustomKernel::create` takes the two terms and checks that neither is null and that they agree on `num_params` and `persist_id`; it returns `InvalidConfig` or `LengthMismatch` otherwise. When the term is a class template with constructor arguments, `make_custom_kernel` builds both instantiations for you:

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [make]
:end-before: // [end]
:dedent: 4
```

`KernelSpec::custom` wraps it into a spec. `CustomKernel` also provides `num_params`, `get_params`, `set_params` (atomic over both precisions), `bounds_into`, `persist_id`, `persist_state_json`, `term<T>()` for read-only access to the typed term, copy and move, and `operator==` (bitwise comparison of the log-parameters). The wrapper reads parameters from the `double` term.

```{code-block} text
num_params = 1, persist_id = example.inverse_quadratic
```

## Verify value, gradient and Hessian

A custom kernel is only as good as its derivatives. Compare against a reference whenever you can, and against finite differences always.

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [compare]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
max |custom - rational quadratic|: value 1.11022e-16, gradient 1.11022e-16, Hessian 5.55112e-17
k(p0, p1) = 0.782609 (hand: 0.782609)
```

The finite-difference check perturbs the log parameter through `set_params`:

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [finite-difference]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
max |gradient - finite difference| = 1.79211e-11
```

The custom leaf composes like any other and contributes its parameters to the flat vector (here the constant first, the lengthscale second). Out-of-bounds updates are rejected:

```{code-block} text
constant * custom: log-parameters = [1.09861, 0.405465]
set_params(log 100) -> value 100.00000000000004 is not strictly inside (0.05, 50)
```

:::{note}
A `CustomKernel` always uses `std::exp` for anything you compute inside it. `KernelExp::FastApprox` only changes the built-in leaves.
:::

:::{note}
A custom leaf receives squared distances only. Through `CompiledKernel::apply_points` and its siblings the library computes them from the coordinates; through `apply` you pass them in. In either case the term never sees the coordinates themselves.
:::

## Make the kernel persistable

Override `persist_id` with a non-empty, unique string that does not start with `cxxgpr.` (that prefix is reserved), and `persist_state_json` with a JSON object that contains everything needed to rebuild the leaf. Both precisions must return the same identifier. A leaf that keeps the defaults can be used for training, but saving a model that contains it fails with a `NotPersistable` error.

When loading, register a factory that rebuilds the `CustomKernel` from that JSON in a `persist::PersistRegistry` and pass the registry to the loader. The registry round trip can be tested without touching disk:

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [persist]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
state = {"lengthscale":1.500000}, restored parameters equal: true
```

Saving and loading whole models, including registries for custom transforms, is covered in {doc}`persist-models`.

## Complete program

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:linenos:
```

## API summary

| Symbol | What it is |
|---|---|
| `KernelTerm<T>` | Abstract leaf: `clone`, `num_params`, `get_params`, `set_params`, `bounds_into`, `apply`, `apply_cross`, `fill_diag`, `grad`, `hess`, `persist_id`, `persist_state_json` |
| `CustomKernel` | Paired `double`/`float` wrapper: `create`, `term<T>`, `num_params`, `get_params`, `set_params`, `bounds_into`, `persist_id`, `persist_state_json`, `operator==` |
| `make_custom_kernel<TermTemplate>(args...)` | Builds both terms from the same constructor arguments and calls `CustomKernel::create` |
| `KernelSpec::custom` | Wraps a `CustomKernel` as a leaf of a kernel tree |
