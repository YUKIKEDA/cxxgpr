# How to use the low-level building blocks

`Gpr` and `FittedGpr` are assembled from public pieces that you can call yourself: dense and triangular linear algebra, deterministic parallel helpers, pre-allocated workspaces, marginal-likelihood kernels, posterior inference and the mixed-precision refinement engine. This guide shows when and how to use them directly, for example to write your own training loop or optimizer, to reuse a factor across many right-hand sides, to implement a custom linear system for refinement, or to check results against an independent computation.

If you only want to fit and predict, stay with the high-level API ({doc}`configure-fit`, {doc}`predict-and-sample`). Everything on this page is a sharp tool: you own the buffers, you choose the precision, and you must respect the preconditions. The generated declarations are in the [linalg](../reference/api/linalg.md), [workspace](../reference/api/workspace.md) and [precision](../reference/api/precision.md) reference pages.

## Headers and conventions

| Header | Namespace | Contents |
|---|---|---|
| `cxxgpr/linalg/cholesky.hpp` | `cxxgpr::linalg` | In-place Cholesky, jittered retries, triangular solves, inverse diagonal, log-determinant |
| `cxxgpr/linalg/dense.hpp` | `cxxgpr::linalg` | Diagonal update, lower-triangle copy and mirror, inner products, matrix-vector product |
| `cxxgpr/linalg/parallel.hpp` | `cxxgpr::linalg` | Area-balanced column partitioning, deterministic parallel loops and reductions |
| `cxxgpr/workspace/buffers.hpp` | `cxxgpr` | `FitPolicy`, `FitBuffers`, `WorkspaceCore`, `HessianScratch`, `QueryWorkspace` |
| `cxxgpr/workspace/mll.hpp` | `cxxgpr` | Negative log marginal likelihood, gradient, Hessian |
| `cxxgpr/workspace/inference.hpp` | `cxxgpr` | Posterior mean, variance, covariance, samples and leave-one-out predictions |
| `cxxgpr/workspace/leaf_cache.hpp` | `cxxgpr` | `LeafCache` and optimizer traits for incremental Gram updates |
| `cxxgpr/workspace/workspace.hpp` | `cxxgpr` | Umbrella header for the four `workspace` headers above |
| `cxxgpr/precision/refine.hpp` | `cxxgpr` | Iterative refinement engine, `RefineSystem`, residual helpers |

Conventions shared by every function:

- **Matrices are column-major views.** `MatrixRef<T>` (read-only) and `MatrixMut<T>` (writable) hold a pointer, the row and column counts and `col_stride()`, the distance in elements between the starts of two columns. `MatrixMut<T>` converts implicitly to `MatrixRef<T>`.
- **Scalars are `float` or `double`**, as required by `KernelScalarConcept`. Functions that take a precision policy `P` use `P::Storage` for matrices and factors and `P::Refine` for weights and outputs.
- **Errors are values.** Every function returns `std::expected<T, GprError>`; none throws. The common codes are `EmptyInput`, `SizeOverflow`, `LengthMismatch`, `DimensionMismatch`, `NonFiniteInput`, `NonPositiveDefiniteMatrix`, `CholeskyFailed` and `WorkspaceTooSmall`.
- **Lower triangle only.** Symmetric matrices are stored and read through their lower triangle (diagonal included). The strictly upper triangle is neither read nor written unless a function says so.

## Factor and solve with `cxxgpr::linalg`

The program below builds a symmetric positive-definite matrix $A$, factors a copy in place, and solves with the factor. The complete source is `low_level_cholesky.cpp`.

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [factor]
:end-before: // [end]
:dedent: 4
```

`cholesky_lower_in_place(a, stage)` overwrites the lower triangle of `a` with $L$ such that $A = L L^\top$. The `stage` argument (`CholeskyStage::Fit` by default, or `CholeskyStage::Predict`) is only recorded in a `CholeskyFailed` error so that you can tell where the failure occurred. `copy_lower_triangle(src, dst)` is the usual way to keep $A$ intact.

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [solve]
:end-before: // [end]
:dedent: 4
```

All solves work in place on the right-hand side and accept either a vector (`std::span<T>`) or a matrix of several right-hand sides (`MatrixMut<T>`):

| Function | Solves | Notes |
|---|---|---|
| `solve_lower_in_place(l, rhs)` | $L X = B$ | Forward substitution |
| `solve_lower_transpose_in_place(l, rhs)` | $L^\top X = B$ | Backward substitution, reads only the lower triangle of $L$ |
| `solve_llt_in_place(l, rhs)` | $L L^\top X = B$ | Both substitutions; use it to solve with $A$ |
| `inv_diag_from_chol_l(l, out_diag, scratch)` | $\operatorname{diag}(A^{-1})$ | Needs an $n \times n$ scratch matrix that is overwritten; accumulates in double precision |
| `log_det_from_l(l)` | $\log\lvert A\rvert = 2\sum_i \log L_{ii}$ | Neumaier-compensated sum, returned as `double` |
| `half_log_det_from_l(l)` | $\tfrac12 \log\lvert A\rvert$ | The term that appears in the marginal likelihood |

These functions validate their input on every call: the factor must be square with a finite lower triangle and a strictly positive diagonal, and the right-hand side must be finite with matching rows. They return `NonPositiveDefiniteMatrix` for a non-positive diagonal entry.

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [diag]
:end-before: // [end]
:dedent: 4
```

:::{warning}
If `cholesky_lower_in_place` fails with `CholeskyFailed`, the lower triangle of `a` has been partially overwritten. Keep a copy of $A$ (or use `cholesky_lower_with_jitter`, which restores it) if you want to retry.
:::

### Skip the validation in a hot loop

The `*_unchecked` functions perform the same substitutions without validating anything and without returning an error. They are `noexcept`. Call them only after one successful validated call on the same factor, with views that are non-empty, correctly strided, non-overlapping and finite:

`solve_lower_in_place_unchecked`, `solve_lower_transpose_in_place_unchecked` and `solve_llt_in_place_unchecked`, each with a matrix and a vector overload.

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [unchecked]
:end-before: // [end]
:dedent: 4
```

### Retry with jitter

`cholesky_lower_with_jitter` first tries the factorization with no jitter. Only if that fails with `CholeskyFailed` does it restore the saved lower triangle, add the next jitter value of the `JitterPolicy` to the diagonal and try again. It returns the jitter that finally worked (`0.0` if none was needed). Two overloads exist: one takes a caller-owned `backup` matrix of the same size and allocates nothing, the other allocates the backup itself.

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [jitter]
:end-before: // [end]
:dedent: 4
```

The jitter value is cast to the matrix scalar type before it is added, so a jitter below the resolution of `float` has no effect on a single-precision matrix. A `JitterPolicy::fixed(0.0)` never retries. The policy types (`FixedJitter`, `AdaptiveJitter`, `JitterSchedule`, `JitterPolicy`) are explained in {doc}`configure-fit` and {doc}`../explanation/numerical-stability`.

Output of the programs so far:

```{code-block} text
L(0,0) = 1.55547, upper sentinel untouched: 1
max |L L^T - A| = 4.44089e-16
max |A x - b| = 1.77636e-15
two triangular solves equal solve_llt_in_place: 1
diag(A^-1) agrees with the explicit inverse to 1.11022e-16
log|A| = 3.64456, 0.5 log|A| = 1.82228, naive sum = 3.64456
unchecked solve identical: 1
unjittered factorization of a rank-one matrix: Cholesky factorization failed (stage=Fit, size=4, jitter=0 already applied)
adaptive jitter policy succeeded with j = 1e-10
fixed(0) policy: CholeskyFailed, error code matches: 1
```

## Dense helpers

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [dense]
:end-before: // [end]
:dedent: 4
```

| Function | Effect |
|---|---|
| `add_to_diag(a, value)` | Adds a finite `value` to every diagonal entry of a square matrix; this is how noise and jitter are applied |
| `copy_lower_triangle(src, dst)` | Copies the lower triangle including the diagonal; the strictly upper triangles are untouched |
| `symmetrize_lower(a)` | Copies the strictly lower triangle into the strictly upper triangle in place |
| `frobenius_lower(a, b)` | $\langle A, B\rangle_F$ of two symmetric matrices, reading only lower triangles (diagonal once, off-diagonal twice), accumulated in `double` |
| `dot(a, b)` | Inner product of two equal-length spans, accumulated in `double` in ascending index order |
| `matvec_columns(a, x, out)` | $\text{out} = A x$ for a column-major $A$, column by column |

`dot`, `frobenius_lower` and `matvec_columns` use plain `double` accumulation in a fixed order. They are deterministic but not compensated; only `log_det_from_l` and the parallel reductions below use Neumaier compensation.

There is a second, header-only `cxxgpr::symmetrize_lower(MatrixMut<T>)` in `precision/refine.hpp`. It does the same job, returns nothing and silently does nothing for an empty, null or non-square view.

Output:

```{code-block} text
asymmetry after symmetrize_lower = 0, <A,A>_F = 68.8858 (full sum of squares 68.8858)
```

## Deterministic parallel helpers

The Gram-matrix and Cholesky code splits the lower triangle of an $n \times n$ matrix into column blocks of equal area, because early columns are longer than late ones. You can use the same partitioning for your own lower-triangular loops.

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [parallel]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
lower_area(1000, 1000) = 500500 (= 1000 * 1001 / 2)
4 area-balanced column blocks start at: 0 134 293 500 1000
every column visited exactly once: 1
sum of 1/k^2 for k = 1..1000 = 1.64393456668, bit-identical for 1, 4 and automatic threads: 1
openmp_max_threads() >= 1: 1
```

| Function | Purpose |
|---|---|
| `lower_area(n, col)` | Number of lower-triangular entries in columns $[0, \min(\text{col}, n))$ |
| `lower_block_start(n, block_idx, n_blocks)` | First column of block `block_idx` when the triangle is split into `n_blocks` blocks of near-equal area; `block_idx == n_blocks` gives `n` |
| `openmp_max_threads()` | The OpenMP thread budget (1 without OpenMP) |
| `parallel_for_lower_blocks(n, num_threads, fn)` | Calls `fn(col_begin, col_end)` once per non-empty block; every column belongs to exactly one block, so writes to per-column slots need no locking |
| `parallel_for_lower_blocks_raw(n, num_threads, fn, ctx)` | The same with a plain function pointer `void(*)(size_t, size_t, void*)` and a context pointer |
| `parallel_lower_reduce_cols(n, num_threads, col_partials, fn)` | Evaluates `fn(col)` for every column in parallel into `col_partials`, then sums them in ascending column order with Neumaier compensation. The result is bit-identical for every thread count |
| `parallel_lower_reduce_cols_raw(n, num_threads, col_partials, fn, ctx)` | The same with a function pointer `double(*)(size_t, void*)` |

`num_threads == 0` means "use `openmp_max_threads()`". The callable is forwarded by pointer through a captureless trampoline, so these templates allocate nothing. `col_partials` must have exactly `n` elements; the `_raw` variant also returns `InvalidConfig` for a null function pointer.

## Marginal likelihood, gradient and Hessian on your own buffers

The functions in `cxxgpr/workspace/mll.hpp` evaluate the negative log marginal likelihood

$$
\mathcal{L}(\theta) = \tfrac12\, y^\top A^{-1} y + \tfrac12 \log\lvert A\rvert + \tfrac n2 \log 2\pi,
\qquad A = K + \sigma_n^2 I
$$

and its derivatives with respect to the log-space parameters $\theta$ (all kernel parameters, then $\log\sigma_n^2$). You provide a `CompiledKernel`, a `GaussianLikelihood`, the training data in storage precision, a `JitterPolicy` and a `FitBuffers<P>`. The complete source is `workspaces.cpp`; its data are 30 noisy samples of $\sin 2x$ and its kernel is `ConstantKernel * RbfKernel`.

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [factor]
:end-before: // [end]
:dedent: 4
```

`FitBuffers<P>::create(n, policy)` allocates everything for $n$ training points once (see the tables below). `factor_and_mll` builds $A$, factors it, solves $A\alpha = y$ and returns $\mathcal{L}$. It leaves $L$ in `buffers.core.k_matrix` (view: `core.k_ref()`), $\alpha$ in `buffers.core.rhs` and the adopted jitter in `core.factor_jitter`. The example recomputes $\mathcal{L}$ from those pieces with `linalg::dot` and `linalg::half_log_det_from_l`.

The third parameter of `factor_and_mll`, `keep_products`, is the number of top-level product terms whose factor Gram matrices are kept for the gradient pass (`CompiledKernel::kept_products()` gives the number the kernel supports). The default `0` keeps none; `mll_and_gradient_into` passes the right value itself.

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [gradient]
:end-before: // [end]
:dedent: 4
```

`mll_and_gradient_into` computes $\mathcal{L}$ and the gradient in one pass over the kernel tree. `out_grad` must have `kernel.num_params() + 1` elements.

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [hessian]
:end-before: // [end]
:dedent: 4
```

`hessian_into` fills the symmetric $p \times p$ Hessian into `out_hess` (length $p^2$, $p$ = `kernel.num_params() + 1`). `value_gradient_hessian_into` returns the value and fills both outputs from one factorization. The `HessianScratch` inside `core.hessian` is empty until the first Hessian call; `is_allocated()` shows it.

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [policies]
:end-before: // [end]
:dedent: 4
```

With `FitPolicy::prefer_memory()` the buffers have no separate $W$ matrix and no distance cache, and gradient and Hessian calls overwrite $L$ and refactor it before returning. The results are identical.

```{code-block} text
negative log marginal likelihood = -10.1376
adopted jitter                   = 0
recomputed from L and alpha      = -10.1376
gradient (analytic | finite difference):
  d/dtheta0:      0.94508 |      0.94508
  d/dtheta1:     -2.56634 |     -2.56634
  d/dtheta2:     -2.18095 |     -2.18095
Hessian scratch allocated before/after the first call: 0 / 1
Hessian vs finite differences of the gradient, worst relative error = 1.6177e-08
prefer_speed : W buffer 900 doubles, distance cache 1, overwrites L 0
prefer_memory: W buffer 0 doubles, distance cache 0, overwrites L 1
same value and gradient under both policies: 1
```

| Function | Returns | Notes |
|---|---|---|
| `factor_and_mll<P>(kernel, likelihood, x, y, jitter, buffers, keep_products = 0)` | $\mathcal{L}$ as `double` | `y` is a `std::span<const P::Storage>`; `CholeskyFailed` after the whole jitter ladder fails |
| `mll_and_gradient_into<P>(kernel, likelihood, x, y, jitter, buffers, out_grad)` | $\mathcal{L}$ | Restores $L$ under `CholeskyBuffer::Reuse` |
| `hessian_into<P>(..., buffers, out_hess)` | nothing | Allocates `HessianScratch` on first use |
| `value_gradient_hessian_into<P>(..., buffers, out_grad, out_hess)` | $\mathcal{L}$ | One factorization for all three |

`P` is deduced from the type of `buffers`.

### The buffer types

`FitPolicy` bundles the two memory decisions: its members `distance_cache` (a `DistanceCachePolicy`) and `cholesky_buffer` (a `CholeskyBuffer`). `FitPolicy::prefer_speed()` selects `Cached` and `Retain`, `FitPolicy::prefer_memory()` selects `Uncached` and `Reuse`, and `operator==` compares both fields.

`FitBuffers<P>` has these members and functions:

| Member | Meaning |
|---|---|
| `create(n, cache = Cached, buffer = Retain)` and `create(n, policy)` | Allocate for $n$ points; `EmptyInput` for `n == 0`, `SizeOverflow` if $n^2$ overflows |
| `core` | The `WorkspaceCore<P>` described below |
| `dist` | `std::optional<DistanceCache<Storage>>`, present when `cache_policy == Cached`, filled on first use |
| `w_matrix` | The $n \times n$ buffer for $W = \alpha\alpha^\top - A^{-1}$ under `Retain`, empty under `Reuse` |
| `cache_policy`, `buffer_policy` | The two policies that were requested |
| `overwrites_cholesky()` | `true` under `Reuse`: gradient and Hessian calls clobber $L$ and refactor it |
| `w_mut()`, `w_ref()` | Views of $W$: `w_matrix` under `Retain`, the `k_matrix` under `Reuse` |

`WorkspaceCore<P>` holds the shared training scratch:

| Member | Meaning |
|---|---|
| `create(n)` | Allocate the core buffers for $n$ points |
| `order()` and `order_` | The training order $n$ |
| `k_matrix`, `k_mut()`, `k_ref()` | $A = K + \sigma_n^2 I$, then $L$ after `factor_and_mll` |
| `exp_buf`, `exp_mut()`, `exp_ref()` | Scratch for kernel evaluation, the jitter backup and derivative scratch |
| `kernel_scratch`, `kernel_scratch_mut()` | Extra $n\times n$ scratch for joint gradient and Hessian walks; empty until `ensure_kernel_scratch()` |
| `ensure_kernel_scratch()` | Allocates `kernel_scratch` once |
| `rhs` | $y$ before the solve, $\alpha$ after |
| `theta` | Parameter backup for rollback after a failed factorization |
| `col_partials` | Per-column partial sums for deterministic reductions (length $3n$) |
| `nested`, `weighted`, `ensure_weighted(n_bufs)` | `NestedBuffers` for tree evaluation and for the single-pass gradient |
| `hessian` | The lazily allocated `HessianScratch` |
| `factor_jitter` | The jitter used by the last successful factorization |

`HessianScratch<S>` stores the quantities of the analytic Hessian, $S_i = L^{-1} A_i L^{-\top}$ for each of the $p$ parameters, $v_i = L^{-1} A_i \alpha$ and two auxiliary buffers:

| Member | Meaning |
|---|---|
| `ensure(n, p)` | Allocates for order $n$ and $p$ parameters; existing capacity is kept |
| `is_allocated()` | `true` after a successful `ensure` |
| `s_mut(i)`, `s_ref(i)` | The $n \times n$ matrix $S_i$ |
| `v_col_mut(i)`, `v_col_ref(i)` | Column $i$ of the $n \times p$ matrix $V$ |
| `scratch_c_mut()` | A third $n\times n$ scratch matrix |
| `order_`, `num_params_`, `s_storage`, `v_storage`, `u`, `scratch_c` | The sizes and the raw storage |

## Incremental Gram updates with `LeafCache`

Optimizers such as `FastSimulatedAnnealing` change one hyperparameter at a time. If only one kernel leaf depends on the changed parameter, the other leaves' Gram matrices need not be recomputed. `LeafCache<T>` stores one Gram matrix per leaf, verifies that no undeclared parameter changed, and recomputes only the dirty leaves.

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [leaf-cache]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
leaves cached: 2, primed: 1
incremental Gram vs full evaluation, max difference = 0
undeclared change rejected: 1
```

The check is bit-exact: `prepare_step` compares every parameter outside `change_indices` with the recorded vector through `std::bit_cast`, so even a change from `+0.0` to `-0.0` is detected and reported as `IndexOutOfRange`.

| Member | Purpose |
|---|---|
| `LeafCache()` | Empty cache; allocates nothing |
| `resize(n_leaves, n)` | Sizes the storage and marks every leaf dirty |
| `prepare_step(params, change_indices, kernel, total_params)` | Validates the step and marks the dirty leaves |
| `evaluate_incremental(kernel, x, dist_cache, params, change_indices, total_params, out, uplo, scratch, nested)` | Recomputes dirty leaves and combines all leaf Grams into `out`; invalidates the cache if anything fails |
| `record(params)` | Stores `params` as the new baseline and marks the cache primed |
| `mark_all_dirty()`, `invalidate()` | Force a full recomputation (the second also clears the primed state) |
| `is_primed()`, `is_leaf_dirty(leaf)`, `num_leaves()`, `order()`, `last_params()` | State queries |
| `leaf_gram(leaf)`, `leaf_gram_mut(leaf)` | The cached Gram matrix of one leaf |

Related free symbols: `require_change_indices(indices, n_params)` rejects empty, out-of-range or duplicated index lists. `OptimizerTraits<O>::uses_change_indices` and the variable template `optimizer_uses_change_indices_v<O>` tell at compile time whether an optimizer type supplies changed indices; the concept `OptimizerHasChangeIndicesConcept` matches a type that has a static member `uses_change_indices` convertible to `bool`. Your own optimizer type can opt in by declaring `static constexpr bool uses_change_indices = true;`.

## Posterior inference on your own factor

Given $L$, $\alpha$ and the training inputs, `cxxgpr/workspace/inference.hpp` computes the posterior without a `FittedGpr`. A `QueryWorkspace<P>` provides the temporaries and grows as needed.

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
query workspace capacity: n = 30, m = 5, d = 1
  x* =  0.25  mean =   0.481545  sd =  0.109424  sin(2x*) =  0.479426
  x* =  0.75  mean =    1.03939  sd =  0.108098  sin(2x*) =  0.997495
  x* =   1.5  mean =   0.155948  sd =  0.107833  sin(2x*) =   0.14112
  x* =  2.25  mean =   -1.03641  sd =  0.108098  sin(2x*) =  -0.97753
  x* =  2.75  mean =  -0.681773  sd =  0.109424  sin(2x*) =  -0.70554
low-level vs FittedGpr::predict, max mean difference = 0
leave-one-out RMSE against the noisy targets = 0.120851
covariance is 5 x 5, 20 sample values, same seed gives identical draws: 1
```

The standard deviation includes the observation noise (`VarianceKind::Observation`, the default of `PredictOptions`), so it is about $\sqrt{0.01 + \text{latent variance}} \approx 0.11$. These functions work on the data exactly as you pass them: there are no input or target transforms and no inverse scaling. Apply your own transforms before and after.

| Function | Result | Notes |
|---|---|---|
| `predict_into<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, options, workspace, out)` | fills `Prediction<P::Refine>` | No allocation once `workspace` and `out` have capacity |
| `predict<P>(...same, without out)` | returns `Prediction<P::Refine>` | Allocates the result |
| `predict_covariance_into<P>(..., out)` | fills `PredictiveCovariance<P::Refine>` | $\Sigma_* = K_{**} - V^\top V$ with $V = L^{-1}K_*$, an $m \times m$ column-major matrix |
| `predict_covariance<P>(...)` | returns `PredictiveCovariance<P::Refine>` | Allocates the result |
| `sample<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, n_draws, seed, options, workspace, jitter = nullopt)` | `std::vector<P::Refine>` of length `n_draws * m` | Draw `d` occupies `[d * m, (d + 1) * m)`; the covariance factorization uses the optional `jitter` policy, by default an adaptive one starting at `1e-12` and capped at `1e-2` |
| `loo_predict_into<P>(likelihood, l_factor, alpha, y_train, options, scratch_n_by_n, scratch_diag, out)` | fills `Prediction<P::Refine>` | Closed-form leave-one-out; needs an $n \times n$ scratch and a length-$n$ scratch |
| `loo_predict<P>(likelihood, l_factor, alpha, y_train, options)` | returns `Prediction<P::Refine>` | Allocates the scratch buffers and result |

`P` is deduced from the `QueryWorkspace<P>` argument, but `loo_predict` and `loo_predict_into` have no such argument and need the explicit form `loo_predict<DoublePrecision>(...)`. Draws are deterministic: the same seed gives the same numbers on every thread count.

`QueryWorkspace<P>` has these members:

| Member | Meaning |
|---|---|
| `ensure(n, m, d, nested_depth)` | Grows every buffer so that the workspace can serve $n$ training points, $m$ queries and $d$ features; never shrinks |
| `query_xs`, `query_x` | The transformed queries in `double` and in `Storage` (m × d) |
| `query_k_star`, `query_scratch` | Cross-covariance $K_*$ (later $L^{-1}K_*$) and its scratch (n × m) |
| `query_nested` | `NestedBuffers` for the cross-covariance tree |
| `query_kss`, `query_diag_scratch` | Prior variances $k(x_*,x_*)$ and the diagonal-evaluation scratch |
| `train_n`, `query_m`, `dims` | The capacities last requested |

## Drive the mixed-precision refinement yourself

`cxxgpr/precision/refine.hpp` contains the engine that `MixedPrecision` uses to improve a single-precision solution. You can reuse it for any linear system $B w = b$ for which you have a cheap approximate solver (the `float` factor) and an accurate residual (a `double` matrix or a re-evaluated kernel). The complete source is `mixed_precision_refine.cpp`.

You describe the system by deriving from `RefineSystem` and implementing four functions (a fifth, `accept`, has a default):

| Virtual function | You provide |
|---|---|
| `rhs()` | The right-hand side $b$ as a `std::span<const double>`; it must stay valid during refinement |
| `residual(w, r)` | Writes $r = b - Bw$ into `r` and returns $\lVert B\rVert_\infty$, or an error |
| `correct(r, w)` | Adds an approximate solution of $B\,\delta = r$ to `w` (the `float` triangular solves) |
| `accept(w, tol)` | Optional: return `false` to reject a converged iterate. The default returns `true` |
| `fallback(w)` | Overwrite `w` with an accurate solution (the double-precision solve) |

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [system]
:end-before: // [end]
```

`RefineSystem` is not copyable or movable; its destructor is virtual and its constructor is protected.

The example builds a radial-basis-function Gram matrix, its `float` copy and `float` factor, and the double-precision reference solution.

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [setup]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
float solve only:   relative error 7.951e-05
tolerance 10 n eps = 1.332e-13, at most 10 corrections, stagnation above ratio 0.9 twice in a row
```

Then it runs `refine_in_place` on five variants of the system and prints the `RefinementSummary` of each:

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [run]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
stored float matrix        Converged, corrections 2, fallback 0, relative residual 1.121e-16, error 2.195e-05
stored float, checked      RejectedByAccept, corrections 2, fallback 1, relative residual 1.121e-16, error 0.000e+00
exact double residual      Converged, corrections 2, fallback 0, relative residual 1.258e-16, error 2.493e-12
useless correction (x0)    Stagnated, corrections 2, fallback 1, relative residual 5.922e-09, error 0.000e+00
weak correction (x0.2)     MaxCorrectionsReached, corrections 10, fallback 1, relative residual 7.948e-10, error 0.000e+00
```

Read it row by row:

- *Stored float matrix.* The residual is computed from the `float` matrix. The loop converges (to a residual far below the tolerance), but it has solved the rounded system: the error against the exact double-precision solution is still $2 \times 10^{-5}$.
- *Checked.* The same iteration, but `accept` recomputes the residual with the exact matrix. The check fails, the status is `RejectedByAccept`, and `fallback` replaces `w` with the exact double-precision solve. This is what `MixedPrecision<PromoteStorage>` does for a fitted model.
- *Exact double residual.* The residual uses the double-precision matrix. Two corrections reach an error of $2.5\times10^{-12}$ and the status is `Converged`. This is the idea behind `MixedPrecision<ReevaluateKernel>`.
- *Useless and weak corrections.* Corrections that do not reduce the residual end in `Stagnated`; corrections that reduce it too slowly use up all ten steps (`MaxCorrectionsReached`). Both fall back to the exact solve.

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [convenience]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
refine(): ||r||_inf = 1.521e-14, ||A||_inf = 2.459e+01, tolerance met: 1, error 2.493e-12
||float(A)||_inf = 2.459e+01, ||b - float(A) w||_inf = 1.858e-07
```

### The loop and its constants

`refine_in_place(system, w, residual_scratch)` repeats up to `max_refinement_corrections` times: compute the residual, test convergence, and otherwise apply `system.correct`. It stops with a fallback if two consecutive steps reduce the residual norm by less than the ratio `refinement_stagnation_ratio`. The constants are public:

| Constant | Value | Meaning |
|---|---|---|
| `max_refinement_corrections` | 10 | Correction steps before the fallback |
| `refinement_stagnation_ratio` | 0.9 | A step with $\lVert r_k\rVert_\infty / \lVert r_{k-1}\rVert_\infty$ above this counts as stagnating |
| `refinement_stagnation_steps` | 2 | Consecutive stagnating steps that trigger the fallback |
| `refinement_column_block` | 32 | Column block width the library uses when it re-evaluates a kernel in double precision |
| `refinement_tolerance(dim)` | $10\,\text{dim}\,\varepsilon_{64}$ | Stopping tolerance on the relative residual |

The convergence test is $\lVert r\rVert_\infty / (\lVert B\rVert_\infty \lVert w\rVert_\infty + \lVert b\rVert_\infty) < \text{tol}$, evaluated by `meets_relative_residual(b_inf, w, b, residual, tol)`. It rejects non-finite operands, a non-positive tolerance and a zero denominator. `RefinementSummary` reports `status` (a `RefinementStatus`), `iterations` (corrections applied before stopping), `final_relative_residual`, and `used_fallback()`, which is true for every status except `Converged`. `RefinementSummary` has a defaulted `operator==`.

| `RefinementStatus` | Meaning |
|---|---|
| `Converged` | The residual test passed and `accept` returned `true` |
| `RejectedByAccept` | The residual test passed but `accept` returned `false`; `fallback` ran |
| `Stagnated` | Two consecutive steps failed to reduce the residual enough; `fallback` ran |
| `MaxCorrectionsReached` | `max_refinement_corrections` steps without convergence; `fallback` ran |

Errors: `refine_in_place` returns `EmptyInput` for an empty `w` or right-hand side, `LengthMismatch` if their sizes differ, and `WorkspaceTooSmall` if `residual_scratch` is shorter than `w`. It allocates nothing itself. `refine(system, w)` is the convenience wrapper: it takes `w` by value, allocates the scratch and returns the refined vector.

Helpers:

| Function | Purpose |
|---|---|
| `inf_norm(values)` | $\max_i \lvert v_i\rvert$; 0 for an empty span, `NaN` if any entry is `NaN` |
| `promoted_residual_into(a_f32, alpha, y, residual, row_abs_scratch, sum_scratch)` | $r = y - A_{f32}\alpha$ with `double` accumulation and caller-owned scratch; returns $\lVert A_{f32}\rVert_\infty$ |
| `promoted_residual(a_f32, alpha, y, residual)` | The same with internal scratch |
| `meets_relative_residual(...)` | The convergence test above |
| `refinement_tolerance(dim)` | The tolerance $10\,\text{dim}\,\varepsilon_{64}$ |
| `symmetrize_lower(matrix)` | Mirror the lower triangle of a view into the upper one (see above) |

## API summary

The tables above list every function with its role. As a checklist of the headers covered on this page:

| Header | Symbols |
|---|---|
| `linalg/cholesky.hpp` | `cholesky_lower_in_place`, `cholesky_lower_with_jitter`, `solve_lower_in_place`, `solve_lower_transpose_in_place`, `solve_llt_in_place`, `inv_diag_from_chol_l`, `log_det_from_l`, `half_log_det_from_l`, `solve_lower_in_place_unchecked`, `solve_lower_transpose_in_place_unchecked`, `solve_llt_in_place_unchecked` |
| `linalg/dense.hpp` | `add_to_diag`, `copy_lower_triangle`, `symmetrize_lower`, `frobenius_lower`, `dot`, `matvec_columns` |
| `linalg/parallel.hpp` | `lower_area`, `lower_block_start`, `openmp_max_threads`, `parallel_for_lower_blocks`, `parallel_for_lower_blocks_raw`, `parallel_lower_reduce_cols`, `parallel_lower_reduce_cols_raw` |
| `workspace/buffers.hpp` | `CholeskyBuffer`, `FitPolicy`, `HessianScratch`, `WorkspaceCore`, `FitBuffers`, `QueryWorkspace` |
| `workspace/mll.hpp` | `factor_and_mll`, `mll_and_gradient_into`, `hessian_into`, `value_gradient_hessian_into` |
| `workspace/inference.hpp` | `predict_into`, `predict`, `predict_covariance_into`, `predict_covariance`, `sample`, `loo_predict_into`, `loo_predict` |
| `workspace/leaf_cache.hpp` | `OptimizerHasChangeIndicesConcept`, `OptimizerTraits`, `optimizer_uses_change_indices_v`, `require_change_indices`, `LeafCache` |
| `precision/refine.hpp` | `max_refinement_corrections`, `refinement_stagnation_ratio`, `refinement_stagnation_steps`, `refinement_column_block`, `refinement_tolerance`, `inf_norm`, `symmetrize_lower`, `meets_relative_residual`, `promoted_residual_into`, `promoted_residual`, `RefinementStatus`, `RefinementSummary`, `RefineSystem`, `refine_in_place`, `refine` |
