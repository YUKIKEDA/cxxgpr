# How to optimize performance

This guide shows how to make cxxgpr predict without touching the heap, how to control the number of threads, how to choose a precision policy, and how to trade memory for speed while fitting. It assumes you already have a working model; for the full list of options on `Gpr` see {doc}`configure-fit`, and for why the library behaves this way see {doc}`../explanation/determinism-and-performance`.

:::{note}
The numbers on this page were measured with the library built in Release mode with MSVC, AVX2 code generation (the default), on a desktop x86-64 machine with 16 hardware threads. They are medians of repeated runs that varied by up to about 20 percent. Treat them as an order-of-magnitude guide and measure your own workload before you decide anything.
:::

## Predict without allocating

A `FittedGpr` owns a `QueryWorkspace` that holds every temporary a prediction needs: the transformed queries, the cross-covariance, the triangular-solve result and the nested kernel scratch. `FittedGpr::predict_into` and `FittedGpr::predict_with_into` write into a `Prediction` that you own. After one warm-up call that sizes the workspace and your output vectors, further calls with the same or a smaller batch size perform no heap allocation.

The program below installs a counting `operator new` (the library is a static library, so its allocations go through yours), fits a two-feature model, and counts allocations.

```{literalinclude} /../examples/zero_alloc_predict.cpp
:language: cpp
:start-after: // [counter]
:end-before: // [end]
```

Warm up once, then predict in a loop with the same `Prediction` object:

```{literalinclude} /../examples/zero_alloc_predict.cpp
:language: cpp
:start-after: // [warmup]
:end-before: // [end]
:dedent: 4
```

```{literalinclude} /../examples/zero_alloc_predict.cpp
:language: cpp
:start-after: // [steady]
:end-before: // [end]
:dedent: 4
```

The complete program prints:

```{code-block} text
warm-up call allocated 8 times
1000 further predict_into calls allocated 0 times
one predict() call allocated 2 times
predict_into equals predict: 1
checksum of the 64-point prediction: edb08ca7add4082b
larger batch (256 points): first call 8 allocations, second call 0
```

`predict` returns a new `Prediction`, so it allocates its two result vectors on every call. `predict_into` and `predict` run the same code and return bit-identical numbers.

### Rules for allocation-free prediction

1. **Reuse one `Prediction` object.** Its `mean` and `variance` vectors keep their capacity between calls.
2. **Any batch size works.** The workspace only grows. A larger batch allocates once (the last line above); smaller batches reuse the existing space, and everything is allocation-free from then on.
3. **Pass `PredictOptions` to choose the variance.** `predict_into(xs, out, options)` and `predict_with_into(xs, options, out)` take a `PredictOptions`; `VarianceKind::Latent` omits the observation noise from the variance and `VarianceKind::Observation` (the default) includes it.
4. **Use one `FittedGpr` per thread.** See the warning below.

:::{warning}
`predict_into` is a `const` member function, but it modifies the model's internal `QueryWorkspace`. Two threads must never call `predict`, `predict_into`, `predict_covariance` or `sample` on the same `FittedGpr` object at the same time. Copy the model instead: the copy constructor gives the copy its own, empty `QueryWorkspace`, so each worker thread can have an independent object.
:::

:::{note}
The counter above sees every call to the global `operator new`, which covers `std::vector` growth. It does not see allocations made by Eigen's own aligned allocator inside the library. The project's own test suite audits both: it forbids Eigen's runtime allocation and counts `operator new` while the prediction runs. Your counter is therefore a good regression check for your own code paths, but not a proof about Eigen internals.
:::

### Low-level equivalents

If you manage the factor and the weights yourself, the free function `predict_into<P>` takes a `QueryWorkspace<P>` you construct. The same holds for `predict_covariance_into<P>` and `loo_predict_into<P>` (which needs two scratch buffers instead of a `QueryWorkspace`). {doc}`use-low-level-building-blocks` shows all of them.

## Know which buffers exist

cxxgpr keeps its temporaries in a small number of named workspace types. Knowing who owns them tells you when memory is allocated and when it is released.

| Type | Owned by | Allocated | Released |
|---|---|---|---|
| `FitBuffers<P>` | the training run inside `Gpr::fit` / `Gpr::factor` | once at the start of a fit, sized for `n` points | when `fit` or `factor` returns; a `FittedGpr` does not keep it |
| `WorkspaceCore<P>` | `FitBuffers<P>::core` | with `FitBuffers`; `kernel_scratch` is allocated lazily | with `FitBuffers` |
| `HessianScratch<S>` | `WorkspaceCore::hessian` | on the first Hessian evaluation (`ensure`), reused afterwards | with `FitBuffers` |
| `NestedBuffers<S>` | `WorkspaceCore::nested` and `weighted`, `QueryWorkspace::query_nested` | sized from the kernel tree depth | with the owner |
| `LeafCache<S>` | the training objective of optimizers that report changed coordinates | at the start of the fit, one Gram matrix per kernel leaf | when `fit` returns |
| `QueryWorkspace<P>` | the `FittedGpr` | on the first prediction, grown on demand | when the `FittedGpr` is destroyed |

Three rules follow from the table:

- **Training memory is temporary.** After `fit` returns, only the fitted model remains: the factor $L$, the weights $\alpha$, the training data and the query workspace.
- **Prediction memory is permanent but bounded.** It grows to the largest batch you ever predict, and it is never shrunk.
- **A `FittedGpr` copy starts with an empty `QueryWorkspace`.** Its first prediction allocates again.

`Gpr::factor` (no optimization) always trains with the memory-saving buffer policy, whatever `with_prefer_speed()` says. Optimizers that only need function values (`NelderMead` and `FastSimulatedAnnealing`) never build the gradient weight matrix, so they train with `CholeskyBuffer::Reuse` as well; for `FastSimulatedAnnealing`, `CholeskyBuffer::Retain` then only decides whether the incremental `LeafCache` is used.

## Control the number of threads

cxxgpr parallelizes its dense kernels with OpenMP. There is no thread-count setting in the `Gpr` API; the library follows the OpenMP runtime.

```{code-block} sh
# Linux, macOS, Git Bash
OMP_NUM_THREADS=4 ./my_program
```

```{code-block} powershell
# Windows PowerShell
$env:OMP_NUM_THREADS = 4
.\my_program.exe
```

From inside your program, call `omp_set_num_threads(k)` from `<omp.h>` before you fit or predict. This requires that your own target is compiled with OpenMP (`/openmp` on MSVC, `-fopenmp` on GCC and Clang, or `find_package(OpenMP)` and `OpenMP::OpenMP_CXX` in CMake). `linalg::openmp_max_threads()` returns the number of threads the library will use right now (1 when it was built without OpenMP), and the `linalg::parallel_*` helpers take an explicit `num_threads` argument (0 means "ask OpenMP").

The library stays serial for small problems: it only opens a parallel region when more than one thread is available and the matrix is large enough to pay for it. Extra threads therefore do nothing for tens of training points.

Measured on the same machine, one gradient evaluation of the negative log marginal likelihood (`n = 1000`, four features) and one `factor` call (`n = 2000`):

| Threads | `mll_and_gradient_into` | `Gpr::factor` |
|---|---|---|
| 1 | about 33 ms | about 200 ms |
| 16 | about 7.6 ms | about 32 ms |

The results do not depend on the thread count. The example above prints the same checksum, including the L-BFGS fit, for every setting:

```{code-block} text
$ OMP_NUM_THREADS=1  ./zero_alloc_predict | grep checksum
checksum of the 64-point prediction: edb08ca7add4082b
$ OMP_NUM_THREADS=3  ./zero_alloc_predict | grep checksum
checksum of the 64-point prediction: edb08ca7add4082b
$ OMP_NUM_THREADS=16 ./zero_alloc_predict | grep checksum
checksum of the 64-point prediction: edb08ca7add4082b
```

How this is achieved is explained in {doc}`../explanation/determinism-and-performance`. If you call cxxgpr from your own parallel loop, give each thread its own `FittedGpr` copy and make sure the OpenMP runtime does not run nested parallel regions with `OMP_NUM_THREADS` threads each; otherwise you oversubscribe the machine.

When using Intel oneMKL (`-DCXXGPR_BLAS_BACKEND=MKL`) on hybrid P-core / E-core processors or alongside MSVC OpenMP loops, note that Windows oneMKL (`libiomp5md.dll`) defaults to `KMP_BLOCKTIME=0` (passive wait). For dedicated workloads executing rapid kernel evaluations or small-to-medium optimization iterations, setting `KMP_BLOCKTIME=200` with `KMP_AFFINITY=disabled` and pinning threads to physical performance cores (`OMP_NUM_THREADS` / `MKL_NUM_THREADS` matching P-core count) avoids repeated thread wake-up latency and E-core spill; leave `KMP_BLOCKTIME=0` when sharing the machine with other processes or independent worker pools.

## Choose a precision policy

`Gpr` and `FittedGpr` take the precision as their second template argument: `Gpr<optimizer::Lbfgs, SinglePrecision>`. The four policies are `DoublePrecision`, `SinglePrecision`, `MixedPrecision<PromoteStorage>` and `MixedPrecision<ReevaluateKernel>`.

The program below factors the same badly conditioned system (a smooth signal, noise variance `2e-5`, 200 points) under every policy at fixed hyperparameters, so the arithmetic is the only difference:

```{literalinclude} /../examples/precision_policies.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
policy                    factor bytes   alpha error  mean error   var error  refinement
DoublePrecision                 320000      0.00e+00    0.00e+00    0.00e+00  none
SinglePrecision                 160000      5.17e-02    1.11e-03    1.01e-07  none
Mixed<PromoteStorage>           160000      1.32e-02    3.83e-04    1.13e-07  Converged, 4 corrections
Mixed<ReevaluateKernel>         160000      1.72e-06    4.93e-04    1.13e-07  Converged, 4 corrections
Double + FastApprox             320000      4.03e-04    1.37e-05    2.25e-10  none
```

"Alpha error" is the largest difference of the prediction weights from the double-precision weights, relative to their largest entry. "Mean error" and "var error" are the largest absolute differences in the predictions at 50 test points.

What the numbers show:

- `SinglePrecision` halves the memory of the factor and is the fastest policy in the table below, but its weights are off by 5 percent on this badly conditioned problem.
- `MixedPrecision` factors in single precision and then refines the weights with double-precision residuals. On this problem the refined weights are correct to about $10^{-6}$ (`ReevaluateKernel`); `PromoteStorage` is only about four times closer than single precision, because it solves the `float`-stored system (see below). The *predictions* still improve only by a factor of about 2, because the cross-covariance $K_*$ is evaluated in single precision.
- `PromoteStorage` refines against the stored single-precision matrix. Its weights solve *that* system to double accuracy but keep the effect of rounding the matrix to `float` (about $10^{-7}$ per entry, amplified by the conditioning). The library accepts the result when the residual against the exact double-precision kernel is within ten times `std::numeric_limits<float>::epsilon()`. It does not require the $10\,n\,\varepsilon_{64}$ tolerance, which a kernel matrix rounded to `float` does not meet unless all of its entries happen to be exactly representable in `float`. Read `FittedGpr::refinement_summary()` to see the outcome.
- `ReevaluateKernel` checks the residual against the exact kernel in double precision, column block by column block, and converges to the double-precision solution.

Timing for `Gpr<optimizer::Fixed, P>::factor` (median of three runs, four features, Release build, 16-thread x86-64 desktop, run-to-run variation up to about 10 percent), in milliseconds with 16 threads:

| `n` | `DoublePrecision` | `SinglePrecision` | `MixedPrecision<PromoteStorage>` | `MixedPrecision<ReevaluateKernel>` |
|---|---|---|---|---|
| 500 | 1.7 | 1.1 | 4.4 | 6.2 |
| 1000 | 6.4 | 4.3 | 13 | 18 |
| 2000 | 28 | 17 | 53 | 70 |
| 4000 | 163 | 83 | 156 | 135 |
| 6000 | 473 | 235 | 472 | 408 |

With one thread, `n = 2000` takes 91 ms (double), 53 ms (single), 87 ms (`PromoteStorage`) and 99 ms (`ReevaluateKernel`); `n = 4000` takes 656, 323, 524 and 521 ms.

Why the mixed policies behave like this:

- Every refinement step evaluates a residual in double precision, and a converged refinement needs two of them (the starting one and the one that confirms convergence). For `ReevaluateKernel` each residual evaluates the exact kernel again, column block by column block and for the full matrix rather than one triangle, so it costs a few times the kernel evaluation of the double-precision path on top of the single-precision factorization. For `PromoteStorage` the residual reads the stored `float` matrix, and only the final acceptance check evaluates the exact kernel once. The factorization saves $O(n^3)$ work and the refinement costs $O(n^2)$, so both policies overtake `DoublePrecision` as `n` grows. On this machine they are slower below roughly `n = 3000`, and from there on about as fast (`PromoteStorage`) or about 1.2 times faster (`ReevaluateKernel`).
- Neither mixed policy is faster than `SinglePrecision`; their benefit over it is accuracy of the weights, and over `DoublePrecision` the half-size factor.

Prefer `DoublePrecision` unless memory is the limit, `SinglePrecision` when you can tolerate its error, and `MixedPrecision<ReevaluateKernel>` when you need single-precision factor storage with weights checked against the exact system, or for large `n` where it is also faster than double precision. `MixedPrecision<PromoteStorage>` sits between single precision and `ReevaluateKernel`: it avoids the repeated kernel evaluation, but the weights keep the storage rounding. See {doc}`../explanation/numerical-stability` for the refinement algorithm and {doc}`use-low-level-building-blocks` to drive it yourself.

## Trade memory for speed while fitting

Two enumerations decide how much memory training uses. `FitPolicy` bundles them, and `Gpr::with_distance_cache_policy`, `Gpr::with_cholesky_buffer`, `Gpr::with_prefer_speed` and `Gpr::with_prefer_memory` set them (see {doc}`configure-fit`).

| Setting | Meaning | Extra memory for `n` points |
|---|---|---|
| `DistanceCachePolicy::Cached` | Compute the training distances once and reuse them in every iteration | $n^2$ values for isotropic kernels, $d \cdot n(n+1)/2$ for ARD kernels, nothing for kernels that read no distances |
| `DistanceCachePolicy::Uncached` | Recompute distances from the coordinates every time | none |
| `CholeskyBuffer::Retain` | Keep $L$ and build the gradient weight matrix $W$ in its own buffer | one $n \times n$ matrix |
| `CholeskyBuffer::Reuse` | Build $W$ over the factor and refactor $L$ afterwards | none, at the cost of one more Cholesky factorization |

`FitPolicy::prefer_speed()` is `Cached` plus `Retain` and is the default. `FitPolicy::prefer_memory()` is `Uncached` plus `Reuse`.

Measured for one `mll_and_gradient_into` call (`n = 1000`, four features, ARD kernel, 16 threads):

| Policy | Time | Large buffers |
|---|---|---|
| `Cached` + `Retain` (`prefer_speed`) | about 7.6 ms | 24 MB for $K$, scratch and $W$, plus the distance cache |
| `Uncached` + `Reuse` (`prefer_memory`) | about 10 ms | 16 MB, no distance cache |
| `Cached` + `Reuse` | about 10 to 14 ms | |
| `Uncached` + `Retain` | about 6 to 10 ms | |

`Reuse` is what costs time (it refactors $L$ at the end of every gradient evaluation), while the distance cache made little difference here. If you must save memory, start by choosing `Uncached` with `Retain` and measure.

The same switches exist for the low-level buffers: `FitBuffers<P>::create(n, cache, buffer)` or `FitBuffers<P>::create(n, policy)`. `overwrites_cholesky()` tells you whether the gradient pass clobbers $L$ and refactors it afterwards. See {doc}`use-low-level-building-blocks`.

## Use the fast exponential

`Gpr::with_kernel_exp(KernelExp::FastApprox)` replaces `std::exp` inside the built-in kernel leaves with `herumi/fmath` fast exponential evaluation. It does not affect `exp(log theta)` parameter transforms or `CustomKernel` leaves.

In the table above, the `Double + FastApprox` row shows the price: predictions differ by $1.4 \times 10^{-5}$ from the accurate ones, and the weights by $4 \times 10^{-4}$ on this ill-conditioned problem. At `n = 1000` the gradient evaluation was no faster (about 7.5 to 8.4 ms against 7.6 ms), because the Cholesky factorization dominates. `FastApprox` can only help when kernel evaluation is a large share of the work: many prediction points relative to the training size, or low-dimensional problems where the factorization is cheap. Measure before you switch it on, and keep `KernelExp::Accurate` otherwise.

## API summary

| Symbol | Purpose |
|---|---|
| `FittedGpr::predict_into`, `FittedGpr::predict_with_into` | Predict into a caller-owned `Prediction`; allocation-free after warm-up |
| `Prediction`, `PredictOptions`, `VarianceKind` | Result container and variance selection (`Latent`, `Observation`) |
| `QueryWorkspace<P>` | Query-time buffers; `ensure` sizes them |
| `FitBuffers<P>`, `WorkspaceCore<P>`, `HessianScratch<S>` | Training buffers and their lazily allocated Hessian part |
| `FitPolicy`, `CholeskyBuffer`, `DistanceCachePolicy` | Memory and speed policy; `prefer_speed`, `prefer_memory` presets |
| `Retain`, `Reuse` | `CholeskyBuffer` enumerators |
| `Cached`, `Uncached` | `DistanceCachePolicy` enumerators |
| `NestedBuffers<S>` | Contiguous per-level matrices for tree evaluation |
| `LeafCache<S>`, `OptimizerTraits` | Incremental per-leaf Gram matrices for optimizers that change one coordinate at a time |
| `KernelExp` | `Accurate` or `FastApprox` exponential |
| `DoublePrecision`, `SinglePrecision`, `MixedPrecision`, `PromoteStorage`, `ReevaluateKernel` | Precision policies and residual formulas |
| `MixedPrecision::Residual` | Alias naming the residual formula (`PromoteStorage` or `ReevaluateKernel`) of a mixed policy |
| `RefinementSummary`, `RefinementStatus` | Outcome of the mixed-precision refinement of a fitted model |
| `linalg::openmp_max_threads` | Current OpenMP thread budget |
