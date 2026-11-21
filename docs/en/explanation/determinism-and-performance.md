# Determinism and performance

cxxgpr makes two promises that usually pull in opposite directions: results that can be reproduced bit for bit, and a fast, parallel, allocation-free implementation. This page explains how the library gets both, what exactly is guaranteed, and what is not. It is background reading; for the practical settings see {doc}`../how-to/optimize-performance`.

## What "reproducible" means here

Reproducibility has levels, and it is useful to be exact about which one a statement refers to.

| Situation | Same result? |
|---|---|
| Same program, same data, run again | Yes. The library has no hidden state, no clock and no unseeded randomness. |
| Same build, different number of OpenMP threads | Yes, bit for bit, for fitting, prediction, covariance and sampling. |
| Same build, different but equal `FittedGpr` (copy, `predict` versus `predict_into`) | Yes. |
| Same source, different compiler, optimization level, operating system or `CXXGPR_X86_ISA` | No guarantee. Results agree to rounding error, not to the last bit. |
| Random streams (`Xoshiro256PlusPlus::next_u64`) across platforms | Yes: it is pure integer arithmetic. |

The third and fourth rows are the reason the library is built without fast-math and with a fixed instruction set per build: within one build the arithmetic is a fixed sequence of IEEE operations, and the guarantee is that threads do not change that sequence.

## Deterministic parallel reductions

Floating-point addition is not associative. A parallel sum that lets each thread add its share and then combines the threads' partial sums gives a different rounding pattern for every thread count, and the difference grows through a Cholesky factorization and an optimizer into visibly different hyperparameters. cxxgpr avoids this by never letting the thread count influence the order of any floating-point operation that contributes to a result.

The strategy has two parts.

**Independent columns.** The work is arranged so that each column (or block of columns) of a matrix is computed by exactly one thread, from the same inputs, with the same instruction sequence that a single thread would use. Which thread does it does not matter, because the column's value does not depend on the others.

**Serial combination in a fixed order.** Whenever columns must be combined into one number (a gradient entry is $\langle W, \partial K/\partial\theta\rangle_F$, a sum over the whole matrix), each thread writes its column's contribution into its own slot of a preallocated array, and a single thread then adds the slots in ascending column order with Neumaier compensation.

```{mermaid}
flowchart LR
  subgraph par ["Parallel phase: independent columns"]
    T0["Thread 0: columns 0..a"] --> P0["partial[0..a]"]
    T1["Thread 1: columns a..b"] --> P1["partial[a..b]"]
    T2["Thread 2: columns b..n"] --> P2["partial[b..n]"]
  end
  P0 --> S["Serial phase: add partial[0], partial[1], ... partial[n-1]<br/>in this order, Neumaier compensated"]
  P1 --> S
  P2 --> S
  S --> R["Result: identical for any number of threads"]
```

The public helper `linalg::parallel_lower_reduce_cols` implements exactly this pattern and you can use it for your own sums ({doc}`../how-to/use-low-level-building-blocks`). The partials array is the `col_partials` buffer that `WorkspaceCore` preallocates, which is why the reduction needs no allocation either.

### Balanced blocks

A lower-triangular loop has unequal work: column $j$ of an $n \times n$ matrix has $n - j$ entries. Splitting the columns into equal-sized blocks would leave the thread with the first block doing most of the work. `linalg::lower_area(n, c)` counts the entries in the first $c$ columns,

$$
\text{area}(c) = c\,n - \frac{c(c-1)}{2},
$$

and `linalg::lower_block_start(n, i, B)` returns the column where the cumulative area is closest to $i/B$ of the total $n(n+1)/2$. For $n = 1000$ and four blocks the boundaries are 0, 134, 293, 500 and 1000: the first block has few, long columns; the last has many short ones. The boundaries are computed with integer arithmetic that cannot overflow.

The partition depends on the thread count, but the results do not, because of the two rules above. The same holds for the blocked Cholesky factorization and the triangular solves with many right-hand sides: the number of threads changes who computes a block and when, never the sequence of operations that produces any one value.

### Where determinism does not extend

Anything that you control and that varies between runs breaks reproducibility: unseeded random inputs, a different order of the training rows, or a different build. The thread count itself is configured outside the library through OpenMP, so it is never part of the model.

## Random numbers

All randomness in cxxgpr is explicit and seeded: posterior samples (`FittedGpr::sample`, `sample` in the workspace API), optimizer restarts and `FastSimulatedAnnealing`. They draw from `Xoshiro256PlusPlus`, also available as `SeededRng`, a 256-bit generator with a very small state and good statistical quality. A 64-bit seed is expanded into the four state words with SplitMix64, so that even seeds such as 0 and 1 give well-separated streams.

- `next_u64()` is pure integer arithmetic and returns the same sequence on every platform; the generator is Xoshiro256++.
- `unit()` takes the upper 53 bits and scales them into $[0, 1)$, which is exact and portable; `open_unit()` clamps that value at least $2^{-53}$ away from 0 and 1 so that logarithms stay finite.
- `up_to(max)` draws an unbiased integer in $[0, \text{max}]$ by Lemire's rejection method.
- `unit_normal()` produces standard normal values with the Box-Muller transform. It calls the C library's `log`, `sqrt` and trigonometric functions, so a different C library can differ in the last bit of a draw while the underlying integer stream is identical.

The generator is cheap to copy and compare (`operator==`), so a draw sequence can be replayed from a saved state.

Samples of a model draw $z \sim \mathcal{N}(0, I)$ and return $\mu_* + L_* z$ with $L_*$ the Cholesky factor of the posterior covariance. The same seed therefore gives the same samples for the same model, whatever the thread count.

## Allocation policy

Heap allocation is slow, takes a lock inside the allocator and has unpredictable latency. In an optimization loop that evaluates the same $n \times n$ problem hundreds of times, it is also pure waste. cxxgpr therefore allocates once, up front, and then reuses.

```{mermaid}
flowchart LR
  A["fit() start:<br/>size everything for n points"] --> B["Optimization loop:<br/>value, gradient, Hessian<br/>(no allocation)"]
  B --> C["fit() returns:<br/>training buffers released"]
  C --> D["FittedGpr keeps L, alpha, X<br/>and a QueryWorkspace"]
  D --> E["predict_into, 1st call:<br/>workspace grows"]
  E --> F["predict_into, later calls:<br/>no allocation"]
```

The rules behind the diagram:

- **Training buffers are sized once.** `FitBuffers<P>` knows $n$ at the start of `fit` and allocates the matrices, scratch and (optionally) distance cache in one go. The Hessian scratch is the exception: it is the largest buffer ($p$ matrices of size $n \times n$) and is created on the first Hessian call, so optimizers that do not use second derivatives never pay for it.
- **Inside the loop nothing is allocated.** Marginal likelihood plus gradient evaluations reuse the buffers; kernel trees evaluate into preallocated `NestedBuffers` whose depth was computed from the tree when it was compiled.
- **Training memory is released at the end.** A `FittedGpr` keeps only what prediction needs.
- **Prediction memory grows monotonically.** The first call sizes the `QueryWorkspace`. Later calls with a batch of equal or smaller size reuse it without allocating; a larger batch grows it once.

The guarantee covers the hot paths: a negative log marginal likelihood evaluation with gradient, and `predict_into` after warm-up. Calls that return a new object (`predict`, `predict_covariance`, `sample`, `loo_predict`) must allocate their result. The test suite enforces the guarantee mechanically: Eigen is built with `EIGEN_RUNTIME_NO_MALLOC` so that any internal allocation inside an audited scope aborts, and a counter on the global `operator new` catches everything else.

### The trade-off

Preallocation means memory is committed for the largest problem even if an iteration needs less, and the dense matrices dominate: a double-precision $n \times n$ matrix needs $8n^2$ bytes, 8 MB at $n = 1000$ and 800 MB at $n = 10\,000$. `FitPolicy` lets you choose the lean variant at some cost in time (see {doc}`../how-to/optimize-performance`), and `SinglePrecision` and `MixedPrecision` halve the matrix storage. The $O(n^3)$ time and $O(n^2)$ memory of exact inference are inherent; the allocation policy decides only how predictable they are.

## SIMD kernels and the instruction set

The expensive loops are written for SIMD: Eigen provides the dense linear algebra, and cxxgpr adds its own vectorized kernels for the exponential in the kernel leaves, for non-finite checks, for the packed triangular products and for distance computation. They exist in three tiers, selected when the library is compiled:

| Tier | Selected by | Typical hardware |
|---|---|---|
| AVX-512 | `-DCXXGPR_X86_ISA=AVX512` | Recent server and some desktop CPUs |
| AVX2 with FMA | `-DCXXGPR_X86_ISA=AVX2` (the default on x86-64) | Almost all x86-64 CPUs since about 2013 |
| SSE2 | `-DCXXGPR_X86_ISA=SSE2` | Every x86-64 CPU |

On 64-bit Arm the cxxgpr kernels use their portable fallbacks and the option does not exist. There is no run-time dispatch: the tier is part of the compiled library, and a library built for AVX2 does not run on a CPU without it. Because fused multiply-add and different vector widths change rounding, two ISA tiers give results that agree to rounding error but not bit for bit, which is the fourth row of the table at the top. See {doc}`../reference/build-options` to choose.

The fast exponential `KernelExp::FastApprox` is an independent speed/accuracy choice inside these kernels: a range-reduced degree-7 polynomial instead of `std::exp`. It changes results at the level of $10^{-5}$ relative and is therefore opt-in.

## Why exactness costs time

Compensated sums, a fixed reduction order, no fast-math and serial combination steps cost a little speed. The serial phases are $O(n)$ or $O(n^2)$ additions after $O(n^3)$ factorization work, so they do not limit scaling, and the measurements in {doc}`../how-to/optimize-performance` show that speedups of about four to six times on 16 threads are achieved with the deterministic implementation. The library chooses reproducibility first because a fitted model that changes with the number of cores is hard to test, to debug and to trust.
