# 0004: External BLAS/LAPACK backends, pinned FetchContent for herumi/fmath, and arbitrary-d ARD incremental updates

- Status: Accepted
- Date: 2026-11-23
- Issue: #130
- Amends: [0001-exact-gpr-cpp23-scope.md](0001-exact-gpr-cpp23-scope.md)

## Context

Three structural issues remained in the internal numeric and build layers of `cxxgpr`:

1. **Custom ISA-specific BLAS-3 kernels**: `src/linalg/packed_blas3.{hpp,cpp}` and `src/linalg/simd_blas3.hpp` re-implemented micro-kernels for AVX-512, AVX2, and SSE2 instead of delegating dense matrix multiplications (`GEMM`), symmetric rank-$k$ updates (`SYRK`), and triangular solves (`TRSM`) to tuned vendor BLAS libraries such as OpenBLAS or Intel oneMKL.
2. **Custom approximate exponential and overly restrictive `FetchContent` policy**: `src/kernel/exp_simd.hpp` maintained hand-written degree-7 polynomial exp intrinsics, while `.cursor/rules/dependencies.mdc` prohibited commit-pinned `FetchContent` even when a library (`herumi/fmath`) is absent from vcpkg.
3. **ARD incremental cache limitations and redundant $s^2$ passes**:
   - `LeafCache<T>` used a fixed 64-dimension capacity (`k_ard_ls_stride_ = 64U`) that silently disabled incremental ARD updates whenever $d > 64$, while `evaluate_incremental` bypassed incremental updates under `DistanceCachePolicy::Uncached`.
   - Rejected FSA single-dimension proposals could overwrite `ard_prev_ls_` or mutate `ard_s2_cache` in-place unless backed up, causing corruption or $2$-dim full rebuilds.
   - `leaf_apply_ard_packed_incremental` scanned `ard_s2_cache` twice (once to apply $\Delta w_k (\Delta x_k)^2$, once to evaluate the kernel transform).
   - `DistanceGpr` (`DistanceObjective::value_at_changes`) ignored `changed_indices` and fell back to full `value(params)`.
   - `factor_and_mll` computed $s^2$ and discarded it so `leaf_ard_weighted_grads` recomputed $s^2 = \sum_d \ell_d^{-2} (\Delta x_d)^2$ from scratch during L-BFGS and Trust-Region evaluations.

## Decision

1. **Selectable dense linear algebra backend (`CXXGPR_BLAS_BACKEND`)**:
   - Add `openblas` (`0.3.34`, `threads` feature, `dynamic-arch` feature on `!windows | mingw` for portable runtime CPU dispatch across cached CI runners, BSD-3-Clause) to `vcpkg.json`, and support `CXXGPR_BLAS_BACKEND=AUTO|OpenBLAS|MKL|BuiltIn`.
   - Under `AUTO`, Linux/MinGW selects OpenBLAS (then Intel oneMKL, then `BuiltIn`), whereas Windows MSVC defaults to `BuiltIn` because vcpkg's Windows MSVC OpenBLAS builds generic C kernels (`OPENBLAS_C_MSVC`, no x86-64 assembly) and exhibits severe spin-thread contention under CPU affinity / hybrid core topologies. Intel oneMKL remains explicitly selectable via `-DCXXGPR_BLAS_BACKEND=MKL`.
   - Keep `<cblas.h>` / `<mkl.h>` strictly isolated inside `src/linalg/blas_backend.cpp` (plus version reporting in `src/core/build_info.cpp`), while compiling `src/linalg/packed_blas3.{hpp,cpp}` and `src/linalg/simd_blas3.hpp` for `BuiltIn` and also using them under `MKL` as a multi-threaded small-matrix (`double`) fallback ($n \le 256$ Cholesky, $n \le 384$ $W$) and promoted posterior covariance helper to prevent dual-OpenMP (`vcomp140` / `libiomp5md`) contention on Windows MSVC.
   - Implement blocked Cholesky, triangular/symmetric BLAS-3 updates, and $\frac{2}{3} n^3$-FLOP lower-triangular $W = \alpha \alpha^T - (L L^T)^{-1}$ (`blas_compute_w_lower`) across all three backends (`BuiltIn` 2-stage SIMD+OpenMP, `MKL` `LAPACKE_?potrf` + `LAPACKE_?potri`, and `OpenBLAS` blocked `trtri + lauum` + CBLAS `trsm`/`syrk`), keeping the strictly upper triangle untouched and allocating zero heap memory in hot paths (`NoAllocationScope`).
   - Relax cross-thread-count bitwise identity expectations specifically on multi-threaded external BLAS operations while preserving single-configuration determinism and keeping OpenMP coordinate/Frobenius reductions (and `BuiltIn` SIMD kernels) bit-exact across thread counts.
2. **Commit-pinned `FetchContent` policy & full-accuracy self-contained Estrin minimax SIMD `exp`**:
   - Amend [ADR 0001](0001-exact-gpr-cpp23-scope.md) and `.cursor/rules/dependencies.mdc` to permit commit-pinned CMake `FetchContent` (`GIT_TAG` = 40-char SHA, verified SHA-256 license notice) when an external dependency is unavailable on vcpkg.
   - `herumi/fmath` (`a81f8e53e9f007ed5df355bb4a99ff799fd75066`, BSD-3-Clause) was initially integrated via `FetchContent`, but superseded before release because its cross-TU array API prevented register fusion with Matérn polynomial factors (requiring scratch buffers and L1 passes) and its single-precision path introduced `3.4e-6` relative error without Cody-Waite reduction.
   - `KernelExp::FastApprox` therefore uses `cxxgpr`'s self-contained inline SIMD Estrin minimax approximation in `include/cxxgpr/kernel/exp.hpp` + `src/kernel/exp_simd.hpp` (pinned Chebyshev degree 11 for `double` $\le 2\text{ ULP}$ / $< 10^{-14}$ and degree 6 for `float` $\le 2\text{ ULP}$ / $< 5\times 10^{-7}$, Cody-Waite subtraction, 2x-unrolled AVX2+FMA / AVX-512 / SSE2 fused with Matérn 1/2, 3/2, and 5/2 in registers) with zero third-party `FetchContent` build dependencies.
3. **Memory-bounded $O(n^2)$ ARD incremental updates and $s^2$ reuse**:
   - Keep `ard_s2_storage_` at $O(n^2)$ per leaf (never $O(d n^2)$) and size `ard_prev_ls_` dynamically to exact ARD dimensionality $O(d)$ ($8d$ bytes), eliminating `k_ard_ls_stride_ = 64U`.
   - Separate trial deltas from accepted baselines in `LeafCache<T>` so rejected FSA proposals discard in $O(1)$ without mutating `ard_s2_cache`.
   - Support $O(n^2)$ single-dim incremental updates both from packed distance slices (`DistanceCachePolicy::Full`/`Auto`) and directly from column `x.col(dim)` under `Uncached`.
   - Fuse delta $s^2$ update and radial kernel evaluation into a single column-wise pass.
   - Add per-leaf incremental $s^2$ caching to `CompiledDistanceKernel` and `DistanceObjective::value_at_changes`.
   - Cache pre-allocated per-ARD-leaf packed $s^2$ during `factor_and_mll` (when `keep_products > 0`) and reuse it directly in `leaf_ard_weighted_grads`.

## Consequences

- `cxxgpr` delegates large dense linear algebra to maintained external implementations (`OpenBLAS` / `oneMKL`) on non-MSVC or explicit builds and uses zero-dependency self-contained inline SIMD for small MSVC matrices and `1–2 ULP` `KernelExp::FastApprox` (`exp`), without exposing any third-party headers in `include/cxxgpr/`.
- Memory footprint per ARD leaf stays $O(n^2)$ for any dimension $d$, while eliminating redundant $O(d n^2)$ work across both `Gpr` and `DistanceGpr` optimization loops.
