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

1. **Selectable external BLAS/LAPACK backend (`CXXGPR_BLAS_BACKEND`)**:
   - Add `openblas` (`0.3.34`, `threads` feature, plus `dynamic-arch` on `!windows | mingw` for portable runtime CPU dispatch across cached CI runners, BSD-3-Clause) to `vcpkg.json` as the portable default runtime dependency, and support `CXXGPR_BLAS_BACKEND=AUTO|OpenBLAS|MKL|Eigen` (selecting OpenBLAS when available under `AUTO`, Intel oneMKL when installed/requested, and Eigen as an explicit fallback).
   - Isolate `<cblas.h>` / `<mkl.h>` strictly inside `src/linalg/blas_backend.cpp` (plus version reporting in `src/core/build_info.cpp`), removing `src/linalg/packed_blas3.{hpp,cpp}` and `src/linalg/simd_blas3.hpp`.
   - Because vcpkg builds OpenBLAS with `-DBUILD_WITHOUT_LAPACK=ON` (CBLAS-only), implement blocked Cholesky and triangular/symmetric updates using vendor `cblas_?trsm` + `cblas_?syrk` + panel `Eigen::LLT<..., Eigen::Lower>` (or `LAPACKE_?potrf` when MKL is active), keeping the strictly upper triangle untouched and allocating zero heap memory in hot paths (`NoAllocationScope`).
   - Relax cross-thread-count bitwise identity expectations specifically on multi-threaded external BLAS operations while preserving single-configuration determinism and keeping OpenMP coordinate/Frobenius reductions bit-exact across thread counts.
2. **Permit commit-pinned `FetchContent` for dependencies absent from vcpkg**:
   - Amend [ADR 0001](0001-exact-gpr-cpp23-scope.md) and `.cursor/rules/dependencies.mdc` to allow CMake `FetchContent` when a library is unavailable on vcpkg, provided `GIT_TAG` is pinned to a full 40-character commit SHA and its license text is copied into `third-party-notices/` and verified by `toolchain.third_party_notices`.
   - Fetch `https://github.com/herumi/fmath.git` at pinned commit `a81f8e53e9f007ed5df355bb4a99ff799fd75066` (BSD-3-Clause), compile its `src/cpu.cpp` + `fmath.asm`/`fmath.S` into an internal object library embedded inside `cxxgpr`, and replace custom `fast_exp_nonpos_*` SIMD code in `src/kernel/exp.cpp` / `src/kernel/exp_simd.hpp` with `fmath`.
3. **Memory-bounded $O(n^2)$ ARD incremental updates and $s^2$ reuse**:
   - Keep `ard_s2_storage_` at $O(n^2)$ per leaf (never $O(d n^2)$) and size `ard_prev_ls_` dynamically to exact ARD dimensionality $O(d)$ ($8d$ bytes), eliminating `k_ard_ls_stride_ = 64U`.
   - Separate trial deltas from accepted baselines in `LeafCache<T>` so rejected FSA proposals discard in $O(1)$ without mutating `ard_s2_cache`.
   - Support $O(n^2)$ single-dim incremental updates both from packed distance slices (`DistanceCachePolicy::Full`/`Auto`) and directly from column `x.col(dim)` under `Uncached`.
   - Fuse delta $s^2$ update and radial kernel evaluation into a single column-wise pass.
   - Add per-leaf incremental $s^2$ caching to `CompiledDistanceKernel` and `DistanceObjective::value_at_changes`.
   - Cache pre-allocated per-ARD-leaf packed $s^2$ during `factor_and_mll` (when `keep_products > 0`) and reuse it directly in `leaf_ard_weighted_grads`.

## Consequences

- `cxxgpr` delegates dense linear algebra and fast vectorized `exp` to maintained external implementations (`OpenBLAS` / `oneMKL` and `herumi/fmath`) without exposing any third-party headers in `include/cxxgpr/`.
- Memory footprint per ARD leaf stays $O(n^2)$ for any dimension $d$, while eliminating redundant $O(d n^2)$ work across both `Gpr` and `DistanceGpr` optimization loops.
