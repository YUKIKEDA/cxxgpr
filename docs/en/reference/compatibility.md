# Compatibility

What cxxgpr runs on, how versions are numbered and what a version number promises, and what you must keep consistent between the library and your program. The facts here come from the project's build files, its continuous integration and its quality policy; where something is not tested or not promised, this page says so.

## Supported toolchains

| Item | Supported | Not supported or not tested |
|---|---|---|
| Language standard | C++23, with `std::expected` and its monadic functions | C++20 and earlier |
| MSVC | 19.44 or later (Visual Studio 17.14 or later) on Windows 11 | Older MSVC versions are rejected at configure time |
| GCC | 14 or later on Linux | Older GCC versions lack the required library support |
| Clang | Not tested | It may configure if its standard library provides `std::expected`, but the project does not build or test with it |
| Operating systems | Windows 11 and Linux (continuous integration uses Windows Server 2025 and Ubuntu 24.04 runners) | macOS and others are not tested |
| Processors | x86-64 (AVX2 by default; AVX-512 and SSE2 selectable) | Arm: CMake recognizes `arm64`, but there is no preset and no test run on it |
| Build system | CMake 3.25 or later, Ninja, vcpkg manifest mode | Other generators may work, but only Ninja is exercised |

The test matrix is: GCC 14 on Linux with Debug, Release and an AddressSanitizer plus UndefinedBehaviorSanitizer build, and MSVC on Windows. {doc}`build-options` lists requirements and options in detail.

## Stability of the API

cxxgpr follows the version scheme `MAJOR.MINOR.PATCH`. The current release series is `0.1`.

:::{important}
Versions before 1.0 are pre-1.0 releases. The project explicitly keeps the right to revise the public API until version 1.0, including default template arguments and auxiliary diagnostic fields. Pin an exact version and read the change log before upgrading.
:::

The project's policy for what each part of the number may change is:

| Bump | Changes allowed |
|---|---|
| `PATCH` | Bug fixes and internal improvements that do not change reference results (bit-identical on all qualified platforms) |
| `MINOR` | Improvements within the numerical budget and new features; if results change, the change is recorded. Backward-compatible additions to configuration |
| `MAJOR` | Breaking changes to the persistence format, the public API or the numerical budget |

Before 1.0 these statements describe intent, not a guarantee: a `0.x` release may break the API. After 1.0, `MAJOR` is the only part that may break source compatibility.

What is stable by design, even before 1.0:

- The error contract. Functions return `std::expected<T, GprError>` and never throw across the API boundary.
- Column-major input and the `Gpr` / `FittedGpr` split.
- The persistence format version (`persist::FORMAT_VERSION`, currently 1). Loading a directory with another `format_version` returns `UnsupportedPersistVersion`; there is no automatic migration. See {doc}`file-format`.
- Public headers contain no Eigen, BLAS/oneMKL, nlohmann/json or OpenMP types ({doc}`../explanation/design-principles`).

## Versions in CMake

`find_package(cxxgpr 0.1.0 CONFIG REQUIRED)` uses the `SameMinorVersion` rule: it accepts an installed `0.1.x` that is not older than the requested version, and rejects `0.2.0` or `1.0.0`. Because minor versions can still change the API before 1.0, this is the intended behaviour.

`cxxgpr::build_info()` reports the version and the exact git commit of the library you linked.

## ABI

cxxgpr is a C++ static library with a C++ interface. It makes no ABI promise across versions or toolchains:

- **Rebuild with every update.** A new library version, a new compiler version or a changed compiler option can change type layouts and mangled names. Rebuild your program against the matching headers and library.
- **Keep the build configuration consistent.** The library and your program must use the same C++ standard library implementation and, on Windows, the same C runtime and the same debug or release settings (iterator debugging and `_ITERATOR_DEBUG_LEVEL`, the `/MD` or `/MT` choice). Mixing a Release library with a Debug program is likely to fail on MSVC.
- **The heavy templates are instantiated inside the library for the four precision policies.** `FitBuffers`, `WorkspaceCore`, `QueryWorkspace`, `LeafCache` and the workspace functions such as `predict_into` are declared `extern template` for `DoublePrecision`, `SinglePrecision`, `MixedPrecision<PromoteStorage>` and `MixedPrecision<ReevaluateKernel>`, and `PrecisionPolicyConcept` rejects every other policy at compile time.
- **OpenMP runtime.** If the library was built with OpenMP, your final link also needs the OpenMP runtime of the same compiler.
- **There is no shared library.** Only a static library is built, so there is no binary interface to keep stable across processes.

## Numerical compatibility

Results are reproducible under the rules in {doc}`../explanation/determinism-and-performance`: bit for bit across thread counts within one build (when external BLAS thread counts are fixed); equal to rounding error across compilers, operating systems and `CXXGPR_X86_ISA` tiers. The random stream of `Xoshiro256PlusPlus` is the same integer sequence everywhere.

A saved model is intended to be portable across platforms: the on-disk format is the safetensors layout with little-endian headers and column-major matrices ({doc}`file-format`), and a model loaded on another platform is expected to predict to within rounding error of the original.

## Scope

cxxgpr implements exact Gaussian process regression only. Sparse approximations and online (incremental) models are not part of the library, and the loader rejects files that describe them.

## License and third-party code

cxxgpr is licensed under Apache-2.0. Its dependencies keep their own licenses (Eigen under MPL-2.0, restricted to its MPL-2 parts by `EIGEN_MPL2_ONLY`; OpenBLAS under BSD-3-Clause; Intel oneMKL under ISSL when selected; nlohmann/json under MIT; OpenMP belongs to your compiler). The texts and notices are installed under `share/cxxgpr` and listed in `THIRD_PARTY_NOTICES.md`.
