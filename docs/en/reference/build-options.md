# Build options

Everything you can configure when you build cxxgpr from source: requirements, CMake options and presets, dependencies with their pinned versions, the compiler flags the library applies, OpenMP and the SIMD instruction set. For installing and linking a finished library see {doc}`../how-to/install-and-link`.

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| C++ compiler | C++23 with `std::expected` (including `transform`) | MSVC 19.44 or later (Visual Studio 17.14 or later) and GCC 14 or later are the supported toolchains |
| CMake | 3.25 or later | |
| Build tool | Ninja | The presets use `Ninja` (Linux) and `Ninja Multi-Config` (Windows) |
| vcpkg | A checkout pointed to by the environment variable `VCPKG_ROOT` | Manifest mode; the presets set the vcpkg toolchain file |

CMake checks the compiler at configure time. It stops with a fatal error if the MSVC version is below 19.44, and compiles a small test that must find `__cpp_lib_expected >= 202211L` and a working `std::expected::transform`; without them configuring fails with "C++23 std::expected with monadic transform is required". Other compilers (Clang, for example) pass the configuration only if they pass this test; the project does not test them.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `CXXGPR_BUILD_TESTS` | `ON` when cxxgpr is the top-level project, otherwise `OFF` | Builds the GoogleTest suites (also needs `BUILD_TESTING` and the vcpkg feature `tests`) |
| `CXXGPR_BUILD_CANARY` | `ON` when top-level, otherwise `OFF` | Builds the numeric canary executable that checks the floating-point environment |
| `CXXGPR_BUILD_BENCHMARKS` | `OFF` | Builds the Google Benchmark programs (needs the vcpkg feature `benchmarks`) |
| `CXXGPR_BUILD_EXAMPLES` | `OFF` | Builds the example programs under `docs/examples`, each registered as a CTest test |
| `CXXGPR_INSTALL` | `ON` when top-level, otherwise `OFF` | Generates the `install` rules and the CMake package files |
| `CXXGPR_BLAS_BACKEND` | `AUTO` | Dense BLAS/LAPACK backend: `AUTO` (`BuiltIn` on Windows MSVC; OpenBLAS, then MKL, then `BuiltIn` elsewhere), `OpenBLAS`, `MKL`, or `BuiltIn`. Any other value is a configure error |
| `CXXGPR_X86_ISA` | `AVX2` | x86-64 only. SIMD tier of the library: `AVX512`, `AVX2` or `SSE2`. Any other value is a configure error |

When cxxgpr is included with `FetchContent` or `add_subdirectory`, it is not the top-level project, so tests, canary and install rules are off unless you switch them on.

Standard variables that matter:

| Variable | Use |
|---|---|
| `CMAKE_BUILD_TYPE` or the multi-config `--config` | `Release` for production use; the library records the configuration in `build_info()` |
| `CMAKE_TOOLCHAIN_FILE` | vcpkg's `scripts/buildsystems/vcpkg.cmake` |
| `VCPKG_MANIFEST_FEATURES` | `tests` and/or `benchmarks` to install their dependencies |
| `VCPKG_INSTALLED_DIR` | Where vcpkg installs packages; the benchmark presets keep a separate directory |
| `CMAKE_PREFIX_PATH` | In a consuming project: the install prefix of cxxgpr and of its dependencies |

## Presets

`CMakePresets.json` defines these configure presets. Each has a matching build preset, and the ones that run tests have a test preset.

| Configure preset | Platform | Generator and compiler | Notes |
|---|---|---|---|
| `windows-msvc` | Windows | Ninja Multi-Config, `cl`; configurations `Debug` and `Release` | vcpkg feature `tests`; build presets `windows-debug`, `windows-release` |
| `windows-benchmark` | Windows | inherits `windows-msvc` | `CXXGPR_BUILD_BENCHMARKS=ON`, tests off, build preset `windows-benchmark` |
| `linux-debug` | Linux | Ninja, `g++-14`, `Debug` | |
| `linux-release` | Linux | Ninja, `g++-14`, `Release` | |
| `linux-sanitize` | Linux | Ninja, `g++-14`, `RelWithDebInfo` with `-fsanitize=address,undefined` | |
| `linux-benchmark` | Linux | Ninja, `g++-14`, `Release` | `CXXGPR_BUILD_BENCHMARKS=ON`, tests off |

Typical use:

```{code-block} sh
# Linux (GCC 14), from the repository root
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
```

```{code-block} powershell
# Windows, from a Developer PowerShell with VCPKG_ROOT set
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
```

The build directories are `out/build/<preset>`. Set `VCPKG_ROOT` first: the presets read it through `$env{VCPKG_ROOT}`.

## Dependencies

All dependencies are implementation details. None appears in a public header, and none is needed to compile your own code. They are `PRIVATE` link dependencies of the static library.

| Dependency | Required version | Source / pin | Used for |
|---|---|---|---|
| Eigen (`eigen3`) | 5.0 (`find_package(Eigen3 5.0 CONFIG REQUIRED)`) | vcpkg `5.0.1` | Dense linear algebra panel fallbacks & transforms. Compiled with `EIGEN_MPL2_ONLY` and `EIGEN_DONT_PARALLELIZE` |
| OpenBLAS (`openblas`, `threads` feature on non-MSVC) or Intel oneMKL | `0.3.34` (OpenBLAS) or oneMKL | vcpkg `0.3.34` (OpenBLAS on `!windows \| mingw`) | Dense BLAS (`GEMM`, `SYRK`, `TRSM`, `GEMV`, `DOT`) and MKL `potrf`/`potri` when selected by `CXXGPR_BLAS_BACKEND` |
| nlohmann/json (`nlohmann-json`) | 3.12 (`find_package(nlohmann_json 3.12 CONFIG REQUIRED)`) | vcpkg `3.12.0` | `config.json` of persisted models |
| OpenMP (`OpenMP::OpenMP_CXX`) | any that the compiler provides; optional | n/a | Parallel loops. MSVC reports OpenMP 2.0 |
| GoogleTest (`gtest`) | 1.18.0 | vcpkg `1.18.0` | Tests only, feature `tests` |
| Google Benchmark (`benchmark`) | 1.9.5 | vcpkg `1.9.5` | Benchmarks only, feature `benchmarks` |

`vcpkg.json` pins the vcpkg registry dependencies with a `builtin-baseline` commit plus explicit `overrides`. To build explicitly with the built-in SIMD backend (the default on Windows MSVC), pass `-DCXXGPR_BLAS_BACKEND=BuiltIn`, or place your own vendor BLAS packages on `CMAKE_PREFIX_PATH`; the version checks above still apply.

The versions of the dependencies a given library was built with are recorded in `cxxgpr::build_info()` (see below).

## Compiler options applied to the library

`cxxgpr_configure_target` gives the library and the canary executable these options. They are private: your target does not inherit them, except the C++23 requirement.

| Compiler | Options |
|---|---|
| MSVC | `/std:c++23preview /W4 /WX /permissive- /utf-8 /EHsc /fp:precise /bigobj` |
| GCC and Clang | `-std=c++23 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion -fno-fast-math` |

The target requests `cxx_std_23` publicly, so a project that links `cxxgpr::cxxgpr` is compiled as C++23 automatically.

Warnings are errors in the project's own build. If you build cxxgpr with a compiler version that issues new warnings, the build may fail; this is a build-time problem only.

Floating-point semantics are strict: `/fp:precise` and `-fno-fast-math` keep `NaN` and infinity checks valid and keep rounding reproducible. Do not add `-ffast-math` or `/fp:fast` to the library build.

## SIMD instruction set (`CXXGPR_X86_ISA`)

On x86-64 the option selects the instruction set that the compiler may use for the whole library, including Eigen and cxxgpr's own SIMD kernels.

| Value | MSVC | GCC and Clang |
|---|---|---|
| `AVX512` | `/arch:AVX512` | `-mavx512f -mavx512dq -mavx512vl -mfma` |
| `AVX2` (default) | `/arch:AVX2` | `-mavx2 -mfma` |
| `SSE2` | (no extra option; the x86-64 baseline) | (no extra option) |

```{code-block} sh
cmake --preset linux-release -DCXXGPR_X86_ISA=AVX512
```

The choice is made when the library is compiled; there is no run-time dispatch. A library built for AVX2 or AVX-512 needs a CPU that supports it. Results of different tiers agree to rounding error but not bit for bit, and the tier is part of the reproducibility context ({doc}`../explanation/determinism-and-performance`). The option is ignored on other architectures. The chosen tier is visible in `BuildInfo::floating_point`, for example `precise (/fp:precise, AVX2)`.

## OpenMP

`find_package(OpenMP COMPONENTS CXX)` runs at configure time. If OpenMP is found, the library links it and its parallel regions are active; if not, the library builds serially and `linalg::openmp_max_threads()` returns 1. The installed package records the decision: when the library was built with OpenMP, `find_package(cxxgpr)` also runs `find_dependency(OpenMP COMPONENTS CXX REQUIRED)`, so your project must be able to find OpenMP too. The thread count is controlled at run time through the usual OpenMP mechanisms; see {doc}`../how-to/optimize-performance`.

## What `install` produces

With `CXXGPR_INSTALL=ON`, `cmake --install <build-dir> --prefix <prefix>` installs:

| Path under the prefix | Contents |
|---|---|
| `include/cxxgpr/` | All public headers (`*.hpp`) |
| `lib/cxxgpr.lib` or `lib/libcxxgpr.a` | The static library |
| `lib/cmake/cxxgpr/` | `cxxgprConfig.cmake`, `cxxgprConfigVersion.cmake`, `cxxgprTargets.cmake` |
| `share/cxxgpr/` | `LICENSE` and `THIRD_PARTY_NOTICES.md` |

The package exports the target `cxxgpr::cxxgpr`. Its version file uses the `SameMinorVersion` compatibility mode: `find_package(cxxgpr 0.1.0 CONFIG REQUIRED)` accepts an installed `0.1.x`, and rejects `0.2.0` and `1.0.0`. {doc}`compatibility` explains the version policy.

## Check what you built

`build_info()` returns a `BuildInfo` describing the compiled library, and `build_info_json()` returns the same data as formatted JSON text. A `BuildInfo` has the fields `schema`, `version`, `git_commit`, `uncommitted_changes`, `compiler`, `platform`, `platform_key`, `configuration`, `floating_point` and `dependencies`, a list of `DependencyVersion` (`name`, `version`).

```{code-block} cpp
#include <cxxgpr/core/build_info.hpp>
#include <iostream>

int main() {
    std::cout << cxxgpr::build_info_json();
}
```

Use it to confirm the version, the ISA tier and the dependency versions of the library that actually got linked.
