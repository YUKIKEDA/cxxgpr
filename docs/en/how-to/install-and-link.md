# How to install cxxgpr and link it into your project

This guide shows how to build cxxgpr, install it, and use it from a CMake project, either from an installed package or directly from source with `FetchContent`. It ends with a runtime check of which build you are actually running.

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| C++ compiler | C++23 with `std::expected` | MSVC 19.44 or newer (Visual Studio 17.14), or GCC 14 or newer |
| CMake | 3.25 or newer | |
| Ninja | any recent | The presets use Ninja (Ninja Multi-Config on Windows) |
| vcpkg | with `VCPKG_ROOT` set | Provides the pinned dependencies |
| Eigen | 5.0 | Used internally, not exposed in public headers |
| OpenBLAS / Intel oneMKL | `0.3.34` (OpenBLAS) or oneMKL | Optional vendor BLAS backend (`CXXGPR_BLAS_BACKEND`; `BuiltIn` SIMD is default on Windows MSVC, OpenBLAS on Linux/MinGW); not exposed in public headers |
| nlohmann/json | 3.12 | Used internally, not exposed in public headers |
| OpenMP | optional | Used for deterministic parallel loops when found |

cxxgpr is a **static library**. Eigen, OpenBLAS/MKL, `herumi/fmath` (fetched automatically during configure), nlohmann/json and OpenMP are implementation details: no public header includes them, so your own code never needs their headers. The installed CMake package still locates the configured libraries with `find_dependency`, so they must be discoverable when you configure your project.

## Build and install from source

Clone the repository and use the presets. The presets read the dependency versions from the vcpkg manifest, so vcpkg downloads and builds exactly the versions the library was verified with.

::::{tab-set}

:::{tab-item} Linux (GCC 14)
```{code-block} sh
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
cmake --install out/build/linux-release --prefix /path/to/prefix
```
:::

:::{tab-item} Windows (MSVC)
Run these in a Developer PowerShell, or any shell where `cl` is on `PATH`.

```{code-block} powershell
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
cmake --install out/build/windows-msvc --config Release --prefix C:\path\to\prefix
```
:::

::::

The install step copies the library, all headers under `include/cxxgpr/`, a CMake package (`cxxgprConfig.cmake`, its version file and the exported targets), the license and the third-party notices. The package version file uses `SameMinorVersion` compatibility: while the major version is 0, `find_package(cxxgpr 0.1.0)` accepts any `0.1.x` release.

The available CMake options are listed in {doc}`../reference/build-options`.

## Use the installed package

In your project's `CMakeLists.txt`:

```{code-block} cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES CXX)

find_package(cxxgpr 0.1.0 CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_23)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

Linking `cxxgpr::cxxgpr` adds the include directory, the static library and the (private) transitive dependencies. Then configure with the prefix you installed to **and** with the location of Eigen, OpenBLAS (if built with OpenBLAS), and nlohmann/json. The simplest way is to use the vcpkg toolchain file, which finds the dependencies for you:

```{code-block} sh
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_PREFIX_PATH=/path/to/prefix
```

If you do not use vcpkg, add the install prefixes of Eigen, OpenBLAS, and nlohmann/json to `CMAKE_PREFIX_PATH` as well, separated by semicolons.

:::{note}
Add a `vcpkg.json` to your own project with `eigen3`, `openblas` (`threads` feature, if using the OpenBLAS backend), and `nlohmann-json` as dependencies if you want vcpkg manifest mode to fetch them automatically. cxxgpr itself is installed as above; it is not in the vcpkg registry.
:::

## Use it with `FetchContent`

If you prefer to build cxxgpr as part of your project, fetch it and link the same target:

```{code-block} cmake
include(FetchContent)
FetchContent_Declare(
    cxxgpr
    GIT_REPOSITORY https://github.com/YUKIKEDA/cxxgpr.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(cxxgpr)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

When cxxgpr is a subproject, its tests, canary, benchmarks, examples and install rules are off by default, so only the library is built. cxxgpr's own `CMakeLists.txt` runs `find_package(Eigen3 5.0 CONFIG REQUIRED)` and `find_package(nlohmann_json 3.12 CONFIG REQUIRED)`, so the dependency location is configured the same way as for the installed package (the vcpkg toolchain file, or `CMAKE_PREFIX_PATH`).

## Include the headers

Include only what you use. Every header lives under `cxxgpr/<module>/` and the namespace is `cxxgpr`, with sub-namespaces `cxxgpr::kernel`, `cxxgpr::transform`, `cxxgpr::optimizer` and `cxxgpr::persist`:

```{code-block} cpp
#include <cxxgpr/gpr/gpr.hpp>             // Gpr, FittedGpr, Prediction results
#include <cxxgpr/kernel/kernel.hpp>       // all kernels, KernelSpec
#include <cxxgpr/transform/transform.hpp> // input / target transforms
#include <cxxgpr/persist/persist.hpp>     // save, LoadedGpr
#include <cxxgpr/core/build_info.hpp>     // build_info(), build_info_json()
```

The complete list of headers and their contents is in the generated {doc}`API reference <../reference/api/index>`.

## Check which build you are running

When something behaves unexpectedly, or when you file a bug report, find out exactly which library build your program links. `build_info()` returns a `BuildInfo` describing the build, and `build_info_json()` returns the same data as a JSON string.

```{literalinclude} /../examples/build_info.cpp
:language: cpp
:start-after: // [struct]
:end-before: // [end]
:dedent: 4
```

On one Windows machine this prints:

```{code-block} text
schema:        cxxgpr.build-info/1
version:       0.1.0
git commit:    21f7f5a1be7b1d1b0aab71d49b2d73e4e3711507 (dirty)
compiler:      MSVC 19.51.36260.0
platform:      Windows AMD64
platform key:  windows-x64-msvc19.51
configuration: Release
fp mode:       precise (/fp:precise, AVX2)
dependency:    eigen 5.0.1
dependency:    nlohmann_json 3.12.0
```

Your values will differ. The fields of `BuildInfo` are described in {doc}`prepare-data`, together with the other utilities of the core module. A `(dirty)` suffix means `uncommitted_changes` was true: the library was built from a working tree with local modifications, which matters when you compare numerical results between builds.

:::{warning}
`build_info()` describes the **library** that was compiled and linked, not your program. The git commit, compiler and configuration are those recorded when cxxgpr was built, which for an installed package is when it was installed.
:::

## Troubleshooting

`Could NOT find Eigen3 (missing: Eigen3_DIR)`
: The dependency prefixes are not on `CMAKE_PREFIX_PATH`. Use the vcpkg toolchain file or add the prefix explicitly.

`error: 'expected' is not a member of 'std'`
: Your compiler or language standard is too old. Use GCC 14 or MSVC 19.44 and set `CMAKE_CXX_STANDARD 23`, or `target_compile_features(... cxx_std_23)`.

`find_package(cxxgpr)` fails with a version error
: The package version file accepts only releases of the same minor version. Ask for the version you installed, for example `0.1.0`.

## Where next

- Run the first program: {doc}`../tutorials/first-regression`.
- Load your own data into the library: {doc}`prepare-data`.
