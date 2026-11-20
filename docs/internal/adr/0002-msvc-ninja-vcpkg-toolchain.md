# 0002: Ninja presets, Windows MSVC environment loading, and the vcpkg manifest

- Status: Accepted
- Date: 2026-11-15
- Issue: #1

## Context

The build configuration, warning policy, compiler flags, and dependency installation procedure must be unified across the target development environments: Windows 11 (Visual Studio MSVC) and Linux (GCC 14). Because several Visual Studio versions can coexist on a Windows machine, depending on a Visual Studio version-specific CMake generator makes presets fragile against environment updates.

## Decision

1. **Unify on Ninja-family generators on both OSes**: Windows uses `Ninja Multi-Config` (`windows-msvc`) and Linux uses `Ninja` (`linux-debug` / `linux-release` / `linux-sanitize`), defined in the `CMakePresets.json` of M0-2.
2. **MSVC environment script (`tools/dev/msvc.ps1`)**: on Windows, VsDevCmd loads the latest C++ toolchain and `cl` is used. Because VsDevCmd can overwrite the `VCPKG_ROOT` environment variable with the Visual Studio bundled one, `msvc.ps1` saves and restores the caller's `VCPKG_ROOT`.
3. **Compiler and floating-point / SIMD policy**:
   - `CMAKE_CXX_STANDARD 23`. MSVC 19.44 or later is required, and `std::expected` (`__cpp_lib_expected >= 202211L`) with `/std:c++23preview` is verified by a configure-time compile test.
   - MSVC uses `/W4 /WX /fp:precise`; GCC 14 / Clang use `-Wall -Wextra -Wpedantic -Werror -fno-fast-math`.
   - On x86-64, `CXXGPR_X86_ISA` (default `AVX2`; choices `AVX512` / `AVX2` / `SSE2`) enables `/arch:AVX2` (`/arch:AVX512` for `AVX512`) on MSVC and `-mavx2 -mfma` (`-mavx512f -mavx512dq -mavx512vl -mfma` for `AVX512`) on GCC / Clang, controlling 256-bit / 512-bit execution of Eigen and the internal SIMD kernels.

## Consequences

- The build works with common presets regardless of the annual Visual Studio version, and both Windows and Linux go through the same vcpkg toolchain and compiler-check path.
- On Windows, commands must be run via `pwsh -NoProfile -File tools/dev/msvc.ps1 <command>` (or a `just` recipe) rather than directly in the shell.
