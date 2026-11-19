# 0002: Ninja プリセット・Windows MSVC 環境読込・vcpkg manifest

- Status: Accepted
- Date: 2026-11-15
- Issue: #1

## Context

開発対象環境である Windows 11（Visual Studio MSVC）と Linux（GCC 14）でビルド構成・警告方針・コンパイラフラグ・依存導入手順を統一する必要がある。Windows 環境には複数の Visual Studio バージョンが混在し得るため、CMake の Visual Studio バージョン固有ジェネレータへ依存するとプリセットが環境更新で壊れやすい。

## Decision

1. **両 OS で Ninja 系ジェネレータへ統一**：Windows は `Ninja Multi-Config`（`windows-msvc`）、Linux は `Ninja`（`linux-debug` / `linux-release` / `linux-sanitize`）を M0-2 の `CMakePresets.json` に定める。
2. **MSVC 環境スクリプト（`tools/dev/msvc.ps1`）**：Windows では VsDevCmd により最新の C++ ツールチェーンを読み込んで `cl` を使用する。VsDevCmd は環境変数 `VCPKG_ROOT` を VS 同梱版へ上書きすることがあるため、`msvc.ps1` が呼び出し元の `VCPKG_ROOT` を保存・復元する。
3. **コンパイラと浮動小数点・SIMD ポリシー**：
   - `CMAKE_CXX_STANDARD 23`、MSVC は 19.44 以上を要求し `/std:c++23preview` による `std::expected`（`__cpp_lib_expected >= 202211L`）を構成時コンパイル試験で検証する。
   - MSVC は `/W4 /WX /fp:precise`、GCC 14 / Clang は `-Wall -Wextra -Wpedantic -Werror -fno-fast-math` を課す。
   - x86-64 では `CXXGPR_X86_ISA`（既定 `AVX2`、選択値 `AVX512` / `AVX2` / `SSE2`）により、MSVC では `/arch:AVX2`（`AVX512` 時は `/arch:AVX512`）、GCC / Clang では `-mavx2 -mfma`（`AVX512` 時は `-mavx512f -mavx512dq -mavx512vl -mfma`）を有効化し、Eigen および内部 SIMD カーネルの 256-bit / 512-bit 実行を制御する。

## Consequences

- Visual Studio の年次バージョン差に依らず共通プリセットでビルドでき、Windows / Linux ともに同一の vcpkg toolchain・コンパイラ検査経路を通る。
- Windows のコマンドは直接シェルではなく `pwsh -NoProfile -File tools/dev/msvc.ps1 <command>`（または `just` レシピ）経由で実行する必要がある。
