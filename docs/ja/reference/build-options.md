# ビルドオプション

cxxgpr をソースからビルドするときに設定できる項目の一覧である。必要な環境、CMake のオプションとプリセット、固定された版つきの依存ライブラリ、ライブラリに適用されるコンパイラオプション、OpenMP、SIMD 命令セットを取り上げる。完成したライブラリのインストールとリンクは {doc}`../how-to/install-and-link` を参照。

## 必要な環境

| 要件 | 版 | 備考 |
|---|---|---|
| C++ コンパイラ | C++23。`std::expected`（`transform` を含む）が使えること | 対応ツールチェーンは、MSVC 19.44 以降（Visual Studio 17.14 以降）と GCC 14 以降 |
| CMake | 3.25 以降 | |
| ビルドツール | Ninja | プリセットは `Ninja`（Linux）と `Ninja Multi-Config`（Windows）を使う |
| vcpkg | 環境変数 `VCPKG_ROOT` が指すチェックアウト | マニフェストモード。プリセットが vcpkg のツールチェーンファイルを設定する |

コンパイラは、CMake が構成時に検査する。MSVC の版が 19.44 未満なら、致命的エラーで止まる。小さなテストをコンパイルし、`__cpp_lib_expected >= 202211L` と、動作する `std::expected::transform` を確認する。満たさなければ、構成は「C++23 std::expected with monadic transform is required」で失敗する。ほかのコンパイラ（たとえば Clang）も、このテストに通れば構成は通るが、プロジェクトはそれらをテストしていない。

## CMake オプション

| オプション | 既定値 | 効果 |
|---|---|---|
| `CXXGPR_BUILD_TESTS` | cxxgpr がトップレベルのプロジェクトなら `ON`、それ以外は `OFF` | GoogleTest のテストをビルドする（`BUILD_TESTING` と vcpkg のフィーチャー `tests` も必要） |
| `CXXGPR_BUILD_CANARY` | トップレベルなら `ON`、それ以外は `OFF` | 浮動小数点環境を調べる数値カナリア実行ファイルをビルドする |
| `CXXGPR_BUILD_BENCHMARKS` | `OFF` | Google Benchmark のプログラムをビルドする（vcpkg のフィーチャー `benchmarks` が必要） |
| `CXXGPR_BUILD_EXAMPLES` | `OFF` | `docs/examples` のサンプルプログラムをビルドし、それぞれを CTest のテストとして登録する |
| `CXXGPR_INSTALL` | トップレベルなら `ON`、それ以外は `OFF` | `install` の規則と CMake パッケージファイルを生成する |
| `CXXGPR_BLAS_BACKEND` | `AUTO` | 密行列 BLAS/LAPACK バックエンド。`AUTO`（Windows MSVC では `BuiltIn`、それ以外では OpenBLAS → MKL → `BuiltIn` の順）、`OpenBLAS`、`MKL`、`BuiltIn` のいずれか。ほかの値は構成エラー |
| `CXXGPR_X86_ISA` | `AVX2` | x86-64 のみ。ライブラリの SIMD 段階。`AVX512`、`AVX2`、`SSE2` のいずれか。ほかの値は構成エラー |

`FetchContent` や `add_subdirectory` で取り込むと、cxxgpr はトップレベルのプロジェクトではない。そのため、テスト、カナリア、インストール規則は、明示的に有効にしない限りオフである。

関係のある標準の変数は次のとおり。

| 変数 | 用途 |
|---|---|
| `CMAKE_BUILD_TYPE`、またはマルチ構成の `--config` | 実運用では `Release`。ライブラリは、構成を `build_info()` に記録する |
| `CMAKE_TOOLCHAIN_FILE` | vcpkg の `scripts/buildsystems/vcpkg.cmake` |
| `VCPKG_MANIFEST_FEATURES` | `tests` や `benchmarks` を指定すると、その依存ライブラリがインストールされる |
| `VCPKG_INSTALLED_DIR` | vcpkg のパッケージのインストール先。ベンチマーク用のプリセットは、別のディレクトリを使う |
| `CMAKE_PREFIX_PATH` | 利用側のプロジェクトで、cxxgpr とその依存ライブラリのインストール先を指す |

## プリセット

`CMakePresets.json` は、次の構成プリセットを定義している。それぞれに対応するビルドプリセットがあり、テストを走らせるものにはテストプリセットもある。

| 構成プリセット | プラットフォーム | ジェネレーターとコンパイラ | 備考 |
|---|---|---|---|
| `windows-msvc` | Windows | Ninja Multi-Config、`cl`。構成は `Debug` と `Release` | vcpkg のフィーチャー `tests`。ビルドプリセットは `windows-debug` と `windows-release` |
| `windows-benchmark` | Windows | `windows-msvc` を継承 | `CXXGPR_BUILD_BENCHMARKS=ON`、テストなし。ビルドプリセットは `windows-benchmark` |
| `linux-debug` | Linux | Ninja、`g++-14`、`Debug` | |
| `linux-release` | Linux | Ninja、`g++-14`、`Release` | |
| `linux-sanitize` | Linux | Ninja、`g++-14`、`RelWithDebInfo`。`-fsanitize=address,undefined` つき | |
| `linux-benchmark` | Linux | Ninja、`g++-14`、`Release` | `CXXGPR_BUILD_BENCHMARKS=ON`、テストなし |

典型的な使い方は次のとおり。

```{code-block} sh
# Linux (GCC 14)、リポジトリのルートで
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
```

```{code-block} powershell
# Windows、VCPKG_ROOT を設定した Developer PowerShell で
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
```

ビルドディレクトリは `out/build/<プリセット名>` である。先に `VCPKG_ROOT` を設定しておくこと。プリセットは `$env{VCPKG_ROOT}` でこの変数を読む。

## 依存ライブラリ

依存ライブラリはすべて実装の詳細である。公開ヘッダーには現れず、利用側のコードのコンパイルにも必要ない。静的ライブラリへの `PRIVATE` なリンク依存である。

| 依存ライブラリ | 要求される版 | 取得元 / 固定版 | 用途 |
|---|---|---|---|
| Eigen（`eigen3`） | 5.0（`find_package(Eigen3 5.0 CONFIG REQUIRED)`） | vcpkg `5.0.1` | 密行列線形代数のパネル計算・変換。`EIGEN_MPL2_ONLY` と `EIGEN_DONT_PARALLELIZE` つきでコンパイルされる |
| OpenBLAS（非 MSVC で `openblas`）または Intel oneMKL | `0.3.34`（OpenBLAS）または oneMKL | vcpkg `0.3.34`（`!windows \| mingw` で OpenBLAS） | `CXXGPR_BLAS_BACKEND` に応じた密行列 BLAS（`GEMM` / `SYRK` / `TRSM` / `GEMV` / `DOT`、MKL では `potrf` / `potri`） |
| nlohmann/json（`nlohmann-json`） | 3.12（`find_package(nlohmann_json 3.12 CONFIG REQUIRED)`） | vcpkg `3.12.0` | 永続化されたモデルの `config.json` |
| OpenMP（`OpenMP::OpenMP_CXX`） | コンパイラが提供するもの。任意 | 対象外 | 並列ループ。MSVC は OpenMP 2.0 を報告する |
| GoogleTest（`gtest`） | 1.18.0 | vcpkg `1.18.0` | テストのみ。フィーチャー `tests` |
| Google Benchmark（`benchmark`） | 1.9.5 | vcpkg `1.9.5` | ベンチマークのみ。フィーチャー `benchmarks` |

`vcpkg.json` は vcpkg レジストリの `builtin-baseline` コミットと明示的な `overrides` でパッケージを固定している。内蔵 SIMD バックエンド（Windows MSVC の既定）を明示指定する場合は `-DCXXGPR_BLAS_BACKEND=BuiltIn` を渡すか、`CMAKE_PREFIX_PATH` でベンダー BLAS の場所を指定する。その場合も、上の版の検査は適用される。

ライブラリがビルドされたときの依存ライブラリの版は、`cxxgpr::build_info()` に記録される（後述）。

## ライブラリに適用されるコンパイラオプション

`cxxgpr_configure_target` が、ライブラリとカナリア実行ファイルに次のオプションを与える。これらは private である。C++23 の要求を除いて、利用側のターゲットには継承されない。

| コンパイラ | オプション |
|---|---|
| MSVC | `/std:c++23preview /W4 /WX /permissive- /utf-8 /EHsc /fp:precise /bigobj` |
| GCC と Clang | `-std=c++23 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion -fno-fast-math` |

ターゲットは `cxx_std_23` を public に要求する。したがって、`cxxgpr::cxxgpr` にリンクするプロジェクトは、自動的に C++23 でコンパイルされる。

プロジェクト自身のビルドでは、警告はエラーである。新しい警告を出すコンパイラの版で cxxgpr をビルドすると、失敗することがある。これはビルド時だけの問題である。

浮動小数点の意味は厳密である。`/fp:precise` と `-fno-fast-math` により、`NaN` と無限大の検査は有効なままで、丸めも再現できる。ライブラリのビルドに `-ffast-math` や `/fp:fast` を加えてはならない。

## SIMD 命令セット（`CXXGPR_X86_ISA`）

x86-64 では、このオプションが、Eigen と cxxgpr 独自の SIMD カーネルを含むライブラリ全体で、コンパイラが使ってよい命令セットを選ぶ。

| 値 | MSVC | GCC と Clang |
|---|---|---|
| `AVX512` | `/arch:AVX512` | `-mavx512f -mavx512dq -mavx512vl -mfma` |
| `AVX2`（既定） | `/arch:AVX2` | `-mavx2 -mfma` |
| `SSE2` | （追加のオプションなし。x86-64 の基本） | （追加のオプションなし） |

```{code-block} sh
cmake --preset linux-release -DCXXGPR_X86_ISA=AVX512
```

選択はライブラリのコンパイル時に行われ、実行時のディスパッチはない。AVX2 や AVX-512 向けにビルドしたライブラリは、それをサポートする CPU を必要とする。段階が違えば、結果は丸め誤差の範囲で一致するが、ビット単位では一致しない。段階は、再現性の条件の一部である（{doc}`../explanation/determinism-and-performance`）。ほかのアーキテクチャでは、このオプションは無視される。選ばれた段階は `BuildInfo::floating_point` で確認できる（たとえば `precise (/fp:precise, AVX2)`）。

## OpenMP

構成時に `find_package(OpenMP COMPONENTS CXX)` が走る。OpenMP が見つかればライブラリはそれにリンクし、並列領域が有効になる。見つからなければ直列でビルドされ、`linalg::openmp_max_threads()` は 1 を返す。インストールされるパッケージは、この判断を記録する。ライブラリが OpenMP つきでビルドされていると、`find_package(cxxgpr)` は `find_dependency(OpenMP COMPONENTS CXX REQUIRED)` も実行する。したがって、利用側のプロジェクトも OpenMP を見つけられなければならない。スレッド数は、通常の OpenMP の仕組みで実行時に制御する。{doc}`../how-to/optimize-performance` を参照。

## `install` が作るもの

`CXXGPR_INSTALL=ON` のとき、`cmake --install <ビルドディレクトリ> --prefix <プレフィックス>` は次をインストールする。

| プレフィックス以下のパス | 内容 |
|---|---|
| `include/cxxgpr/` | すべての公開ヘッダー（`*.hpp`） |
| `lib/cxxgpr.lib` または `lib/libcxxgpr.a` | 静的ライブラリ |
| `lib/cmake/cxxgpr/` | `cxxgprConfig.cmake`、`cxxgprConfigVersion.cmake`、`cxxgprTargets.cmake` |
| `share/cxxgpr/` | `LICENSE` と `THIRD_PARTY_NOTICES.md` |

パッケージは、ターゲット `cxxgpr::cxxgpr` をエクスポートする。バージョンファイルの互換モードは `SameMinorVersion` である。`find_package(cxxgpr 0.3.0 CONFIG REQUIRED)` は、インストール済みの `0.3.x` を受け入れ、`0.4.0` と `1.0.0` を拒否する。版の方針は {doc}`compatibility` で説明する。

## ビルドした内容を確かめる

`build_info()` は、コンパイルされたライブラリを記述する `BuildInfo` を返す。`build_info_json()` は、同じデータを整形した JSON テキストで返す。`BuildInfo` のフィールドは、`schema`、`version`、`git_commit`、`uncommitted_changes`、`compiler`、`platform`、`platform_key`、`configuration`、`floating_point`、`dependencies` である。`dependencies` は `DependencyVersion`（`name`、`version`）のリストである。

```{code-block} cpp
#include <cxxgpr/core/build_info.hpp>
#include <iostream>

int main() {
    std::cout << cxxgpr::build_info_json();
}
```

実際にリンクされたライブラリの版、ISA の段階、依存ライブラリの版を、これで確認できる。
