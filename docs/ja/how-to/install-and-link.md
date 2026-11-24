# cxxgpr をインストールしてプロジェクトにリンクする

cxxgpr をビルドしてインストールし、CMake プロジェクトから使う手順を示す。インストール済みパッケージを使う方法と、`FetchContent` でソースから直接取り込む方法の両方を扱う。最後に、実行中のプログラムが実際にどのビルドを使っているかを確かめる方法を示す。

## 必要なもの

| 必要なもの | バージョン | 備考 |
|---|---|---|
| C++ コンパイラ | `std::expected` を備えた C++23 | MSVC 19.44 以降（Visual Studio 17.14）、または GCC 14 以降 |
| CMake | 3.25 以降 | |
| Ninja | 最近のもの | プリセットは Ninja を使う（Windows は Ninja Multi-Config） |
| vcpkg | `VCPKG_ROOT` を設定済み | 版を固定した依存ライブラリを提供する |
| Eigen | 5.0 | 内部でのみ使い、公開ヘッダには現れない |
| OpenBLAS / Intel oneMKL | `0.3.34`（OpenBLAS）または oneMKL | `CXXGPR_BLAS_BACKEND` に応じた BLAS バックエンド。公開ヘッダには現れない |
| nlohmann/json | 3.12 | 内部でのみ使い、公開ヘッダには現れない |
| OpenMP | 任意 | 見つかれば、決定的な並列ループに使う |

cxxgpr は**静的ライブラリ**である。Eigen、OpenBLAS/MKL、`herumi/fmath`（構成時に自動取得）、nlohmann/json、OpenMP は実装の詳細で、公開ヘッダのどれもこれらをインクルードしない。そのため自分のコードがそれらのヘッダを必要とすることはない。ただし、インストールされた CMake パッケージは構成時に選んだ依存ライブラリを `find_dependency` で探す。プロジェクトを構成するときには、見つけられる状態にしておく必要がある。

## ソースからビルドしてインストールする

リポジトリをクローンし、プリセットを使う。プリセットは vcpkg のマニフェストから依存ライブラリの版を読むので、ライブラリを検証したときと同じ版が取得・ビルドされる。

::::{tab-set}

:::{tab-item} Linux（GCC 14）
```{code-block} sh
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
cmake --install out/build/linux-release --prefix /path/to/prefix
```
:::

:::{tab-item} Windows（MSVC）
Developer PowerShell など、`cl` が `PATH` にあるシェルで実行する。

```{code-block} powershell
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
cmake --install out/build/windows-msvc --config Release --prefix C:\path\to\prefix
```
:::

::::

インストールすると、ライブラリ、`include/cxxgpr/` 以下のすべてのヘッダ、CMake パッケージ（`cxxgprConfig.cmake`、バージョンファイル、エクスポートしたターゲット）、ライセンス、第三者通知がコピーされる。パッケージのバージョンファイルは `SameMinorVersion` 互換を使う。メジャーバージョンが 0 の間は、`find_package(cxxgpr 0.1.0)` が `0.1.x` のどのリリースも受け付ける。

CMake のオプションは {doc}`../reference/build-options` に一覧がある。

## インストール済みパッケージを使う

自分のプロジェクトの `CMakeLists.txt` に、次のように書く。

```{code-block} cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES CXX)

find_package(cxxgpr 0.1.0 CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_23)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

`cxxgpr::cxxgpr` にリンクすると、インクルードディレクトリ、静的ライブラリ、（非公開の）推移的な依存が加わる。構成のときには、インストール先のプレフィックスと、Eigen・OpenBLAS（OpenBLAS バックエンド利用時）・nlohmann/json の場所を指定する。vcpkg のツールチェーンファイルを使うと、依存ライブラリは自動で見つかるので、いちばん簡単である。

```{code-block} sh
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_PREFIX_PATH=/path/to/prefix
```

vcpkg を使わない場合は、Eigen、OpenBLAS、nlohmann/json のインストール先も `CMAKE_PREFIX_PATH` にセミコロン区切りで加える。

:::{note}
vcpkg のマニフェストモードで自動取得したいなら、自分のプロジェクトの `vcpkg.json` に `eigen3`、`openblas`（`threads` 機能、OpenBLAS 利用時）、`nlohmann-json` を依存として書く。cxxgpr 自体は上の手順でインストールする。vcpkg のレジストリには登録されていない。
:::

## `FetchContent` で使う

cxxgpr をプロジェクトの一部としてビルドしたいなら、取得して同じターゲットにリンクする。

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

cxxgpr がサブプロジェクトのとき、テスト、カナリア、ベンチマーク、サンプル、インストール規則は既定で無効になり、ライブラリだけがビルドされる。cxxgpr の `CMakeLists.txt` は `find_package(Eigen3 5.0 CONFIG REQUIRED)` と `find_package(nlohmann_json 3.12 CONFIG REQUIRED)` を実行する。依存ライブラリの場所は、インストール済みパッケージの場合と同じ方法（vcpkg のツールチェーンファイル、または `CMAKE_PREFIX_PATH`）で指定する。

## ヘッダをインクルードする

使うものだけをインクルードする。ヘッダはすべて `cxxgpr/<module>/` の下にあり、名前空間は `cxxgpr` である。サブ名前空間として `cxxgpr::kernel`、`cxxgpr::transform`、`cxxgpr::optimizer`、`cxxgpr::persist` がある。

```{code-block} cpp
#include <cxxgpr/gpr/gpr.hpp>             // Gpr, FittedGpr, Prediction results
#include <cxxgpr/kernel/kernel.hpp>       // all kernels, KernelSpec
#include <cxxgpr/transform/transform.hpp> // input / target transforms
#include <cxxgpr/persist/persist.hpp>     // save, LoadedGpr
#include <cxxgpr/core/build_info.hpp>     // build_info(), build_info_json()
```

ヘッダの全一覧とその内容は、生成された {doc}`API リファレンス <../reference/api/index>` にある。

## 実行中のビルドを確認する

動作が想定と違うときや、バグ報告を書くときは、プログラムが実際にどのビルドのライブラリをリンクしているかを調べる。`build_info()` はビルドを記述する `BuildInfo` を返す。`build_info_json()` は同じ内容を JSON 文字列で返す。

```{literalinclude} /../examples/build_info.cpp
:language: cpp
:start-after: // [struct]
:end-before: // [end]
:dedent: 4
```

ある Windows 環境では、次のように出力された。

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

値は環境によって異なる。`BuildInfo` の各フィールドは、core モジュールのほかの機能とあわせて {doc}`prepare-data` で説明している。`(dirty)` は `uncommitted_changes` が真だったことを表す。ローカルに変更のある作業ツリーからビルドされたという意味で、ビルド間で数値結果を比べるときに重要になる。

:::{warning}
`build_info()` が記述するのは、コンパイルされてリンクされた**ライブラリ**であり、自分のプログラムではない。Git のコミット、コンパイラ、構成は、cxxgpr をビルドしたときに記録されたものである。インストール済みパッケージなら、それはインストールしたときにあたる。
:::

## トラブルシューティング

`Could NOT find Eigen3 (missing: Eigen3_DIR)`
: 依存ライブラリのプレフィックスが `CMAKE_PREFIX_PATH` にない。vcpkg のツールチェーンファイルを使うか、プレフィックスを明示的に加える。

`error: 'expected' is not a member of 'std'`
: コンパイラか言語標準が古い。GCC 14 または MSVC 19.44 を使い、`CMAKE_CXX_STANDARD 23` または `target_compile_features(... cxx_std_23)` を設定する。

`find_package(cxxgpr)` がバージョンのエラーで失敗する
: パッケージのバージョンファイルは、同じマイナーバージョンのリリースしか受け付けない。インストールしたバージョン（たとえば `0.1.0`）を指定する。

## 次に読むもの

- 最初のプログラムを動かす：{doc}`../tutorials/first-regression`
- 自分のデータをライブラリに渡す：{doc}`prepare-data`
