[English](README.md) | 日本語

# cxxgpr

Rust 製ガウス過程回帰ライブラリ [`gprx`](https://github.com/YUKIKEDA/gprx) の **Exact GPR** モデルを C++23 へ移植した静的ライブラリ。

> **現在地（`0.1.0`）**：Exact GPR 本体、永続化、CMake install / パッケージ対応、および検証スイートを実装済み。公開 API は 1.0 未満（`0.x`）の事前安定性規約に従う。

## 対象範囲

`cxxgpr` は **Exact GPR**（`Gpr` / `FittedGpr`）に特化して移植した。Sparse GPR（`Sgpr` / `Svgp`）およびオンライン学習（`OnlineGpr`）はスコープ外とする。

- **推論・モデル操作**：`Gpr::fit`（`const` メンバ、失敗時も不変）、`Gpr::factor`、`predict`、`predict_with`（`VarianceKind::Observation` / `Latent`）、`predict_into`、`predict_with_into`（ウォームアップ後アロケーションゼロ）、`predict_covariance`、`sample`（決定的 `Xoshiro256++`）、`loo_predict`、`neg_log_marginal_likelihood`、`value_and_gradient_into`、`hessian_into`、`get_params` / `set_params`、`refit`、`into_trainer`。
- **カーネル**：組込み 10 種の葉（`RbfKernel`、`RbfArdKernel`、`MaternKernel` $\nu \in \{1/2, 3/2, 5/2\}$、`MaternArdKernel`、`PeriodicKernel`、`RationalQuadraticKernel`、`RationalQuadraticArdKernel`、`ConstantKernel`、`LinearKernel`、`WhiteKernel`）、`+` / `*` 合成（`KernelSpec` -> `CompiledKernel<T>`）、`Custom` 葉（`KernelTerm<T>`）、`LeafCache` 増分更新、供給二乗距離（`ScalarDistance` / `ArdDistance` / `DistanceKernel`）。
- **最適化器**：外部最適化ライブラリに頼らない自前実装（`Lbfgs` / `NelderMead` / `TrustRegion` / `FastSimulatedAnnealing` / `Fixed`）、区間 logit 変換、決定的リスタート。
- **前処理・精度・方針**：入力／目的変数変換（`Identity*` / `Standardize*` / `MinMax*` / `Pipeline` / `TargetPipeline` / `ColumnwiseInput` / ユーザー定義）、`DoublePrecision` / `SinglePrecision` / `MixedPrecision<PromoteStorage | ReevaluateKernel>`、`KernelExp::{Accurate, FastApprox}`、`DistanceCachePolicy`、`CholeskyBuffer`、`JitterPolicy::{Fixed, Adaptive}`。
- **永続化**：`config.json` と `model.safetensors`（原子的一時ファイル＋リネーム、`f64` 因子 mmap 読込、`PersistRegistry`）。

## 設計の特徴

- **C++23 とエラー契約**：API 境界に例外を送出せず `[[nodiscard]] std::expected<T, GprError>` を返す。
- **列優先（Column-major）**：データ行列は `MatrixRef<double>` または `std::span<const double>`（`x`, `n_rows`, `n_cols`）として列優先で受け渡す。
- **公開ヘッダ非漏洩**：Eigen・`nlohmann::json`・OpenMP は内部実装に閉じ込め、`include/cxxgpr/` に露出させない。
- **ライセンス**：Apache-2.0（`LICENSE`）。第三者通知は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) に記録する。

## ビルドと組み込み

### 動作要件

- `std::expected` に対応した C++23 コンパイラ（MSVC 19.44+ / Visual Studio 17.14+、または GCC 14+）
- CMake 3.25+、Ninja、および `vcpkg`（環境変数 `VCPKG_ROOT` を設定）

### プリセットによるビルドと試験

```sh
# Linux (GCC 14)
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release

# Windows (MSVC、Developer PowerShell または VCPKG_ROOT 設定済み環境)
cmake --preset windows-msvc
cmake --build --preset windows-release
ctest --preset windows-release
```

### インストールと `find_package(cxxgpr)`

```sh
cmake --install out/build/linux-release --prefix /path/to/install
```

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES CXX)

find_package(cxxgpr 0.1.0 CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

### `FetchContent`

```cmake
include(FetchContent)
FetchContent_Declare(
    cxxgpr
    GIT_REPOSITORY https://github.com/YUKIKEDA/cxxgpr.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(cxxgpr)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

### 最小利用例

```cpp
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>

#include <array>
#include <iostream>

int main() {
    const std::array<double, 4> x_train{0.0, 1.0, 2.0, 3.0};
    const std::array<double, 4> y_train{0.0, 1.0, 0.5, 1.2};
    const std::array<double, 2> x_query{0.5, 1.5};

    const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec).with_noise_variance(1e-4);
    if (!trainer) {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }

    auto fitted = trainer->fit(x_train, 4U, 1U, y_train);
    if (!fitted) {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }

    auto pred = fitted->predict(x_query, 2U, 1U);
    if (!pred) {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    std::cout << pred->mean[0] << ", " << pred->mean[1] << '\n';
    return 0;
}
```

## 文書

| 文書 | 内容 |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | マイルストーン・依存グラフ・現在地 |
| [`docs/design.md`](docs/design.md) | 要求・C++23 API・数理式・バッファ設計 |
| [`docs/persist-format.md`](docs/persist-format.md) | `config.json` と `model.safetensors` 仕様 |
| [`docs/architecture.md`](docs/architecture.md) | レイヤ構成と依存方向 |
| [`docs/conventions.md`](docs/conventions.md) | ディレクトリ構成と命名規約 |
| [`docs/quality/quality-plan.md`](docs/quality/quality-plan.md) | 追跡・基準結果・結果変更管理・来歴 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 開発手順とゲート |
