[English](README.md) | 日本語

# cxxgpr

**C++23 のための厳密ガウス過程回帰ライブラリ。** データにモデルを当てはめ、校正された不確かさつきで予測できます。数値・メモリ・スレッドを自分で制御できます。例外は投げず、ホットパスで確保せず、スレッド数が違っても結果はビット単位で一致します。

[![CI](https://github.com/YUKIKEDA/cxxgpr/actions/workflows/ci.yml/badge.svg)](https://github.com/YUKIKEDA/cxxgpr/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **状態：`0.1.0`。** 機能は揃い、検証も済んでいます。1.0 までは API が変わる可能性があります。

## 特長

- **エラーは値で返る。** 失敗しうる呼び出しはすべて `std::expected<T, GprError>` を返し、API の境界で例外は飛びません。
- **再現できる。** 並列計算の結果はスレッド数によらずビット単位で一致し、乱数による標本抽出もシード付きで可搬です。
- **コストが読める。** 2 回目以降の予測は確保ゼロで、メモリと速度のどちらを優先するかを選べます。
- **ヘッダがきれい。** Eigen・nlohmann/json・OpenMP は公開ヘッダに現れず、利用側のビルドへ漏れません。
- **構造で安全にする。** 失敗した呼び出しは対象を変えず、範囲つきのハイパーパラメータは範囲外にできず、学習前と学習後のモデルは別の型です。

## できること

| 領域 | 内容 |
|---|---|
| **モデル** | `Gpr`（学習器）と `FittedGpr`（学習済みモデル）。学習、予測平均と分散、同時共分散、事後標本、1 つ抜き交差検証、解析勾配とヘッセ行列つきの周辺尤度 |
| **カーネル** | 組込み 10 種（RBF、Matérn 1/2・3/2・5/2、周期、有理二次、定数、線形、白色、および ARD 版）、和と積、独自カーネル、事前計算した距離 |
| **最適化器** | L-BFGS、信頼領域、Nelder–Mead、高速焼きなまし、または固定。決定的な再開始 |
| **データ処理** | 入力と目的変数の標準化、最小最大、列ごとの変換。パイプラインで連結できる |
| **数値計算** | 倍精度・単精度・混合精度（反復改良）、適応的 Jitter、必要に応じた高速近似指数関数 |
| **永続化** | 学習済みモデルを可搬なディレクトリ（`config.json` と `model.safetensors`）へ保存・読込。メモリマップ読込も選べる |

疎な近似モデルとオンライン更新は、意図して含めていません。

## はじめの一歩

```cpp
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <iostream>

int main()
{
    // Eight training inputs: one column, so the column-major buffer is just the values.
    const std::array<double, 8> x_train{0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3};
    // y = sin(x) plus small hand-picked noise (standard deviation about 0.1).
    const std::array<double, 8> y_train{0.08, 0.83, 1.01, 0.33, -0.52, -1.07, -0.67, 0.12};
    // Three query inputs, one of them (3.15) between two training points.
    const std::array<double, 3> x_query{1.35, 3.15, 5.85};

    // k(x, x') = c * exp(-(x - x')^2 / (2 l^2)): signal variance c times an RBF kernel.
    auto signal = cxxgpr::ConstantKernel::create(1.0);
    auto shape = cxxgpr::RbfKernel::create(1.0);
    if (!signal || !shape)
    {
        std::cerr << "kernel construction failed\n";
        return 1;
    }
    const cxxgpr::KernelSpec spec = cxxgpr::KernelSpec(*signal) * cxxgpr::KernelSpec(*shape);

    // Initial observation noise variance 0.01 (it is optimised too); targets are standardised before fitting.
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec).with_noise_variance(0.01);
    if (!trainer)
    {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }
    const auto configured = trainer->with_target_transform(cxxgpr::transform::StandardizeTarget{});

    auto fitted = configured.fit(x_train, 8U, 1U, y_train);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }

    auto pred = fitted->predict(x_query, 3U, 1U);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }

    std::cout << "mean = " << pred->mean[0] << ", std = " << std::sqrt(pred->variance[0]) << '\n';
    return 0;
}
```

ノイズ入りの正弦曲線に当てはめ、予測平均と標準偏差を表示します。一行ずつの解説は[最初の回帰チュートリアル](docs/ja/tutorials/first-regression.md)にあります。

## 導入

**ビルド済みパッケージ。** [GitHub Release](https://github.com/YUKIKEDA/cxxgpr/releases) に、Windows（MSVC）と Linux（GCC 14）の利用可能なパッケージ、文書、SHA-256 の一覧を置きます。展開して、CMake から指すだけで使えます。

```cmake
find_package(cxxgpr 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cxxgpr::cxxgpr)
```

パッケージは、構成時に Eigen、nlohmann/json、（あれば）OpenMP を探します。vcpkg などで導入しておいてください。

**ソースから `FetchContent` で。**

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

必要なもの：`std::expected` に対応した C++23 コンパイラ（MSVC 19.44 以降、または GCC 14 以降）、CMake 3.25 以降、Ninja、`VCPKG_ROOT` を設定した vcpkg。ビルドオプションとプリセットは[導入ガイド](docs/ja/how-to/install-and-link.md)にあります。

## 文書

| | |
|---|---|
| [チュートリアル](docs/ja/tutorials/index.md) | 作りながら学ぶ：最初の回帰、多次元モデル、保存と読込 |
| [ハウツーガイド](docs/ja/how-to/index.md) | 目的別の手順。全体で公開 API をすべて網羅 |
| [リファレンス](docs/ja/reference/index.md) | API リファレンス、エラーコード、ビルドオプション、ファイル形式 |
| [解説](docs/ja/explanation/index.md) | ガウス過程、カーネル、数値安定性、設計の背景 |

文書サイトは `just docs` で自分でビルドできます。開発に参加する方は [`CONTRIBUTING.md`](CONTRIBUTING.md) と、開発者向け文書 [`docs/internal/`](docs/internal/roadmap.md)（英語）を読んでください。

## ライセンス

Apache-2.0。[`LICENSE`](LICENSE) を参照してください。第三者の通知は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) にあります。
