# はじめての回帰

このチュートリアルでは、$\sin(x)$ にノイズを加えた 8 個の点にガウス過程を当てはめ、新しい 3 点について予測値とその不確かさを求める。終わるころには、動くプログラムが 1 本手元に残り、その各行が何をしているかも分かるようになる。

cxxgpr が使える状態であることが前提である。まだなら先に {doc}`../how-to/install-and-link` を済ませてほしい。ガウス過程の知識は要らない。必要になった時点で、その場で説明する。理論の全体は {doc}`../explanation/gaussian-processes` にある。

## 作るもの

次の処理をするプログラムである。

1. $y_i \approx \sin(x_i)$ となる 8 個の訓練点 $(x_i, y_i)$ を持つ。
2. カーネルで関数の滑らかさを表す。
3. モデルを当てはめる。このとき、カーネルのハイパーパラメータもデータから学習される。
4. $x = 1.35$、$3.15$、$5.85$ での予測平均と標準偏差を表示する。

## ステップ 1：ヘッダとデータ

`first_regression.cpp` を作る。まずインクルードと `main` の開始部分を書く。

```{code-block} cpp
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>

int main()
{
```

次にデータを書く。

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [data]
:end-before: // [end]
:dedent: 4
```

入力特徴量は 1 つなので、訓練行列は 8 行 1 列である。cxxgpr は行列を**列優先**（column-major）で読む。0 列目の値がすべて先に並び、続いて 1 列目の値が並ぶ。列が 1 つなら 8 個の数が並ぶだけなので、普通の `std::array<double, 8>` がそのまま行列になる。目的変数 `y_train` は、$\sin(x)$ に約 $\pm 0.1$ のノイズを手で足した値である。実行のたびに同じ結果になるよう、乱数ではなく固定値にしてある。

## ステップ 2：モデルを記述する

ガウス過程には**カーネル**が要る。カーネル $k(x, x')$ は、2 つの入力での関数値がどれほど強く結びついているかを表す関数である。ここでは 2 つの部品の積を使う。

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [model]
:end-before: // [end]
:dedent: 4
```

1 行ずつ見ていく。

- `ConstantKernel::create(1.0)` は定数 $c$ である。ほかのカーネルに掛けると**信号分散**になる。関数が平均からどこまで離れうるかを決める量である。
- `RbfKernel::create(1.0)` は二乗指数カーネル $\exp\!\bigl(-(x - x')^2 / (2\ell^2)\bigr)$ で、長さスケールは $\ell = 1$ である。近い入力には近い出力が対応する。「近い」の尺度が長さスケールである。
- `create` は失敗しうる（たとえば長さスケールが負のとき）。そのため `std::expected` を返す。cxxgpr は API の境界で例外を投げない。失敗しうる呼び出しはすべて値を返し、`if (!result)` で確かめる。ここでは 2 つのカーネルを一度に確かめて先へ進む。
- `KernelSpec(*signal) * KernelSpec(*shape)` は、各カーネルを `KernelSpec` に包んで掛け合わせる。和は `+` で同様に書ける。
- `Gpr<optimizer::Lbfgs>(spec)` は、L-BFGS でハイパーパラメータを調整する、未学習のモデル記述を作る。`with_noise_variance(0.01)` は観測ノイズの*初期値* $\sigma_n^2 = 0.01$ を設定する。この値も最適化で調整される。ノイズ分散が 0 以下だと拒否されるため、戻り値は `std::expected` である。
- `with_target_transform(StandardizeTarget{})` は、学習の前に目的変数を平均 0・標準偏差 1 へ変換し、予測を返すときに元へ戻すよう指示する。標準化しておくと、どんな大きさのデータでも初期値が妥当になる。

`with_...` は、どれも元の記述を変えずに**新しい**記述を返す。基本の設定から複数の変種を枝分かれさせて作れる。

:::{note}
`Gpr` は学習器であって、モデルではない。学習済みのオブジェクトは次のステップで得る。
:::

## ステップ 3：当てはめる

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

`fit(x, n_rows, n_cols, y)` は、列優先の行列をスパンと形状で受け取り、目的変数も受け取る。目的変数を標準化し、データがもっとも尤もらしくなるハイパーパラメータを探し、最終的な共分散行列を分解して、`FittedGpr` を返す。`fit` は `const` なので、成功しても失敗しても学習器 `configured` は変わらない。失敗したときは `GprError` が返り、`message()` で原因が分かる。たとえばデータに非有限値が含まれる場合や、サイズが合わない場合である。

## ステップ 4：予測する

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
:dedent: 4
```

`predict(xs, n_rows, n_cols)` は、クエリ点を同じ列優先の配置で受け取る。結果は `Prediction` で、`mean` と `variance` の 2 本のベクトルを持つ。要素はクエリ点ごとに 1 つで、値は目的変数の元のスケールである。分散は既定では、*新しいノイズ付き観測*の分散であり、測定ノイズ $\sigma_n^2$ を含む。この分散の平方根が、次に表示する標準偏差である。

## ステップ 5：結果を見る

プログラムの残りは、学習結果を表示し、予測を真の $\sin(x)$ と比べる。

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [report]
:end-before: // [end]
:dedent: 4
```

カーネルのパラメータは、最適化中も正の値に保たれるよう、対数で保持されている。そのため `std::exp` で戻している。このブロックの後に、クエリ点ごとに 1 行を表示するループを書き、`main` を閉じる（完全なリストはページの末尾にある）。ビルドして実行しよう。ビルドの手順は {doc}`../how-to/install-and-link` にある。次のように出力される。

```{code-block} text
signal variance = 1.91418, lengthscale = 1.71027
noise variance = 0.00549421 (standardised scale)
log marginal likelihood = 5.74717
x = 1.35  mean = 1.0122  std = 0.0652796  sin(x) = 0.975723
x = 3.15  mean = -0.108219  std = 0.0648688  sin(x) = -0.00840725
x = 5.85  mean = -0.289365  std = 0.067871  sin(x) = -0.419764
```

数値は次のように読む。

- 最適化は、信号分散 1・長さスケール 1 から出発し、$c \approx 1.91$、$\ell \approx 1.71$ に移った。長さスケール 1.7 は、$0$ から $6.3$ にわたる正弦波のゆるやかな波に見合っている。ノイズ分散も、こちらが与えた 0.01 から約 0.0055（標準化後の単位）へ下がった。データが滑らかだからである。
- どのクエリでも、平均は真の $\sin(x)$ に近く、差は標準偏差約 0.065 の数倍以内に収まっている。プログラムは、差がすべて標準偏差の 3 倍未満のときだけ終了コード 0 で終わる。つまり、この主張は実行のたびに検査される。
- 3 点の標準偏差がほぼ同じなのは、どの点も、0.9 間隔の訓練点のあいだにあるからである。$[0, 6.3]$ のはるか外側を尋ねると、標準偏差が増える。次のチュートリアルでそれを確かめる。

## 完全なプログラム

:::{dropdown} ファイル全体を表示する
```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:linenos:
```
:::

## 学んだこと

カーネルでモデルを記述し、変更しないビルダー呼び出しで設定し、`fit` で当てはめ、`predict` で平均と不確かさを得た。途中で、`std::expected` によるエラー処理、列優先の行列、目的変数の標準化にも触れた。

次は {doc}`multi-dimensional-regression` に進み、尺度が大きく違う複数の入力を扱う。個々の部品については {doc}`../how-to/build-kernels`、{doc}`../how-to/transform-data`、{doc}`../how-to/predict-and-sample` を参照してほしい。
