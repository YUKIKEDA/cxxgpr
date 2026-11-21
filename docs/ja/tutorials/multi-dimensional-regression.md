# 複数の入力を持つ回帰

このチュートリアルでは、尺度が 100 倍違う 2 つの入力を持つモデルを当てはめる。各入力がどれほど効くかをモデルに学習させ、データから離れると不確かさがどう増えるかを見て、最後に 1 点抜き交差検証（leave-one-out）で当てはまりを確かめる。{doc}`first-regression` の続きなので、先にそちらを済ませてほしい。

## 作るもの

知らないことにする関数は次のものである。

$$
f(x_1, x_2) = \sin(x_1) + \frac{x_2}{500}, \qquad x_1 \in [0, 6],\quad x_2 \in [0, 500].
$$

$x_1$ 方向には速く振動し、$x_2$ 方向にはゆっくり直線的に増える。この関数からノイズ付きの 30 点を取り、データだけからこの構造を復元する。

## ステップ 1：訓練データを作る

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [data]
:end-before: // [end]
:dedent: 4
```

訓練行列は 30 行（サンプル）2 列（特徴量）である。列優先では、$x_1$ の 30 個の値がすべて先に並び、続いて $x_2$ の 30 個が並ぶ。ループが第 1 特徴量を `x[i]`、第 2 特徴量を `x[n + i]` に書くのはそのためである。列 $j$ は、オフセット $j \cdot n$ から始まる。乱数生成器には `std::mt19937` を使っている。その出力は C++ 標準が完全に規定しているので、どの環境でも同じデータになる。

## ステップ 2：ARD カーネルと変換を選ぶ

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [model]
:end-before: // [end]
:dedent: 4
```

最初のチュートリアルとの違いは 2 つある。

**ARD カーネル。** `RbfArdKernel` は、入力の列ごとに長さスケールを 1 つずつ持つ二乗指数カーネルである（ARD は automatic relevance determination の略）。長さスケールが短い入力に沿っては、関数が速く変化する。長い入力は、ほとんど効かない。初期値は列ごとに 1 つずつ、スパンで与える。最適化が、それぞれを別々に学習する。

**標準化した入力。** `with_input_transform(StandardizeInput{})` は、カーネルがデータを見る前に、各列から平均を引いて標準偏差で割る。これがないと $x_2$ は $x_1$ の約 100 倍の大きさになり、初期の長さスケール 1 は $x_2$ にとって無意味な値になる。この変換は `fit` の中で訓練データから*学習*され、以後のクエリへは自動で適用される。クエリを自分で変換する必要はない。`with_target_transform(StandardizeTarget{})` も、目的変数に同じことをする。

あとで読み出す長さスケールは、$x_1$、$x_2$ の元の単位ではなく、**標準化後の単位**である。

## ステップ 3：当てはめて、ハイパーパラメータを読む

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

カーネル `ConstantKernel * RbfArdKernel` のパラメータは 3 つで、順に、定数、0 列目の長さスケール、1 列目の長さスケールである。`get_params` は、それらを対数のまま、ちょうどこの大きさのスパンへ複製する。大きさが違えば、黙って切り詰めずにエラーを返す。

## ステップ 4：データの内側と外側で予測する

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
:dedent: 4
```

クエリ行列は 3 行 2 列で、これも列優先である。先頭の 3 個が 3 つのクエリの $x_1$、後ろの 3 個が $x_2$ である。最初の 2 つのクエリは訓練領域の内側にある。3 つ目は $x_1 = 12$ で、データの $x_1$ の最大値の 2 倍にあたる。

## ステップ 5：1 点抜き予測で確かめる

テストデータがないとき、モデルの良し悪しはどう分かるのか。`loo_predict()` が答える。各訓練点について、その点を除いてあった場合の予測を返す。計算は既存の分解から閉じた式で行われ、30 回も当てはめ直す必要はない。

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [loo]
:end-before: // [end]
:dedent: 4
```

結果は 2 つの数に要約される。二乗平均平方根誤差は、1 点抜き予測の平均と観測値を比べる。平均二乗 $z$ スコアは、各誤差を予測標準偏差で割った値の二乗平均で、よく校正されたモデルなら 1 に近い。

## 実行する

次のように出力される。

```{code-block} text
noise variance = 0.00623201 (standardised scale)
lengthscale for x1 = 1.38772, for x2 = 9.63727 (standardised units)
query (3, 250): mean = 0.64008, std = 0.0652656, truth = 0.64112
query (1, 100): mean = 1.06294, std = 0.0732807, truth = 1.04147
query (12, 250): mean = 0.591749, std = 1.50877, truth = -0.0365729
leave-one-out RMSE = 0.0683095, mean squared z-score = 0.92736
```

数値から次のことが読み取れる。

- **ARD が構造を見つけた。** $x_2$ の長さスケールは約 9.6 で、$x_1$ の 1.39 の 7 倍ほどある。$x_2$ はゆっくり、$x_1$ は速く変わることを、教えなくてもモデルが学習した。
- **内挿は正確で、自信もある。** 内側の 2 つのクエリでは、平均が真値の 0.03 以内にあり、標準偏差は約 0.07 である。その大半は観測ノイズによる。データは標準偏差 $0.2/\sqrt{12} \approx 0.058$ の一様ノイズで作ってある。
- **外挿は正直である。** $x_1 = 12$ では標準偏差が 1.5 に跳ね上がる。約 20 倍である。平均 0.59 は真値 $-0.04$ から大きく外れている。モデルは分かったふりをせず、誤差の幅でそれを伝えている。この比が 5 を超えたときだけ、プログラムは終了コード 0 で終わる。
- **当てはまりは校正されている。** 1 点抜きの RMSE 0.068 はノイズの水準に近く、平均二乗 $z$ スコア 0.93 は 1 に近い。予測された不確かさが、実際の誤差とよく合っている。

## 完全なプログラム

:::{dropdown} ファイル全体を表示する
```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:linenos:
```
:::

## 学んだこと

ARD カーネルで、入力ごとに別の長さスケールを持たせた。入力と目的変数の標準化をモデルの内側で当てはめ、`get_params` で学習したハイパーパラメータを読み出した。データから離れるほど不確かさが増える様子を見て、`loo_predict` で当てはまりを検証した。

次の {doc}`save-and-reload` では、このようなモデルを保存し、あとで使う方法を扱う。個々の部品については {doc}`../how-to/build-kernels`、{doc}`../how-to/transform-data`、{doc}`../how-to/predict-and-sample` を参照してほしい。
