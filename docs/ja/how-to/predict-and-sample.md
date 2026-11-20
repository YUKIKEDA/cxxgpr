# 予測・サンプリング・交差検証を行う

このガイドは、学習済みの `FittedGpr` に問い合わせる方法を示す。点ごとの平均と分散、同時共分散、事後サンプル、閉形式の leave-one-out 予測を扱う。サンプリングに使う決定的な乱数生成器も紹介する。結果はすべて**元の目的変数のスケール**で返る。入力と目的変数の変換は、ライブラリが元に戻す。

スニペットは、ページ末尾に全文を載せた 1 つのプログラムから取っている。`using namespace cxxgpr;` を前提とし、`fitted` は、$[0, 6]$ 上の $\sin x$ にノイズを加えた 25 点で学習した `FittedGpr<optimizer::Lbfgs>` とする（学習の方法は {doc}`configure-fit` を参照）。

## 平均と分散を予測する

`predict` は `Prediction<Refine>` を返す。事後平均 `mean`、クエリ行ごとの分散 `variance`、`variance_kind` を持つ構造体である。クエリは、列優先の生データに行数と列数を添えて渡すか、`MatrixRef<double>` で渡す。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
```

```{code-block} text
x = 0.5000  mean = 0.4481  sd = 0.1006  sin(x) = 0.4794
x = 1.5000  mean = 1.0660  sd = 0.0993  sin(x) = 0.9975
x = 3.0000  mean = 0.1754  sd = 0.0991  sin(x) = 0.1411
x = 5.0000  mean = -1.0728  sd = 0.0999  sin(x) = -0.9589
```

平均は $\sin x$ をよく追っており、真の値は各予測の約 1.2 標準偏差以内に収まる。標準偏差（約 0.1）の大半は、学習で得たノイズによる。既定の分散が、新しい**観測値**の分散だからだ。

`xs` の列数は `n_features()` に一致し、すべて有限でなければならない。そうでなければ、`GprErrorCode::DimensionMismatch`、`GprErrorCode::NonFiniteInput`、`GprErrorCode::EmptyInput` で失敗する。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
```

```{code-block} text
two-column query: input dimension mismatch: X.ncols()=2, expected 1 (x_dim 2, expected 1)
```

予測分散が負になることはない。内部で潜在分散を 0 で切り上げてからノイズ分散を足すため、丸め誤差で負の値が出ることはない。

### 潜在分散と観測分散

`PredictOptions` のフィールドは `variance_kind` だけで、型は `VarianceKind` である。

| 値 | 意味 | 用途 |
|---|---|---|
| `VarianceKind::Observation`（既定） | 新しいノイズ入り測定値 $y_*$ の分散。潜在分散に $\sigma_n^2$ を**加えた**もの。 | これから測定するデータの予測区間。 |
| `VarianceKind::Latent` | ノイズを含まない関数 $f_*$ の分散。 | 背後にある関数の不確かさ。関数の事後サンプル。 |

返される `Prediction` の `variance_kind` が、どちらを得たかを記録する。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [variance-kind]
:end-before: // [end]
```

```{code-block} text
observation - latent variance = 0.0080 (noise variance 0.0080)
```

目的変数の変換が恒等なら、2 つの分散の差はちょうどノイズ分散になる。スケールを変える目的変数の変換を使う場合、差は、ノイズ分散にスケールの 2 乗を掛けた値になる。

`PredictOptions` は `==` で比較できる集成体なので、指示付き初期化子が使える。`PredictOptions{.variance_kind = VarianceKind::Latent}` のように書く。

## メモリ確保なしで予測する

`predict` は結果のために確保を行う。ループでは、1 つの `Prediction` を `predict_into` で使い回す。最初の呼び出しで、モデル内部の作業領域と出力ベクトルの大きさが決まる。その後、クエリ行数が同じか、それ以下なら、ヒープ確保は発生しない。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [predict-into]
:end-before: // [end]
```

平均と分散を返す関数は 4 つあり、役割が分かれている。

| 関数 | 結果 | 確保 | オプション |
|---|---|---|---|
| `predict(xs, options = {})` | `Prediction` を返す | あり | 省略可 |
| `predict_with(xs, options)` | `Prediction` を返す | あり | 必須 |
| `predict_into(xs, out, options = {})` | `out` を埋める | なし（ウォームアップ後） | 省略可 |
| `predict_with_into(xs, options, out)` | `out` を埋める | なし（ウォームアップ後） | 必須 |

`predict` と `predict_into` は、それぞれ `predict_with` と `predict_with_into` に処理を渡す。一般的な呼び出しで `PredictOptions` を書かずに済むための窓口である。どの関数にも、生の `std::span<const double>` に `n_rows` と `n_cols` を添えるオーバーロードがある。

:::{warning}
予測関数は `const` だが、`FittedGpr` の内部に置いた作業領域に書き込む。同じオブジェクトに対して、複数のスレッドから**同時に**呼んではならない。スレッドごとにコピーを持たせる。`FittedGpr` をコピーすると、独立した作業領域ができる。
:::

## 同時共分散を予測する

`predict_covariance` は `PredictiveCovariance<Refine>` を返す。`mean`、クエリ点間の $m \times m$ 共分散行列 `covariance`（列優先）、点数 `n_points`、`variance_kind` を持つ。要素 $(i, j)$ は `covariance[j * n_points + i]` である。`VarianceKind::Observation` では、対角に $\sigma_n^2$ が加わる。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [covariance]
:end-before: // [end]
```

```{code-block} text
cov(0, 1) = -0.0000, var(0) = 0.0022
```

対角は点ごとの潜在分散に等しく、行列は対称である（例のプログラムが両方を検査している）。2 つのクエリ点ではデータが関数をほぼ決めるため、事後誤差の相関はほとんどなく、非対角成分は 0 に近い。

`predict_covariance_with(xs, options)` はオプションを明示する形である。どちらにも生スパンのオーバーロードがある。結果は確保して返す。$m$ 点の共分散は $O(m^2)$ のメモリを使うので、$m$ は控えめにすること。確保済みバッファに書き込む `predict_covariance_into` は、低レベルのワークスペース API に属する（{doc}`use-low-level-building-blocks` を参照）。

## 事後サンプルを引く

`sample` は、クエリ点での同時事後分布から関数を引く。返す `std::vector<Refine>` は、$m \times n_\text{draws}$ の列優先行列である。`j` 番目のサンプルは、添字 `j * m` から `j * m + m - 1` までを占める。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [sample]
:end-before: // [end]
```

```{code-block} text
sample mean 0.4484 (posterior 0.4481), sample variance 0.0021 (posterior 0.0022)
```

4000 個のサンプルで、経験平均と経験分散は事後のモーメントを再現する。使われる共分散は `options.variance_kind` に従う。ノイズを含まない関数を引くには `VarianceKind::Latent` を指定する。既定の `VarianceKind::Observation` では、サンプルの各値に観測ノイズも加わる。

### シードと再現性

引数 `seed` は、呼び出しの内部で `Xoshiro256PlusPlus` を初期化する。モデル、クエリ、シード、オプションが同じなら、サンプルはビット単位で同一になる。シードを変えると、サンプルも変わる。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [seed]
:end-before: // [end]
```

```{code-block} text
same seed identical: 1, different seed identical: 0
```

### サンプリング用のジッタ

サンプリングでは、クエリ共分散を分解する。この行列は半正定値にすぎず、近い点を含むと数値的に特異に近づく。そのため、分解はジッタで安定化される。何も渡さなければ、適応ポリシーが使われる（最初のオフセット `1e-12`、倍率 10、再試行は最大 10 回、上限 `1e-2`）。最後の引数で上書きできる。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [jitter]
:end-before: // [end]
```

それでも分解に失敗すると、`sample` は `GprErrorCode::CholeskyFailed` を返す。`n_draws = 0` なら、空のベクトルを返す。`sample_with(xs, n_draws, seed, options, jitter)` はオプションを明示する形で、`sample` には生スパンのオーバーロードもある。`JitterPolicy` は {doc}`configure-fit` を参照。

## leave-one-out 予測

`loo_predict` は、各学習点について、**その点を除く**全データで学習したモデルの予測を返す。閉形式の恒等式（$A = K + \sigma_n^2 I$ の逆行列の対角を使う。Rasmussen と Williams の 5.4.2 節）を使うので、再学習は要らない。コストは、逆行列の対角を 1 回求める程度である。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [loo]
:end-before: // [end]
```

```{code-block} text
LOO for training point 10: mean 0.6425, sd 0.1018, observed 0.5218
explicit refit without point 10: mean 0.6425, sd 0.1018
```

2 行目は、プログラム内の照合結果である。点 10 を取り除き、同じハイパーパラメータでモデルを分解し、その点を予測している。2 つは $10^{-6}$ より良く一致する。LOO で固定されるのは**ハイパーパラメータ**である。除外した点を抜いてハイパーパラメータを再最適化するわけではない。

`loo_predict(options = {})` と `loo_predict_with(options)` は、学習行と対応する `Prediction` を返す。既定の分散はノイズ入りの観測値のものである。ノイズを差し引くには `VarianceKind::Latent` を使う。失敗するのは、逆スケーリングで非有限値が出たときの `GprErrorCode::NonFiniteKernelValue` だけである。

## 乱数を自分で生成する

`Xoshiro256PlusPlus`（別名 `SeededRng`）は、ライブラリがサンプリングと最適化器の再スタートに使う決定的な生成器と同じものである。SplitMix64 でシードを展開し、プラットフォームによらず同じ列を生成する。再現可能な合成データの作成に便利である。`constexpr` で構築でき、`==` で比較できる。

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:start-after: // [rng]
:end-before: // [end]
```

```{code-block} text
u = 0.8143, normal = 1.5043, pick = 7, open = 0.7935, word = 10848501901068131965, fresh generator equals advanced one: 0
```

| メソッド | 戻り値 |
|---|---|
| `next_u64()` | 次の生の 64 ビット語 |
| `unit()` | $[0, 1)$ の一様乱数（`double`） |
| `open_unit()` | $(0, 1)$ の内部の一様乱数（`std::log` に安全に渡せる） |
| `up_to(max)` | $[0, \text{max}]$ の一様な整数（上端を含む。偏りなし） |
| `unit_normal()` | Box-Muller 法による標準正規乱数 |

:::{note}
この生成器はスレッドセーフではない。スレッドごとに別々のシードで 1 つずつ使うこと。
:::

## プログラム全文

```{literalinclude} /../examples/predict_and_sample.cpp
:language: cpp
:linenos:
```

## API 一覧

| シンボル | 用途 |
|---|---|
| `FittedGpr::predict`、`predict_with` | 事後平均と分散。結果を確保して返す。 |
| `predict_into`、`predict_with_into` | 同じ計算を、呼び出し側が持つ `Prediction` に書く。ウォームアップ後は確保なし。 |
| `predict_covariance`、`predict_covariance_with` | 平均と完全な共分散を `PredictiveCovariance` で返す。 |
| `sample`、`sample_with` | 決定的な事後サンプル。$m \times n_\text{draws}$ の列優先。 |
| `loo_predict`、`loo_predict_with` | 閉形式の leave-one-out の平均と分散。 |
| `Prediction`（`mean`、`variance`、`variance_kind`） | 点ごとの結果。 |
| `PredictiveCovariance`（`mean`、`covariance`、`n_points`、`variance_kind`） | 同時分布の結果。 |
| `PredictOptions`（`variance_kind`） | 呼び出しごとのオプション。 |
| `VarianceKind`（`Latent`、`Observation`） | 報告する分散の種類。 |
| `Xoshiro256PlusPlus`、`SeededRng` | 決定的な生成器。`next_u64`、`unit`、`open_unit`、`up_to`、`unit_normal`。 |
| `MatrixRef` | 所有しない列優先のクエリビュー。 |

生成された API リファレンス: [`gpr`](../reference/api/gpr.md)、[`core`](../reference/api/core.md)。
