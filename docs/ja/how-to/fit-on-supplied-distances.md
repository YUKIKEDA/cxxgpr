# 与えた距離でモデルを学習する

自分で計算した二乗距離（弧長、グラフ距離、編集距離など）を読むカーネルでガウス過程を学習し、予測・サンプリング・交差検証まで行う手順を示す。学習器は `DistanceGpr`、学習済みモデルは `FittedDistanceGpr` で、どちらも `cxxgpr/gpr/distance_gpr.hpp` にある。

先に {doc}`use-supplied-distances` を読むこと。距離表、スロット、`DistanceSource` はそちらで説明しており、ここでは繰り返さない。シグネチャは [gpr API リファレンス](../reference/api/gpr.md) を参照。

## DistanceGpr と Gpr の対応

`DistanceGpr` は、`DistanceKernel` で作ったカーネル向けの `Gpr` に当たる。学習の手順、最適化器、精度ポリシー、数値上の安全策は同じで、入力だけが違う。

| | `Gpr` / `FittedGpr` | `DistanceGpr` / `FittedDistanceGpr` |
|---|---|---|
| カーネル | `KernelSpec` | `DistanceKernel<DistanceOnly>` または `DistanceKernel<WithPoints>` |
| 学習データ | 座標 `x`（n × d）と `y` | 二乗距離表を持つ `DistanceSource` と `y`（`WithPoints` は `x` も） |
| 予測点のデータ | 座標 `xq` | 学習点から予測点への表を持つ `DistanceSource`（`WithPoints` は `xq` も） |

`DistanceOnly` のカーネルは座標を一切見ないので、モデルにも座標は要らない。`WithPoints` のカーネルは距離の葉と通常の座標の葉を組み合わせるため、表に加えて座標も渡す。

## 手順 1: データとカーネルを用意する

データは円周上の周期関数にノイズを乗せた標本で、尺度は弧長である。カーネルが読むのは弧長の二乗になる。

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [data]
:end-before: // [end]
```

カーネルは弧長の指数カーネルで、円周上では妥当である（理由は {doc}`use-supplied-distances` を参照）。`DistanceGpr` にはそのまま渡す。

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
```

## 手順 2: 学習する

スロットごとに表を `DistanceSource` へ結び付けてから `fit` を呼ぶ。表は 1 度だけ検査され、有限・非負・対称・対角が 0 でなければならない。`DistanceSource::tidy` を呼べば、決められた範囲の修復を許す。

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
```

`with_noise_variance` は引数を検査し、`Gpr` と同じく `std::expected` を返す。ほかの設定メソッドは `with_jitter`、`with_kernel_exp`、`with_target_transform`、`with_optimizer` で、`WithPoints` のカーネルに限り `with_input_transform` もある。getter は `kernel`、`likelihood`、`jitter`、`optimizer` である。

`fit` は周辺尤度を最大化して、カーネルのハイパーパラメータとノイズ分散を最適化する。最適化せずに現在のハイパーパラメータのまま分解したいときは `factor` を呼ぶ。`WithPoints` のカーネルでは、どちらも座標を追加の引数に取る（`fit(train, x, y)`、`factor(train, x, y)`）。

:::{note}
既定の精度ポリシーは倍精度である。`DistanceGpr<Lbfgs, MixedPrecision<ReevaluateKernel>>` など {doc}`optimize-performance` の各ポリシーも同じように使え、`refinement_summary` を報告する。
:::

## 手順 3: 予測する

すべての学習点から予測点への二乗距離の表（n × m、行が学習点）を作り、同じスロットへ結び付ける。スロットが複数あるカーネルでは、それぞれに表が要る。

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
```

`predict` は元の目的変数の尺度で、平均と周辺分散を返す。`predict_into` は自分で持つ `Prediction` へ書き込むので、繰り返し呼ぶときに記憶領域を再利用できる。`WithPoints` のモデルでは、表の後ろに予測点の座標を渡す（`predict(cross, xq)`）。

## 手順 4: 共分散・サンプル・交差検証

同時予測共分散と事後サンプルには、予測点どうしの距離表（m × m）も要る。

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [covariance]
:end-before: // [end]
```

- `predict_covariance(cross, query)` は、予測点の m × m の事後共分散全体を返す。
- `sample(cross, query, n_draws, seed)` は `n_draws` 個の同時サンプルを引く。結果は同じシードなら決定的で、スレッド数に依存しない。
- `loo_predict` は、再学習せずに学習点での leave-one-out 予測を返す。

## 手順 5: 調べる・更新する・再学習する

`FittedDistanceGpr` の参照用メソッドは `kernel`、`likelihood`（どちらも学習後の値）、`n_train`、`n_features`、`factor_jitter`、`neg_log_marginal_likelihood`、`alpha`、`l_factor`、`refinement_summary`、`optimizer_result` である。

`into_trainer` は、学習後のハイパーパラメータから始まる `DistanceGpr` を返す。`refit(train, y)` は、同じ設定のまま新しい表と目的変数で学習し直す。学習後の値から再開し、元のモデルは変わらない。`FittedDistanceGpr` のコピーは軽い。コピー同士は学習結果を共有し、作業領域は別々に持つので、スレッドごとに自分のコピーを使うこと。

## 不正な表とエラー

失敗はすべて `GprError` で返る。距離に固有のものは次のとおり。

| `GprErrorCode` | 原因 |
|---|---|
| `InvalidDistance` | 表が非有限・負・非対称、または対角が 0 でない |
| `DistanceSlot` | スロットが無い・二重に結び付いている、または表の形が合わない |
| `UnsupportedKernelOperation` | `WithPoints` のモデルに `predict(cross)` を呼んだ（座標を渡すこと） |

```{literalinclude} /../examples/distance_model.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
```

表は API の境界で検査されるので、不正な表は黙って誤ったモデルになるのではなく、エラーとして報告される。ほかのコードは {doc}`handle-errors` を参照。

## まとめ

| シンボル | 用途 |
|---|---|
| `DistanceGpr` | `DistanceKernel` の学習器。`fit`、`factor`、`with_*` の設定 |
| `FittedDistanceGpr` | 学習済みモデル。`predict`、`predict_into`、`predict_covariance`、`sample`、`loo_predict`、`into_trainer`、`refit` |

距離モデルの保存と読み込みは {doc}`persist-models` を参照。
