# データを変換する

ガウス過程のハイパーパラメータは、入力と目的変数のスケールがそろっているほど学習しやすい。このガイドでは、`cxxgpr/transform/` の変換クラスを選び、学習し、組み合わせ、適用し、自作する手順を示す。あわせて、モデルが予測値を元の単位へ戻す仕組みも説明する。対象は `IdentityInput`・`StandardizeInput`・`MinMaxInput`、対応する `Fitted*` クラス、目的変数の変換、`ColumnwiseInput`、`Pipeline`、`TargetPipeline`、そして自作変換で実装する 2 つの基底インターフェースである。

このページのコードは、ドキュメントと一緒にビルド・実行している 2 つのプログラム `transforms.cpp` と `custom_transform.cpp` から取り出したものである。ここで扱うクラスはすべて `cxxgpr/transform/transform.hpp` を include すれば使える。

## 変換の 2 つの姿を知る

変換には、入力側でも目的変数側でも、次の 2 つの姿がある。

| 姿 | 入力側 | 目的変数側 | 持つもの | 役割 |
|---|---|---|---|---|
| 未学習（レシピ） | `UnfittedInputTransform` | `UnfittedTargetTransform` | 設定のみ | `fit` が学習データを読み、学習済み変換を返す |
| 学習済み（固定した写像） | `FittedInputTransform` | `FittedTargetTransform` | 学習データから得た統計量 | データを順方向・逆方向に写す |

モデルに渡すのは未学習の変換である。モデルは学習データでそれを `fit` し、レシピと学習結果の両方を保持する。レシピは新しいデータでの `refit` に使い、学習済みの写像は `predict` が使う。学習済み変換を直接触るのは、自分でデータを前処理したいときか、学習された統計量を確かめたいときだけである。

## 変換を選ぶ

| 変換 | 写像 | 向く場面 | 避ける場面 |
|---|---|---|---|
| `IdentityInput`, `IdentityTarget` | なし（検証だけ行う） | データがすでに適切なスケールにある。既定値である | 特徴量の桁が大きく異なる |
| `StandardizeInput`, `StandardizeTarget` | $z = (x - \mu)/\sigma$ | ほとんどの場面での第一候補。特徴量のスケールが違う、目的変数に大きなオフセットがある | 外れ値が $\mu$ と $\sigma$ を支配している |
| `MinMaxInput`, `MinMaxTarget` | $[lo, hi]$ への線形写像（既定は $[0, 1]$） | 特徴量に意味のある範囲がある。学習範囲を既知の区間にちょうど写したい | 新しいデータが学習範囲から大きくはみ出す |
| `ColumnwiseInput` | 列ごとに別の変換 | 列ごとに扱いを変えたい。たとえば 1 列は $[0,1]$ のままにし、別の列には独自の写像を使う | 全列に同じ変換を使う（素の変換を使えばよい） |
| `Pipeline`, `TargetPipeline` | 複数の変換を順に適用 | 非線形な写像のあとに再スケールしたい。対数のあとに標準化する、など | 変換が 1 つで足りる |

統計量は、つねに学習データだけから計算する。`StandardizeInput` と `StandardizeTarget` は*母*標準偏差（$n$ で割る）を使い、定数列には $0$ ではなく尺度 `1.0` を与える。`MinMaxInput` も定数列を同様に扱う。幅を `1.0` とみなすので、すべての値が範囲の下端に写る。

## モデルの中で変換を使う

トレーナーに `with_input_transform` と `with_target_transform` で変換を付ける。どちらも未学習の変換を参照で受け取り、コピーを保持する。

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [use-in-gpr]
:end-before: // [end]
:dedent: 4
```

このプログラムのデータでは、第 1 特徴量が $[0, 1]$、第 2 特徴量が $[1000, 5000]$、目的変数が $50$ 前後にある。モデルは学習前に入力と目的変数を標準化する。問い合わせ点は元の単位のまま渡せばよい。

```{code-block} text
prediction at (0.30, 3000): mean = 69.0443, sd = 0.0701, truth = 69.0211
```

### 予測値を元に戻す仕組み

学習済みモデルは、すべての計算を変換後の空間で行い、返す直前に結果を元の単位へ戻す。戻す写像は、目的変数の逆変換 $y = g(z)$ である。

| 量 | 変換式 | `FittedTargetTransform` のメソッド |
|---|---|---|
| 平均 | $g(\mu_z)$ | `inverse_transform_mean` |
| 分散 | $\sigma_z^2 \cdot \bigl(g'\bigr)^2$ | `inverse_transform_variance` |
| 共分散行列 | 全要素に $\bigl(g'\bigr)^2$ を掛ける | `inverse_transform_covariance` |

`StandardizeTarget` では $g(z) = \sigma z + \mu$ なので、分散に $\sigma^2$ を掛ける。`MinMaxTarget` では、導関数がデータ幅と出力幅の比になる。どちらも線形（アフィン）なので、写した分布はふたたび正確にガウス分布になり、共分散の変換も正確である。観測分散（既定）と潜在分散は同じ方法で戻す。入力変換は、カーネルを評価する前に問い合わせ点へ適用する。予測のために入力変換を逆にたどることはない。

:::{note}
基底クラスの `inverse_transform_covariance` は、`inverse_transform_variance` を呼ぶだけである。これが正確なのは、目的変数の変換がアフィンのときだけである。非線形な目的変数変換を自作する場合は、`inverse_transform_covariance` も上書きすること。非線形に写したガウス分布はガウス分布でなくなり、共分散を 1 つの倍率では表せないからである。
:::

## 自分で変換を学習して適用する

学習結果の具体的な型とアクセサが欲しいときは `fit_typed` を、`std::unique_ptr<FittedInputTransform>` として多態的に持ちたいときは `fit` を使う。どちらも `std::expected` を返し、例外は投げない。

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [standardize-input]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
column means: 0.5000, 2800.0000
column stds:  0.3035, 1107.8666
```

`FittedStandardizeInput` は学習した統計量を `mean()` と `std()` で公開する。`FittedMinMaxInput` は `min()`、`max()`、`feature_range()` で公開する。`MinMaxInput::create(lo, hi)` で作った `MinMaxInput` では、`feature_range()` が設定した区間を返す。既定構築の `MinMaxInput` は $[0, 1]$ に写す。

### その場で変換するか、別の領域へ出力するか

入力変換には 4 つの入口がある。行列は列優先で、サンプルが `n_rows` 行、特徴量が `n_cols` 列である。

| メソッド | 動作 |
|---|---|
| `apply(x, n_rows, n_cols)` | `x` をその場で変換する |
| `apply_into(x, n_rows, n_cols, out)` | 確保済みの `out` へ変換結果を書き出す。メモリ確保は行わない |
| `inverse_apply(x, n_rows, n_cols)` | 変換後のデータを元の座標へその場で戻す |
| `inverse_apply_into(x, n_rows, n_cols, out)` | 同じ処理を `out` へ書き出す |

目的変数の変換にも同じ対応がある。順方向は `transform` と `transform_into`、予測平均の逆変換は `inverse_transform_mean` と `inverse_transform_mean_into`、分散の逆変換は `inverse_transform_variance` と `inverse_transform_variance_into` である。

逆変換は、浮動小数点の丸めの範囲で正確である。

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [round-trip]
:end-before: // [end]
:dedent: 4
```

どの入口も、最初に入力を検証する。バッファが空か、いずれかの extent が 0 なら `GprErrorCode::EmptyInput` を返す。`n_rows * n_cols` がオーバーフローすれば `GprErrorCode::SizeOverflow`、バッファの大きさが `n_rows * n_cols` と違えば `GprErrorCode::LengthMismatch` を返す。`n_cols` が学習時の列数と違うと `GprErrorCode::DimensionMismatch`、値が `NaN` か無限大なら `GprErrorCode::NonFiniteInput` になる。`IdentityInput` も同じ検証を行う。変換の有無でエラーの振る舞いが変わらないようにするためである。

### 保存済みの統計量から学習済み変換を作る

統計量がすでにわかっているとき（前回の実行で得た値など）は、学習を省いて学習済み変換を直接作れる。`FittedStandardizeInput::from_parts(mean, std)`、`FittedMinMaxInput::from_parts(data_min, data_max, range_lo, range_hi)`、`FittedStandardizeTarget::from_parts(mean, std)`、`FittedMinMaxTarget::from_parts(data_min, data_max, range_lo, range_hi)` である。これらは引数を検証する。ベクトルは空でなく同じ長さであること、値が有限であること、尺度が正であること（違反は `GprErrorCode::InvalidHyperparameter`）、最小値が最大値を超えないこと、範囲の両端が有限で `range_lo < range_hi` であること（違反は `GprErrorCode::InvalidConfig`）を確かめる。`FittedIdentityInput` と `FittedIdentityTarget` は状態を持たず、既定構築できる。組込みの変換クラスはすべて `operator==` を持ち、設定や統計量を値で比較する。

## 列ごとに別のスケールを使う

`ColumnwiseInput` は、特徴量の列ごとに未学習の変換を 1 つずつ持ち、それぞれを独立に適用する。`then` で列の順に追加する。`fit`（または `fit_typed`）には、行列の列数と同じ数の変換が必要である。

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [min-max-and-columnwise]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
columnwise: row 0 -> (0.0000, -0.8160), row 19 -> (1.0000, 0.8160)
```

列 0 は `IdentityInput` を通り、列 1 は $[-1, 1]$ へ写された。`then` は `const UnfittedInputTransform&`（コピーされる）か `std::unique_ptr<UnfittedInputTransform>`（移動される）を受け取り、自分自身を返すので、呼び出しをつなげて書ける。`size()`、`empty()`、`maps()` が、保持する変換の数と中身を返す。学習結果の `FittedColumnwiseInput` にも `size()`、`empty()`、`maps()` と 4 つの適用メソッドがある。列数が合わなければ `GprErrorCode::DimensionMismatch`、変換が null なら `GprErrorCode::InvalidConfig` になる。どちらのクラスも `clone()` で深くコピーされる。

## 変換をつなぐ

`Pipeline` は、未学習の入力変換を順に適用する。学習のとき、各ステップは直前までのステップの*出力*で学習される。適用は順方向に、逆変換は逆順に実行される。

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [pipeline]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
pipeline steps: 2
```

`TargetPipeline` は目的変数に対して同じことを行う。平均・分散・共分散の逆変換は、ステップごとに逆順で適用される。どちらのパイプラインも `then(const T&)` と `then(std::unique_ptr<T>)` を受け付け、`size()`、`empty()`、`steps()` を持つ。空のパイプラインは恒等変換とまったく同じに振る舞う。学習結果が `FittedPipeline` と `FittedTargetPipeline` で、`fit_typed` はそのまま返し、`fit` はポインタに包んで返す。

## 目的変数を変換する

目的変数の変換は、ただのベクトルに対して働く。`fit_typed` が返す学習済み変換のアクセサで、学習結果を確かめられる。

```{literalinclude} /../examples/transforms.cpp
:language: cpp
:start-after: // [target-transform]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
target mean = 49.0000, std = 15.4347
original-scale mean = 56.7173, variance = 9.5292
```

変換後の空間で平均 $0.5$ は $0.5 \sigma + \mu$ に、分散 $0.04$ は $0.04\,\sigma^2$ に戻る。`FittedStandardizeTarget` には `mean()` と `std()`、`FittedMinMaxTarget` には `min()`、`max()`、`feature_range()` がある。`MinMaxTarget::create(lo, hi)` と `MinMaxTarget::feature_range()` は入力側と対応している。`StandardizeTarget` と `MinMaxTarget` は、学習用の目的変数だけから学習する。

## 変換を自作する

未学習のクラスと学習済みのクラスを 1 つずつ実装する。次の例は対数による入力変換である。特徴量が数桁にわたり、目的変数が $\log x$ に対して滑らかに変化する場合に役立つ。学習済み側は列数を覚えておき、$z = \log x$ とその逆を計算する。

```{literalinclude} /../examples/custom_transform.cpp
:language: cpp
:start-after: // [fitted]
:end-before: // [end]
:dedent: 4
```

未学習側は、学習用の行列を検証して学習済み側を作る。

```{literalinclude} /../examples/custom_transform.cpp
:language: cpp
:start-after: // [unfitted]
:end-before: // [end]
:dedent: 4
```

組込みの変換と同じように使える。ここでは `Pipeline` で対数のあとに標準化をつなぐ。

```{literalinclude} /../examples/custom_transform.cpp
:language: cpp
:start-after: // [use]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
x = 3: mean = 2.96786, truth = 2.9683, |error| = 0.00043564
x = 30: mean = 1.19237, truth = 1.19253, |error| = 0.000163227
x = 300: mean = 2.53224, truth = 2.53233, |error| = 9.33199e-05
negative query rejected: yes
```

### 実装が必須のもの

| インターフェース | 必須 | 任意（既定の実装がある） |
|---|---|---|
| `FittedInputTransform` | `apply`、`inverse_apply`、`clone` | `apply_into`、`inverse_apply_into`（コピーしてから、その場で変換する版を呼ぶ）、`persist_id`、`persist_state_json` |
| `UnfittedInputTransform` | `fit`、`clone` | `persist_id`、`persist_state_json` |
| `FittedTargetTransform` | `transform`、`inverse_transform_mean`、`inverse_transform_variance`、`clone` | `transform_into`、`inverse_transform_mean_into`、`inverse_transform_variance_into`、`inverse_transform_covariance`、`inverse_transform_covariance_into`、`persist_id`、`persist_state_json` |
| `UnfittedTargetTransform` | `fit`、`clone` | `persist_id`、`persist_state_json` |

組込みの変換と同じ振る舞いにするため、次の規則を守る。

- 呼び出しのたびに検証する。入力が空でないこと、列数が期待どおりであること（`GprError::dimension_mismatch`）、バッファが隙間なく詰まっていること（`GprError::length_mismatch`）、値が有限であること（`GprError::non_finite_input`）を確かめる。`apply` の中の検証は基底クラスが肩代わりしてくれない。既定の `apply_into` と `inverse_apply_into` は、`x` と `out` が別の領域なら `x` を `out` へコピーし、そのあとで自作のその場版を呼ぶ。
- `inverse_apply` は、モデルが生成する範囲で `apply` の正確な逆にする。
- `clone` は深いコピーを返す。ライブラリは、モデル、パイプライン、トレーナーをコピーするときに変換を `clone` する。
- 写せない入力にはドメインエラーを返す。`LogInput` が正でない値を拒否するのと同じである。モデルは、そのエラーを `fit` や `predict` の戻り値としてそのまま返す。
- 目的変数の変換では、`inverse_transform_variance` が逆写像の導関数の 2 乗を掛けなければならない。写像が非線形なら、前述の注意のとおり `inverse_transform_covariance` も上書きする。

既定の `persist_id()` は空文字列を返し、その変換は保存できないものとみなされる。`save` は `PersistErrorKind::NotPersistable` で失敗する。自作の変換を使うモデルを保存するには、`persist_id` と `persist_state_json` を上書きし、ファクトリを登録する。手順は {doc}`persist-models` にある。

## エラーを扱う

このモジュールの失敗しうる呼び出しは、すべて `std::expected<..., GprError>` を返す。よく出会うコードは次のとおりである。

| コード | 原因 |
|---|---|
| `GprErrorCode::EmptyInput` | バッファが空、または `n_rows` か `n_cols` が 0 |
| `GprErrorCode::LengthMismatch` | `x.size() != n_rows * n_cols`、または `out` の大きさが違う |
| `GprErrorCode::DimensionMismatch` | `n_cols` が学習時の列数と違う |
| `GprErrorCode::NonFiniteInput` | データに `NaN` か無限大がある |
| `GprErrorCode::InvalidConfig` | `MinMaxInput::create` の範囲が不正、またはステップや変換が null |
| `GprErrorCode::InvalidHyperparameter` | `from_parts` で尺度が正でない、または `data_min > data_max` |

全コードの一覧と `GprError` の調べ方は {doc}`handle-errors` を参照。

## API 一覧

### 入力変換（`input.hpp`）

| シンボル | 役割 |
|---|---|
| `FittedInputTransform` | 学習済み入力写像のインターフェース。`apply`、`apply_into`、`inverse_apply`、`inverse_apply_into`、`clone`、`persist_id`、`persist_state_json` |
| `UnfittedInputTransform` | 入力変換のレシピのインターフェース。`fit`、`clone`、`persist_id`、`persist_state_json` |
| `IdentityInput`, `FittedIdentityInput` | 値を変えない。行列を検証する。`fit_typed`、`fit`、`clone`、`operator==` |
| `StandardizeInput` | 列ごとの平均と母標準偏差。`fit_typed`、`fit`、`clone`、`operator==` |
| `FittedStandardizeInput` | `from_parts`、`mean`、`std`、4 つの適用メソッド、`clone`、`operator==` |
| `MinMaxInput` | $[lo, hi]$ への線形写像。`create`、`feature_range`、`fit_typed`、`fit`、`clone`、`operator==` |
| `FittedMinMaxInput` | `from_parts`、`feature_range`、`min`、`max`、4 つの適用メソッド、`clone`、`operator==` |

### 目的変数の変換（`target.hpp`）

| シンボル | 役割 |
|---|---|
| `FittedTargetTransform` | `transform`、`transform_into`、`inverse_transform_mean`、`inverse_transform_mean_into`、`inverse_transform_variance`、`inverse_transform_variance_into`、`inverse_transform_covariance`、`inverse_transform_covariance_into`、`clone`、`persist_id`、`persist_state_json` |
| `UnfittedTargetTransform` | `fit`、`clone`、`persist_id`、`persist_state_json` |
| `IdentityTarget`, `FittedIdentityTarget` | 値を変えない。`fit_typed`、`fit`、`clone`、`operator==` |
| `StandardizeTarget` | 目的変数の平均と母標準偏差 |
| `FittedStandardizeTarget` | `from_parts`、`mean`、`std` |
| `MinMaxTarget` | `create`、`feature_range`、`fit_typed`、`fit` |
| `FittedMinMaxTarget` | `from_parts`、`feature_range`、`min`、`max` |

### 合成（`columnwise.hpp`、`pipeline.hpp`）

| シンボル | 役割 |
|---|---|
| `ColumnwiseInput`, `FittedColumnwiseInput` | 特徴量の列ごとに 1 つの写像。`then`、`size`、`empty`、`maps`、`fit_typed`、`fit`、`clone` |
| `Pipeline`, `FittedPipeline` | 入力写像を順に適用。`then`、`size`、`empty`、`steps`、`fit_typed`、`fit`、`clone` |
| `TargetPipeline`, `FittedTargetPipeline` | 目的変数の写像を順に適用。メンバーは同じで、目的変数の逆変換メソッドを持つ |

これらのヘッダの生成リファレンスは [transform API リファレンス](../reference/api/transform.md) にある。
