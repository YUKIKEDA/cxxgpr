# 学習を設定する

このガイドは、学習の前に `Gpr` トレーナーを設定する方法を示す。観測ノイズ、ジッタ、精度、最適化器、指数関数のバックエンド、ワークスペースのメモリを扱う。末尾に、関係する公開シンボルの一覧表を置く。

各スニペットは、ページ末尾に全文を載せた 1 つのプログラムから取っている。`using namespace cxxgpr;` と、`cxxgpr/gpr/gpr.hpp`、`cxxgpr/kernel/kernel.hpp`、`cxxgpr/optimizer/optimizer.hpp`、`cxxgpr/transform/transform.hpp` のインクルードを前提とする。

## トレーナーから始める

`Gpr<O, P, K>` は**未学習**のトレーナーである。テンプレート引数は 3 つある。最適化器 `O`（既定は `optimizer::Lbfgs`）、精度ポリシー `P`（既定は `DoublePrecision`）、カーネル仕様 `K`（常に `KernelSpec`）である。トレーナーに `predict` はない。学習すると `FittedGpr` になり、予測はそちらで行う（{doc}`predict-and-sample` を参照）。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
```

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [builder]
:end-before: // [end]
```

コンストラクタ `Gpr(KernelSpec, O)` は、カーネル（{doc}`build-kernels` を参照）と、省略可能な最適化器を受け取る。`with_*` メソッドはすべて `const` で、変更を加えた**コピー**を返す。元のトレーナーは変わらない。1 つの基底トレーナーから複数の派生を安全に作れる。

新しいトレーナーの初期設定は次のとおり。

| 設定 | 既定値 | 変更するメソッド |
|---|---|---|
| 観測ノイズ分散 | `1e-2`、範囲 $(10^{-5}, 10^{5})$ | `with_noise_variance`、`with_likelihood` |
| 入力・目的変数の変換 | 恒等変換 | `with_input_transform`、`with_target_transform` |
| ジッタポリシー | 固定、オフセット `0`（再試行なし） | `with_jitter` |
| 指数関数のバックエンド | `KernelExp::Accurate` | `with_kernel_exp` |
| ワークスペースのポリシー | `FitPolicy::prefer_speed()` | `with_prefer_speed`、`with_prefer_memory`、`with_distance_cache_policy`、`with_cholesky_buffer` |
| 最適化器 | `O{}` | `with_optimizer` |

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [defaults]
:end-before: // [end]
```

```{code-block} text
default jitter kind is fixed, kernel_exp is Accurate, policy is prefer_speed
```

:::{note}
対応するゲッターは `kernel_spec()`、`optimizer()`、`likelihood()`、`jitter()`、`kernel_exp()`、`fit_policy()` である。`FittedGpr` にも同じゲッターがある。`refit` と `into_trainer` のために設定を保持しているからだ。
:::

`Gpr` をコピーすると変換も深くコピーされる。ムーブは `noexcept` である。コピー代入とムーブ代入も使えるため、トレーナーをコンテナに格納したり、再代入したりできる。

## 観測ノイズを設定する

ノイズ分散 $\sigma_n^2$ はハイパーパラメータである。`optimizer::Fixed` を使わない限り、ここで与える値は最適化器が改善していく**初期値**になる。

`with_noise_variance` は引数を検証するため、`std::expected<Gpr, GprError>` を返す。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [noise]
:end-before: // [end]
```

```{code-block} text
noise variance = 0.0500 in (0.0001, 1.0000)
```

探索範囲を変えるには、`GaussianLikelihood` を作り、`GaussianLikelihood::with_bounds` で範囲を絞って、`with_likelihood` に渡す。尤度は検証済みなので、`with_likelihood` は失敗しない。$\log\sigma_n^2$ から作るなら `GaussianLikelihood::from_log_noise_variance` を使う。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [reject]
:end-before: // [end]
```

```{code-block} text
negative noise rejected: 1
```

:::{warning}
範囲は**開**区間である。既定は $(10^{-5}, 10^{5})$ なので、ノイズ分散がちょうど `1e-5` だと `GprErrorCode::InvalidInterval` で拒否される。`2e-5` を使うか、範囲を広げる。0 以下の分散は `GprErrorCode::InvalidNoiseVariance` になる。
:::

:::{tip}
ノイズとジッタは別物である。ノイズは統計的な量で、センサーの精度の限界を表す。ジッタは数値的な量で、コレスキー分解が失敗したときの救済に使う。データが実際にそれだけ雑音を含むのでない限り、数値的な失敗をノイズの増加で解決してはならない。
:::

## `fit` と `factor` を使い分ける

| | `fit` | `factor` |
|---|---|---|
| ハイパーパラメータの最適化 | する（`O` が `optimizer::Fixed` の場合を除く） | しない |
| 使われるハイパーパラメータ | 最適化器の結果 | トレーナーに設定されたものそのまま |
| ワークスペースのポリシー | トレーナーの `FitPolicy` | 常に `FitPolicy::prefer_memory()` |
| `optimizer_result().iterations` | 最適化器の反復回数 | `0` |
| 典型的な用途 | データからハイパーパラメータを学習する | 既知のハイパーパラメータでデータを条件付ける |

どちらにも 2 つのオーバーロードがある。`MatrixRef<double>` と `std::span<const double>` の目的変数を取るものと、列優先の生の `std::span<const double>` に `n_rows` と `n_cols` を添えるものである。入力は**列優先**である。$n$ 行 $d$ 列なら、要素 $(i, j)$ は添字 `j * n + i` にある。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [fit-factor]
:end-before: // [end]
```

```{code-block} text
factor: neg-MLL 4.7716, noise 0.0500, iterations 0
fit:    neg-MLL 0.7080, noise 0.0227, iterations 15, converged 1
```

`log_marginal_likelihood()` が返すのは**負の**対数周辺尤度である（小さいほどよい）。最適化により、値は 4.77 から 0.71 に下がり、標準化後の目的変数でノイズ分散は 0.023 になった。データと整合する。加えたノイズは分散 0.01 で、信号の分散は約 0.45 だから、0.01 / 0.45 ≈ 0.022 になる。

:::{note}
最適化が収束しなくても、**エラーにはならない**。学習済みモデルの `optimizer_result().converged` と `iterations` を確認すること。{doc}`choose-optimizer` を参照。
:::

## ジッタポリシーを選ぶ

$K + \sigma_n^2 I$ のコレスキー分解が失敗すると、トレーナーは対角に小さなオフセット（ジッタ）を加えて再試行する。方針は `JitterPolicy` で選ぶ。

- `JitterPolicy::fixed(j)` は、まずオフセット `0` で試す。失敗したら `j` で 1 回だけ再試行する。`fixed(0.0)` は再試行しない。既定のトレーナーは `j = 0` のこの方針を使う。
- `JitterPolicy::adaptive(initial, multiplier, max_retries, max_jitter)` は、`initial`、`initial * multiplier`、… の順に再試行する。分解が成功するか、`max_retries` を使い切るか、次のオフセットが `max_jitter` を超えるまで続ける。

どちらのファクトリも `std::expected<JitterPolicy, GprError>` を返し、非有限値や範囲外の引数は `GprErrorCode::InvalidConfig` で拒否する。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [jitter]
:end-before: // [end]
```

```{code-block} text
accepted factor jitter: 0.0000e+00
```

採用されたオフセットは `FittedGpr::factor_jitter()` で読める。ここでは 0 で、助けなしに分解が成功したことを示す。どのオフセットでも失敗した場合、呼び出しは `GprErrorCode::CholeskyFailed` を返す（{doc}`handle-errors` を参照）。

ポリシーの中身は、`kind()`（`JitterPolicy::Kind::Fixed` か `JitterPolicy::Kind::Adaptive`）、`is_fixed()`、`is_adaptive()`、`as_fixed()`、`as_adaptive()` で調べる。後ろの 2 つは `std::optional<FixedJitter>` と `std::optional<AdaptiveJitter>` を返す。保持している値は `FixedJitter::jitter()`、`AdaptiveJitter::initial()`、`multiplier()`、`max_retries()`、`max_jitter()` で読む。`retry_schedule()` は `JitterSchedule` を返す。その `next()` を呼ぶと、オフセットを順に 1 つずつ返し、尽きると `std::nullopt` を返す。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [jitter-schedule]
:end-before: // [end]
```

```{code-block} text
adaptive retry ladder: 1.0000e-10 1.0000e-09 1.0000e-08 1.0000e-07 1.0000e-06 1.0000e-05 1.0000e-04
max_retries 8, multiplier 10.0000, fixed retry 1.0000e-08, fixed has no adaptive part: 1
```

`max_retries` は 8 だが、段は 7 つで止まる。次の値 `1e-3` が `max_jitter = 1e-4` を超えるためである。

`FixedJitter`、`AdaptiveJitter`、`JitterSchedule` には公開コンストラクタがあるが、検証はしない。`JitterPolicy` のファクトリを使うこと。

事後サンプリングには別のジッタがあり、`FittedGpr::sample` に渡す（{doc}`predict-and-sample` を参照）。

## 精度ポリシーを選ぶ

精度ポリシー `P` は、カーネル行列とコレスキー因子の保存に使うスカラー型（`P::Storage`）と、重みと出力の型（`P::Refine`）を決める。

| ポリシー | `Storage` | `Refine` | `refines_in_f64` | 備考 |
|---|---|---|---|---|
| `DoublePrecision` | `double` | `double` | `false` | 既定。 |
| `SinglePrecision` | `float` | `float` | `false` | メモリは半分。結果は `float`。 |
| `MixedPrecision<PromoteStorage>` | `float` | `double` | `true` | `float` の因子と、反復改良で求めた `double` の重み。残差は保存済みの `float` 行列を昇格して計算する。 |
| `MixedPrecision<ReevaluateKernel>` | `float` | `double` | `true` | 上と同じだが、残差は保存済みの `float` を使わず、カーネル要素を `double` で再評価する。 |

`MixedPrecision` は `ResidualFormulaConcept` を満たす型を取る。タグ `PromoteStorage` か `ReevaluateKernel` で、既定は `PromoteStorage` である。`KernelScalarConcept` は `float` と `double` を受け付ける。`PrecisionPolicyConcept`（別名 `GpScalarConcept`）は上の 4 つのポリシーだけを受け付け、それ以外の型はコンパイル時に拒否される。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [precision]
:end-before: // [end]
```

```{code-block} text
single neg-MLL 0.7079, mixed neg-MLL 0.7079, refinement iterations 2
```

混合精度では、`FittedGpr::refinement_summary()` が `RefinementSummary`（`status`、`iterations`、`final_relative_residual`、`used_fallback()`）を保持する。ほかの 2 つでは `std::nullopt` になる。`FittedGpr::Storage` と `FittedGpr::Refine` は、ポリシーの型をそのまま公開する。

各ポリシーは、ディスク上の保存形式も持つ。`persist_kind()` は `PersistPrecisionKind`（`Double`、`Single`、`MixedPromote`、`MixedReevaluate`）を返す。`to_precision_string` と `to_residual_string` は、それを保存設定の文字列に変換する。タグ `PromoteStorage` と `ReevaluateKernel` には、`name()` と `reads_storage` がある。

生のバッファに精度ポリシーを適用する `inline` ヘルパーが 2 つある。`copy_storage_to_refine<P>` は `P::Storage` のスパンを `P::Refine` のスパンへ広げてコピーする。`compute_storage_means<P>` は $K_*^\top\alpha$ を `double` で累積して計算する。スパンが不正なら、どちらも `GprErrorCode::EmptyInput` か `GprErrorCode::LengthMismatch` を返す。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [precision-helpers]
:end-before: // [end]
```

```{code-block} text
mixed / promote_storage, reevaluate variant: reevaluate_kernel, tag name reevaluate_kernel
k^T alpha = -1.0000, -1.0000
```

:::{tip}
メモリや速度の要求がなければ `DoublePrecision` を使う。`MixedPrecision` は `float` の分解で倍精度に近い精度を得る。`SinglePrecision` は `DoublePrecision` なら分解できる行列で失敗することがあるため、適応ジッタと組み合わせる。精度ごとのモデルの保存は {doc}`persist-models` を参照。
:::

## 最適化器、再スタート、反復の上限を設定する

最適化器は `with_optimizer` で差し替える。組込みの最適化器は、それぞれ独自の `with_*` メソッドで設定する。たとえば `optimizer::Lbfgs::with_restarts(count, seed)` は、決定的なランダム再スタートを `count` 回加える。開始点は対数一様分布から引き、乱数は `seed` で初期化した `Xoshiro256PlusPlus` が生成する。最良の結果を採用する。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [optimizer]
:end-before: // [end]
```

```{code-block} text
with 3 restarts: neg-MLL 0.7080
```

再スタートは、悪い局所解を避けるためのものだ。すべての開始点の中で最良の結果が採用される。`optimizer::Fixed` は探索せず、トレーナーのハイパーパラメータで目的関数を 1 回評価するだけである。ほかの最適化器とそのオプションは {doc}`choose-optimizer` で説明する。

## 指数関数のバックエンドを選ぶ

`with_kernel_exp(KernelExp::FastApprox)` を使うと、組込みカーネル内の `std::exp` がベクトル化 Estrin minimax 近似（`1–2 ULP`）に置き換わり、カーネル評価が速くなる。既定の `KernelExp::Accurate` は `std::exp` を使う。この設定は、パラメータ変換と `CustomKernel` の葉には効かない。この設定を使う行は、次の例に含まれる。

## メモリと速度のトレードオフを決める

`FitPolicy` は、独立した 2 つの選択をまとめたものである。

| フィールド | 値 | 効果 |
|---|---|---|
| `distance_cache` | `DistanceCachePolicy::Cached` / `Uncached` | 最適化の各ステップ間で $O(n^2)$ の二乗距離を保持するか、座標から毎回再計算するか。 |
| `cholesky_buffer` | `CholeskyBuffer::Retain` / `Reuse` | 勾配の重み行列用に $n \times n$ のバッファを別に持つか、`K` を上書きして後で再分解するか。 |

プリセットは 2 つある。`FitPolicy::prefer_speed()` は `Cached` と `Retain` の組で、既定である。`FitPolicy::prefer_memory()` は `Uncached` と `Reuse` の組である。トレーナーには `with_prefer_speed`、`with_prefer_memory` と、フィールド単位の `with_distance_cache_policy`、`with_cholesky_buffer` がある。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [policy]
:end-before: // [end]
```

```{code-block} text
prefer_memory neg-MLL 0.7080, custom policy neg-MLL 0.7080
```

ポリシーが変えるのはメモリ使用量と実行時間で、結果は変わらない。2 つのモデルは、表示した桁の範囲で同じ最適値に達している。`FitPolicy` は `==` で比較できる。

## 前処理を加える

`with_input_transform` と `with_target_transform` は、渡された変換を複製する。同じ原型を使い回せる。変換については {doc}`transform-data` で説明する。学習済みの変換は `FittedGpr::input_transform()` と `FittedGpr::target_transform()` で、未学習の原型は `unfitted_input_transform()` と `unfitted_target_transform()` で取得する。

## 失敗の保証を利用する

`fit` と `factor` は `const` である。失敗しても状態は何も変わらない。トレーナーは呼び出し前のままで、そのまま再利用できる。

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:start-after: // [failure]
:end-before: // [end]
```

```{code-block} text
bad targets: input contains a non-finite value (NaN/Inf)
trainer still usable: 1, noise still 0.0500
```

成功すると、返される `FittedGpr` は独立したオブジェクトになる。学習データのコピー（`x_raw()`、`y_raw()`）、コレスキー因子（`l_factor()`）、重み（`alpha()`）、自身の設定を所有する。そのほかの参照メソッドは次のとおり。`n_train()`、`n_features()`、`kernel_spec()` と `likelihood()`（どちらも**学習後**のハイパーパラメータ）、`optimizer_result()`、`jitter()`、`kernel_exp()`、`fit_policy()`。`FittedGpr` をコピーすると、コピー側は独立した作業領域を持つ。ムーブは `noexcept` である。

エラーとその意味は {doc}`handle-errors` にある。

## プログラム全文

```{literalinclude} /../examples/configure_fit.cpp
:language: cpp
:linenos:
```

## API 一覧

| シンボル | 用途 |
|---|---|
| `Gpr`、`Gpr::Gpr` | 未学習のトレーナー。コンストラクタは `KernelSpec` と最適化器を取る。 |
| `with_optimizer`、`with_likelihood`、`with_noise_variance`、`with_jitter` | 最適化器、ノイズ、ジッタを変えたコピー。 |
| `with_input_transform`、`with_target_transform`、`with_kernel_exp` | 変換、指数関数のバックエンドを変えたコピー。 |
| `with_distance_cache_policy`、`with_cholesky_buffer`、`with_prefer_memory`、`with_prefer_speed` | ワークスペースのポリシーを変えたコピー。 |
| `kernel_spec`、`optimizer`、`likelihood`、`jitter`、`kernel_exp`、`fit_policy` | トレーナーのゲッター（`FittedGpr` にもある）。 |
| `fit`、`factor` | 最適化あり・なしの学習。`const` で、失敗しても状態は変わらない。 |
| `FittedGpr`、`Storage`、`Refine` | 学習済みモデルとそのスカラー型。 |
| `n_train`、`n_features`、`factor_jitter`、`log_marginal_likelihood`、`alpha`、`l_factor`、`x_raw`、`y_raw` | 学習済みモデルの参照メソッド。 |
| `input_transform`、`target_transform`、`unfitted_input_transform`、`unfitted_target_transform` | 学習済みの変換と未学習の原型。 |
| `refinement_summary`、`optimizer_result` | 反復改良と最適化の診断。 |
| `GaussianLikelihood`、`create`、`from_log_noise_variance`、`with_bounds`、`noise_variance`、`bounds` | 観測ノイズ（詳細は {doc}`tune-hyperparameters`）。 |
| `JitterPolicy`、`fixed`、`adaptive`、`Kind`、`kind`、`is_fixed`、`is_adaptive`、`as_fixed`、`as_adaptive`、`retry_schedule` | コレスキー分解の安定化方針。 |
| `FixedJitter`、`AdaptiveJitter`、`JitterSchedule`、`next`、`initial`、`multiplier`、`max_retries`、`max_jitter`、`jitter` | ジッタのパラメータ型とスケジュール型。 |
| `DoublePrecision`、`SinglePrecision`、`MixedPrecision`、`PromoteStorage`、`ReevaluateKernel` | 精度ポリシーと残差タグ。 |
| `PrecisionPolicyConcept`、`GpScalarConcept`、`KernelScalarConcept`、`ResidualFormulaConcept` | コンパイル時の制約。 |
| `refines_in_f64`、`reads_storage`、`name`、`persist_kind`、`PersistPrecisionKind`、`to_precision_string`、`to_residual_string` | ポリシーの性質と保存名。 |
| `copy_storage_to_refine`、`compute_storage_means` | 精度の変換と平均計算のヘルパー。 |
| `FitPolicy`、`prefer_speed`、`prefer_memory`、`DistanceCachePolicy`、`CholeskyBuffer` | ワークスペースのトレードオフ。 |
| `KernelExp` | `Accurate` か `FastApprox` の指数関数。 |
| `RefinementSummary` | 混合精度の反復改良の結果。 |

生成された API リファレンス: [`gpr`](../reference/api/gpr.md)、[`core`](../reference/api/core.md)、[`precision`](../reference/api/precision.md)、[`linalg`](../reference/api/linalg.md)。
