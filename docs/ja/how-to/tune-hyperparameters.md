# ハイパーパラメータを調整する

このガイドは、カーネルとノイズ尤度のハイパーパラメータを読み書きし、範囲を設定し、学習済みモデルから新しい学習をウォームスタートし、周辺尤度の目的関数（値、勾配、ヘッセ行列）を自分で評価する方法を示す。最適化器の選び方は {doc}`choose-optimizer` を参照。

スニペットは、ページ末尾に全文を載せた 1 つのプログラムから取っている。`using namespace cxxgpr;` を前提とする。

## パラメータベクトルを理解する

ハイパーパラメータには 2 通りの表現がある。

- **ユーザー単位**: 書き下したとおりの値（長さスケール、定数、ノイズ分散）で、常に正である。範囲はここで、開区間として与える。
- **対数空間**: $\theta = \log(\text{値})$。すべての `get_params` と `set_params`、勾配、ヘッセ行列はこちらを使う。最適化器は、これにより制約のないスケールで動く。

最適化器が見るベクトルは $p$ 個の成分を持つ。カーネルのパラメータが、葉の深さ優先・左から右の順に並び、最後にノイズ 1 個 $\log\sigma_n^2$ が付く。勾配とヘッセ行列は、この対数空間のベクトルに関するものである。

## パラメータの範囲を設定する

`Interval` は、検証済みの**開**区間 $(lo, hi)$ で、端点は有限、`lo < hi` を満たす。`Interval::create(lo, hi)`（任意の有限な端点）か、`Interval::positive(lo, hi)`（`0 < lo` が必要）で作る。`Interval::default_positive()` は、範囲を指定しないときに使う標準の $(10^{-5}, 10^{5})$ である。どのファクトリも `std::expected<Interval, GprError>` を返す。端点が不正なら `GprErrorCode::InvalidInterval` になる。区間は `lo()`、`hi()`、`width()`、`contains(value)` で調べる。区間同士は `==` で比較できる。

カーネルの葉は、作成時に区間を受け取る。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [bounds]
:end-before: // [end]
```

これで、最適化器は長さスケールを $(0.1, 10)$ の外に出せない。初期値は、区間の厳密な内側になければならない。

`BoundedParam` は、その土台となる型である。値はユーザー単位で、`Interval` の厳密な内側にあることが保証される。葉と `GaussianLikelihood` が内部で使う。自分のパラメータにも使える。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [bounded-param]
:end-before: // [end]
```

```{code-block} text
ln(2) = 0.6931, with_value(4).value() = 4.0000, with_value(9) fails: 1, interval width 7.5000
```

| 関数 | 結果 |
|---|---|
| `BoundedParam::create(value, interval)` | 検証済みのパラメータ。失敗時は `GprErrorCode::InvalidInterval` |
| `BoundedParam::default_positive(value)` | 同上。区間は $(10^{-5}, 10^{5})$ |
| `value()`、`interval()` | 値（ユーザー単位）と範囲を読む |
| `ln()` | $\log(\text{値})$。最適化器の座標 |
| `with_value(v)`、`with_interval(i)` | 新しいパラメータ。結果が範囲外ならエラー |

どの関数も、元のオブジェクトを変更しない。

## カーネルのハイパーパラメータを読む

`KernelSpec` は値型である。`num_params()` は対数空間のパラメータ数を返す。`get_params` は、ちょうどその長さのスパンを埋める。`bounds_into` は、対応する `Interval` を埋める。`parameter_bindings` は、通し番号を葉と葉内のスロットに対応づける（`ParameterBinding`）。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [params]
:end-before: // [end]
```

```{code-block} text
kernel parameters (log): 0.0000, 0.6931  -> constant 1.0000, lengthscale 2.0000
lengthscale bounds: (0.1000, 10.0000)
parameter 0 is leaf 0, local slot 0
parameter 1 is leaf 1, local slot 0
```

長さが合わなければ `GprErrorCode::LengthMismatch` になる。対数空間の値をユーザー単位に戻すには `std::exp` を使う。

## ハイパーパラメータを設定する

`set_params` は、対数空間のスパンからすべてのパラメータを置き換える。操作は**アトミック**である。1 つでも非有限値や範囲外の値があれば失敗し、仕様は元のまま残る。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [set-params]
:end-before: // [end]
```

```{code-block} text
out-of-bounds update rejected: 1, spec unchanged: 1
```

失敗の種類は、`GprErrorCode::LengthMismatch`、`GprErrorCode::NonFiniteInput`、`GprErrorCode::InvalidInterval` である。`KernelSpec` は値セマンティクスなので、古い値が必要なら先にコピーしておく。葉の型（`RbfKernel` など）にも、自身のパラメータ用に同じ `get_params` と `set_params` がある。{doc}`build-kernels` を参照。

## ノイズのハイパーパラメータを設定する

ノイズは 1 パラメータの尤度 `GaussianLikelihood` で、$\theta = \log\sigma_n^2$ で表す（`num_params()` は常に 1）。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [likelihood]
:end-before: // [end]
```

```{code-block} text
noise variance 0.0400 = exp(-3.2189), num_params 1, default bounds (1.0000e-05, 1.0000e+05)
diagonal after adding noise: 1.0400, gradient diagonal 0.0400
```

| 関数 | 用途 |
|---|---|
| `create(variance)`、`from_log_noise_variance(theta)` | ユーザー単位または対数単位から作る |
| `noise_variance()`、`log_noise_variance()`、`bounds()` | 読み出し |
| `with_bounds(interval)` | 範囲を変えた新しい尤度。現在の値が範囲内に残る必要がある |
| `get_params(out)`、`set_params(params)`、`bounds_into(out)` | 長さ 1 の対数空間インターフェース（`set_params` はアトミック） |
| `add_noise_diag(k_diag)` | 対角の各要素に $\sigma_n^2$ を足す |
| `noise_grad_diag(dk_diag, idx)` | $\partial K / \partial\theta$ の対角（$\sigma_n^2$）を書く。`idx` は 0 のみ（それ以外は `GprErrorCode::IndexOutOfRange`） |

`create` は、非正または非有限の入力に `GprErrorCode::InvalidNoiseVariance` を返す。値が範囲外なら `GprErrorCode::InvalidInterval` を返す。

## 学習されたハイパーパラメータを読む

`fit` の後、モデルは最適化された値を持つ。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
```

```{code-block} text
learned constant 1.1308, lengthscale 1.8844, noise 0.0134, neg-MLL -8.8970
```

`FittedGpr::kernel_spec()` と `likelihood()` は、学習後の `KernelSpec` と `GaussianLikelihood` を返す。`neg_log_marginal_likelihood()` は、最終的な**負の**対数周辺尤度を返す。小さいほど当てはまりがよい。ここでの値（定数 1.13、長さスケール 1.88、ノイズ分散 0.013）は、データの生成条件（振幅 1、ノイズ分散 0.01）に近い。

:::{note}
値は、最適化器が最小化する量にあたる。対数周辺尤度そのものが欲しいときは、符号を反転する。
:::

## 新しいデータで再調整する

`refit` は、新しいデータで新しいモデルを学習する。探索はこのモデルのハイパーパラメータから始まり、設定も引き継ぐ。`into_trainer` は、同じウォームスタート済みの設定を、未学習の `Gpr` として返す。`fit` の前に設定をさらに変えたいときに使う。どちらも元のモデルを変更しない。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [refit]
:end-before: // [end]
```

```{code-block} text
refit: 12 iterations, cold fit: 13 iterations
```

ウォームスタートは、コールドスタートと同じ最適値に達する（プログラムが、2 つの負の尤度が 0.05 以内で一致することを検査している）。この小さな問題では、コールドスタートも安いので反復数は近い。目的関数の評価が高く、新しいデータが最適値をわずかしか動かさないときほど、節約が大きくなる。`refit` には、`fit` と同じ `MatrixRef` 版と生スパン版があり、エラーも同じである。

探索せずに、**指定した**ハイパーパラメータでモデルを評価したいときは、`optimizer::Fixed` のトレーナーを作る（または `factor` を呼ぶ）。カーネルと尤度は、使いたいものを渡す。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [into-trainer]
:end-before: // [end]
```

```{code-block} text
factor at learned values reproduces neg-MLL: -8.8970 vs -8.8970
```

## 目的関数、その勾配、ヘッセ行列を評価する

`fit` の間、最適化器は `optimizer::TwiceDifferentiable` の目的関数を受け取る。対数空間のベクトルを引数とする負の対数周辺尤度である。最適化器の型を自作すれば、この目的関数に触れられる。要件は少ない。

- `static constexpr bool uses_change_indices`（`false` にする）。
- `optimize(optimizer::TwiceDifferentiable&, std::span<const double>) const`。戻り値は `std::expected<optimizer::OptimizerResult, GprError>`。

次の例は、開始点で解析的な勾配とヘッセ行列を評価し、有限差分と比較する。

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [probe]
:end-before: // [end]
```

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:start-after: // [objective]
:end-before: // [end]
```

```{code-block} text
neg-MLL at the start -1.5600, max gradient error 3.6651e-09, max Hessian error 9.3551e-09
```

目的関数のインターフェースは次のとおり。

| メソッド | 意味 |
|---|---|
| `dim()` | $p$。対数空間のパラメータ数（カーネルのパラメータ数 + 1） |
| `value(params)` | 負の対数周辺尤度 |
| `value_and_gradient_into(params, grad)` | 1 回の分解で値と勾配を求める |
| `hessian_into(params, hess)` | $p \times p$ のヘッセ行列。要素 $(i, j)$ は `hess[i * p + j]` に平坦化される |
| `value_gradient_hessian_into(params, grad, hess)` | 1 つのコレスキー因子で 3 つすべてを求める |

勾配とヘッセ行列は、$\theta$ に関する厳密な解析的導関数であり、有限差分ではない。例では、約 $10^{-8}$ の精度で一致している。どのメソッドも `std::expected` を返し、`GprErrorCode::CholeskyFailed` などのエラーを伝える。

`Gpr` を使わずに同じ計算を行うには、ワークスペース関数の `mll_and_gradient_into`、`hessian_into`、`value_gradient_hessian_into` を使う。{doc}`use-low-level-building-blocks` を参照。

## プログラム全文

```{literalinclude} /../examples/tune_hyperparameters.cpp
:language: cpp
:linenos:
```

## API 一覧

| シンボル | 用途 |
|---|---|
| `Interval`、`create`、`positive`、`default_positive`、`lo`、`hi`、`width`、`contains` | ユーザー単位の検証済み開区間。 |
| `BoundedParam`、`create`、`default_positive`、`value`、`interval`、`ln`、`with_value`、`with_interval` | 区間内にあることが保証されたパラメータ。 |
| `KernelSpec::num_params`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings` | カーネルのハイパーパラメータ（対数空間）。 |
| `ParameterBinding` | 通し番号、葉 ID、葉内のスロット。 |
| `GaussianLikelihood`、`create`、`from_log_noise_variance`、`noise_variance`、`log_noise_variance`、`bounds`、`with_bounds`、`num_params`、`get_params`、`set_params`、`bounds_into`、`add_noise_diag`、`noise_grad_diag` | ノイズ尤度。 |
| `FittedGpr::kernel_spec`、`likelihood`、`neg_log_marginal_likelihood`、`optimizer_result` | 学習された値と、最適点での目的関数。 |
| `refit`、`into_trainer` | 学習済みモデルからのウォームスタート。 |
| `optimizer::TwiceDifferentiable`、`dim`、`value`、`value_and_gradient_into`、`hessian_into`、`value_gradient_hessian_into` | 最適化器から見える目的関数。 |
| `optimizer::OptimizerResult` | パラメータ、値、反復回数、収束フラグ。 |

生成された API リファレンス: [`core`](../reference/api/core.md)、[`gpr`](../reference/api/gpr.md)、[`optimizer`](../reference/api/optimizer.md)、[`kernel`](../reference/api/kernel.md)。
