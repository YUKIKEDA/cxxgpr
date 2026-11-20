# 最適化器を選んで設定する

`Gpr` のトレーナーは、負の対数周辺尤度を最小化して、カーネルのハイパーパラメータとノイズ分散を決める。このガイドでは、5 種類の組込み最適化器から 1 つを選び、設定し、結果を読み、複数回の初期化（リスタート）を再現可能にする手順を示す。自分の目的関数に最適化器を直接使う方法と、自作の最適化器を差し込む方法も扱う。最適化器が 5 種類ある理由と、それぞれの仕組みは {doc}`../explanation/optimizers` を参照。

このページのプログラムは `optimizers.cpp`（最初の表以外のすべて）と `transforms.cpp` である。ここで使うクラスは `cxxgpr/optimizer/optimizer.hpp` を include すればすべて使える。

## 何を最適化しているかを知る

最適化器が最小化するのは、ベクトル $\theta$ の 1 つのスカラー関数である。$\theta$ は、すべてのハイパーパラメータの**自然対数**を並べたものだ。並び順は、`KernelSpec::get_params` の順にカーネルのパラメータ、最後にノイズ分散である。関数は、変換後のデータで評価した負の対数周辺尤度（NLML）である。各パラメータには、ユーザー単位の開区間がある。既定は `(1e-5, 1e5)` で、最適化器はこの区間の内側だけを動く。

実際には次の 3 点が重要になる。

- **出発点はあなたのカーネルである。** `RbfKernel::create(1.0)` や `with_noise_variance(1e-2)` に渡した値から探索が始まる。妥当なスケールから遠い出発点では、悪い局所解に落ちることがある（後述の警告を参照）。
- **範囲が到達できる領域を決める。** カーネルの `create` に `Interval` を渡せば、間違いとわかっている領域に長さスケールが入り込むのを防げる。
- **小さいほど良い。** `FittedGpr::log_marginal_likelihood()` が返すのは、*負の*対数周辺尤度の最終値である。`OptimizerResult::value` も同じ向きだ。

## 最適化器を選ぶ

| 最適化器 | 必要なもの | 手法 | 向く場面 | 1 反復のコスト |
|---|---|---|---|---|
| `Lbfgs` | 勾配 | 準ニュートン法と Moré–Thuente 直線探索 | 既定。パラメータが少数から中程度の滑らかな問題 | 値と勾配の評価 1 回（直線探索では数回） |
| `TrustRegion` | 勾配とヘッセ行列 | 信頼領域内のニュートン法 | パラメータが少なく反復を最小にしたい、または曲面が悪条件 | 値・勾配・ヘッセ行列の評価 1 回 |
| `NelderMead` | 値のみ | 単体探索 | 微分を使わない手軽な確認、または勾配が信用できない曲面 | 値の評価 1 回（反復あたり数回） |
| `FastSimulatedAnnealing` | 値のみ | 確率的な座標探索 | 凹凸が多く、局所法が捕まる曲面 | 値の評価 1 回（和のカーネルでは安価な増分更新） |
| `Fixed` | なし | 評価 1 回 | ハイパーパラメータがすでにわかっている。`fit` はカーネル行列の分解だけを行う | なし（反復 0 回） |

`Gpr` の目的関数は値・勾配・ヘッセ行列を提供するので、どの最適化器もモデルで使える。まず `Lbfgs` を試す。反復が多い、または進まないときは `TrustRegion` に替える。出発点によって答えが変わるときは、リスタート（後述）を加える。

## 同じ問題で比べる

次のプログラムは、同じノイズ入りの 1 次元データを、すべての最適化器で学習する。最後に自作の最適化器も加えている（このガイドの末尾を参照）。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [compare]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
truth at 0.35 = 1.0382
optimizer               iters  converged  neg-log-ML mean@0.35
Lbfgs                      17        yes    -14.5571    1.0564
TrustRegion                16        yes    -14.5571    1.0564
NelderMead                101        yes    -14.5571    1.0564
FastSimulatedAnnealing    400        yes    -14.5462    1.0561
Lbfgs + 3 restarts         17        yes    -14.5571    1.0564
Fixed                       0        yes    -10.4670    1.0605
RandomSearch (custom)     300        yes    -14.4880    1.0560
```

表の読み方は次のとおりである。

- 3 つの局所最適化器は、小数点以下 4 桁まで同じ最小値を見つけた。`TrustRegion` と `Lbfgs` は 16〜17 反復で済む。`NelderMead` は微分を使わないので、その 6 倍ほどかかる。
- `FastSimulatedAnnealing` は反復の予算（ここでは 400）を使い切り、最小値のそばには着くが、ぴたりとは一致しない。
- `Fixed` は最適化しない。初期のハイパーパラメータを 1 回評価するだけである。NLML は $-10.47$ で、最適化した $-14.56$ より悪い。ただし 0.35 での予測は、真の値に近いままである。
- NLML が小さいほど、最適化器がより良いハイパーパラメータを見つけたことになる。比べるのはこの列である。データが易しいので、予測はどれも近い。

:::{warning}
出発点は結果を左右する。上のプログラムは長さスケールを $0.3$ から始めている。同じデータで $1.0$ から始めると、素の `Lbfgs` は 1 反復で止まる。そのときの NLML は $210.7$ で、最小値よりはるかに悪い。0.35 での予測も、$1.04$ ではなく $0.25$ になる。それでも最適化器は `converged = yes` と報告する。モデルが良いからではなく、最適化器自身の停止判定が働いたからである。結果が怪しいときは、複数の実行で `log_marginal_likelihood()` を比べる。`with_restarts` を加える、`TrustRegion` を試す、という対処も有効である。
:::

## 最適化器を設定する

最適化器はどれも小さな値型である。`with_*` メソッドは変更を加えたコピーを返し、元のオブジェクトは変わらない。いずれも失敗しない。検証は `optimize`（したがって `fit`）の中で行われ、不正な設定は `GprErrorCode::InvalidConfig` になる。

### `Lbfgs`

| 設定 | メソッド | アクセサ | 既定値 | 有効な値 |
|---|---|---|---|---|
| 1 回の出発あたりの反復数 | `with_max_iterations` | `max_iterations` | 100 | 任意 |
| 収束判定の許容値 | `with_tolerance` | `tolerance` | $\sqrt{\varepsilon}\approx 1.5\times 10^{-8}$ | 有限、$\ge 0$ |
| 保持する曲率ペアの数 | `with_history_size` | `history_size` | 10 | $\ge 1$ |
| リスタート | `with_restarts` | `restarts` | なし | — |

### `TrustRegion`

| 設定 | メソッド | アクセサ | 既定値 | 有効な値 |
|---|---|---|---|---|
| 1 回の出発あたりの反復数 | `with_max_iterations` | `max_iterations` | 100 | 任意 |
| 収束判定の許容値 | `with_tolerance` | `tolerance` | $\sqrt{\varepsilon}$ | 有限、$\ge 0$ |
| 初期半径 | `with_initial_radius` | `initial_radius` | 1.0 | 有限、$> 0$、最大半径以下 |
| 最大半径 | `with_max_radius` | `max_radius` | 100.0 | 有限、初期半径以上 |
| リスタート | `with_restarts` | `restarts` | なし | — |

半径は、{doc}`../explanation/optimizers` で説明する非拘束座標で測る。既定値を変える必要はまずない。

### `NelderMead`

| 設定 | メソッド | アクセサ | 既定値 | 有効な値 |
|---|---|---|---|---|
| 1 回の出発あたりの反復数 | `with_max_iterations` | `max_iterations` | 100 | 任意 |
| 収束判定の許容値（関数値の広がりと単体の大きさ） | `with_tolerance` | `tolerance` | $\sqrt{\varepsilon}$ | 有限、$\ge 0$ |
| リスタート | `with_restarts` | `restarts` | なし | — |

### `FastSimulatedAnnealing`

| 設定 | メソッド | アクセサ | 既定値 | 有効な値 |
|---|---|---|---|---|
| 1 回の出発あたりの反復数（冷却の長さでもある） | `with_max_iterations` | `max_iterations` | 100 | 任意 |
| 初期温度 $T_0$ | `with_initial_temperature` | `initial_temperature` | 1.0 | 有限、$> 0$ |
| 冷却率 $c$ | `with_cooling_rate` | `cooling_rate` | 3.0 | 有限、$\ge 0$ |
| 乱数シード | `with_seed` | `seed` | 0 | 任意 |
| 範囲外の提案の扱い | `with_boundary_policy` | `boundary_policy` | `BoundaryPolicy::Clamp` | `Clamp` か `Reflect` |
| リスタート | `with_restarts` | `restarts` | なし | — |

`BoundaryPolicy::Clamp` は、範囲外の提案を区間の内側で最も近い点に置く。`BoundaryPolicy::Reflect` は、境界で折り返す。反復数は冷却スケジュールの長さでもあるので、反復を増やすと、末尾が長くなるだけでなく、冷却が全体にゆっくりになる。`FastSimulatedAnnealing` だけは独自の `with_seed` を持ち、リスタートでは実行番号がこのシードに足される。

### `Fixed`

`Fixed` に設定項目はない。ハイパーパラメータがわかっているときは、最適化器の型として `Gpr<optimizer::Fixed>` を使う。あるいは、任意のトレーナーで `Gpr::factor` を呼べば、最適化せずに現在のハイパーパラメータで分解できる。

### 許容値と反復の予算

反復の上限は、1 回の出発ごとに適用される。許容値を満たす前に上限で止まると、`OptimizerResult::converged` は `false` になるが、見つかった最良の点は返され、`fit` も成功する。`GprErrorCode::OptimizationNotConverged` では失敗しない。気になるときは `FittedGpr::optimizer_result()` を確認する。

```cpp
const auto& r = model->optimizer_result();   // OptimizerResult
// r.params: best log-space parameters, r.value: NLML there,
// r.iterations: iterations of the best run, r.converged: stopping test met
```

## リスタートを再現可能にする

局所最適化器は、出発点によって別の局所解に着くことがある。`with_restarts(count, seed)` は、初期点と、範囲内から（対数空間で）一様に引いた `count` 個の追加の点から最適化器を走らせ、最良の結果を返す。追加の点は、シードを与えた `Xoshiro256PlusPlus` 乱数生成器から作るので、同じ `(count, seed)` なら必ず同じ列になる。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [restarts]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
restarts with the same seed give bit-identical results: yes
```

`RestartConfig` は、この組（`count` と `seed`）を保持する単純な構造体である。`restarts()` がそれを返し、`operator==` も持つ。リスタートの費用は `count + 1` 倍になるので、`count` は小さくする。シードを変えれば、再現可能な別の追加点の集合が得られる。

## 自分の目的関数に最適化器を使う

最適化器はガウス過程に縛られない。目的関数のインターフェースを実装した関数なら、何でも最小化できる。値だけを使う最適化器には `Objective`、勾配を与えられる（`Lbfgs`）なら `Differentiable`、ヘッセ行列も与えられる（`TrustRegion`）なら `TwiceDifferentiable` から派生する。

| クラス | 実装するもの | 受け付ける最適化器 |
|---|---|---|
| `Objective` | `dim`、`value`、`fill_intervals`。任意で `value_at_changes` | `NelderMead`、`FastSimulatedAnnealing`、`Fixed` |
| `Differentiable` | 加えて `gradient_into`、`value_and_gradient_into` | 上記と `Lbfgs` |
| `TwiceDifferentiable` | 加えて `hessian_into`。任意で `value_gradient_hessian_into` | 上記すべてと `TrustRegion` |

受け取るパラメータは対数である。`fill_intervals` は各パラメータの*ユーザー単位*の開区間を返す。最適化器がその対数を取るので、`lo` はすべて正でなければならない。次は、解析的な導関数を持つ 2 次の目的関数の完全な実装である。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [objective]
:end-before: // [end]
:dedent: 4
```

ヘッセ行列は、`dim * dim` の 1 次元配列に行優先で格納する。$(i, j)$ 成分は `i * dim + j` に置く。`value_at_changes` の既定は `value` の呼び出しである。上書きするのは、`FastSimulatedAnnealing` が行う 1 座標ずつの更新を活かしたいときだけでよい。`value_gradient_hessian_into` の既定は、`value_and_gradient_into` と `hessian_into` を続けて呼ぶ実装である。3 つの計算で作業を共有したいときに上書きする。

`optimize(objective, initial_params)` で任意の最適化器を実行する。`initial_params` は、対数空間の範囲の厳密に内側になければならない。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [direct]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
Lbfgs: converged = true, iterations = 11, value = 8.70e-20
  parameters = 2.0000 0.5000 10.0000 
TrustRegion value = 2.82e-15 (7 iterations)
NelderMead  value = 3.62e-18 (128 iterations)
FSA         value = 1.60e-03 (3000 iterations)
Fixed       value = 1.86e+00 (0 iterations)
```

この滑らかな問題では、微分を使う最適化器は 7〜11 反復で厳密な最小値に届く。`NelderMead` は 128 反復、`FastSimulatedAnnealing` は 3000 反復で近くまで来る。これは桁の目安であり、ベンチマークではない。最適な選択は問題によって変わる。

`OptimizerResult` は `params`（対数空間で最良のベクトル）、`value`、`iterations`、`converged` を持つ。`Fixed::optimize` は `initial_params` をそのまま返し、`iterations = 0`、`converged = true` とする。

### `optimize` が返すエラー

| コード | 原因 |
|---|---|
| `GprErrorCode::InvalidConfig` | 許容値が負または有限でない、`history_size == 0`、半径が不正、温度が正でない、冷却率が負 |
| `GprErrorCode::EmptyInput` | `obj.dim() == 0`、または `initial_params` が空 |
| `GprErrorCode::LengthMismatch` | `initial_params.size() != obj.dim()` |
| `GprErrorCode::NonFiniteInput` | 出発点、値、勾配、ヘッセ行列が有限でない |
| `GprErrorCode::InvalidInterval` | 区間が `0 < lo < hi` を満たさない、または出発点が区間の外にある |

自作の `value` や導関数が返したエラーは、そのまま伝わる。

## 境界を直接扱う

`LogBounds` は、最適化器が共有する座標変換である。開区間を対数空間に直し、区間に拘束された対数パラメータ $\theta$ と、最適化器が動く非拘束ベクトル $z$ を相互に写す。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [log-bounds]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
z = (0.6985, -0.6985, -0.0000); round trip error = 0.0000
```

区間の中点は $z = 0$ に写る。上の出力の最後の成分が 0 なのはそのためである（出発点 $\log 1 = 0$ は、$(\log 0.01, \log 100)$ の中点にある）。`LogBounds` は `LogBounds::from_objective(obj)` か `LogBounds::from_intervals(intervals)` で作る。失敗すると `GprErrorCode::EmptyInput` か `GprErrorCode::InvalidInterval` になる。メンバーは次のとおりである。

| メンバー | 役割 |
|---|---|
| `dim` | パラメータの数 |
| `lo_log`、`hi_log` | 対数空間の下限と上限 |
| `validate_log_params` | $\theta$ が有限で、境界の厳密に内側にあることを確かめる |
| `to_unconstrained` | $\theta \to z$（$\theta$ が境界の外なら失敗する） |
| `to_constrained` | $z \to \theta$（失敗しない。結果は必ず厳密に内側に収まる） |
| `to_constrained_with_jacobian` | $d\theta/dz$ も返す |
| `to_constrained_with_derivatives` | $d^2\theta/dz^2$ も返す |
| `chain_gradient`、`chain_hessian` | 勾配やヘッセ行列を $\theta$ から $z$ へ変換する |
| `sample_log_uniform` | 境界の厳密に内側から、決定的な乱数点を引く |

`EvalCache` は直近の評価を覚えておき、直線探索が同じ点に 2 度コストを払わないようにする。目的関数、境界、点 $z$、作業用バッファを渡す。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [eval-cache]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
two evaluations at the same point cost 1 objective call(s)
```

`EvalCache::eval_unconstrained` は、点・値・勾配のどれかが有限でなければ `GprErrorCode::NonFiniteInput` を返す。保持している点を捨てるには `clear` を呼ぶ。`EvalCache` が必要になるのは、同じ点を繰り返し評価する最適化器を自作するときだけである。

## 最適化器を自作する

`Gpr<O>` は、次を備えたコピー可能な任意の型 `O` を受け付ける。

- `static constexpr bool uses_change_indices`（`OptimizerTraits<O>::uses_change_indices` 経由で読まれる）。`true` にするのは、変更した座標の添字を付けて `value_at_changes` を呼ぶ最適化器だけである。そうするとモデルは、キャッシュしたカーネル行列を再利用できる。
- `std::expected<OptimizerResult, GprError> optimize(TwiceDifferentiable& obj, std::span<const double> initial_params) const`

モデルは、組込みの最適化器と同じ目的関数を渡す。次の例は、最良点を揺らして改善だけを採る最適化器で、`LogBounds`、`Xoshiro256PlusPlus`、`OptimizerResult` を使っている。

```{literalinclude} /../examples/optimizers.cpp
:language: cpp
:start-after: // [custom-optimizer]
:end-before: // [end]
:dedent: 4
```

先の比較表では、NLML が $-14.49$ になり、勾配を使う最適化器の $-14.56$ に近い値に達している。組込みの最適化器と同じように `Gpr<RandomSearch>(kernel, RandomSearch{})` で使う。

:::{note}
モデルは自作の最適化器にも `TwiceDifferentiable` の目的関数を渡す。必要なら `gradient_into` や `hessian_into` を呼んでよい。モデルが値のみとして扱うのは `NelderMead` と `FastSimulatedAnnealing` だけである。自作の最適化器に、それ以外の特別扱いはない。
:::

最適化器が提案する点は、`LogBounds::validate_log_params` で確かめるか、`sample_log_uniform` で引き、開区間の厳密に内側に保つ。外側の点は、目的関数が `GprErrorCode::InvalidInterval` などのエラーで拒否する。

## API 一覧

| シンボル | 役割 |
|---|---|
| `Lbfgs` | limited-memory BFGS。`with_max_iterations`、`with_tolerance`、`with_history_size`、`with_restarts`、対応するアクセサ、`optimize`、`operator==` |
| `TrustRegion` | 信頼領域ニュートン法。`with_max_iterations`、`with_tolerance`、`with_initial_radius`、`with_max_radius`、`with_restarts`、アクセサ、`optimize`、`operator==` |
| `NelderMead` | 単体探索。`with_max_iterations`、`with_tolerance`、`with_restarts`、アクセサ、`optimize`、`operator==` |
| `FastSimulatedAnnealing` | 焼きなまし。`with_max_iterations`、`with_initial_temperature`、`with_cooling_rate`、`with_seed`、`with_boundary_policy`、`with_restarts`、アクセサ、`optimize`、`operator==` |
| `BoundaryPolicy` | 焼きなましの範囲外の提案を扱う `Clamp` と `Reflect` |
| `Fixed` | 1 回だけ評価する。`optimize`、`operator==` |
| `Objective`、`Differentiable`、`TwiceDifferentiable` | 目的関数のインターフェース。`dim`、`value`、`value_at_changes`、`fill_intervals`、`gradient_into`、`value_and_gradient_into`、`hessian_into`、`value_gradient_hessian_into` |
| `OptimizerResult` | `params`、`value`、`iterations`、`converged`、`operator==` |
| `RestartConfig` | `count`、`seed`、`operator==` |
| `OptimizerTraits` | 最適化器の型の `uses_change_indices` |
| `LogBounds` | 対数空間の境界と $\theta \leftrightarrow z$ の変換（メンバーは上の表） |
| `EvalCache` | 1 エントリのキャッシュ。`eval_unconstrained`、`clear` |

生成リファレンスは [optimizer API リファレンス](../reference/api/optimizer.md) にある。
