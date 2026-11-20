# カーネルを組み立てて評価する

組込みの 10 種のカーネル葉を作り、`+` と `*` で合成し、コンパイルして、グラム行列・交差共分散・微分を評価する手順を示す。カーネルの意味と選び方は {doc}`../explanation/kernels` に、独自の葉の書き方は {doc}`write-custom-kernel` にある。

ここで扱うシンボルはすべて `cxxgpr/kernel/kernel.hpp` から使える。生成された仕様は [kernel API リファレンス](../reference/api/kernel.md) を参照。

## 葉を選ぶ

| 目的 | 葉 | 備考 |
|---|---|---|
| 滑らかな関数、長さスケールは 1 つ | `RbfKernel` | 無限回微分可能。最初の候補にしやすい |
| 滑らかな関数、特徴量ごとに長さスケール | `RbfArdKernel` | どの特徴量が効くかを学習する |
| より粗い関数 | `MaternKernel`（`MaternNu::OneHalf` または `ThreeHalves`） | `FiveHalves` は広く使われる既定値 |
| 粗い関数、特徴量ごとの長さスケール | `MaternArdKernel` | |
| 繰り返しのあるパターン | `PeriodicKernel` | 周期がずれていくなら減衰カーネルを掛ける |
| 複数のスケールが混在する | `RationalQuadraticKernel`、`RationalQuadraticArdKernel` | `RbfKernel` のスケール混合 |
| 出力のスケール（信号分散） | `ConstantKernel` | 因子として使う: `ConstantKernel * RbfKernel` |
| 線形トレンド | `LinearKernel` | 定常カーネルに足す |
| カーネルの内部に独立な観測ノイズを持たせる | `WhiteKernel` | `Gpr` は既にノイズを持つ。下の警告を参照 |

## 葉を作る

どの葉にも静的関数 `create` があり、引数を検証して `std::expected<Leaf, GprError>` を返す。次の断片は、エラーで止まる補助関数 `unwrap`（完全な例に含まれる）を使っている。自分のコードでは `if (!result)` で結果を確かめる。

```{literalinclude} /../examples/kernels_tour.cpp
:language: cpp
:start-after: // [create-leaves]
:end-before: // [end]
:dedent: 4
```

ARD の 2 つの葉は、特徴量ごとの長さスケールを `std::span<const double>` で受け取る。`double` の `std::array` や `std::vector` をそのまま渡せる。`MaternKernel` と `MaternArdKernel` は `MaternNu`（`OneHalf`、`ThreeHalves`、`FiveHalves`）を取る。滑らかさの次数は構築時に固定され、最適化の対象にはならない。

### 数式とパラメータ

$r = \lVert x - x' \rVert$ とし、ARD の葉では $s^2 = \sum_k \bigl((x_k - x'_k)/\ell_k\bigr)^2$ とおく。

| 葉 | $k(x, x')$ | ハイパーパラメータ（順序どおり） |
|---|---|---|
| `RbfKernel` | $\exp\!\bigl(-r^2 / 2\ell^2\bigr)$ | $\ell$ |
| `RbfArdKernel` | $\exp\!\bigl(-s^2 / 2\bigr)$ | $\ell_1, \dots, \ell_d$ |
| `MaternKernel`、`OneHalf` | $\exp(-r/\ell)$ | $\ell$ |
| `MaternKernel`、`ThreeHalves` | $\bigl(1 + \sqrt3\, r/\ell\bigr)\exp\!\bigl(-\sqrt3\, r/\ell\bigr)$ | $\ell$ |
| `MaternKernel`、`FiveHalves` | $\bigl(1 + u + u^2/3\bigr)\exp(-u)$、$u = \sqrt5\, r/\ell$ | $\ell$ |
| `MaternArdKernel` | Matern の式の $r/\ell$ を $s$ に置き換えたもの | $\ell_1, \dots, \ell_d$ |
| `PeriodicKernel` | $\exp\!\bigl(-2\sin^2(\pi r / p)/\ell^2\bigr)$ | $\ell$、$p$（周期） |
| `RationalQuadraticKernel` | $\bigl(1 + r^2/(2\alpha\ell^2)\bigr)^{-\alpha}$ | $\ell$、$\alpha$ |
| `RationalQuadraticArdKernel` | $\bigl(1 + s^2/(2\alpha)\bigr)^{-\alpha}$ | $\ell_1, \dots, \ell_d$、$\alpha$ |
| `ConstantKernel` | $c$ | $c$ |
| `LinearKernel` | $v\,(x \cdot x')$ | $v$ |
| `WhiteKernel` | 学習用グラム行列では $v\,\delta_{ij}$、それ以外は 0 | $v$ |

例では、各値を標準ライブラリの数式でもう一度計算して比べている。

```{code-block} text
kernel                          k(p0,p0)    k(p0,p1)    expected
RbfKernel                       1.000000    0.329193    0.329193   ok
RbfArdKernel                    1.000000    0.367879    0.367879   ok
MaternKernel (1/2)              1.000000    0.225212    0.225212   ok
MaternKernel (3/2)              1.000000    0.270882    0.270882   ok
MaternKernel (5/2)              1.000000    0.286713    0.286713   ok
MaternArdKernel (5/2)           1.000000    0.317283    0.317283   ok
PeriodicKernel                  1.000000    0.200271    0.200271   ok
RationalQuadraticKernel         1.000000    0.413265    0.413265   ok
RationalQuadraticArdKernel      1.000000    0.444444    0.444444   ok
ConstantKernel                  2.000000    2.000000    2.000000   ok
LinearKernel                    0.625000    1.875000    1.875000   ok
WhiteKernel                     0.100000    0.000000    0.000000   ok
```

:::{warning}
`WhiteKernel` は、**正方の学習用**グラム行列の対角に分散を足す。「同じ点」かどうかは座標ではなく行番号で決まり、2 つの点集合の間の交差共分散には何も寄与しない。`Gpr` は別に観測ノイズ分散を足すため、`WhiteKernel` を重ねるとノイズ項が 2 つになって互いに競合する。`WhiteKernel` が役に立つのは、カーネルを自分で評価する場合か、ノイズ項をカーネルのパラメータベクトルに含めたい場合に限る。
:::

### パラメータ・境界・アクセサ

ハイパーパラメータはどれも厳密に正で、保存も受け渡しも**対数**で行う。どの葉も同じ小さなインターフェースを持つ。

| メンバ | 役割 |
|---|---|
| `num_params()` | ハイパーパラメータの数。ほとんどの葉は 1、`PeriodicKernel` と `RationalQuadraticKernel` は 2、ARD の RBF と Matern は次元数、`RationalQuadraticArdKernel` は次元数 + 1 |
| `get_params(out)` | 長さ `num_params()` のスパンに $\log\theta$ を書く |
| `set_params(params)` | $\log\theta$ から値を置き換える。原子的で、エラー時は何も変わらない |
| `bounds_into(out)` | 各パラメータの開区間（ユーザー単位）を書く |
| `operator==` | 値と区間を比べる |

型付きのアクセサもある。`RbfKernel`・`MaternKernel` は `lengthscale()` と `interval()`、`RbfArdKernel`・`MaternArdKernel` は `lengthscales()` と `intervals()`、2 つの Matern は `nu()`、`PeriodicKernel` は `lengthscale()`・`period()`・`lengthscale_interval()`・`period_interval()`、`RationalQuadraticKernel` は `lengthscale()`・`alpha()`・`lengthscale_interval()`・`alpha_interval()`、`RationalQuadraticArdKernel` は `lengthscales()`・`alpha()`・`lengthscale_intervals()`・`alpha_interval()`、`ConstantKernel` は `constant()` と `interval()`、`LinearKernel` と `WhiteKernel` は `variance()` と `interval()` を持つ。

既定の区間は `Interval::default_positive()` で、開区間 $(10^{-5}, 10^5)$ である。区間を狭めるには、`create` のうち区間を受け取るオーバーロードに自分の `Interval`（パラメータごとに 1 つ）を渡す。区間の外の値は `GprErrorCode::InvalidInterval` で拒否される。

```{literalinclude} /../examples/kernels_tour.cpp
:language: cpp
:start-after: // [custom-bounds]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
bounded lengthscale in (0.100000, 10.000000); create(0.05, narrow) -> value 0.05 is not strictly inside (0.1, 10)
after set_params: lengthscale = 2.500000, second update refused, lengthscale still 2.500000
```

## カーネルを合成する

葉を `KernelSpec` で包み、`+` と `*` で組み合わせる。`KernelLeaf` は 10 種の葉と `CustomKernel` を束ねた `std::variant` で、`KernelSpec` はこれから暗黙に変換できる。ただし C++ はユーザー定義変換を 1 回しか許さないため、演算子が両方のオペランドの変換を要求する場面では `KernelSpec(leaf)` と明示する。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [compose]
:end-before: // [end]
:dedent: 4
```

これは $\ell = 1.5$ のときの $k(x, x') = 2\,\exp(-r^2/2\ell^2) + 0.1\,\delta$ である。`*` は `+` より強く結合する。`KernelSpec::sum` と `KernelSpec::product` は演算子の名前付き版で、`KernelSpec::custom` は `CustomKernel` を包む（{doc}`write-custom-kernel` を参照）。`KernelSpec` は値型の不変な木である。コピーは安価で、コピーを変更しても元は変わらない。

### 有効なパラメータベクトル

木は、対数ハイパーパラメータの**平坦な 1 本のベクトル**を公開する。葉は深さ優先・左から右の順に並び、各葉の中のパラメータは上の表の順になる。最適化器が操作するのはこのベクトルである。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [parameters]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
num_params = 3
log-parameters = [0.693147, 0.405465, -2.30259]
  parameter 0 -> leaf 0, local slot 0
  parameter 1 -> leaf 1, local slot 0
  parameter 2 -> leaf 2, local slot 0
coord_mode = Dist
```

- `KernelSpec::num_params`、`get_params`、`set_params`、`bounds_into` は木全体に作用する。
- `parameter_bindings` は、平坦な添字ごとに 1 つの `ParameterBinding` を返す。持つのは `index`、所属する葉の `leaf_id`、その葉の中での `local_index` である。
- `set_params` は木全体で原子的に動く。区間を外れる値があると `InvalidInterval` で失敗し、すべての葉が元の値を保つ。

### カーネルが要求する入力

`coord_mode` が返す `CoordMode` は、コンパイル済みカーネルが何から評価できるかを表す。

| `CoordMode` | 意味 | 該当する葉 |
|---|---|---|
| `Dist` | 二乗距離行列だけで足りる | `RbfKernel`、`MaternKernel`、`PeriodicKernel`、`RationalQuadraticKernel`、`CustomKernel` |
| `Points` | 座標が必要 | ARD の葉、`LinearKernel` |
| `Either` | どちらの入力でもよい | `ConstantKernel`、`WhiteKernel` のみ |
| `Mixed` | 距離だけの葉と座標が必要な葉が混在する | そのような和や積 |

## コンパイルして評価する

`KernelSpec::compile<T>()` は `CompiledKernel<T>` を作る。`T` は `double` か `float`（`KernelScalarConcept`）である。`CompiledKernel<T>::from_spec` が実体のファクトリ関数にあたる。コンパイルでは、入れ子の和と積を平坦化し、ARD の長さスケールをキャッシュし、評価に必要な作業バッファの数を求める。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [compile]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
leaves = 3, nested_depth = 1, reads_distances = true, needs_ard_sq_diff = false
```

`num_leaves`、`nested_depth`、`reads_distances`、`needs_ard_sq_diff`、`coord_mode` がコンパイル済みの木の性質を表す。`CompiledKernel` には `num_params`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings` もあり、意味は `KernelSpec` と同じである。コピー（深いコピー）もムーブもできる。

### グラム行列

行列は列優先のビューで渡す。読み取り用が `MatrixRef<T>`、書き込み用が `MatrixMut<T>` である。`Triangle`（`Lower`、`Upper`、`Full`）は、対称な結果のどの部分を書くかを選ぶ。評価にはどれも、出力と同じ形のスクラッチ行列が要る。出力と別の領域でなければならない。`NestedBuffers` を取らないオーバーロードは自前で確保する。ループの中で確保を避けるには、`make_nested_buffers` で 1 つ作って渡す。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [gram]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
K(0,0) = 2.1  K(1,0) = 1.60147  K(2,0) = 0.822225
```

手計算では $K_{00} = 2 + 0.1 = 2.1$、$K_{10} = 2\exp(-1/4.5) = 1.60147$ である。例はこれらの値と、距離経由と座標経由の結果が一致することを検査している。

| 目的 | 二乗距離から | 座標から |
|---|---|---|
| 学習用グラム $K(X, X)$ | `apply` | `apply_points` |
| 交差共分散 $K(X_1, X_2)$ | `apply_cross` | `apply_cross_points` |
| 事前分散 $k(x_i, x_i)$ | | `fill_diag` |
| 1 階微分 $\partial K / \partial \log\theta_p$ | `grad` | `grad_points` |
| 2 階微分 $\partial^2 K / \partial \log\theta_i \partial \log\theta_j$ | `hess` | `hess_points` |

`apply_cross_points` は `x1` と `x2` の列数が等しくなければならない。`WhiteKernel` の寄与は 0 である。`fill_diag` は長さ $n$ のスパンに書き、白色ノイズの分散を含む。

:::{note}
微分はどれも各パラメータの**対数**に関するもので、`get_params` と揃っている。$k = c\,e^{-r^2/2\ell^2}$ の $\log\ell$ による微分は $k\, r^2/\ell^2$ である。例は手計算でこれを検査している。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [gradient]
:end-before: // [end]
:dedent: 4
```
:::

### 出会うエラー

評価は例外を投げず、`std::expected<void, GprError>` を返す。

| コード | 典型的な原因 |
|---|---|
| `UnsupportedKernelOperation` | 座標が必要な `Points` または `Mixed` のカーネルに `apply`、`apply_cross`、`grad`、`hess` を使った |
| `DimensionMismatch` | `x.cols()` が ARD の葉の長さスケール数と合わない |
| `LengthMismatch`、`EmptyInput` | 出力や入力の形が合わない、または入力が空 |
| `WorkspaceTooSmall`、`InvalidConfig` | スクラッチやネストバッファが小さい、またはスクラッチが出力と重なっている |
| `NonFiniteInput` | 入力に NaN か無限大がある |
| `IndexOutOfRange` | パラメータの添字が `num_params()` 以上 |

### 単精度

同じ `KernelSpec` から `CompiledKernel<float>` もコンパイルできる。ハイパーパラメータは spec の中では `double` のままで、コンパイル時に変換される。`double` との一致は $10^{-6}$ 程度である。

## 指数関数のバックエンドを選ぶ

RBF や Matern のように指数関数を使う組込みの葉は、`std::exp` の代わりにインラインの多項式を使える。`CompiledKernel::kernel_exp` は現在の `KernelExp`（既定は `Accurate`）を返し、`set_kernel_exp` で `FastApprox` に切り替える。トレーナーでは `Gpr::with_kernel_exp` が同じ切り替えである。近似の相対誤差は `double` で数 $10^{-9}$ あり、完全な精度が要る場面の代用品にはならない。トレードオフは {doc}`../explanation/kernels` で述べる。スカラー関数は公開されている。

```{literalinclude} /../examples/kernel_exp.cpp
:language: cpp
:start-after: // [scalar]
:end-before: // [end]
:dedent: 4
```

```{literalinclude} /../examples/kernel_exp.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
max relative error of FastApprox on [-100, 0]: double 7.02e-09, float 1.19e-07
max |Gram(FastApprox) - Gram(Accurate)| = 1.33e-09
max |dK(FastApprox) - dK(Accurate)|     = 6.12e-08
```

`eval_kernel_exp(x, mode)` は、選んだバックエンドでの $e^x$ を返す。`eval_kernel_exp_jet(x, mode)` は、値 `v` と 1 階・2 階微分 `d1`、`d2` を持つ `ExpJet` を返す。`Accurate` は同じ数を 3 回返す。`FastApprox` は自分の多項式の微分を返す。

```{literalinclude} /../examples/kernel_exp.cpp
:language: cpp
:start-after: // [jet]
:end-before: // [end]
:dedent: 4
```

このため、`FastApprox` のカーネルから得た勾配は `FastApprox` のグラム行列と整合する。例では、近似したグラム行列の中心差分と比べて確かめている。

```{code-block} text
finite difference vs FastApprox gradient:  5.35e-11
finite difference vs Accurate gradient:    6.11e-08
```

パラメータ変換、`CustomKernel` の葉、およびカーネル評価の外側は、常に `std::exp` を使う。

## カーネルをトレーナーに渡す

`Gpr` は `KernelSpec` を受け取る。観測ノイズは `Gpr` が自分で足すので、`WhiteKernel` は入れない。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
predict(1.25) = 0.948457 (sin = 0.948985)
predict(3.75) = -0.571757 (sin = -0.571561)
```

目的変数は標準化し、信号分散を最適化器が学べるように `ConstantKernel * RbfKernel` を使っている。トレーナーと予測 API は {doc}`configure-fit` と {doc}`predict-and-sample` を参照。

## 単独の葉とキャッシュ値を扱う

`CompiledKernel` のこれらのメンバは、最適化器や独自のソルバーのためにある。多くの利用者は直接呼ばない。`Gpr` が内部で使っているものである。

### 葉のグラム行列を再利用する

`apply_leaf_points` は 1 枚の葉だけを評価する。`apply_leaf_cached` は同じことを `DistanceCache` から行い、`apply_leaf_cached_incremental` は、一部の長さスケールだけが変わったときに、呼び出し側が持つパック済み ARD 行列を差分更新する。`leaf_index_for_param` は平坦なパラメータ添字から葉を引く。`combine_from_leaf_grams` は、葉ごとの行列から、どの葉も再計算せずに合成後の行列を組み立てる。最適化の 1 ステップで 1 枚の葉のパラメータしか変わらないときに効く。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [leaf-level]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
recombined == gram: true, parameter 2 belongs to leaf 2
```

### 1 回の走査で重み付き勾配を得る

対数周辺尤度の勾配には、すべてのパラメータ $p$ について $g_p = \langle W, \partial K/\partial\theta_p\rangle$ が要る。`weighted_grads` は、木を 1 回たどってこれをすべて計算する。`W` は下三角で渡し、鏡映して使う。非対角成分は 2 回、対角成分は 1 回数えられる。積については、順方向の計算で得た因子のグラム行列を保持できる。`kept_products` と `kept_buffers(products)` がそのバッファの大きさ、`weighted_buffers(products)` が作業領域の大きさを与え、`apply_keeping` と `apply_cached_keeping` はグラム行列を評価しながら因子を保存する。

```{literalinclude} /../examples/kernel_composition.cpp
:language: cpp
:start-after: // [weighted]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
weighted gradients = [14.0503, 5.77053, 0.3]
```

白色ノイズのパラメータの結果は $0.1 \times$ 対角 3 要素である。例は、`grad_points` の出力を手で縮約した値とも全要素で比べている。

### DistanceCache を使う

`apply_cached`、`grad_cached`、`hess_cached` は `DistanceCache` を受け取り、距離の再計算を省く。{doc}`use-supplied-distances` を参照。

## NestedBuffers

`NestedBuffers<T>` は、`rows` $\times$ `cols` の行列を `depth` 枚、連続した 1 つのブロックに持つ。再帰的な木の評価が確保を必要としない。`make_nested_buffers(rows, cols)` は適切な深さのものを作る。`depth`、`rows`、`cols` は容量を報告し、`resize` は縮めずに拡張し、`level(index, rows, cols)` は 1 段分の `MatrixMut<T>` ビューを返す。デフォルト構築した `NestedBuffers` の容量は 0 である。

## API 一覧

| シンボル | 内容 |
|---|---|
| `MaternNu` | `OneHalf`、`ThreeHalves`、`FiveHalves` |
| `RbfKernel`、`RbfArdKernel`、`MaternKernel`、`MaternArdKernel`、`PeriodicKernel`、`RationalQuadraticKernel`、`RationalQuadraticArdKernel`、`ConstantKernel`、`LinearKernel`、`WhiteKernel` | 組込みの葉。`create`、`num_params`、`get_params`、`set_params`、`bounds_into`、`operator==` を持つ |
| `KernelLeaf` | 10 種の葉と `CustomKernel` を束ねる `std::variant` |
| `KernelSpec` | 不変な式の木: `custom`、`sum`、`product`、`operator+`、`operator*`、`num_params`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings`、`coord_mode`、`compile` |
| `ParameterBinding` | `index`、`leaf_id`、`local_index` |
| `CoordMode` | `Dist`、`Points`、`Either`、`Mixed` |
| `CompiledKernel<T>` | 実行用の木: `from_spec`、`num_params`、`num_leaves`、`coord_mode`、`nested_depth`、`reads_distances`、`needs_ard_sq_diff`、`kernel_exp`、`set_kernel_exp`、`make_nested_buffers`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings` |
| `CompiledKernel<T>` の評価 | `apply`、`apply_points`、`apply_cross`、`apply_cross_points`、`fill_diag`、`grad`、`grad_points`、`hess`、`hess_points` |
| `CompiledKernel<T>` のキャッシュ・葉単位 | `apply_cached`、`grad_cached`、`hess_cached`、`leaf_index_for_param`、`apply_leaf_points`、`apply_leaf_cached`、`apply_leaf_cached_incremental`、`combine_from_leaf_grams`、`kept_products`、`kept_buffers`、`weighted_buffers`、`apply_keeping`、`apply_cached_keeping`、`weighted_grads` |
| `NestedBuffers<T>` | `depth`、`rows`、`cols`、`resize`、`level` |
| `KernelExp` | `Accurate`、`FastApprox` |
| `ExpJet<T>` | `v`、`d1`、`d2` |
| `eval_kernel_exp`、`eval_kernel_exp_jet` | `KernelExp` のバックエンドによるスカラーの指数関数 |
