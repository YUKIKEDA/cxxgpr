# 与えた距離でカーネルを評価する

座標ベクトル間のユークリッド距離ではなく、自分で計算した二乗距離（球面上の弧長、グラフ距離、編集距離など）からカーネルを評価する手順を示す。座標から計算した距離を最適化の反復をまたいで再利用する `DistanceCache` も扱う。

先に {doc}`build-kernels` を読むこと。ここで扱うシンボルは `cxxgpr/kernel/distance.hpp` と `cxxgpr/kernel/distance_cache.hpp` にある。シグネチャは [kernel API リファレンス](../reference/api/kernel.md) を参照。

:::{warning}
この文書は**カーネル評価**の API を扱う。`CompiledDistanceKernel` でグラム行列・交差共分散・微分を作る。`Gpr` トレーナーが受け取るのは `KernelSpec` だけである。`DistanceKernel` で学習・予測するには `DistanceGpr` を使う（{doc}`fit-on-supplied-distances`）。
:::

## 与えた距離を使う場面

次のようなときに使う。

- 自然な尺度がユークリッドではない（測地距離、グラフ距離、編集距離、分野固有の距離）
- 距離が表としてすでにあり、座標から再計算するのが無駄、または不可能である
- 1 つのカーネルに、特徴量グループごとなど複数の距離表を読ませたい

:::{warning}
カーネルが妥当な共分散になるのは、グラム行列が半正定値のときだけである。組込みのカーネルが保証するのはユークリッド距離の場合で、任意の距離尺度ではない。弧長を距離とする円周上では、指数カーネル $\exp(-d/\ell)$（`OneHalf` の `MaternKernel`）は妥当だが、弧長の二乗指数カーネルは妥当ではない。選んだ葉が自分の尺度に合うかを確かめること。
:::

## 処理の流れ

```{mermaid}
flowchart LR
    T["distance tables<br/>(double, column-major)"] --> S["DistanceSource<br/>bind_scalar / bind_ard / tidy"]
    K["DistanceKernel (mode tag)<br/>ScalarDistance, ArdDistance leaves"] -->|compile| C["CompiledDistanceKernel"]
    S --> P["prepare_square / prepare_cross"]
    C --> P
    P --> V["CheckedDistanceStore"]
    V --> E["apply / apply_cross / grad / hess / fill_diag"]
    C --> E
```

## 手順 1: 距離表を作る

表に入れるのは**二乗**距離である。平方根が要る葉が、中で取る。$n$ 点の表は、対角が 0 の $n \times n$ 列優先行列になる。例では単位円上の弧長を使う。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [distance-table]
:end-before: // [end]
:dedent: 4
```

## 手順 2: スロットとカーネルを作る

**スロット**は、カーネルの中で 1 つの距離表を指す名前である。`ScalarDistance` は等方的な表 1 つのスロットで、そのファクトリ関数が、等方的な葉をスロットに結び付けて `DistanceKernel<DistanceOnly>` を返す。`rbf`、`matern`、`periodic`、`rational_quadratic`、そして `CustomKernel` 用の `custom` がある。`ConstantKernel` と `WhiteKernel` は表が要らず、`DistanceKernel` に暗黙に変換される。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [kernel]
:end-before: // [end]
:dedent: 4
```

これは $k(a, b) = 2\exp(-\operatorname{arc}(a, b)/1) + 0.1\,\delta$ である。`DistanceKernel` のパラメータ用インターフェースは `KernelSpec` と同じで、`num_params`、`get_params`、`set_params`（原子的）、`bounds_into`、`parameter_bindings` がある。安価な値型で、コピー同士は不変な木を共有する。

スロットごとにプロセス内で一意な `SlotId` が付く。`value` が番号を返し、`operator==` と `operator<=>` で比べられる。`slots` は、カーネルが読むスロットを最初に現れた順に `SlotDescriptor` の値として返す。持つのは `id`、`ard_dims`（スカラーのスロットでは空）、同じ形のスロットの中での添字 `at` である。`num_scalar_slots` と `num_ard_slots` がそれぞれの数を返す。1 つの `ScalarDistance` を複数の葉で使い回すと、それらは同じ表を読む。

```{code-block} text
num_params = 3, scalar slots = 1, ARD slots = 0
```

## 手順 3: 表を検証する

`DistanceKernel::compile<T>` は `CompiledDistanceKernel<T>` を返す。`DistanceSource` は 1 回の呼び出しで使う表を集める。`bind_scalar(slot, d2)` は `MatrixRef<double>` と `MatrixRef<float>` のどちらも受け取り、`bind_ard(slot, tables)` は `MatrixRef<double>` または `MatrixRef<float>` のスパンを受け取る。どちらもメソッドチェーンできるよう自身を返す。`prepare_square` は正方の学習集合を検証して `CheckedDistanceStore<T>` を返し、`prepare_cross` は $n_1 \times n_2$（クエリ × 学習）の表について同じことをする。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [prepare]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
store: 5 x 5, square = true, requires_points = false
```

検証では、スロットの過不足、形、有限性、非負性、そして正方の表では対角が 0 であることと対称性を調べ、値を `T` に変換する。組込みの等方的な葉（`Rbf`、`Matern`、`Periodic`、`RationalQuadratic`）だけを読むスカラー正方表は、下三角順に詰めて保持する。ストアは `rows`、`cols`、`is_square` を報告する。この型は**ムーブ専用**（コピー不可）であり、ムーブ済みのストアを渡すと `EmptyInput` で失敗する。同じストアを何度でも評価に使い回せる。

### 既定は厳格、`tidy` で修復、または `assume_valid` で値検査を省略

浮動小数点のデータから計算した距離には、丸めの誤差が乗りやすい。`DistanceSource::tidy(rel_tol)` は、最大要素に対する相対許容差の範囲で小さな違反を許す。$-\mathrm{rel\_tol}\cdot\max|d^2|$ までの負値は 0 に切り上げ、許容差内の対角は 0 にし、許容差内の非対称な組は平均する。`tidy_tolerance` は設定値を返し、厳格なあいだは `std::nullopt` を返す。

呼び出し側ですでに検証済みの表を繰り返し渡す場合など、要素ごとの値検査の負荷を避けたいときは `DistanceSource::assume_valid(true)` を使う（`assumes_valid()` が現在の設定を返す）。この場合もスロットの一致、形、オーバーフローの検査は行うが、有限性・非負性・対称性・対角の要素検査は省く。`assume_valid(true)` と `tidy` は併用できず、同時に指定すると `GprErrorCode::InvalidConfig` になる。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [tidy]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
strict: invalid squared distance in slot 0 at (0, 0): diagonal must be zero
tidy(1e-9): accepted, tidy_tolerance = 1e-09
```

| `GprErrorCode` | 原因 |
|---|---|
| `InvalidDistance` | 許容差を超える非有限・負・非対称・対角が非 0 の値、または `float` への変換オーバーフロー。エラーはスロットと要素を持つ |
| `DistanceSlot` | カーネルのスロットが欠けている、二重に結び付けられている、または表の種別が違う |
| `UnsupportedKernelOperation` | カーネルが使わないスロットを source が結び付けている |
| `EmptyInput`、`LengthMismatch`、`DimensionMismatch` | 空の表、非正方または形の不一致、ARD の表の数が違う |
| `InvalidConfig` | `tidy` の許容差が負または非有限、または `tidy` と `assume_valid(true)` の併用 |

## 手順 4: 評価する

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [gram]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
K(0,0) = 2.1  K(1,0) = 1.21306  K(3,0) = 0.0995741  K(4,0) = 0.913896
```

手計算で確かめる。角度 $0$ と $0.5$ は $0.5$ 離れているので、$K_{10} = 2e^{-0.5} = 1.21306$ である。角度 $0$ と $5.5$ は、近いほうの回り方で $2\pi - 5.5 = 0.783$ 離れているので、$K_{40} = 2e^{-0.783} = 0.9139$ になる。例はこれらをすべて検査している。

`CompiledDistanceKernel` は `CompiledKernel` と同じ評価の組を持つ。距離行列の代わりにストアを取り、座標は `std::optional<MatrixRef<T>>` で渡す。純粋な距離カーネルでは `std::nullopt` を渡す。

| 目的 | メンバ |
|---|---|
| 学習用グラム | `apply` |
| 交差共分散 | `apply_cross`（`prepare_cross` のストアを使う） |
| 事前分散 | `fill_diag(n, x, out)`。ストアは不要 |
| $\partial K/\partial\log\theta_p$ | `grad` |
| $\partial^2 K/\partial\log\theta_i\partial\log\theta_j$ | `hess` |

どれにも、確保なしで使える `NestedBuffers` 付きのオーバーロードと、自前で確保するオーバーロードがある。`make_nested_buffers`、`nested_depth`、`num_params`、`num_leaves`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings`、`slots`、`kernel_exp`、`set_kernel_exp` は `CompiledKernel` と同じ働きをする。`requires_points` は、カーネルが座標も読むかどうかを返す。

クエリ点と学習点の間の交差共分散には、クエリごとに 1 行の表を使う。`prepare_cross` でストアを作り、白色ノイズは寄与しない。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [cross]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
k(1.0, 0.0) = 0.735759 (hand 0.735759)
```

パラメータ 1 は対数の長さスケールなので、`param_idx = 1` の `grad` は $2e^{-r/\ell}$ の $\log\ell$ による微分、つまり $2e^{-r/\ell}\,r/\ell$ を返す。例は 1 要素をこの値と照合している。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [gradient]
:end-before: // [end]
:dedent: 4
```

## ARD: 特徴量ごとに 1 つの表

`ArdDistance` のスロットが期待するのは、**特徴量ごとの二乗差の表 1 枚ずつ**である。`ArdDistance::create(dims)` は `dims == 0` を `EmptyInput` で拒否する。`dims` は個数を、`slot` は ID を返す。ファクトリ関数 `rbf_ard`、`matern_ard`、`rational_quadratic_ard` は `std::expected<DistanceKernel<DistanceOnly>, GprError>` を返す。葉の長さスケール数が `dims` と違うと `DimensionMismatch` で拒否されるためである。表は `bind_ard(slot, MatrixRef<double> のスパン)` で結び付ける。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [ard]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
ARD K(1,0) = 0.367879 (hand 0.367879), dims = 2
```

## 与えた距離と座標を混ぜる

`DistanceKernel` はモードタグを取る。`DistanceOnly`（既定）は座標が要らない。`WithPoints` は、`KernelSpec` 由来の `LinearKernel` や ARD の葉など、座標を使う葉も含められる。`DistanceModeConcept` は、タグをこの 2 つに制約する。`with_points` は `DistanceOnly` のカーネルを昇格させる。`operator+` と `operator*` は、同じモードの 2 つのカーネル、または `DistanceKernel<DistanceOnly>` と `KernelSpec`（どちらの順でもよい）を結び、`WithPoints` のカーネルを返す。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [with-points]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
mixed requires_points = true, K(2,1) = 1.36788
```

ここでは $K_{21} = e^{-1} + 0.5\cdot(1\cdot2) = 1.36788$ である。座標なしでこのようなカーネルに `apply` を呼ぶと、`UnsupportedKernelOperation` で失敗する。

## 座標から計算した距離を再利用する

通常の座標カーネルでは、距離を 1 回だけ計算して最適化の反復をまたいで再利用できる。`DistanceCache<T>` は、等方的な葉のための密な $n \times n$ 二乗距離行列と、ARD の葉のための特徴量ごとのパック済みの表を保持する。

```{literalinclude} /../examples/supplied_distances.cpp
:language: cpp
:start-after: // [cache]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
cache: points = 3, dims = 2, has_dist = true, has_ard = true, packed size = 12
```

| メンバ | 役割 |
|---|---|
| `DistanceCache::build(x, cache_dist, cache_ard, policy)` | 構築済みのキャッシュかエラー（`EmptyInput`、`NonFiniteInput`、`SizeOverflow`）を返す静的ファクトリ |
| `ensure(x, cache_dist, cache_ard, policy)` | 既存のキャッシュをその場で埋め直す。フラグが false か方針が `Uncached` ならバッファを空にする |
| `reset` | 容量を残したままキャッシュを空にする |
| `policy`、`num_points`、`num_dims` | 直近の `build` か `ensure` が記録した内容 |
| `has_dist`、`has_ard` | どのバッファが埋まっているか |
| `dist_view`、`dist_mut` | 二乗距離の $n \times n$ ビュー（キャッシュしていなければ空） |
| `ard_packed`、`ard_packed_mut` | $d \cdot n(n+1)/2$ 要素のスパン |

パック済み ARD の配置は、特徴量ごとに $d$ 個のブロックを並べ、各ブロックが下三角を列ごとに保持する。行 $i \ge j$ の要素は、ブロック内の `j * (2n - j + 1) / 2 + (i - j)` にある。例は、$(s_1 - s_0)^2$ がこの式で求めた位置にあることを確かめている。

`DistanceCachePolicy` は、メモリと速度のどちらを取るかを選ぶ。`Cached` は表を事前に計算して保持する。`Uncached` はバッファを空にしたままにし、呼び出しのたびに座標から距離を計算する。`Gpr` ではこれを `with_distance_cache_policy` と `FitPolicy` で指定する。{doc}`configure-fit` を参照。キャッシュを使うのは `CompiledKernel::apply_cached`、`grad_cached`、`hess_cached` と、{doc}`build-kernels` で説明した葉単位の版である。

## API 一覧

| シンボル | 内容 |
|---|---|
| `DistanceOnly`、`WithPoints`、`DistanceModeConcept` | `DistanceKernel` のモードタグと、それを許すコンセプト |
| `SlotId` | プロセス内で一意なスロット ID: `value`、`operator==`、`operator<=>` |
| `SlotDescriptor` | `id`、`ard_dims`、`at` |
| `ScalarDistance` | 等方的なスロット: `slot`、`rbf`、`matern`、`periodic`、`rational_quadratic`、`custom` |
| `ArdDistance` | ARD のスロット: `create`、`slot`、`dims`、`rbf_ard`、`matern_ard`、`rational_quadratic_ard` |
| `DistanceKernel<Mode>` | 式の木: `with_points`、`num_params`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings`、`slots`、`num_scalar_slots`、`num_ard_slots`、`compile`、`operator+`、`operator*` |
| `DistanceSource` | `bind_scalar`（`double`/`float`）、`bind_ard`（`double`/`float`）、`tidy`、`tidy_tolerance`、`assume_valid`、`assumes_valid` |
| `CheckedDistanceStore<T>` | ムーブ専用の検証済みの表: `rows`、`cols`、`is_square` |
| `CompiledDistanceKernel<T>` | `num_params`、`num_leaves`、`requires_points`、`nested_depth`、`kernel_exp`、`set_kernel_exp`、`make_nested_buffers`、`get_params`、`set_params`、`bounds_into`、`parameter_bindings`、`slots`、`prepare_square`、`prepare_cross`、`apply`、`apply_cross`、`fill_diag`、`grad`、`hess` |
| `DistanceCachePolicy` | `Cached`、`Uncached` |
| `DistanceCache<T>` | `build`、`ensure`、`reset`、`policy`、`num_points`、`num_dims`、`has_dist`、`has_ard`、`dist_view`、`dist_mut`、`ard_packed`、`ard_packed_mut` |
