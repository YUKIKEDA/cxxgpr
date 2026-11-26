# 低レベルの部品を使う

`Gpr` と `FittedGpr` は、公開された部品の組み合わせでできている。密行列と三角行列の線形代数、決定的な並列ヘルパー、事前確保のワークスペース、周辺尤度のカーネル、事後推論、混合精度の反復改良エンジンである。これらは利用者が直接呼べる。このガイドは、いつ、どのように直接呼ぶかを示す。たとえば、独自の学習ループや最適化器を書くとき、1 つの因子を多数の右辺に使い回すとき、反復改良に独自の線形系を与えるとき、独立な計算で結果を検算するときに役立つ。

学習と予測だけが目的なら、高水準の API（{doc}`configure-fit`、{doc}`predict-and-sample`）を使う。このページの内容はどれも鋭い道具である。バッファは利用者が持ち、精度は利用者が選び、事前条件は利用者が守る。宣言の一覧は、[linalg](../reference/api/linalg.md)、[workspace](../reference/api/workspace.md)、[precision](../reference/api/precision.md) の各リファレンスにある。

## ヘッダーと規約

| ヘッダー | 名前空間 | 内容 |
|---|---|---|
| `cxxgpr/linalg/cholesky.hpp` | `cxxgpr::linalg` | インプレースのコレスキー分解、ジッター付き再試行、三角求解、逆行列の対角、対数行列式 |
| `cxxgpr/linalg/dense.hpp` | `cxxgpr::linalg` | 対角の更新、下三角のコピーと鏡映、内積、行列ベクトル積 |
| `cxxgpr/linalg/parallel.hpp` | `cxxgpr::linalg` | 面積で均衡させた列の分割、決定的な並列ループと縮約 |
| `cxxgpr/workspace/buffers.hpp` | `cxxgpr` | `FitPolicy`、`FitBuffers`、`WorkspaceCore`、`HessianScratch`、`QueryWorkspace` |
| `cxxgpr/workspace/mll.hpp` | `cxxgpr` | 負の対数周辺尤度、勾配、ヘッセ行列 |
| `cxxgpr/workspace/inference.hpp` | `cxxgpr` | 事後平均、分散、共分散、サンプル、1 個抜き交差検証の予測 |
| `cxxgpr/workspace/leaf_cache.hpp` | `cxxgpr` | 増分グラム更新のための `LeafCache` と最適化器トレイト |
| `cxxgpr/workspace/workspace.hpp` | `cxxgpr` | 上の 4 つの `workspace` ヘッダーをまとめる統合ヘッダー |
| `cxxgpr/precision/refine.hpp` | `cxxgpr` | 反復改良エンジン、`RefineSystem`、残差ヘルパー |

すべての関数に共通する規約は次のとおり。

- **行列は列優先のビューである。** `MatrixRef<T>`（読み取り専用）と `MatrixMut<T>`（書き込み可）は、ポインタ、行数、列数、`col_stride()`（隣り合う列の先頭の間隔、要素数）を持つ。`MatrixMut<T>` は `MatrixRef<T>` に暗黙に変換できる。
- **スカラーは `float` か `double` である。** これは `KernelScalarConcept` の要求による。精度ポリシー `P` を取る関数は、行列と因子に `P::Storage`、重みと出力に `P::Refine` を使う。
- **エラーは値である。** すべての関数が `std::expected<T, GprError>` を返し、例外は投げない。よく出るコードは `EmptyInput`、`SizeOverflow`、`LengthMismatch`、`DimensionMismatch`、`NonFiniteInput`、`NonPositiveDefiniteMatrix`、`CholeskyFailed`、`WorkspaceTooSmall` である。
- **下三角だけを使う。** 対称行列は下三角（対角を含む）で保持し、読む。厳密な上三角は、関数が明記しない限り、読みも書きもしない。

## `cxxgpr::linalg` で因子分解して解く

次のプログラムは、対称正定値行列 $A$ を作り、そのコピーをインプレースで因子分解し、因子で解く。全文は `low_level_cholesky.cpp` にある。

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [factor]
:end-before: // [end]
:dedent: 4
```

`cholesky_lower_in_place(a, stage)` は、`a` の下三角を $A = L L^\top$ を満たす $L$ で上書きする。`stage` 引数（既定は `CholeskyStage::Fit`、もう一方は `CholeskyStage::Predict`）は、`CholeskyFailed` エラーに記録されるだけである。失敗がどこで起きたかを区別するために使う。$A$ を残したいときは、`copy_lower_triangle(src, dst)` でコピーを取るのが普通である。

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [solve]
:end-before: // [end]
:dedent: 4
```

求解はすべて右辺をその場で書き換える。右辺はベクトル（`std::span<T>`）でも、複数の右辺を持つ行列（`MatrixMut<T>`）でもよい。

| 関数 | 解く式 | 備考 |
|---|---|---|
| `solve_lower_in_place(l, rhs)` | $L X = B$ | 前進代入 |
| `solve_lower_transpose_in_place(l, rhs)` | $L^\top X = B$ | 後退代入。$L$ の下三角だけを読む |
| `solve_llt_in_place(l, rhs)` | $L L^\top X = B$ | 2 つの代入を続けて行う。$A$ で解くときに使う |
| `inv_diag_from_chol_l(l, out_diag, scratch)` | $\operatorname{diag}(A^{-1})$ | $n \times n$ のスクラッチが必要で、上書きされる。倍精度で累積する |
| `log_det_from_l(l)` | $\log\lvert A\rvert = 2\sum_i \log L_{ii}$ | Neumaier の補償付き和。`double` で返る |
| `half_log_det_from_l(l)` | $\tfrac12 \log\lvert A\rvert$ | 周辺尤度に現れる項 |

これらの関数は、呼ぶたびに入力を検証する。因子は正方で、下三角が有限、対角が厳密に正でなければならない。右辺は有限で、行数が一致しなければならない。対角に非正の要素があれば `NonPositiveDefiniteMatrix` を返す。

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [diag]
:end-before: // [end]
:dedent: 4
```

:::{warning}
`cholesky_lower_in_place` が `CholeskyFailed` で失敗したとき、`a` の下三角は部分的に上書きされている。再試行したいときは、$A$ のコピーを取っておくか、`cholesky_lower_with_jitter` を使う（こちらは元に戻す）。
:::

### ホットループで検証を省く

`*_unchecked` 関数は、同じ代入を、検証もエラーの返却もなしで行う。`noexcept` である。同じ因子に対して検証つきの呼び出しを 1 回成功させた後で、空でなく、ストライドが正しく、重ならず、値が有限なビューに対してだけ呼ぶ。

`solve_lower_in_place_unchecked`、`solve_lower_transpose_in_place_unchecked`、`solve_llt_in_place_unchecked` があり、それぞれ行列用とベクトル用のオーバーロードを持つ。

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [unchecked]
:end-before: // [end]
:dedent: 4
```

### ジッターで再試行する

`cholesky_lower_with_jitter` は、まずジッターなしで因子分解を試みる。`CholeskyFailed` で失敗したときだけ、保存しておいた下三角を復元し、`JitterPolicy` の次のジッター値を対角に足して再試行する。最終的に成功したジッター値を返す（不要だったときは `0.0`）。オーバーロードは 2 つある。1 つは、同じ大きさの `backup` 行列を利用者が渡し、何も確保しない。もう 1 つは、バックアップを自分で確保する。

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [jitter]
:end-before: // [end]
:dedent: 4
```

ジッターの値は、足す前に行列のスカラー型へ変換される。そのため、`float` の分解能より小さいジッターは、単精度の行列には何も効かない。`JitterPolicy::fixed(0.0)` は再試行しない。ポリシーの型（`FixedJitter`、`AdaptiveJitter`、`JitterSchedule`、`JitterPolicy`）は {doc}`configure-fit` と {doc}`../explanation/numerical-stability` で説明する。

ここまでのプログラムの出力は次のとおり。

```{code-block} text
L(0,0) = 1.55547, upper sentinel untouched: 1
max |L L^T - A| = 4.44089e-16
max |A x - b| = 1.77636e-15
two triangular solves equal solve_llt_in_place: 1
diag(A^-1) agrees with the explicit inverse to 1.11022e-16
log|A| = 3.64456, 0.5 log|A| = 1.82228, naive sum = 3.64456
unchecked solve identical: 1
unjittered factorization of a rank-one matrix: Cholesky factorization failed (stage=Fit, size=4, jitter=0 already applied)
adaptive jitter policy succeeded with j = 1e-10
fixed(0) policy: CholeskyFailed, error code matches: 1
```

## 密行列のヘルパー

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [dense]
:end-before: // [end]
:dedent: 4
```

| 関数 | 働き |
|---|---|
| `add_to_diag(a, value)` | 正方行列の対角要素すべてに、有限の `value` を足す。ノイズやジッターの付加はこの関数で行う |
| `copy_lower_triangle(src, dst)` | 対角を含む下三角をコピーする。厳密な上三角には触れない |
| `symmetrize_lower(a)` | 厳密な下三角を、その場で厳密な上三角へコピーする |
| `frobenius_lower(a, b)` | 2 つの対称行列の $\langle A, B\rangle_F$。下三角だけを読み（対角は 1 回、非対角は 2 回数える）、`double` で累積する |
| `dot(a, b)` | 長さの等しい 2 つのスパンの内積。添字の昇順に `double` で累積する |
| `matvec_columns(a, x, out)` | 列優先の $A$ について $\text{out} = A x$ を、列ごとに計算する |

`dot`、`frobenius_lower`、`matvec_columns` は、固定した順序で素直に `double` へ累積する。決定的だが、補償はしない。Neumaier の補償を使うのは、`log_det_from_l` と、次節の並列縮約だけである。

`precision/refine.hpp` には、もう 1 つ、ヘッダーだけで定義された `cxxgpr::symmetrize_lower(MatrixMut<T>)` がある。働きは同じで、戻り値はなく、空、ヌル、非正方のビューでは黙って何もしない。

出力は次のとおり。

```{code-block} text
asymmetry after symmetrize_lower = 0, <A,A>_F = 68.8858 (full sum of squares 68.8858)
```

## 決定的な並列ヘルパー

グラム行列の計算とコレスキー分解は、$n \times n$ 行列の下三角を、面積の等しい列ブロックに分ける。前のほうの列は後ろの列より長いからである。自分の下三角ループにも、同じ分割を使える。

```{literalinclude} /../examples/low_level_cholesky.cpp
:language: cpp
:start-after: // [parallel]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
lower_area(1000, 1000) = 500500 (= 1000 * 1001 / 2)
4 area-balanced column blocks start at: 0 134 293 500 1000
every column visited exactly once: 1
sum of 1/k^2 for k = 1..1000 = 1.64393456668, bit-identical for 1, 4 and automatic threads: 1
openmp_max_threads() >= 1: 1
```

| 関数 | 用途 |
|---|---|
| `lower_area(n, col)` | 列 $[0, \min(\text{col}, n))$ に含まれる下三角の要素数 |
| `lower_block_start(n, block_idx, n_blocks)` | 下三角を面積がほぼ等しい `n_blocks` 個のブロックに分けたときの、ブロック `block_idx` の先頭列。`block_idx == n_blocks` なら `n` を返す |
| `openmp_max_threads()` | OpenMP のスレッド数の上限（OpenMP なしなら 1） |
| `parallel_for_lower_blocks(n, num_threads, fn)` | 空でない各ブロックについて `fn(col_begin, col_end)` を 1 回ずつ呼ぶ。どの列もちょうど 1 つのブロックに属するので、列ごとのスロットへの書き込みにロックは要らない |
| `parallel_for_lower_blocks_raw(n, num_threads, fn, ctx)` | 同じ働きを、関数ポインタ `void(*)(size_t, size_t, void*)` とコンテキストポインタで行う |
| `parallel_lower_reduce_cols(n, num_threads, col_partials, fn)` | すべての列について `fn(col)` を並列に評価して `col_partials` に入れ、列番号の昇順に Neumaier の補償付きで足し上げる。結果はスレッド数によらずビット単位で一致する |
| `parallel_lower_reduce_cols_raw(n, num_threads, col_partials, fn, ctx)` | 同じ働きを、関数ポインタ `double(*)(size_t, void*)` で行う |

`num_threads == 0` は「`openmp_max_threads()` を使う」の意味である。呼び出し可能オブジェクトは、キャプチャなしのトランポリン経由でポインタとして渡されるので、これらのテンプレートは確保を行わない。`col_partials` の要素数はちょうど `n` でなければならない。`_raw` 版は、関数ポインタがヌルのとき `InvalidConfig` も返す。

## 自分のバッファで周辺尤度、勾配、ヘッセ行列を計算する

`cxxgpr/workspace/mll.hpp` の関数は、負の対数周辺尤度

$$
\mathcal{L}(\theta) = \tfrac12\, y^\top A^{-1} y + \tfrac12 \log\lvert A\rvert + \tfrac n2 \log 2\pi,
\qquad A = K + \sigma_n^2 I
$$

と、対数空間のパラメータ $\theta$（すべてのカーネルパラメータに続いて $\log\sigma_n^2$）に関する導関数を計算する。入力は、`CompiledKernel`、`GaussianLikelihood`、記憶精度の学習データ、`JitterPolicy`、`FitBuffers<P>` である。全文は `workspaces.cpp` にある。データは $\sin 2x$ にノイズを加えた 30 点で、カーネルは `ConstantKernel * RbfKernel` である。

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [factor]
:end-before: // [end]
:dedent: 4
```

`FitBuffers<P>::create(n, policy)` は、$n$ 個の学習点に必要な領域を 1 回で確保する（下の表を参照）。`factor_and_mll` は $A$ を組み、因子分解し、$A\alpha = y$ を解き、$\mathcal{L}$ を返す。$L$ は `buffers.core.k_matrix`（ビューは `core.k_ref()`）、$\alpha$ は `buffers.core.rhs`、採用したジッターは `core.factor_jitter` に残る。例では、これらの値から `linalg::dot` と `linalg::half_log_det_from_l` で $\mathcal{L}$ を計算し直している。

`factor_and_mll` の第 3 引数 `keep_products` は、勾配の計算のために因子ごとのグラム行列を残す、トップレベルの積項の個数である（カーネルが対応する個数は `CompiledKernel::kept_products()` が返す）。既定値の `0` は何も残さない。`mll_and_gradient_into` は、適切な値を自分で渡す。

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [gradient]
:end-before: // [end]
:dedent: 4
```

`mll_and_gradient_into` は、カーネル木を 1 回たどって $\mathcal{L}$ と勾配を計算する。`out_grad` の要素数は `kernel.num_params() + 1` でなければならない。

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [hessian]
:end-before: // [end]
:dedent: 4
```

`hessian_into` は、対称な $p \times p$ のヘッセ行列を `out_hess`（長さ $p^2$、$p$ は `kernel.num_params() + 1`）に書く。`value_gradient_hessian_into` は、1 回の因子分解から、値を返し、勾配とヘッセ行列の両方を埋める。`core.hessian` の `HessianScratch` は、最初にヘッセ行列を計算するまで空である。`is_allocated()` で確かめられる。

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [policies]
:end-before: // [end]
:dedent: 4
```

`FitPolicy::prefer_memory()` では、バッファに $W$ 用の別行列も距離キャッシュもない。勾配とヘッセ行列の計算は $L$ を上書きし、戻る前に再分解する。結果は同じである。

```{code-block} text
negative log marginal likelihood = -10.1376
adopted jitter                   = 0
recomputed from L and alpha      = -10.1376
gradient (analytic | finite difference):
  d/dtheta0:      0.94508 |      0.94508
  d/dtheta1:     -2.56634 |     -2.56634
  d/dtheta2:     -2.18095 |     -2.18095
Hessian scratch allocated before/after the first call: 0 / 1
Hessian vs finite differences of the gradient, worst relative error = 1.6177e-08
prefer_speed : W buffer 900 doubles, distance cache 1, overwrites L 0
prefer_memory: W buffer 0 doubles, distance cache 0, overwrites L 1
same value and gradient under both policies: 1
```

| 関数 | 戻り値 | 備考 |
|---|---|---|
| `factor_and_mll<P>(kernel, likelihood, x, y, jitter, buffers, keep_products = 0)` | `double` の $\mathcal{L}$ | `y` は `std::span<const P::Storage>`。ジッターの段階をすべて試して失敗すると `CholeskyFailed` |
| `mll_and_gradient_into<P>(kernel, likelihood, x, y, jitter, buffers, out_grad)` | $\mathcal{L}$ | `CholeskyBuffer::Reuse` では $L$ を復元する |
| `hessian_into<P>(..., buffers, out_hess)` | なし | 最初の呼び出しで `HessianScratch` を確保する |
| `value_gradient_hessian_into<P>(..., buffers, out_grad, out_hess)` | $\mathcal{L}$ | 1 回の因子分解で 3 つとも求める |

`P` は `buffers` の型から推論される。

### バッファの型

`FitPolicy` は、メモリに関する 2 つの選択をまとめる。メンバは `distance_cache`（`DistanceCachePolicy`）と `cholesky_buffer`（`CholeskyBuffer`）である。`FitPolicy::prefer_speed()` は `Cached` と `Retain`、`FitPolicy::prefer_memory()` は `Uncached` と `Reuse` を選ぶ。`operator==` は 2 つのフィールドを比較する。

`FitBuffers<P>` のメンバと関数は次のとおり。

| メンバ | 意味 |
|---|---|
| `create(n, cache = Cached, buffer = Retain)` と `create(n, policy)` | $n$ 点分を確保する。`n == 0` なら `EmptyInput`、$n^2$ がオーバーフローすれば `SizeOverflow` |
| `core` | 後述の `WorkspaceCore<P>` |
| `dist` | `std::optional<DistanceCache<Storage>>`。`cache_policy == Cached` のときに存在し、最初の使用時に埋まる |
| `w_matrix` | `Retain` のとき、$W = \alpha\alpha^\top - A^{-1}$ を置く $n \times n$ のバッファ。`Reuse` では空 |
| `cache_policy`、`buffer_policy` | 要求された 2 つのポリシー |
| `overwrites_cholesky()` | `Reuse` なら `true`。勾配とヘッセ行列の計算が $L$ を壊し、後で再分解する |
| `w_mut()`、`w_ref()` | $W$ のビュー。`Retain` では `w_matrix`、`Reuse` では `k_matrix` |

`WorkspaceCore<P>` は、学習で共有するスクラッチを持つ。

| メンバ | 意味 |
|---|---|
| `create(n)` | $n$ 点分のコアバッファを確保する |
| `order()` と `order_` | 学習点の個数 $n$ |
| `k_matrix`、`k_mut()`、`k_ref()` | $A = K + \sigma_n^2 I$。`factor_and_mll` の後は $L$ |
| `exp_buf`、`exp_mut()`、`exp_ref()` | カーネル評価、ジッターのバックアップ、導関数のためのスクラッチ |
| `kernel_scratch`、`kernel_scratch_mut()` | 勾配とヘッセ行列を同時にたどるための追加の $n\times n$ スクラッチ。`ensure_kernel_scratch()` まで空 |
| `ensure_kernel_scratch()` | `kernel_scratch` を 1 回確保する |
| `rhs` | 求解前は $y$、求解後は $\alpha$ |
| `theta` | 因子分解の失敗後にパラメータを戻すためのバックアップ |
| `col_partials` | 決定的な縮約のための、列ごとの部分和（長さ $3n$） |
| `nested`、`weighted`、`ensure_weighted(n_bufs)` | 木の評価用と、1 パス勾配用の `NestedBuffers` |
| `ard_s2_storage`、`ard_s2_ready`、`ensure_ard_s2(n_ard_leaves)`、`clear_ard_s2_ready()`、`ard_s2_mut(slot)`、`ard_s2_ref(slot)` | 順方向の評価で保存し、逆方向の `weighted_grads` で再利用する ARD 葉ごとのパック済み $s^2$ 記憶域 |
| `hessian` | 遅延確保される `HessianScratch` |
| `factor_jitter` | 直近に成功した因子分解で使ったジッター |

`HessianScratch<S>` は、解析的ヘッセ行列の途中量を持つ。$p$ 個のパラメータそれぞれの $S_i = L^{-1} A_i L^{-\top}$、$v_i = L^{-1} A_i \alpha$、補助バッファ 2 つである。

| メンバ | 意味 |
|---|---|
| `ensure(n, p)` | 次数 $n$、パラメータ数 $p$ の分を確保する。既存の容量は保つ |
| `is_allocated()` | `ensure` が成功した後は `true` |
| `s_mut(i)`、`s_ref(i)` | $n \times n$ 行列 $S_i$ |
| `v_col_mut(i)`、`v_col_ref(i)` | $n \times p$ 行列 $V$ の第 $i$ 列 |
| `scratch_c_mut()` | 3 つ目の $n\times n$ スクラッチ行列 |
| `order_`、`num_params_`、`s_storage`、`v_storage`、`u`、`scratch_c` | 大きさと、生の記憶領域 |

## `LeafCache` で増分的にグラム行列を更新する

`FastSimulatedAnnealing` のような最適化器は、ハイパーパラメータを 1 つずつ動かす。変わったパラメータに依存するカーネル葉が 1 つだけなら、ほかの葉のグラム行列を計算し直す必要はない。`LeafCache<T>` は葉ごとに 1 つのグラム行列を持ち、宣言されていないパラメータの変更がないことを確かめ、汚れた葉だけを再計算する。

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [leaf-cache]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
leaves cached: 2, primed: 1
incremental Gram vs full evaluation, max difference = 0
undeclared change rejected: 1
```

検査はビット単位で厳密である。`prepare_step` は、`change_indices` に含まれないすべてのパラメータを、記録済みのベクトルと `std::bit_cast` で比べる。`+0.0` から `-0.0` への変化でも検出し、`IndexOutOfRange` を返す。

| メンバ | 用途 |
|---|---|
| `LeafCache()` | 空のキャッシュ。何も確保しない |
| `resize(n_leaves, n, ard_dims_per_leaf = {})` | グラム行列、ARD 葉ごとの $s^2$ スロット、動的な $O(d)$ 長さスケールスナップショットを確保し、すべての葉を汚れた状態にする |
| `prepare_step(params, change_indices, kernel, total_params)` | ステップを検証し、汚れた葉に印をつける |
| `prime(kernel, x, dist_cache, params, out, uplo, scratch, nested)` | すべての葉を評価し、`params` から ARD の $s^2$ 基準を初期化する |
| `evaluate_incremental(kernel, x, dist_cache, params, change_indices, total_params, out, uplo, scratch, nested, commit_immediately = true)` | 汚れた葉だけを再計算して合成後のグラム行列を `out` に書く（`commit_immediately == false` では非破壊の試行評価として保持）。失敗時はキャッシュを無効化する |
| `commit_candidate(accepted, kernel, x, dist_cache)` | `accepted == true` のとき試行した ARD $s^2$ 基準を確定し、`false` なら受理済みの基準を壊さず $O(1)$ で破棄する |
| `record(params)` | `params` を新しい基準として保存し、キャッシュを初期化済みにする |
| `mark_all_dirty()`、`invalidate()` | 全再計算を強制する（後者は初期化済みの状態も解除する） |
| `is_primed()`、`is_leaf_dirty(leaf)`、`num_leaves()`、`order()`、`last_params()` | 状態の問い合わせ |
| `leaf_gram(leaf)`、`leaf_gram_mut(leaf)` | 1 つの葉のキャッシュされたグラム行列 |

関連する自由シンボルがある。`require_change_indices(indices, n_params)` は、空、範囲外、重複のあるインデックス列を拒否する。`OptimizerTraits<O>::uses_change_indices` と変数テンプレート `optimizer_uses_change_indices_v<O>` は、最適化器の型が変更インデックスを供給するかどうかを、コンパイル時に示す。コンセプト `OptimizerHasChangeIndicesConcept` は、`bool` に変換できる静的メンバ `uses_change_indices` を持つ型に合致する。自分の最適化器の型は、`static constexpr bool uses_change_indices = true;` と宣言すれば、この仕組みに参加できる。

## 自分の因子で事後推論する

$L$、$\alpha$、学習入力があれば、`cxxgpr/workspace/inference.hpp` は `FittedGpr` なしで事後分布を計算する。`QueryWorkspace<P>` が一時領域を提供し、必要に応じて増える。

```{literalinclude} /../examples/workspaces.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
query workspace capacity: n = 30, m = 5, d = 1
  x* =  0.25  mean =   0.481545  sd =  0.109424  sin(2x*) =  0.479426
  x* =  0.75  mean =    1.03939  sd =  0.108098  sin(2x*) =  0.997495
  x* =   1.5  mean =   0.155948  sd =  0.107833  sin(2x*) =   0.14112
  x* =  2.25  mean =   -1.03641  sd =  0.108098  sin(2x*) =  -0.97753
  x* =  2.75  mean =  -0.681773  sd =  0.109424  sin(2x*) =  -0.70554
low-level vs FittedGpr::predict, max mean difference = 0
leave-one-out RMSE against the noisy targets = 0.120851
covariance is 5 x 5, 20 sample values, same seed gives identical draws: 1
```

標準偏差には観測ノイズが含まれる（`PredictOptions` の既定は `VarianceKind::Observation`）ので、$\sqrt{0.01 + \text{潜在分散}} \approx 0.11$ になる。これらの関数は、渡されたデータをそのまま扱う。入力変換も目的変数変換も、逆スケーリングもない。変換が必要なら、呼び出しの前後で自分で行う。

| 関数 | 結果 | 備考 |
|---|---|---|
| `predict_into<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, options, workspace, out)` | `Prediction<P::Refine>` を埋める | `workspace` と `out` に容量があれば確保しない |
| `predict<P>(...同じ引数、out なし)` | `Prediction<P::Refine>` を返す | 結果を確保する |
| `predict_covariance_into<P>(..., out)` | `PredictiveCovariance<P::Refine>` を埋める | $\Sigma_* = K_{**} - V^\top V$（$V = L^{-1}K_*$）。$m \times m$ の列優先行列 |
| `predict_covariance<P>(...)` | `PredictiveCovariance<P::Refine>` を返す | 結果を確保する |
| `sample<P>(kernel, likelihood, x_train, l_factor, alpha, x_query, n_draws, seed, options, workspace, jitter = nullopt)` | 長さ `n_draws * m` の `std::vector<P::Refine>` | 第 `d` 回のサンプルは `[d * m, (d + 1) * m)` に入る。共分散の因子分解には、任意の `jitter` を使う。既定は `1e-12` から始まり `1e-2` で打ち切る適応的ポリシーである |
| `loo_predict_into<P>(likelihood, l_factor, alpha, y_train, options, scratch_n_by_n, scratch_diag, out)` | `Prediction<P::Refine>` を埋める | 閉形式の 1 個抜き予測。$n \times n$ と長さ $n$ のスクラッチが必要 |
| `loo_predict<P>(likelihood, l_factor, alpha, y_train, options)` | `Prediction<P::Refine>` を返す | スクラッチと結果を確保する |

`P` は `QueryWorkspace<P>` の引数から推論される。ただし `loo_predict` と `loo_predict_into` にはそのような引数がないので、`loo_predict<DoublePrecision>(...)` のように明示する。サンプルは決定的で、同じシードならスレッド数によらず同じ値になる。

`QueryWorkspace<P>` のメンバは次のとおり。

| メンバ | 意味 |
|---|---|
| `ensure(n, m, d, nested_depth)` | $n$ 個の学習点、$m$ 個のクエリ、$d$ 個の特徴量に対応できるよう、すべてのバッファを広げる。縮めることはない |
| `query_xs`、`query_x` | 変換後のクエリを `double` と `Storage` で持つ（m × d） |
| `query_k_star`、`query_scratch` | 交差共分散 $K_*$（のちに $L^{-1}K_*$）とそのスクラッチ（n × m） |
| `query_nested` | 交差共分散の木のための `NestedBuffers` |
| `query_kss`、`query_diag_scratch` | 事前分散 $k(x_*,x_*)$ と、対角評価のスクラッチ |
| `train_n`、`query_m`、`dims` | 直近に要求された容量 |

## 混合精度の反復改良を自分で動かす

`cxxgpr/precision/refine.hpp` は、`MixedPrecision` が単精度の解を改良するのに使うエンジンを含む。安価な近似解法（`float` の因子）と正確な残差（`double` の行列、または再評価したカーネル）があれば、任意の線形系 $B w = b$ に使える。全文は `mixed_precision_refine.cpp` にある。

系は、`RefineSystem` から派生して 4 つの関数を実装することで記述する（5 つ目の `accept` には既定の実装がある）。

| 仮想関数 | 利用者が与えるもの |
|---|---|
| `rhs()` | 右辺 $b$ を `std::span<const double>` で返す。反復改良の間、有効でなければならない |
| `residual(w, r)` | $r = b - Bw$ を `r` に書き、$\lVert B\rVert_\infty$ を返す。失敗時はエラー |
| `correct(r, w)` | $B\,\delta = r$ の近似解を `w` に足す（`float` の三角求解） |
| `accept(w, tol)` | 任意。収束した反復解を拒否するなら `false` を返す。既定は `true` |
| `fallback(w)` | `w` を正確な解で上書きする（倍精度の求解） |

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [system]
:end-before: // [end]
```

`RefineSystem` はコピーも移動もできない。デストラクタは仮想で、コンストラクタは protected である。

例では、放射基底関数のグラム行列、その `float` のコピーと `float` の因子、倍精度の基準解を作る。

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [setup]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
float solve only:   relative error 7.951e-05
tolerance 10 n eps = 1.332e-13, at most 10 corrections, stagnation above ratio 0.9 twice in a row
```

続いて、系の 5 つの変種に `refine_in_place` を実行し、それぞれの `RefinementSummary` を表示する。

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [run]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
stored float matrix        Converged, corrections 2, fallback 0, relative residual 1.121e-16, error 2.195e-05
stored float, checked      RejectedByAccept, corrections 2, fallback 1, relative residual 1.121e-16, error 0.000e+00
exact double residual      Converged, corrections 2, fallback 0, relative residual 1.258e-16, error 2.493e-12
useless correction (x0)    Stagnated, corrections 2, fallback 1, relative residual 5.922e-09, error 0.000e+00
weak correction (x0.2)     MaxCorrectionsReached, corrections 10, fallback 1, relative residual 7.948e-10, error 0.000e+00
```

行ごとに読むと次のようになる。

- *stored float matrix*（保存された float 行列）。残差を `float` の行列から計算する。ループは許容値をはるかに下回る残差まで収束する。ただし、解いたのは丸めた系である。倍精度の厳密な解との誤差は、まだ $2 \times 10^{-5}$ ある。
- *checked*（検査つき）。反復は同じだが、`accept` が厳密な行列で残差を計算し直す。検査は失敗し、状態は `RejectedByAccept` になり、`fallback` が `w` を厳密な倍精度の解で置き換える。学習済みモデルの `MixedPrecision<PromoteStorage>` が行うのは、これである。
- *exact double residual*（厳密な倍精度の残差）。残差を倍精度の行列から計算する。2 回の補正で誤差は $2.5\times10^{-12}$ になり、状態は `Converged` である。`MixedPrecision<ReevaluateKernel>` の考え方はこれにあたる。
- *useless / weak corrections*（役に立たない補正、弱い補正）。残差を減らさない補正は `Stagnated` で終わる。減り方が遅すぎる補正は 10 回の補正を使い切って `MaxCorrectionsReached` になる。どちらも厳密な求解に切り替わる。

```{literalinclude} /../examples/mixed_precision_refine.cpp
:language: cpp
:start-after: // [convenience]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
refine(): ||r||_inf = 1.521e-14, ||A||_inf = 2.459e+01, tolerance met: 1, error 2.493e-12
||float(A)||_inf = 2.459e+01, ||b - float(A) w||_inf = 1.858e-07
```

### ループと定数

`refine_in_place(system, w, residual_scratch)` は、最大 `max_refinement_corrections` 回、次を繰り返す。残差を計算し、収束を判定し、収束していなければ `system.correct` を適用する。連続する 2 ステップで残差のノルムが比 `refinement_stagnation_ratio` ほども減らなければ、停滞とみなして打ち切り、フォールバックする。定数は公開されている。

| 定数 | 値 | 意味 |
|---|---|---|
| `max_refinement_corrections` | 10 | フォールバックするまでの補正回数 |
| `refinement_stagnation_ratio` | 0.9 | $\lVert r_k\rVert_\infty / \lVert r_{k-1}\rVert_\infty$ がこの値を超えるステップは、停滞とみなす |
| `refinement_stagnation_steps` | 2 | フォールバックを引き起こす、停滞ステップの連続回数 |
| `refinement_column_block` | 32 | ライブラリがカーネルを倍精度で再評価するときの列ブロック幅 |
| `refinement_tolerance(dim)` | $10\,\text{dim}\,\varepsilon_{64}$ | 相対残差の停止許容値 |

収束判定は $\lVert r\rVert_\infty / (\lVert B\rVert_\infty \lVert w\rVert_\infty + \lVert b\rVert_\infty) < \text{tol}$ で、`meets_relative_residual(b_inf, w, b, residual, tol)` が評価する。非有限のオペランド、非正の許容値、分母が 0 の場合は、すべて不合格にする。`RefinementSummary` は、`status`（`RefinementStatus`）、`iterations`（停止までに適用した補正の回数）、`final_relative_residual`、`used_fallback()` を持つ。`used_fallback()` は、`Converged` 以外のすべての状態で真になる。`RefinementSummary` の `operator==` は、3 つのフィールドをそのまま比較する。

| `RefinementStatus` | 意味 |
|---|---|
| `Converged` | 残差判定に合格し、`accept` が `true` を返した |
| `RejectedByAccept` | 残差判定には合格したが、`accept` が `false` を返し、`fallback` が実行された |
| `Stagnated` | 連続する 2 ステップで残差が十分に減らず、`fallback` が実行された |
| `MaxCorrectionsReached` | `max_refinement_corrections` 回で収束せず、`fallback` が実行された |

エラーは次のとおり。`refine_in_place` は、`w` か右辺が空なら `EmptyInput`、大きさが違えば `LengthMismatch`、`residual_scratch` が `w` より短ければ `WorkspaceTooSmall` を返す。自分では何も確保しない。`refine(system, w)` は便利なラッパーで、`w` を値で受け取り、スクラッチを確保して、改良したベクトルを返す。

ヘルパーは次のとおり。

| 関数 | 用途 |
|---|---|
| `inf_norm(values)` | $\max_i \lvert v_i\rvert$。空のスパンなら 0、要素に `NaN` があれば `NaN` |
| `promoted_residual_into(a_f32, alpha, y, residual, row_abs_scratch, sum_scratch)` | `double` で累積し、利用者が持つスクラッチを使って $r = y - A_{f32}\alpha$ を求める。$\lVert A_{f32}\rVert_\infty$ を返す |
| `promoted_residual(a_f32, alpha, y, residual)` | 同じ計算を、内部でスクラッチを確保して行う |
| `meets_relative_residual(...)` | 上の収束判定 |
| `refinement_tolerance(dim)` | 許容値 $10\,\text{dim}\,\varepsilon_{64}$ |
| `symmetrize_lower(matrix)` | ビューの下三角を上三角へ鏡映する（前述） |

## API の一覧

上の表が、各関数を役割とともに列挙している。このページが扱うヘッダーごとのシンボルを、確認用に一覧にする。

| ヘッダー | シンボル |
|---|---|
| `linalg/cholesky.hpp` | `cholesky_lower_in_place`、`cholesky_lower_with_jitter`、`solve_lower_in_place`、`solve_lower_transpose_in_place`、`solve_llt_in_place`、`inv_diag_from_chol_l`、`log_det_from_l`、`half_log_det_from_l`、`solve_lower_in_place_unchecked`、`solve_lower_transpose_in_place_unchecked`、`solve_llt_in_place_unchecked` |
| `linalg/dense.hpp` | `add_to_diag`、`copy_lower_triangle`、`symmetrize_lower`、`frobenius_lower`、`dot`、`matvec_columns` |
| `linalg/parallel.hpp` | `lower_area`、`lower_block_start`、`openmp_max_threads`、`parallel_for_lower_blocks`、`parallel_for_lower_blocks_raw`、`parallel_lower_reduce_cols`、`parallel_lower_reduce_cols_raw` |
| `workspace/buffers.hpp` | `CholeskyBuffer`、`FitPolicy`、`HessianScratch`、`WorkspaceCore`、`FitBuffers`、`QueryWorkspace` |
| `workspace/mll.hpp` | `factor_and_mll`、`mll_and_gradient_into`、`hessian_into`、`value_gradient_hessian_into` |
| `workspace/inference.hpp` | `predict_into`、`predict`、`predict_covariance_into`、`predict_covariance`、`sample`、`loo_predict_into`、`loo_predict` |
| `workspace/leaf_cache.hpp` | `OptimizerHasChangeIndicesConcept`、`OptimizerTraits`、`optimizer_uses_change_indices_v`、`require_change_indices`、`LeafCache` |
| `precision/refine.hpp` | `max_refinement_corrections`、`refinement_stagnation_ratio`、`refinement_stagnation_steps`、`refinement_column_block`、`refinement_tolerance`、`inf_norm`、`symmetrize_lower`、`meets_relative_residual`、`promoted_residual_into`、`promoted_residual`、`RefinementStatus`、`RefinementSummary`、`RefineSystem`、`refine_in_place`、`refine` |
