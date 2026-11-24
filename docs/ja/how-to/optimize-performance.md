# 性能を最適化する

このガイドは、予測でヒープを使わない方法、スレッド数の制御、精度ポリシーの選び方、学習時のメモリと速度の兼ね合いを説明する。動くモデルがすでにあることを前提とする。`Gpr` のオプション一覧は {doc}`configure-fit`、ライブラリがこう振る舞う理由は {doc}`../explanation/determinism-and-performance` を参照。

:::{note}
このページの数値は、Release ビルドのライブラリ（MSVC、既定の AVX2 コード生成）を、16 ハードウェアスレッドの x86-64 デスクトップ機で測ったものである。値は繰り返し測定の中央値で、実行ごとに最大 2 割ほどばらついた。桁を把握する目安として読み、判断の前に自分のワークロードで測ること。
:::

## 確保なしで予測する

`FittedGpr` は `QueryWorkspace` を 1 つ持つ。予測に必要な一時領域（変換後のクエリ、交差共分散、三角求解の結果、カーネルの入れ子スクラッチ）はすべてここにある。`FittedGpr::predict_into` と `FittedGpr::predict_with_into` は、利用者が持つ `Prediction` に結果を書き込む。ワークスペースと出力ベクトルの大きさを決めるウォームアップを 1 回行えば、同じか小さいバッチサイズの呼び出しはヒープを確保しない。

次のプログラムは、確保回数を数えるグローバル `operator new` を差し込み（ライブラリは静的ライブラリなので、ライブラリ内の確保もこの `operator new` を通る）、2 特徴量のモデルを学習して確保回数を数える。

```{literalinclude} /../examples/zero_alloc_predict.cpp
:language: cpp
:start-after: // [counter]
:end-before: // [end]
```

ウォームアップを 1 回行い、同じ `Prediction` を使って繰り返し予測する。

```{literalinclude} /../examples/zero_alloc_predict.cpp
:language: cpp
:start-after: // [warmup]
:end-before: // [end]
:dedent: 4
```

```{literalinclude} /../examples/zero_alloc_predict.cpp
:language: cpp
:start-after: // [steady]
:end-before: // [end]
:dedent: 4
```

プログラム全体の出力は次のとおり。

```{code-block} text
warm-up call allocated 8 times
1000 further predict_into calls allocated 0 times
one predict() call allocated 2 times
predict_into equals predict: 1
checksum of the 64-point prediction: edb08ca7add4082b
larger batch (256 points): first call 8 allocations, second call 0
```

`predict` は新しい `Prediction` を返すので、呼ぶたびに 2 本の結果ベクトルを確保する。`predict_into` と `predict` は同じコードを走らせ、ビット単位で同じ数値を返す。

### 確保なし予測の規則

1. **`Prediction` を 1 つ使い回す。** `mean` と `variance` の容量が呼び出し間で残る。
2. **バッチサイズは自由に変えてよい。** ワークスペースは増えるだけである。大きなバッチでは 1 回だけ確保が起き（上の最終行）、小さなバッチは既存の領域を再利用し、以後は確保なしになる。
3. **分散の種類は `PredictOptions` で選ぶ。** `predict_into(xs, out, options)` と `predict_with_into(xs, options, out)` は `PredictOptions` を受け取る。`VarianceKind::Latent` は分散から観測ノイズを除き、`VarianceKind::Observation`（既定）は含める。
4. **スレッドごとに `FittedGpr` を 1 つ持つ。** 下の警告を参照。

:::{warning}
`predict_into` は `const` メンバ関数だが、モデル内部の `QueryWorkspace` を書き換える。同じ `FittedGpr` に対して、複数のスレッドが同時に `predict`、`predict_into`、`predict_covariance`、`sample` を呼んではならない。代わりにモデルをコピーする。コピーコンストラクタは、コピー側に空の `QueryWorkspace` を持たせる。ワーカースレッドごとに独立したオブジェクトを渡せる。
:::

:::{note}
上のカウンタは、グローバル `operator new` の呼び出しをすべて数える。`std::vector` の増加はこれで捉えられる。ライブラリ内部で Eigen 自身のアラインドアロケータが行う確保は数えない。プロジェクトのテストは両方を監査している。予測の実行中に Eigen の実行時確保を禁止し、`operator new` も数える。したがって、このカウンタは利用者側コードの回帰検査としては有効だが、Eigen 内部についての証明にはならない。
:::

### 低レベルでの同等の呼び出し

因子と重みを自分で管理するなら、自由関数 `predict_into<P>` が、自分で作った `QueryWorkspace<P>` を受け取る。`predict_covariance_into<P>` と `loo_predict_into<P>`（こちらは `QueryWorkspace` ではなく 2 つのスクラッチを取る）も同様である。{doc}`use-low-level-building-blocks` にすべての例がある。

## どのバッファがあるかを知る

cxxgpr は一時領域を、名前のついた少数のワークスペース型に置く。持ち主を知れば、メモリがいつ確保されいつ解放されるかがわかる。

| 型 | 持ち主 | 確保 | 解放 |
|---|---|---|---|
| `FitBuffers<P>` | `Gpr::fit` / `Gpr::factor` の中の学習処理 | 学習の開始時に、`n` 点分を 1 回で確保する | `fit` または `factor` の復帰時。`FittedGpr` は保持しない |
| `WorkspaceCore<P>` | `FitBuffers<P>::core` | `FitBuffers` と同時。`kernel_scratch` は遅延確保 | `FitBuffers` と同時 |
| `HessianScratch<S>` | `WorkspaceCore::hessian` | 最初のヘッセ行列計算（`ensure`）で確保し、以後は再利用する | `FitBuffers` と同時 |
| `NestedBuffers<S>` | `WorkspaceCore::nested` と `weighted`、`QueryWorkspace::query_nested` | カーネル木の深さから大きさを決める | 持ち主と同時 |
| `LeafCache<S>` | 変更座標を報告する最適化器の学習目的関数 | 学習の開始時に、カーネル葉ごとに 1 つのグラム行列 | `fit` の復帰時 |
| `QueryWorkspace<P>` | `FittedGpr` | 最初の予測で確保し、必要に応じて拡張する | `FittedGpr` の破棄時 |

この表から 3 つの規則が導かれる。

- **学習用のメモリは一時的である。** `fit` が戻ると、残るのは学習済みモデル、つまり因子 $L$、重み $\alpha$、学習データ、クエリ用ワークスペースだけである。
- **予測用のメモリは常駐するが、上限がある。** これまでに予測した最大バッチの大きさまで増え、縮むことはない。
- **`FittedGpr` のコピーは空の `QueryWorkspace` から始まる。** コピーの最初の予測では、再び確保が起きる。

`Gpr::factor`（最適化なし）は、`with_prefer_speed()` の設定にかかわらず、常にメモリ節約側のバッファポリシーで学習する。関数値だけを使う最適化器（`NelderMead` と `FastSimulatedAnnealing`）は勾配の重み行列を作らないので、やはり `CholeskyBuffer::Reuse` で学習する。`FastSimulatedAnnealing` では、`CholeskyBuffer::Retain` は増分 `LeafCache` を使うかどうかだけを決める。

## スレッド数を制御する

cxxgpr は密行列の計算を OpenMP で並列化する。`Gpr` の API にスレッド数の設定はない。ライブラリは OpenMP ランタイムの設定に従う。

```{code-block} sh
# Linux、macOS、Git Bash
OMP_NUM_THREADS=4 ./my_program
```

```{code-block} powershell
# Windows PowerShell
$env:OMP_NUM_THREADS = 4
.\my_program.exe
```

プログラムの中からは、学習や予測の前に `<omp.h>` の `omp_set_num_threads(k)` を呼ぶ。そのためには、自分のターゲットも OpenMP つきでコンパイルしなければならない（MSVC は `/openmp`、GCC と Clang は `-fopenmp`、CMake なら `find_package(OpenMP)` と `OpenMP::OpenMP_CXX`）。`linalg::openmp_max_threads()` は、ライブラリがいま使うスレッド数を返す（OpenMP なしでビルドされていれば 1）。`linalg::parallel_*` ヘルパーは、明示的な `num_threads` 引数を取る（0 は「OpenMP に任せる」）。

小さな問題では、ライブラリは直列のまま動く。複数のスレッドが使え、かつ行列が並列化の費用に見合う大きさのときだけ、並列領域を開く。学習点が数十点なら、スレッドを増やしても何も変わらない。

同じマシンで、負の対数周辺尤度の勾配計算 1 回（`n = 1000`、特徴量 4 個）と、`factor` 1 回（`n = 2000`）を測った。

| スレッド数 | `mll_and_gradient_into` | `Gpr::factor` |
|---|---|---|
| 1 | 約 33 ms | 約 200 ms |
| 16 | 約 7.6 ms | 約 32 ms |

結果はスレッド数に依存しない。上の例は、L-BFGS による学習を含めて、どの設定でも同じチェックサムを出力する。

```{code-block} text
$ OMP_NUM_THREADS=1  ./zero_alloc_predict | grep checksum
checksum of the 64-point prediction: edb08ca7add4082b
$ OMP_NUM_THREADS=3  ./zero_alloc_predict | grep checksum
checksum of the 64-point prediction: edb08ca7add4082b
$ OMP_NUM_THREADS=16 ./zero_alloc_predict | grep checksum
checksum of the 64-point prediction: edb08ca7add4082b
```

この性質を実現する仕組みは {doc}`../explanation/determinism-and-performance` で説明する。自分の並列ループから cxxgpr を呼ぶ場合は、スレッドごとに `FittedGpr` のコピーを持たせる。また、入れ子の並列領域がそれぞれ `OMP_NUM_THREADS` 個のスレッドを起こさないように、OpenMP ランタイムの設定を確かめる。そうしないと、マシンの能力を超えてスレッドを走らせてしまう。

## 精度ポリシーを選ぶ

`Gpr` と `FittedGpr` は、精度を第 2 テンプレート引数で受け取る（`Gpr<optimizer::Lbfgs, SinglePrecision>`）。ポリシーは 4 つある。`DoublePrecision`、`SinglePrecision`、`MixedPrecision<PromoteStorage>`、`MixedPrecision<ReevaluateKernel>` である。

次のプログラムは、同じ悪条件の系（滑らかな信号、ノイズ分散 `2e-5`、200 点）を、固定のハイパーパラメータですべてのポリシーで因子分解する。違いは演算の精度だけである。

```{literalinclude} /../examples/precision_policies.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
policy                    factor bytes   alpha error  mean error   var error  refinement
DoublePrecision                 320000      0.00e+00    0.00e+00    0.00e+00  none
SinglePrecision                 160000      5.17e-02    1.11e-03    1.01e-07  none
Mixed<PromoteStorage>           160000      1.32e-02    3.83e-04    1.13e-07  Converged, 4 corrections
Mixed<ReevaluateKernel>         160000      1.72e-06    4.93e-04    1.13e-07  Converged, 4 corrections
Double + FastApprox             320000      4.03e-04    1.37e-05    2.25e-10  none
```

「alpha error」は、予測重みが倍精度の重みからずれた最大値を、重みの最大要素で割った相対値である。「mean error」と「var error」は、50 個のテスト点での予測の差の最大絶対値である。

数値から読み取れることは次のとおり。

- `SinglePrecision` は因子のメモリを半分にし、下の表では最も速いポリシーである。ただし、この悪条件の問題では重みが 5 % ずれる。
- `MixedPrecision` は単精度で因子分解し、倍精度の残差で重みを反復改良する。この問題では、改良後の重みは約 $10^{-6}$（`ReevaluateKernel`）になる。`PromoteStorage` は、`float` で保存した系を解くので、単精度より約 4 倍近いだけである（下記）。ただし**予測**の改善は 2 倍程度にとどまる。交差共分散 $K_*$ が単精度で評価されるためである。
- `PromoteStorage` は、保存された単精度行列に対して反復改良する。重みは、*その*系を倍精度の精度で解くが、行列を `float` に丸めた影響（1 要素あたり約 $10^{-7}$ を、条件数が増幅する）は残る。ライブラリは、厳密な倍精度カーネルに対する残差が `std::numeric_limits<float>::epsilon()` の 10 倍以内であれば受理する。許容値 $10\,n\,\varepsilon_{64}$ は要求しない。`float` に丸めたカーネル行列は、全要素が `float` で厳密に表せる場合を除いて、この許容値を満たさないからである。結果は `FittedGpr::refinement_summary()` で確認できる。
- `ReevaluateKernel` は、残差を倍精度の厳密なカーネルで列ブロックごとに検査し、倍精度の解に収束する。

`Gpr<optimizer::Fixed, P>::factor` の所要時間（3 回の中央値、特徴量 4 つ、Release ビルド、16 スレッドの x86-64 デスクトップ、実行ごとのばらつきは最大で約 10 %）を、16 スレッドでミリ秒で示す。

| `n` | `DoublePrecision` | `SinglePrecision` | `MixedPrecision<PromoteStorage>` | `MixedPrecision<ReevaluateKernel>` |
|---|---|---|---|---|
| 500 | 1.7 | 1.1 | 4.4 | 6.2 |
| 1000 | 6.4 | 4.3 | 13 | 18 |
| 2000 | 28 | 17 | 53 | 70 |
| 4000 | 163 | 83 | 156 | 135 |
| 6000 | 473 | 235 | 472 | 408 |

1 スレッドでは、`n = 2000` が 91 ms（倍精度）、53 ms（単精度）、87 ms（`PromoteStorage`）、99 ms（`ReevaluateKernel`）。`n = 4000` は 656、323、524、521 ms である。

混合精度のポリシーがこう振る舞う理由は次のとおり。

- 反復改良の各ステップは、倍精度で残差を評価する。収束までに、この残差が 2 回（出発点の残差と、収束を確かめる残差）要る。`ReevaluateKernel` では、各残差が厳密なカーネルを、列ブロックごとに、三角ではなく行列全体について評価し直すので、倍精度の経路のカーネル評価の数倍のコストになり、そこに単精度の分解が加わる。`PromoteStorage` では、残差は保存した `float` 行列を読むだけで、最後の受理の検査だけが厳密なカーネルを 1 回評価する。分解は $O(n^3)$ の計算を節約し、反復改良は $O(n^2)$ を要するので、`n` が大きくなるほど、どちらのポリシーも `DoublePrecision` を追い越す。この環境では、おおよそ `n = 3000` 未満では遅く、それ以上では `PromoteStorage` が同程度、`ReevaluateKernel` が約 1.2 倍速い。
- どちらの混合精度も `SinglePrecision` より速くはない。単精度に対する利点は重みの精度、倍精度に対する利点は半分の大きさの因子である。

メモリが制約でなければ `DoublePrecision`、誤差を許せるなら `SinglePrecision`、単精度の因子の保存と、厳密な系に対して検証した重みが要る場合や、`n` が大きく倍精度より速くしたい場合は `MixedPrecision<ReevaluateKernel>` を選ぶ。`MixedPrecision<PromoteStorage>` は単精度と `ReevaluateKernel` の中間にあたる。カーネルの繰り返しの評価は避けられるが、重みには保存時の丸めが残る。反復改良のアルゴリズムは {doc}`../explanation/numerical-stability`、自分で動かす方法は {doc}`use-low-level-building-blocks` を参照。

## 学習時にメモリと速度を取引する

学習のメモリ使用量は 2 つの列挙型が決める。`FitPolicy` がこの 2 つをまとめる。設定は `Gpr::with_distance_cache_policy`、`Gpr::with_cholesky_buffer`、`Gpr::with_prefer_speed`、`Gpr::with_prefer_memory` で行う（{doc}`configure-fit` を参照）。

| 設定 | 意味 | `n` 点での追加メモリ |
|---|---|---|
| `DistanceCachePolicy::Cached` | 学習点間の距離を 1 回だけ計算し、毎回の反復で再利用する | 等方カーネルで $n^2$ 個、ARD カーネルで $d \cdot n(n+1)/2$ 個、距離を読まないカーネルでは 0 |
| `DistanceCachePolicy::Uncached` | 毎回、座標から距離を計算し直す | なし |
| `CholeskyBuffer::Retain` | $L$ を残したまま、勾配の重み行列 $W$ を別のバッファに作る | $n \times n$ 行列 1 つ |
| `CholeskyBuffer::Reuse` | $W$ を因子の上に作り、後で $L$ を再分解する | なし。代わりにコレスキー分解が 1 回増える |

`FitPolicy::prefer_speed()` は `Cached` と `Retain` で、既定値である。`FitPolicy::prefer_memory()` は `Uncached` と `Reuse` である。

`mll_and_gradient_into` 1 回の時間を測った（`n = 1000`、特徴量 4 個、ARD カーネル、16 スレッド）。

| ポリシー | 時間 | 大きなバッファ |
|---|---|---|
| `Cached` + `Retain`（`prefer_speed`） | 約 7.6 ms | $K$、スクラッチ、$W$ で 24 MB と、距離キャッシュ |
| `Uncached` + `Reuse`（`prefer_memory`） | 約 10 ms | 16 MB、距離キャッシュなし |
| `Cached` + `Reuse` | 約 10〜14 ms | |
| `Uncached` + `Retain` | 約 6〜10 ms | |

時間がかかる原因は `Reuse` である（勾配計算のたびに最後に $L$ を再分解する）。距離キャッシュの影響はここでは小さかった。メモリを節約する必要があるなら、まず `Uncached` と `Retain` の組み合わせを選び、測定する。

低レベルのバッファにも同じ切り替えがある。`FitBuffers<P>::create(n, cache, buffer)` または `FitBuffers<P>::create(n, policy)` を使う。`overwrites_cholesky()` は、勾配計算が $L$ を壊して再分解するかどうかを返す。{doc}`use-low-level-building-blocks` を参照。

## 高速な指数関数を使う

`Gpr::with_kernel_exp(KernelExp::FastApprox)` は、組込みのカーネル葉の `std::exp` を `herumi/fmath` の高速近似指数計算に置き換える。`exp(log theta)` のパラメータ変換と `CustomKernel` の葉には影響しない。

上の表の `Double + FastApprox` の行が、その代償を示す。予測は正確な値から $1.4 \times 10^{-5}$、重みは $4 \times 10^{-4}$ ずれる（この悪条件の問題での値）。`n = 1000` の勾配計算は速くならなかった（約 7.5〜8.4 ms に対し、正確版は約 7.6 ms）。コレスキー分解が支配的だからである。`FastApprox` が効くのは、カーネル評価が仕事の大きな割合を占める場合だけである。学習点に対して予測点が多い場合や、分解の安い低次元の問題がこれに当たる。切り替える前に測り、そうでなければ `KernelExp::Accurate` のままにする。

## API の一覧

| シンボル | 役割 |
|---|---|
| `FittedGpr::predict_into`、`FittedGpr::predict_with_into` | 利用者が持つ `Prediction` に予測を書く。ウォームアップ後は確保なし |
| `Prediction`、`PredictOptions`、`VarianceKind` | 結果の入れ物と、分散の選択（`Latent`、`Observation`） |
| `QueryWorkspace<P>` | 予測時のバッファ。`ensure` で大きさを決める |
| `FitBuffers<P>`、`WorkspaceCore<P>`、`HessianScratch<S>` | 学習用バッファと、遅延確保されるヘッセ行列用の部分 |
| `FitPolicy`、`CholeskyBuffer`、`DistanceCachePolicy` | メモリと速度のポリシー。プリセットは `prefer_speed` と `prefer_memory` |
| `Retain`、`Reuse` | `CholeskyBuffer` の列挙子 |
| `Cached`、`Uncached` | `DistanceCachePolicy` の列挙子 |
| `NestedBuffers<S>` | 木の評価用に、段ごとの行列を連続して持つ領域 |
| `LeafCache<S>`、`OptimizerTraits` | 1 座標ずつ動かす最適化器のための、葉ごとの増分グラム行列 |
| `KernelExp` | `Accurate` か `FastApprox` の指数関数 |
| `DoublePrecision`、`SinglePrecision`、`MixedPrecision`、`PromoteStorage`、`ReevaluateKernel` | 精度ポリシーと残差の式 |
| `MixedPrecision::Residual` | 混合ポリシーの残差の式（`PromoteStorage` か `ReevaluateKernel`）を指す別名 |
| `RefinementSummary`、`RefinementStatus` | 学習済みモデルの混合精度反復改良の結果 |
| `linalg::openmp_max_threads` | 現在の OpenMP スレッド数 |
