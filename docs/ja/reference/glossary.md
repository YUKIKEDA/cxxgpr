# 用語集

このドキュメント全体で使う用語と記号の定義を集めた。見出し語は五十音順ではなく、英語の用語のアルファベット順に並べている。各項目は、詳しい説明の場所を示す。

## 記法

| 記号 | 意味 |
|---|---|
| $n$ | 学習点の数 |
| $d$ | 特徴量の数（$X$ の列数） |
| $m$ | クエリ点の数 |
| $X$、$y$ | 学習入力（$n \times d$、列優先）と学習目的変数（長さ $n$） |
| $K$ | 学習入力でのカーネルのグラム行列。$K_{ij} = k(x_i, x_j)$ |
| $\sigma_n^2$ | 観測ノイズの分散 |
| $j$ | 因子分解が失敗したときに対角へ足すジッター |
| $A$ | 系の行列 $K + \sigma_n^2 I$（ジッターを含めると $K + (\sigma_n^2 + j) I$） |
| $L$ | 下三角のコレスキー因子。$A = L L^\top$ |
| $\alpha$ | 予測重み。$\alpha = A^{-1} y$ |
| $K_*$ | 学習点とクエリ点の間の交差共分散（$n \times m$） |
| $\theta$ | 対数空間のハイパーパラメータのベクトル。すべてのカーネルパラメータに続いて $\log\sigma_n^2$ |
| $p$ | $\theta$ の長さ |
| $W$ | 勾配の重み行列 $\alpha\alpha^\top - A^{-1}$ |
| $\varepsilon_{64}$、$u_{32}$ | `double` の機械イプシロン（$2^{-52}$）と、`float` の単位丸め誤差（$\approx 6 \times 10^{-8}$） |
| $\kappa(A)$ | $A$ の条件数 |

## 用語

Accurate（正確な指数関数）
: `KernelExp::Accurate`。カーネル評価の中で、標準ライブラリの `std::exp` を使う設定で、既定値である。{doc}`../how-to/optimize-performance` を参照。

ARD（automatic relevance determination、自動関連度決定）
: 特徴量ごとに 1 つの長さスケールを持つカーネル（`RbfArdKernel`、`MaternArdKernel`、`RationalQuadraticArdKernel`）。学習された長さスケールが大きいほど、その特徴量はほとんど効かない。{doc}`../how-to/build-kernels` を参照。

バッチ
: 1 回の予測呼び出しに渡すクエリ点の集合。バッチの大きさが `QueryWorkspace` の大きさを決める。

コレスキー分解
: 対称正定値行列 $A$ を、下三角行列とその転置の積 $A = L L^\top$ に分解すること。厳密なガウス過程回帰の中心となる演算で、計算量は $O(n^3)$ である。{doc}`../explanation/numerical-stability` を参照。

列優先（column-major）
: 列の要素がメモリ上で隣り合う格納順序。$n$ 行の行列の要素 $(i, j)$ は添字 $jn + i$ にある。cxxgpr の行列はすべて列優先である。{doc}`../explanation/design-principles` を参照。

コンパイル済みカーネル（compiled kernel）
: `CompiledKernel<T>`。`KernelSpec` の実行用の形で、平坦化され、`float` か `double` に特殊化されている。数値計算はすべてこれで行う。{doc}`../how-to/build-kernels` を参照。

条件数
: 対称正定値行列の、最大固有値と最小固有値の比。値が大きいと、入力の小さな誤差が出力で大きな誤差になり、単精度では精度が落ちる。

交差共分散
: すべての学習点とすべてのクエリ点の間のカーネル値の行列 $K_*$。

距離キャッシュ
: 学習点間の二乗距離を 1 回だけ計算し、最適化器の毎回の反復で再利用するもの（`DistanceCachePolicy::Cached`）。{doc}`../how-to/optimize-performance` を参照。

Exact GPR（厳密なガウス過程回帰）
: $n \times n$ の共分散行列全体を、近似なしで因子分解するガウス過程回帰。cxxgpr が実装するのは、このモデルだけである。{doc}`../explanation/relationship-to-gprx` を参照。

因子（factor）
: 名詞では、コレスキー因子 $L$ のこと。API の動詞（`Gpr::factor`）では、ハイパーパラメータを最適化せず、現在の値のままモデルを学習することを指す。

FastApprox
: `KernelExp::FastApprox`。組込みカーネルの `std::exp` を、範囲縮約つきの 7 次多項式に置き換える設定。1 回あたりは速いが、精度は相対で約 $10^{-5}$ である。

フォールバック（反復改良の）
: 混合精度の反復改良が収束しないとき、または結果が拒否されたときに、系を倍精度で解き直すこと。{doc}`../explanation/numerical-stability` を参照。

学習済みモデル（fitted model）
: `FittedGpr`。予測、サンプリング、1 個抜き予測、保存ができる、学習済みのモデル。未学習のトレーナー `Gpr` とは別の型である。

グラム行列
: すべての学習点の組のカーネル値からなる行列 $K$。

ヘッセ行列
: $\theta$ に関する負の対数周辺尤度の 2 階微分の行列。`hessian_into` が解析的に計算する。`TrustRegion` が使う。

ハイパーパラメータ
: カーネルやノイズのパラメータ（長さスケール、信号分散、周期、ノイズ分散）。cxxgpr は、ハイパーパラメータを対数スケールで最適化する。{doc}`../how-to/tune-hyperparameters` を参照。

区間（Interval）
: ハイパーパラメータが内側にとどまらなければならない開区間 `(lo, hi)`。`Interval::default_positive()` は $(10^{-5}, 10^{5})$ である。

反復改良（iterative refinement）
: 近似解の残差を、より高い精度で繰り返し計算し、安価な因子で補正量を求めて、線形系の解を改善すること。{doc}`../explanation/numerical-stability` を参照。

ジッター
: コレスキー分解が失敗したときに、対角へ足す小さな正の数。数値上の手段であり、観測ノイズとは別のものである。{doc}`../explanation/numerical-stability` を参照。

カーネル
: 関数値の間の事前共分散を定める、半正定値の関数 $k(x, x')$。共分散関数とも呼ぶ。{doc}`../explanation/kernels` を参照。

カーネル葉（kernel leaf）
: 10 種類の組込みカーネル（または `CustomKernel`）の 1 つで、`KernelSpec` の式の木の構成要素。

KernelSpec
: 葉を `+` と `*` で組み合わせた木として、カーネルを宣言的に記述する不変のオブジェクト。評価の前に `CompiledKernel` になる。

潜在分散（latent variance）
: ノイズを含まない関数 $f(x_*)$ の予測分散（`VarianceKind::Latent`）。観測分散は、これに $\sigma_n^2$ を加えたものである。

1 個抜き交差検証（leave-one-out、LOO）
: 各学習目的変数を、残りのすべてから予測すること。$L$ と $\alpha$ から閉形式で得られる（`loo_predict`）。

長さスケール（length scale）
: 関数値の相関が保たれる距離。RBF、Matern などのカーネルの主要なハイパーパラメータ。

対数周辺尤度
: 入力とハイパーパラメータが与えられたときの、目的変数の確率の対数。cxxgpr は、その符号を反転したものを最小化する（最適化器の目的関数）。{doc}`../explanation/gaussian-processes` を参照。

混合精度
: `MixedPrecision<R>`。`float` で因子分解し、`float` の行列を保持し、重みを `double` で反復改良する。残差の式 `R` は、`PromoteStorage` か `ReevaluateKernel` である。

NestedBuffers
: カーネル木の再帰的な評価が確保なしで済むように、事前確保しておく行列のまとまり。

ノイズ分散
: ガウス観測ノイズの分散 $\sigma_n^2$。`GaussianLikelihood` が持ち、カーネルパラメータと同じように最適化される。

観測分散（observation variance）
: 新しいノイズ込みの観測の予測分散。潜在分散に $\sigma_n^2$ を加えたもの（`VarianceKind::Observation`、既定）。

OpenMP
: ライブラリがループの並列化に使う、ディレクティブベースのスレッドの標準。`OMP_NUM_THREADS` で制御する。{doc}`../how-to/optimize-performance` を参照。

最適化器（optimizer）
: 負の対数周辺尤度を最小化するアルゴリズム。`Lbfgs`、`NelderMead`、`TrustRegion`、`FastSimulatedAnnealing`、`Fixed`。{doc}`../how-to/choose-optimizer` を参照。

事後分布（posterior）
: 学習データで条件づけた後の、新しい点での関数値の分布。予測平均と予測共分散を持つガウス分布。

精度ポリシー（precision policy）
: 記憶精度と改良精度のスカラー型を選ぶ型。`DoublePrecision`、`SinglePrecision`、`MixedPrecision<PromoteStorage>`、`MixedPrecision<ReevaluateKernel>`。

予測重み
: 交差共分散を予測平均へ変えるベクトル $\alpha = A^{-1}y$。

クエリ点
: モデルを評価する点（テスト点ともいう）。

Refine 型
: 精度ポリシーでの、重みと予測のスカラー型（`P::Refine`）。`DoublePrecision` と `MixedPrecision` では `double`、`SinglePrecision` では `float`。

残差（residual）
: 線形系 $Bw = b$ と反復解 $w$ に対する、ベクトル $r = b - Bw$。

停滞（stagnation）
: 反復改良が進まなくなること。連続する 2 ステップで、残差のノルムが 0.9 倍より小さくならない状態。

標準化（standardization）
: 平均を引き、標準偏差で割る変換（`StandardizeInput`、`StandardizeTarget`）。{doc}`../how-to/transform-data` を参照。

Storage 型
: 精度ポリシーでの、行列とコレスキー因子のスカラー型（`P::Storage`）。`double` か `float`。

供給された距離（supplied distances）
: 座標から計算するのではなく、呼び出し側が与える二乗距離。`DistanceKernel` で使う。{doc}`../how-to/use-supplied-distances` を参照。

学習用バッファ（training buffers）
: モデルの学習中だけ存在する、一時的なワークスペース（`FitBuffers`）。

ビュー（view）
: メモリを所有せずに指す窓。`MatrixRef<T>`（読み取り専用）と `MatrixMut<T>`（書き込み可）。ポインタ、形状、列のストライドで記述される。

ウォームアップ（warm-up）
: 再利用するワークスペースの大きさを決める、最初の呼び出し。同じ形の以降の呼び出しは、何も確保しない。

ワークスペース（workspace）
: 呼び出しの間で再利用する、事前確保のバッファのまとまり。学習用の `FitBuffers` と、予測用の `QueryWorkspace`。
