# ガウス過程回帰

cxxgpr が当てはめるモデルの数学を説明する。ガウス過程とは何か、データで条件付けるとなぜ誤差の幅つきの予測が得られるのか、ハイパーパラメータをどう選ぶのかを扱う。ライブラリが評価する式を導き、それぞれをコードと対応づける。標準的な参考書は Rasmussen と Williams の *Gaussian Processes for Machine Learning*（GPML）で、記号もそれに従う。

## 関数についての推論としての回帰

$n$ 個の組 $(x_i, y_i)$ を観測したとする。入力 $x_i \in \mathbb{R}^d$ を行列 $X$ に、目的変数をベクトル $\mathbf{y}$ にまとめる。目的変数は入力の関数であり、測定ノイズを通して見えていると考える。

$$
y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2).
$$

通常の回帰は、$f$ にパラメトリックな族を仮定し、そのパラメータを当てはめる。ガウス過程（GP）は、*関数*そのものに確率分布を与える。ガウス過程は確率変数の集まりで、その任意の有限個の部分集合が同時ガウス分布に従う。平均関数と共分散関数（**カーネル**）で完全に決まる。

$$
f \sim \mathcal{GP}\bigl(m(\cdot),\, k(\cdot, \cdot)\bigr), \qquad
m(x) = \mathbb{E}[f(x)], \quad
k(x, x') = \operatorname{Cov}\bigl(f(x), f(x')\bigr).
$$

cxxgpr は事前平均をゼロにする。実用上、これは制約にならない。目的変数の変換 `StandardizeTarget` が、学習の前にデータを平均ゼロへずらし、予測の後で元へ戻すからである。

任意の入力 $X$ に対し、関数値のベクトルは多変量正規分布 $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, K)$ からの標本になる。ここで $K_{ij} = k(x_i, x_j)$ である。仮定はすべてカーネルに込められる。$f$ がどれほど滑らかか、どれほど速く変化するか、周期的かどうか、といったことである。使えるカーネルとその組み合わせ方は {doc}`kernels` で扱う。

## 事後分布：データで条件付ける

$X_*$ を $m$ 個のクエリ入力、$\mathbf{f}_*$ をそこでの関数値とする。観測した目的変数と未知の関数値は、同時ガウス分布に従う。

$$
\begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix}
\sim \mathcal{N}\!\left(\mathbf{0},
\begin{bmatrix} K + \sigma_n^2 I & K_* \\ K_*^\top & K_{**} \end{bmatrix}\right).
$$

ただし $K = k(X, X)$、$K_* = k(X, X_*)$（$n \times m$ 行列）、$K_{**} = k(X_*, X_*)$ である。同時ガウス分布の一方のブロックで条件付けるとやはりガウス分布になるので、予測分布は閉じた式で得られる。

$$
\mathbf{f}_* \mid X, \mathbf{y}, X_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)
$$

ここで

$$
\boldsymbol{\mu}_* = K_*^\top \boldsymbol{\alpha}, \qquad
\boldsymbol{\alpha} = (K + \sigma_n^2 I)^{-1} \mathbf{y},
$$

$$
\Sigma_* = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*.
$$

言葉で読むと、こうなる。平均は、訓練入力を中心とするカーネル関数を、重み $\boldsymbol{\alpha}$ で足し合わせたものである。共分散は、事前共分散から、データが取り除いた不確かさの分を引いたものである。どの訓練点からも遠い場所では $K_*$ がほぼゼロになり、平均は事前平均（ゼロ）へ、分散は事前分散 $k(x_*, x_*)$ へ戻る。{doc}`../tutorials/multi-dimensional-regression` でモデルが外挿したとき、まさにこの振る舞いを見ることになる。

### 潜在分散と観測分散

$\Sigma_*$ は、ノイズのない関数 $f$ についての不確かさである。$x_*$ で新しく*測定*すれば、その上に独立なノイズが乗るので、分散は $\sigma_n^2$ だけ大きくなる。

$$
\operatorname{Var}[y_*] = \operatorname{Var}[f_*] + \sigma_n^2.
$$

cxxgpr は両方を `VarianceKind::Latent` と `VarianceKind::Observation` で提供する。`predict` の既定は `Observation` である。背後にある関数の誤差の幅がほしいときは `Latent`、これから測る値を予測したいときは `Observation` を使う。

### コレスキー因子による計算

$(K + \sigma_n^2 I)^{-1}$ を陽に作ることはしない。この行列は対称正定値なので、下三角行列 $L$ によるコレスキー分解 $K + \sigma_n^2 I = L L^\top$ を持つ。すると

$$
\boldsymbol{\alpha} = L^{-\top} (L^{-1} \mathbf{y}), \qquad
V = L^{-1} K_*, \qquad
\boldsymbol{\mu}_* = K_*^\top \boldsymbol{\alpha}, \qquad
\Sigma_* = K_{**} - V^\top V
$$

となる。各 $L^{-1}(\cdot)$ は三角行列の求解である。当てはめの計算量は、分解で $O(n^3)$ の時間、$K$ と $L$ で $O(n^2)$ のメモリである。その後 $m$ 点で予測する計算量は、平均が $O(n m)$、分散が $O(n^2 m)$ で、分散には三角求解が要る。因子 $L$ と重み $\boldsymbol{\alpha}$ は、まさに `FittedGpr` が保持し、`persist::save_with_factor` がディスクへ書くものである。$n$ が大きいと、この 3 乗のコストが厳密な GP 回帰の最大の制約になる。近似的なスパースモデルが存在する理由もそこにある。cxxgpr が実装するのは厳密なモデルだけである。

```{mermaid}
flowchart LR
    D["data X, y"] --> K["kernel matrix<br/>K + noise I"]
    K --> C["Cholesky<br/>L L^T"]
    C --> A["weights alpha"]
    C --> V["solve L V = K*"]
    A --> M["mean = K*^T alpha"]
    V --> S["variance = K** - V^T V"]
```

## ハイパーパラメータを選ぶ

カーネルにはパラメータがある。長さスケール、信号分散、周期などである。ノイズ分散 $\sigma_n^2$ もその 1 つで、これらをまとめたものがハイパーパラメータのベクトル $\boldsymbol{\theta}$ になる。GPML は、**周辺尤度**を最大化して $\boldsymbol{\theta}$ を選ぶ。周辺尤度とは、関数値を積分で消した、観測された目的変数の確率である。$\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K_\theta + \sigma_n^2 I)$ なので、その対数は

$$
\log p(\mathbf{y} \mid X, \boldsymbol{\theta})
= -\tfrac{1}{2}\, \mathbf{y}^\top \boldsymbol{\alpha}
  \;-\; \tfrac{1}{2} \log \lvert K_\theta + \sigma_n^2 I \rvert
  \;-\; \tfrac{n}{2} \log 2\pi
$$

である。コレスキー因子を使えば行列式は安く求まる。$\log \lvert K_\theta + \sigma_n^2 I \rvert = 2 \sum_i \log L_{ii}$ である。3 つの項の意味は明快である。第 1 項は、モデルがデータをどれだけよく説明するかを測る。第 2 項はモデルの複雑さに罰を与える。柔軟すぎるカーネルは確率を多くのありうるデータセットに分散させるので、行列式が大きくなる。第 3 項は定数である。これらが合わさって、検証用データも、調整が必要な正則化定数もなしに、オッカムの剃刀を実現している。

cxxgpr は、この量の符号を反転したものを、最適化器のどれか（{doc}`optimizers` を参照）で最小化する。各ハイパーパラメータについての解析的な勾配を使う。`Gpr::fit` の後には、最小化した値を `FittedGpr::neg_log_marginal_likelihood` で得られる。これは*負の*対数周辺尤度である。同じデータで 2 つのカーネルを比べるときに役立つ数である。小さいほど、複雑さを考慮したうえでデータをよく説明している。

:::{note}
周辺尤度の曲面は一般に凸ではない。勾配法が見つけるのは局所的な最大値で、長さスケールが短い場合やノイズが小さい場合には、質の悪い解に落ちやすい。入力と目的変数を標準化する、妥当な初期値を選ぶ、必要なら複数回リスタートする、といった対処が有効である。{doc}`../how-to/tune-hyperparameters` を参照してほしい。
:::

## 1 点抜き予測

GP の枠組みでは、1 点抜き（LOO）交差検証がほとんど無料で手に入る。$Q = (K + \sigma_n^2 I)^{-1}$ とすると、訓練点 $i$ を除いたときのその点の予測平均と予測分散は

$$
\mu_{-i} = y_i - \frac{\alpha_i}{Q_{ii}}, \qquad
\sigma_{-i}^2 = \frac{1}{Q_{ii}}
$$

である（GPML 5.4.2 節）。$Q$ の対角は既存のコレスキー因子から得られる。そのため、$n$ 個の LOO 予測をすべてそろえても、追加の計算量は $O(n^2)$ で済み、$n$ 回の当てはめ直しは要らない。`FittedGpr::loo_predict` は、それらを目的変数の元のスケールで返す。{doc}`../tutorials/multi-dimensional-regression` のように観測値と比べれば、モデルとその誤差の幅が信頼できるかどうかが分かる。

## ノイズとジッターは別のもの

観測ノイズ $\sigma_n^2$ は*モデル*の一部である。測定にノイズがあることを表し、データから学習され、予測の観測分散に現れる。**ジッター**は数値上の道具である。丸め誤差で $K + \sigma_n^2 I$ が数値的に不定になり、コレスキー分解が失敗したときに限って、対角に足す、ごく小さな非負の数 $j$ のことである。cxxgpr は両者を厳密に分ける。最初の分解は常に $j = 0$ で試し、ジッターは失敗した後にだけ `JitterPolicy` に従って適用し、実際に使った値は `factor_jitter` としてモデルとともに保存する。詳細は {doc}`numerical-stability` にある。

## 数学からライブラリへ

| 概念 | cxxgpr での対応 |
|---|---|
| カーネル $k$ とそのハイパーパラメータ | カーネルの葉から作り、`+` と `*` で組み合わせる `KernelSpec` |
| ノイズ分散 $\sigma_n^2$ | `Gpr::with_noise_variance`、`fit` で学習される |
| 周辺尤度の最大化 | `optimizer::Lbfgs` などの最適化器を使った `Gpr::fit` |
| コレスキー因子 $L$、重み $\boldsymbol{\alpha}$ | `FittedGpr` が保持し、`persist::save_with_factor` が保存する |
| $\boldsymbol{\mu}_*$ と $\operatorname{diag}\Sigma_*$ | `Prediction` を返す `FittedGpr::predict` |
| 完全な $\Sigma_*$ と事後標本 | `predict_covariance`、`sample` |
| LOO 予測 | `FittedGpr::loo_predict` |

実践的な手順は {doc}`../how-to/build-kernels` と {doc}`../how-to/predict-and-sample` を参照してほしい。
