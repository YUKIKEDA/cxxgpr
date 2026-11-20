# 独自のカーネルを書く

独自のカーネル葉を追加し、`KernelSpec` の合成に組み込み、微分を検査し、永続化できるようにする手順を示す。実例として $k(d^2) = 1/(1 + d^2/2\ell^2)$ を実装し、組込みの `RationalQuadraticKernel`（$\alpha = 1$ のとき一致する）と照合する。

先に {doc}`build-kernels` を読むこと。`KernelSpec` と `CompiledKernel` の説明がある。ここで扱う宣言はすべて `cxxgpr/kernel/custom.hpp` にある。シグネチャは [kernel API リファレンス](../reference/api/kernel.md) を参照。

## 独自カーネルとは何か

独自の葉は、**二乗ユークリッド距離** $d^2 = \lVert x - x'\rVert^2$ の関数で、$p \ge 1$ 個のハイパーパラメータを持つ。二乗距離はライブラリが計算し、その行列に対してあなたのコードが呼ばれる。実装者が用意するのは次の 3 つである。

- 距離行列に対するカーネルの値
- 各パラメータの**対数**に関する 1 階・2 階微分
- 対数空間のパラメータベクトルと、各パラメータの開区間

葉が見るのは距離だけなので、異方的にはできず、生の座標にも依存できない。自分で計算した距離を使う場合は {doc}`use-supplied-distances` を参照。

## `KernelTerm<T>` インターフェース

`KernelTerm<T>` を継承する。`T` は `double` か `float`（`KernelScalarConcept`）で、クラステンプレート 1 つで両方の精度を扱える。下のメンバは、任意と書いたもの以外すべて純粋仮想関数である。

| メンバ | 契約 |
|---|---|
| `clone` | 深いコピーを `std::unique_ptr<KernelTerm<T>>` で返す |
| `num_params` | ハイパーパラメータの数。`noexcept` |
| `get_params(out)` | 対数空間のパラメータを書く。`out.size() != num_params()` なら `LengthMismatch` |
| `set_params(params)` | 検証してから代入する。どのエラーでも、オブジェクトは変化させない。返すのは `LengthMismatch`、`NonFiniteInput`、`InvalidInterval` |
| `bounds_into(out)` | パラメータごとの開区間 `Interval` を、ユーザー単位・パラメータ順に書く |
| `apply(dist, out, uplo)` | $n \times n$ の二乗距離行列から学習用グラム行列を作る。書くのは三角 `uplo` だけでよい |
| `apply_cross(dist, out)` | $n_1 \times n_2$ の行列から交差共分散を作る。全要素を書く |
| `fill_diag(out)` | すべての $i$ について $k(x_i, x_i)$、つまり距離 0 でのカーネル値 |
| `grad(dist, d_k, param_idx, uplo)` | $\partial k/\partial\log\theta_{p}$ |
| `hess(dist, d2_k, i, j, uplo)` | $\partial^2 k/\partial\log\theta_i\,\partial\log\theta_j$ |
| `persist_id`（任意） | 空でない識別子を返すと保存できる。既定は空 |
| `persist_state_json`（任意） | 状態を表す JSON 文字列。既定は、種別 `NotPersistable` の `PersistFailed` エラーを返す |

二乗距離は丸めで微小な負の値になることがある。平方根やべき乗を取る前に 0 で切り上げる。

### 微分は対数パラメータで取る

最適化器は $\vartheta = \log\theta$ の空間で動く。そのため `grad` と `hess` は、$\theta$ ではなく $\vartheta$ による微分を返さなければならない。例では $w = d^2/(2\ell^2)$、$dw/d\vartheta = -2w$ として次のようになる。

$$
k = \frac{1}{1 + w}, \qquad
\frac{\partial k}{\partial\vartheta} = \frac{2w}{(1 + w)^2}, \qquad
\frac{\partial^2 k}{\partial\vartheta^2} = \frac{4w\,(w - 1)}{(1 + w)^3}.
$$

## 項を実装する

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [term]
:end-before: // [end]
:dedent: 4
```

補助関数 `fill` は、要求された三角を走査してスカラー関数を $w$ に適用する。値・勾配・ヘッセ行列で共有している。注意点は 3 つある。

- **`set_params` は原子的にする。** サイズ、有限性、境界を先に検証し、代入は最後に行う。
- **エラーコードを分ける。** 範囲外の値は `GprError::invalid_interval_out_of_range` で返す。これは `GprErrorCode::InvalidInterval` を持つので、呼び出し側は境界違反と形状エラーを区別できる。
- **すべて `T` で書く。** `InverseQuadraticTerm<double>` と `InverseQuadraticTerm<float>` の 2 つのインスタンス化は、精度の差を除いて同じ挙動でなければならない。

## 項を `CustomKernel` で包む

`CustomKernel` は値のように扱えるラッパーで、`KernelTerm<double>` と `KernelTerm<float>` を 1 つずつ所有する。これにより、同じ葉が `DoublePrecision`、`SinglePrecision`、混合精度のどれでも動く。`CustomKernel::create` は 2 つの項を受け取り、どちらも null でないこと、`num_params` と `persist_id` が一致することを検査する。満たさなければ `InvalidConfig` または `LengthMismatch` を返す。項がコンストラクタ引数を持つクラステンプレートなら、`make_custom_kernel` が両方のインスタンス化をまとめて作る。

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [make]
:end-before: // [end]
:dedent: 4
```

`KernelSpec::custom` で spec に包める。`CustomKernel` には `num_params`、`get_params`、`set_params`（両方の精度で原子的）、`bounds_into`、`persist_id`、`persist_state_json`、型付きの項を読み取り専用で返す `term<T>()`、コピーとムーブ、`operator==`（対数パラメータをビット単位で比較）がある。パラメータは `double` の項から読む。

```{code-block} text
num_params = 1, persist_id = example.inverse_quadratic
```

## 値・勾配・ヘッセ行列を検証する

独自カーネルの品質は微分で決まる。参照があるなら必ず比べ、有限差分とは常に比べる。

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [compare]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
max |custom - rational quadratic|: value 1.11022e-16, gradient 1.11022e-16, Hessian 5.55112e-17
k(p0, p1) = 0.782609 (hand: 0.782609)
```

有限差分の検査では、`set_params` で対数パラメータを動かす。

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [finite-difference]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
max |gradient - finite difference| = 1.79211e-11
```

独自の葉は他の葉と同じように合成でき、パラメータは平坦なベクトルに加わる（ここでは定数が先、長さスケールが後）。範囲外の更新は拒否される。

```{code-block} text
constant * custom: log-parameters = [1.09861, 0.405465]
set_params(log 100) -> value 100.00000000000004 is not strictly inside (0.05, 50)
```

:::{note}
`CustomKernel` の内部で行う計算は、常に `std::exp` を使う。`KernelExp::FastApprox` が切り替えるのは組込みの葉だけである。
:::

:::{note}
独自の葉が受け取るのは二乗距離だけである。`CompiledKernel::apply_points` とその仲間では、ライブラリが座標から距離を計算する。`apply` では距離を自分で渡す。どちらの場合も、項は座標そのものを見ない。
:::

## カーネルを永続化できるようにする

`persist_id` を、空でない一意な文字列でオーバーライドする。`cxxgpr.` で始まる名前は予約されているので使えない。あわせて `persist_state_json` を、葉を再構築するのに必要な情報をすべて含む JSON オブジェクトを返すようにオーバーライドする。2 つの精度は同じ識別子を返さなければならない。既定のままの葉は学習には使えるが、それを含むモデルを保存しようとすると `NotPersistable` エラーになる。

読み込むときは、その JSON から `CustomKernel` を再構築するファクトリを `persist::PersistRegistry` に登録し、レジストリをローダーに渡す。往復の検証はディスクに触れずにできる。

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:start-after: // [persist]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
state = {"lengthscale":1.500000}, restored parameters equal: true
```

モデル全体の保存と読み込み、独自の変換用レジストリは {doc}`persist-models` で扱う。

## 完全なプログラム

```{literalinclude} /../examples/custom_kernel.cpp
:language: cpp
:linenos:
```

## API 一覧

| シンボル | 内容 |
|---|---|
| `KernelTerm<T>` | 抽象的な葉: `clone`、`num_params`、`get_params`、`set_params`、`bounds_into`、`apply`、`apply_cross`、`fill_diag`、`grad`、`hess`、`persist_id`、`persist_state_json` |
| `CustomKernel` | `double` と `float` を組にしたラッパー: `create`、`term<T>`、`num_params`、`get_params`、`set_params`、`bounds_into`、`persist_id`、`persist_state_json`、`operator==` |
| `make_custom_kernel<TermTemplate>(args...)` | 同じコンストラクタ引数から両方の項を作り、`CustomKernel::create` を呼ぶ |
| `KernelSpec::custom` | `CustomKernel` をカーネルの木の葉として包む |
