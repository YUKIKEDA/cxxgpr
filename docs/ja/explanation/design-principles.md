# 設計原則

このページは、cxxgpr の API を形づくる決定とその理由を説明する。エラーを値として返すこと、不正な状態を作れない型、学習と推論の厳密な分離、明確な所有権、列優先のデータ、サードパーティの型を含まない公開インターフェースを取り上げる。原則を知れば、ライブラリの振る舞いを予測でき、呼び出しがなぜその形をしているかがわかる。どの原則にも代償があり、それも書いておく。

## エラーは値である

失敗しうる操作はすべて `std::expected<T, GprError>` を返し、`[[nodiscard]]` が付いている。ライブラリは、API の境界を越えて例外を投げない。失敗は普通の戻り値であり、`GprErrorCode` とメッセージを持つ。コードによっては、構造化された詳細も持つ。問題のある次元、因子分解が失敗したときに最後に試したジッター、最適化器の反復回数、不正な距離の位置などである。

```cpp
auto fitted = trainer.fit(x, n_rows, n_cols, y);
if (!fitted) {
    // fitted.error().code(), fitted.error().message(), ...
    return;
}
auto prediction = fitted->predict(xs, m, n_cols);
```

理由は実際的である。数値計算は、データに依存する理由で失敗する（行列が正定値でない、カーネルの値が無限大へ溢れた）。失敗は例外的な出来事ではなく、予想される制御の流れである。値で返すエラーは、例外を無効にしたリアルタイムや組み込みのコードでも扱える。シグネチャに現れ、`[[nodiscard]]` により、無視するとコンパイラの警告になる。合成もできる。C++23 の `and_then`、`transform`、`or_else` で、入れ子の `if` なしに呼び出しを連ねられる。

代償は冗長さである。呼び出し箇所ごとに検査が要る。カーネルや尤度の構築も 2 段階になる（`create(...)` のあとに `.value()` か検査）。読みやすく保つパターンは {doc}`../how-to/handle-errors` にある。

そこから導ける性質は信頼してよい。`Gpr::fit` か `Gpr::factor` が失敗しても、トレーナーは以前のままで、書きかけの出力引数も残らない。両方とも `const` メンバ関数であり、トレーナーを変更できない。新しい `FittedGpr` を返すか、何も返さないかのどちらかである。

検証は決定的である。複数の問題が同時にあるときは、決まった順序で最初のエラーを報告する。空の入力、サイズのオーバーフロー、長さの不一致、次元の不一致、非有限値、データ不足の順である。同じ不正な呼び出しは、いつも同じエラーを返す。

## 不正な状態は作れない

不正な状態で作れる型は、それを受け取るすべての関数に、再検査を強いる。cxxgpr は代わりに、検査を構築時に移し、検証済みのオブジェクトだけが値を保持する唯一の手段になるようにする。

- **失敗しうる構築は、名前のあるファクトリである。** `RbfKernel::create(length_scale)`、`GaussianLikelihood::create(noise_variance)`、`Interval::create(lo, hi)`、`JitterPolicy::fixed(j)`、`JitterPolicy::adaptive(...)` は `std::expected` を返す。コンストラクタは private である。存在する `RbfKernel` の長さスケールは、必ず正で有限である。存在する `JitterPolicy` は、必ず筋の通った段階を持つ。
- **失敗しない構築は、普通のコンストラクタである。** `Gpr`、`Lbfgs`、`IdentityInput`、`StandardizeTarget` のように、既定値が常に有効な型は、コンストラクタを使う。
- **有界のパラメータは区間を持ち歩く。** ハイパーパラメータと許容範囲は一緒に運ばれる（`BoundedParam`、`Interval`）。最適化器が範囲外に出ることはなく、保存された値は常に範囲内にある。
- **閉じた集合は、列挙型とコンセプトである。** `MaternNu` は 3 値、`VarianceKind` は 2 値、`CholeskyBuffer` は 2 値である。精度はテンプレート引数で、`PrecisionPolicyConcept` が、対応する 4 つのポリシー（`DoublePrecision`、`SinglePrecision`、`MixedPrecision<PromoteStorage>`、`MixedPrecision<ReevaluateKernel>`）だけに制約する。`KernelScalarConcept` は `float` と `double` だけを受け付ける。打ち間違いや未対応の組み合わせは、実行時の驚きではなくコンパイルエラーになる。
- **ビルダーはコピーを返す。** `Gpr` の `with_...` 関数は、元のオブジェクトに触れず、変更を加えたコピーを返す。設定途中の状態のトレーナーが存在することはない。

## 学習と推論は別の型である

学習していないモデルは予測できない。`predict` が「未学習」のエラーを返すようにする代わりに、cxxgpr はその問いが成り立たないようにしている。

```{mermaid}
stateDiagram-v2
  [*] --> Gpr : 構築 (カーネル、最適化器)
  Gpr --> Gpr : with_... (変更したコピーを返す)
  Gpr --> FittedGpr : fit / factor
  FittedGpr --> FittedGpr : predict, predict_into, sample, loo_predict
  FittedGpr --> FittedGpr : refit
  FittedGpr --> Gpr : into_trainer
  FittedGpr --> Files : save, save_with_factor
  Files --> LoadedGpr : load
  LoadedGpr --> FittedGpr : into_fitted
```

`Gpr` には、`predict` も `save` も、因子へのアクセサもない。`FittedGpr` には `fit` がない。あるのは、自分のハイパーパラメータから新しい最適化を始める `refit` と、学習したハイパーパラメータを開始点とする未学習の `Gpr` に戻す `into_trainer` である。前処理にも同じ分割がある。`UnfittedInputTransform` と `UnfittedTargetTransform` は `fit` しかできず、`FittedInputTransform` と `FittedTargetTransform` を作る。後者が `apply` と `inverse_apply` を持つ。

利点は、実行時の検査の一群が不要になり、オブジェクトの状態が型に表れることである。代償は、フラグではなく `FittedGpr` を持ち回る必要があること、そして保存したモデルを `FittedGpr` に明示的に戻す必要があることである。

## 所有権は明示的である

オブジェクトは 3 種類あり、それぞれの規則は単純である。

- **結果は、所有された値である。** `Prediction`、`PredictiveCovariance`、サンプルのベクトル、`FittedGpr` は、自分のメモリ（`std::vector`、`std::unique_ptr`）を所有する。渡したバッファより長く生きる。
- **入力は、所有しないビューである。** `MatrixRef<T>`、`MatrixMut<T>`、`std::span` は、利用者が所有するメモリを指す。ライブラリは呼び出しの間だけ読み、ポインタを保持しない。`Gpr::fit` は、必要なもの（生の学習データを含む）を `FittedGpr` にコピーする。学習の直後に、自分の配列を解放してかまわない。
- **ワークスペースは、作った側が所有する。** 利用者が作った `QueryWorkspace` は利用者のものである。`FittedGpr` の中のものは、そのオブジェクトに属する。共有したモデルで `predict_into` がスレッドセーフでないのも、モデルのコピーが自分用のワークスペースを持って始まるのも、これが理由である。

出力パラメータ（`_into` 関数）を使うと、確保を利用者が制御できる。出力先を利用者が与え、ライブラリがそこに書く。

## 列優先のデータ

行列は列優先で、$n \times d$ 行列の要素 $(i, j)$ は添字 $j\,n + i$ にある（一般には $j \cdot \text{col\_stride} + i$）。そのため、特徴量の 1 列はメモリ上で連続した領域になる。

これは Eigen、BLAS、LAPACK、Fortran、MATLAB、R と同じなので、これらの世界のデータをコピーなしで渡せる。カーネルの評価の仕方にも合う。距離やグラム行列は列ごとに作られ、下三角の因子分解は列をたどる。ビュー型は `col_stride` を持つので、大きな行列の部分ブロックは、コピーなしのビューになる。

代償を払うのは、サンプルを行として保持するコード（`std::vector<std::array<double, d>>`、構造体の C 配列、NumPy の既定の順序）である。こうしたデータは 1 回だけ転置する必要がある。生のスパンを取るオーバーロード `fit(x, n_rows, n_cols, y)` は、平らな列優先のバッファを受け取る。行優先の $n \times d$ バッファをそのまま渡すと、黙って別の $d \times n$ 行列として読まれる。データの並べ方は {doc}`../how-to/prepare-data` を参照。

## 公開ヘッダーにサードパーティの型を含めない

`include/cxxgpr/` のヘッダーは、C++ 標準ライブラリと互いのヘッダーしかインクルードしない。Eigen（密行列の線形代数）、nlohmann/json（永続化）、OpenMP（並列化）は、静的ライブラリの実装の詳細であり、シグネチャにも型にもマクロにも現れない。

```{mermaid}
flowchart LR
  subgraph yours ["利用者のプログラム"]
    U["cxxgpr のヘッダー<br/>(標準ライブラリの型だけ)"]
  end
  subgraph lib ["libcxxgpr (静的)"]
    API["公開 API"] --> IMPL["実装"]
    IMPL --> E["Eigen"]
    IMPL --> J["nlohmann/json"]
    IMPL --> O["OpenMP"]
  end
  U --> API
```

おかげで、コンパイル時間が短くなり、インクルードパスに Eigen と JSON のヘッダーが入らず、ライブラリ内部の選択を API を変えずに差し替えられる。カーネル木は `CompiledKernel<T>`（実装へのポインタ）の背後に隠れ、永続化は、素の文字列とパスを受け取って返す。

この境界は、コンパイル時の境界にすぎない。ライブラリが OpenMP つきでビルドされていれば、リンクには OpenMP ランタイムが必要で、インストールされた CMake パッケージが `find_dependency` で面倒を見る。サードパーティのライセンスも、出荷するバイナリには引き続き適用される（`THIRD_PARTY_NOTICES.md` を参照）。ビルド手順は {doc}`../reference/build-options` にある。

## モジュールと依存関係

ライブラリは 9 つのモジュールからなる。依存は一方向である。モジュールが使えるのは、図の左にあるモジュールだけで、逆はない。

```{mermaid}
flowchart LR
  core["core<br/>エラー、ビュー、区間、<br/>尤度、乱数、ビルド情報"]
  precision["precision<br/>ポリシー、反復改良"]
  linalg["linalg<br/>コレスキー、ジッター、<br/>並列ヘルパー"]
  kernel["kernel<br/>葉、KernelSpec、<br/>CompiledKernel、距離"]
  transform["transform<br/>入力と目的変数の変換"]
  workspace["workspace<br/>バッファ、MLL、推論、<br/>LeafCache"]
  optimizer["optimizer<br/>Lbfgs、NelderMead など"]
  gpr["gpr<br/>Gpr、FittedGpr"]
  persist["persist<br/>保存と読み込み"]

  core --> precision --> linalg --> kernel --> workspace --> gpr --> persist
  core --> transform --> gpr
  core --> optimizer --> gpr
```

| モジュール | 名前空間 | 使う目的 | ガイド |
|---|---|---|---|
| `core` | `cxxgpr` | エラーの処理、行列の受け渡し、区間の設定、乱数のシード | {doc}`../how-to/handle-errors`、{doc}`../how-to/prepare-data` |
| `precision` | `cxxgpr` | `DoublePrecision`、`SinglePrecision`、`MixedPrecision` の選択、解の反復改良 | {doc}`../how-to/optimize-performance` |
| `linalg` | `cxxgpr::linalg` | 因子分解、求解、ジッター、決定的な並列ループ | {doc}`../how-to/use-low-level-building-blocks` |
| `kernel` | `cxxgpr` | カーネルの構築、コンパイル、評価、距離の供給、カスタム葉の作成 | {doc}`../how-to/build-kernels` |
| `transform` | `cxxgpr::transform` | 入力と目的変数の標準化、スケーリングなどの変換 | {doc}`../how-to/transform-data` |
| `workspace` | `cxxgpr` | 自分のバッファで周辺尤度、勾配、ヘッセ行列、推論を動かす | {doc}`../how-to/use-low-level-building-blocks` |
| `optimizer` | `cxxgpr::optimizer` | ハイパーパラメータ最適化器の選択と設定 | {doc}`../how-to/choose-optimizer` |
| `gpr` | `cxxgpr` | `Gpr` と `FittedGpr` による学習と予測 | {doc}`../how-to/configure-fit` |
| `persist` | `cxxgpr::persist` | モデルの保存と読み込み | {doc}`../how-to/persist-models` |

`cxxgpr/gpr/gpr.hpp` は、必要なモジュールのヘッダーをインクルードする。そのため、多くのプログラムは、このヘッダーと `cxxgpr/kernel/kernel.hpp` だけをインクルードすれば足りる。下位のモジュールは単独でも使える。`linalg` と `precision` は、モデルをまったく必要としない。

## 決定性と性能は設計上の制約である

再現できる数値と、確保のないホットパスは、最初からの要件であり、API の形を決めている。`_into` 関数、明示的なワークスペース、引数としてのシード、固定した縮約の順序がそうである。理由は {doc}`determinism-and-performance` と {doc}`numerical-stability` にある。

## これらの選択の代償

| 選択 | 利用者にとっての代償 |
|---|---|
| エラーを値で返す | 呼び出しごとの検査 |
| 検証するファクトリ | すべてのカーネルと尤度で、`create(...)` と、`.value()` か検査 |
| 学習済みの型を分ける | 保存したモデルは `into_fitted` で戻す。新しいデータには `refit` |
| 列優先のビュー | 行優先のデータは 1 回転置する |
| サードパーティの型を隠す | 内部の Eigen 行列にはアクセスできない。数値計算の部分には、低レベル API で到達する |
| 静的ライブラリのみ | ライブラリは各プログラムにリンクされる。共有ライブラリは提供しない |
| 1.0 より前のバージョン | 既定値と補助的なフィールドが変わる可能性がある（{doc}`../reference/compatibility`） |
