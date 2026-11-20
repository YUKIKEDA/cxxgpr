# モデルを保存して読み込む

このチュートリアルでは、モデルを一度当てはめてディスクへ書き出し、別のプログラムでもありうる場面を想定して読み込み直し、読み込んだモデルがまったく同じ予測を返すことを確かめる。最初のモデルを再利用するので、先に {doc}`first-regression` を済ませてほしい。

## ステップ 1：モデルを当てはめる

`save_and_reload.cpp` を作る。インクルードは最初のチュートリアルのものに、`cxxgpr/persist/persist.hpp` と `<filesystem>` を加えたものである。完全なリストはページの末尾にある。プログラムは最初のチュートリアルと同じ当てはめを繰り返し、3 つのクエリ点での予測を `before` に取っておく。

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

## ステップ 2：保存する

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [save]
:end-before: // [end]
:dedent: 4
```

`persist::save_with_factor(model, dir)` は、当てはめたモデルをディレクトリ `dir` へ書き出す。保存されたモデルは、2 つのファイルを持つディレクトリである。

- `config.json`：カーネルと学習したハイパーパラメータ、ノイズ、変換、数値設定を記した、人が読める記述。
- `model.safetensors`：訓練データと、分解（因子）を残す指定をしたので、コレスキー因子と重みベクトルを生のテンソルとして収めたもの。

例にある `remove_all` は、プログラムを繰り返し実行しても安全にするためだけの呼び出しである。失敗しうる cxxgpr の関数の例に漏れず、`save_with_factor` も `std::expected` を返す。典型的な失敗は、書き込めないディレクトリや、ライブラリが保存方法を知らないカスタムカーネルを使ったモデルである。

:::{tip}
`persist::save` は、因子を除いて同じモデルを書き出す。ファイルは小さくなるが、読み込み時に因子を再計算するので遅くなる。読み込み時間が重要なら `save_with_factor` を使う。
:::

## ステップ 3：読み込む

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [load]
:end-before: // [end]
:dedent: 4
```

`persist::LoadedGpr::load(dir)` は、ディレクトリを読んで検証する。結果の `LoadedGpr` は、保存したモデルのコンパイル時の型（最適化器や精度ポリシー）を持たず、予測に要るものだけを持つ。`n()` と `d()` は訓練点の数と特徴量の数を返す。`has_factor()` は、因子をディスクから読んだかどうかを返す。

ディレクトリがない、壊れている、または新しい形式バージョンで書かれている場合、`load` は中途半端なモデルではなくエラーを返す。

## ステップ 4：読み込んだモデルで予測する

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [compare]
:end-before: // [end]
:dedent: 4
```

`LoadedGpr::predict` の引数は `FittedGpr::predict` と同じで、省略可能な `PredictOptions` も取り、同じ `Prediction` を返す。入力と目的変数の変換はモデルとともに保存されているので、読み込んだモデルは、クエリの標準化も答えの逆変換も自分で行う。

プログラムを実行する。次のように出力される。

```{code-block} text
saved to sin_model
loaded: n = 8, d = 1, has_factor = true
x = 1.35  before: 1.0122 / 0.00426142  after: 1.0122 / 0.00426142
x = 3.15  before: -0.108219 / 0.00420796  after: -0.108219 / 0.00420796
x = 5.85  before: -0.289365 / 0.00460647  after: -0.289365 / 0.00460647
bit-identical = true
```

どのクエリでも、平均と分散がビットまで完全に一致している。プログラムは、一致した場合に限って終了コード 0 で終わる。作業ディレクトリに `sin_model` というディレクトリができているはずである。`config.json` をテキストエディタで開くと、何が保存されたかが分かる。

## 完全なプログラム

:::{dropdown} ファイル全体を表示する
```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:linenos:
```
:::

## 学んだこと

`persist::save_with_factor` で当てはめたモデルを保存し、`persist::LoadedGpr::load` で復元し、往復しても予測が変わらないことを確かめた。

カスタムカーネルを持つモデルの扱い、`save` と `save_with_factor` の使い分け、型付きの `FittedGpr` へ戻す方法は {doc}`../how-to/persist-models` にある。ディスク上のレイアウトは {doc}`../reference/file-format` に記されている。
