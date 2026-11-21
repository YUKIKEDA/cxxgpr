# cxxgpr に渡すデータを準備する

訓練データとクエリデータを、cxxgpr が期待する配置で用意する方法を示す。所有しない行列ビューの使い方、サイズや値を検査用ヘルパー（checked ヘルパー）で自分で確かめる方法、結果の型とビルド情報の読み方も扱う。対象は core モジュール全体、つまり `cxxgpr/core/views.hpp`、`cxxgpr/core/checked.hpp`、`cxxgpr/core/build_info.hpp` である。

## 行列を列優先で並べる

ライブラリのどの行列引数も**列優先**（column-major、Fortran 順）である。1 つの列の要素はメモリ上で隣り合い、列 $j$ はオフセット $j \cdot \text{col\_stride}$ から始まる。サンプル数 $n$（行）、特徴量数 $d$（列）の訓練行列では、バッファに特徴量 0 の $n$ 個の値が先に並び、続いて特徴量 1 の $n$ 個が並ぶ。要素 $(i, j)$ の添字は $j \cdot n + i$ である。目的変数は $n$ 個の値の普通のベクトルである。

データが 1 サンプル 1 行の形で手元にあるなら、列優先のバッファへコピーする。

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [layout]
:end-before: // [end]
:dedent: 4
```

出力を見ると、特徴量 0 の 3 つの値が先に並んでいる。

```{code-block} text
x(1, 1) = 20, buffer = 1 2 3 | 10 20 30
```

:::{warning}
行優先のデータを誤って渡しても、検出する手段はない。要素数は正しく、値も有限だからである。結果は、間違った関数についての正しいモデルになる。迷ったときは、答えが分かっている角の `x(i, j)` を表示して確かめる。
:::

特徴量が 1 つなら、列優先のバッファと値のベクトルは同じものである。

## データをライブラリに渡す

モデルの入り口はどれも、行列の渡し方だけが違う 2 つの形を持つ。

| 形 | 例 | 使う場面 |
|---|---|---|
| スパンと形状 | `fit(x, n_rows, n_cols, y)` | `std::vector<double>` や `std::array<double, N>` のような連続バッファを自分が持っている。 |
| `MatrixRef<double>` | `fit(MatrixRef<double>(...), y)` | 行列がすでに、自分の持つメモリ上のビューになっている。 |

スパンの形は、列ストライドが `n_rows` に等しい密な行列を前提にする。どちらの形も `std::expected` を返す。たとえば、目的変数の長さが行数と違えば `GprErrorCode::LengthMismatch`、データが空なら `GprErrorCode::EmptyInput`、NaN や無限大があれば `GprErrorCode::NonFiniteInput` になる。

:::{note}
スパンの形は、要素を読む前に、スパンと形状が合っているかを検査する。スパンが空か、`n_rows` または `n_cols` が 0 なら `GprErrorCode::EmptyInput` になる。次元の積があふれる場合は `GprErrorCode::SizeOverflow`、スパンの大きさが `n_rows * n_cols` と一致しない場合は `GprErrorCode::LengthMismatch` になる。複数が同時に成り立つときは、この順で最初のものを返す。`MatrixRef` の形はバッファ長を再検査しない。`MatrixRef` が持つのはポインタと寸法だけで、バッファの長さは持たないためである。生ポインタから `MatrixRef` を作る場合は、正しい寸法を与える責任が呼び出し側にある。データが `std::span` なら、`MatrixRef<double>::from_span(span, n_rows, n_cols)` またはスパンのオーバーロードを直接使うのがよい。
:::

## `MatrixRef` と `MatrixMut` を使う

`MatrixRef<T>` は、列優先の行列を読み取り専用で参照する、所有しないビューである。`MatrixMut<T>` はその書き込み可能版である。どちらもコピーは安い（ポインタ 1 つと大きさ 3 つ）。メモリを所有しないので、バッファはビューより長く生きていなければならない。これらのおかげで、Eigen やライブラリ内部の記憶域が公開ヘッダに現れない。

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [views]
:end-before: // [end]
:dedent: 4
```

出力は次のとおりである。

```{code-block} text
a(2, 1) = 6, rows = 3, cols = 2, col_stride = 3, contiguous = true
column 1 has 3 elements, first = 4
block(0, 0) = 2, block(1, 1) = 6, contiguous = false
from_span(5 values, 3 x 2): length mismatch: expected 6 matrix values, got 5
after write: a(0, 0) = 10, via MatrixMut::as_ref = 10, via conversion = 10
default view: is_empty = true, is_square = true
```

### ビューを作る

2 つのクラスは、同じ 3 つのコンストラクタと、検査付きのファクトリ 1 つを持つ。

| コンストラクタ・ファクトリ | 意味 |
|---|---|
| `MatrixRef()` / `MatrixMut()` | 空のビュー。ポインタは null、大きさは 0。 |
| `(data, rows, cols, col_stride)` | ポインタ上の一般のストライドつきビュー。`col_stride >= rows` が必要。上の `block` のように部分ブロックに使う。 |
| `(data, rows, cols)` | ポインタ上の密なビュー。`col_stride` は `rows` に等しい。 |
| `from_span(span, rows, cols)` | スパン上の密なビューを作る静的ファクトリ。`std::expected` を返し、`span.size() == rows * cols` のときだけビューを返す。 |

コンストラクタはポインタを取り（`MatrixRef<T>` は `const T*`、`MatrixMut<T>` は `T*`）、検査をしない。寸法とストライドが指す全ての要素をポインタが指していることは、呼び出し側が保証する。`from_span` は、`MatrixRef<T>` なら `std::span<const T>`、`MatrixMut<T>` なら `std::span<T>` を取り、ポインタを結び付ける前にスパンと形状を照合する。スパンが空か、`rows` または `cols` が 0 なら `GprErrorCode::EmptyInput` になる。寸法の積が `std::size_t` をあふれる場合は `GprErrorCode::SizeOverflow`、スパンの大きさが `rows * cols` とちょうど一致しない場合は `GprErrorCode::LengthMismatch` になる。複数が同時に成り立つときは、この順で最初のものを返す。長すぎるスパンも拒否する。バッファの先頭部分だけを見たいときは、`span.first(rows * cols)` を渡す。

### アクセサ

| メンバ | 戻り値 |
|---|---|
| `data()` | 要素 $(0, 0)$ へのポインタ。空のビューでは null。 |
| `rows()`、`cols()` | 各次元の大きさ。 |
| `col_stride()` | 隣り合う列の先頭の間隔（要素数）。密なら `rows()` に等しい。 |
| `is_empty()` | `rows() == 0` または `cols() == 0` のとき真。 |
| `is_square()` | `rows() == cols()` のとき真。$0 \times 0$ のビューも正方とみなす。 |
| `is_contiguous()` | `col_stride() == rows()`、つまり列の間に隙間がないとき真。 |
| `operator()(row, col)` | 1 要素への参照。**範囲検査はしない**。`row < rows()` と `col < cols()` は呼び出し側が守る。 |
| `col(j)` | 列 $j$ への連続したスパン（`MatrixRef` は読み取り専用、`MatrixMut` は書き込み可能）。null のビュー、行数 0、`j >= cols()` では空のスパンを返す。 |
| `as_ref()` | `MatrixMut` のみ。同じ領域を `MatrixRef` として返す。 |
| `operator MatrixRef<T>` | `MatrixMut` のみ。暗黙の変換なので、書き込み可能なビューを、読み取り専用のビューを要求する場所へそのまま渡せる。 |

`MatrixMut` の `data()`、`operator()`、`col()` は `const` メンバ関数でありながら、書き込み可能なアクセスを返す。`const MatrixMut` は、`std::span<T>` と同様、書き込み可能なメモリへの*ハンドル*だからである。

:::{warning}
`operator()` は内側のループで使われるため、意図して検査していない。添字が外部から来る場合は、列の添字を検査する `col(j)` を使う。
:::

## 結果とオプションの型を選ぶ

`FittedGpr` と `LoadedGpr` の予測関数は、`views.hpp` の次の型を使う。

| 型 | 役割 |
|---|---|
| `VarianceKind` | 列挙型。`VarianceKind::Latent` はノイズを含まない関数 $f_*$ の分散を返す。`VarianceKind::Observation` は観測ノイズ $\sigma_n^2$ を加えた分散を返す（既定）。 |
| `PredictOptions` | フィールドが 1 つ（`variance_kind`、既定は `Observation`）の構造体。`operator==` で比較できる。`predict`、`loo_predict` などの最後の引数として渡す。 |
| `Prediction<T>` | 点ごとの予測結果。`mean` と `variance`（クエリごとに 1 要素）と、計算した `variance_kind` を持つ。`operator==` で比較できる。 |
| `PredictiveCovariance<T>` | 同時予測の結果。`mean`、列優先の $m \times m$ の `covariance`（要素 $(r, c)$ は `c * n_points + r`）、`n_points`、`variance_kind` を持つ。`operator==` で比較できる。 |
| `Triangle` | 列挙型 `Lower`、`Upper`、`Full`。対称行列のどの部分を読み書きするかを選ぶ。カーネルと距離のルーチンが使う（`kernel::Triangle` としても見える）。範囲は対角を含む。 |

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [results]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
default options are Observation: true, options equal: false, prediction == itself: true, Cov(0, 1) = 0.01
```

`T` は、モデルの精度ポリシーに応じて `double` か `float` になる。結果の取得は {doc}`predict-and-sample`、潜在分散と観測分散の違いは {doc}`../explanation/gaussian-processes` を参照してほしい。

## 検査付き演算でオーバーフローを防ぐ

大きな行列の添字計算は、`std::size_t` の中で気づかないうちに一周してしまうことがある。checked ヘルパーは `std::expected<std::size_t, GprError>` を返し、そのときは `GprErrorCode::SizeOverflow` を報告する。

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [checked]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
2 + 3 = 5, 1000 * 1000 = 1000000, 30 x 2 -> 60 elements, packed_len(4) = 10
checked_mul(max, 2): size overflows std::size_t
```

| 関数 | 結果 |
|---|---|
| `checked_add(a, b)` | $a + b$。あふれたら `SizeOverflow`。 |
| `checked_mul(a, b)` | $a \cdot b$。あふれたら `SizeOverflow`。どちらかが 0 なら即座に 0 を返す。 |
| `column_major_len(n_rows, n_cols)` | 空でない行列の $n_\text{rows} \cdot n_\text{cols}$。どちらかが 0 なら `EmptyInput`、あふれたら `SizeOverflow`。 |
| `packed_len(n)` | 下三角を詰めた長さ $n(n+1)/2$。途中の偽のオーバーフローを避けて計算する。 |

## 入力を自分で検査する

長い処理を始める前にデータを検査し、不正なデータファイルを正確なメッセージで拒否できる。次の検査関数を呼ぶとよい。どれも `std::expected<void, GprError>` を返し、`[[nodiscard]]` が付いている。

```{literalinclude} /../examples/views_and_checked.cpp
:language: cpp
:start-after: // [validate]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
require_nonempty(0): input is empty
require_count(3, 4): length mismatch: expected 4 y, got 3
require_finite(bad): input contains a non-finite value (NaN/Inf)
validate_matrix(2 x 3 over 4 values): length mismatch: expected 6 X values, got 4
validate_training(y too short): length mismatch: expected 4 targets, got 3
validate_training(wrong d): input dimension mismatch: X.ncols()=1, expected 2
validate_training(min_points = 5): insufficient data: n=4, at least 5 points required
validate_distance_targets(n = 3): length mismatch: expected 3 targets, got 4
validate_query(d = 2 expected): input dimension mismatch: X.ncols()=1, expected 2
```

| 関数 | 検査の内容 | エラーコード（優先順） |
|---|---|---|
| `require_nonempty(n)` | `n > 0` | `EmptyInput` |
| `require_count(actual, expected, what)` | `actual == expected`。`what` がメッセージ中の量の名前になる | `LengthMismatch` |
| `require_finite(values)` | `std::span<const double>` または `std::span<const float>` に NaN や無限大がない（非正規化数は許す。空のスパンは通る） | `NonFiniteInput` |
| `validate_matrix(x, n_rows, n_cols, name)` | 空でなく有限で、`x.size() == n_rows * n_cols`。`name` の既定は `"X"` | `EmptyInput`、`SizeOverflow`、`LengthMismatch`、`NonFiniteInput` |
| `validate_training(x, n_rows, n_cols, y, expected_d, min_points)` | 訓練セット。行列と `y` の大きさ、必要なら次元 `expected_d`（`std::optional<std::size_t>`、既定は指定なし）、`min_points` 行以上（既定は 1）。$n \times n$ のグラム行列の大きさがあふれないことも検査する | `EmptyInput`、`SizeOverflow`、`LengthMismatch`、`DimensionMismatch`、`NonFiniteInput`、`InsufficientData` |
| `validate_distance_targets(n, y, min_points)` | 与えた距離から学習するときの目的変数。特徴量行列は存在しない | `EmptyInput`、`SizeOverflow`、`LengthMismatch`、`NonFiniteInput`、`InsufficientData` |
| `validate_query(xs, n_rows, n_cols, expected_d)` | クエリ行列を、学習済みモデルの特徴量数と照合する | `EmptyInput`、`SizeOverflow`、`LengthMismatch`、`DimensionMismatch`、`NonFiniteInput` |

複数の条件が同時に成り立たないときは、表で先に書かれたエラーが返る。同じ入力なら、常に同じエラーになる。各コードへの対処は {doc}`handle-errors` を参照してほしい。

## 実行時にビルド情報を読む

`build_info.hpp` を使うと、プログラム、ログ、バグ報告から、ライブラリのビルドを特定できる。

| シンボル | 意味 |
|---|---|
| `DependencyVersion` | リンクされた第三者ライブラリ 1 つの `name` と `version`（`major.minor.patch`）を持つ構造体。 |
| `BuildInfo` | 下で説明する来歴レコード。 |
| `build_info()` | コンパイルされたライブラリの `BuildInfo` を返す。 |
| `build_info_json()` | 同じ内容を、インデント付きで末尾に改行のある JSON として返す。 |

`BuildInfo` のフィールドは次のとおりである。

| フィールド | 意味 |
|---|---|
| `schema` | レコードのレイアウトのバージョン。`cxxgpr.build-info/1` など。 |
| `version` | ライブラリのセマンティックバージョン。 |
| `git_commit` | ビルド時に記録されたコミットハッシュ。 |
| `uncommitted_changes` | ビルド時の作業ツリーにローカルの変更があったとき真。 |
| `compiler` | コンパイラを識別する文字列。 |
| `platform` | OS とアーキテクチャ。 |
| `platform_key` | プラットフォームごとの基準に名前を付けるための、正規化した `os-arch-compiler` キー。 |
| `configuration` | ビルド構成。`Release` など。 |
| `floating_point` | 浮動小数点のモード。縮約と命令セットを含む。 |
| `dependencies` | リンクされた第三者ライブラリの `std::vector<DependencyVersion>`。 |

```{literalinclude} /../examples/build_info.cpp
:language: cpp
:linenos:
```

出力例は {doc}`install-and-link` にある。

## API 一覧

| シンボル | ヘッダ | 節 |
|---|---|---|
| `MatrixRef`、`MatrixMut` | `core/views.hpp` | `MatrixRef` と `MatrixMut` を使う |
| `Triangle`、`VarianceKind`、`PredictOptions`、`Prediction`、`PredictiveCovariance` | `core/views.hpp` | 結果とオプションの型を選ぶ |
| `checked_add`、`checked_mul`、`column_major_len`、`packed_len` | `core/checked.hpp` | 検査付き演算でオーバーフローを防ぐ |
| `require_nonempty`、`require_count`、`require_finite`、`validate_matrix`、`validate_training`、`validate_distance_targets`、`validate_query` | `core/checked.hpp` | 入力を自分で検査する |
| `DependencyVersion`、`BuildInfo`、`build_info`、`build_info_json` | `core/build_info.hpp` | 実行時にビルド情報を読む |

これらのヘッダから生成したリファレンスは [core の API リファレンス](../reference/api/core.md) にある。
