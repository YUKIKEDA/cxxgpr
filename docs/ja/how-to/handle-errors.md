# エラーを処理する

このガイドは、失敗を検出し、内容を調べ、対処する方法を示す。cxxgpr は API の境界で例外を投げない。失敗しうる操作はすべて `std::expected<T, GprError>`（または `std::expected<void, GprError>`）を返し、失敗した操作は、入力も呼び出し先のオブジェクトも変更しない。コードの簡潔な一覧は[エラーコード](../reference/error-codes.md)にある。

スニペットは、ページ末尾に全文を載せた 1 つのプログラムから取っている。`using namespace cxxgpr;` と `#include <cxxgpr/core/error.hpp>` を前提とする。

## 結果を毎回確認する

値を使う前に、返された `std::expected` を検査する。成功なら `*result` または `result.value()` が値になる。失敗なら `result.error()` が `GprError` である。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [check]
:end-before: // [end]
```

ほとんどの関数は `[[nodiscard]]` なので、結果を無視するとコンパイラが警告する。

:::{warning}
失敗した結果に `result.value()` を呼ぶと、`std::bad_expected_access<GprError>` が**投げられる**。ライブラリ自身は例外を投げないが、この標準ライブラリの呼び出しは投げる。失敗を意図的に例外へ変えるコードか、失敗がありえない箇所でだけ使うこと。**成功した**結果に `result.error()` を呼ぶと、未定義動作になる。
:::

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [value-throws]
:end-before: // [end]
```

```{code-block} text
value() threw; carried code InvalidInterval
```

## コードとメッセージを読む

`GprError::code()` は `GprErrorCode` を返す。分岐にはコードを使い、メッセージの文面は使わない。`GprError::message()` は、ログや利用者向けの `std::string_view` を返す。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [report]
:end-before: // [end]
```

:::{warning}
`message()` は、エラーオブジェクトの中を指すビューである。`GprError` が破棄されると、ビューはぶら下がる。使っている間はエラーを生かしておくか、先に `std::string` へコピーする。一時オブジェクトに対する `trainer.fit(...).error().message()` は、ぶら下がったビューになる。
:::

`GprErrorCode` を文字列にする関数は、ライブラリにない。例では、`switch` で `code_name` を定義している。この `switch` はすべての列挙子を並べ、`default` を置かない。新しいコードが追加されると、コンパイラが警告する。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [name]
:end-before: // [end]
```

## よくある入力エラーを起こして読む

次の失敗はすべて、呼び出し側の前提条件違反から生じる。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [bad-inputs]
:end-before: // [end]
```

```{code-block} text
empty fit -> EmptyInput: input is empty
short targets -> LengthMismatch: length mismatch: target length must match training row count
infinite target -> NonFiniteInput: input contains a non-finite value (NaN/Inf)
2-column query -> DimensionMismatch: input dimension mismatch: X.ncols()=2, expected 1
```

エラーによっては、メッセージのほかに構造化データも持つ。各アクセサは `std::optional` を返し、対応する種類のエラーでなければ空になる。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [accessors]
:end-before: // [end]
```

```{code-block} text
got 2 columns, expected 1
```

設定値の検証関数も、それぞれのエラーを返す。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [validators]
:end-before: // [end]
```

```{code-block} text
noise 0 -> InvalidNoiseVariance: invalid observation noise variance: noise variance must be positive
negative jitter -> InvalidConfig: invalid configuration: jitter must be finite and non-negative, got -1
negative lengthscale -> InvalidInterval: value -1 is not strictly inside (1e-05, 1e+05)
reversed interval -> InvalidInterval: interval bounds must be finite and satisfy lo < hi (got lo=2, hi=1)
value outside interval -> InvalidInterval: value 5 is not strictly inside (0, 1)
```

カーネルの長さスケールが 0 以下のときは、`GprErrorCode::InvalidHyperparameter` ではなく `GprErrorCode::InvalidInterval` になる（既定の正の区間の外だからである）。`InvalidInterval` の 2 つの場合は、`interval_error_kind()` で区別する。`IntervalErrorKind::InvalidBounds` は端点が不正な場合、`IntervalErrorKind::OutOfRange` は値が区間の外にある場合である。後者では `interval_value()`、`interval_lo()`、`interval_hi()` が読める。

## 対処を決める

コードは、呼び出し側が取れる対処で分類できる。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [classify]
:end-before: // [end]
```

| 対処 | コード | すること |
|---|---|---|
| 呼び出しを直す | `EmptyInput`、`SizeOverflow`、`LengthMismatch`、`DimensionMismatch`、`IndexOutOfRange`、`InsufficientData`、`NonFiniteInput`、`InvalidHyperparameter`、`InvalidNoiseVariance`、`InvalidInterval`、`InvalidConfig`、`InvalidDistance`、`DistanceSlot`、`UnsupportedKernelOperation`、`WorkspaceTooSmall` | 前提条件が破られている。同じ呼び出しを繰り返しても再び失敗する。データか設定を直す。 |
| 数値設定を変えて再試行する | `CholeskyFailed`、`NonPositiveDefiniteMatrix`、`NonFiniteKernelValue`、`OptimizationNotConverged` | 入力は正しいが、演算が破綻した。ジッタ、ノイズ、カーネルのハイパーパラメータ、精度、最適化器を変える。 |
| 利用者に知らせる | `PersistFailed`、`UnsupportedPersistVersion` | ファイルシステムか、保存されたモデルに原因がある。 |

以下では、各コードが生じる状況を挙げる。1 行の要約は[エラーコード](../reference/error-codes.md)にある。

### コード別の説明

**`EmptyInput`**: 入力の行列、ベクトル、スパンが、大きさ 0 かヌルポインタである。`fit`、`factor`、`predict*`、`sample*` と、ほとんどの低レベルルーチンで生じる。データが読み込まれたかを確認する。

**`SizeOverflow`**: $n^2$ や $p\,n^2$ のような大きさが `std::size_t` を超える。問題の規模が極端に大きいか、次元が壊れている。

**`LengthMismatch`**: ベクトルやスパンの長さが合わない。たとえば `fit` で `y.size() != x.rows()` の場合や、`get_params` と `set_params` のスパンの長さが違う場合である。

**`DimensionMismatch`**: 特徴の列数が、モデルまたは ARD カーネルの期待と違う。`x_dim()` が渡した列数で、`expected_dim()` が必要な列数である。

**`IndexOutOfRange`**: パラメータ、距離スロット、葉の添字が範囲外である。たとえば、`GaussianLikelihood::noise_grad_diag` に 0 以外の添字を渡した場合。

**`InsufficientData`**: 検証ヘルパーが要求するより点が少ない。`n_points()` と `min_points()` が数を返す。`Gpr::fit` は空でないデータをすべて受け付け、このコードを返さない。`cxxgpr/core/checked.hpp` の検証関数は返す。

**`NonFiniteInput`**: 学習入力、目的変数、クエリ点、パラメータに NaN か無限大がある。先にデータを整える。

**`NonFiniteKernelValue`**: カーネル評価、周辺尤度、潜在予測分散、逆変換後の値が NaN か無限大になった。極端なハイパーパラメータを避けるか、範囲を狭める。

**`InvalidHyperparameter`**: 学習済みの変換が、`from_parts` またはモデルの読み込み時に、不正なパラメータから作られた。標準化のスケールが正でない場合や、最小値が最大値を超える場合などである。通常の `fit` では生じない。定数の列や目的変数は、変換が処理する。

**`InvalidNoiseVariance`**: ノイズ分散が有限でないか正でない。対数ノイズが有限でないか、`exp` がオーバーフローする場合も含む。

**`InvalidInterval`**: 端点が有限でないか順序が正しくない。または、値が開区間の外にある（上の説明を参照）。

**`InvalidConfig`**: 設定オブジェクトが不正である。負のジッタ、0 回の再試行、負の最適化許容誤差、整合しない変換パイプラインなど。メッセージが該当する項目を示す。

**`InvalidDistance`**: 与えた二乗距離行列が、対称でない、対角が 0 でない、負の要素がある、形が違う、のいずれかである。`distance_slot()`、`distance_dim()`、`distance_pair()`、`distance_reason()` で場所を特定する。{doc}`use-supplied-distances` を参照。

**`DistanceSlot`**: 与えた距離ソースが、カーネルのスロットと合わない。`slot_error_kind()` は、`SlotErrorKind::NotRead`（カーネルが読まないスロットのソース）、`Duplicate`（重複）、`Missing`（不足）のいずれかである。`distance_slot()` がスロットを返す。

**`UnsupportedKernelOperation`**: カーネルが要求された操作を行えない。距離専用カーネルに座標評価を求めた場合や、項のない合成カーネルなど。

**`CholeskyFailed`**: ジッタのスケジュールをすべて試しても、分解に失敗した。回復の方法は次の節で説明する。`jitter()` は最後に試したオフセット、`matrix_size()` は行列の次数、`cholesky_stage()` は `CholeskyStage::Fit`（学習）か `CholeskyStage::Predict`（予測とサンプリング）である。

**`NonPositiveDefiniteMatrix`**: ジッタによる再試行なしの厳密な正定値性の検査に失敗した（低レベルの分解と、退化した因子に対する leave-one-out の評価）。

**`OptimizationNotConverged`**: 最適化器が諦めた。組込みの最適化器はこのコードを返さない。`optimizer_result().converged` で収束の有無を報告し、それまでに見つけた最良のパラメータを返す。自作の最適化器は、`iterations()` を設定してこのコードを返してよい（後述）。

**`WorkspaceTooSmall`**: 呼び出し側が渡した作業バッファが、必要な大きさより小さい（低レベル API のみ）。

**`PersistFailed`**: 保存または読み込みに失敗した。`persist_error_kind()` が `PersistErrorKind` を返す。`Io`、`Config`、`Tensor`、`InvalidPersistId`、`NotPersistable`、`UnregisteredId`、`WrongModel` のいずれかである。{doc}`persist-models` を参照。

**`UnsupportedPersistVersion`**: 保存された形式のバージョンに対応していない。`found_version()` と `supported_version()` が両方の番号を返す。

## `CholeskyFailed` から回復する

グラム行列 $K + \sigma_n^2 I$ は正定値のはずだが、浮動小数点の丸めで崩れることがある。`SinglePrecision`、非常に大きなカーネルの振幅、ほぼ重複した点で起きやすい。エラーは、何が起きたかを記録している。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [cholesky]
:end-before: // [end]
```

```{code-block} text
CholeskyFailed: last jitter 0, matrix order 300, stage Fit
adaptive jitter recovered, accepted jitter 0.01
```

次の対策を順に試す。

1. 適応ジッタ（`JitterPolicy::adaptive`）を使う。必要だったオフセットは `FittedGpr::factor_jitter()` で分かる。
2. データが許すなら、ノイズ分散か、その下限を上げる。
3. 精度を上げる（`MixedPrecision` か `DoublePrecision`）。
4. 重複した点を取り除く。長すぎる長さスケールを短くする。巨大な振幅を下げる。

既定のトレーナーのジッタは `0` なので、頑健さが必要なトレーナーでは、ポリシーを明示的に設定すること。{doc}`configure-fit` を参照。

## 分岐を書かずに処理を連鎖させる

`std::expected` にはモナド的なメンバーがある。最初の失敗で止まり、元の `GprError` を最後まで運ぶパイプラインが書ける。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [and-then]
:end-before: // [end]
```

```{code-block} text
chained mean at 0.5: 0.141139, failing chain stops at: InvalidNoiseVariance
```

| メンバー | 用途 |
|---|---|
| `and_then(f)` | 値に `f` を適用する。`f` 自身が `expected` を返すので、エラーが伝わる |
| `transform(f)` | 失敗しない関数で値を変換する |
| `or_else(f)` | エラーに `f` を適用して、回復または置き換える |
| `value_or(x)` | 値を返す。エラーなら `x` を返す（エラーは捨てられる） |

`or_else` を使うと、失敗を代替値に置き換えられる。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [or-else]
:end-before: // [end]
```

## 自分のエラーを返す

自作のカーネル、最適化器、変換は、ライブラリと同じ静的ファクトリで `GprError` を返す。コードごとに 1 つずつある。たとえば、`GprError::invalid_config(reason)`、`GprError::optimization_not_converged(iterations)`、`GprError::cholesky_failed(jitter, matrix_size, stage)` である。ライブラリは、そのエラーを変えずに伝え、トレーナーは元のまま残る。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [give-up]
:end-before: // [end]
```

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [custom-error]
:end-before: // [end]
```

```{code-block} text
custom optimizer error passes through: OptimizationNotConverged after 3 iterations
```

`GprError` は値型である。コピーでき、`operator==` はコード、メッセージ、すべてのペイロードを比較する。`with_slot_if_unset` と `with_dim_if_unset` は、`InvalidDistance` エラーのコピーを返し、スロットまたは次元が未設定なら埋める。内側の層が知らなかった文脈を、外側の層が足せる。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [factories]
:end-before: // [end]
```

```{code-block} text
invalid_distance: invalid squared distance in slot 2, dimension 0 at (1, 2): negative entry | slot 2, dim 0, pair (1, 2), reason "negative entry"
distance_slot: distance slot mismatch in slot 1: a kernel slot has no supplied distance source
persist_failed: persist failed (config): bad json
cholesky_failed: Cholesky factorization failed (stage=Predict, size=10, jitter=1e-06 already applied)
```

プログラムは、各ファクトリが作るメッセージを出力する。全文は[エラーコード](../reference/error-codes.md)にある。

## ディスクからの読み込み

存在しないモデルのディレクトリも、通常のエラーとして返る。

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:start-after: // [persist]
:end-before: // [end]
```

```{code-block} text
load -> PersistFailed, kind Io
```

## プログラム全文

```{literalinclude} /../examples/handle_errors.cpp
:language: cpp
:linenos:
```

## API 一覧

| シンボル | 用途 |
|---|---|
| `GprError`、`code`、`message` | エラー値。分類と文面。 |
| `GprErrorCode` | 21 種類のエラー分類。 |
| `CholeskyStage`（`Fit`、`Predict`）、`SlotErrorKind`（`NotRead`、`Duplicate`、`Missing`）、`PersistErrorKind`、`IntervalErrorKind`（`InvalidBounds`、`OutOfRange`） | 一部のコードの下位分類。 |
| `x_dim`、`expected_dim`、`n_points`、`min_points`、`jitter`、`matrix_size`、`cholesky_stage`、`iterations` | 次元、データ、コレスキー、最適化器のエラーのペイロード用アクセサ。 |
| `interval_error_kind`、`interval_value`、`interval_lo`、`interval_hi` | `InvalidInterval` のペイロード。 |
| `distance_slot`、`distance_dim`、`distance_pair`、`distance_reason`、`slot_error_kind` | 距離エラーのペイロード。 |
| `persist_error_kind`、`found_version`、`supported_version` | 永続化エラーのペイロード。 |
| `with_slot_if_unset`、`with_dim_if_unset`、`operator==` | 文脈の追加と比較。 |
| `empty_input`、`size_overflow`、`length_mismatch`、`dimension_mismatch`、`index_out_of_range`、`insufficient_data`、`non_finite_input`、`non_finite_kernel_value`、`invalid_hyperparameter`、`invalid_noise_variance`、`invalid_interval_bounds`、`invalid_interval_out_of_range`、`invalid_config`、`invalid_distance`、`distance_slot`、`unsupported_kernel_operation`、`cholesky_failed`、`non_positive_definite_matrix`、`optimization_not_converged`、`workspace_too_small`、`persist_failed`、`unsupported_persist_version` | 静的ファクトリ。コードごとに 1 つ（`InvalidInterval` は 2 つ）。 |

生成された API リファレンス: [`core`](../reference/api/core.md)。
