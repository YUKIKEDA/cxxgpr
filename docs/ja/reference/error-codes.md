# エラーコード

失敗しうる cxxgpr の関数は、すべて `std::expected<T, GprError>` を返す。このページは、各 `GprErrorCode`、下位分類、`GprError` のインターフェースを列挙する。ヘッダーは `cxxgpr/core/error.hpp`。使い方は[エラーを処理する](../how-to/handle-errors.md)を参照。

## GprErrorCode

列挙型の基底型は `std::uint8_t` である。値は表の順に割り当てられる（`EmptyInput` が 0）。数値を永続化してはならない。

| コード | 発生条件 | ペイロードのアクセサ | ファクトリ |
|---|---|---|---|
| `EmptyInput` | 入力の行列、ベクトル、スパンが空、大きさ 0、またはヌルポインタ。 | なし | `empty_input()` |
| `SizeOverflow` | 次元またはパック済みバッファの大きさが `std::size_t` を超える。 | なし | `size_overflow()` |
| `LengthMismatch` | ベクトルやスパンの長さが、必要な長さと違う。 | なし | `length_mismatch(reason)` |
| `DimensionMismatch` | 特徴の列数が、モデルまたは ARD カーネルの期待と違う。 | `x_dim()`、`expected_dim()` | `dimension_mismatch(x_dim, expected_dim)` |
| `IndexOutOfRange` | パラメータ、距離スロット、葉、要素の添字が範囲外。 | なし | `index_out_of_range(reason)` |
| `InsufficientData` | 操作や検証ヘルパーが要求する点数より少ない。 | `n_points()`、`min_points()` | `insufficient_data(n_points, min_points)` |
| `NonFiniteInput` | 入力座標、目的変数、スカラーが NaN か無限大。 | なし | `non_finite_input()` |
| `NonFiniteKernelValue` | カーネル行列の要素、周辺尤度、潜在予測分散、逆変換後の値が NaN か無限大。 | なし | `non_finite_kernel_value()` |
| `InvalidHyperparameter` | 構成要素のパラメータが検証に失敗した（たとえば変換のスケールが正でない）。 | なし | `invalid_hyperparameter(reason)` |
| `InvalidNoiseVariance` | ノイズ分散または対数ノイズ分散が不正（非有限、非正、オーバーフロー）。 | なし | `invalid_noise_variance(reason)` |
| `InvalidInterval` | 区間の端点が不正、または値が開区間の外にある。 | `interval_error_kind()`、`interval_value()`、`interval_lo()`、`interval_hi()` | `invalid_interval_bounds(lo, hi)`、`invalid_interval_out_of_range(value, lo, hi)` |
| `InvalidConfig` | モデル、最適化器、ジッタ、変換、カーネルの設定が不正。 | なし | `invalid_config(reason)` |
| `InvalidDistance` | 与えた二乗距離行列が、対称性、対角、非負性、形のいずれかに違反する。 | `distance_slot()`、`distance_dim()`、`distance_pair()`、`distance_reason()` | `invalid_distance(slot, dim, pair, reason)` |
| `DistanceSlot` | 与えた距離ソースが、カーネルの距離スロットと合わない。 | `slot_error_kind()`、`distance_slot()` | `distance_slot(kind, slot)` |
| `UnsupportedKernelOperation` | カーネルが、要求された操作に対応しない。 | なし | `unsupported_kernel_operation(reason)` |
| `CholeskyFailed` | ジッタのスケジュールをすべて試しても、コレスキー分解に失敗した。 | `jitter()`、`matrix_size()`、`cholesky_stage()` | `cholesky_failed(jitter, matrix_size, stage)` |
| `NonPositiveDefiniteMatrix` | ジッタによる回復なしの、厳密な正定値性の検査に失敗した。 | なし | `non_positive_definite_matrix()` |
| `OptimizationNotConverged` | 最適化器が収束しなかったと報告した。組込みの最適化器は返さず、`OptimizerResult::converged` で報告する。 | `iterations()` | `optimization_not_converged(iterations)` |
| `WorkspaceTooSmall` | 呼び出し側が渡したワークスペースまたは作業バッファが小さすぎる。 | なし | `workspace_too_small()` |
| `PersistFailed` | モデルの保存または読み込みに失敗した（I/O、JSON、テンソル、レジストリ）。 | `persist_error_kind()` | `persist_failed(kind, reason)` |
| `UnsupportedPersistVersion` | 保存された形式のバージョンが、このビルドに対応していない。 | `found_version()`、`supported_version()` | `unsupported_persist_version(found, supported)` |

## 下位分類

| 列挙型 | 列挙子 | 対応するコード |
|---|---|---|
| `CholeskyStage` | `Fit`、`Predict` | `CholeskyFailed`: 学習中か、予測・サンプリング中か。 |
| `SlotErrorKind` | `NotRead`、`Duplicate`、`Missing` | `DistanceSlot`: カーネルが読まないスロット、二重に与えたスロット、必要なのに無いスロット。 |
| `PersistErrorKind` | `Io`、`Config`、`Tensor`、`InvalidPersistId`、`NotPersistable`、`UnregisteredId`、`WrongModel` | `PersistFailed`。 |
| `IntervalErrorKind` | `InvalidBounds`、`OutOfRange` | `InvalidInterval`。 |

4 つとも基底型は `std::uint8_t` である。

## GprError

`GprError` はコピー可能な値型である。コンストラクタは非公開で、上記の静的ファクトリで作る。

| メンバー | 戻り値 | 備考 |
|---|---|---|
| `code()` | `GprErrorCode` | `noexcept`。 |
| `message()` | `std::string_view` | エラーオブジェクト内のビュー。エラーが生きている間だけ有効。`noexcept`。 |
| `x_dim()`、`expected_dim()` | `std::optional<std::size_t>` | `dimension_mismatch` が設定する。 |
| `n_points()`、`min_points()` | `std::optional<std::size_t>` | `insufficient_data` が設定する。 |
| `jitter()` | `std::optional<double>` | 最後に試したジッタ。`cholesky_failed` が設定する。 |
| `matrix_size()` | `std::optional<std::size_t>` | 分解に失敗した行列の次数。 |
| `cholesky_stage()` | `std::optional<CholeskyStage>` | |
| `iterations()` | `std::optional<std::size_t>` | `optimization_not_converged` が設定する。 |
| `interval_error_kind()` | `std::optional<IntervalErrorKind>` | |
| `interval_value()` | `std::optional<double>` | `IntervalErrorKind::OutOfRange` のみ。 |
| `interval_lo()`、`interval_hi()` | `std::optional<double>` | |
| `distance_slot()` | `std::optional<std::size_t>` | `InvalidDistance` または `DistanceSlot`。分かっている場合のみ。 |
| `distance_dim()` | `std::optional<std::size_t>` | ARD の次元。分かっている場合のみ。 |
| `distance_pair()` | `std::optional<std::pair<std::size_t, std::size_t>>` | 問題のある行列要素（行、列）。 |
| `distance_reason()` | `std::optional<std::string_view>` | `code() == InvalidDistance` でなければ空。 |
| `slot_error_kind()` | `std::optional<SlotErrorKind>` | |
| `persist_error_kind()` | `std::optional<PersistErrorKind>` | |
| `found_version()`、`supported_version()` | `std::optional<std::uint32_t>` | |
| `with_slot_if_unset(slot)` | `GprError` | スロットが未設定なら設定したコピー。メッセージを作り直す。 |
| `with_dim_if_unset(dim)` | `GprError` | ARD の次元について同様。 |
| `operator==` | `bool` | コード、メッセージ、全ペイロードを比較する。 |

対応するファクトリで作られていないエラーでは、アクセサは `std::nullopt` を返す。

## メッセージ

`message()` の文面は人間向けであり、リリース間で変わりうる。現在の文面を、ファクトリごとに示す。

| ファクトリ | メッセージ |
|---|---|
| `empty_input()` | `input is empty` |
| `size_overflow()` | `size overflows std::size_t` |
| `length_mismatch("expected 3 values")` | `length mismatch: expected 3 values` |
| `dimension_mismatch(2, 3)` | `input dimension mismatch: X.ncols()=2, expected 3` |
| `index_out_of_range("leaf 4 of 3")` | `index out of range: leaf 4 of 3` |
| `insufficient_data(2, 5)` | `insufficient data: n=2, at least 5 points required` |
| `non_finite_input()` | `input contains a non-finite value (NaN/Inf)` |
| `non_finite_kernel_value()` | `kernel evaluation produced a non-finite value` |
| `invalid_hyperparameter("...")` | `invalid hyperparameter: ...` |
| `invalid_noise_variance("...")` | `invalid observation noise variance: ...` |
| `invalid_interval_bounds(2, 1)` | `interval bounds must be finite and satisfy lo < hi (got lo=2, hi=1)` |
| `invalid_interval_out_of_range(5, 0, 1)` | `value 5 is not strictly inside (0, 1)` |
| `invalid_config("...")` | `invalid configuration: ...` |
| `invalid_distance(2, 0, (1, 2), "negative entry")` | `invalid squared distance in slot 2, dimension 0 at (1, 2): negative entry` |
| `distance_slot(Missing, 1)` | `distance slot mismatch in slot 1: a kernel slot has no supplied distance source` |
| `unsupported_kernel_operation("...")` | `unsupported kernel operation: ...` |
| `cholesky_failed(1e-6, 10, Predict)` | `Cholesky factorization failed (stage=Predict, size=10, jitter=1e-06 already applied)` |
| `non_positive_definite_matrix()` | `matrix is not positive semidefinite` |
| `optimization_not_converged(100)` | `optimizer did not converge after 100 iterations` |
| `workspace_too_small()` | `workspace capacity is insufficient` |
| `persist_failed(Config, "bad json")` | `persist failed (config): bad json` |
| `unsupported_persist_version(2, 1)` | `unsupported persist format version 2; this library reads version 1` |

`reason` を取るファクトリは `[[nodiscard]]` で、引数は `std::string` である。引数のないファクトリは、メモリを確保しない。
