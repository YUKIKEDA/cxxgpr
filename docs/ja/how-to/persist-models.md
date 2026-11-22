# モデルを保存して読み込む

学習済みモデルは、ディレクトリに書き出して、あとで別のプロセスや別のマシンで読み戻せる。読み戻したモデルは、同じ予測を返す。このガイドでは、`save` と `save_with_factor` で保存し、`LoadedGpr` で読み込む手順を示す。読み込んだモデルで予測する方法、型付きの `FittedGpr` へ戻す方法、自作のカーネルや変換を含むモデルを `PersistRegistry` で保存する方法、ファイルに伴うエラーの扱いも扱う。ディレクトリの構造そのものは {doc}`../reference/file-format` にある。

このページのプログラムは `persist_roundtrip.cpp` と `custom_persist_registry.cpp` である。どちらも一時ディレクトリに書き出し、終了前に削除する。`cxxgpr/persist/persist.hpp` を include して使う。

## 保存方法を選ぶ

モデルを書き出す自由関数は 2 つある。どちらも、学習済みモデル、保存先ディレクトリ、省略可能な `PersistRegistry` を受け取り、`std::expected<void, GprError>` を返す。

| 関数 | 書き出すもの | 読み込み時間 | ファイルサイズ | 向く場面 |
|---|---|---|---|---|
| `save` | 学習データ、ハイパーパラメータ、変換 | カーネル行列を再分解する（$O(n^3)$） | 小さい | モデルを配布し、読み込み時の再分解を許容できる |
| `save_with_factor` | 上記に加えて、コレスキー因子 $L$ と重み $\alpha$ | 再分解しない。倍精度の因子はメモリマップされる | $n^2$ 個ほど大きい | 頻繁に読み込む、または $n$ が大きい |

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [save]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
config.json: 1176 bytes
model.safetensors: 5440 bytes
```

ディレクトリがなければ作成される。保存に成功すると、ディレクトリには `config.json` と `model.safetensors` の 2 ファイルだけが残る。例のモデルの学習点は 24 個で、因子として加わる $24 \times 24$ の倍精度行列が、5440 バイトの大半を占める。

どちらの関数も、すべての精度ポリシーと、すべての組込みカーネル・変換で使える。自作のカーネルや変換を含むモデルには、レジストリの引数が必要である。{ref}`persist-models-custom` を参照。

## 保存が保証することを知る

保存はモデルを変更しない。保存先のディレクトリに一時ファイル `model.safetensors.tmp` と `config.json.tmp` を書き、フラッシュしてから、それぞれを最終的な名前へリネームする。これには次の意味がある。

- 読み手が書きかけのファイルを見ることはない。各ファイルは、リネームによって丸ごと置き換わる。
- 書き込みやリネームに失敗すると、`save` は一時ファイルを削除し、`PersistErrorKind::Io` を返す。
- 保存できないモデルは、何かを書く前に検出され、保存先のディレクトリには触れない。有効な `persist_id` を持たない自作の構成要素があると `PersistErrorKind::NotPersistable` か `PersistErrorKind::InvalidPersistId`、UTF-8 の JSON に書けない自作の状態があると `PersistErrorKind::NotPersistable`（{ref}`persist-models-custom` を参照）、`load` が拒む 128 段より深い入れ子の `config.json` になると `PersistErrorKind::Config` を返す。

:::{warning}
2 つのファイルは、1 つのトランザクションではなく、順にリネームされる。2 回のリネームの間にプロセスが強制終了したとき、または 2 回目のリネームに失敗したとき（`save` は `PersistErrorKind::Io` を返す）、ディレクトリには新しい `model.safetensors` と古い `config.json` が残る。`save` は古い `model.safetensors` を元に戻さない。両ファイルには内容ハッシュに基づくペアリング識別子（`config.json` の `save_id` と `model.safetensors` の `__metadata__["cxxgpr.save_id"]`）が書かれるため、ローダーは異なる保存から来た組を `PersistErrorKind::Tensor` で拒む（2 回の保存内容が完全に同一である場合を除く）。中断時も直前のモデルを読み込める状態を保つ必要があるなら、新しいディレクトリへ保存し、ディレクトリごと所定の位置へ移す。
:::

:::{warning}
Windows では、メモリマップ中のファイルを置き換えられない。`model.safetensors` を生きている `LoadedGpr`（後述）がマップしているディレクトリへ保存すると、`PersistErrorKind::Io` で失敗し、既存のファイルはそのまま残る。古いファイルをマップしている `LoadedGpr` をすべて破棄してから保存するか、別のディレクトリへ保存する。Linux では、マップ中のファイルも置き換えられる。生きている `LoadedGpr` は古いデータで予測を続け、新たに `load` すると新しいファイルを読む。
:::

## モデルを読み込む

`LoadedGpr::load` はディレクトリを読み、`LoadedGpr` を返す。これは型を消去したモデルで、どの精度で学習したモデルでも、倍精度で予測する。

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [load]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
n = 24, d = 1, has_factor = true, mmap factor = true
save_with_factor round trip is bit-identical: true
save() (refactorized on load): has_factor = false, max |mean difference| = 0
```

`DoublePrecision` のモデルを `save_with_factor` で保存した場合、読み込んだモデルは、元のモデルと**ビット単位で同じ**平均と分散を予測する。因子と重みは変更されずに読み戻され、ハイパーパラメータと変換の統計量は、正確に再現できるだけの桁数で保存されるからである。この例では、`save` で保存したモデルでも平均が完全に一致した。同じ行列を同じコードで再分解すれば、結果は決定的になるためである。ただし、それは「そうなる可能性が高い」にとどまる。形式が保証するのは、因子を保存した場合のビット単位の一致だけである。`save` を使う場合は、許容誤差を付けて比べること。

`load` は第 2 引数でレジストリを受け取る（`load(dir, registry)`）。組込みの構成要素だけを使うモデルでは省略できる。

### 読み込んだモデルを調べる

`LoadedGpr` は、ファイルに入っていた内容を報告する。

| メンバー | 返すもの |
|---|---|
| `n`、`d` | 学習サンプル数と特徴量の数 |
| `precision_kind` | 学習時の `PersistPrecisionKind`。`Double`、`Single`、`MixedPromote`、`MixedReevaluate` のいずれか |
| `has_factor` | ファイルに $L$ と $\alpha$ が保存されていたか |
| `has_mmap_factor` | $L$ をメモリマップしたファイルから直接読んでいるか |
| `factor_jitter` | 分解が成功したときに採用された対角ジッタ |
| `kernel_spec` | 復元された `KernelSpec`（学習済みのハイパーパラメータを持つ） |
| `likelihood` | 復元された `GaussianLikelihood`（ノイズ分散とその区間） |
| `jitter` | 復元された `JitterPolicy` |
| `kernel_exp` | `KernelExp` のモード |
| `distance_cache_policy` | `DistanceCachePolicy` |

### 読み込んだモデルで予測する

`LoadedGpr` には、`FittedGpr` と同じ予測メソッドがあり、結果はつねに `double` で、元の目的変数のスケールである。入力変換と目的変数の変換は自動で適用される。

| メソッド | 結果 |
|---|---|
| `predict(xs, options)`、`predict(span, n_rows, n_cols, options)` | 平均と分散を持つ `Prediction<double>` |
| `predict_into(xs, out, options)` | 呼び出し側が用意した `Prediction<double>` へ同じ結果を書く |
| `predict_covariance(xs, options)` | `PredictiveCovariance<double>`。$m \times m$ の完全な共分散 |
| `sample(xs, n_draws, seed, options, jitter)` | `std::vector<double>`。事後サンプルの $m \times n_\text{draws}$ 列優先行列。`seed` に対して決定的 |
| `loo_predict(options)` | 学習点に対する閉形式の leave-one-out 予測 |

`options` 引数は `PredictOptions` で、`FittedGpr` と同様に、潜在分散か観測分散かを選ぶ。

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [loaded-api]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
covariance entries = 16, samples = 12, loo points = 24
```

これらはすべて、問い合わせが空なら `GprErrorCode::EmptyInput`、列数が `d()` と違えば `GprErrorCode::DimensionMismatch`、`NaN` や無限大を含めば `GprErrorCode::NonFiniteInput` を返す。`sample` は `GprErrorCode::CholeskyFailed` を返すこともある。

## 読み込んだモデルを `FittedGpr` に戻す

`into_fitted` は、自分でデータを所有する型付きのモデルを取り出す。このモデルは、ファイルにもメモリマップにも依存しない。最適化器の型と精度ポリシーを渡す。精度ポリシーは、ファイルに保存されているものと一致していなければならない。

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [into-fitted]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
into_fitted prediction is bit-identical: true
into_fitted<SinglePrecision> on a double model: persist failed (wrong_model): requested precision policy does not match persisted model precision
```

`P` が保存されたポリシーと違うと、`into_fitted<O, P>(optimizer)` は `PersistErrorKind::WrongModel` を返す。倍精度のモデルを単精度として読み替えることはできない。最適化器はファイルに保存されないので、ここで選ぶ。結果に対して `refit` や `into_trainer` を呼ぶときだけ意味を持つ。取り出したモデルはコレスキー因子を保持しているので、学習したばかりのモデルと同じように振る舞う。

| 保存された `precision_kind()` | 指定するポリシー |
|---|---|
| `PersistPrecisionKind::Double` | `DoublePrecision` |
| `PersistPrecisionKind::Single` | `SinglePrecision` |
| `PersistPrecisionKind::MixedPromote` | `MixedPrecision<PromoteStorage>` |
| `PersistPrecisionKind::MixedReevaluate` | `MixedPrecision<ReevaluateKernel>` |

## 因子がメモリマップされる条件を知る

`has_factor == true` の `DoublePrecision` モデルを読み込むと、因子 $L$ はコピーされない。`has_mmap_factor()` は `true` になり、オペレーティングシステムが必要に応じてデータをページインする。そのため大きなモデルの読み込みはほぼ瞬時に終わり、複数のプロセスが 1 つのコピーを共有できる。単精度と混合精度の因子は 32 ビット浮動小数点で保存され、メモリへコピーされるので、`has_mmap_factor()` は `false` になる。`save` で保存したモデルでも `false` である。

マップは、`LoadedGpr` のコピーが 1 つでも生きている間は保たれる。コピーは内部のデータを共有し、最後の 1 つが破棄されるまでファイルはマップされたままである。

:::{warning}
`LoadedGpr` がマップしている `model.safetensors` を、その場で書き換えてはならない。読み込んだモデルが、変更されたバイトを読むことになる。予測は `load` の後に因子を検査し直さないので、結果は不定になり、エラーなしに誤った値を返すこともある。ファイルは、`save` と同様に置き換えること（上の Windows の制約に従う）。
:::

(persist-models-custom)=
## 自作のカーネルと変換を保存する

保存されたモデルは、カーネルと変換を名前で記述する。組込みのものはローダーが認識できる。`CustomKernel` 経由で `KernelTerm` から作ったカーネルや、変換のインターフェースから派生した自作の変換は認識できない。そこで、それぞれに安定した識別子を与え、再構築の方法をローダーに教える。

1. クラスで、一意な識別子を返す `persist_id()` と、オブジェクトを再構築するための状態を返す `persist_state_json()` を上書きする。変換については {doc}`transform-data` で説明した 2 つの仮想関数と同じである。カーネルの葉では `KernelTerm` にある。
2. 識別子ごとに 1 つのファクトリを `PersistRegistry` へ登録する。`register_kernel`、`register_unfitted_input`、`register_fitted_input`、`register_unfitted_target`、`register_fitted_target` を使う。
3. 同じレジストリを `save` や `save_with_factor` と `LoadedGpr::load` の両方に渡す。

例では、自作のカーネルと自作の入力変換（対数のあとに標準化）を持つモデルを保存する。クラス `GaussTerm`、`FittedLog`、`UnfittedLog` が、インターフェースと保存用のペアを実装している。たとえば次のようになる。

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [transform]
:end-before: // [end]
:dedent: 4
```

1 つのレジストリが、すべての自作の構成要素の再構築方法を知っている。ファクトリは状態の文字列を受け取り、オブジェクトかエラーを返す。

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [registry]
:end-before: // [end]
:dedent: 4
```

あとは、レジストリを付けた通常の呼び出しで保存と読み込みを行う。

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [save-load]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
load without registry fails with UnregisteredId: true
predictions after load are bit-identical: true
```

### 識別子と状態の規則

- 識別子は空であってはならず、`cxxgpr.`（定数 `RESERVED_PREFIX`）で始まってもならず、有効な UTF-8 でなければならない。`validate_persist_id` がこの規則を適用し、登録と保存の両方で使われる。違反すると `PersistErrorKind::InvalidPersistId` になる。
- 同じ識別子は、カテゴリごとに 1 回だけ登録できる。2 回目の登録は `PersistErrorKind::InvalidPersistId` で失敗する。5 つの登録カテゴリ（カーネル、未学習と学習済みの入力変換、未学習と学習済みの目的変数変換）は互いに独立している。変換には、未学習と学習済みの登録が 1 つずつ必要である。例では、別々の識別子（`example.log.unfitted`、`example.log.fitted`）を与えている。
- 状態は、有効な JSON であることが望ましい文字列である。JSON のオブジェクト、配列、数値、`true`、`false`、`null` として解析できれば、ライブラリは `config.json` の中に構造化された JSON として保存し、ファクトリには再シリアライズした文字列を渡す。これは返した文字列と同じとは限らない。数値は書き直され（`2.500000` は `2.5`、`1e2` は `100.0` になる）、空白は除かれ、オブジェクトのキーは整列される。それ以外の状態は、JSON の文字列リテラル（たとえば `"abc"`）であっても単なる文字列として保存し、ファクトリには返した文字列が引用符も含めてそのまま渡る。単なる文字列として保存する状態は、有効な UTF-8 でなければならない。そうでなければ、保存は `PersistErrorKind::NotPersistable` で失敗する。JSON のオブジェクトか数値を使うのがよい。状態は、ファクトリが必要とするすべてを含めること。カーネルでは、*学習後*のハイパーパラメータも含める。
- ビット単位の一致を保つには、浮動小数点の状態をすべての桁で書く。`std::format("{}", x)` は、読み戻すと正確に元の値になる最短の文字列を作る。6 桁しか残さない `std::to_string` は使わない。
- `persist_id` がない（既定の）自作の構成要素は保存できず、保存は `PersistErrorKind::NotPersistable` で失敗する。

```{literalinclude} /../examples/custom_persist_registry.cpp
:language: cpp
:start-after: // [registration-errors]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
reserved prefix rejected: true, duplicate rejected: true
```

ファクトリのエイリアス `CustomKernelFactory`、`UnfittedInputFactory`、`FittedInputFactory`、`UnfittedTargetFactory`、`FittedTargetFactory` は、`PersistRegistry` が受け付ける関数の型である。レジストリは、内部で使う検索も公開している。`create_kernel`、`create_unfitted_input`、`create_fitted_input`、`create_unfitted_target`、`create_fitted_target` である。これらは、再構築したオブジェクトを返す。識別子が未登録なら `PersistErrorKind::UnregisteredId` になる。通常は自分で呼ぶ必要はないが、ファクトリを単独でテストするときに便利である。

## エラーを扱う

失敗は `GprError` で報告される。保存に関するエラーでは、`GprError::code()` が `GprErrorCode::PersistFailed` になり、`persist_error_kind()` が何が起きたかを示す `PersistErrorKind` を返す。

| `PersistErrorKind` | 意味 | 典型的な原因 |
|---|---|---|
| `Io` | ファイルシステムの失敗 | ディレクトリがない、権限がない、ディスクが満杯、マップに失敗した |
| `Config` | `config.json` が壊れている | JSON が不正、キーが欠けている、入れ子が 128 段を超える |
| `Tensor` | `model.safetensors` が不正 | ファイルが切り詰められた、形状や `dtype` が違う、有限でない値、アラインメントが合わない |
| `InvalidPersistId` | 自作の識別子が許されない | 空、予約された接頭辞、有効な UTF-8 でない、すでに登録済み |
| `NotPersistable` | 保存できない構成要素がある | `persist_id` のない自作のカーネルや変換、JSON でも有効な UTF-8 でもない状態 |
| `UnregisteredId` | 登録されていない自作の構成要素がファイルにある | `save` で使ったレジストリなしで `load` した |
| `WrongModel` | ファイルが呼び出し側の求めるものと違う | `into_fitted` での精度の不一致、または別のモデル族のファイル |

```{literalinclude} /../examples/persist_roundtrip.cpp
:language: cpp
:start-after: // [errors]
:end-before: // [end]
:dedent: 4
```

```{code-block} text
missing directory -> PersistErrorKind::Io: true
```

読み込みはファイルを信用しない。テンソルのどこかに有限でない値があれば拒否し、すべての形状、`dtype`、アラインメントを検査し、JSON の入れ子の深さにも上限を設ける。`save` で保存したモデルでは、カーネル行列がもう分解できなくなっていると、読み込みが `GprErrorCode::CholeskyFailed` で失敗することもある。

## 形式のバージョンを扱う

定数 `FORMAT_VERSION` は、このリリースが書き出し、読み込めるディスク形式のバージョンで、現在は `1` である。`format_version` が違うファイルを `load` すると、`GprErrorCode::UnsupportedPersistVersion` が返る。エラーは両方の番号を持つ。

```cpp
auto loaded = cxxgpr::persist::LoadedGpr::load(dir);
if (!loaded && loaded.error().code() == cxxgpr::GprErrorCode::UnsupportedPersistVersion)
{
    // loaded.error().found_version()     -> version in the file (std::optional<std::uint32_t>)
    // loaded.error().supported_version() -> version this library supports
}
```

自動の移行機能はない。古いモデルを読む必要があるなら、それを書き出した版のライブラリを残しておくか、学習データから再学習する。学習データは、どのファイルにも `x` と `y` として入っている。

## 供給された距離のモデルを保存する

`FittedDistanceGpr` は、同じ `save` と `save_with_factor` で保存し、`LoadedDistanceGpr<C>` で読み戻す。`C` は `DistanceOnly` か `WithPoints` で、保存したモデルのモードに合わせる。チェックポイントには、距離スロットごとの `d2.<k>` テンソル（検査済みの学習距離の下三角）と、`config.json` の `distance` セクションが加わる。詳細は {doc}`../reference/file-format` を参照。

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [save]
:end-before: // [end]
```

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [load]
:end-before: // [end]
```

読み込むと、**新しいスロット**が発行される。保存の前に使った `ScalarDistance` や `ArdDistance` は、読み込んだモデルには使えない。読み込んだモデルの `kernel().slots()` から、カーネルの順にスロットを取り出し、予測点の表を結び付ける。

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [bind]
:end-before: // [end]
```

`LoadedDistanceGpr` が持つのは、`load`、`n`、`d`、`precision_kind`、`has_factor`、`has_mmap_factor`、`factor_jitter`、`kernel`、`likelihood`、`jitter`、`kernel_exp`、`predict`、`predict_covariance`、`sample`、`loo_predict`、`into_fitted` である。予測系のメソッドは `FittedDistanceGpr` と同じ表（`WithPoints` では予測点の座標も）を受け取り、`double` の結果を返す。`into_fitted` は、保存した精度ポリシーの型付き `FittedDistanceGpr` を返す。

- `save_with_factor` で保存すると、因子と重みは保存した値のまま復元され、予測は保存前のモデルとビット単位で一致する。倍精度の因子は `LoadedGpr` と同じく読み取り専用のメモリマップで読む（`has_mmap_factor()` が `true`）。そのため同じ決まりに従い、モデルやそのコピーが生きている間はファイルをその場で書き換えないこと。`into_fitted` は因子をコピーするので、返すモデルはファイルに依存しない。`save` で保存した場合は、読み込み時に因子を計算し直し、予測は丸めの範囲で一致する。
- 読み込んだ `MixedPrecision` のモデルは `refinement_summary` を持たない。`neg_log_marginal_likelihood` は保存した因子と改良後の重みから計算し直すので、学習時に報告した値（単精度の解から得た値）とわずかに違うことがある。
- 表は、読み込み時にもう一度検査される。負の要素や 0 でない対角は `InvalidDistance`、非有限の値は壊れたテンソル（`PersistErrorKind::Tensor`）として拒否する。
- 別のモードのチェックポイントや座標モデルのチェックポイントは、`PersistErrorKind::WrongModel` で拒否する。

```{literalinclude} /../examples/persist_distance_model.cpp
:language: cpp
:start-after: // [wrong-loader]
:end-before: // [end]
```

`MixedPrecision` のモデルは、表を単精度で保存する。予測に使う精度と同じである。

## API 一覧

| シンボル | 役割 |
|---|---|
| `FORMAT_VERSION` | このリリースが読み書きする形式のバージョン（`1`） |
| `RESERVED_PREFIX` | 自作の識別子に使えない接頭辞 `cxxgpr.` |
| `save` | 因子なしでモデルを書き出す。ファイルごとに原子的 |
| `save_with_factor` | $L$ と $\alpha$ を付けてモデルを書き出す |
| `LoadedGpr` | 読み込んだ、型を消去したモデル。`load`、`n`、`d`、`precision_kind`、`has_factor`、`has_mmap_factor`、`factor_jitter`、`kernel_spec`、`likelihood`、`jitter`、`kernel_exp`、`distance_cache_policy`、`predict`、`predict_into`、`predict_covariance`、`sample`、`loo_predict`、`into_fitted` |
| `LoadedDistanceGpr` | 読み込んだ供給距離のモデル（モードは `DistanceOnly` か `WithPoints`）。`load`、`n`、`d`、`precision_kind`、`has_factor`、`has_mmap_factor`、`factor_jitter`、`kernel`、`likelihood`、`jitter`、`kernel_exp`、`predict`、`predict_covariance`、`sample`、`loo_predict`、`into_fitted` |
| `PersistRegistry` | 自作の構成要素のファクトリ。`register_kernel`、`register_unfitted_input`、`register_fitted_input`、`register_unfitted_target`、`register_fitted_target`、および `create_kernel`、`create_unfitted_input`、`create_fitted_input`、`create_unfitted_target`、`create_fitted_target` |
| `validate_persist_id` | 識別子を、空文字列、予約接頭辞、UTF-8 の規則に照らして検査する |
| `CustomKernelFactory`、`UnfittedInputFactory`、`FittedInputFactory`、`UnfittedTargetFactory`、`FittedTargetFactory` | ファクトリ関数の型 |

生成リファレンスは [persist API リファレンス](../reference/api/persist.md) にある。
