# 永続化フォーマット仕様

本仕様は、`cxxgpr`（Exact GPR）がディスク上の 1 ディレクトリへ保存・読込する形式（`FMT-01`〜`FMT-08`）を定める。形式版は `FORMAT_VERSION = 1` であり、`gprx` の Exact GPR 永続化形式（`config.json` + `model.safetensors`）互換を基本とする（ただしオンラインモデル用の `factor_kind: "ldlt"`・`point_ids` および Sparse モデルはスコープ外のため読込拒否する）。

---

## FMT-01 ディレクトリ構成とバージョン契約

モデル保存先は 1 ディレクトリとし、直下に次の 2 ファイルのみを配置する。

- `config.json`：モデル構造、精度、ポリシー、ハイパーパラメータ値・区間、前処理状態、距離スロット構成を記述する UTF-8 JSON。
- `model.safetensors`：学習データおよび計算済み Cholesky 因子等を格納する safetensors バイナリ。

定数 `cxxgpr::persist::FORMAT_VERSION = 1`（`std::uint32_t`）を定める。読込時に `config.json` の `format_version` が整数 `1` でない場合は `GprError::UnsupportedPersistVersion { found, supported }` を返す（別バージョンの自動移行機構は置かない）。

---

## FMT-02 原子的書き込みと入力信頼境界

- **原子的保存**：`save` / `save_with_factor` は出力先ディレクトリ（存在しなければ作成）内に一時ファイルを書き出し、フラッシュ完了後に同一ディレクトリ内の最終名（`config.json`, `model.safetensors`）へリネーム置換する。既存ディレクトリ上への再保存中もマップ済みファイルをその場で破壊書き換えしない。
- **信頼境界と検査**：読込対象ディレクトリは信頼できない外部入力として検証する。
  - JSON の入れ子深さ（`sum`/`product`/`pipeline`/`columnwise`）が 128 段を超える場合は `PersistFailed(Config)` で拒否する。
  - 読込時、全テンソル（`x`, `y`, `l`, `alpha`, `d2.*`）の形状・dtype・8 バイト整列および **全要素が有限（非 `NaN`・非 `±∞`）** であることを検証する。
  - 変換適用結果や再構成値に非有限が生じた場合もエラーとし、不整合な `LoadedGpr` を返さない。

---

## FMT-03 `config.json` スキーマ

| キー | 型・取り得る値 | 省略時既定値（読込） | 説明 |
|---|---|---|---|
| `format_version` | 整数 `1` | （必須） | 形式バージョン |
| `n` | 整数（$> 0$） | （必須） | 訓練データ点数 |
| `d` | 整数（$0$ は `distance_only` 時のみ） | （必須） | 訓練データ特徴次元数 |
| `has_factor` | bool | （必須） | `model.safetensors` に `l` と `alpha` を含むか（`save` は `false`、`save_with_factor` は `true`） |
| `factor_kind` | 文字列 `"llt"` | （必須） | Exact GPR 因子種別。`"llt"` 以外は `PersistFailed(WrongModel)` |
| `factor_jitter` | 数値（$\ge 0$） | `0.0` | 分解成功時に採用された数値安定化 jitter |
| `precision` | `"double"` / `"single"` / `"mixed"` | `"double"` | 精度ポリシー種別 |
| `residual` | `"promote_storage"` / `"reevaluate_kernel"` | `"promote_storage"` | `mixed` 時の残差方式 |
| `math` | `"accurate"` / `"fast_approx"` | `"accurate"` | `KernelExp` モード |
| `distance_cache` | `"always"` / `"never"` | `"always"` | `DistanceCachePolicy`（`Cached` / `Uncached`） |
| `kernel` | JSON オブジェクト | （必須） | カーネル木（後述） |
| `likelihood` | `{"noise_variance": <区間つき>}` | （必須） | $\sigma_n^2$ の実数値（対数ではない）と区間 |
| `jitter` | `{"fixed": {"jitter": j}}` または `{"adaptive": {"initial", "multiplier", "max_retries", "max_jitter"}}` | （必須） | `JitterPolicy`（デコード時コンストラクタ検査を通す） |
| `x_unfitted`, `x_transform` | 変換表現 | （必須） | 入力変換（学習前設定と学習済み統計） |
| `y_unfitted`, `y_transform` | 変換表現 | （必須） | 目的変数変換（学習前設定と学習済み統計） |
| `distance` | オブジェクト | 省略可（座標モデル） | 供給二乗距離メタデータ（`FMT-07`） |
| `model` | 存在しないこと | なし | `"sgpr"`・`"online_sgpr"`・`"svgp"` 等が存在すれば `PersistFailed(WrongModel)` |

浮動小数点 JSON 値は丸め損ねのない高精度 roundtrip 表記で書き出す（既定 17 桁／shortest roundtrip）。未知の任意トップレベルキーは読込時に無視する。

- **カーネル表現**：区間つき数値は `{"value": v, "lo": a, "hi": b}`。各組込み葉（`rbf`, `rbf_ard`, `matern`, `matern_ard`, `periodic`, `rq`, `rq_ard`, `constant`, `linear`, `white`）、複合ノード（`{"sum": {"left": ..., "right": ...}}`、`{"product": {"left": ..., "right": ...}}`）、距離ノード（`{"distance": {"slot": k, "leaf": ...}}`）、カスタム葉（`{"custom": {"persist_id": "...", "state": ...}}`）を表現する。
- **変換表現**：`"identity"`、`"standardize"`（学習後は `mean`, `std`）、`min_max`（学習前は `range_lo`, `range_hi`、学習後は `data_min`, `data_max`, `range_lo`, `range_hi`）、`pipeline`（`steps`）、`columnwise`（`maps`、`x_*` のみ）、`custom`（`persist_id`, `state`）。

---

## FMT-04 `model.safetensors` レイアウトと列優先契約

[safetensors](https://github.com/huggingface/safetensors) 仕様に準拠する（先頭 8 バイト小エンディアン `uint64` ヘッダ長 + JSON ヘッダ + 生テンソルデータ列。`__metadata__` は出力しない）。**2 次元行列（`x`, `l`）は列優先（Column-major / Fortran 順）** で格納し、`(i, j)` 要素の位置は `j * n + i` とする。一方、`d2.<k>` の ARD スロットは次元ブロック連結（`dims` 軸が外側、`n(n+1)/2` パック軸が内側ストライド 1）で格納する。データ部は 8 バイト境界に整列されていなければならない。

| テンソル名 | 形状 `shape` | `dtype` | 保存条件 | 内容 |
|---|---|---|---|---|
| `x` | `[n, d]` | `F64` | 常に | 入力変換前の学習用 `X` |
| `y` | `[n]` | `F64` | 常に | 目的変数変換前の学習用 `y` |
| `l` | `[n, n]` | `double`→`F64`、`single`/`mixed`→`F32` | `has_factor == true` | $A + (\sigma_n^2 + j)I$ の下三角 Cholesky 因子 $L$（対角より上は `0` 詰め） |
| `alpha` | `[n]` | `single`→`F32`、`double`/`mixed`→`F64` | `has_factor == true` | 解重み $\alpha$（変換後 `y` 空間の値） |
| `d2.<k>` | スカラー `[n(n+1)/2]` / ARD `[dims, n(n+1)/2]` | `single`→`F32`、`double`/`mixed`→`F64` | 距離モデル時のみ | スロット `k` の修復済み下三角二乗距離。スカラーはパック位置 $p = jn - j(j-1)/2 + (i-j)$、ARD は次元 $m \in [0, \text{dims})$ ごとに長さ $\text{packed\_len} = n(n+1)/2$ のパック配列を順に連結（フラット添字 `m * packed_len + p`）（`FMT-07`） |

NumPy から直接検証・利用する際は、`x` や `l` 等の列優先行列は `flat.reshape(meta["shape"], order="F")`、ARD 距離テンソル `d2.<k>` は `flat.reshape(meta["shape"], order="C")` により元の形状・意味へビット一致復元できる（1 次元テンソル `y`, `alpha`, スカラー `d2.<k>` はいずれでも同一）。

---

## FMT-05 `save` / `save_with_factor` と `f64` 因子 `mmap` 読込

| メソッド | `has_factor` | `model.safetensors` 内容 | 読込時の挙動（`LoadedGpr::load` / `LoadedDistanceGpr<C>::load`） |
|---|---|---|---|
| `save(dir)` | `false` | `x`, `y`（+ `d2.*`） | 保存済み変換・$\theta$・`jitter`・`factor_jitter` を用いて Gram 行列を組み直し $O(n^3)$ で Cholesky 分解・$\alpha$ を再構築する |
| `save_with_factor(dir)` | `true` | `x`, `y`, `l`, `alpha`（+ `d2.*`） | 再分解をスキップする。`DoublePrecision` の `F64` 因子 `l` は読取り専用メモリマップ（`mmap`）で直接参照し、`F32` 因子はメモリへコピーして取り出す |

- **`mmap` の安全要件**：`F64` 因子 `l` の `mmap` はデータ先頭オフセットが 8 バイト境界整列であることを検証した上でマッピング（Windows `CreateFileMapping`/`MapViewOfFile`、Linux `mmap(PROT_READ, MAP_PRIVATE)`）を行う。モデルの有効期間中、外部プロセスによる対象ファイルその場書き換えを行わない契約とし、`save` による置換は原子リネームで行うため進行中のマップを壊さない。
- **ローダー戻り型**：`LoadedGpr`（`double`・`single`・`mixed_promote`・`mixed_reevaluate` の Exact LLT 4 変種を保持するタグ付きユニオン／`std::variant` ラッパ）および `LoadedDistanceGpr<C>`（`C = DistanceOnly | WithPoints`）。モデル種別に依存しない `predict`（`double` 拡大返却）・`n()`・`d()` を共通提供し、型付き取り出しで `FittedGpr` へ変換できる。`load` されたモデルは `Retain` ポリシーを使用し、`with_optimizer` 後の `refit` が可能である。

---

## FMT-06 カスタム拡張登録と `PersistRegistry`

非組込みの `KernelTerm` やカスタム入力・目的変数変換を保存・復元するため、`PersistRegistry` を提供する。

- **`persist_id` 制約**：空文字列および予約接頭辞 `"cxxgpr."`（`RESERVED_PREFIX`）で始まる ID を禁止する。違反時は `save` 時および登録時に `PersistFailed(InvalidPersistId)` を返す。同一カテゴリへの重複登録も `InvalidPersistId` とする。
- **登録関数**（4 カテゴリ）：`register_kernel`, `register_unfitted_input`, `register_fitted_input`, `register_unfitted_target`, `register_fitted_target`（計 5 登録関数：入力・目的変数の学習前後各 2 種＋カーネル 1 種）。
- **未対応・未登録エラー**：`persist_id()` 未実装または永続化非対応カスタム型の保存は `PersistFailed(NotPersistable)`、読込時に未登録 ID が出現した場合は `PersistFailed(UnregisteredId)` を返す。

---

## FMT-07 供給二乗距離モデルの永続化（`LoadedDistanceGpr<C>`）

`DistanceKernel<C>` のモデルでは `config.json` に `distance` セクションを追加し、`model.safetensors` にスロット別下三角テンソル `d2.<k>` を保存する。

```json
"distance": {
  "points": "distance_only",
  "slots": [ { "kind": "scalar" }, { "kind": "ard", "dims": 2 } ]
}
```

- `points` は `"distance_only"` または `"with_points"`。`DistanceKernel<WithPoints>` を `DistanceOnly` ローダー（またはその逆・座標用 `LoadedGpr`）で読んだ場合はすべて `PersistFailed(WrongModel)` を返す。
- `slots` は `DistanceKernel::slots()` の順（深さ優先初回出現順）に一致しなければならない。`distance_only` カーネルが座標依存葉（`rbf_ard`, `linear` 等）を含む場合も `PersistFailed(Config)` で拒否する。
- **`d2.<k>` 格納規則**：スカラー距離スロットでは列 $j$ の下三角要素 `j..n`（対角含む）を列順に詰めて格納し（パック位置 $p = jn - j(j-1)/2 + (i - j)$、長さ $n(n+1)/2$）、ARD 距離スロット（ヘッダ `shape` は `[dims, n(n+1)/2]`）では次元 $m \in [0, \text{dims})$ ごとに長さ $\text{packed\_len} = n(n+1)/2$ の下三角パック配列を順に連結して格納する（フラット添字 `m * packed_len + p`、内側軸が `packed_len`）。したがって NumPy 等で読む際は、2 次元列優先行列（`x`, `l`）は `flat.reshape(meta["shape"], order="F")`、距離テンソル `d2.<k>` は `flat.reshape(meta["shape"], order="C")` で解釈する（1 次元のスカラー `d2.<k>` は `F`/`C` いずれでも同一）。`MixedPrecision` モデルでは保持している高精度な `double` 側の `d²` を書き出し、読込時に `float` へ再丸めする。

```python
# Python / NumPy 最小読込規則（x, l は列優先、ARD の d2.<k> は次元ブロック連結）
def reshape_tensor(name: str, flat: "np.ndarray", shape: list[int]) -> "np.ndarray":
    order = "C" if name.startswith("d2.") else "F"
    return flat.reshape(shape, order=order)
```
- **新スロット発行**：読込時にモデル専用の新規スロットハンドル列を生成するため、保存前の `ScalarDistance` / `ArdDistance` は読込後のモデルへ使えない。利用者は `slots()` / `to_kernel()` から新スロットを取得して予測供給を作成する。
- 読込時、`d2.<k>` の有限性・非負性・対角ゼロ・形状をすべて検証し、違反時は `InvalidDistance` を返す。

---

## FMT-08 エラー種別・外部ツール互換マッピング

| 異常状況 | 返す `GprError` |
|---|---|
| ファイル開閉・読込・書込・`mmap` 失敗 | `PersistFailed { PersistErrorKind::Io, reason }` |
| JSON 構文誤り・必須キー欠落・ネスト超過・`factor_kind != "llt"` | `PersistFailed { PersistErrorKind::Config, reason }` |
| テンソル欠落・形状／dtype 不一致・8 バイト未整列・非有限テンソル | `PersistFailed { PersistErrorKind::Tensor, reason }` |
| 座標／距離／マーカー不一致、Sparse/Online モデル用ファイル指定 | `PersistFailed { PersistErrorKind::WrongModel, reason }` |
| 未登録のカスタム `persist_id` | `PersistFailed { PersistErrorKind::UnregisteredId, reason }` |
| 空または `"cxxgpr."` 接頭辞・重複登録された `persist_id` | `PersistFailed { PersistErrorKind::InvalidPersistId, reason }` |
| `persist_id` 非対応カスタムの `save` | `PersistFailed { PersistErrorKind::NotPersistable, reason }` |
| `format_version != 1` | `UnsupportedPersistVersion { found, supported: 1 }` |
| `d2.<k>` 非有限・負値・対角非零 | `InvalidDistance` |
| 保存された区間・Jitter・カーネル値が不正 | 各コンストラクタ／ファクトリの `GprError` |
