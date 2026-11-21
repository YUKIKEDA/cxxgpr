# モデルファイル形式

このページは、`cxxgpr::persist::save`、`cxxgpr::persist::save_with_factor`、`cxxgpr::persist::LoadedGpr::load` が、厳密なガウス過程モデルについて読み書きするディスク形式を定める。保存したモデルを他のツールで調べるとき、他の言語で読み取り側を書くとき、ライブラリの 2 つの版がファイルをやり取りできるかを判断するときに使う。保存と読み込みの手順は {doc}`../how-to/persist-models` を参照。

## ディレクトリ構成

モデルは、次の 2 つのファイルだけを含む 1 つのディレクトリである。

| ファイル | 内容 |
|---|---|
| `config.json` | UTF-8 の JSON。モデルの構造、精度、ポリシー、区間付きのハイパーパラメータ、変換の状態 |
| `model.safetensors` | [safetensors](https://github.com/huggingface/safetensors) コンテナ形式のバイナリテンソル。学習データと、任意でコレスキー因子と重み |

保存の最中は、同じディレクトリに `config.json.tmp` と `model.safetensors.tmp` が一時的に存在することがある。保存が完了すると、上の 2 つのファイルだけが残る。

## バージョンと互換性

| 項目 | 値 |
|---|---|
| 形式のバージョン | `1`（定数 `cxxgpr::persist::FORMAT_VERSION`） |
| バージョンのフィールド | `config.json` の `format_version`（整数） |
| 読み取り側の動作 | `format_version` が `1` 以外なら `GprErrorCode::UnsupportedPersistVersion`。見つかった版と対応する版を持つ |
| 移行 | なし。版どうしの変換はない |
| 未知の最上位キー | 読み込み時に無視する |
| 書き込み側の動作 | 完成した一時ファイルを最終的な名前へリネームして書き込む |

他のモデル族は、`PersistErrorKind::WrongModel` で拒否する。対象は、疎なモデルとオンラインモデル（`model` キー）、`"llt"` 以外の分解（`factor_kind`）である。供給された二乗距離で学習したモデルは `distance` キーを持ち、`LoadedGpr` ではなく `LoadedDistanceGpr` が読む。

## `config.json`

ファイルは 1 つの JSON オブジェクトである。書き込み側は、2 スペースのインデントとキーの整列を付けて出力する。読み取り側は、有効な JSON ならどのレイアウトでも受け付けなければならない。

### 最上位のキー

| キー | 型 | 必須 | 欠けているときの読み込み値 | 意味 |
|---|---|---|---|---|
| `format_version` | 整数 | はい | — | 形式のバージョン。`1` |
| `n` | 整数、$> 0$ | はい | — | 学習点の数 |
| `d` | 整数、$> 0$（距離のみのモデルは 0） | はい | — | 特徴量の数 |
| `has_factor` | 真偽値 | はい | — | `model.safetensors` に `l` と `alpha` があれば `true` |
| `factor_kind` | 文字列 | はい | — | `"llt"`（コレスキー $LL^\top$）でなければならない |
| `factor_jitter` | 数値、$\ge 0$ | いいえ | `0.0` | モデルを分解したときに採用された対角ジッタ |
| `precision` | 文字列 | いいえ | `"double"` | `"double"`、`"single"`、`"mixed"` |
| `residual` | 文字列 | いいえ | `"promote_storage"` | `"mixed"` のときだけ。`"promote_storage"` か `"reevaluate_kernel"` |
| `math` | 文字列 | いいえ | `"accurate"` | `"accurate"` か `"fast_approx"`（`KernelExp`） |
| `distance_cache` | 文字列 | いいえ | `"always"` | `"always"` か `"never"`（`DistanceCachePolicy`） |
| `kernel` | オブジェクト | はい | — | カーネルの木 |
| `likelihood` | オブジェクト | はい | — | `{"noise_variance": <区間付きの数値>}` |
| `jitter` | オブジェクト | はい | — | コレスキー分解のジッタポリシー |
| `x_unfitted` | 変換 | はい | — | 学習前の入力変換の設定 |
| `x_transform` | 変換 | はい | — | 学習済みの入力変換 |
| `y_unfitted` | 変換 | はい | — | 学習前の目的変数変換の設定 |
| `y_transform` | 変換 | はい | — | 学習済みの目的変数変換 |
| `distance` | オブジェクト | 供給距離のモデルのみ | — | モードと距離スロット。下記を参照 |
| `save_id` | 文字列（小文字 16 進 32 桁） | いいえ | なし（ペアリング検査なし） | `model.safetensors` の `cxxgpr.save_id` と対になる内容ハッシュ識別子 |

書き込み側は、`precision` が `"mixed"` でない限り `residual` を出力しない。`precision` と `residual` の組が、モデルの `PersistPrecisionKind` を決める。

| `precision` | `residual` | `PersistPrecisionKind` | ポリシー |
|---|---|---|---|
| `"double"` | — | `Double` | `DoublePrecision` |
| `"single"` | — | `Single` | `SinglePrecision` |
| `"mixed"` | `"promote_storage"` | `MixedPromote` | `MixedPrecision<PromoteStorage>` |
| `"mixed"` | `"reevaluate_kernel"` | `MixedReevaluate` | `MixedPrecision<ReevaluateKernel>` |

浮動小数点数は、正確な値を読み戻せるだけの桁数で書く。整数値の数も、JSON の浮動小数点数として書く（`0.0`、`100000.0`）。

### 区間付きの数値

ハイパーパラメータは、**ユーザー単位の値**（対数ではない）と、その開区間の組で保存する。

```json
{ "value": 1.1164913909984453, "lo": 1e-05, "hi": 100000.0 }
```

制約は $0 < lo < value < hi$ である。

### カーネルの木

カーネルは、ノードの種類を表すキーを 1 つだけ持つ JSON オブジェクトである。

| ノード | 本体 |
|---|---|
| `"sum"`、`"product"` | `{"left": <kernel>, "right": <kernel>}` |
| `"rbf"` | `{"length_scale": <区間付きの数値>}` |
| `"rbf_ard"` | `{"length_scales": [<区間付きの数値>, ...]}` |
| `"matern"` | `{"nu": "1/2" \| "3/2" \| "5/2", "length_scale": <区間付きの数値>}` |
| `"matern_ard"` | `{"nu": ..., "length_scales": [<区間付きの数値>, ...]}` |
| `"periodic"` | `{"length_scale": <区間付きの数値>, "period": <区間付きの数値>}` |
| `"rq"` | `{"length_scale": <区間付きの数値>, "alpha": <区間付きの数値>}` |
| `"rq_ard"` | `{"length_scales": [<区間付きの数値>, ...], "alpha": <区間付きの数値>}` |
| `"constant"` | `{"value": <区間付きの数値>}` |
| `"linear"` | `{"variance": <区間付きの数値>}` |
| `"white"` | `{"variance": <区間付きの数値>}` |
| `"custom"` | `{"persist_id": <文字列>, "state": <JSON>}` |

モデルの対数パラメータベクトルでの並び順は、葉を深さ優先・左から右にたどった順で、各葉の中ではこの表に示した順である。どの構造（カーネル、パイプライン、列ごとの変換）でも、入れ子の深さは 128 段を超えてはならない。

### 尤度とジッタ

`likelihood` は、ノイズ分散を区間付きの数値で持つ。`{"noise_variance": {"value": ..., "lo": ..., "hi": ...}}` である。

`jitter` は、次の 2 つのポリシーのどちらかを持つ。

| ポリシー | 本体 |
|---|---|
| 固定オフセット | `{"fixed": {"jitter": <数値>}}` |
| 適応的 | `{"adaptive": {"initial": <数値>, "multiplier": <数値>, "max_retries": <整数>, "max_jitter": <数値>}}` |

値は、`JitterPolicy` のコンストラクターと同じ検査を通る。

### 変換

変換は、文字列 `"identity"` か、キーが 1 つのオブジェクトである。

| ノード | 未学習の本体（`x_unfitted`、`y_unfitted`） | 学習済みの本体（`x_transform`、`y_transform`） |
|---|---|---|
| `"identity"`（オブジェクトではなく文字列） | — | — |
| `"standardize"` | `{}` | 入力: `{"mean": [...], "std": [...]}`、目的変数: `{"mean": <数値>, "std": <数値>}` |
| `"min_max"` | `{"range_lo": <数値>, "range_hi": <数値>}` | 入力: `{"data_min": [...], "data_max": [...], "range_lo": ..., "range_hi": ...}`、目的変数: 同じキーでスカラー |
| `"pipeline"` | `{"steps": [<transform>, ...]}` | `{"steps": [<transform>, ...]}` |
| `"columnwise"`（入力のみ） | `{"maps": [<transform>, ...]}` | `{"maps": [<transform>, ...]}` |
| `"custom"` | `{"persist_id": <文字列>, "state": <JSON>}` | `{"persist_id": <文字列>, "state": <JSON>}` |

`custom` ノードでは、未学習の変換と学習済みの変換は別々の登録になる。未学習側の `persist_id` は `register_unfitted_input` か `register_unfitted_target` で登録したファクトリへ、学習済み側の `persist_id` は `register_fitted_input` か `register_fitted_target` で登録したファクトリへ渡される。

`custom` ノードの `state` には、構成要素の `persist_state_json` が返した文字列を収める。JSON のオブジェクト、配列、数値、`true`、`false`、`null` として解析できる文字列は、その JSON の値として保存し、ファクトリには値を詰めて再シリアライズした文字列を渡す。これは一般に元の文字列と同じではない。数値は同じ `double` に読み戻せる最短の形に書き直され（`2.500000` → `2.5`、`1e2` → `100.0`、`-0` → `0`、`12345678901234567890123` → `1.2345678901234568e+22`）、文字列の外の空白は除かれ、オブジェクトのキーは整列され、重複したキーは最後の 1 つだけが残る。それ以外の文字列は、`"abc"` のような JSON の文字列リテラルも含めて JSON の文字列として保存し、ファクトリには元の文字列が引用符も含めて 1 バイトも違わずに渡る。この文字列は有効な UTF-8 でなければならず、そうでない文字列を保存すると `PersistErrorKind::NotPersistable` になる。

### 供給距離のモデル

供給された二乗距離で学習したモデル（`FittedDistanceGpr`）は、上と同じセクションを持ち、次の点が異なる。

```json
"distance": {
  "points": "distance_only",
  "slots": [ { "kind": "scalar" }, { "kind": "ard", "dims": 2 } ]
}
```

- `distance.points` は `"distance_only"`（このとき `d` は `0`）か `"with_points"`（このとき `d > 0` で、`x` も保存する）。もう一方のモードのローダーは、`PersistErrorKind::WrongModel` でファイルを拒否する。
- `distance.slots` は、カーネルに最初に現れる深さ優先の順で距離スロットを並べる。スロットは `{"kind": "scalar"}` か `{"kind": "ard", "dims": <整数>}` である。
- カーネルの木では、距離スロットを読む葉は `{"distance": {"slot": <slots の添字>, "leaf": <カーネルの葉>}}` と書く。スロット番号は `distance.slots` の順に現れる。木を深さ優先・左から右にたどると、最初の新しい番号が `0`、次の新しい番号が `1` となる（2 つの葉が同じスロットを読むときは番号が繰り返される）。`d2.<k>` はスロット `k` のものなので、これと違う番号付けは `PersistErrorKind::Config` で拒否する。スカラースロットが取るのは `rbf`、`matern`、`periodic`、`rq`、`custom`、ARD スロットが取るのは `dims` 個の長さスケールを持つ `rbf_ard`、`matern_ard`、`rq_ard` である。それ以外の葉は、上の表のとおりに書く座標の葉である。距離のみのモデルが持てる座標の葉は、`constant` と `white` だけである。
- `distance_cache` は `"always"` と書き、使わない。
- `"distance_only"` では、`x_unfitted` と `x_transform` はどちらも `"identity"` である。それ以外は `PersistErrorKind::Config` で拒否する。

**距離のテンソル。** スロット $k$ のテンソル `d2.<k>` は、修復済みの学習距離の下三角（対角を含む）を持つ。$i \ge j$ の要素 $(i, j)$ の詰めた位置は $p = j n - j(j-1)/2 + (i - j)$ で、詰めた長さは $n(n+1)/2$ である。ARD スロットは、次元ごとに詰めた配列を順に連結する。次元 `m` の平坦な添字は `m * packed_len + p` である。このため `x` や `l` と違って、`d2.<k>` は `reshape(shape, order="C")` で読む（1 次元であるスカラースロットでは、順序は結果に影響しない）。`"mixed"` の精度のモデルは、予測に使う精度と同じ `F32` の表を保存する。

読み込み時には、保存した表が、自分で与える表と同じ検査を通る。有限であること、非負であること、対角が 0 であることである。違反は `InvalidDistance`。非有限の要素は、その前に壊れたテンソルとして拒否する。

### 例

カーネルが `Constant * RbfArd`（学習点 24 個、特徴量 1 個）で、`StandardizeInput` と `StandardizeTarget` を使い、倍精度で `save_with_factor` により保存したモデルである。

```json
{
  "d": 1,
  "distance_cache": "always",
  "factor_jitter": 0.0,
  "factor_kind": "llt",
  "format_version": 1,
  "has_factor": true,
  "jitter": {
    "fixed": {
      "jitter": 0.0
    }
  },
  "kernel": {
    "product": {
      "left": {
        "constant": {
          "value": {
            "hi": 100000.0,
            "lo": 1e-05,
            "value": 1.9301390140085495
          }
        }
      },
      "right": {
        "rbf_ard": {
          "length_scales": [
            {
              "hi": 100000.0,
              "lo": 1e-05,
              "value": 1.1164913909984453
            }
          ]
        }
      }
    }
  },
  "likelihood": {
    "noise_variance": {
      "hi": 100000.0,
      "lo": 1e-05,
      "value": 0.0017863553180453662
    }
  },
  "math": "accurate",
  "n": 24,
  "precision": "double",
  "x_transform": {
    "standardize": {
      "mean": [
        0.5
      ],
      "std": [
        0.300964632714423
      ]
    }
  },
  "x_unfitted": {
    "standardize": {}
  },
  "y_transform": {
    "standardize": {
      "mean": 10.373560792920012,
      "std": 2.136314909314304
    }
  },
  "y_unfitted": {
    "standardize": {}
  }
}
```

## `model.safetensors`

ファイルは safetensors の仕様に従う。

| オフセット | サイズ | 内容 |
|---|---|---|
| 0 | 8 バイト | ヘッダーのバイト長 $H$。リトルエンディアンの `uint64` |
| 8 | $H$ バイト | 各テンソルの名前を `{"dtype", "shape", "data_offsets"}` に対応づける JSON オブジェクト |
| $8 + H$ | 残り | テンソルのバイト列。`data_offsets` は、この位置を基準とした `[begin, end)` |

書き込み側は、$8 + H$ が 8 の倍数になるようにヘッダーを詰め物で埋め、各テンソルを 8 の倍数のオフセットから始める。`__metadata__` には `config.json` の `save_id` と一致する `{"cxxgpr.save_id": "<小文字16進32桁>"}` を書く。数値はすべて、リトルエンディアンの IEEE 754 である。

### テンソル

| 名前 | `shape` | `dtype` | 存在する条件 | 内容 |
|---|---|---|---|---|
| `x` | `[n, d]` | `F64` | 常に（距離のみのモデルは $d = 0$ なので省く） | 入力変換の前の学習入力 |
| `y` | `[n]` | `F64` | 常に | 目的変数変換の前の学習目的変数 |
| `l` | `[n, n]` | `double`: `F64`、`single` と `mixed`: `F32` | `has_factor` が `true` | $K + (\sigma_n^2 + j) I$ の下三角コレスキー因子 $L$。上三角は 0 で埋める |
| `alpha` | `[n]` | `single`: `F32`、`double` と `mixed`: `F64` | `has_factor` が `true` | 変換後の目的変数空間での重み $\alpha = (K + (\sigma_n^2 + j) I)^{-1}\mathbf{y}'$。$j$ は `factor_jitter` |
| `d2.<k>` | スカラースロット: `[n(n+1)/2]`、ARD スロット: `[dims, n(n+1)/2]` | `double`: `F64`、`single` と `mixed`: `F32` | 供給距離のモデル。スロット $k$ ごとに 1 つ | スロット $k$ の、検査済みの学習距離の下三角（下記） |

すべてのテンソルのすべての要素は、有限でなければならない。

### レイアウトとバイト順

行列 `x` と `l` は**列優先**（Fortran 順）で格納する。要素 $(i, j)$ は、平坦なインデックス $j \cdot n + i$ にある（`x` は $n$ 行 $d$ 列、`l` は $n$ 行 $n$ 列）。1 次元のテンソルは、ただの配列である。

学習点が 6 個で特徴量が 1 個のモデルを、倍精度で `save_with_factor` により保存すると、ヘッダーは次のようになる（整形して示す。ファイルでは余分な空白なしで格納される）。

```json
{
  "alpha": { "dtype": "F64", "shape": [6],    "data_offsets": [384, 432] },
  "l":     { "dtype": "F64", "shape": [6, 6], "data_offsets": [96, 384] },
  "x":     { "dtype": "F64", "shape": [6, 1], "data_offsets": [0, 48] },
  "y":     { "dtype": "F64", "shape": [6],    "data_offsets": [48, 96] }
}
```

`"mixed"` のモデルでは、同じヘッダーで `l` が `F32`（`data_offsets` の幅は $4 n^2$ バイト）、`alpha` が `F64` になる。

### Python での読み取り

```python
import json, struct
import numpy as np

def read_safetensors(path):
    raw = open(path, "rb").read()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    data = memoryview(raw)[8 + header_len :]
    dtypes = {"F64": "<f8", "F32": "<f4"}
    tensors = {}
    for name, meta in header.items():
        lo, hi = meta["data_offsets"]
        flat = np.frombuffer(data[lo:hi], dtype=dtypes[meta["dtype"]])
        tensors[name] = flat.reshape(meta["shape"], order="F")  # column-major
    return tensors
```

`order="F"` を指定すると、`x` と `l` は保存時の形状と意味のまま、ビット単位で正確に復元される。1 次元のテンソル `y` と `alpha` では、指定しても結果は変わらない。

## 読み込み時の動作

`LoadedGpr::load` と `LoadedDistanceGpr::load` は、ディレクトリを信頼できない入力として扱い、次の検査を行う。`LoadedGpr::load` は `distance` キーのあるファイルを、`LoadedDistanceGpr::load` は `distance` キーのないファイルや、`distance.points` が別のファイルを拒否する。どちらも `PersistErrorKind::WrongModel` である。

1. 2 つのファイルを開いて読む。失敗は `PersistErrorKind::Io`。
2. JSON の深さが 128 以下で、テキストが解析でき、ルートがオブジェクトで、`model` キーがない。`model` キーや `distance` キーがあれば `PersistErrorKind::WrongModel`。そのほかの構文やスキーマの誤りは `PersistErrorKind::Config`。
3. `format_version` が存在し、整数である。`1` 以外の値は `GprErrorCode::UnsupportedPersistVersion`。
4. `factor_kind` が `"llt"` である。そうでなければ `PersistErrorKind::WrongModel`。
5. 各テンソルのヘッダーのエントリが、既知の `dtype`（`F64` か `F32`）、想定した形状、アラインメントの合った開始位置を持ち、バイト長が形状と要素サイズの積に等しい。失敗は `PersistErrorKind::Tensor`。
6. すべてのテンソル要素が有限である。失敗は `PersistErrorKind::Tensor`。
7. ハイパーパラメータ、区間、ジッタ、変換の統計量が、それぞれのコンストラクターの検査を通る。失敗は、そのコンストラクターのエラー（たとえば `GprErrorCode::InvalidHyperparameter` や `GprErrorCode::InvalidInterval`）。
8. `custom` ノードをレジストリで検索する。未知の `persist_id` は `PersistErrorKind::UnregisteredId`。
9. `has_factor` が `false` なら、保存されたデータとハイパーパラメータから分解を再計算する。失敗は `GprErrorCode::CholeskyFailed`。

`has_factor == true` のとき、ローダーは `l` と `alpha` を保存されたとおりに使い、再分解しない。`double` 精度では、`l` を読み取り専用のメモリマップへ直接結びつける。それ以外の `l` は、メモリへコピーする。

## エラー

| 条件 | エラー |
|---|---|
| ファイルを開けない、読めない、書けない、マップできない | `PersistFailed`、`PersistErrorKind::Io` |
| JSON が不正、キーが欠けている・型が違う、入れ子が 128 段を超える（読込時、または保存で書く `config.json`） | `PersistFailed`、`PersistErrorKind::Config` |
| テンソルがない、形状や `dtype` が違う、アラインメントが合わない、有限でない要素がある | `PersistFailed`、`PersistErrorKind::Tensor` |
| `model` キーがある、`LoadedGpr` で `distance` キーがある、`LoadedDistanceGpr` で `distance` キーがないか `distance.points` が別、`factor_kind` が `"llt"` 以外、`into_fitted` で精度が合わない | `PersistFailed`、`PersistErrorKind::WrongModel` |
| ファイルの `persist_id` に登録されたファクトリがない | `PersistFailed`、`PersistErrorKind::UnregisteredId` |
| `persist_id` が空、`cxxgpr.` で始まる、有効な UTF-8 でない、または二重登録 | `PersistFailed`、`PersistErrorKind::InvalidPersistId` |
| `persist_id` のない自作の構成要素、または文字列として保存する状態が有効な UTF-8 でない構成要素を保存した | `PersistFailed`、`PersistErrorKind::NotPersistable` |
| 保存した距離の表に負の要素や 0 でない対角がある | `InvalidDistance` |
| `format_version` が `1` 以外 | `UnsupportedPersistVersion`（`found_version` と `supported_version` を持つ） |
