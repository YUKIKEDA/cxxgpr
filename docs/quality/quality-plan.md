# 品質保証計画

`cxxgpr` の任意のバージョン・出力結果について、次を検証記録とツール出力から追跡・再現できる状態を保つ。

1. この仕様（`REQ-*` / `DES-*` / `FMT-*`）はどこでテストされ、何を保証しているか。
2. この数値結果はどのバージョン、どの commit、どのコンパイラ／ビルド設定から出たか。
3. バージョンや依存を更新したとき、基準値がなぜ・どれだけ変化したか。

決定には ID（`QA-01`〜`QA-10`）を付け、[追跡表](traceability.md) によって仕様・テスト間で整合管理する。

## QA-01 追跡

要求（`REQ-*`）、設計決定（`DES-*`）、永続化契約（`FMT-*`）、品質契約（`QA-*`）は各仕様文書で定義し、テストおよび検証記録と自動対応づける。

- 追跡表 `docs/quality/traceability.md` は `tools/qa/trace.py matrix` が生成する。手で直接書き換えない。
- GoogleTest のテストケースは検証対象の仕様 ID をメタデータまたは近傍コメント（`// Spec: REQ-01, DES-05` / `RecordProperty("spec", "DES-05")`）で明示する。
- 公開ヘッダ `include/cxxgpr/` の Doxygen コメントには内部仕様 ID（`REQ-`・`DES-`・`FMT-`・`QA-`・`M*-`）を混入させない（`just doc-check` が拒否する）。
- ゲート `just trace-check` は未定義 ID への参照および最新の `traceability.md` との差分を失敗として検出する。

## QA-02 検証記録

重要な数値認定・独立比較のまとめは `docs/verification/<M-ID>/` に記録を残す（該当記録を作成する PR でディレクトリを追加する）。採取済みの検証記録は書き換えず、訂正時は新しい記録を追加する。

## QA-03 基準結果（hexfloat ベースライン）

数値を返す能力について、固定入力・丸めなしの 16 進浮動小数点表記（hexfloat、`std::hexfloat`）で基準結果を保存する。

- 配置先は `qa/baselines/<case>/<platform>.json` とする。
- 最初の基準ケースは `numeric-canary` とし、IEEE-754 演算・補償加算・Eigen LLT 分解・初等数学関数・決定的乱数を評価して、コンパイラ・フラグ・Eigen・標準数学ライブラリの微細な変化を検知する。
- 基準結果の書込みは `tools/qa/results.py accept` のみが行う。

## QA-04 結果の変更管理

**認定済みプラットフォーム上の基準結果が 1 ビットでも変化する変更は、[結果変更記録](result-changes/README.md) なしに merge しない。**

- `just results-check` は現在のビルド出力と `qa/baselines/` を比較し、同一プラットフォームでのビット一致を要求する。
- 異なるプラットフォーム（Windows MSVC vs Linux GCC 等）間の差異（例：標準数学関数の 1 ulp 差）は許容確認のうえプラットフォーム別に基準認定し、記録を残す。

## QA-05 版の意味

バージョンは `MAJOR.MINOR.PATCH` とする。

| 更新桁 | 含めてよい変更 |
|---|---|
| `PATCH` | 基準結果を変えないバグ修正・内部改善（全認定プラットフォームでビット一致） |
| `MINOR` | 数値予算内の改善・機能追加（結果変更がある場合は記録必須）、後方互換な設定追加 |
| `MAJOR` | 永続化フォーマット・公開 API・数値予算の破壊的変更 |

公開前は `0.x.y-dev` として扱い、v1.0 まで公開 API の見直し余地を保つ。

## QA-06 来歴（Provenance）

`cxxgpr::build_info()`（M0-2 で実装）が、バージョン、Git commit SHA、`uncommitted_changes` フラグ、コンパイラ名とバージョン、ターゲット OS／アーキテクチャ、ビルド構成（Debug / Release / Sanitize）、浮動小数点ポリシー（`/fp:strict` 等）、依存パッケージ固定版を機械可読で返す。基準結果採取は原則 `uncommitted_changes == false` のツリーで行う。

## QA-07 環境の認定

プラットフォーム鍵は `<os>-<arch>-<compiler><major>.<minor>`（および関連 ABI/命令セット設定）とする。未認定プラットフォームや更新されたコンパイラ（CI ランナー更新等）で `just results-check` が実行された場合は即座に失敗させ、原因確認と認定記録のうえで基準を追加する。

## QA-08 不具合と既知の問題

不具合 Issue には影響範囲（カーネル・精度ポリシー・永続化互換性）と基準結果への影響を記載する。既知の制限は `CHANGELOG.md` および `README.md` / `README.ja.md` へ明記する。

## QA-09 公開前の品質ゲート

REL-2 公開審査前に次を全て満たす：全 `just` ゲート通過、`include/cxxgpr/` の第三者ヘッダ完全非露出、`gprx` Exact GPR との golden 数値照合（fixed / fit / LOO / predict / persist）合格、ホットパス確保ゼロ試験合格、`THIRD_PARTY_NOTICES.md` と `LICENSE` の整合確認。

## QA-10 役割と承認

実装とレビューは観点を分けて検証する。仕様変更（`design` Issue）・結果変更記録の承認・公開タグ打ちは所有者の承認を経る。エージェントは単独で merge やリリース公開を行わない。
