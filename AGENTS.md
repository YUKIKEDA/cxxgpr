# cxxgpr：エージェントの入口

**Exact GPR** モデルを提供する C++23 の静的ライブラリプロジェクト。品質・設計基準を学習用デモや簡易版で妥協しない。人間向けの手順は [CONTRIBUTING.md](CONTRIBUTING.md)、守るべき規則は `.cursor/rules/` にある。

## 正本

| 知りたいこと | 場所 |
|---|---|
| 利用者向け文書（Diataxis、日英） | `docs/en/`、`docs/ja/`（例は `docs/examples/`）。`just docs` / `just docs-coverage` |
| 作業の順序と現在地 | [docs/internal/roadmap.md](docs/internal/roadmap.md) |
| 設計仕様（要求・API・アルゴリズム・バッファ） | [docs/internal/design.md](docs/internal/design.md) |
| 永続化仕様（`config.json`・`model.safetensors`） | [docs/internal/persist-format.md](docs/internal/persist-format.md) |
| レイヤ構成とモジュール依存の向き | [docs/internal/architecture.md](docs/internal/architecture.md) |
| ディレクトリ構成と命名規約 | [docs/internal/conventions.md](docs/internal/conventions.md) |
| 決定の理由（ADR） | [docs/internal/adr/](docs/internal/adr/0001-exact-gpr-cpp23-scope.md) |
| 品質保証（追跡・基準結果・変更管理・来歴） | [docs/internal/quality/quality-plan.md](docs/internal/quality/quality-plan.md) |
| ビルド・試験・ゲートのレシピ | `justfile`（M0-2 で追加） |

現在の行のIDは [docs/internal/roadmap.md](docs/internal/roadmap.md) だけが持つ。`.dev/` は手元の一時領域であり、決定を置かない。

## スコープの基本原則

- 対象は **Exact GPR**（`Gpr` / `FittedGpr`、全 10 種の組込みカーネル葉・合成・`Custom`、供給二乗距離 `DistanceKernel`、前処理変換、`Double`/`Single`/`MixedPrecision`、`KernelExp`、5 種最適化器、`persist`）に限る。
- `Sgpr` / `Svgp` / `OnlineGpr`（`LdltStore` / `PointId` / `ldlt` persist）と誘導点座標微分は対象外とする。
- 公開ヘッダ（`include/cxxgpr/`）へ Eigen・nlohmann/json・OpenMP の型やヘッダを露出させない。例外は API 境界へ送出せず、`std::expected<T, GprError>` を返す。入力行列は列優先（column-major）とする。

## 必ず読む規則

作業に合う規則を読む。`alwaysApply` の規則は全ての変更に適用する。

| 規則 | 一行 |
|---|---|
| `workflow` | Grill、Issue、ブランチ、PR、人の merge。commit 前にブランチを確かめる |
| `consent` | 質問は決定ではない。何を利用者が決め、何を自分で決めるか |
| `git` | 通常の commit・push は確認なしで PR まで進める。破壊的操作は名指しで |
| `conventional-commits` | `type(scope): subject`。`perf` は使わない |
| `pull-requests` | テンプレート見出し。Verification は完了条件の各行から書く |
| `github-closing` | 番号の前の閉じる動詞は否定文でも Issue を閉じる |
| `review-comments` | 指摘は、それが書かれた head についてのもの |
| `scope` | 範囲は要求が決める。半端な能力を出さない。状態を分けて言う |
| `defer` | 先送りは、名前の付いたロードマップの行だけ |
| `pace` | 依存を整理し独立作業は並列に進める。CI 完了を原則待たない |
| `open-ends` | 未了の作業や説明できない数値を残したまま PR を出さない |
| `spec-checks` | 試験は、仕様の文の全体を実装する |
| `similar-findings` | 同じ種類の問題を、リポジトリ全体を検索して直す |
| `record-corrections` | 訂正は、同じ変更の中で規則・Skill・仕様・ADR へ書く |
| `scratch` | 一時ファイルを残さない。未追跡ファイルをまとめて消さない |
| `no-dead-code` | 取り除いた経路だけに仕えていたものを全て取り除く |
| `layout` | 置き場所。必要になる前に作らない。役目を終えたものを残さない |
| `design-docs` | 仕様の置き場所と変え方。実装が先に仕様から外れない |
| `dev-docs` | 図は Mermaid、数式は LaTeX。Doxygen コメントは英語で内部 ID を書かない |
| `quality` | 試験に決定 ID のタグ・コメント。基準結果の変化には結果変更記録。環境の認定 |
| `publish` | Apache-2.0 準拠、公開ヘッダの非漏洩、第三者通知、英語 README |
| `evidence` | 凍結した基準・参照・結果を書き換えない |
| `reference-data` | 独立参照（参照実装の出力／解析解／R&D スクリプト）との比較・不確かさ |
| `api-contract` | 所有結果、`Gpr::fit` / `factor` は `const`、`std::expected`、非例外、決定性 |
| `types` | 不正状態を表現できなくする（強い型、区間型、学習前後分離） |
| `cpp` | C++23、未定義動作禁止、検査付き寸法、`clang-tidy` / `clang-format`、命名規約 |
| `numerics` | 非有限検査、Jitter とノイズの分離、混合精度 IR、補償と安定化 |
| `numeric-tests` | 独立参照、固定許容差、変異確認、性質試験（並列ビット一致・確保 0） |
| `testing` | GoogleTest、CTest ラベル（`unit` / `property` / `negative` / `integration` / `verification`） |
| `performance` | ホットパス確保 0（`EIGEN_RUNTIME_NO_MALLOC` + 監査）、決定的並列畳み込み |
| `bench` | Google Benchmark の測り方・比べ方・前後の数値報告 |
| `python` | `tools/` の Python は `ruff` と `pyright` strict を通す |
| `dependencies` | vcpkg manifest で版固定。公開ヘッダ非漏洩と第三者通知 |
| `review` | レビューする側の範囲と指摘の書き方 |
| `language` | 文書・規則・Issue・PR 本文は日本語（常体）、コードと件名は英語 |
| `powershell-shell` | Windows のビルドとシェルスクリプトの決まり |

## Skills

正本は `.agents/skills/`。`.claude/skills/` は同一の複製で、Skill 変更時は両方を更新する（M0-2 の `just skills-sync` / `just skills-check`）。

- `grilling`：設計が分岐するとき、Issue の前に仕様を詰める。
- `github-closing`：`gh pr create` 前に、PR が終えるものだけを閉じることを確かめる。
- `natural-japanese`：日本語文書を常体で自然に推敲・検証する。
- `publish-review`：公開基準（ライセンス、API 非漏洩、文書整合）の最終審査を行う。

## 環境

Windows 11（PowerShell 7、MSVC + Ninja + vcpkg `VCPKG_ROOT`）および Linux（GCC 14 + Ninja + vcpkg）。詳細なビルド構成とゲートは M0-2（`CMakePresets.json` / `justfile` / `.github/workflows/ci.yml`）で整備する。
