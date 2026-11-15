# 開発の手順

エージェント向けの入口は [AGENTS.md](AGENTS.md)、規則は `.cursor/rules/` にある。ここには開発手順をまとめる。

## 順序

Grill（設計が分岐するとき）→ Issue → ブランチ → 作業 → PR → レビュー → squash merge → 振り返り。CI は PR 上で非同期に実行し、merge に必要な成功は merge 前に確認する。

- Issue は `.github/ISSUE_TEMPLATE/` のテンプレートから作る。白紙の Issue は作らない。Issue には [docs/roadmap.md](docs/roadmap.md) の行 ID を書く。
- ブランチ名は `type/<Issue番号>-<slug>`。`main` へ直接 commit しない。
- 1 Issue は 1 PR。merge は所有者が行い、merge したブランチは消す。
- 依頼されたタスクは、検査・commit・通常の push・PR 作成まで進める。設計や範囲の判断が分かれる場合は PR 前に確認する。
- PR 作成後、エージェントは CI の完了を原則待たず、PR の URL と実行状態を報告して返す。
- commit の件名は `type(scope): subject`（英語、命令形、末尾ピリオドなし）。

| テンプレート | ラベル | commit の type |
|---|---|---|
| feat | `type:feat` | `feat` |
| bug | `type:bug` | `fix` |
| refactor | `type:refactor` | `refactor` |
| test | `type:test` | `test` |
| design | `type:docs` | `docs` |
| task | `type:chore` | `chore` `build` `ci` |
| spike | `type:spike` | `chore` |

## 振り返り

merge 後、PR に 3 点を書く：得た教訓、それを書き込むファイル、追加のリファクタリングの要否。教訓があれば `chore` の Issue を立て、規則・Skill・仕様・ADR へ書く。

## ゲート

差分と依存先に影響する検査を選ぶ。作業中は局所試験を使い、全体検証が必要な場合は編集を終えた最後に 1 回へまとめる。ゲート群は M0-2（Issue #2）で `justfile` と CI へ実装する。

| コマンド | 内容 | ゲート |
|---|---|---|
| `just gate` | 構成、ビルド、試験（Windows Debug/Release、Linux Debug/Release/Sanitize） | はい |
| `just format-check` | `clang-format` の差分がないこと | はい |
| `just lint` | `clang-tidy`（WarningsAsErrors）の指摘がないこと | はい |
| `just lint-python` | Python ツール：`ruff`（検査と整形）と `pyright` strict | はい |
| `just trace-check` | 仕様 ID（`REQ-`/`DES-`/`FMT-`/`QA-`）の追跡表整合 | はい |
| `just arch-check` | 依存方向および `include/cxxgpr/` の第三者ヘッダ非露出 | はい |
| `just doc-check` | 公開 Doxygen コメント有無と内部 ID 非混入 | はい |
| `just skills-check` | `.claude/skills` が `.agents/skills` の一致複製であること | はい |
| `just results-check` | 基準結果（`qa/baselines/`）との一致 | はい |
| `just format` / `just skills-sync` | 整形、Skill の複製 | いいえ |

## 並列に進める

着手前にタスクの先行条件・成果物・編集ファイル・共有状態を整理する。依存する作業は順に行い、互いの結果を必要とせず共有ファイルを衝突させない作業は並列に進める。同一ファイルの編集とそのビルド・解析を同時に行わない。

## 言語

文書（`docs/`、`AGENTS.md`、`CONTRIBUTING.md`、`.cursor/rules/`）、Issue・PR の本文は日本語（常体）。コード、Doxygen コメント、識別子、commit 件名、PR 題名は英語。`README.md`（英語）と `README.ja.md`（日本語）は両方維持する。
