# 構成とモジュール

リポジトリの配置と命名規約の正本。新しいモジュールやディレクトリを追加する PR が同じ PR でこの文書を更新する。

## ディレクトリ

| パス | 内容 |
|---|---|
| `include/cxxgpr/<module>/` | 公開 C++23 ヘッダ。Eigen・`nlohmann/json`・OpenMP を露出させない |
| `src/<module>/` | 静的ライブラリ `cxxgpr` の実装および非公開 `detail/` ヘッダ |
| `tests/` | GoogleTest / CTest による製品試験（`tests/<module>/` に分割） |
| `tests/support/` | 確保監査（`EIGEN_RUNTIME_NO_MALLOC` 等）・比較補助などの試験共通枠 |
| `benchmarks/` | Google Benchmark による kernel / 結合計測（製品・CTest へリンクしない） |
| `cmake/` | コンパイラ方針、警告設定、vcpkg 依存・第三者通知検査、ビルド情報生成 |
| `docs/` | 設計仕様・永続化仕様・アーキテクチャ・品質保証計画・ADR・ロードマップ |
| `docs/adr/` | 設計判断の記録。1 決定 1 ファイル |
| `docs/quality/` | 品質保証計画、追跡表（自動生成）、結果変更記録 |
| `qa/numeric_canary/` | 浮動小数点環境検針カナリア実行ファイル（M0-2 で構築） |
| `qa/baselines/` | プラットフォーム別 hexfloat 基準結果（`tools/qa/results.py` だけが書く） |
| `tools/dev/` | 開発補助（MSVC 環境起動・Skill 複製・CI スコープ判定等。M0-2 で構築） |
| `tools/qa/` | 品質ゲート（`trace.py`・`arch_check.py`・`doc_check.py`・`results.py`。M0-2 で構築） |
| `.agents/skills/`、`.claude/skills/` | Skill の正本と完全一致複製 |
| `.cursor/rules/` | 開発・設計・数値・C++23 規則 |
| `out/` | ビルド生成物、vcpkg 導入領域（git 対象外） |
| `build/` | 一時作業領域（git 対象外） |
| `.dev/` | ローカル覚え書き（git 対象外） |

## 製品モジュール

依存の向きは [docs/architecture.md](architecture.md) に従う。

| モジュール | 責任 | 導入パッケージ |
|---|---|---|
| `core` | `BuildInfo`、`GprError`、`Interval`、checked 寸法、`Xoshiro256++`、`MatrixRef`/`MatrixMut`、`std::expected` 補助 | M0-2（`BuildInfo`）、M1-1 |
| `precision` | `DoublePrecision`、`SinglePrecision`、`MixedPrecision`、反復改良制御定数・エンジン | M1-2 |
| `linalg` | Eigen LLT（`uplo=Lower`）、三角求解、`JitterPolicy` 再試行ループ、OpenMP 決定的補助 | M1-3 |
| `transform` | `Identity`/`Standardize`/`MinMax`/`Pipeline`/`ColumnwiseInput`・目的変数逆変換・カスタム I/F | M1-4 |
| `kernel` | 10 組込みカーネル葉、`KernelSpec`→`CompiledKernel`、`KernelTerm<T>`、`DistanceKernel`、`KernelExp` | M2-A 〜 M2-E |
| `workspace` | `WorkspaceCore`、`FitBuffers`、`QueryWorkspace`、`HessianScratch`、`LeafCache`、`CholeskyBuffer` | M2-E 〜 M3-B |
| `optimizer` | `Objective` 階層、`Lbfgs`・`NelderMead`・`TrustRegion`・`FastSimulatedAnnealing`・`Fixed`・logit 変換・リスタート | M4-A |
| `gpr` | `GaussianLikelihood`、`Gpr`、`FittedGpr`、`GprObjective`、LOO・共分散・標本抽出・`predict_into` | M3-C、M4-B |
| `persist` | `save`/`save_with_factor`、`LoadedGpr`/`LoadedDistanceGpr`、`config.json`/`model.safetensors`、`mmap`、`PersistRegistry` | M4-C |

現在はリポジトリ基盤整備中であり、M0-2 で `core`（`build_info`）とビルド・QA 基盤を作成する。

## 命名

- **CMake ターゲット**：`cxxgpr`（静的ライブラリ）、試験ヘルパ `cxxgpr_add_test(<module>.<topic> ...)`。
- **名前空間**：`cxxgpr`（`cxxgpr::kernel`、`cxxgpr::transform`、`cxxgpr::persist`）。内部詳細は `cxxgpr::detail`。
- **C++ 識別子**：型名 `PascalCase`、関数・メソッド・ローカル変数・定数 `snake_case`、private メンバ末尾 `_`、マクロ接頭辞 `CXXGPR_`。
- **ロードマップ行 ID**：`M0-1` 〜 `M5-4`、`REL-1` 〜 `REL-2`。
- **仕様・品質 ID**：要求 `REQ-01`〜`REQ-12`、設計決定 `DES-01`〜`DES-20`、永続化 `FMT-01`〜`FMT-08`、品質 `QA-01`〜`QA-10`。
