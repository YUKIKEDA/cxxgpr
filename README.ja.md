[English](README.md) | 日本語

# cxxgpr

Rust 製ガウス過程回帰ライブラリ `gprx` の **Exact GPR** モデルを C++23 へ移植した静的ライブラリ。

> **現在地**：リポジトリ基盤（設計仕様・永続化仕様・品質計画・ロードマップ・規則）を整備している段階。ビルド／QA ゲート（M0-2）およびライブラリ本体（M1 以降）は [`docs/roadmap.md`](docs/roadmap.md) に沿って順次実装する。

## 対象範囲

`cxxgpr` は **Exact GPR**（`Gpr` / `FittedGpr`）に特化して移植する。Sparse GPR（`Sgpr` / `Svgp`）およびオンライン学習（`OnlineGpr`）はスコープ外とする。

- **推論・モデル操作**：`Gpr::fit`（`const` メンバ、失敗時も不変）、`Gpr::factor`、`predict`、`predict_with`、`predict_into`（ウォームアップ後アロケーションゼロ）、`predict_covariance`、`sample`（決定的 `Xoshiro256++`）、`loo_predict`、`neg_log_marginal_likelihood`、`value_and_gradient_into`、`hessian_into`、`get_params` / `set_params`、`refit`、`into_trainer`。
- **カーネル**：組込み 10 種の葉（`RbfKernel`、`RbfArdKernel`、`MaternKernel` $\nu \in \{1/2, 3/2, 5/2\}$、`MaternArdKernel`、`PeriodicKernel`、`RationalQuadraticKernel`、`RationalQuadraticArdKernel`、`ConstantKernel`、`LinearKernel`、`WhiteKernel`）、`+` / `*` 合成、`Custom` 葉（`KernelTerm<T>`）、`LeafCache` 増分更新、供給二乗距離（`ScalarDistance` / `ArdDistance` / `DistanceKernel`）。
- **最適化器**：外部最適化ライブラリに頼らない忠実な自前実装（`Lbfgs` / `NelderMead` / `TrustRegion` / `FastSimulatedAnnealing` / `Fixed`）、区間 logit 変換、決定的リスタート。
- **前処理・精度・方針**：入力／目的変数変換（`Identity*` / `Standardize*` / `MinMax*` / `Pipeline` / `TargetPipeline` / `ColumnwiseInput` / ユーザー定義）、`DoublePrecision` / `SinglePrecision` / `MixedPrecision<PromoteStorage | ReevaluateKernel>`、`KernelExp::{Accurate, FastApprox}`、`DistanceCachePolicy`、`CholeskyBuffer`、`JitterPolicy::{Fixed, Adaptive}`。
- **永続化**：`config.json` と `model.safetensors`（原子的一時ファイル＋リネーム、`f64` 因子 mmap 読込、`PersistRegistry`）。

## 設計の特徴

- **C++23 とエラー契約**：API 境界に例外を送出せず `[[nodiscard]] std::expected<T, GprError>` を返す。
- **列優先（Column-major）**：データ行列は `std::span<const double>`（`x`, `n_rows`, `n_cols`）として列優先で受け渡す。
- **公開ヘッダ非漏洩**：Eigen・`nlohmann::json`・OpenMP は内部実装に閉じ込め、`include/cxxgpr/` に露出させない。
- **ライセンス**：Apache-2.0（`LICENSE`）。第三者通知は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) に記録する。

## 文書

| 文書 | 内容 |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | マイルストーン・依存グラフ・現在地 |
| [`docs/design.md`](docs/design.md) | 要求・C++23 API・数理式・バッファ設計 |
| [`docs/persist-format.md`](docs/persist-format.md) | `config.json` と `model.safetensors` 仕様 |
| [`docs/architecture.md`](docs/architecture.md) | レイヤ構成と依存方向 |
| [`docs/conventions.md`](docs/conventions.md) | ディレクトリ構成と命名規約 |
| [`docs/quality/quality-plan.md`](docs/quality/quality-plan.md) | 追跡・基準結果・結果変更管理・来歴 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 開発手順とゲート |
