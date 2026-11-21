# 変更履歴

公開した版を、新しい順に [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) の形式で記す。版の意味は [品質保証計画 QA-05](docs/internal/quality/quality-plan.md) が定める。PATCH は基準結果を変えない変更、MINOR は数値予算内の結果変更を含む変更、MAJOR は形式・API・予算を超える変更とする。

公開した版は、まだ無い。

## [Unreleased]

### 追加

- GitHub Release のワークフロー（`v*` タグで、Windows・Linux のパッケージと文書サイトを集めて下書きの Release を作る）と、リリース手順（`docs/internal/release.md`）。README（英・日）を読みやすい構成へ書き直した。
- 利用者向け文書（英語・日本語）：Diataxis の 4 区分、全公開 API の使用方法、Doxygen + Sphinx（Furo）のサイト、コンパイル・実行される例（`docs/examples/`）、API 網羅検査（`just docs-coverage`）。開発者向け文書は `docs/internal/`（英語）へ移動。
- Exact GPR（`Gpr` / `FittedGpr`）、全 10 種の組込みカーネル葉・合成・`Custom` 葉、`LeafCache` 増分更新、供給二乗距離（`DistanceKernel`）。
- 5 種の自己完結型最適化器（`Lbfgs` / `NelderMead` / `TrustRegion` / `FastSimulatedAnnealing` / `Fixed`）。
- 入力／目的変数前処理変換、`DoublePrecision` / `SinglePrecision` / `MixedPrecision`、`KernelExp` ポリシー。
- `config.json` / `model.safetensors` による永続化（原子保存、`f64` 因子 mmap 読込、`PersistRegistry`）。
- CMake `install` / `find_package(cxxgpr CONFIG)` / `FetchContent` パッケージ体裁と `0.1.0` 版統一（`REL-1`）。
- 総合検証スイート（`path_matrix`・golden 数値照合・ホットパス確保ゼロラチェット・Google Benchmark・QA ゲート）。
