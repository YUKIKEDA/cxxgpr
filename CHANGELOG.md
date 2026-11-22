# 変更履歴

公開した版を、新しい順に [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) の形式で記す。版の意味は [品質保証計画 QA-05](docs/internal/quality/quality-plan.md) が定める。PATCH は基準結果を変えない変更、MINOR は数値予算内の結果変更を含む変更、MAJOR は形式・API・予算を超える変更とする。

## [Unreleased]

## [0.1.0] - 2026-11-22

### 追加

- Exact GPR モデル（`Gpr` / `FittedGpr`）：学習（`fit` / `factor`）、予測平均・分散（`predict` / `predict_into`）、同時予測共分散（`predict_covariance`）、事後標本抽出（`sample`）、1 つ抜き交差検証（`loo_predict` / `loo_predict_into`）、解析勾配・解析ヘッセ行列つきの負対数周辺尤度。
- 供給二乗距離で学習・推論するモデル（`DistanceGpr` / `FittedDistanceGpr`）：距離のみ（`DistanceOnly`）および座標併用（`WithPoints`）の双方で、学習・予測・共分散・標本抽出・LOO・再学習（`refit` / `into_trainer`）を提供。
- 全 10 種の組込みカーネル葉（RBF、Matérn 1/2・3/2・5/2、周期、有理二次、定数、線形、白色、および各 ARD 版）、和・積による合成、利用者定義の `CustomKernel`（`KernelTerm<T>`）、供給二乗距離スロット（`ScalarDistance` / `ArdDistance` / `DistanceKernel`）、`LeafCache` による増分更新と OpenMP 並列評価。
- 5 種の自己完結型最適化器（`Lbfgs` / `NelderMead` / `TrustRegion` / `FastSimulatedAnnealing` / `Fixed`）と決定的なマルチスタート再開始（`with_restarts`）。
- 入力・目的変数の前処理変換（`Identity` / `Standardize` / `MinMax` / `ColumnwiseInput` / `Pipeline` / `TargetPipeline` およびカスタム変換インターフェース）。
- 精度ポリシー（`DoublePrecision` / `SinglePrecision` / `MixedPrecision` 反復改良）、適応的 Cholesky Jitter ポリシー、指数関数評価モード（`KernelExp::{Accurate, FastApprox}`）、距離キャッシュポリシー（`DistanceCachePolicy::{Cached, Uncached}`）。
- `config.json` と `model.safetensors` による永続化（`persist::save` / `save_with_factor`、`LoadedGpr` / `LoadedDistanceGpr<C>`、倍精度因子のゼロコピー `mmap` 読込、`PersistRegistry` によるカスタム拡張復元、保存ごとの SHA-256 ペアリング識別子 `save_id` によるファイル対整合検査）。
- `MatrixRef<T>::from_span(span, rows, cols)` と `MatrixMut<T>::from_span(span, rows, cols)`：スパンの長さを寸法と照合してからビューを返す検査付きファクトリ（`EmptyInput` → `SizeOverflow` → `LengthMismatch`）。
- CMake `install` / `find_package(cxxgpr CONFIG)` / `FetchContent` パッケージ構成と、GitHub Release ワークフロー（`v*` タグで Windows・Linux のパッケージ、文書サイト、`SHA256SUMS.txt` を揃えた下書き Release を作成）。
- 利用者向け文書（英語・日本語）：Diataxis の 4 区分（チュートリアル・ハウツー・リファレンス・解説）、全公開 API シンボルの網羅検査（`just docs-coverage`）、コンパイル・実行される例（`docs/examples/`）。
- 総合検証スイート（`path_matrix` 全組合せ試験、参照実装・解析解との golden 数値照合、ホットパス確保ゼロ監査と深さ 128 小スタック試験、Google Benchmark、QA ゲート）。
