# 変更履歴

公開した版を、新しい順に [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) の形式で記す。版の意味は [品質保証計画 QA-05](docs/internal/quality/quality-plan.md) が定める。PATCH は基準結果を変えない変更、MINOR は数値予算内の結果変更を含む変更、MAJOR は形式・API・予算を超える変更とする。

## [Unreleased]

## [0.3.0] - 2026-11-26

### 追加

- `DistanceGpr::with_likelihood(GaussianLikelihood)`（カスタム観測ノイズ区間・初期値の指定）および `with_retain_distances(bool)` / `retains_distances()`（学習済み距離テーブルの保持制御。非保持モデルの永続化は `PersistErrorKind::NotPersistable` を返却）を追加。
- `DistanceSource::assume_valid(bool)` / `assumes_valid()`（構造検査を維持しつつ値走査のみを省略する明示的オプトイン、`tidy()` 併用時は `InvalidConfig`）を追加。
- `DistanceGpr` / `FittedDistanceGpr` で事前検証済み `CheckedDistanceStore<Storage>` を学習・予測・共分散・標本抽出・再学習へ直接渡せるオーバーロードを追加し、`CheckedDistanceStore<T>` を共有不変保持による $O(1)$ コピー可能型として提供。
- `Gpr` / `FittedGpr` / `LoadedGpr` / `DistanceGpr` / `FittedDistanceGpr` / `LoadedDistanceGpr` / `DistanceSource` 全般に `float` 精度の入力行列（`MatrixRef<float>`・検査付き `std::span<const float>`）・ターゲット（`std::span<const float>`）・供給距離テーブル入力を追加（`SinglePrecision` 距離テーブルは変換なしで直接消費、`predict_into` はウォームアップ後の確保ゼロを維持）。
- `LoadedGpr::predict_covariance` / `sample` に検査付き `std::span<const double>` オーバーロードを追加。

### 変更

- 供給二乗距離経路を最適化（定数乗算カーネル畳み込み、並列・SIMD 値検証、BLAS ベースの事後共分散評価、遅延因子バッファ確保）。
- `with_retain_distances(false)` かつ `tidy` 未指定で入力型が `Storage` と一致する場合、スカラー・ARD 双方の正方および交差距離テーブルをゼロコピー借用して評価する経路を導入。組込み葉のみに結合された所有正方スカラーテーブルを下三角パック化（メモリ半減）し、`CustomKernel` 結合スロットのみ密配置を維持。

### 修正

- `DistanceSource::tidy(rel_tol)` で `*tidy_tol * max_abs` の `+Inf` 溢れを `InvalidDistance` として検出し、対称化平均での `(b - a)` 溢れを回避。
- 供給距離の構造検査で非零寸法かつ `data() == nullptr` の `MatrixRef` を `EmptyInput` として拒否し、`WithPoints` 座標コピーに `checked_mul` による `SizeOverflow` 検査を追加。

## [0.2.0] - 2026-11-26

### 追加

- 選択可能な密線形代数バックエンド `CXXGPR_BLAS_BACKEND`（`AUTO` / `OpenBLAS` / `MKL` / `BuiltIn`）を追加。
- `LeafCache` の固定 64 次元上限を撤廃し、記憶量 $O(n^2)$ + $O(d)$ のまま任意次元 $d$ で ARD 差分更新と `DistanceCachePolicy::Uncached` 列座標直接更新に対応。
- `FastSimulatedAnnealing` の棄却ステップで採用済み $s^2$ ベースラインを汚染しない非破壊 trial 隔離（`LeafCache::prime` / `commit_candidate`、`TwiceDifferentiable::notify_candidate_accepted`）と、差分更新・放射カーネル評価の 1 パス融合を追加。
- `DistanceGpr` / `CompiledDistanceKernel`（`ArdDistance`）の `value_at_changes` へ ARD 差分更新を追加し、`mll_and_gradient_into` で `factor_and_mll` が計算した ARD $s^2$ を `leaf_ard_weighted_grads` へ再利用する経路（`WorkspaceCore::ensure_ard_s2`）を追加。

### 変更

- `KernelExp::FastApprox` を依存ゼロの 1–2 ULP Estrin ミニマックス インライン SIMD 実装（倍精度 11 次・単精度 6 次固定端点チェビシェフ＋正確 Cody-Waite、Matérn 1/2・3/2・5/2 との 1 パスレジスタ融合、倍精度誤差 $< 10^{-14}$・単精度誤差 $< 5\times 10^{-7}$）へ刷新。
- Windows MSVC 既定の密線形代数バックエンドを依存ゼロの `BuiltIn`（AVX-512 / AVX2 / SSE2 + OpenMP）、Linux / MinGW 既定を OpenBLAS（`0.3.34`）とし、`DES-20` のスレッド数間ビット一致契約へ外部マルチスレッド BLAS 利用時のスレッド固定条件を明記。

### 修正

- `MKL` 構成における小行列並列競合を抑制（`MklLocalThreadsGuard`、多スレッド時の小行列 `double` 経路への `BuiltIn` フォールバック、および `LAPACKE_?potrf` / `LAPACKE_?potri` の負のエラーコード・`lapack_int` 溢れ検査）。
- インストール済み `cxxgprConfig.cmake` へのビルド環境ローカル MKL パス非埋め込み化と `VerifyPackage.cmake` での漏洩検査、`BuiltIn` 構成の事後共分散 1 パス融合による性能回復。

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
