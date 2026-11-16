# 仕様追跡表

このファイルは `tools/qa/trace.py matrix` によって自動生成される。手編集しない。

| ID | 項目 | 正本位置 | テスト | 検証記録 |
|---|---|---|---|---|
| `REQ-01` | Exact GPR スコープ | [design.md#L11](../design.md#L11) | — | — |
| `REQ-02` | C++23 非例外・値返却 API | [design.md#L12](../design.md#L12) | `tests/core/error_test.cpp`<br>`tests/core/interval_test.cpp`<br>`tests/core/likelihood_test.cpp`<br>`tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp`<br>`tests/transform/target_test.cpp` | — |
| `REQ-03` | 列優先入力と入力検査順序 | [design.md#L13](../design.md#L13) | `tests/core/checked_test.cpp`<br>`tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp` | — |
| `REQ-04` | カーネル体系と合成 | [design.md#L14](../design.md#L14) | — | — |
| `REQ-05` | 供給二乗距離 | [design.md#L15](../design.md#L15) | — | — |
| `REQ-06` | ノイズと Jitter の分離 | [design.md#L16](../design.md#L16) | `tests/core/likelihood_test.cpp`<br>`tests/linalg/cholesky_test.cpp` | — |
| `REQ-07` | 3 種精度ポリシーと IR | [design.md#L17](../design.md#L17) | `tests/precision/policy_test.cpp`<br>`tests/precision/refine_test.cpp` | — |
| `REQ-08` | MLL・微分・LOO | [design.md#L18](../design.md#L18) | — | — |
| `REQ-09` | 自前最適化器群 | [design.md#L19](../design.md#L19) | — | — |
| `REQ-10` | 前処理パイプライン | [design.md#L20](../design.md#L20) | `tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp`<br>`tests/transform/target_test.cpp` | — |
| `REQ-11` | ゼロ確保・決定的並列・境界 | [design.md#L21](../design.md#L21) | `tests/core/build_info_test.cpp`<br>`tests/core/error_test.cpp`<br>`tests/core/likelihood_test.cpp`<br>`tests/core/rng_test.cpp`<br>`tests/core/views_test.cpp`<br>`tests/linalg/cholesky_test.cpp`<br>`tests/linalg/parallel_test.cpp`<br>`tests/precision/policy_test.cpp`<br>`tests/precision/refine_test.cpp`<br>`tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp`<br>`tests/transform/target_test.cpp` | — |
| `REQ-12` | 永続化・乱数・独立検証 | [design.md#L22](../design.md#L22) | `tests/core/rng_test.cpp` | — |
| `DES-01` | 型設計・生成契約・公開境界 | [design.md#L28](../design.md#L28) | `tests/core/error_test.cpp`<br>`tests/core/interval_test.cpp`<br>`tests/core/likelihood_test.cpp`<br>`tests/core/views_test.cpp`<br>`tests/linalg/jitter_test.cpp`<br>`tests/precision/policy_test.cpp`<br>`tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp`<br>`tests/transform/target_test.cpp` | — |
| `DES-02` | エラー型 `GprError` と入力検査契約 | [design.md#L42](../design.md#L42) | `tests/core/checked_test.cpp`<br>`tests/core/error_test.cpp`<br>`tests/core/interval_test.cpp`<br>`tests/linalg/cholesky_test.cpp`<br>`tests/linalg/dense_test.cpp`<br>`tests/linalg/jitter_test.cpp`<br>`tests/linalg/parallel_test.cpp`<br>`tests/precision/policy_test.cpp`<br>`tests/precision/refine_test.cpp`<br>`tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp`<br>`tests/transform/target_test.cpp` | — |
| `DES-03` | 線形代数バックエンド（Eigen + OpenMP） | [design.md#L77](../design.md#L77) | `tests/core/build_info_test.cpp`<br>`tests/linalg/cholesky_test.cpp`<br>`tests/linalg/dense_test.cpp` | — |
| `DES-04` | 観測ノイズ `GaussianLikelihood` と `JitterPolicy` | [design.md#L86](../design.md#L86) | `tests/core/interval_test.cpp`<br>`tests/core/likelihood_test.cpp`<br>`tests/linalg/cholesky_test.cpp`<br>`tests/linalg/jitter_test.cpp` | — |
| `DES-05` | 精度ポリシーと混合精度反復改良（IR） | [design.md#L98](../design.md#L98) | `tests/precision/policy_test.cpp`<br>`tests/precision/refine_test.cpp` | — |
| `DES-06` | `KernelSpec`（宣言層）と `CompiledKernel<T>`（実行層） | [design.md#L126](../design.md#L126) | — | — |
| `DES-07` | 全 10 種組込みカーネル葉の数理定義と区間 | [design.md#L135](../design.md#L135) | — | — |
| `DES-08` | ユーザー定義カーネル拡張 `KernelTerm<T>` | [design.md#L156](../design.md#L156) | `tests/core/views_test.cpp` | — |
| `DES-09` | 距離キャッシュ方針（`DistanceCachePolicy`） | [design.md#L182](../design.md#L182) | — | — |
| `DES-10` | 供給二乗距離（`ScalarDistance` / `ArdDistance` / `DistanceKernel<C>`） | [design.md#L189](../design.md#L189) | — | — |
| `DES-11` | 増分更新対応（`LeafCache` と `USES_CHANGE_INDICES`） | [design.md#L202](../design.md#L202) | — | — |
| `DES-12` | 指数評価バックエンド `KernelExp` | [design.md#L209](../design.md#L209) | — | — |
| `DES-13` | MLL と解析勾配の単一パス算法（`keep_plan`） | [design.md#L220](../design.md#L220) | — | — |
| `DES-14` | 解析ヘッセ行列（`HessianScratch`） | [design.md#L241](../design.md#L241) | — | — |
| `DES-15` | `CholeskyBuffer::{Retain, Reuse}` とメモリ・速度優先 | [design.md#L252](../design.md#L252) | — | — |
| `DES-16` | 推論 API・共分散・標本抽出・LOO | [design.md#L257](../design.md#L257) | `tests/core/views_test.cpp` | — |
| `DES-17` | ワークスペース構造とライフサイクル | [design.md#L273](../design.md#L273) | — | — |
| `DES-18` | 前処理クラス階層（`cxxgpr::transform`） | [design.md#L291](../design.md#L291) | `tests/transform/columnwise_test.cpp`<br>`tests/transform/input_test.cpp`<br>`tests/transform/pipeline_test.cpp`<br>`tests/transform/target_test.cpp` | — |
| `DES-19` | 自前最適化器群（`cxxgpr::optimizer`） | [design.md#L298](../design.md#L298) | — | — |
| `DES-20` | 決定的乱数（`Xoshiro256++`）とスレッド不変 OpenMP 並列 | [design.md#L311](../design.md#L311) | `tests/core/rng_test.cpp`<br>`tests/linalg/parallel_test.cpp` | — |
| `FMT-01` | ディレクトリ構成とバージョン契約 | [persist-format.md#L7](../persist-format.md#L7) | — | — |
| `FMT-02` | 原子的書き込みと入力信頼境界 | [persist-format.md#L18](../persist-format.md#L18) | — | — |
| `FMT-03` | `config.json` スキーマ | [persist-format.md#L28](../persist-format.md#L28) | `tests/precision/policy_test.cpp` | — |
| `FMT-04` | `model.safetensors` レイアウトと列優先契約 | [persist-format.md#L57](../persist-format.md#L57) | — | — |
| `FMT-05` | `save` / `save_with_factor` と `f64` 因子 `mmap` 読込 | [persist-format.md#L73](../persist-format.md#L73) | — | — |
| `FMT-06` | カスタム拡張登録と `PersistRegistry` | [persist-format.md#L85](../persist-format.md#L85) | — | — |
| `FMT-07` | 供給二乗距離モデルの永続化（`LoadedDistanceGpr<C>`） | [persist-format.md#L95](../persist-format.md#L95) | — | — |
| `FMT-08` | エラー種別・外部ツール互換マッピング | [persist-format.md#L121](../persist-format.md#L121) | — | — |
| `QA-01` | 追跡 | [quality-plan.md#L11](quality-plan.md#L11) | — | — |
| `QA-02` | 検証記録 | [quality-plan.md#L20](quality-plan.md#L20) | — | — |
| `QA-03` | 基準結果（hexfloat ベースライン） | [quality-plan.md#L24](quality-plan.md#L24) | — | — |
| `QA-04` | 結果の変更管理 | [quality-plan.md#L32](quality-plan.md#L32) | — | — |
| `QA-05` | 版の意味 | [quality-plan.md#L39](quality-plan.md#L39) | — | — |
| `QA-06` | 来歴（Provenance） | [quality-plan.md#L51](quality-plan.md#L51) | `tests/core/build_info_test.cpp` | — |
| `QA-07` | 環境の認定 | [quality-plan.md#L55](quality-plan.md#L55) | — | — |
| `QA-08` | 不具合と既知の問題 | [quality-plan.md#L59](quality-plan.md#L59) | — | — |
| `QA-09` | 公開前の品質ゲート | [quality-plan.md#L63](quality-plan.md#L63) | — | — |
| `QA-10` | 役割と承認 | [quality-plan.md#L67](quality-plan.md#L67) | — | — |
