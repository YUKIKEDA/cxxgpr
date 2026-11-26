# 第三者ライセンス通知

cxxgpr 自身のソースのライセンスは、ルートの `LICENSE` にある Apache-2.0 である。第三者のソース・ヘッダには、それぞれのライセンスが適用される。

| 依存 | 固定版 | ライセンス | 用途 |
|---|---|---|---|
| Eigen | 5.0.1 | MPL-2.0、および同梱の互換ライセンス | 内部パネル線形代数・前処理・次元検査。`EIGEN_MPL2_ONLY` を定義し、公開ヘッダに露出しない |
| OpenBLAS | 0.3.34 | BSD-3-Clause | 非 Windows MSVC 環境の既定外部 BLAS（任意切替で選択可、`GEMM` / `SYRK` / `TRSM` / `GEMV` / `DOT`）。Windows MSVC の既定は依存ゼロの自前 `BuiltIn` SIMD バックエンド。公開ヘッダに露出しない |
| Intel oneMKL（任意切替） | 導入環境準拠 | Intel Simplified Software License | `-DCXXGPR_BLAS_BACKEND=MKL` 指定時の BLAS / LAPACKE 実装。公開ヘッダに露出しない |
| nlohmann/json | 3.12.0 | MIT | 永続化 `config.json` と safetensors ヘッダの解析。公開ヘッダに露出しない |
| GoogleTest | 1.18.0 | BSD-3-Clause | `tests` feature、試験専用。製品ライブラリへリンクしない |
| Google Benchmark | 1.9.5 | Apache-2.0 | `benchmarks` feature、計測専用。製品ライブラリへリンクしない |
| OpenMP | ツールチェーン同梱 | ツールチェーン準拠 | 距離行列・カーネル行列構築の並列化。公開ヘッダに露出しない |

上の表はライセンス全文や著作権表示の代わりではない。CMake の構成時に、vcpkg が導入した各パッケージの `share/<package>/copyright` と任意選択された Intel oneMKL のライセンス文書を構成先の `third-party-notices/<package>.txt` へバイトを変えずに複製し、CTest の整合検査（`toolchain.third_party_notices`）で原本と複製の SHA-256 一致を検証する。

依存の取得・版の正本は `vcpkg.json` および [ADR 0001](docs/internal/adr/0001-exact-gpr-cpp23-scope.md) / [ADR 0002](docs/internal/adr/0002-msvc-ninja-vcpkg-toolchain.md) / [ADR 0004](docs/internal/adr/0004-external-blas-and-fmath.md) である。`EIGEN_MPL2_ONLY` により GPL/LGPL ライセンスのモジュール混入を禁止し、`include/cxxgpr/` に第三者ヘッダを露出させない。
