# 0001: Exact GPR スコープと C++23 設計・依存方針

- Status: Accepted
- Date: 2026-11-15
- Issue: #1

## Context

Rust ライブラリ `gprx` には Exact GPR（`Gpr` / `FittedGpr`）、オンライン更新版 Exact GPR（`OnlineGpr`）、および近似疎モデル（`Sgpr` / `Svgp` / `OnlineSgpr`）が含まれる。本プロジェクト `cxxgpr` の目的は、実務・組込み・C++ エコシステムから直接利用できる **Exact GPR** の全機能を、C++23 で忠実かつ安全に再構築することにある。

あわせて、依存管理方式（Eigen / JSON / 最適化器 / テスト / ベンチマーク）、API エラー契約、公開ヘッダ境界を初期段階で固定し、実装着手後の大規模な作り直しを防ぐ必要がある。

## Decision

1. **移植対象スコープの明確化**：
   - `gprx` の Exact GPR（`Gpr` / `FittedGpr`、全 10 種組込みカーネル葉・合成・`Custom`、供給二乗距離 `DistanceKernel`、前処理変換、`Double`/`Single`/`MixedPrecision`、`KernelExp`、5 種最適化器、`persist`）を完全移植する。
   - `Sgpr` / `Svgp` / `OnlineGpr`（`into_online` / `LdltStore` / `PointId` / `ldlt` persist）および疎誘導点座標微分は移植対象から除外する。
2. **C++23 公開 API 契約**：
   - 名前空間は `cxxgpr`（`kernel`, `transform`, `persist`）。入力行列は列優先 `std::span<const double>`。
   - 例外を API 境界へ送出せず、`[[nodiscard]] std::expected<T, GprError>` でエラーを返す。
   - `Gpr::fit(...)` と `Gpr::factor(...)` は `const` メンバ関数とし、失敗時も `*this` を変更しない。
   - 失敗しない型はコンストラクタ、失敗し得る型は `static std::expected<T, GprError> create(...)` を使う。
3. **依存選定と公開ヘッダ非漏洩**：
   - 依存は vcpkg manifest（`vcpkg.json` の `builtin-baseline` + `overrides`）で固定する。
   - 内部線形代数は `eigen3`（5.0.1、`EIGEN_MPL2_ONLY` + `EIGEN_DONT_PARALLELIZE`）、永続化 JSON は `nlohmann-json`（3.12.0）、並列化は OpenMP を使う。いずれも静的ライブラリ `cxxgpr` の `PRIVATE` 依存に閉じ込め、`include/cxxgpr/` に露出させない。
   - 最適化器（`Lbfgs`・`NelderMead`・`TrustRegion`・`FastSimulatedAnnealing`・`Fixed`）は外部ソルバライブラリを導入せず、`gprx` / argmin アルゴリズム仕様に忠実な自前実装とする。
   - 試験は `gtest`（1.18.0、`tests` feature）、計測は `benchmark`（1.9.5、`benchmarks` feature）を分離導入する。

## Consequences

- `include/cxxgpr/` の利用者は Eigen や `nlohmann/json` や OpenMP のヘッダ導入なしに静的ライブラリ `cxxgpr` を利用できる（ABI・ビルド依存・ライセンス露出の最小化）。
- 自前最適化器の実装コストが増える反面、L-BFGS Moré–Thuente や TrustRegion Steihaug–Toint CG、FSA 増分評価の数値挙動と決定性を完全に制御できる。
- Apache-2.0（`cxxgpr` 本体）および MPL-2.0（Eigen、`EIGEN_MPL2_ONLY`）/ MIT（`nlohmann-json`）との互換性が保たれる。
