## Summary

<!-- 何を、なぜ変えたか。実装した決定ID（REQ-01、DES-01、FMT-01、QA-01 など）。 -->

## Related

Closes #N

## Verification

<!-- 実際に走らせたコマンドと結果。走らせていないものは書かない。失敗はそのまま書く。 -->

- [ ] 差分と依存先から、検証する対象・構成と省略理由を記載した
- [ ] 対象とした試験・静的検査が通った（必要なOSだけを記載）
- [ ] 全体検証が必要な場合は、編集後の最後に1回へまとめた

## Claimed state

<!-- このPRが主張するものだけに印を付ける。主張しないものは理由を一言。 -->

- [ ] 実装が採用仕様に適合
- [ ] 数値認定
- [ ] 製品規模の性能
- [ ] 実務受入

## Risk / Rollback

N/A

## Checklist

- [ ] 題名が Conventional Commits の形式（英語）
- [ ] 各決定に、独立な参照による試験と負対照がある。許容差は数値で書き、出典を添えた
- [ ] 試験が検出する変異を入れて、落ちることを確かめた（何を入れたかを Verification に書いた）
- [ ] 失敗の後、`const` メソッドや親オブジェクトが不変であることを試験した
- [ ] 繰り返し呼ばれる経路（MLL+grad、`predict_into`）で確保していない。OpenMP 並列は逐次・スレッド数変更間で決定的に一致する
- [ ] 公開ヘッダ（`include/cxxgpr/`）に Eigen・nlohmann/json・OpenMP が露出していない
- [ ] 数値試験は確かめる決定IDを記し、`just trace-check` / `just arch-check` / `just doc-check` が通る
- [ ] 基準結果は変わらない。変わるなら、結果変更記録を足し、所有者の承認を得た（QA-04）
- [ ] 採用仕様からの逸脱、見つけた仕様欠陥を書いた（なければ「なし」）
- [ ] 先送りしたものは `docs/internal/roadmap.md` の行になっている
- [ ] `docs/internal/conventions.md`、`docs/internal/roadmap.md` を必要に応じて更新した
