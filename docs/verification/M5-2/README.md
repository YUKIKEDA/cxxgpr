# M5-2 独立参照・解析解 Golden 数値検証記録

## 目的

`QA-02` および `REQ-12` に基づき、`cxxgpr` の Exact GPR 実装（fixed $\theta$、L-BFGS 最適化、LOO 予測、混合精度推論）を、独立した純粋 Python 参照ソルバ（[tools/qa/m5_2_golden.py](../../../tools/qa/m5_2_golden.py)）および閉形式解析解の golden 値と照合した結果を記録する。

## 検証ケースと許容差

| ケース | 検証対象 | 独立参照の算出方法 | 照合許容差 |
|---|---|---|---|
| `AnalyticalClosedFormOneAndTwoPoints` | 1 点・2 点 RBF + noise（$\text{NLL}$, $\alpha$, $\mu_*$, $\sigma_*^2$, $\text{LOO}$） | 閉形式解析解 | $10^{-14}$（1 点）/ $10^{-13}$（2 点） |
| `Forrester1dFixedLbfgsAndLooMatchIndependentReference` (Fixed) | 1D スケール済み Forrester ($n=18$, $c=1.0, \ell=0.25, \sigma_n^2=10^{-3}$) | 純粋 Python Cholesky + 三角求解 + LOO | $\text{NLL}$/予測/LOO: $10^{-11}$ |
| `Forrester1dFixedLbfgsAndLooMatchIndependentReference` (L-BFGS) | 1D スケール済み Forrester 対数空間 L-BFGS 最適化（$\ell \in (0.05, 5.0)$） | 純粋 Python 2-loop L-BFGS + 中心差分 | $\text{NLL}$: $10^{-4}$, $\mu_*$: $10^{-4}$, $\sigma_*^2$: $10^{-5}$ |
| `TwoDimArdMatern52FixedLbfgsAndLooMatchIndependentReference` (Fixed) | 2D ARD Matérn 5/2 ($n=10$, $c=1.5, \ell=(0.5, 0.7), \sigma_n^2=10^{-3}$) | 純粋 Python Cholesky + 三角求解 + LOO | $\text{NLL}$/予測/LOO: $10^{-12}$ |
| `TwoDimArdMatern52FixedLbfgsAndLooMatchIndependentReference` (L-BFGS) | 2D ARD Matérn 5/2 対数空間 L-BFGS 最適化 | 純粋 Python 2-loop L-BFGS + 中心差分 | $\text{NLL}$: $10^{-6}$, $\mu_*$: $10^{-4}$, $\sigma_*^2$: $10^{-5}$ |

## 再現手順

```powershell
python -X utf8 tools/qa/m5_2_golden.py
ctest --preset windows-debug -R verification\.golden --output-on-failure
```

## 結果

[golden_test.cpp](../../../tests/verification/golden_test.cpp)（CTest: `verification.golden`）にて全ケースが上記許容差内で通過することを確認した。
