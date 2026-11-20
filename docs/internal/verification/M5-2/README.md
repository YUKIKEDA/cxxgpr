# M5-2 Independent-reference and analytic-solution golden verification record

## Purpose

Following `QA-02` and `REQ-12`, this record documents the numerical comparison of the `cxxgpr` Exact GPR implementation (fixed $\theta$, L-BFGS optimization, LOO prediction, mixed-precision inference) against the output of the external library `scikit-learn` 1.9.1 (`GaussianProcessRegressor`) and against closed-form analytic solutions.

## Cases and tolerances

| Case | Subject | How the independent reference is computed | Comparison tolerance |
|---|---|---|---|
| `AnalyticalClosedFormOneAndTwoPoints` | One- and two-point RBF + noise ($\text{NLL}$, $\alpha$, $\mu_*$, $\sigma_*^2$, $\text{LOO}$) | Closed-form analytic solution | $10^{-14}$ (one point) / $10^{-13}$ (two points) |
| `Forrester1dFixedLbfgsAndLooMatchIndependentReference` (Fixed) | 1D scaled Forrester ($n=18$, $c=1.0, \ell=0.25, \sigma_n^2=10^{-3}$) | `scikit-learn` `GaussianProcessRegressor` + actual leave-one-out refits | $\text{NLL}$/prediction/LOO: $10^{-11}$ |
| `Forrester1dFixedLbfgsAndLooMatchIndependentReference` (L-BFGS) | 1D scaled Forrester, log-space L-BFGS optimization ($\ell \in (0.05, 5.0)$) | `scikit-learn` `ConstantKernel * RBF + WhiteKernel` (`L-BFGS-B`) | $\text{NLL}$: $10^{-6}$, $\mu_*$: $10^{-5}$, $\sigma_*^2$: $10^{-5}$ |
| `TwoDimArdMatern52FixedLbfgsAndLooMatchIndependentReference` (Fixed) | 2D ARD Matérn 5/2 ($n=10$, $c=1.5, \ell=(0.5, 0.7), \sigma_n^2=10^{-3}$) | `scikit-learn` `GaussianProcessRegressor` + actual leave-one-out refits | $\text{NLL}$/prediction/LOO: $10^{-12}$ |
| `TwoDimArdMatern52FixedLbfgsAndLooMatchIndependentReference` (L-BFGS) | 2D ARD Matérn 5/2, log-space L-BFGS optimization | `scikit-learn` `ConstantKernel * Matern(nu=2.5) + WhiteKernel` (`L-BFGS-B`) | $\text{NLL}$: $10^{-6}$, $\mu_*$: $10^{-4}$, $\sigma_*^2$: $10^{-5}$ |

## Reproduction

```powershell
uv run tools/qa/m5_2_golden.py
ctest --preset windows-debug -R verification\.golden --output-on-failure
```

## Result

All cases were confirmed to pass within the tolerances above in [golden_test.cpp](../../../../tests/verification/golden_test.cpp) (CTest: `verification.golden`).
