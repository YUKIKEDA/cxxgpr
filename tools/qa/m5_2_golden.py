# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "scikit-learn>=1.5",
# ]
# ///
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Generate M5-2 golden reference values using scikit-learn GaussianProcessRegressor."""

from __future__ import annotations

import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as W


def main() -> None:
    """Compute and print M5-2 golden values from scikit-learn."""
    # Case 1: 1D scaled noisy Forrester (n=18)
    x_forr = np.array([[i / 17.0] for i in range(18)], dtype=np.float64)
    y_forr = np.array(
        [
            0.1 * ((6.0 * p[0] - 2.0) ** 2) * math.sin(12.0 * p[0] - 4.0)
            + 0.02 * math.cos(50.0 * p[0])
            for p in x_forr
        ],
        dtype=np.float64,
    )
    xq_forr = np.array([[0.15], [0.50], [0.85]], dtype=np.float64)

    k_forr_fixed = C(1.0, "fixed") * RBF(0.25, "fixed")
    gp_forr_fixed = GaussianProcessRegressor(kernel=k_forr_fixed, alpha=1e-3, optimizer=None).fit(
        x_forr, y_forr
    )
    mean_ff, std_ff = gp_forr_fixed.predict(xq_forr, return_std=True)
    loo_mean_f: list[float] = []
    loo_var_f: list[float] = []
    for i in range(4):
        gp_i = GaussianProcessRegressor(kernel=k_forr_fixed, alpha=1e-3, optimizer=None).fit(
            np.delete(x_forr, i, axis=0), np.delete(y_forr, i)
        )
        m_i, s_i = gp_i.predict(x_forr[i : i + 1], return_std=True)
        loo_mean_f.append(float(m_i[0]))
        loo_var_f.append(float(s_i[0] ** 2 + 1e-3))

    print("=== FORRESTER FIXED (scikit-learn) ===")
    print(f"nll={-float(gp_forr_fixed.log_marginal_likelihood_value_):.17g}")
    print("mean=", [f"{float(v):.17g}" for v in mean_ff])
    print("obs_var=", [f"{float(s**2 + 1e-3):.17g}" for s in std_ff])
    print("loo_mean=", [f"{v:.17g}" for v in loo_mean_f])
    print("loo_var=", [f"{v:.17g}" for v in loo_var_f])

    k_forr_opt = C(1.0, (1e-5, 1e5)) * RBF(0.25, (0.05, 5.0)) + W(1e-3, (1e-5, 1e5))
    gp_forr_opt = GaussianProcessRegressor(
        kernel=k_forr_opt, alpha=0.0, n_restarts_optimizer=0
    ).fit(x_forr, y_forr)
    mean_fo, std_fo = gp_forr_opt.predict(xq_forr, return_std=True)
    print("=== FORRESTER OPT (scikit-learn) ===")
    print(f"nll={-float(gp_forr_opt.log_marginal_likelihood_value_):.17g}")
    print("kernel=", gp_forr_opt.kernel_)
    print("mean=", [f"{float(v):.17g}" for v in mean_fo])
    print("obs_var=", [f"{float(s**2):.17g}" for s in std_fo])

    # Case 2: 2D ARD Matern 5/2 (n=10)
    x_2d = np.array([[0.1 * i, 0.1 * ((3 * i + 1) % 10)] for i in range(10)], dtype=np.float64)
    y_2d = np.array(
        [math.sin(3.0 * p[0]) + 0.5 * math.cos(4.0 * p[1]) for p in x_2d], dtype=np.float64
    )
    xq_2d = np.array([[0.25, 0.45], [0.75, 0.15]], dtype=np.float64)

    k_2d_fixed = C(1.5, "fixed") * Matern([0.5, 0.7], length_scale_bounds="fixed", nu=2.5)
    gp_2d_fixed = GaussianProcessRegressor(kernel=k_2d_fixed, alpha=1e-3, optimizer=None).fit(
        x_2d, y_2d
    )
    mean_2f, std_2f = gp_2d_fixed.predict(xq_2d, return_std=True)
    loo_mean_2: list[float] = []
    loo_var_2: list[float] = []
    for i in range(4):
        gp_i = GaussianProcessRegressor(kernel=k_2d_fixed, alpha=1e-3, optimizer=None).fit(
            np.delete(x_2d, i, axis=0), np.delete(y_2d, i)
        )
        m_i, s_i = gp_i.predict(x_2d[i : i + 1], return_std=True)
        loo_mean_2.append(float(m_i[0]))
        loo_var_2.append(float(s_i[0] ** 2 + 1e-3))

    print("=== 2D ARD FIXED (scikit-learn) ===")
    print(f"nll={-float(gp_2d_fixed.log_marginal_likelihood_value_):.17g}")
    print("mean=", [f"{float(v):.17g}" for v in mean_2f])
    print("obs_var=", [f"{float(s**2 + 1e-3):.17g}" for s in std_2f])
    print("loo_mean=", [f"{v:.17g}" for v in loo_mean_2])
    print("loo_var=", [f"{v:.17g}" for v in loo_var_2])

    k_2d_opt = C(1.0, (1e-5, 1e5)) * Matern(
        [0.5, 0.5], length_scale_bounds=(1e-5, 1e5), nu=2.5
    ) + W(1e-2, (1e-5, 1e5))
    gp_2d_opt = GaussianProcessRegressor(kernel=k_2d_opt, alpha=0.0, n_restarts_optimizer=0).fit(
        x_2d, y_2d
    )
    mean_2o, std_2o = gp_2d_opt.predict(xq_2d, return_std=True)
    print("=== 2D ARD OPT (scikit-learn) ===")
    print(f"nll={-float(gp_2d_opt.log_marginal_likelihood_value_):.17g}")
    print("kernel=", gp_2d_opt.kernel_)
    print("mean=", [f"{float(v):.17g}" for v in mean_2o])
    print("obs_var=", [f"{float(s**2):.17g}" for s in std_2o])


if __name__ == "__main__":
    main()
