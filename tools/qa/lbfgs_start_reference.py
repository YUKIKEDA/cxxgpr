# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "scipy>=1.13",
#   "scikit-learn>=1.5",
# ]
# ///
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
"""Independent reference optimum for the L-BFGS starting-point test, from scikit-learn.

Reproduces the data of ``GprModelsTest.LbfgsConvergesToTheSameOptimumFromEveryStartingLengthScale``
(``tests/gpr/gpr_test.cpp``): ``x_i = i / 29`` and
``y_i = sin(6 x_i) + 0.5 x_i + 0.05 ((37 i mod 29) / 14 - 1)`` for ``i = 0..29``, the kernel
``ConstantKernel * RBF + WhiteKernel`` with every parameter in ``(1e-5, 1e5)``, and a standardized
target (population standard deviation, as ``normalize_y=True``). The negative log marginal
likelihood is minimized in log-parameter space by scikit-learn's ``L-BFGS-B`` with 50 restarts and
then polished with SciPy ``L-BFGS-B`` at ``gtol = 1e-12`` on the scikit-learn objective.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as W


def main() -> None:
    """Print the reference optimum, the gradient there, and the Hessian eigenvalues."""
    n = 30
    x = np.array([[i / (n - 1)] for i in range(n)], dtype=np.float64)
    y = np.array(
        [
            math.sin(6.0 * float(p[0])) + 0.5 * float(p[0]) + 0.05 * (((37 * i) % 29) / 14.0 - 1.0)
            for i, p in enumerate(x)
        ],
        dtype=np.float64,
    )
    kernel = C(1.0, (1e-5, 1e5)) * RBF(0.3, (1e-5, 1e5)) + W(1e-2, (1e-5, 1e5))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=50, random_state=0
    ).fit(x, y)
    kernel_opt = gp.kernel_

    def objective(theta):
        lml, grad = gp.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=True)
        return -lml, -grad

    bounds = kernel_opt.bounds
    polished = minimize(
        objective,
        kernel_opt.theta,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"gtol": 1e-12, "ftol": 1e-16, "maxiter": 1000},
    )
    nll, grad = objective(polished.x)
    print("scikit-learn kernel:", kernel_opt)
    print(f"nll={nll:.17g}")
    print("theta=", [f"{float(t):.17g}" for t in polished.x])
    print("grad=", [f"{float(g):.3e}" for g in grad])
    # Central-difference Hessian of the scikit-learn objective, for the parameter
    # uncertainty |dtheta| <= |g| / lambda_min.
    h = 1e-5
    dim = len(polished.x)
    hess = np.zeros((dim, dim))
    for j in range(dim):
        step = np.zeros(dim)
        step[j] = h
        hess[:, j] = (objective(polished.x + step)[1] - objective(polished.x - step)[1]) / (2.0 * h)
    eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    print("hessian_eigenvalues=", [f"{float(e):.6g}" for e in eig])


if __name__ == "__main__":
    main()
