# SPDX-License-Identifier: Apache-2.0
"""Independent pure-Python reference solver for M5-2 Exact GPR golden verification."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

Matrix = list[list[float]]
Vector = list[float]


def cholesky_lower(a: Matrix) -> Matrix:
    """Compute lower-triangular Cholesky factor L such that A = L @ L^T."""
    n = len(a)
    l_mat: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = a[i][j]
            for k in range(j):
                acc -= l_mat[i][k] * l_mat[j][k]
            if i == j:
                if acc <= 0.0:
                    raise ValueError("Matrix is not positive definite")
                l_mat[i][j] = math.sqrt(acc)
            else:
                l_mat[i][j] = acc / l_mat[j][j]
    return l_mat


def forward_solve(l_mat: Matrix, b: Vector) -> Vector:
    """Solve L @ z = b."""
    n = len(b)
    z: Vector = [0.0] * n
    for i in range(n):
        acc = b[i]
        for k in range(i):
            acc -= l_mat[i][k] * z[k]
        z[i] = acc / l_mat[i][i]
    return z


def backward_solve(l_mat: Matrix, z: Vector) -> Vector:
    """Solve L^T @ x = z."""
    n = len(z)
    x: Vector = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = z[i]
        for k in range(i + 1, n):
            acc -= l_mat[k][i] * x[k]
        x[i] = acc / l_mat[i][i]
    return x


def inverse_diag_from_cholesky(l_mat: Matrix) -> Vector:
    """Compute diagonal entries of A^{-1} = (L L^T)^{-1} via unit vector solves."""
    n = len(l_mat)
    diag: Vector = [0.0] * n
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        z = forward_solve(l_mat, e)
        diag[col] = sum(v * v for v in z)
    return diag


def rbf_gram(x1: list[Vector], x2: list[Vector], const_val: float, ls: float) -> Matrix:
    """Evaluate c * exp(-0.5 * ||x - x'||^2 / ls^2)."""
    inv_ls2 = 1.0 / (ls * ls)
    mat: Matrix = []
    for p in x1:
        row: Vector = []
        for q in x2:
            sq = sum((a - b) * (a - b) for a, b in zip(p, q, strict=True))
            row.append(const_val * math.exp(-0.5 * sq * inv_ls2))
        mat.append(row)
    return mat


def matern52_ard_gram(x1: list[Vector], x2: list[Vector], const_val: float, ls: Vector) -> Matrix:
    """Evaluate c * Matern5/2_ARD(x, x')."""
    sqrt5 = math.sqrt(5.0)
    mat: Matrix = []
    for p in x1:
        row: Vector = []
        for q in x2:
            r2 = sum(((a - b) / ell) ** 2 for a, b, ell in zip(p, q, ls, strict=True))
            r = math.sqrt(r2)
            s = sqrt5 * r
            k = const_val * (1.0 + s + (5.0 / 3.0) * r2) * math.exp(-s)
            row.append(k)
        mat.append(row)
    return mat


@dataclass
class GprEval:
    """Exact GPR evaluation outputs."""

    nll: float
    alpha: Vector
    mean: Vector
    variance: Vector
    cov: Matrix
    loo_mean: Vector
    loo_var: Vector


def evaluate_exact_gpr(
    k_tt: Matrix,
    k_tq: Matrix,
    k_qq: Matrix,
    y: Vector,
    noise_var: float,
) -> GprEval:
    """Evaluate NLL, posterior mean/variance/covariance, and LOO from Gram matrices."""
    n = len(y)
    m = len(k_qq)
    a_mat: Matrix = [[k_tt[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        a_mat[i][i] += noise_var

    l_mat = cholesky_lower(a_mat)
    alpha = backward_solve(l_mat, forward_solve(l_mat, y))
    quad = sum(yi * ai for yi, ai in zip(y, alpha, strict=True))
    log_det = 2.0 * sum(math.log(l_mat[i][i]) for i in range(n))
    nll = 0.5 * (quad + log_det + n * math.log(2.0 * math.pi))

    mean: Vector = [sum(k_tq[i][j] * alpha[i] for i in range(n)) for j in range(m)]

    v_cols: list[Vector] = []
    for j in range(m):
        col = [k_tq[i][j] for i in range(n)]
        v_cols.append(forward_solve(l_mat, col))

    cov: Matrix = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            vdot = sum(v_cols[i][k] * v_cols[j][k] for k in range(n))
            cov[i][j] = k_qq[i][j] - vdot
    variance = [cov[j][j] + noise_var for j in range(m)]

    inv_diag = inverse_diag_from_cholesky(l_mat)
    loo_mean = [y[i] - alpha[i] / inv_diag[i] for i in range(n)]
    loo_var = [1.0 / inv_diag[i] for i in range(n)]
    return GprEval(
        nll=nll,
        alpha=alpha,
        mean=mean,
        variance=variance,
        cov=cov,
        loo_mean=loo_mean,
        loo_var=loo_var,
    )


def optimize_lbfgs(
    eval_fn: Callable[[Vector], float],
    theta0: Vector,
    max_iter: int = 150,
) -> tuple[Vector, float]:
    """Minimize NLL using two-loop L-BFGS with Armijo backtracking and central FD gradients."""
    dim = len(theta0)
    eps = 1e-6

    def grad_fn(th: Vector) -> tuple[float, Vector]:
        val = eval_fn(th)
        g = [0.0] * dim
        for k in range(dim):
            tp = list(th)
            tm = list(th)
            tp[k] += eps
            tm[k] -= eps
            g[k] = (eval_fn(tp) - eval_fn(tm)) / (2.0 * eps)
        return val, g

    x = list(theta0)
    f, g = grad_fn(x)
    s_hist: list[Vector] = []
    y_hist: list[Vector] = []
    rho_hist: list[float] = []
    m_mem = 10

    for _ in range(max_iter):
        gnorm = math.sqrt(sum(gi * gi for gi in g))
        if gnorm < 1e-8:
            break
        q = list(g)
        k_hist = len(s_hist)
        alphas = [0.0] * k_hist
        for i in range(k_hist - 1, -1, -1):
            alphas[i] = rho_hist[i] * sum(s_hist[i][j] * q[j] for j in range(dim))
            for j in range(dim):
                q[j] -= alphas[i] * y_hist[i][j]
        gamma = 1.0
        if k_hist > 0:
            sy = sum(s_hist[-1][j] * y_hist[-1][j] for j in range(dim))
            yy = sum(y_hist[-1][j] * y_hist[-1][j] for j in range(dim))
            if yy > 0.0:
                gamma = sy / yy
        r = [gamma * qj for qj in q]
        for i in range(k_hist):
            beta = rho_hist[i] * sum(y_hist[i][j] * r[j] for j in range(dim))
            for j in range(dim):
                r[j] += s_hist[i][j] * (alphas[i] - beta)
        direction = [-rj for rj in r]
        slope = sum(g[j] * direction[j] for j in range(dim))
        if slope >= 0.0:
            direction = [-gj for gj in g]
            slope = -sum(gj * gj for gj in g)

        step = 1.0
        accepted = False
        x_next = list(x)
        f_next = f
        for _ls in range(30):
            cand = [x[j] + step * direction[j] for j in range(dim)]
            if all(math.log(1e-5) < c < math.log(1e5) for c in cand) and cand[1] > math.log(0.05):
                try:
                    f_cand = eval_fn(cand)
                    if f_cand <= f + 1e-4 * step * slope:
                        x_next = cand
                        f_next = f_cand
                        accepted = True
                        break
                except ValueError:
                    pass
            step *= 0.5
        if not accepted:
            break
        _, g_next = grad_fn(x_next)
        s = [x_next[j] - x[j] for j in range(dim)]
        y_vec = [g_next[j] - g[j] for j in range(dim)]
        sy = sum(s[j] * y_vec[j] for j in range(dim))
        if sy > 1e-12:
            if len(s_hist) == m_mem:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s)
            y_hist.append(y_vec)
            rho_hist.append(1.0 / sy)
        x = x_next
        f = f_next
        g = g_next
    return x, f


def main() -> None:
    """Compute and print M5-2 golden reference values."""
    # Case 1: 1D scaled noisy Forrester (n=18) so MLE noise_variance is strictly interior
    x_forr = [[i / 17.0] for i in range(18)]
    y_forr = [
        0.1 * ((6.0 * p[0] - 2.0) ** 2) * math.sin(12.0 * p[0] - 4.0) + 0.02 * math.cos(50.0 * p[0])
        for p in x_forr
    ]
    xq_forr = [[0.15], [0.50], [0.85]]

    res_forr_fixed = evaluate_exact_gpr(
        rbf_gram(x_forr, x_forr, 1.0, 0.25),
        rbf_gram(x_forr, xq_forr, 1.0, 0.25),
        rbf_gram(xq_forr, xq_forr, 1.0, 0.25),
        y_forr,
        1e-3,
    )
    print("=== FORRESTER FIXED ===")
    print(f"nll={res_forr_fixed.nll:.17g}")
    print("mean=", [f"{v:.17g}" for v in res_forr_fixed.mean])
    print("var=", [f"{v:.17g}" for v in res_forr_fixed.variance])
    print("loo_mean=", [f"{v:.17g}" for v in res_forr_fixed.loo_mean[:4]])
    print("loo_var=", [f"{v:.17g}" for v in res_forr_fixed.loo_var[:4]])

    def forr_nll(th: Vector) -> float:
        c = math.exp(th[0])
        ls = math.exp(th[1])
        nv = math.exp(th[2])
        return evaluate_exact_gpr(
            rbf_gram(x_forr, x_forr, c, ls),
            rbf_gram(x_forr, xq_forr, c, ls),
            rbf_gram(xq_forr, xq_forr, c, ls),
            y_forr,
            nv,
        ).nll

    th_opt_f, nll_opt_f = optimize_lbfgs(forr_nll, [math.log(1.0), math.log(0.25), math.log(1e-3)])
    res_forr_opt = evaluate_exact_gpr(
        rbf_gram(x_forr, x_forr, math.exp(th_opt_f[0]), math.exp(th_opt_f[1])),
        rbf_gram(x_forr, xq_forr, math.exp(th_opt_f[0]), math.exp(th_opt_f[1])),
        rbf_gram(xq_forr, xq_forr, math.exp(th_opt_f[0]), math.exp(th_opt_f[1])),
        y_forr,
        math.exp(th_opt_f[2]),
    )
    print("=== FORRESTER OPT ===")
    print(
        f"nll={nll_opt_f:.17g} c={math.exp(th_opt_f[0]):.17g} "
        f"ls={math.exp(th_opt_f[1]):.17g} nv={math.exp(th_opt_f[2]):.17g}"
    )
    print("mean=", [f"{v:.17g}" for v in res_forr_opt.mean])
    print("var=", [f"{v:.17g}" for v in res_forr_opt.variance])

    # Case 2: 2D ARD Matern 5/2 (n=10)
    x_2d = [[0.1 * i, 0.1 * ((3 * i + 1) % 10)] for i in range(10)]
    y_2d = [math.sin(3.0 * p[0]) + 0.5 * math.cos(4.0 * p[1]) for p in x_2d]
    xq_2d = [[0.25, 0.45], [0.75, 0.15]]

    res_2d_fixed = evaluate_exact_gpr(
        matern52_ard_gram(x_2d, x_2d, 1.5, [0.5, 0.7]),
        matern52_ard_gram(x_2d, xq_2d, 1.5, [0.5, 0.7]),
        matern52_ard_gram(xq_2d, xq_2d, 1.5, [0.5, 0.7]),
        y_2d,
        1e-3,
    )
    print("=== 2D ARD FIXED ===")
    print(f"nll={res_2d_fixed.nll:.17g}")
    print("mean=", [f"{v:.17g}" for v in res_2d_fixed.mean])
    print("var=", [f"{v:.17g}" for v in res_2d_fixed.variance])
    print("loo_mean=", [f"{v:.17g}" for v in res_2d_fixed.loo_mean[:4]])
    print("loo_var=", [f"{v:.17g}" for v in res_2d_fixed.loo_var[:4]])

    def m2d_nll(th: Vector) -> float:
        c = math.exp(th[0])
        ls = [math.exp(th[1]), math.exp(th[2])]
        nv = math.exp(th[3])
        return evaluate_exact_gpr(
            matern52_ard_gram(x_2d, x_2d, c, ls),
            matern52_ard_gram(x_2d, xq_2d, c, ls),
            matern52_ard_gram(xq_2d, xq_2d, c, ls),
            y_2d,
            nv,
        ).nll

    th_opt_2d, nll_opt_2d = optimize_lbfgs(
        m2d_nll, [math.log(1.0), math.log(0.5), math.log(0.5), math.log(1e-2)]
    )
    res_2d_opt = evaluate_exact_gpr(
        matern52_ard_gram(
            x_2d,
            x_2d,
            math.exp(th_opt_2d[0]),
            [math.exp(th_opt_2d[1]), math.exp(th_opt_2d[2])],
        ),
        matern52_ard_gram(
            x_2d,
            xq_2d,
            math.exp(th_opt_2d[0]),
            [math.exp(th_opt_2d[1]), math.exp(th_opt_2d[2])],
        ),
        matern52_ard_gram(
            xq_2d,
            xq_2d,
            math.exp(th_opt_2d[0]),
            [math.exp(th_opt_2d[1]), math.exp(th_opt_2d[2])],
        ),
        y_2d,
        math.exp(th_opt_2d[3]),
    )
    print("=== 2D ARD OPT ===")
    print(
        f"nll={nll_opt_2d:.17g} c={math.exp(th_opt_2d[0]):.17g} "
        f"l0={math.exp(th_opt_2d[1]):.17g} l1={math.exp(th_opt_2d[2]):.17g} "
        f"nv={math.exp(th_opt_2d[3]):.17g}"
    )
    print("mean=", [f"{v:.17g}" for v in res_2d_opt.mean])
    print("var=", [f"{v:.17g}" for v in res_2d_opt.variance])


if __name__ == "__main__":
    main()
