# Numerical stability

Gaussian process regression is a sequence of linear-algebra operations on a matrix that is mathematically positive definite and numerically often barely so. This page explains how cxxgpr keeps the computation trustworthy: why observation noise and numerical jitter are kept apart, how the jitter schedule rescues a failing Cholesky factorization, what the mixed-precision refinement does and does not guarantee, how non-finite values are rejected, and where compensated summation is used. For recipes see {doc}`../how-to/optimize-performance` and {doc}`../how-to/use-low-level-building-blocks`; for the error codes see {doc}`../reference/error-codes`.

## Why the covariance matrix is hard to factor

With training inputs $X$, kernel $k$ and observation noise variance $\sigma_n^2$, the system matrix is

$$
A = K + \sigma_n^2 I, \qquad K_{ij} = k(x_i, x_j).
$$

The Gram matrix $K$ is positive semi-definite for a valid kernel. For smooth kernels such as the squared exponential its eigenvalues decay very fast, so $K$ is close to singular whenever points are close together or the length scale is large. Rounding errors of size $u\,\lVert K\rVert$ (with unit roundoff $u$) can then push a tiny eigenvalue below zero, and the Cholesky factorization $A = LL^\top$ fails because it meets a non-positive pivot. The noise term helps: all eigenvalues of $A$ are at least $\sigma_n^2$, so the condition number is bounded by

$$
\kappa(A) \le \frac{\lambda_{\max}(K) + \sigma_n^2}{\sigma_n^2}.
$$

A small noise variance, a large length scale, duplicated inputs, a single-precision factorization or a custom kernel that is not positive semi-definite all push the factorization toward failure.

## Observation noise is not jitter

cxxgpr treats two diagonal additions as different things, in the type system as well as in the maths.

**Observation noise** $\sigma_n^2$ is part of the model. It lives in `GaussianLikelihood`, is a hyperparameter that the optimizer learns (as $\theta_{\text{noise}} = \log\sigma_n^2$, the last entry of the parameter vector), is bounded by an interval (by default $(10^{-5}, 10^{5})$), and appears in the marginal likelihood and, for `VarianceKind::Observation`, in predictive variances.

**Jitter** $j$ is a numerical device. It lives in `JitterPolicy`, is not a parameter, is never optimized, and is added only when the plain factorization has failed. The jitter that was finally used is stored in the model (`FittedGpr::factor_jitter()`, and in saved files) so that every later solve refers to the same matrix.

Keeping them apart matters for three reasons:

- Mixing them would change the model. If jitter were folded into $\sigma_n^2$, a purely numerical rescue would change the reported noise level and the predictive variance.
- The marginal likelihood must be consistent. The value, gradient and Hessian of the objective are those of the matrix that was actually factored, $A + jI$, and the gradient with respect to $\theta_{\text{noise}}$ is $\sigma_n^2 I$ only; jitter has no derivative.
- Refinement must target one system. The weights $\alpha = (A + jI)^{-1} y$ and the mixed-precision refinement refer to the same jittered matrix; no step tries to "undo" the jitter.

Formally, the factored matrix is $B = K + (\sigma_n^2 + j) I$ with $j \ge 0$, and $j = 0$ unless a retry was needed.

## The jitter schedule

Every factorization starts with $j = 0$. Only if it fails with `CholeskyFailed` does the `JitterPolicy` come into play.

```{mermaid}
flowchart TD
  A["Factor A with j = 0"] -->|success| Done["Record factor_jitter = 0"]
  A -->|"CholeskyFailed"| S{"Policy has a next jitter j?"}
  S -->|no| F["Return CholeskyFailed<br/>(last j tried, matrix size, stage)"]
  S -->|yes| R["Restore lower triangle of A<br/>from the backup, add j to the diagonal"]
  R --> T["Factor A + jI"]
  T -->|success| Done2["Record factor_jitter = j"]
  T -->|"CholeskyFailed"| S
```

The two policy kinds generate the sequence of retry values:

- `JitterPolicy::fixed(j)` makes at most one retry at $j$. `fixed(0.0)` never retries and is the default policy of a default-constructed `JitterPolicy` and of `Gpr`.
- `JitterPolicy::adaptive(initial, multiplier, max_retries, max_jitter)` tries $j_1 = \text{initial}$, then $j_{k+1} = m\, j_k$ with $m = \text{multiplier} > 1$, for at most `max_retries` attempts. The schedule ends early if the next value would exceed `max_jitter` or overflow to infinity.

`JitterSchedule::next()` yields these values one by one; `JitterPolicy::retry_schedule()` creates it, and `AdaptiveJitter` and `FixedJitter` are the validated parameter holders behind the two kinds. The factory functions reject non-finite values, negative jitter, a multiplier not above 1, zero retries and a cap below the initial value with `InvalidConfig`.

:::{warning}
A `Gpr` that you construct with default settings has the policy `JitterPolicy::fixed(0.0)`: it makes no retries. A near-singular system then fails with `CholeskyFailed` instead of being repaired. Opt in with `Gpr::with_jitter(JitterPolicy::adaptive(...))` when you expect trouble. The doc comment of the `Gpr` constructor says "adaptive jitter", but the code leaves the policy at `fixed(0.0)`.
:::

Three details are worth knowing:

- Each retry restarts from a saved copy of the original lower triangle. The retries do not accumulate: attempt $k$ factors $A + j_k I$, not $A + (j_1 + \dots + j_k) I$.
- The jitter is added in the matrix's own scalar type. For a `float` matrix a jitter below roughly $10^{-7}$ times the diagonal changes nothing.
- When the mixed-precision refinement fails to converge, the jitter is not increased. The fallback solves the same system in double precision.

A good jitter scale is small against the noise variance, so that the model is hardly changed, yet large against $u\,\lVert K\rVert$, so that it cures the failure; a starting value of $10^{-10}$ times the typical diagonal of $K$ and a geometric growth factor of 10 is a reasonable rule of thumb, not a guarantee.

### What a Cholesky failure tells you

`CholeskyFailed` carries the last jitter that was tried (`GprError::jitter()`), the matrix size (`matrix_size()`) and the stage (`cholesky_stage()`, `CholeskyStage::Fit` or `CholeskyStage::Predict`). `Predict` marks the factorization of the posterior covariance during sampling; sampling uses its own adaptive policy by default (initial $10^{-12}$, growth 10, at most 10 retries, cap $10^{-2}$), because a posterior covariance can be numerically rank-deficient. The triangular solves report a different code: `NonPositiveDefiniteMatrix` means that a *given* factor has a non-positive diagonal entry.

On failure `Gpr::fit` and `Gpr::factor` leave the trainer unchanged. The low-level `cholesky_lower_in_place` leaves its argument partially overwritten, which is why `cholesky_lower_with_jitter` works from a backup.

## Mixed-precision iterative refinement

`MixedPrecision` stores the matrix and its Cholesky factor in `float` (roughly half the memory and time) and then repairs the loss of accuracy in the weights $\alpha$ with a few cheap iterations in `double`. This is classical iterative refinement.

Let $B = K + (\sigma_n^2 + j)I$ be the system to solve, $b$ the (transformed) targets and $\tilde L$ the `float` Cholesky factor, so that $\tilde L \tilde L^\top \approx B$. The initial iterate $w_0$ is the `float` solution widened to `double`. Each step computes the residual in double precision and solves for a correction with the cheap factor:

$$
r_k = b - B\,w_k, \qquad \tilde L \tilde L^\top d_k = r_k \ (\text{in float}), \qquad w_{k+1} = w_k + d_k.
$$

The error contracts by about a factor $\kappa(B)\,u_{32}$ per step, with $u_{32} \approx 6 \times 10^{-8}$ the unit roundoff of `float`. The iteration therefore converges quickly for moderately conditioned problems and not at all when $\kappa(B)\,u_{32} \gtrsim 1$.

### Stopping, stagnation and fallback

With $\varepsilon_{64} = 2^{-52}$ and $n$ the number of training points, the loop stops successfully when the scaled residual is below the tolerance

$$
\frac{\lVert r_k\rVert_\infty}{\lVert B\rVert_\infty \lVert w_k\rVert_\infty + \lVert b\rVert_\infty} < 10\, n\, \varepsilon_{64}.
$$

It gives up and re-solves the system with a full double-precision Cholesky factorization if

- the residual norm fails to drop below 0.9 times its previous value in two consecutive steps, or becomes non-finite (`Stagnated`),
- ten corrections have not been enough (`MaxCorrectionsReached`), or
- a converged iterate is rejected by the acceptance check (`RejectedByAccept`; for `PromoteStorage` this means the residual against the exact kernel exceeds the storage-rounding tolerance below).

The fallback is an error-free path: no `GprError` is returned for it, and `FittedGpr::refinement_summary()` records which of the four outcomes (`Converged` or one of the three fallback reasons) occurred, together with the number of corrections and the final residual.

### Two ways to evaluate the residual

The two residual formulas differ in which matrix defines "$B$":

- `PromoteStorage` computes $r = b - \operatorname{double}(B_{32})\,w$ from the stored `float` matrix, promoted entry by entry. This is cheap, but it measures the residual of the *rounded* matrix $B_{32}$, which differs from the true matrix by relative errors of order $u_{32}$. The iterate therefore solves $B_{32}$ to double accuracy and carries the effect of the storage rounding into the weights. cxxgpr adds an acceptance check: after convergence it recomputes the residual with the true double-precision kernel, evaluated column block by column block (blocks of 32 columns, so the full double matrix is never stored), and accepts the iterate when the relative residual is within ten times the float epsilon. Entrywise storage errors of a few float epsilons bound that residual independently of $n$, so a correctly stored matrix passes, while a stored matrix that is far from the true kernel still falls back.
- `ReevaluateKernel` computes the residual from the kernel evaluated in double precision, block by block, in every step. No separate acceptance check is needed.

The tolerance $10\,n\,\varepsilon_{64}$ is a few hundred rounding errors of double precision, which a matrix stored only in `float` does not meet unless all of its entries are exactly representable in `float`: otherwise the data themselves are perturbed by $u_{32} \gg \varepsilon_{64}$. `ReevaluateKernel` is held to it, because its residual uses the exact kernel. `PromoteStorage` is accepted at storage rounding instead, which is the accuracy it can deliver: its weights are the double-accuracy solution of the stored system, closer to the double-precision weights than a plain single-precision solve but not equal to them, with the difference growing with the conditioning (see the measurements in {doc}`../how-to/optimize-performance`). Check `refinement_summary()` for the outcome.

### Small residual is not small error

Refinement controls the residual. The error in the solution can be much larger when $B$ is ill conditioned:

$$
\frac{\lVert w - B^{-1}b\rVert}{\lVert B^{-1}b\rVert} \lesssim \kappa(B)\,\frac{\lVert r\rVert}{\lVert B\rVert\,\lVert w\rVert}.
$$

In the example of {doc}`../how-to/optimize-performance`, the converged `ReevaluateKernel` residual is at the level of $10^{-13}$, yet the weights differ from the double-precision weights by $2 \times 10^{-6}$, because $\kappa(B)$ is of the order $10^{7}$ there. This is not a defect; it is the conditioning of the problem. The predictions are less sensitive than the weights.

### What refinement does not improve

Refinement improves $\alpha$ only. The training Gram matrix and its factor remain `float`, and so does the cross-covariance $K_*$ and the triangular solve $L^{-1}K_*$ that give the predictive variance. The predictive mean therefore improves by far less than the weights do. The marginal likelihood and its gradient during training also use the `float` factor.

## Non-finite values are rejected, not propagated

A `NaN` or infinity in a Gram matrix would silently poison every number after it. cxxgpr checks at the boundaries instead:

| Where | What is checked | Error |
|---|---|---|
| Training and query inputs, targets | Every entry is finite | `NonFiniteInput` |
| Hyperparameters, bounds, noise | Finite and inside the allowed interval | `InvalidHyperparameter`, `InvalidNoiseVariance`, `InvalidInterval` |
| Objective, prediction, refinement and inverse scaling | The marginal likelihood value, the latent predictive variance before it is clamped at zero, the kernel entries re-evaluated during refinement, and the inverse-scaled predictive moments are finite | `NonFiniteKernelValue` |
| Supplied squared distances | Finite, non-negative, symmetric, zero diagonal | `InvalidDistance` |
| Factorization input | Lower triangle finite; pivots finite and positive | `NonFiniteInput`, `CholeskyFailed` |
| Triangular solves | Factor finite with positive diagonal, right-hand side finite | `NonFiniteInput`, `NonPositiveDefiniteMatrix` |
| Reductions | Dot products and Frobenius products check their inputs | `NonFiniteInput` |

Subnormal numbers are valid finite values and are processed normally; the library does not change the floating-point control registers. The library is compiled without fast-math (`/fp:precise` on MSVC, `-fno-fast-math` elsewhere), because fast-math would allow the compiler to assume that `NaN` and infinity never occur and optimize these checks away.

## Compensated summation

Adding many numbers of different size loses low-order digits. cxxgpr uses Neumaier's compensated summation in two places, where sums collect many terms and an order must be fixed:

1. `linalg::log_det_from_l` and `half_log_det_from_l` sum $2\log L_{ii}$ over all pivots.
2. `linalg::parallel_lower_reduce_cols` sums the per-column contributions of gradient and Hessian terms after they have been computed in parallel.

For a running sum $s$ with compensation $c$ and a new term $x$:

$$
t = s + x, \qquad c \leftarrow c + \begin{cases} (s - t) + x & \lvert s\rvert \ge \lvert x\rvert \\ (x - t) + s & \text{otherwise,} \end{cases} \qquad s \leftarrow t,
$$

and the result is $s + c$. The term $c$ collects the rounding error of each addition exactly, so the final error is almost independent of the number of terms.

Other reductions are plain `double` accumulations in a fixed ascending order: `linalg::dot`, `linalg::frobenius_lower`, `linalg::matvec_columns`, the diagonal of the inverse, and the stored-matrix products of the refinement residual. They accumulate in `double` even for `float` inputs, which already removes most of the error, and the fixed order makes them deterministic. They are not compensated.

## Signals the library gives you

You do not have to guess whether a fit was numerically comfortable. A fitted model reports `factor_jitter()` (zero means the plain factorization worked), `refinement_summary()` for mixed precision (status, corrections, final residual), the optimizer result, and the negative log marginal likelihood. Errors identify the stage, the matrix size and the last jitter tried. Together with {doc}`determinism-and-performance`, which explains why these numbers are reproducible, they let you tell a numerical problem from a modelling problem.
