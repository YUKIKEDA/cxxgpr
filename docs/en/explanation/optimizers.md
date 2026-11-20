# How the optimizers work

This page explains what cxxgpr's optimizers are asked to solve, how the problem is reshaped so that unconstrained algorithms can solve it, and how each of the five optimizers works, at the level you need to predict their behaviour and choose between them. It does not tell you how to configure them; {doc}`../how-to/choose-optimizer` does. The likelihood itself is derived in {doc}`gaussian-processes`.

## The problem

Fitting a Gaussian process means choosing hyperparameters $\boldsymbol\phi$, which collect the kernel parameters and the noise variance $\sigma_n^2$. cxxgpr chooses them by maximum marginal likelihood, which is the same as minimizing the negative log marginal likelihood

$$
\mathcal{L}(\boldsymbol\phi) = \tfrac12\,\mathbf{y}^\top K_y^{-1}\mathbf{y} + \tfrac12\log\lvert K_y\rvert + \tfrac n2\log 2\pi,
\qquad K_y = K_{\boldsymbol\phi} + \sigma_n^2 I.
$$

Three properties of $\mathcal{L}$ shape every design decision below.

- **It is expensive.** Each evaluation factors an $n \times n$ matrix, which costs $O(n^3)$. Gradients and Hessians reuse that factorization but add further $O(n^3)$ and $O(n^2)$ work. An optimizer that needs fewer evaluations is worth more than one that is cheap per step.
- **It is smooth but not convex.** Different starting points can lead to different local minima, for example one that explains the data with a long length scale and low noise and another with a short length scale and high noise. No local method can promise the global minimum.
- **Its parameters are positive and span orders of magnitude.** Length scales, signal variances and noise variances are all positive, and the useful values range from $10^{-3}$ to $10^{3}$.

The library supplies exact values, gradients and Hessians of $\mathcal{L}$ for the kernels it ships, which is why gradient-based and second-order methods are available at all.

## Reshaping the problem: two coordinate changes

The optimizers do not search the positive orthant directly. The parameters pass through two changes of variable, so that the solvers see an unconstrained problem whose axes have comparable scale.

```{mermaid}
flowchart LR
    A["user parameters<br/>p in (lo, hi)"] -- "log" --> B["log parameters<br/>theta in (log lo, log hi)"]
    B -- "bounded logit" --> C["unconstrained<br/>z in R^d"]
    C -- "optimizer works here" --> C
```

**Logarithm.** The objective is a function of $\theta_i = \log p_i$. Positivity is automatic, and a step of a given size means the same *relative* change whatever the scale of the parameter. Gradients with respect to $\theta_i$ are the natural sensitivities for scale parameters.

**Bounded logit.** Each parameter has an open interval $(lo_i, hi_i)$ in user units. In log space this is $(\ell_i, u_i)$ with $\ell_i = \log lo_i$, $u_i = \log hi_i$ and width $w_i = u_i - \ell_i$. The gradient-based optimizers need an unconstrained variable, so $\theta_i$ is written as a scaled sigmoid of $z_i$:

$$
\theta_i = \ell_i + w_i\,s_i, \qquad s_i = \frac{1}{1 + e^{-4 z_i / w_i}}, \qquad
z_i = \frac{w_i}{4}\log\frac{p_i}{1 - p_i},\quad p_i = \frac{\theta_i - \ell_i}{w_i}.
$$

The factor $4/w_i$ is chosen so that the map has unit slope at the midpoint of the interval:

$$
\frac{d\theta_i}{dz_i} = 4 s_i (1 - s_i), \qquad \left.\frac{d\theta_i}{dz_i}\right|_{z_i = 0} = 1.
$$

Near the middle of each interval, a step of length one in $z$ is a step of length one in $\theta$, regardless of how wide the interval is. Without that normalization a very wide interval such as $(10^{-5}, 10^5)$ would make the sigmoid nearly flat and the problem badly scaled. Close to a bound the slope tends to zero, so the optimizer is slowed down before it can reach the boundary, and the result is always strictly inside the interval.

Derivatives follow from the chain rule. With $J = \operatorname{diag}(d\theta_i/dz_i)$,

$$
\nabla_z \mathcal{L} = J\,\nabla_\theta \mathcal{L}, \qquad
H_z = J\,H_\theta\,J + \operatorname{diag}\!\Bigl(\nabla_\theta \mathcal{L} \odot \tfrac{d^2\theta}{dz^2}\Bigr), \qquad
\frac{d^2\theta_i}{dz_i^2} = \frac{16}{w_i}\,s_i (1 - s_i)(1 - 2 s_i).
$$

The second term in $H_z$ is the curvature of the coordinate map itself. It is easy to forget and the second-order method depends on it. `LogBounds` implements exactly these maps and chain rules.

Two optimizers deviate: `FastSimulatedAnnealing` works directly in $\theta$ and handles the interval itself, and `Fixed` does not move at all.

## L-BFGS

`Lbfgs` is a quasi-Newton method. Newton's method would solve $H_z\,\mathbf{p} = -\nabla_z\mathcal{L}$ for the step $\mathbf{p}$, but that needs the Hessian. L-BFGS never forms it. It keeps the last $m$ pairs of step and gradient change,

$$
\mathbf{s}_k = z_{k+1} - z_k, \qquad \mathbf{y}_k = \nabla\mathcal{L}(z_{k+1}) - \nabla\mathcal{L}(z_k),
$$

and uses the two-loop recursion to compute the product of an implicit inverse-Hessian approximation with the current gradient from those pairs alone. The history size $m$ (default 10) bounds the memory and the cost per step. A pair is kept only if its curvature $\mathbf{s}_k^\top\mathbf{y}_k$ is safely positive, which keeps the approximation positive definite, and the method falls back to steepest descent if the computed direction is not a descent direction.

The step length comes from a line search along the quasi-Newton direction. cxxgpr uses the Moré–Thuente search, which finds a step satisfying the strong Wolfe conditions: a sufficient decrease of the value and a flattening of the slope,

$$
\mathcal{L}(z + \alpha\mathbf{p}) \le \mathcal{L}(z) + c_1\alpha\,\nabla\mathcal{L}^\top\mathbf{p}, \qquad
\lvert\nabla\mathcal{L}(z + \alpha\mathbf{p})^\top\mathbf{p}\rvert \le c_2\,\lvert\nabla\mathcal{L}^\top\mathbf{p}\rvert,
$$

with $c_1 = 10^{-4}$ and $c_2 = 0.9$. The search bracket-and-interpolates and evaluates the value and gradient together, so the cost of a line search is a few evaluations. The Wolfe conditions guarantee that the curvature pair is usable. The optimizer stops when the gradient is small relative to the size of $z$, or when both the relative change of the value and the gradient are small. The tolerance controls these tests.

What to expect: fast convergence (typically tens of iterations) on smooth, well-scaled problems, with a cost of one value-and-gradient evaluation per iteration plus line-search extras. What can go wrong: on a plateau, where the gradient is almost zero, the stopping test can fire although the point is not a minimum. `converged` then means "the algorithm's own test was satisfied", not "this is the best model".

## Trust-region Newton

`TrustRegion` uses the exact Hessian. At the current point it minimizes the local quadratic model

$$
m(\mathbf{p}) = \nabla\mathcal{L}^\top\mathbf{p} + \tfrac12\,\mathbf{p}^\top H\,\mathbf{p}
\qquad\text{subject to}\qquad \lVert\mathbf{p}\rVert \le \Delta.
$$

The radius $\Delta$ is the distance within which the model is trusted. After computing a step, the optimizer compares the *actual* decrease of $\mathcal{L}$ to the decrease the model *predicted*:

$$
\rho = \frac{\mathcal{L}(z) - \mathcal{L}(z + \mathbf{p})}{-m(\mathbf{p})}.
$$

A small $\rho$ means the model was poor, so the step is rejected or the radius shrinks. The rules in cxxgpr: a step is accepted if $\rho > 10^{-4}$; the radius is multiplied by $1/4$ if $\rho < 0.25$ and doubled (up to the maximum radius) if $\rho > 0.75$ and the step reached the boundary. Because the radius adapts, the method needs no line search and works even where the Hessian is indefinite.

The subproblem is solved approximately by the **Steihaug–Toint** conjugate-gradient method. It runs the usual conjugate-gradient iteration on $H\mathbf{p} = -\nabla\mathcal{L}$ starting from $\mathbf{p} = 0$, and stops early in three cases: the iterate would leave the trust region (the step is cut at the boundary), a direction of non-positive curvature is found (the step follows it to the boundary, which is how the method escapes saddle points), or the residual is small enough. This makes the solver safe on the indefinite Hessians that appear far from a minimum, at the price of a few $d \times d$ matrix-vector products.

What to expect: few iterations (under twenty in the how-to comparison), but each iteration needs the Hessian, which costs more than a gradient. Because it also handles negative curvature well, it is the best choice on difficult surfaces. With only a handful of hyperparameters the Hessian is cheap relative to the $O(n^3)$ factorization it shares.

## Nelder–Mead

`NelderMead` is derivative-free: it uses only values of $\mathcal{L}$. It maintains a simplex of $d + 1$ points in $z$-space, the vertices of a $d$-dimensional triangle. In each iteration it reflects the worst vertex through the centroid of the others, and depending on how the new point compares it expands further, contracts toward the centroid, or shrinks the whole simplex toward the best vertex. cxxgpr uses the standard coefficients: reflection 1, expansion 2, contraction 0.5 and shrink 0.5. The first simplex is built around the start by moving each coordinate by 5% (or by $0.25$ for a coordinate near zero). It stops when the spread of values across the simplex and the simplex diameter are both below the tolerance.

What to expect: robust and simple, and indifferent to noise in the gradients, but slow, because each iteration discards information that gradient methods use. In the table of the how-to guide it needs about six times as many iterations as L-BFGS on a three-parameter problem. The cost grows quickly with the number of parameters. It is still a local method: the result depends on the starting simplex.

## Fast simulated annealing

`FastSimulatedAnnealing` is stochastic and global in intent. It works in the log parameters $\theta$ directly. In iteration $k$ it cycles to one coordinate and proposes a heavy-tailed move,

$$
\theta_j' = \theta_j + T_k \tan\bigl(\pi(u - \tfrac12)\bigr), \qquad u \sim \mathcal{U}(0, 1),
$$

a Cauchy step whose scale is the current temperature $T_k$. The Cauchy distribution has heavy tails, so most steps are small but occasional steps are large; that is what lets the search leave a local basin. A proposal outside the interval is clamped to just inside it (`BoundaryPolicy::Clamp`) or reflected back (`BoundaryPolicy::Reflect`). The move is accepted by the Metropolis rule: always if it lowers $\mathcal{L}$, and with probability $e^{-\Delta\mathcal{L}/T_k}$ if it raises it. The temperature follows Ingber's exponential schedule

$$
T_k = T_0\,\exp\!\bigl(-c\,(k/K)^{1/d}\bigr),
$$

where $K$ is the iteration budget, $d$ the number of parameters, $T_0$ the initial temperature and $c$ the cooling rate. Early on, uphill moves are accepted often and the search explores; as $T_k$ falls it behaves more and more like a descent. The optimizer returns the best point it visited, not the last one. A proposal whose evaluation fails or is not finite is skipped.

Because each proposal changes exactly one coordinate, the optimizer reports which coordinates changed through `Objective::value_at_changes`. The GP objective uses this to recompute only the kernel terms affected by the change, which makes a sum kernel much cheaper per proposal. This is the reason `FastSimulatedAnnealing` declares `uses_change_indices = true`.

What to expect: it uses its whole budget, lands near the minimum rather than on it, and needs hundreds to thousands of evaluations. It has no convergence test, so `converged` is `true` once the budget is spent. It is a good first stage for a rough surface, or a source of a starting point for a local optimizer.

## Fixed

`Fixed` runs no search. It validates the bounds and the starting point, evaluates the objective once, and returns the starting point with zero iterations. It exists so the same `Gpr` code path serves a model whose hyperparameters you already know.

## Restarts and determinism

Every optimizer except `Fixed` can restart. After the run from your starting point, it draws `count` further starting points uniformly in the log-space box,

$$
\theta_i \sim \mathcal{U}(\ell_i, u_i),
$$

from a seeded `Xoshiro256PlusPlus` generator, runs from each, and returns the run with the lowest finite objective. A run that fails is skipped. Because the generator is seeded and the solvers are deterministic, the whole procedure is reproducible, including bit for bit: the same seed gives identical results. `FastSimulatedAnnealing` also draws its own random numbers, seeded by `with_seed`; each run adds its run index to the seed so that restarts explore different trajectories yet remain reproducible.

Restarts cost proportionally more and help most when the likelihood has several basins. They do not help when the problem is a plateau; a better start does.

## Why five optimizers

No single algorithm dominates, because the trade-offs are real:

| | Gradient needed | Handles indefinite curvature | Global exploration | Typical evaluations | Weakness |
|---|---|---|---|---|---|
| `Lbfgs` | yes | via fallback | no | tens | can stop early on plateaus |
| `TrustRegion` | gradient and Hessian | yes | no | tens | needs exact Hessians |
| `NelderMead` | no | n/a | no | hundreds | slow, scales badly with $d$ |
| `FastSimulatedAnnealing` | no | n/a | yes | hundreds to thousands | lands near the minimum, not on it |
| `Fixed` | no | n/a | no | one | does not optimize |

```{mermaid}
flowchart TD
    Q{"Hyperparameters known?"} -- yes --> F["Fixed"]
    Q -- no --> G{"Result stable across starts?"}
    G -- yes --> L["Lbfgs"]
    G -- "no, ill-conditioned" --> T["TrustRegion"]
    G -- "no, many local minima" --> R["add restarts, or start with FastSimulatedAnnealing"]
    L --> S{"Stalls or gradients unreliable?"}
    S -- yes --> N["NelderMead"]
```

The library keeps all five self-contained so that none of them depends on an external optimization package, and so that an optimizer's behaviour is the same on every platform.

## Limits worth knowing

- **Local minima are real.** The reported optimum is a local one unless restarts or annealing found a better basin. Compare the final negative log marginal likelihood across runs.
- **`converged` is the algorithm's own opinion.** A stalled L-BFGS run can report success on a plateau; a run that exhausts its budget reports `false` but still returns the best point found.
- **Noise can be driven to its lower bound.** On noise-free data the optimizer may push $\sigma_n^2$ to the lower end of its interval. That is a property of the data, not a failure; widen or fix the interval if you need a floor.
- **The bounds are a model assumption.** A minimum that lies outside an interval cannot be found. If a fitted parameter sits at an interval end, widen the interval or reconsider the kernel.
