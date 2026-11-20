# Gaussian process regression

This page explains the mathematics behind the model that cxxgpr fits: what a Gaussian process is, how conditioning on data gives a prediction with an error bar, and how the hyperparameters are chosen. It derives the formulas the library evaluates and connects each of them to the code. The standard reference is Rasmussen and Williams, *Gaussian Processes for Machine Learning* (GPML), whose notation we follow.

## Regression as inference over functions

Suppose we observe $n$ pairs $(x_i, y_i)$, with inputs $x_i \in \mathbb{R}^d$ stacked in a matrix $X$ and targets in a vector $\mathbf{y}$. We believe that the targets are a function of the input, seen through measurement noise:

$$
y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2).
$$

Ordinary regression picks a parametric family for $f$ and fits its parameters. A Gaussian process (GP) instead places a probability distribution directly over *functions*. It is a collection of random variables, any finite subset of which is jointly Gaussian. A GP is fully specified by a mean function and a covariance function (the **kernel**):

$$
f \sim \mathcal{GP}\bigl(m(\cdot),\, k(\cdot, \cdot)\bigr), \qquad
m(x) = \mathbb{E}[f(x)], \quad
k(x, x') = \operatorname{Cov}\bigl(f(x), f(x')\bigr).
$$

cxxgpr uses a zero prior mean. That is not a restriction in practice, because the target transform `StandardizeTarget` shifts the data to zero mean before training and shifts predictions back afterwards.

For any inputs $X$, the vector of function values is a draw from a multivariate normal distribution, $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, K)$, where $K_{ij} = k(x_i, x_j)$. The kernel is where all of your assumptions live: how smooth $f$ is, how fast it varies, whether it is periodic. {doc}`kernels` discusses the available kernels and how to combine them.

## The posterior: conditioning on the data

Let $X_*$ hold $m$ query inputs and $\mathbf{f}_*$ the function values there. The observed targets and the unknown function values are jointly Gaussian:

$$
\begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix}
\sim \mathcal{N}\!\left(\mathbf{0},
\begin{bmatrix} K + \sigma_n^2 I & K_* \\ K_*^\top & K_{**} \end{bmatrix}\right),
$$

with $K = k(X, X)$, $K_* = k(X, X_*)$ (an $n \times m$ matrix) and $K_{**} = k(X_*, X_*)$. Conditioning a joint Gaussian on one block gives another Gaussian, so the predictive distribution is available in closed form:

$$
\mathbf{f}_* \mid X, \mathbf{y}, X_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)
$$

with

$$
\boldsymbol{\mu}_* = K_*^\top \boldsymbol{\alpha}, \qquad
\boldsymbol{\alpha} = (K + \sigma_n^2 I)^{-1} \mathbf{y},
$$

$$
\Sigma_* = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*.
$$

Read these in words. The mean is a weighted sum of kernel functions centred on the training inputs, with weights $\boldsymbol{\alpha}$. The covariance is the prior covariance minus the amount of uncertainty that the data removed. Far from every training point $K_*$ is close to zero, so the mean falls back to the prior mean (zero) and the variance falls back to the prior variance $k(x_*, x_*)$. This is the behaviour you see in {doc}`../tutorials/multi-dimensional-regression` when the model extrapolates.

### Latent and observation variance

$\Sigma_*$ is the uncertainty about the noise-free function $f$. A new *measurement* at $x_*$ would add independent noise on top, so its variance is larger by $\sigma_n^2$:

$$
\operatorname{Var}[y_*] = \operatorname{Var}[f_*] + \sigma_n^2.
$$

cxxgpr offers both through `VarianceKind::Latent` and `VarianceKind::Observation`; the default for `predict` is `Observation`. Use `Latent` when you want an error bar on the underlying function, and `Observation` for predicting what you will measure.

### Computation with a Cholesky factor

Nobody forms $(K + \sigma_n^2 I)^{-1}$ explicitly. The matrix is symmetric positive definite, so it has a Cholesky factorisation $K + \sigma_n^2 I = L L^\top$ with $L$ lower triangular. Then

$$
\boldsymbol{\alpha} = L^{-\top} (L^{-1} \mathbf{y}), \qquad
V = L^{-1} K_*, \qquad
\boldsymbol{\mu}_* = K_*^\top \boldsymbol{\alpha}, \qquad
\Sigma_* = K_{**} - V^\top V,
$$

where each $L^{-1}(\cdot)$ is a triangular solve. Fitting costs $O(n^3)$ time for the factorisation and $O(n^2)$ memory for $K$ and $L$; every later prediction at $m$ points costs $O(n m)$ for the means and $O(n^2 m)$ for the variances, which need the triangular solves. The factor $L$ and the weights $\boldsymbol{\alpha}$ are exactly what a `FittedGpr` stores and what `persist::save_with_factor` writes to disk. For large $n$, this cubic cost is the main limitation of exact GP regression, and the reason why approximate sparse models exist (see {doc}`relationship-to-gprx` for why cxxgpr does not include them).

```{mermaid}
flowchart LR
    D["data X, y"] --> K["kernel matrix<br/>K + noise I"]
    K --> C["Cholesky<br/>L L^T"]
    C --> A["weights alpha"]
    C --> V["solve L V = K*"]
    A --> M["mean = K*^T alpha"]
    V --> S["variance = K** - V^T V"]
```

## Choosing the hyperparameters

A kernel has parameters: lengthscales, signal variances, periods. The noise variance $\sigma_n^2$ is one more. Together they form the hyperparameter vector $\boldsymbol{\theta}$. GPML selects $\boldsymbol{\theta}$ by maximising the **marginal likelihood**, the probability of the observed targets with the function values integrated out. Because $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K_\theta + \sigma_n^2 I)$, its logarithm is

$$
\log p(\mathbf{y} \mid X, \boldsymbol{\theta})
= -\tfrac{1}{2}\, \mathbf{y}^\top \boldsymbol{\alpha}
  \;-\; \tfrac{1}{2} \log \lvert K_\theta + \sigma_n^2 I \rvert
  \;-\; \tfrac{n}{2} \log 2\pi,
$$

and with the Cholesky factor the determinant is cheap: $\log \lvert K_\theta + \sigma_n^2 I \rvert = 2 \sum_i \log L_{ii}$. The three terms have a clear meaning. The first measures how well the model explains the data. The second penalises model complexity: a very flexible kernel spreads its probability over many possible datasets and gets a large determinant. The third is a constant. Together they implement Occam's razor without a separate validation set or a regularisation constant that you must tune.

cxxgpr minimises the negative of this quantity with one of its optimizers (see {doc}`optimizers`) using analytic gradients with respect to every hyperparameter. After `Gpr::fit`, the value at the optimum is available as `FittedGpr::log_marginal_likelihood`. It is a useful number for comparing two kernels on the same data: the larger, the better the data are explained after accounting for complexity.

:::{note}
The marginal likelihood surface is generally not convex. A gradient-based optimizer finds a local maximum, and for short lengthscales or small noise, a poor one is easy to reach. Standardising inputs and targets, choosing sensible starting values and, when needed, running several restarts all help. See {doc}`../how-to/tune-hyperparameters`.
:::

## Leave-one-out predictions

The GP framework gives leave-one-out (LOO) cross-validation almost for free. With $Q = (K + \sigma_n^2 I)^{-1}$, the predictive mean and variance for training point $i$ when that point is held out are

$$
\mu_{-i} = y_i - \frac{\alpha_i}{Q_{ii}}, \qquad
\sigma_{-i}^2 = \frac{1}{Q_{ii}},
$$

(GPML section 5.4.2). The diagonal of $Q$ comes from the existing Cholesky factor, so all $n$ LOO predictions together cost $O(n^2)$ extra, not $n$ separate fits. `FittedGpr::loo_predict` returns them on the original target scale. Comparing them with the observed targets, as in {doc}`../tutorials/multi-dimensional-regression`, tells you whether the model and its error bars are trustworthy.

## Noise and jitter are different things

The observation noise $\sigma_n^2$ is part of the *model*: it states that measurements are noisy, it is learned from the data, and it appears in the observation variance of predictions. **Jitter** is a numerical device: a tiny non-negative number $j$ added to the diagonal only when a Cholesky factorisation fails because rounding made $K + \sigma_n^2 I$ numerically indefinite. cxxgpr keeps the two strictly separate. The first factorisation attempt always uses $j = 0$, jitter is applied only after a failure according to a `JitterPolicy`, and the value that was actually used is stored with the model as `factor_jitter`. Read {doc}`numerical-stability` for the details.

## From mathematics to the library

| Concept | In cxxgpr |
|---|---|
| Kernel $k$ and its hyperparameters | `KernelSpec` built from kernel leaves, combined with `+` and `*` |
| Noise variance $\sigma_n^2$ | `Gpr::with_noise_variance`, learned during `fit` |
| Maximising the marginal likelihood | `Gpr::fit` with an optimizer such as `optimizer::Lbfgs` |
| Cholesky factor $L$, weights $\boldsymbol{\alpha}$ | Stored in `FittedGpr`; saved by `persist::save_with_factor` |
| $\boldsymbol{\mu}_*$ and $\operatorname{diag}\Sigma_*$ | `FittedGpr::predict` returning `Prediction` |
| Full $\Sigma_*$ and posterior samples | `predict_covariance`, `sample` |
| LOO predictions | `FittedGpr::loo_predict` |

For practical recipes see {doc}`../how-to/build-kernels` and {doc}`../how-to/predict-and-sample`.
