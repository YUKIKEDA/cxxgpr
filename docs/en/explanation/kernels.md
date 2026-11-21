# Kernels: what they mean and how cxxgpr evaluates them

A Gaussian process is defined by its kernel. This page explains what the built-in kernel families say about the functions they describe, why combining them works, how cxxgpr turns a kernel description into fast code, and what the choice between the two exponential backends costs and buys. It does not list the API; for that see {doc}`../how-to/build-kernels`.

## A kernel states a belief about similarity

The kernel $k(x, x')$ is the prior covariance between the function values $f(x)$ and $f(x')$. Everything the model believes before it sees data is in $k$: how rough the function is, whether it repeats, how far apart two inputs must be before they stop informing each other. Training data then updates that belief (see {doc}`gaussian-processes`).

A function $k$ is a valid kernel when every Gram matrix $K_{ij} = k(x_i, x_j)$ is symmetric and positive semi-definite. That single requirement is why you cannot invent a kernel by writing down any decreasing function of distance: the matrix must stay factorable for every set of points. The built-in leaves satisfy it, and so does every sum and product of them.

### Stationary kernels and length scales

Most of the leaves are **stationary**: $k$ depends only on the distance $r = \lVert x - x'\rVert$. Each one has a **length scale** $\ell$ that sets the distance over which function values stay correlated. Doubling $\ell$ stretches the function along $x$ by a factor of two.

With **automatic relevance determination (ARD)** there is one length scale per feature and the distance becomes

$$
s^2 = \sum_{k=1}^{d} \frac{(x_k - x'_k)^2}{\ell_k^2}.
$$

A feature with a large fitted $\ell_k$ barely changes the output, so ARD doubles as feature selection. The price is $d$ hyperparameters instead of one, which needs more data to determine and gives the optimizer more local optima. ARD leaves also need the coordinates, not just a distance matrix, because they must weight each feature separately.

## The leaf families

### Squared exponential

$$
k(r) = \exp\!\Bigl(-\frac{r^2}{2\ell^2}\Bigr)
$$

The `RbfKernel` produces functions that are infinitely differentiable. That is a strong assumption: it fits smooth signals well and extrapolates confidently, but it is a poor model for data with kinks or noise-like roughness, and its Gram matrices are numerically close to singular when points are close together relative to $\ell$ (see {doc}`numerical-stability`).

### Matern

The Matern family has a smoothness parameter $\nu$:

$$
k_\nu(r) = \frac{2^{1-\nu}}{\Gamma(\nu)}\Bigl(\frac{\sqrt{2\nu}\,r}{\ell}\Bigr)^{\nu} K_\nu\!\Bigl(\frac{\sqrt{2\nu}\,r}{\ell}\Bigr),
$$

where $K_\nu$ is the modified Bessel function. Sample paths are $\lceil\nu\rceil - 1$ times differentiable. cxxgpr provides the three half-integer cases that reduce to an exponential times a polynomial:

| `MaternNu` | $\nu$ | $k(r)$ with $z = r/\ell$ | Sample paths |
|---|---|---|---|
| `OneHalf` | $1/2$ | $e^{-z}$ | continuous, nowhere differentiable |
| `ThreeHalves` | $3/2$ | $(1 + \sqrt3 z)\,e^{-\sqrt3 z}$ | once differentiable |
| `FiveHalves` | $5/2$ | $(1 + \sqrt5 z + \tfrac53 z^2)\,e^{-\sqrt5 z}$ | twice differentiable |

As $\nu \to \infty$ the family converges to the squared exponential. Smaller $\nu$ is the more cautious assumption: physical and noisy data are rarely infinitely smooth, which is why `FiveHalves` is a widespread default. The smoothness is a discrete choice made at construction, not a hyperparameter that the optimizer tunes.

### Rational quadratic

$$
k(r) = \Bigl(1 + \frac{r^2}{2\alpha\ell^2}\Bigr)^{-\alpha}
$$

`RationalQuadraticKernel` is an infinite sum of squared-exponential kernels with different length scales: if the inverse squared length scale $\tau = \ell'^{-2}$ is drawn from a Gamma distribution with shape $\alpha$ and rate $\alpha\ell^2$, then $\int e^{-\tau r^2/2}\,p(\tau)\,d\tau$ is exactly this expression. It suits data that vary on several scales at once. As $\alpha \to \infty$ it becomes `RbfKernel`; for small $\alpha$ it has heavier tails and long-range correlation.

### Periodic

$$
k(r) = \exp\!\Bigl(-\frac{2\sin^2(\pi r/p)}{\ell^2}\Bigr)
$$

`PeriodicKernel` makes $f(x) = f(x + p)$ exactly. The length scale controls how smooth one period is. Real signals rarely repeat exactly, so the usual remedy is a product with a decay kernel (see below).

### Constant, linear and white noise

- `ConstantKernel` is $k = c$: all values share a common random offset of variance $c$. Multiplying any kernel by it sets the **signal variance**.
- `LinearKernel` is $k = v\,x\cdot x'$: the function is a random linear map through the origin, which is Bayesian linear regression with weight variance $v$. Add a `ConstantKernel` for an intercept.
- `WhiteKernel` is independent noise of variance $v$ on each training point. It is $v\,\delta_{ij}$ on a square training Gram matrix and 0 between different point sets, so it models measurement noise but never predicts it. `Gpr` has its own observation noise variance, which makes a `WhiteKernel` inside a model redundant (see the warning in {doc}`../how-to/build-kernels`).

## Why sums and products work

If $k_1$ and $k_2$ are kernels, so are $k_1 + k_2$ and $k_1 k_2$. Each has a meaning:

- **Sum**: the function is the sum of two independent random functions, $f = f_1 + f_2$. `RbfKernel(long) + RbfKernel(short)` models a slow trend plus faster variation; `LinearKernel + RbfKernel` a trend plus smooth wiggle.
- **Product**: the covariance requires both factors to be high, like a logical AND. `ConstantKernel * RbfKernel` rescales the signal variance. `PeriodicKernel * RbfKernel` gives a **locally periodic** function: the pattern repeats, but its shape drifts over distances of the RBF length scale. With different kernels on different features it models interactions.

Parameters stay independent: the parameter vector of a composite is the concatenation of its leaves' parameters, so the optimizer sees one flat vector. The how-to guide shows how that vector is ordered.

## Parameters live in log space

Every hyperparameter is a positive quantity: a length scale, a variance, a period. cxxgpr stores and exchanges them as $\vartheta = \log\theta$. Three reasons:

1. An unconstrained optimizer can take any step without leaving the valid region: $e^\vartheta > 0$ always.
2. Scale differences matter multiplicatively. Going from $\ell = 0.1$ to $0.2$ is as large a change as from $10$ to $20$; in log space both are the same step $\log 2$.
3. The derivative of a kernel with respect to $\log\theta$ is often simpler. For $\theta = \ell$ in the squared exponential, $\partial k/\partial\log\ell = k\,r^2/\ell^2$.

Each parameter additionally has an **open interval** (default $(10^{-5}, 10^5)$) in user units. The interval is a guard: it keeps an optimizer from collapsing a length scale to a numerical zero or running it off to infinity, where the Gram matrix degenerates. A value on or outside the boundary is rejected, and `set_params` is atomic, so an invalid step never leaves a half-updated kernel behind. For derivatives, the library always reports $\partial/\partial\log\theta$, never $\partial/\partial\theta$.

## From description to evaluation

cxxgpr separates *describing* a kernel from *evaluating* it.

```{mermaid}
flowchart LR
    L["Leaves<br/>RbfKernel, MaternKernel, ..."] --> S["KernelSpec<br/>immutable tree, double"]
    C["CustomKernel"] --> S
    S -->|"compile for double"| D["CompiledKernel (double)"]
    S -->|"compile for float"| F["CompiledKernel (float)"]
    D --> G["Gram, cross-covariance,<br/>gradient, Hessian"]
    F --> G
    DC["DistanceCache"] -.-> G
    NB["NestedBuffers"] -.-> G
```

**`KernelSpec` is the declaration.** It is a tree of value types whose parameters are always `double`. Copies share immutable subtrees, so composing and passing kernels is cheap, and `set_params` rebuilds a validated copy and swaps it in only if every leaf accepts. Because it holds no scratch memory and no precision choice, it is the type that is saved and loaded.

**`CompiledKernel<T>` is the execution plan.** Compiling for `T = double` or `float` flattens nested sums and products into one level (associativity), caches the length scales that ARD leaves need, works out how many scratch matrices a recursive evaluation will want, and binds each parameter to its leaf. The result evaluates matrices without heap allocation when you supply the buffers: `NestedBuffers` is one contiguous block with a matrix per recursion level. A `DistanceCache` supplies precomputed squared distances so that an optimizer, which changes only hyperparameters between steps, does not rescan the coordinates.

This split also explains the precision story: a `KernelSpec` is compiled once per precision, while the hyperparameter vector itself stays in double.

## The cost of the exponential

Almost all of the evaluation time of an RBF or Matern Gram matrix is spent on $e^{-x}$ for $n^2/2$ entries, and again for the gradient. `std::exp` is correct to the last bit or two, but it is a call into the C library. The `FastApprox` backend replaces it with an inline sequence that a compiler can vectorize.

### How the approximation works

Write $x = n\ln 2 + g$ with integer $n$ and $|g| \le \tfrac12\ln 2 \approx 0.3466$. Then $e^x = 2^n e^g$. The factor $2^n$ is built by writing the exponent bits directly. The remaining $e^g$ is evaluated with the degree-7 Taylor polynomial

$$
e^g \approx \sum_{i=0}^{7} \frac{g^i}{i!}.
$$

The truncation error of the polynomial is the first omitted term, $g^8/8!$. At the edge of the reduced range this is $0.3466^8/40320 \approx 5\times10^{-9}$, and relative to $e^g$ it reaches about $7\times10^{-9}$. That is what the example measures:

```{code-block} text
max relative error of FastApprox on [-100, 0]: double 7.02e-09, float 1.19e-07
```

In `float` the error is dominated by the 24-bit mantissa, so the approximation is practically exact; in `double` it throws away about seven of sixteen digits. Special values behave: underflow gives 0, overflow gives $+\infty$, and NaN stays NaN.

### What it does to a model

The Gram matrix entries change by at most about $10^{-9}$ for kernels with values of order one. That is far above the rounding of `std::exp` but far below the observation noise of any realistic data set, so predictions usually move in the seventh digit or later. Two consequences are worth knowing:

- **Derivatives stay consistent.** Under `FastApprox` the library differentiates the *polynomial*, not $e^x$: `eval_kernel_exp_jet` returns an `ExpJet` whose `d1` and `d2` are the polynomial's derivatives. The analytic gradient therefore matches a finite difference of the approximate Gram matrix (error about $5\times10^{-11}$ in the example) instead of differing from it at the $10^{-8}$ level, which keeps line searches and gradient checks coherent.
- **Results differ from `Accurate`.** The gradient itself differs by about $6\times10^{-8}$ from the accurate one. Do not mix backends when you compare a run against stored reference values, and do not use `FastApprox` when a test needs agreement with another implementation to better than $10^{-8}$.

`Accurate` is the default and the right choice unless profiling shows the exponential dominates; the speedup depends on the compiler, the CPU and the matrix size and is not claimed here. Only the exponentials inside the built-in kernel leaves are affected. Parameter transforms $e^\vartheta$ and custom kernels always use `std::exp`.

## When distance is not enough

A stationary leaf needs only the squared distance, so you may supply your own distances (geodesic, graph, edit) and evaluate the kernel on those, or train a model on them; see {doc}`../how-to/use-supplied-distances` and {doc}`../how-to/fit-on-supplied-distances`. A leaf that you write yourself as a function of squared distance plugs into the same composition and optimization machinery; see {doc}`../how-to/write-custom-kernel`. In both cases the kernel must remain positive semi-definite for your metric, which cxxgpr cannot check for you.
