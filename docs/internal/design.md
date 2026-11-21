# cxxgpr design specification (Exact GPR)

This specification is the canonical source of the requirements (`REQ-01` to `REQ-12`) and design decisions (`DES-01` to `DES-20`) for the **Exact GPR** model implemented by the C++23 static library `cxxgpr`. The details of the persistence format are defined in [docs/internal/persist-format.md](persist-format.md), and the dependency boundaries between modules in [docs/internal/architecture.md](architecture.md).

---

## 1. Product requirements (REQ-01 to REQ-12)

| ID | Title | Requirement |
|---|---|---|
| REQ-01 | Exact GPR scope | Port `Gpr` / `FittedGpr`, all 10 built-in kernel leaves, `+`/`*` composition, `Custom` leaves, supplied squared distances `DistanceKernel`, preprocessing transforms, `Double`/`Single`/`MixedPrecision`, `KernelExp`, the 5 optimizers, and `persist` completely. `Sgpr`/`Svgp`/`OnlineGpr` and inducing-point coordinate derivatives are out of scope |
| REQ-02 | C++23 non-throwing, value-returning API | Do not throw exceptions across the API boundary; return `[[nodiscard]] std::expected<T, GprError>`. `Gpr::fit(...)` and `Gpr::factor(...)` are `const` members and, on failure, leave `*this` and output arguments unchanged |
| REQ-03 | Column-major input and input validation order | Matrix input is column-major `std::span<const double>` (`x`, `n_rows`, `n_cols`). Empty input, dimension overflow, array length mismatch, dimension mismatch, non-finite values (`NaN`/`±∞`), and insufficient data are detected in a deterministic order and returned as `GprError` |
| REQ-04 | Kernel system and composition | Provide the 10 built-in kernel leaves (Isotropic/ARD variants of RBF, Matérn, and RQ; Periodic; Constant; Linear; White), `+`/`*` flattening composition, `Custom` leaves (`KernelTerm<T>`), and depth-first left-to-right $\theta$ binding |
| REQ-05 | Supplied squared distances | Provide `ScalarDistance` / `ArdDistance` / `DistanceKernel<DistanceOnly | WithPoints>` with `tidy(rel_tol)`, strict validation, and slot-misuse detection. Must be neither slower than nor allocate more than a `Cached` coordinate model under the same conditions. Models trained on supplied distances (`DistanceGpr` / `FittedDistanceGpr`, DES-21) are part of this requirement |
| REQ-06 | Separation of noise and Jitter | Separate the log observation noise $\theta_{\text{noise}} = \log(\sigma_n^2)$ from the numerical-stabilization `JitterPolicy` (`Fixed`/`Adaptive`). The first attempt always uses $j=0$, retries happen only on failure, and the adopted `factor_jitter` is retained and saved |
| REQ-07 | Three precision policies and IR | Provide `DoublePrecision`, `SinglePrecision`, and `MixedPrecision<PromoteStorage | ReevaluateKernel>`. `MixedPrecision` refines $\alpha$ by `f64`-residual iterative refinement on an `f32` Cholesky factor (re-solving in `f64` on non-convergence) |
| REQ-08 | MLL, derivatives, LOO | Implement the negative marginal log likelihood (MLL), the single-pass analytic gradient, the analytic Hessian, and Leave-One-Out inference of GPML §5.4.2 according to the exact formulas |
| REQ-09 | In-house optimizers | Without relying on external solvers, implement `Lbfgs` (Moré–Thuente), `NelderMead`, `TrustRegion` (Steihaug–Toint CG), `FastSimulatedAnnealing` (Cauchy/Metropolis/Ingber + incremental evaluation), `Fixed`, the logit interval transform, and deterministic restarts |
| REQ-10 | Preprocessing pipeline | Provide `Identity`, `Standardize`, `MinMax`, `Pipeline`, `ColumnwiseInput`, `TargetPipeline`, and custom transforms, and map the predictive mean, variance, and covariance back to the original scale with a closed-form affine inverse transform |
| REQ-11 | Zero allocation, deterministic parallelism, boundaries | The hot paths (MLL+grad, and `predict_into` after warm-up) perform zero new allocations. OpenMP parallelism keeps bitwise agreement independent of the thread count. Eigen, `nlohmann/json`, and OpenMP are not exposed in `include/cxxgpr/` |
| REQ-12 | Persistence, random numbers, independent verification | Provide `config.json` + `model.safetensors` saving, `mmap` loading, and `PersistRegistry`; perform deterministic sampling with `Xoshiro256++`; and compare numerically against analytic solutions and frozen reference-implementation output |

---

## 2. Architecture and type contracts (DES-01 to DES-02)

### DES-01 Type design, construction contracts, and the public boundary

The public namespace is `cxxgpr`, with `cxxgpr::kernel`, `cxxgpr::transform`, and `cxxgpr::persist` beneath it (detailed types live in `cxxgpr::detail`). Public headers are placed in `include/cxxgpr/<module>/` and never `#include` `<Eigen/...>`, `<nlohmann/...>`, or `<omp.h>` ([architecture.md](architecture.md)).

1. **Separation of construction and validation**:
   - Types whose validation cannot fail (default-valued configurations such as `Gpr`, `Lbfgs`, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing`, `Fixed`, `IdentityInput`, `StandardizeInput`, `MinMaxInput`, and non-empty default settings) are built with ordinary constructors.
   - Types that can fail on input ranges or constraints (`RbfKernel`, `RbfArdKernel`, `MaternKernel`, `MaternArdKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, `RationalQuadraticArdKernel`, `ConstantKernel`, `LinearKernel`, `WhiteKernel`, `GaussianLikelihood`, `Interval`, `JitterPolicy::fixed/adaptive`, `MinMaxInput::create(lo, hi)`, etc.) make the constructor private and expose only a `[[nodiscard]] static std::expected<T, GprError> create(...)` factory.
2. **Separate types for unfitted and fitted**:
   - The unfitted trainer `Gpr<O, P, K>` has no `predict`, `loo_predict`, or `save`.
   - Both `fit(...)` and `factor(...)` (for `O = Fixed`, or the fixed-factorization factory) are **`const` member functions** returning `[[nodiscard]] std::expected<FittedGpr<O, P, K>, GprError>`. On failure, `*this` is never modified.
   - Preprocessing likewise separates types into `UnfittedInputTransform` / `FittedInputTransform` and `UnfittedTargetTransform` / `FittedTargetTransform`. An unfitted `apply` / `transform` cannot be called, by type.
3. **Extensible and closed enumerations**:
   - Code that exhaustively handles the error classification of `GprError` or extension tags that may grow in the future is expected to have a default branch for unknown values, whereas `Triangle` (`Lower`, `Upper`, `Full`), `MaternNu` (`Nu1_2`, `Nu3_2`, `Nu5_2`), `VarianceKind` (`Latent`, `Observation`), and `CholeskyBuffer` (`Retain`, `Reuse`) are closed enumerations.

### DES-02 Error type `GprError` and the input validation contract

There is no `NotFitted`. All failures of numerical computation, input, and persistence are returned as the value type `GprError`.

| Error code (`GprErrorCode`) | Main conditions and attached information |
|---|---|
| `EmptyInput` | $n = 0$ or $d = 0$ (except for `DistanceOnly`), or an empty length-scale array |
| `SizeOverflow` | A size product such as `n_rows * n_cols` or $n(n+1)/2$ exceeds the `std::size_t` limit |
| `LengthMismatch` | `x.size() != n_rows * n_cols`, `y.size() != n_rows`, `out.size() != p`, etc. |
| `DimensionMismatch` | Mismatch of the number of features $d$ in prediction, query, or transform application |
| `IndexOutOfRange` | Parameter index `i >= p`, or a coordinate not designated changed in an incremental update |
| `InsufficientData` | $n$ is below the minimum number of points of the kernel / model (in principle $n \ge 1$ for Exact GPR, including special requirements such as LOO) |
| `NonFiniteInput` | `NaN` / `±∞` in `x`, `y`, the parameters $\theta$, or configured floating-point values |
| `NonFiniteKernelValue` | A kernel evaluation, or a quantity derived from it (the marginal likelihood, a latent predictive variance before its clamp at $0$, an inverse-scaled moment), became non-finite |
| `InvalidHyperparameter` | Domain violation of $\ell, \sigma^2, \alpha, p$, etc. (non-positive, non-finite, or outside the interval) |
| `InvalidNoiseVariance` | $\sigma_n^2 \le 0$ or non-finite in `GaussianLikelihood` |
| `InvalidInterval` | `Interval` lower bound $\ge$ upper bound, non-positive (for positive intervals), or non-finite |
| `InvalidConfig` | Invalid optimizer settings, Jitter settings, `tidy` tolerance, or `ColumnwiseInput` configuration |
| `InvalidDistance` | Supplied squared distance that is non-finite, negative, has a non-zero diagonal, or violates symmetry (`slot`, `dim`, `pair`, `reason`) |
| `DistanceSlot` | Missing, duplicate, or unknown slot (`SlotErrorKind`, `slot`) |
| `UnsupportedKernelOperation` | Passing a supply meant for another kernel, performing a coordinate operation on a `DistanceOnly` kernel, etc. |
| `CholeskyFailed` | Cholesky factorization failure (the last `jitter` tried, `matrix_size`, `CholeskyStage::{Fit, Predict}`) |
| `NonPositiveDefiniteMatrix` | Failure for a matrix that is not positive definite |
| `OptimizationNotConverged` | A user-defined optimizer gave up (elapsed `iterations`). The built-in optimizers never return it: non-convergence is reported as success with `OptimizerResult::converged == false` (DES-19) |
| `WorkspaceTooSmall` | Insufficient external or pre-allocated buffer |
| `PersistFailed` | Persistence errors: I/O, JSON configuration, tensor validation, wrong model, unregistered ID, etc. (`PersistErrorKind`) |
| `UnsupportedPersistVersion` | `format_version != 1` (`found`, `supported`) |

**Principle of validation order**: when several validation violations hold at the same time, the first `GprError` that applies in the following strict total order is returned.
`EmptyInput` ($n = 0$, $d = 0$ for coordinate models, empty slices and arrays) → `SizeOverflow` (`std::size_t` multiplication overflow in `n_rows * n_cols`, $n(n+1)/2$, etc.) → `LengthMismatch` (`x.size() != n_rows * n_cols`, `y.size() != n_rows`, output slice length mismatch) → `DimensionMismatch` (mismatch between the feature dimension $d$ expected by the fitted model, preprocessing transform, or ARD kernel and the number of input columns `n_cols`) → `NonFiniteInput` (`NaN` / `±∞` in `x`, `y`, $\theta$, etc.) → `InsufficientData` (fewer samples than the processing requires). IEEE 754 subnormal numbers are processed as valid finite values, and MXCSR / FPCR are not manipulated.

---

## 3. Linear algebra, noise, Jitter, and precision (DES-03 to DES-05)

### DES-03 Linear algebra backend (Eigen + OpenMP)

Internal linear algebra uses Eigen 5.0.1 pinned by vcpkg (`EIGEN_MPL2_ONLY` + `EIGEN_DONT_PARALLELIZE`).

- **Storage order**: all dense matrices are column-major (`Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>`). The input data `x` ($n \times d$), the kernel matrices, and the factor $L$ are all held column-major.
- **Cholesky factorization**: only `uplo = Lower` is built and referenced, and the factorization is performed with `Eigen::LLT<..., Eigen::Lower>` on pre-allocated buffers, effectively in place (using `matrixL()`).
- **Allocation detection**: `EIGEN_RUNTIME_NO_MALLOC` is used; inside the test audit scope `AllocationAudit`, `Eigen::internal::set_is_malloc_allowed(false)` is combined with a standard heap-allocation hook to enforce zero hot-path allocation mechanically.
- **Floating-point and SIMD policy**: MSVC `/fp:precise` and GCC/Clang `-fno-fast-math` are imposed, allowing optimization by SIMD auto-vectorization and FMA contraction while preserving the correctness of IEEE 754 `NaN`/`Inf` checks. On x86-64, `CXXGPR_X86_ISA` (default `AVX2`; `AVX512` / `AVX2` / `SSE2` selectable) optimizes Eigen and the internal SIMD kernels across three tiers: AVX-512 / AVX2+FMA / SSE2.

### DES-04 Observation noise `GaussianLikelihood` and `JitterPolicy`

The observation noise $\sigma_n^2$ and the numerical regularization jitter $j$ are cleanly separated.

- **`GaussianLikelihood`**: a concrete class holding the log-space $\theta_{\text{noise}} = \log(\sigma_n^2)$ over an `Interval` (default `(1e-5, 1e5)`). Since $\sigma_n^2 = \exp(\theta_{\text{noise}})$, $\partial K / \partial \theta_{\text{noise}} = \sigma_n^2 I$.
- **`JitterPolicy`**: takes `Fixed` (default: $j = 0.0$; on failure, one retry, or immediate termination when $j=0$) and `Adaptive(initial, multiplier, max_retries, max_jitter)`.
- **Retry algorithm**:
  1. Always try the lower-triangular Cholesky factorization of $A = K + \sigma_n^2 I$ with $j = 0$ first.
  2. On success, set `factor_jitter = 0.0`.
  3. Only on failure, follow the `JitterPolicy`: `Fixed(j)` retries exactly once when $j > 0$, and `Adaptive` retries the factorization of $(A + jI)$ starting from `initial`, multiplying by `multiplier` each time, within `max_retries` attempts and while $j \le \text{max\_jitter}$.
  4. The final successful $j$ is recorded and persisted as `factor_jitter`, and the subsequent solve for $\alpha$ and the `MixedPrecision` iterative refinement are consistently performed on the same system $(A + jI)$ (no iterative refinement back to $A$ is performed for a jittered factor).

### DES-05 Precision policies and mixed-precision iterative refinement (IR)

```cpp
struct PromoteStorage {};    // residual r = y_f64 - cast_f64(A_f32) * alpha
struct ReevaluateKernel {};  // residual r = y_f64 - kernel_f64(X) * alpha (blocked O(n) buffer)

struct DoublePrecision { using Storage = double; using Refine = double; };
struct SinglePrecision { using Storage = float;  using Refine = float;  };
template <typename ResidualFormula = PromoteStorage>
struct MixedPrecision  { using Storage = float;  using Refine = double; using Residual = ResidualFormula; };
```

- **`DoublePrecision`**: storage, arithmetic, and inference are all `double` (the default).
- **`SinglePrecision`**: storage, arithmetic, and inference are all `float`. No iterative refinement.
- **`MixedPrecision<R>`**: `K + σn²I + jI` is factorized and stored in `float` ($u_f \approx 1.19 \times 10^{-7}$), and only the prediction weights $\alpha$ are iteratively refined with `double` residuals. The MLL and analytic gradient during the training loop use the `float` factor as is.
- **Notation**: $B = A + (\sigma_n^2 + j)I$ (the training-system matrix including the adopted jitter $j \ge 0$), $w_k = \alpha^{(k)} \in \mathbb{R}^n$ (the approximate weights at iteration $k$), $b = y \in \mathbb{R}^n$ (the transformed target vector), $r_k = b - B w_k$ (the residual vector evaluated in `double`). $\|\cdot\|_\infty$ is the maximum-absolute-value norm for vectors ($\|v\|_\infty = \max_i |v_i|$) and the maximum absolute row-sum norm for matrices ($\|B\|_\infty = \max_i \sum_j |B_{ij}|$). $\varepsilon_{64} = 2^{-52} \approx 2.220446049250313 \times 10^{-16}$ (`std::numeric_limits<double>::epsilon()`), and the stopping threshold is $\text{tol} = 10 \cdot n \cdot \varepsilon_{64}$.
- **Iterative-refinement parameters and the stopping / fallback contract** (internal constants fixed):
  - Maximum number of correction iterations: `10`
  - Convergence condition: the denominator $\|B\|_\infty \|w_k\|_\infty + \|b\|_\infty > 0$ and $\|r_k\|_\infty / (\|B\|_\infty \|w_k\|_\infty + \|b\|_\infty) < 10 \cdot n \cdot \varepsilon_{64}$
  - Stagnation cutoff: the residual-norm ratio $\|r_k\|_\infty / \|r_{k-1}\|_\infty > 0.9$ twice in a row
  - Post-convergence verification for `PromoteStorage`: for a solution $w$ that satisfied the convergence test under `PromoteStorage`, the system $B_{\text{f64}} = K_{\text{f64}}(X) + (\sigma_n^2 + j)I$ from the true `f64` kernel matrix is evaluated in column blocks (block width 32; the whole $n \times n$ `f64` matrix is not held) to compute the true residual $r_{\text{f64}} = b - B_{\text{f64}} w$ and the row-sum norm $\|B_{\text{f64}}\|_\infty$ once, and it is checked whether the same relative-residual formula as the iteration stopping condition
    $$\|r_{\text{f64}}\|_\infty / \left(\|B_{\text{f64}}\|_\infty \|w\|_\infty + \|b\|_\infty\right) < 10 \cdot n \cdot \varepsilon_{64}$$
    (with denominator $> 0$) holds. If it does not hold (for example $\kappa(A) u_{32}$ too large), or if IR did not converge within the maximum iterations or by the stagnation test, **no error is returned; the same system $(A + (\sigma_n^2 + j)I)$ is solved again with an `f64` Cholesky factorization**. Jitter is not increased when IR fails to converge.

---

## 4. Kernels, distance caches, and supplied squared distances (DES-06 to DES-12, DES-21)

### DES-06 `KernelSpec` (declaration layer) and `CompiledKernel<T>` (execution layer)

`KernelSpec` holds the parameters and the tree structure in `double`, independent of precision, and is composed with the operators `+` (sum) and `*` (product). `compile<T>()` flattens `Sum` and `Product` according to associativity and compiles to `CompiledKernel<T>` (`double` or `float`) for static dispatch at run time.

- **Parameter binding (`ParameterBinding`)**: the optimization vector $\theta$ of the whole model is the concatenation `[\text{kernel\_params} \mid \theta_{\text{noise}}]`, and the kernel part is assigned flatly in depth-first, left-to-right leaf order and, within a leaf, in local index order.
- **Atomicity of `set_params`**: if an out-of-range or non-finite value is detected in a partial leaf, the state of all leaves, including those already modified, is kept as before the call (unchanged on failure).
- **`CoordMode`**: the whole tree is inspected at compile time and classified as `Dist` (isotropic distance leaves only), `Points` (contains ARD/Linear coordinate leaves), `Either` (Constant/White only), or `Mixed` (a mix of distance leaves and coordinate leaves).
- **`nested_depth` and scratch estimation**: the number of intermediate `n×n` (and, for prediction, `n×m`) buffers required is precomputed from the tree structure and borrowed from the workspace, so that dynamic allocation during the tree traversal is zero.

### DES-07 Mathematical definitions and intervals of all 10 built-in kernel leaves

Every positive parameter $v > 0$ uses the logarithmic representation $\theta = \log(v)$ as the optimization space, and the default interval is `Interval::DEFAULT_POSITIVE = (1e-5, 1e5)`. For isotropic kernels, write the Euclidean distance as $r = \|x - x'\|_2$ and the squared distance as $d^2 = \|x - x'\|_2^2$; for ARD, write the scaled distance as $s^2 = \sum_{k=1}^d (x_k - x'_k)^2 / \ell_k^2$ ($r_{\text{ard}} = \sqrt{s^2}$).

| Kernel leaf class | Formula $k(x, x')$ | Optimization parameters $\theta$ |
|---|---|---|
| `RbfKernel` | $\exp\!\left(-\frac{d^2}{2\ell^2}\right)$ | $[\log \ell]$ |
| `RbfArdKernel` | $\exp\!\left(-\frac{1}{2}\sum_k \frac{(x_k - x'_k)^2}{\ell_k^2}\right)$ | $[\log \ell_0, \dots, \log \ell_{d-1}]$ |
| `MaternKernel` ($\nu = 1/2$) | $\exp\!\left(-\frac{r}{\ell}\right)$ | $[\log \ell]$ |
| `MaternKernel` ($\nu = 3/2$) | $\left(1 + \frac{\sqrt{3}\,r}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right)$ | $[\log \ell]$ |
| `MaternKernel` ($\nu = 5/2$) | $\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5 r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right)$ | $[\log \ell]$ |
| `MaternArdKernel` | The same $1/2, 3/2, 5/2$ radial functions as above, applied to $u = \sqrt{s^2}$ | $[\log \ell_0, \dots, \log \ell_{d-1}]$ |
| `PeriodicKernel` | $\exp\!\left(-\frac{2\sin^2(\pi r / p)}{\ell^2}\right)$ | $[\log \ell, \log p]$ |
| `RationalQuadraticKernel` | $\left(1 + \frac{d^2}{2\alpha\ell^2}\right)^{-\alpha}$ | $[\log \ell, \log \alpha]$ |
| `RationalQuadraticArdKernel` | $\left(1 + \frac{s^2}{2\alpha}\right)^{-\alpha}$ | $[\log \ell_0, \dots, \log \ell_{d-1}, \log \alpha]$ |
| `ConstantKernel` | $c$ | $[\log c]$ |
| `LinearKernel` | $\sigma^2 x^\top x'$ | $[\log \sigma^2]$ |
| `WhiteKernel` | $\sigma_w^2 \delta_{x, x'}$ (non-zero only on the training Gram diagonal; cross-covariance 0) | $[\log \sigma_w^2]$ |

The first derivatives $\partial K / \partial \theta_i$ and the second derivatives $\partial^2 K / \partial \theta_i \partial \theta_j$ (`hess` for squared-distance input and `hess_points` for coordinate leaves) are implemented analytically for each leaf. The first and second derivatives of `Matern(Ard)` at $r=0$ (the diagonal and duplicate points) take the continuous limit value $0$ and never divide by zero.

### DES-08 User-defined kernel extension `KernelTerm<T>`

A user-defined custom leaf is defined as a leaf interface evaluated from a squared-distance matrix `dist` (or a distance block), and is accepted through the virtual base classes `KernelTerm<float>` and `KernelTerm<double>` (via a common class template or a type-erasing wrapper). Even when used in a coordinate model, the `cxxgpr` side computes the squared-distance matrix `dist` from the coordinates `x` (or supplies it from the distance cache) before calling `apply`, `apply_cross`, `grad`, and `hess`. To keep Eigen types out of public headers, the distance and output matrices are passed as the lightweight views `MatrixRef<T>` / `MatrixMut<T>` (`data()`, `rows()`, `cols()`, `col_stride()`).

```cpp
// Public concept / interface signature overview (no Eigen headers)
template <typename T>
class KernelTerm {
public:
    virtual ~KernelTerm() = default;
    virtual std::size_t num_params() const = 0;
    virtual std::expected<void, GprError> get_params(std::span<double> out) const = 0;
    virtual std::expected<void, GprError> set_params(std::span<const double> params) = 0;
    virtual std::expected<void, GprError> bounds_into(std::span<Interval> out) const = 0;
    virtual std::expected<void, GprError> apply(MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo) const = 0;
    virtual std::expected<void, GprError> apply_cross(MatrixRef<T> dist, MatrixMut<T> out) const = 0;
    virtual std::expected<void, GprError> fill_diag(std::span<T> out) const = 0;
    virtual std::expected<void, GprError> grad(MatrixRef<T> dist, MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo) const = 0;
    virtual std::expected<void, GprError> hess(MatrixRef<T> dist, MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo) const = 0;
    virtual std::string_view persist_id() const { return ""; }
    virtual std::expected<std::string, GprError> persist_state_json() const;
};
```

Built-in kernel leaves use `std::variant` / `enum` static dispatch on the hot path, and dynamic dispatch is limited to `Custom` leaves.

### DES-09 Distance cache policy (`DistanceCachePolicy`)

`DistanceCachePolicy::{Cached, Uncached}` is held as a run-time enumeration (default `Cached`).

- **`Cached`**: at the start of `fit`/`factor`, the squared Euclidean distances between training points (a dense $n \times n$ column-major matrix, for isotropic leaves) and the per-dimension raw squared differences $(\Delta x_m)^2$ (for ARD leaves; on a contiguous array of length $\text{dims} \cdot n(n+1)/2$, the blocks for dimensions $m \in [0, \text{dims})$ are laid out in order, and within each block the lower-triangular elements $(i \ge j)$ are stored at the column-major packed position $p = jn - j(j-1)/2 + (i - j)$, that is, at flat index `m * (n*(n+1)/2) + p`) are computed and held exactly once and reused in every iteration of the optimization. If the compiled kernel reads neither the distance nor the ARD cache (for example a standalone `Constant` / `Linear` / `White`), nothing is allocated.
- **`Uncached`**: these intermediate distance buffers are not allocated, and each evaluation computes directly from the coordinates `X`.

### DES-10 Supplied squared distances (`ScalarDistance` / `ArdDistance` / `DistanceKernel<C>`)

Kernels are evaluated from a table of squared distances computed by the caller (geodesic distance, graph distance, etc.) instead of coordinates.

- **Slots and type markers**:
  - `ScalarDistance` (one scalar $d^2$ per pair of points) and `ArdDistance(dims)` (one $d^2$ per pair and per dimension) are declared, and kernel leaves are bundled into a `DistanceKernel<DistanceOnly>` or a `DistanceKernel<WithPoints>` mixed with coordinate leaves.
  - Each slot has a unique slot ID in depth-first order (and a number within its kind); passing the slot supply of another kernel returns `GprError::UnsupportedKernelOperation`. Missing or duplicate slots are `DistanceSlot`.
- **Table validation and `tidy(rel_tol)`**:
  - The supplied `DistanceSource` is verified at training time, in a single pass ($O(n^2)$), to be $n \times n$, symmetric, non-negative, finite, and zero on the diagonal. By default, tiny negative values and mirror mismatches are not silently repaired and `InvalidDistance` is returned (recording the first failing `(row, col)`, dimension, and slot).
  - Only for a supply for which the caller has explicitly enabled `source.tidy(rel_tol)`, diagonal entries and negative values within `rel_tol` are repaired to $0$, mirror pairs $(i, j),(j, i)$ are repaired to their mean, and when the tolerance is exceeded `InvalidDistance` is returned for the worst pair.
  - When ingesting an ARD supply into `Cached` or internal storage, validation and lower-triangular packing are done together in a single pass, and no extra dense `d · n²` temporary copy is made. The `fill` interface also packs directly into the lower triangle through a column-block recursion buffer.
- **Performance guarantee**: relative to a coordinate `Cached` model with the same $n, d$, precision, and kernel, neither time nor allocation count may be higher.

### DES-11 Incremental update support (`LeafCache` and `USES_CHANGE_INDICES`)

When `OptimizerTraits<O>::uses_change_indices` (`true` for `FastSimulatedAnnealing`, `false` for others) holds and `CholeskyBuffer::Retain` is in effect, `GprObjective` keeps the previous $\theta$ and the Gram matrix of each leaf in a `LeafCache<S>`.

- When `value_at_changes(params, indices)` is called, the supplied `params` are first compared bitwise, component by component, with the previous `theta` using `std::bit_cast<std::uint64_t>`; if a component outside `indices` has changed, `GprError::IndexOutOfRange` is returned immediately (the same applies to duplicate or out-of-range indices).
- Only the kernel leaves affected by the changed indices have `apply` recomputed, and the sums and products of the tree are reconstructed. Since the Cholesky factorization is not low-rank, it is redone at $O(n^3)$ each time. Under `CholeskyBuffer::Reuse`, no partial update is done and everything is rebuilt.

### DES-12 Exponential evaluation backend `KernelExp`

`KernelExp::{Accurate, FastApprox}` is held in each model as a run-time choice (default `Accurate`).

- `Accurate`: uses `std::exp` (storage type `float`/`double`) as is.
- `FastApprox`: replaces the $\exp(-x)$ computation during kernel evaluation with range reduction $x = k \ln 2 + f$ and a degree-7 Taylor polynomial (coefficients $1/j!$) in the reduced argument $f$, evaluated in the storage type (it is not used for restoring length scales by `exp(θ)` or for operations inside `Custom` leaves).

---

### DES-21 Supplied-distance models (`DistanceGpr` / `FittedDistanceGpr`)

A model whose covariance is a `DistanceKernel<C>` evaluated on squared distances supplied by the caller. It has the same lifecycle as the coordinate model (DES-01, DES-02, DES-16): an unfitted trainer returns a fitted model by value, errors are `std::expected<T, GprError>`, and a failed call leaves the trainer and every previously fitted model unchanged.

- **Types**: `DistanceGpr<O = optimizer::Lbfgs, P = DoublePrecision, C = DistanceOnly>` (trainer) and `FittedDistanceGpr<O, P, C>` (fitted model), with `C` in `{DistanceOnly, WithPoints}`, `P` any of the four precision policies and `O` any of the five optimizers or a user optimizer. The trainer is built from a `DistanceKernel<C>` and an optimizer: `DistanceGpr(DistanceKernel<C> kernel, O optimizer = O{})`.
- **Builder methods** (each returns a new `DistanceGpr`, or `std::expected` when the argument can be invalid, with the same validation and errors as `Gpr`): `with_noise_variance`, `with_jitter`, `with_kernel_exp`, `with_fit_policy`, `with_cholesky_buffer`, `with_target_transform`, `with_optimizer`. `with_input_transform` exists only for `C = WithPoints` and transforms the coordinates; supplied distances are never transformed.
- **Training inputs**: `fit(train, y)` for `DistanceOnly` and `fit(train, x, y)` for `WithPoints`, where `train` is a `DistanceSource` bound to the slots of the kernel (square $n \times n$ tables), `x` is the column-major $n \times d$ coordinate matrix and `y` the targets. `factor(...)` takes the same arguments and factorizes at the current hyperparameters without optimizing. Raw-span overloads of `x` check `x.size() == n * d` like the coordinate model.
- **Validation order**: empty inputs, target length, finiteness of `y` (and of `x`), then `prepare_square` (slot coverage, shapes, finiteness, non-negativity, zero diagonal, symmetry; `tidy` only when the caller enabled it on the source, DES-10), then the numerics. The first failing check is reported, and the table is validated once per `fit`, not per objective evaluation.
- **Objective and derivatives**: the parameters are the kernel's log-space parameters followed by $\log \sigma_n^2$, with the intervals of `bounds_into` and of the likelihood. Value, gradient and Hessian are the exact formulas of DES-13 and DES-14 with $A_i = \partial A / \partial \theta_i$ and $A_{ij}$ supplied by `CompiledDistanceKernel::grad` and `::hess`; $\sigma_n^2$ and $j$ are separated as in DES-04 and DES-06. All five optimizers work, including `FastSimulatedAnnealing` (which rebuilds every leaf each time, because distance leaves are not cached incrementally) and `Fixed`.
- **Precision**: `DoublePrecision`, `SinglePrecision` and the two `MixedPrecision` residual formulas behave as in DES-05. The tables are converted to the storage scalar when the store is built; the mixed-precision residual is evaluated from the `double` store, so the supplied tables are held in `double` once.
- **Fitted-model API**: `predict`, `predict_with`, `predict_into`, `predict_with_into` (mean and diagonal variance, `VarianceKind`), `predict_covariance`, `sample` (seeded, deterministic) and `loo_predict`, plus `refit`, `into_trainer` and the getters of `FittedGpr`. A query is described by a rectangular `DistanceSource` of $n \times m$ tables (column $j$ holds the squared distances from the $n$ training points to query $j$), bound to the slots of the fitted kernel, and, for `WithPoints`, the $m \times d$ query coordinates. `predict_covariance` and `sample` additionally take a square $m \times m$ `DistanceSource` of query-to-query distances. The prior variance $k(x_*, x_*)$ is evaluated at zero distance (`fill_diag`) and needs only the coordinates of a `WithPoints` kernel. Validation of the query supply (`prepare_cross`) precedes any workspace growth, and a failed query leaves the model unchanged.
- **Workspace and allocation**: the query workspace grows monotonically and repeated `predict_into` calls of non-increasing size do not allocate after warm-up (DES-17). The supplied tables are owned by the fitted model; the caller's tables are not referenced after `fit` returns.
- **Independent reference**: feeding the Euclidean squared distances of a coordinate dataset to a `DistanceKernel` of the same leaves must reproduce the coordinate model, in MLL, gradient, Hessian, weights, predictions, covariance and LOO, within $10^{-10}$ in `double` and a documented float tolerance in `float`, for scalar and ARD slots, products and sums.
- **Persistence**: `save` and `save_with_factor` accept a `FittedDistanceGpr`, and `LoadedDistanceGpr<C>::load` restores it (FMT-05, FMT-07).

## 5. Exact GPR numerical algorithms, buffers, and inference (DES-13 to DES-17)

### DES-13 Single-pass algorithm for MLL and the analytic gradient (`keep_plan`)

The negative marginal log likelihood (the minimization target) and its gradient are defined by the following formulas ($A = K + \sigma_n^2 I = L L^\top$, $\alpha = A^{-1} y$, $W = \alpha \alpha^\top - A^{-1}$):

$$
\mathcal{L}(\theta) = \frac{1}{2} y^\top A^{-1} y + \frac{1}{2} \log |A| + \frac{n}{2} \log(2\pi), \qquad
\log |A| = 2 \sum_{i=0}^{n-1} \log L_{ii}
$$

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} = -\frac{1}{2} \alpha^\top \frac{\partial A}{\partial \theta_i} \alpha + \frac{1}{2} \operatorname{Tr}\!\left(A^{-1} \frac{\partial A}{\partial \theta_i}\right) = -\frac{1}{2} \langle W, \frac{\partial A}{\partial \theta_i} \rangle_F
$$

`value_and_gradient_into` carries out the following steps in a single pass (total cost $O(n^3 + p n^2)$):

1. Build $A = K + \sigma_n^2 I$ in the lower triangle of `k_matrix`. When a multi-factor product other than a constant factor ($A \times B$) benefits the gradient computation, intermediate factor Grams are kept in the reserved buffers according to `keep_plan`.
2. Perform the in-place lower-triangular Cholesky factorization on `k_matrix` (on failure, retry with $A + jI$ according to `JitterPolicy`) to obtain $L$.
3. Obtain $\log |A|$ from the diagonal and $\alpha$ by forward/back substitution.
4. Build `W = ααᵀ - A⁻¹` for the lower triangle only (`Retain` uses a dedicated `w_matrix`; `Reuse` overwrites `k_matrix` via `exp_buf`).
5. Traverse the tree once with `CompiledKernel::weighted_grads`: sums pass through as is, and for products the product of the other factors is multiplied into the weight, computing $\langle W_{\text{eff}}, \partial K/\partial \theta_i \rangle_F$ for each leaf. The gradient of `GaussianLikelihood` is obtained from the diagonal sum alone ($\sigma_n^2 \sum_k W_{kk}$). The sums of the Frobenius inner products are folded deterministically in column-index order after parallel evaluation.

### DES-14 Analytic Hessian (`HessianScratch`)

$$
H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}
= -\frac{1}{2} \langle W, \frac{\partial^2 A}{\partial \theta_i \partial \theta_j} \rangle_F
  - \frac{1}{2} \operatorname{Tr}\!\left(A^{-1} A_i A^{-1} A_j\right)
  + \alpha^\top A_i A^{-1} A_j \alpha
$$

Letting $A_i = \partial A / \partial \theta_i$, $S_i = L^{-1} A_i L^{-\top}$, and $v_i = L^{-1} A_i \alpha$, we obtain $\operatorname{Tr}(A^{-1} A_i A^{-1} A_j) = \langle S_i, S_j \rangle_F$ and $\alpha^\top A_i A^{-1} A_j \alpha = v_i^\top v_j$, and the analytic Hessian (column-major / symmetric $p \times p$) is computed in $O(p n^3 + p^2 n^2)$ overall. `HessianScratch` ($p$ matrices $S_i$ of size $n\times n$ and an $n \times p$ matrix $v$) is empty inside `WorkspaceCore` until the first call and is reused from the second call on without additional allocation.

### DES-15 `CholeskyBuffer::{Retain, Reuse}` and memory / speed preference

- **Speed preference (`with_prefer_speed()`, the default)**: `DistanceCachePolicy::Cached` + `CholeskyBuffer::Retain`. $L$ and $W$ are kept in separate buffers, avoiding unnecessary refactorization.
- **Memory preference (`with_prefer_memory()`)**: `DistanceCachePolicy::Uncached` + `CholeskyBuffer::Reuse`. There is no `w_matrix`; $W$ overwrites `k_matrix`, and $L$ is refactorized at the end of `fit` and at the end of a standalone `value_and_gradient_into` / `hessian_into` to restore consistency. On persistence, the model is always loaded as the equivalent of `Retain`.

### DES-16 Inference API, covariance, sampling, and LOO

A fitted model `FittedGpr<O, P, K>` holds `L`, `factor_alpha`, `alpha`, `X`, the kernel, the likelihood, and the preprocessing, and provides the following operations (the output scalar type is `P::Refine`):

- `predict(xs, n_rows, n_cols)` / `predict_with(xs, n_rows, n_cols, PredictOptions)`: mean $\mu_* = K_*^\top \alpha$ and variance $v_* = \max\!\left(0,\; \operatorname{diag}(K_{**}) - \|L^{-1} K_*\|_2^2\right)$ (adding $\sigma_n^2$ for `VarianceKind::Observation`, not for `Latent`). The clamp at $0$ only absorbs rounding: a latent variance that is `NaN` or $\pm\infty$ before the clamp (for example from a factor whose solve overflows) is not clamped and the prediction fails with `NonFiniteKernelValue`. Finally the mean and the variance (multiplied by $s^2$) are inverse-transformed by `TargetTransform` and returned.
- `predict_into(..., Prediction<Refine>& out)` / `predict_with_into(...)`: once a call has sized them, later calls whose query count $m$ is no larger than the largest earlier $m$ reuse the `QueryWorkspace` and the output `mean`/`variance` capacity, achieving **zero new allocations**. Query batch sizes may change freely between calls: a larger $m$ grows the buffers once, and a smaller $m$ gives the same values as on a model that never saw a larger batch.
- `predict_covariance` / `predict_covariance_with`: return the full posterior covariance matrix $\Sigma_* = K_{**} - (L^{-1} K_*)^\top (L^{-1} K_*)$ (column-major $m \times m$; $\sigma_n^2$ is added to the diagonal for `Observation`).
- `sample(xs, n_rows, n_cols, n_draws, seed)` / `sample_with`: compute the Cholesky factor of the symmetrized posterior covariance with a tiny numerical jitter, and generate $n_{\text{draws}}$ posterior samples (length $n_{\text{draws}} \cdot m$) from `Xoshiro256++` + Box–Muller standard normal random numbers.
- `loo_predict()` / `loo_predict_with()` (the closed form of GPML §5.4.2): using $Q_{ii} = (A^{-1})_{ii} = \|L^{-1} e_i\|_2^2$ and the training targets $\tilde{y}_i$ after applying `TargetTransform`, compute $\tilde{\mu}_{\text{LOO}, i}$ and $\tilde{\sigma}_{\text{LOO}, i}^2$ in the transformed space, and **then** apply the closed-form affine inverse transform of `TargetTransform` (inverse transform of the mean and multiplication of the variance by $s^2$) to return a `Prediction<T>` on the original scale:

$$
\tilde{\mu}_{\text{LOO}, i} = \tilde{y}_i - \frac{\alpha_i}{Q_{ii}}, \qquad
\tilde{\sigma}_{\text{LOO}, \text{obs}, i}^2 = \frac{1}{Q_{ii}}, \qquad
\tilde{\sigma}_{\text{LOO}, \text{lat}, i}^2 = \max\!\left(0,\; \frac{1}{Q_{ii}} - \sigma_n^2\right)
$$

### DES-17 Workspace structure and lifecycle

```mermaid
flowchart LR
  FITSTART["fit start: n, d fixed, bulk allocation"] --> DIST["DistanceCache first build (when Cached)"]
  DIST --> LOOP["optimization loop: MLL+grad reuse, zero allocation"]
  LOOP -->|"done / Reuse refactorization"| FITTED["FittedGpr: keeps L, α, X, QueryWorkspace (W/H discarded)"]
  FITTED --> PRED["predict_into: 1st call allocates Query, 2nd and later zero allocation"]
```

- `WorkspaceCore<P>`: `k_matrix`, `exp_buf`, `kernel_scratch`, `thread_scratch` (per-OpenMP-thread scratch), `rhs`, the `theta` backup, `nested`, `hessian`, and `factor_jitter`.
- `FitBuffers<P>`: `WorkspaceCore<P>` + `dist` (optional `DistCache`) + `w_matrix` (optional, under `Retain`). Released at the end of `fit` and not kept in `FittedGpr`.
- `QueryWorkspace<P>`: belongs to `FittedGpr` and holds the query transform, `k_star`, `k_ss`, and nested scratch.

---

## 6. Preprocessing, optimizers, random numbers, and parallelism (DES-18 to DES-20)

### DES-18 Preprocessing class hierarchy (`cxxgpr::transform`)

All preprocessing is applied in `double` (independent of the precision policy).

- **Input transforms (`UnfittedInputTransform` / `FittedInputTransform`)**: `IdentityInput` (the default), `StandardizeInput` (matching the convention for zero-variance columns), `MinMaxInput` (default range `[0, 1]`, `create(lo, hi)`), `Pipeline` (serial composition), `ColumnwiseInput` (a separate transform per feature, with the check `maps.size() == d`), and user-defined custom bases. All provide the inverse `inverse_apply`.
- **Target transforms (`UnfittedTargetTransform` / `FittedTargetTransform`)**: `IdentityTarget` (the default), `StandardizeTarget`, `MinMaxTarget`, `TargetPipeline`, and user-defined custom bases. The inverse transform of variances and covariances multiplies by the affine scale $s^2$.

### DES-19 In-house optimizers (`cxxgpr::optimizer`)

The objective abstraction has three tiers: `Objective` (`value`, `value_at_changes`, `fill_intervals`) $\subset$ `Differentiable` (`gradient_into`, `value_and_gradient_into`) $\subset$ `TwiceDifferentiable` (`hessian_into`).

- **Interval mapping (logit transform)**: `Lbfgs`, `NelderMead`, and `TrustRegion` logit-map the log-transformed $\theta \in (\log a, \log b)$, on the positive open interval $(a, b)$ of each parameter, to the unconstrained space $z \in \mathbb{R}$ (with Jacobian scaling $1$ at the midpoint), and pass `TrustRegion` the chain-rule-transformed Hessian as well.
- **Specification of each algorithm (defaults `max_iterations = 100`, `tolerance = sqrt(eps)`)**:
  - `Lbfgs`: the Hestenes/Nocedal two-loop recursion (default `history_size = 10`) + the **Moré–Thuente line search**. It has a shared-evaluation `EvalCache`.
    - **First step**: while no curvature pair is stored, the direction is the raw negative gradient, so the initial trial step is $\min(1, 1/\lVert g_z\rVert)$ in the unconstrained space. The reciprocal gradient norm follows the first step of Liu and Nocedal's L-BFGS (the 1990 Fortran `LBFGS.f` sets `STP1 = ONE/GNORM`; checked in a published copy of that source, not in the netlib original); the cap at $1$ is this implementation's choice. A unit step along a large gradient otherwise jumps to the saturated bounds and stops on a flat plateau.
    - **Value resolution in the line search**: let $\varepsilon_f = \sqrt{\varepsilon}\,(1 + \lvert f_0\rvert)$ at the start $f_0$ of the line search ($\varepsilon$ the machine epsilon). When the first-order decrease of a trial step $\alpha$ is at most $\varepsilon_f$ ($-\alpha\,\varphi'(0) \le \varepsilon_f$) and the computed value satisfies $\lvert f(\alpha) - f_0\rvert \le \varepsilon_f$, the value difference is rounding noise and the line search uses the trapezoidal value $f_0 + \tfrac{\alpha}{2}(\varphi'(0) + \varphi'(\alpha))$ in place of $f(\alpha)$ in its acceptance tests and interpolation. The iterate's value is still the computed $f(\alpha)$, so successive iterates may rise by at most $\varepsilon_f$.
  - `TrustRegion`: the analytic Hessian and the **Steihaug–Toint conjugate-gradient subproblem** (`initial_radius = 1.0`, `max_radius = 100.0`). Indefinite or singular Hessians and out-of-bounds trial steps are absorbed by shrinking the trust-region radius. When the predicted reduction is at most $\varepsilon_f = \sqrt{\varepsilon}\,(1 + \lvert f\rvert)$ and the actual change stays within $\varepsilon_f$, the reduction ratio is taken as $1$.
  - `NelderMead`: the standard reflection, expansion, contraction, and shrink simplex method (uses only `Objective`).
  - `FastSimulatedAnnealing`: searches the log-$\theta$ space directly (no logit), using Cauchy proposals, Metropolis acceptance, Ingber exponential cooling (defaults `initial_temperature = 1.0`, `cooling_rate = 3.0`, `seed = 0`, `BoundaryPolicy::Clamp | Reflect`), and incremental updates via `value_at_changes` when one coordinate changes.
  - `Fixed`: a marker dedicated to `factor()` of `Gpr<Fixed>` (no optimization iterations).
- **Stopping rule of `Lbfgs` and `TrustRegion`**: both decide convergence in the log-parameter space $\theta$ on the closed box $[\log a, \log b]$, never in the unconstrained space $z$, whose gradient $g_z = g_\theta \odot \partial\theta/\partial z$ vanishes wherever the logit transform saturates.
  - **Stationarity measure**: the projected gradient $\chi(\theta) = \lVert \theta - \Pi(\theta - g_\theta)\rVert_2$, where $\Pi$ clamps each component to $[\log a_i, \log b_i]$ and $g_\theta$ is the objective's own gradient in $\theta$ (it is not recovered from $g_z$). An interior component equals $g_{\theta,i}$; a component whose descent direction points out of the box is cut to the distance to that bound, so a minimizer on a bound with an outward gradient has $\chi \to 0$, while an inward gradient at a bound keeps its full size.
  - **Test**: an iterate is stationary when $\chi(\theta) \le \text{tolerance} \cdot \max(1, \lVert\theta\rVert_2)$. The loop stops there; the joint test (relative function change at most $\text{tolerance}^2$ together with the same stationarity test) only saves one iteration. Relation to a test in $z$: $\lvert g_{z,i}\rvert = \lvert g_{\theta,i}\rvert\,\partial\theta_i/\partial z_i \le \lvert g_{\theta,i}\rvert$, and for an interval whose log-midpoint is $0$ (such as the default $(10^{-5}, 10^5)$) $\lvert z_i\rvert \ge \lvert\theta_i\rvert$, with equality at the midpoint; so the test coincides with $\lVert g_z\rVert_2 \le \text{tolerance} \cdot \max(1, \lVert z\rVert_2)$ at the midpoint and is stricter elsewhere, by up to the factor $1/(\partial\theta/\partial z)$ that grows without limit as a parameter approaches a bound.
  - **Failed step**: a line search that finds no acceptable step (`Lbfgs`) or a trust-region radius that collapses below $10^{-15}$ (`TrustRegion`) ends the run. It counts as convergence only when $\chi(\theta) \le \sqrt{\varepsilon}\,(1 + \lvert f\rvert)$ at the iterate: along a direction of curvature $h$, $\chi$ can buy at most the decrease $\chi^2 / (2h)$, which with the rounding level $\varepsilon\,(1 + \lvert f\rvert)$ and $h = 1 + \lvert f\rvert$ stays below half the rounding level exactly at that floor. The floor depends only on $\varepsilon$ and the scale of $f$, never on `tolerance`.
  - **Returned point and `converged`**: when the last iterate passes the test (or the floor after a failed step), it is returned with `converged = true`. Otherwise the run returns the lowest-value point it evaluated (for `Lbfgs` this can be a line-search trial point), and `converged` states whether that returned point passes the stationarity test. Exhausting `max_iterations` therefore returns `converged = false` unless the returned point is stationary. `converged` always describes the returned parameters.
- **`with_restarts(n, seed)`**: in addition to the first search point, searches from $n$ additional points generated deterministically by `Xoshiro256++`, log-uniformly within the interval (logit-space aware), and returns the minimum value and the corresponding parameters.
- **Non-convergence is not an error**: when `Lbfgs`, `TrustRegion`, or `NelderMead` stops before its stopping test holds (the `max_iterations` limit of a start is reached, or no further step is accepted, such as a Moré–Thuente line search that finds no acceptable step), it returns success with the best point found, the completed `iterations`, and `converged == false`, and `fit` succeeds with that result. `FastSimulatedAnnealing` has no stopping test, runs its whole `max_iterations` schedule, and reports `converged == true`. `Fixed` reports `iterations == 0` and `converged == true`. With `with_restarts`, the result (including `iterations` and `converged`) is that of the best run. The built-in optimizers return `GprError` only for invalid configuration or input and for objective failures.
- **User-defined optimizers**: `Gpr<O>` accepts a copyable optimizer type `O` that is none of the built-in ones when it provides a `const` member `optimize(TwiceDifferentiable&, std::span<const double>)` returning `std::expected<OptimizerResult, GprError>` (an optional `static constexpr bool uses_change_indices` selects DES-11; it defaults to `false`). Such an optimizer may report giving up with `GprError::optimization_not_converged(iterations)` or any other `GprError`; `fit` returns that error unchanged (same code, message, and payload) and leaves `*this` unchanged (REQ-02).

### DES-20 Deterministic random numbers (`Xoshiro256++`) and thread-invariant OpenMP parallelism

- **`Xoshiro256++`**: every `seed` is expanded into the 256-bit state by SplitMix64, and `next_u64()`, a uniform $[0, 1)$ `double`, and Box–Muller standard normal random numbers are generated. The output sequence for the same seed matches exactly across OSes (Windows / Linux) and build configurations.
- **OpenMP parallel design**:
  - The lower-triangular column blocks of the Gram matrix are parallelized with OpenMP using a partition (`lower_block_start`) that equalizes the area assigned to each worker.
  - Sums such as `weighted_grads` compute per-worker, per-column partial sums and then **add them sequentially in ascending column-index order**, which guarantees **bitwise-identical** floating-point results regardless of the number of OpenMP threads (1 vs. any).
