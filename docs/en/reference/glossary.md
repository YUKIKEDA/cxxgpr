# Glossary

Definitions of the terms and symbols used throughout this documentation. Entries are alphabetical; each says where to read more.

## Notation

| Symbol | Meaning |
|---|---|
| $n$ | Number of training points |
| $d$ | Number of features (columns of $X$) |
| $m$ | Number of query points |
| $X$, $y$ | Training inputs ($n \times d$, column-major) and training targets (length $n$) |
| $K$ | Gram matrix of the kernel on the training inputs, $K_{ij} = k(x_i, x_j)$ |
| $\sigma_n^2$ | Observation noise variance |
| $j$ | Jitter added to the diagonal when a factorization fails |
| $A$ | The system matrix $K + \sigma_n^2 I$ (with jitter, $K + (\sigma_n^2 + j) I$) |
| $L$ | Lower Cholesky factor, $A = L L^\top$ |
| $\alpha$ | Prediction weights, $\alpha = A^{-1} y$ |
| $K_*$ | Cross-covariance between the training and the query points, $n \times m$ |
| $\theta$ | Vector of log-space hyperparameters: all kernel parameters followed by $\log\sigma_n^2$ |
| $p$ | Length of $\theta$ |
| $W$ | Gradient weight matrix $\alpha\alpha^\top - A^{-1}$ |
| $\varepsilon_{64}$, $u_{32}$ | Machine epsilon of `double` ($2^{-52}$) and unit roundoff of `float` ($\approx 6 \times 10^{-8}$) |
| $\kappa(A)$ | Condition number of $A$ |

## Terms

Accurate (exponential)
: `KernelExp::Accurate`: the standard-library `std::exp` inside kernel evaluation. The default. See {doc}`../how-to/optimize-performance`.

ARD (automatic relevance determination)
: A kernel with one length scale per feature (`RbfArdKernel`, `MaternArdKernel`, `RationalQuadraticArdKernel`). A large learned length scale means that the feature barely matters. See {doc}`../how-to/build-kernels`.

Batch
: The set of query points passed to one prediction call. The size of the batch decides the size of the `QueryWorkspace`.

Cholesky factorization
: The decomposition $A = L L^\top$ of a symmetric positive-definite matrix into a lower-triangular matrix and its transpose. It is the central operation of exact GP regression and costs $O(n^3)$. See {doc}`../explanation/numerical-stability`.

Column-major
: Storage order in which the elements of a column are adjacent in memory; element $(i, j)$ of an $n$-row matrix is at index $jn + i$. All matrices in cxxgpr are column-major. See {doc}`../explanation/design-principles`.

Compiled kernel
: `CompiledKernel<T>`: the execution form of a `KernelSpec`, flattened and specialized for `float` or `double`, used for all numerical work. See {doc}`../how-to/build-kernels`.

Condition number
: The ratio of the largest to the smallest eigenvalue of a symmetric positive-definite matrix. A large value means that small input errors produce large output errors, and that single precision loses accuracy.

Cross-covariance
: The matrix $K_*$ of kernel values between every training point and every query point.

Distance cache
: Squared distances between training points, computed once and reused in every optimizer iteration (`DistanceCachePolicy::Cached`). See {doc}`../how-to/optimize-performance`.

Exact GPR
: Gaussian process regression that factorizes the full $n \times n$ covariance matrix without approximation. This is the only model cxxgpr implements.

Factor
: Used as a noun for the Cholesky factor $L$. As a verb in the API (`Gpr::factor`): fit the model at the current hyperparameters without optimizing them.

FastApprox
: `KernelExp::FastApprox`: vectorized 1–2 ULP Estrin minimax exponential approximation replacing `std::exp` in the built-in kernels. Faster per call, accurate to $\le 2\text{ ULP}$ ($< 10^{-14}$ relative in `double`, $< 5 \times 10^{-7}$ in `float`).

Fallback (refinement)
: The re-solve of the system in double precision when mixed-precision iterative refinement does not converge or is rejected. See {doc}`../explanation/numerical-stability`.

Fitted model
: `FittedGpr`: a trained model that can predict, sample, compute leave-one-out predictions and be saved. It is a different type from the unfitted trainer `Gpr`.

Gram matrix
: The matrix $K$ of kernel values between all pairs of training points.

Hessian
: The matrix of second derivatives of the negative log marginal likelihood with respect to $\theta$, computed analytically by `hessian_into`. Used by `TrustRegion`.

Hyperparameter
: A parameter of the kernel or the noise (length scale, signal variance, period, noise variance). cxxgpr optimizes hyperparameters on a logarithmic scale. See {doc}`../how-to/tune-hyperparameters`.

Interval
: An open range `(lo, hi)` that a hyperparameter must stay inside. `Interval::default_positive()` is $(10^{-5}, 10^{5})$.

Iterative refinement
: Improving an approximate solution of a linear system by repeatedly computing the residual in higher precision and solving for a correction with a cheap factor. See {doc}`../explanation/numerical-stability`.

Jitter
: A small positive number added to the diagonal when the Cholesky factorization fails. It is a numerical device and is separate from the observation noise. See {doc}`../explanation/numerical-stability`.

Kernel
: A positive semi-definite function $k(x, x')$ that defines the prior covariance between function values. Also called covariance function. See {doc}`../explanation/kernels`.

Kernel leaf
: One of the ten built-in kernels (or a `CustomKernel`), the building block of a `KernelSpec` expression tree.

KernelSpec
: The declarative, immutable description of a kernel as a tree of leaves combined with `+` and `*`. It becomes a `CompiledKernel` before evaluation.

Latent variance
: The predictive variance of the noise-free function $f(x_*)$ (`VarianceKind::Latent`). The observation variance adds $\sigma_n^2$.

Leave-one-out (LOO)
: The prediction of each training target from all the others, available in closed form from $L$ and $\alpha$ (`loo_predict`).

Length scale
: The distance over which function values stay correlated; the main hyperparameter of RBF, Matern and related kernels.

Log marginal likelihood
: The log of the probability of the targets given the inputs and hyperparameters. cxxgpr minimizes its negative (the objective of the optimizers). See {doc}`../explanation/gaussian-processes`.

Mixed precision
: `MixedPrecision<R>`: factor in `float`, store `float` matrices, refine the weights in `double`. The residual formula `R` is `PromoteStorage` or `ReevaluateKernel`.

NestedBuffers
: A block of preallocated matrices used by the recursive evaluation of a kernel tree so that evaluation does not allocate.

Noise variance
: The variance $\sigma_n^2$ of the Gaussian observation noise, held by `GaussianLikelihood` and optimized like a kernel parameter.

Observation variance
: The predictive variance of a new noisy observation: latent variance plus $\sigma_n^2$ (`VarianceKind::Observation`, the default).

OpenMP
: The directive-based threading standard used by the library for parallel loops. Controlled with `OMP_NUM_THREADS`. See {doc}`../how-to/optimize-performance`.

Optimizer
: The algorithm that minimizes the negative log marginal likelihood: `Lbfgs`, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing` or `Fixed`. See {doc}`../how-to/choose-optimizer`.

Posterior
: The distribution of the function values at new points after conditioning on the training data; a Gaussian with the predictive mean and covariance.

Precision policy
: A type that selects the scalar types of storage and refinement: `DoublePrecision`, `SinglePrecision`, `MixedPrecision<PromoteStorage>` or `MixedPrecision<ReevaluateKernel>`.

Prediction weights
: The vector $\alpha = A^{-1}y$ that turns cross-covariances into predictive means.

Query point
: A point at which the model is evaluated (also test point).

Refine type
: The scalar type of weights and predictions in a precision policy (`P::Refine`): `double` for `DoublePrecision` and `MixedPrecision`, `float` for `SinglePrecision`.

Residual
: For a linear system $Bw = b$ and an iterate $w$, the vector $r = b - Bw$.

Stagnation
: A refinement stops making progress: the residual norm shrinks by less than a factor of 0.9 in two consecutive steps.

Standardization
: A transform that subtracts the mean and divides by the standard deviation (`StandardizeInput`, `StandardizeTarget`). See {doc}`../how-to/transform-data`.

Storage type
: The scalar type of matrices and Cholesky factors in a precision policy (`P::Storage`): `double` or `float`.

Supplied distances
: Squared distances given by the caller instead of computed from coordinates; used with `DistanceKernel`. See {doc}`../how-to/use-supplied-distances`.

Training buffers
: The temporary workspace (`FitBuffers`) that exists while a model is being fitted.

View
: A non-owning window onto memory: `MatrixRef<T>` (read-only) or `MatrixMut<T>` (writable), described by a pointer, the shape and the column stride.

Warm-up
: The first call that sizes a reusable workspace. Later calls of the same shape allocate nothing.

Workspace
: A set of preallocated buffers reused between calls: `FitBuffers` for training, `QueryWorkspace` for prediction.
