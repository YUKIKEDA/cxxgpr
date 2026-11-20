# Your first regression

In this tutorial you fit a Gaussian process to eight noisy samples of $\sin(x)$, then ask the model for its prediction and its uncertainty at three new points. By the end you will have a complete, working program and you will know what every line of it does.

You need a working installation of cxxgpr. If you have none yet, follow {doc}`../how-to/install-and-link` first and come back. No knowledge of Gaussian processes is required; the tutorial explains each idea at the moment you need it, and {doc}`../explanation/gaussian-processes` tells the full story afterwards.

## What you will build

A program that

1. holds eight training points $(x_i, y_i)$ where $y_i \approx \sin(x_i)$,
2. describes the smoothness of the function with a kernel,
3. fits the model, which also learns the kernel's hyperparameters from the data, and
4. prints the predicted mean and standard deviation at $x = 1.35$, $3.15$ and $5.85$.

## Step 1: Include the headers and write the data

Create a file `first_regression.cpp`. Start with the includes and the opening of `main`:

```{code-block} cpp
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>

int main()
{
```

Then add the data:

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [data]
:end-before: // [end]
:dedent: 4
```

There is one input feature, so the training matrix has eight rows and one column. cxxgpr reads matrices in **column-major** order: the values of column 0 come first, then the values of column 1, and so on. With a single column that is just the eight numbers in a row, so a plain `std::array<double, 8>` is already a valid matrix. The targets `y_train` are $\sin(x)$ plus a little noise of about $\pm 0.1$ that we typed in by hand to keep the run reproducible.

## Step 2: Describe the model

A Gaussian process needs a **kernel**, a function $k(x, x')$ that says how strongly the function values at two inputs are tied together. We use the product of two building blocks:

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [model]
:end-before: // [end]
:dedent: 4
```

Walk through it line by line:

- `ConstantKernel::create(1.0)` is a constant $c$. Multiplied with another kernel it sets the **signal variance**: how far the function is allowed to stray from its mean.
- `RbfKernel::create(1.0)` is the squared-exponential kernel $\exp\!\bigl(-(x - x')^2 / (2\ell^2)\bigr)$ with lengthscale $\ell = 1$. Nearby inputs get similar outputs; the lengthscale says what "nearby" means.
- `create` can fail (for example for a negative lengthscale), so it returns a `std::expected`. cxxgpr never throws exceptions across its API: every fallible call returns a value that you test with `if (!result)`. We check both kernels once and move on.
- `KernelSpec(*signal) * KernelSpec(*shape)` wraps each kernel in a `KernelSpec` and multiplies them. Sums work the same way with `+`.
- `Gpr<optimizer::Lbfgs>(spec)` creates an untrained model description that will tune hyperparameters with the L-BFGS optimizer. `with_noise_variance(0.01)` sets the *starting* observation noise, $\sigma_n^2 = 0.01$. The optimizer adjusts it, too. The call returns a `std::expected`, because a non-positive noise variance is rejected.
- `with_target_transform(StandardizeTarget{})` tells the model to shift and scale the targets to mean 0 and standard deviation 1 before training, and to undo this when it reports predictions. Standardising keeps the starting hyperparameters sensible for data of any size.

Each `with_...` call returns a **new** description and leaves the old one untouched, so you can branch one base configuration into several variants.

:::{note}
`Gpr` is a trainer, not a model. The trained object appears in the next step.
:::

## Step 3: Fit

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

`fit(x, n_rows, n_cols, y)` takes the column-major matrix as a span together with its shape, and the targets. It standardises the targets, searches for the hyperparameters that make the data most probable, factorises the final covariance matrix, and returns a `FittedGpr`. Because `fit` is `const`, the trainer `configured` is the same afterwards, whether the fit succeeded or failed. On failure you get a `GprError` whose `message()` explains what went wrong, such as non-finite values in the data or mismatched sizes.

## Step 4: Predict

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
:dedent: 4
```

`predict(xs, n_rows, n_cols)` takes the query points in the same column-major layout. The result is a `Prediction` with two vectors, `mean` and `variance`, one entry per query point, both on the original scale of your targets. By default the variance is that of a *new noisy observation*: it contains the measurement noise $\sigma_n^2$. The square root of the variance is the standard deviation we print next.

## Step 5: Look at the results

The rest of the program prints what the model learned and compares the predictions with the true $\sin(x)$:

```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:start-after: // [report]
:end-before: // [end]
:dedent: 4
```

The kernel stores its parameters as logarithms so that they stay positive during optimisation, hence `std::exp`. After this block, add a loop that prints one line per query point and finish `main` (the complete listing is at the end of this page). Build and run the program; {doc}`../how-to/install-and-link` has the build commands. You should see:

```{code-block} text
signal variance = 1.91418, lengthscale = 1.71027
noise variance = 0.00549421 (standardised scale)
log marginal likelihood = 5.74717
x = 1.35  mean = 1.0122  std = 0.0652796  sin(x) = 0.975723
x = 3.15  mean = -0.108219  std = 0.0648688  sin(x) = -0.00840725
x = 5.85  mean = -0.289365  std = 0.067871  sin(x) = -0.419764
```

Read these numbers as follows:

- The optimizer started from signal variance 1 and lengthscale 1 and moved to $c \approx 1.91$ and $\ell \approx 1.71$. A lengthscale of 1.7 matches the gentle wiggle of a sine over $0$ to $6.3$. It also shrank the noise variance from the 0.01 we proposed to about 0.0055 (in standardised units), because the data are smooth.
- At each query the mean is close to the true $\sin(x)$, and the difference is within a few standard deviations of about 0.065. The program exits with status 0 only if every difference is below three standard deviations, so the claim is checked each time you run it.
- The standard deviation is nearly the same at all three points because each lies between training points that are 0.9 apart. Ask for a point far outside $[0, 6.3]$ and you will see it grow. The next tutorial does exactly that.

## The complete program

:::{dropdown} Show the whole file
```{literalinclude} /../examples/first_regression.cpp
:language: cpp
:linenos:
```
:::

## What you learned

You described a model with kernels, configured it with immutable builder calls, fitted it with `fit`, and predicted with `predict` to get a mean and an uncertainty. Along the way you met `std::expected` error handling, column-major matrices and target standardisation.

Continue with {doc}`multi-dimensional-regression` to handle several inputs on very different scales. For the individual pieces, see {doc}`../how-to/build-kernels`, {doc}`../how-to/transform-data` and {doc}`../how-to/predict-and-sample`.
