# Regression with several inputs

In this tutorial you fit a model with two inputs whose scales differ by a factor of one hundred, let the model learn how much each input matters, look at how its uncertainty grows away from the data, and check the fit with leave-one-out predictions. It builds on {doc}`first-regression`; do that one first.

## What you will build

The function we pretend not to know is

$$
f(x_1, x_2) = \sin(x_1) + \frac{x_2}{500}, \qquad x_1 \in [0, 6],\quad x_2 \in [0, 500].
$$

It oscillates quickly in $x_1$ and rises slowly and linearly in $x_2$. You will draw 30 noisy samples of it and recover this structure from the data alone.

## Step 1: Generate the training data

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [data]
:end-before: // [end]
:dedent: 4
```

The training matrix has 30 rows (samples) and 2 columns (features). In column-major order that means all 30 values of $x_1$ come first, then all 30 values of $x_2$. That is why the loop writes `x[i]` for the first feature and `x[n + i]` for the second: column $j$ starts at offset $j \cdot n$. The generator is `std::mt19937`, whose output the C++ standard fully specifies, so you get exactly the same data on every platform.

## Step 2: Choose an ARD kernel and transforms

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [model]
:end-before: // [end]
:dedent: 4
```

Two things are new compared with the first tutorial.

**An ARD kernel.** `RbfArdKernel` is the squared-exponential kernel with one lengthscale per input column (ARD stands for automatic relevance determination). A short lengthscale means the function changes quickly along that input; a long one means the input hardly matters. You give the starting values as a span, one per column. The optimizer then learns each of them separately.

**Standardised inputs.** `with_input_transform(StandardizeInput{})` subtracts each column's mean and divides by its standard deviation before the kernel sees the data. Without this step, $x_2$ would be about 100 times larger than $x_1$ and a starting lengthscale of 1 would be absurd for it. The transform is *fitted* on the training data inside `fit` and applied automatically to every query, so you never transform queries yourself. `with_target_transform(StandardizeTarget{})` does the same for the targets, as before.

The lengthscales you read back later are therefore in **standardised units**, not in the units of $x_1$ and $x_2$.

## Step 3: Fit and read the hyperparameters

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

The kernel `ConstantKernel * RbfArdKernel` has three parameters, in this order: the constant, the lengthscale for column 0, and the lengthscale for column 1. `get_params` copies them, as logarithms, into a span of exactly that size; a wrong size is an error, not a silent truncation.

## Step 4: Predict inside and outside the data

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [predict]
:end-before: // [end]
:dedent: 4
```

The query matrix has three rows and two columns, again column-major: the first three numbers are the $x_1$ values of the three queries and the last three are their $x_2$ values. The first two queries lie inside the training region. The third has $x_1 = 12$, twice the largest $x_1$ in the data.

## Step 5: Check with leave-one-out predictions

How do you know whether the model is any good without a test set? `loo_predict()` answers: for every training point it returns the prediction you would have made had that point been left out of the data, computed in closed form from the existing factorisation, without refitting 30 times.

```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:start-after: // [loo]
:end-before: // [end]
:dedent: 4
```

Two numbers summarise the result. The root-mean-square error compares the leave-one-out means with the observed targets. The mean squared $z$-score divides each error by the predicted standard deviation; for a well-calibrated model it is close to 1.

## Run it

You should see:

```{code-block} text
noise variance = 0.00623201 (standardised scale)
lengthscale for x1 = 1.38772, for x2 = 9.63727 (standardised units)
query (3, 250): mean = 0.64008, std = 0.0652656, truth = 0.64112
query (1, 100): mean = 1.06294, std = 0.0732807, truth = 1.04147
query (12, 250): mean = 0.591749, std = 1.50877, truth = -0.0365729
leave-one-out RMSE = 0.0683095, mean squared z-score = 0.92736
```

What the numbers tell you:

- **ARD found the structure.** The lengthscale for $x_2$ is about 9.6, seven times the one for $x_1$ (1.39). The model learned, without being told, that $x_2$ varies slowly and $x_1$ quickly.
- **Interpolation is accurate and confident.** At the two inner queries the mean is within 0.03 of the truth and the standard deviation is about 0.07. Most of that is observation noise: the data were generated with uniform noise of standard deviation $0.2/\sqrt{12} \approx 0.058$.
- **Extrapolation is honest.** At $x_1 = 12$ the standard deviation jumps to 1.5, about twenty times larger, and the mean (0.59) is far from the truth ($-0.04$). The model does not pretend to know; its error bar says so. The program exits with status 0 only if this ratio is above five.
- **The fit is calibrated.** The leave-one-out RMSE of 0.068 is close to the noise level, and the mean squared $z$-score of 0.93 is close to 1, so the predicted uncertainties match the actual errors.

## The complete program

:::{dropdown} Show the whole file
```{literalinclude} /../examples/multi_dimensional_regression.cpp
:language: cpp
:linenos:
```
:::

## What you learned

You used an ARD kernel to let each input have its own lengthscale, fitted input and target standardisation inside the model, read learned hyperparameters back with `get_params`, saw uncertainty grow away from the data, and validated the fit with `loo_predict`.

Next, {doc}`save-and-reload` shows how to store this kind of model and use it again later. For more on the pieces, see {doc}`../how-to/build-kernels`, {doc}`../how-to/transform-data` and {doc}`../how-to/predict-and-sample`.
