# Save a model and load it again

In this tutorial you fit a model once, write it to disk, load it back in what could be a different program, and confirm that the loaded model returns exactly the same predictions. Do {doc}`first-regression` first; this tutorial reuses its model.

## Step 1: Fit a model

Create `save_and_reload.cpp`. Its includes are those of the first tutorial plus two new ones, `cxxgpr/persist/persist.hpp` and `<filesystem>`; the complete listing is at the end of this page. The program starts by repeating the fit of the first tutorial and keeping the predictions at three query points in `before`:

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [fit]
:end-before: // [end]
:dedent: 4
```

## Step 2: Save

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [save]
:end-before: // [end]
:dedent: 4
```

`persist::save_with_factor(model, dir)` writes the fitted model into the directory `dir`. A saved model is a directory with two files:

- `config.json`, a human-readable description of the kernel with its learned hyperparameters, the noise, the transforms and the numerical settings;
- `model.safetensors`, the training data and, because we asked for the factor, the Cholesky factor and the weight vector as raw tensors.

The `remove_all` call in the example only makes it safe to run the program repeatedly. Like every cxxgpr function that can fail, `save_with_factor` returns a `std::expected`. Typical failures are an unwritable directory, or a model that uses a custom kernel the library does not know how to store.

:::{tip}
`persist::save` writes the same model without the factor. The file is smaller; loading is slower because the factor is recomputed. Use `save_with_factor` when load time matters.
:::

## Step 3: Load

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [load]
:end-before: // [end]
:dedent: 4
```

`persist::LoadedGpr::load(dir)` reads and validates the directory. The result, a `LoadedGpr`, does not carry the compile-time types of the model you saved (optimizer, precision policy); it carries only what prediction needs. `n()` and `d()` report the number of training points and the number of features; `has_factor()` says whether the factor came from disk.

If the directory is missing, damaged, or written by a newer format version, `load` returns an error instead of a half-initialised model.

## Step 4: Predict with the loaded model

```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:start-after: // [compare]
:end-before: // [end]
:dedent: 4
```

`LoadedGpr::predict` has the same signature as `FittedGpr::predict`, including the optional `PredictOptions`, and returns the same `Prediction`. The input and target transforms are stored with the model, so the loaded model standardises queries and un-standardises answers by itself.

Run the program. You should see:

```{code-block} text
saved to sin_model
loaded: n = 8, d = 1, has_factor = true
x = 1.35  before: 1.0122 / 0.00426142  after: 1.0122 / 0.00426142
x = 3.15  before: -0.108219 / 0.00420796  after: -0.108219 / 0.00420796
x = 5.85  before: -0.289365 / 0.00460647  after: -0.289365 / 0.00460647
bit-identical = true
```

The mean and variance of every query agree exactly, bit for bit, and the program exits with status 0 only because they do. A directory named `sin_model` is now in your working directory; open `config.json` in a text editor to see what was stored.

## The complete program

:::{dropdown} Show the whole file
```{literalinclude} /../examples/save_and_reload.cpp
:language: cpp
:linenos:
```
:::

## What you learned

You persisted a fitted model with `persist::save_with_factor`, restored it with `persist::LoadedGpr::load`, and verified that predictions survive the round trip unchanged.

For models with custom kernels, the choice between `save` and `save_with_factor`, and the typed round trip back to a `FittedGpr`, see {doc}`../how-to/persist-models`; the on-disk layout is documented in {doc}`../reference/file-format`.
