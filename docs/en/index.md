# cxxgpr

**Exact Gaussian Process Regression for C++23.** A static library with a value-returning, exception-free API,
ten built-in kernels, five self-contained optimizers, double/single/mixed-precision solvers and a portable
on-disk model format.

```{code-block} cpp
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
// Fit, then ask for the mean *and* the uncertainty. Errors are values, never exceptions.
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Learn by doing. Fit your first model, then grow it into a realistic multi-dimensional workflow.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

Solve a specific task: build kernels, transform data, choose an optimizer, save a model, tune performance.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Look things up: every public class, function and enumerator, error codes, build options, file format.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

Understand why: the maths, the numerical choices, and the design decisions behind the library.
:::
::::

## Where to start

- New to Gaussian processes? Read {doc}`explanation/gaussian-processes`, then follow {doc}`tutorials/first-regression`.
- Already know the theory? Jump to {doc}`how-to/install-and-link` and {doc}`how-to/index`.
- Coming from the Rust crate `gprx`? See {doc}`explanation/relationship-to-gprx`.

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
how-to/index
reference/index
explanation/index
```
