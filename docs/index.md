```{raw} html
<div class="raptor-hero">
  <img class="raptor-reveal dark-light" src="_static/brand/wide_family_hawk.svg" alt="hawk">
</div>
```

# hawk

**One kernel, two machines.** hawk is a Python DSL (domain-specific
language) for writing a numerical kernel once and running it, unchanged,
on CPU or GPU.

```{image} _static/ecosystem/ecosystem_hawk_light.svg
:alt: The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at hawk.
:class: only-light
:align: center
```

```{image} _static/ecosystem/ecosystem_hawk_dark.svg
:alt: The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at hawk.
:class: only-dark
:align: center
```

[aether](https://amasat01.github.io/aether/) · [hawk](https://amasat01.github.io/hawk/) · [eagle](https://amasat01.github.io/eagle/) · [raptor](https://amasat01.github.io/raptor/) · [the family](https://amasat01.github.io/)

A kernel is a plain Python function decorated `@hawk.kernel`. hawk turns
it into a small typed representation of what it computes, derives its
forward- and reverse-mode derivatives from that representation, and
compiles it for the [eagle](https://amasat01.github.io/eagle/) runtime to
run.

```python
import hawk
from hawk import Mutable, Param, Scalar

@hawk.kernel
def scale(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b
```

That is the whole authoring surface: a declared vocabulary of planes — a
plane is the role one argument plays for every sample, such as a per-sample
input, a shared constant, or an output — and each parameter's type says
which plane it fills (`Scalar` a per-sample input, `Param` a uniform
constant, `Mutable[...]` an output plane; defined in full in the first
tutorial, next), plus an ordinary arithmetic body.
`hawk.artifact.build` compiles it for host and device in one call —
`pip install raptor-hawk` (the CPU route; for a GPU add the `[cuda12]` or `[cuda13]` extra, see the
[full install guide](content/installation)) is all it takes to try it.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Start here
:link: content/tutorials/01_your_first_kernel
:link-type: doc
Write, build and run a kernel in about five minutes — then run the same
one on a GPU with the same call.
:::

:::{grid-item-card} Tutorials
:link: content/tutorials
:link-type: doc
Arguments and shapes, stopping conditions, gradients backward and
forward, lookups and sums.
:::

:::{grid-item-card} Vocabulary
:link: content/vocabulary
:link-type: doc
Write your own vocabulary and pair it to a new kernel kind: inheritance, a
custom derivative, and a capstone running the whole stack through PyTorch.
:::

:::{grid-item-card} Reference
:link: content/api/index
:link-type: doc
Every public name in `hawk`, `hawk.math`, `hawk.ext`, `hawk.diff`,
`hawk.compile` and `hawk.runtime`.
:::
::::

New to the whole RAPTOR family? The landing page's
[Start here](https://amasat01.github.io/start_here.html) walks through one
kernel, a million samples and a PyTorch fit in ten minutes, across every
repo at once.

## Where hawk sits

hawk is the kernel-authoring layer of the RAPTOR family. It writes and
compiles kernels, and `hawk.runtime` runs a host build directly; it never
launches a DEVICE artifact itself — that is deployed and run by
[eagle](https://amasat01.github.io/eagle/), the family's GPU execution
layer, over the protocol [raptor](https://amasat01.github.io/raptor/)
defines and [aether](https://amasat01.github.io/aether/) provides the
vector-algebra vocabulary for.

```text
aether (numerics core)  --->  eagle (execution)  <---  raptor (contracts)  --->  hawk (kernel authoring)
```

aether lays out arrays and provides the device-safe vector-algebra
vocabulary; eagle launches and captures what runs on top of it; raptor is
the dependency-free foundation that defines the manifest and protocol
contracts eagle and hawk both build to without depending on each other;
hawk authors and compiles the kernels eagle deploys.

```{toctree}
:caption: Start here
:maxdepth: 1
:hidden:

content/installation
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:

content/tutorials
```

```{toctree}
:caption: Vocabulary
:maxdepth: 1
:hidden:

content/vocabulary
```

```{toctree}
:caption: How-to guides
:maxdepth: 1
:hidden:

content/examples
```

```{toctree}
:caption: Explanation
:maxdepth: 1
:hidden:

content/compile_cache
content/compile_options
content/interop
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

content/api/index
```

```{toctree}
:caption: Contributing
:maxdepth: 1
:hidden:

content/contributing
```
