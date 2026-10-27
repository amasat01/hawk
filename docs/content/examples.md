# How-to guides

Standalone pages, one task each. Start wherever the task matches what you need.

## Write kernels

Every way to write and call a kernel, reusing one kernel family, and what to do when hawk cannot spell a construct.

```{toctree}
:maxdepth: 1

examples/06_every_form_of_a_kernel
examples/03_kernelkind_reuse
vocabulary/when_hawk_cant_spell_it
examples/01_ensemble_drag_throughput
```

## Derivatives

Check a gradient, compare it with torch, and mix analytic physics with a learned correction. The names and options are on the [derivatives reference](derivatives).

```{toctree}
:maxdepth: 1

examples/02_autodiff_finite_difference_check
examples/autodiff_vs_torch
examples/two_body_hybrid
```

## Linear algebra

Solve, factor and decompose a small matrix in every lane.

```{toctree}
:maxdepth: 1

howto_linear_algebra
```

## Embedded C

Emit a kernel as standalone C for a target without a runtime.

```{toctree}
:maxdepth: 1

examples/05_embedded_c_target
```
