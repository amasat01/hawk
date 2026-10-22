# Examples

Standalone notebooks, each one task. Unlike the tutorials, these do not
build on each other — start wherever the task matches what you need. The last page,
*When hawk can't spell it*, helps choose between a primitive, `raw_device` and a native op.

- [A drag kernel over a large ensemble](examples/01_ensemble_drag_throughput): CPU and GPU agree, with the draws made inside the kernel.
- [A gradient checked against finite differences](examples/02_autodiff_finite_difference_check) and [against torch](examples/autodiff_vs_torch).
- [One kernel family, reused](examples/03_kernelkind_reuse) and [hawk next to Warp](examples/04_hawk_next_to_warp).
- [A hybrid orbit](examples/two_body_hybrid): analytic physics plus a learned correction, on the CPU and, if you have one, the GPU.

```{toctree}
:maxdepth: 1

examples/01_ensemble_drag_throughput
examples/02_autodiff_finite_difference_check
examples/03_kernelkind_reuse
examples/04_hawk_next_to_warp
examples/autodiff_vs_torch
examples/two_body_hybrid
vocabulary/when_hawk_cant_spell_it
```
