# Low-level control, next to NVIDIA Warp

[NVIDIA Warp](https://nvidia.github.io/warp/) and hawk both let you write a GPU kernel in
Python. They start from different places, and that shows in what each one lets you control.
This page lines the two up. Warp is described as its own documentation describes it; for
anything beyond that, read Warp's docs. The two tools exchange arrays without a copy
([hawk next to Warp](examples/04_hawk_next_to_warp) shows it), so the choice is per kernel.

## What you control

| | Warp, as documented by Warp | hawk and eagle |
|---|---|---|
| The unit you write | one thread, found with `wp.tid()`; you index arrays yourself | one sample; planes declare what each argument is (`Scalar`, `Param`, `Mutable[...]`) |
| Reaching another sample | read any array element; update with atomics such as `wp.atomic_add` | by name: `hawk.all_samples(x)[i]` to read, `[key] += v` to accumulate, `.sum() .max() .min() .mean()` to reduce ([Across samples](across_samples)) |
| Sums over the threads | atomics, or Warp's array and tile reductions | a reduction stored to a cell; the policy picks `"compensated"` or `"ordered"` (the same bits on every run, CPU and GPU) |
| Launch and graph | `wp.launch`, streams, CUDA graph capture | `eagle.deploy`, `eagle.Graph`, graph capture and replay, `eagle.simulate` ([Compiling](compiling)) |
| Gradients | a recorded tape of launches, replayed backward | a derived `jvp`/`vjp` per kernel, `step_vjp` for a stepping kernel, `eagle.diff.reverse` over a recorded graph |
| Escape hatch | native code from Python | `hawk.raw_device`, for a block of device code |

## Where hawk and eagle now come close

Cross-sample steps used to be separate plane kinds (`Accum`, `Reduce`). Now they are
written in the body, as named forms, and each says what it costs. A read is
`all_samples(x)[i]`. A by-key `+=` is the target's atomic add, so its order is not fixed. A
reduction is finished in a fixed order if you ask for `policy="ordered"`. A worked example: [One sample, one lane](tutorials/09_per_sample_control).

## Where Warp still offers more

- **Thread-level freedom.** In Warp any thread can read or write any element, loop over
  neighbours or call a function on a mesh. hawk keeps one sample per body, and a read of
  another sample is recomputed from the inputs, so it cannot read a draw or a plane this
  launch writes.
- **Cooperation inside a kernel.** Warp documents tile operations for block-level work. In
  hawk a reduced value read later in the body ends the launch: the kernel runs as launches in
  sequence (the Limits section of [Across samples](across_samples)).
- **More atomics.** Warp has atomic min and max. hawk's by-key store is `+=` or `=`; take a
  max or min over all samples with `.max()` / `.min()` into a cell, not by key.
- **Libraries.** Warp ships geometry, meshes and simulation building blocks, and a tape
  over a whole simulation. hawk differentiates one kernel or one recorded graph at a time.

Neither list is a ranking. If a job needs thread-level freedom or Warp's geometry, write it in
Warp and hand the arrays to hawk; if it is many independent samples with a few named
exchanges between them, hawk keeps the kernel small and the sums reproducible.
