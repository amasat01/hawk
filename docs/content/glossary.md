# Glossary

hawk's own terms. For words shared across the family (manifest, sample,
DLPack, certification row, ...) see the
[family glossary](https://amasat01.github.io/raptor/content/glossary.html).

```{glossary}
all-samples view
  `hawk.all_samples(x)`: every sample's `x` at once, as against `x`, which is
  this sample's. It is indexed, masked or reduced, never used in arithmetic.
  See [Across samples](across_samples).

cell
  One value for the whole run, made by storing a reduction over samples
  (`total = hawk.all_samples(x).sum()`) to a `Mutable` output. A vector or
  matrix cell comes back as one array per entry (`total_0`, `total_1`, ...).

stage
  One launch of a kernel that reads a value reduced over samples. The read is
  a launch boundary: the kernel is built as stages, run in sequence, with the
  cell passed between them. See [Across samples](across_samples).

jump width
  The `jump=` of a kernel: how far from a switch a sample still pulls on a
  parameter that moves the switch (about 0.02 of the compared values' range).
  See [Gradients through branches and stops](tutorials/07_gradients_through_branches_and_stops).

extent
  How many values a plane holds: one per sample, one per key (a by-key
  store), or one for the whole run (a cell).

reduction policy
  How eagle adds up the rows of a cell: `"compensated"` (default),
  `"ordered"` (the same bits on every run) or `"auto"`.

plane
  The role one argument plays for every sample: a per-sample input
  (`Scalar`), a shared constant (`Param`) or an output (`Mutable[...]`).

kernel kind
  A declared shared shape for a family of kernels (`KernelKind`), grown by
  ordinary Python inheritance.

primitive
  A function you register with `hawk.ext.primitive` together with its
  forward and, optionally, `jvp`/`vjp` rules, so it can appear in a kernel.

guard
  Which masks gate a sample (`hawk.ext.Guard`): a sample runs only when
  none of the named masks, such as `terminated` or `rejected`, is set.

dispatch policy
  How a per-sample branch between several formulas is compiled. It never
  changes what is computed.

tape
  The record of the state a stepping kernel read at each step, kept so a
  gradient can be taken backward through a whole rollout.

jvp
  Forward-mode derivative: how an output moves for a given change of the inputs.

vjp
  Reverse-mode derivative: how a change of the output maps back to the inputs.

step vjp
  The vjp of one step of a kernel that updates its planes in place
  (`hawk.diff.step_vjp`). It reads each plane's value before the step, takes
  `bar_<plane>_next` (the gradient with respect to the value after the step)
  and returns `bar_<plane>` (the gradient with respect to the value before it).

derived
  A kernel that hawk builds from another one, such as its `jvp`, `vjp` or
  taping twin.

compile cache
  The store of compiled kernels. Building the same kernel again copies it
  from the cache instead of recompiling.

device image
  The compiled GPU code (`DeviceImage`, PTX or SASS) hawk produces for a
  kernel; eagle launches it.

integer plane
  A per-sample plane of 32-bit integers, declared `Int32` (`np.int32`) or
  `UInt32` (`np.uint32`); `Param[Int32]` declares an integer parameter.
  `+ - *` wrap modulo 2^32, `//` and `%` follow Python's sign rules (a
  zero divisor gives 0), `& | ^ ~ << >>` act on the bits (a shift count
  outside `[0, 32)` gives 0, or -1 for a negative value shifted right), as
  NumPy does. An integer never mixes with a float without a cast:
  `hawk.math.real(n)` gives the float, `hawk.math.int32(x)` /
  `hawk.math.uint32(x)` the integer (truncating toward zero, saturating at
  the type's range, NaN to 0). An integer plane is never differentiated.
  `Index` is the separate 64-bit index type for addresses and selectors.

payload
  The sealed bundle of aether's C++ headers that hawk's runtime GPU
  compiles build against. It ships in the `aether-dsc` package, so a GPU
  build needs no aether checkout.
```
