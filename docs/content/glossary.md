# Glossary

hawk's own terms. For words shared across the family (manifest, sample,
DLPack, certification row, ...) see the
[family glossary](https://amasat01.github.io/raptor/content/glossary.html).

```{glossary}
vocabulary
  The named per-sample arrays and parameters a family of kernels shares, declared once as a `Kind`; each kernel then
  reads them by name. See {doc}`hawk in your domain <vocabulary>`.

lane
  One work item: one sample, seen through the shape the kernel declares for
  that axis. A kernel body is one lane's code. See [Across lanes](across_lanes).

lane space
  The shape a kernel gives its flat sample axis, as an ordered list of axes.
  Memory stays flat; the shape only says how the lanes are grouped.

Grid
  The parameter type that declares the lane space:
  `lane: Grid["channel", "sample":8192]`. A bare label has its extent given
  at run time (first axis only); `"sample":8192` fixes it.

fastest axis
  The last axis a lane space declares. Consecutive lanes along it are
  consecutive threads or SIMD lanes.

group
  A lane count, the unit in which finished lanes are compacted away
  (`@hawk.kernel(group=32)`, default 1).

cell
  One value for the whole run, made by storing a reduction over lanes
  (`total = hawk.sum(x)`) to a `Mutable` output. A vector or
  matrix cell comes back as one array per entry (`total_0`, `total_1`, ...).

Sorted
  `Sorted[Index]`: an index plane whose values never decrease along the
  lanes. A promise that lets hawk add each run of equal keys once.

Writer
  How one language writes each IR construct as source text for a target.

Wrapper
  How a kernel is wrapped for one execution model: the declaration, the
  index, the argument unpacking and the reduction frame.

target
  A Writer and a Wrapper with their compile and package facts. Its id is
  what `hawk.build(targets=...)` takes. See
  [Add a target](contributing/add_a_target).

stage
  One launch of a kernel that reads a value reduced over samples. The read is
  a launch boundary: the kernel is built as stages, run in sequence, with the
  cell passed between them. See [Across lanes](across_lanes).

jump width
  The `jump=` of a kernel: how far from a switch a sample still pulls on a
  parameter that moves the switch (about 0.02 of the compared values' range).
  See [Gradients through branches and stops](tutorials/07_gradients_through_branches_and_stops).

extent
  How many values a plane holds: one per sample, one per key (a by-key
  store), or one for the whole run (a cell).

policy
  How a sum over lanes is added up: `policy="auto"` (default),
  `"compensated"` (tightest) or `"ordered"` (the same bits on every run).
  The same three words as `eagle.reduce`. Every sum is within k ULP of the
  exactly rounded sum.

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
