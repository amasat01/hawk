# Glossary

hawk's own terms. For words shared across the family (manifest, sample,
DLPack, certification row, ...) see the
[family glossary](https://amasat01.github.io/raptor/content/glossary.html).

```{glossary}
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

payload
  The sealed bundle of aether's C++ headers that hawk's runtime GPU
  compiles build against. It ships in the `aether-dsc` package, so a GPU
  build needs no aether checkout.
```
