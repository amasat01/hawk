# Compiling

How hawk turns a kernel into something that runs.

- [The compile cache](compile_cache): why building the same kernel twice compiles once.
- [Compile options](compile_options): the flags and profiles a build takes.

The `c` target emits standalone MISRA-C for an embedded build instead of a
native library; see [Embedded C target](embedded.md).

## The compile guard

Two guards keep a compile from running away unseen: a very large source is refused before the
compiler sees it, and a compiler that does not finish is stopped.

- A translation unit over 8 MiB is refused with an error that names its size. Set
  `HAWK_COMPILE_SOURCE_BUDGET` (bytes) to change the limit; `0` turns it off.
- A compiler subprocess is stopped after 1800 seconds. Set `HAWK_COMPILE_TIMEOUT` (seconds) to
  change it; `0` turns it off.

A source this large is usually a wide kernel unrolled. Roll the loop, or read a wide table at a
runtime index with [`hawk.math.index`](tutorials/06_lookups_scatters_and_sums), which keeps the
source small and the gradient cheap.

## Wide values

A vector with hundreds of components used in a loop (`bf[j]`) is read from memory where it is indexed, not built in registers; a value that is used whole is placed by its size against the device's registers and local memory, so a wide kernel no longer spills because it was written naturally. There is nothing to set: the placement follows the device a kernel is built for, and the [spill report](emit_form) shows the result.

`hawk.ir.recognize(kernel)` reads a sum of products over one loop (a matrix-vector product, a masked sum, several outputs sharing an operand, with `bf[j]` or table reads as operands) as a plain record of operands, reduce axis and outputs, for tools that route such a sum elsewhere. It only reads: it never changes the code a kernel generates.

## Local memory

A thread on the GPU can hold only so much local memory (spilled registers, arrays and matrices it indexes at run
time): 512 KB. A kernel past that would fail at launch, so `hawk.build` refuses it by name, with the number and the
way out: hold fewer live matrices per lane, build with `linalg_form="team"` (see
[small linear algebra](linear_algebra)), or use a smaller `n`. The sidecar records the figure as
`launch.local_bytes`, with `local_source` (how it was measured) and `local_limit`. A launch of `n` threads needs about
`n * local_bytes` of device memory beside your planes, and eagle refuses a bind that cannot fit it.

```{toctree}
:maxdepth: 1
:hidden:

compile_cache
compile_options
```
