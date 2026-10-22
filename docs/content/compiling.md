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

```{toctree}
:maxdepth: 1
:hidden:

compile_cache
compile_options
```
