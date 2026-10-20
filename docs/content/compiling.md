# Compiling

How hawk turns a kernel into something that runs.

- [The compile cache](compile_cache): why building the same kernel twice compiles once.
- [Compile options](compile_options): the flags and profiles a build takes.

The `c` target emits standalone MISRA-C for an embedded build instead of a
native library; see [Embedded C target](embedded.md).

```{toctree}
:maxdepth: 1
:hidden:

compile_cache
compile_options
```
