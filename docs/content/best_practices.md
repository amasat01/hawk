# Best practices

The short list of habits that keep hawk kernels fast and builds quiet. Each point is a rule and a small example.

## Constants and lookup tables: `Table` and `.at()`

- Put a constant or a lookup table in a `Table` and read it with `.at()`.
- Do not pass it as a `Vector` plane that holds the same numbers for every sample.
- Index with the value you have; `.at()` clamps the index to the table, so a read can never leave it.

```python
import hawk
from hawk import Index, Mutable, Scalar, Table

@hawk.kernel
def scale(x: Scalar, k: Index, gain: Table["i": 8], y: Mutable[Scalar]):
    y = gain.at(k) * x
```

A broadcast `Vector` is one copy of the table per sample: it costs memory, bandwidth and registers, and grows with `n`. A `Table` is held once. See [wide values](howto_wide_values).

## Both sides of an `if` run

- Treat `if`/`else` on a per-sample value as a select: both branches are computed, one result is kept.
- Guard a read with a clamped index, not with a branch around it.

```python
@hawk.kernel
def pick(i: Index, tab: Table["j": 8], y: Mutable[Scalar]):
    # tab.at(i + 3) is read for every sample, taken or not; the clamp keeps it in the table
    y = tab.at(i + 3) if i < 5 else 0.0
```

See [gradients through branches](tutorials/07_gradients_through_branches_and_stops) and [dispatch strategies](tutorials/08_choosing_a_dispatch_strategy).

## Wide values: placed, or per lane

- Write a wide vector the natural way and let hawk place it. Do not split it by hand to avoid spills.
- Read a wide value with a loop index (`bf[j]`) when you need one component at a time: it is read from memory where it is indexed.
- Keep a `Mutable[Vector[W]]` for an output of several components per lane; use `WideOut` only when the length is set at run time.

See [wide values](howto_wide_values).

## Compose kernels with `hawk.chain` or graphs

- Build small kernels and join them with `hawk.chain`, or run them as nodes of a graph.
- Do not hand-fuse many steps into one giant kernel: it compiles slowly, uses many registers and is harder to test.

```python
step = hawk.chain(predict, correct, finish, name="step")
```

hawk cuts a chain where one body would spill, so you keep the small pieces and the speed. See [composition](composition).

## Precompile with the build's targets

- When you warm the compile cache ahead of a run, call the same `hawk.build` with the same `targets=` (and mode and options) as the real build.
- A different target set is a different compile: it warms nothing the real build can use.

```python
TARGETS = ("host", "cuda")
hawk.build([step], "warm/", targets=TARGETS)    # in the install step
hawk.build([step], "out/", targets=TARGETS)     # in the program: a cache hit
```

See [the compile cache](compile_cache).

## Let hawk pick the host compiler

- Leave `HAWK_CXX` unset. hawk finds a suitable compiler on its own.
- Set `HAWK_CXX=/path/to/c++` only for a real alternative, such as Homebrew's clang on macOS or a newer `g++`.
- Do not generate or commit wrapper scripts that stand in for the compiler; set the variable to the compiler itself.

See [installation](installation).
