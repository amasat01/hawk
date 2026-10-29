# Broadcasting and the run-time first axis

A kernel's lanes have a shape, declared with a `Grid` parameter: `lane: Grid["channel", "sample":8192]`. Two rules decide how arrays of other shapes meet that shape.

## Broadcasting

A per-lane input you only read may be given broadcast, as in NumPy: a scalar, an array with axes of size 1, or one with fewer axes, lined up from the last axis. hawk binds your array as it is, without copying it, and compiles the kernel to read it along only the axes it varies along. The gradient of such an input comes back in its own shape. A per-lane `Matrix[R, C]` broadcasts too: one matrix for every lane, or one per index along the axes it varies along (`(R, C, 64, 1, 1)`). Its gradient is summed the same way. Planes a kernel writes, and the `terminated` mask, keep the exact lane shape.

The same holds for a deployed kernel: `eagle.deploy(k)(...)`, `.run` and `.bind` take the broadcast shape as `k(...)` does, and build the kernel for that shape once. A bound plane is rebound with the compact buffer of the same shape; a different shape is a new bind. The gradient of a broadcast input through a recorded graph is the same sum, in the input's own shape. A kernel loaded from a saved artifact has no source to rebuild, so it refuses a broadcast shape by name.

## The run-time first axis

A bare label (`Grid["channel", ...]`, `Mutable["node"]`, `Table["elem", "c":3]`) is an extent given when you run, and only the first axis may be bare. Give it as `node=N` or by passing a plane of that length; one build serves every size. A label that a `Grid` axis shares takes the lane extent. `"sample":8192` fixes an axis at build time instead.

The fastest axis is the last one declared: consecutive lanes along it are consecutive threads or SIMD lanes.

## A list of lanes

A launch need not cover the whole lane space. `Kind(..., lanes=IndexSet())` runs the lanes in a map you pass, and the extent you give (`node=N`, `pair=P`) is the size of the dense space the map points into. See [Run only the lanes you list](howto_index_set).

See {doc}`Your own vocabulary <vocabulary>` for the same words in context, the [glossary](glossary) for one-line definitions, and [Heat diffusion](vocabulary/09_heat_diffusion) for a two-axis grid.
