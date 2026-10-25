# Add a target

A target is one row of `hawk.targets.TARGETS`. The build, the package step and the compile drivers read the row, and
none of them compares a target's name, so a new target is a new row and the code it points to. A test fails if
`"cuda"` or `"host"` appears as a string anywhere outside `hawk/targets`.

Three jobs, smallest first:

| You want | You add |
|---|---|
| another file layout for the same language (a second export) | a row that reuses a Writer |
| a new execution model for a language hawk already writes (HIP) | a Wrapper, a row, and a compile entry |
| a new language (OpenCL C, MSL) | a Writer, then the same as above |

## The row

```python
Target(id="cuda", about="builds for the GPU", writer=AETHER, wrapper=CUDA, compile=GPU,
       entry=ENTRY_DEVICE, format="ptx", artifact=".ptx", source=".cu", features=frozenset(FEATURES))
```

`writer` says how the language spells each construct and `wrapper` how a kernel is wrapped for one execution model.
`wrapper=None` means the Writer produces the whole file (the `c` row). `compile` is the toolchain, `GPU`, `CPU` or
`None`: with `None` the row is exported, not built, and `export` names the module that renders and checks it.
`entry` is how eagle enters a built kernel (`"device"` or `"host"`, or `None` when eagle never runs it), and
`format` is the manifest's artifact format. `features` are the names from the shared table in
`hawk/targets/features.py` that the row builds; a kernel that uses one the row lacks is refused by name before
anything renders.

## A row that reuses a Writer

The smallest target is a copy of an existing row. This one exports the same C under new suffixes, with no change to
the build:

```python
from dataclasses import replace

import hawk
from hawk import Mutable, Param, Scalar
from hawk.targets import TARGETS


@hawk.kernel
def drag(v: Scalar, k: Param, a: Mutable[Scalar]):
    a = -k * v


TARGETS["toy"] = replace(TARGETS["c"], id="toy", about="a toy export", artifact=".toyh", source=".toy")
bundle = hawk.build([drag], "out", targets=("toy",))      # out/drag.toy and out/drag.toyh
```

Remove the row and the same call is refused: `unknown target 'toy'`, with the valid ones listed.

## A new execution model: the Wrapper

A Wrapper is the part of the unit that depends on how the kernel runs. It has one attribute and six methods:

| Member | It writes |
|---|---|
| `qualifier` | the prefix of the helper functions (`__device__ __forceinline__`, `static inline`) |
| `entry_signature(walk, name)` | the entry's declaration: the arguments, then the partition triple `base`, `count`, `nSamples` |
| `index_prologue(...)` | the one place that binds the body's lane index (a CUDA block and thread, a serial host loop) |
| `index_epilogue()` | the closing half of that seam |
| `unpack(walk)` | the lines that rebuild each argument's view from what the entry receives |
| `sample_sum_frame(...)` | the frame around a body with a reduction over lanes |
| `dispatch(...)` | a per-lane branch between formulas |

A Wrapper may also supply optional hooks (`simd_safe`, `lane_guards`, `interchange`), which the renderer reads with
`getattr`. See `hawk/emit/cuda.py` and `hawk/emit/host.py` for the two that exist. The launch geometry is eagle's:
a Wrapper names no stream and no occupancy.

Then add the row. A HIP target that reuses the aether C++ Writer would be (a sketch; hawk builds no HIP today):

```python
Target("hip", "builds for AMD GPUs", AETHER, HipWrapper(), HIP, ENTRY_DEVICE, "hsaco",
       ".hsaco", ".hip.cpp", frozenset(FEATURES))
```

## A toolchain

`compile=GPU` and `compile=CPU` name the two toolchains hawk drives: the device compiler and the host compiler.
A row that uses one of them needs nothing more. A new compiler (hiprtc) is a new value and a branch in
`hawk.compile.drivers.compile_source`, the one function that forks on the row's toolchain.

## A new language: the Writer

A Writer answers the renderer's questions in one language: declare a local, write an operator, store to a sink,
write a sum over lanes. `hawk.emit.writer.WRITER_MEMBERS` lists them, and each member's docstring gives the aether
writer's text. `hawk.embedded.writer.CWriter` is the second one. A language that cannot compile aether's C++ templates
also needs its own core for the numerics contract: see [the numerics contract](https://amasat01.github.io/aether/content/numerics_contract.html).

## The family side

A built row also reaches the other repos:

- `entry` and `format` are written to the manifest. A new format word is added to
  `raptor.schema.manifest.MANIFEST_FORMATS` and to eagle's `plugin/roles.h`; a test in each repo pins the two lists
  against each other.
- eagle runs the kernel through an execution backend for that device kind. See
  [Add an execution backend](https://amasat01.github.io/eagle/content/devguide/add_an_execution_backend.html).

## Renamed

| Was | Now |
|---|---|
| `Dialect` (`render_body(..., dialect=)`) | `Writer` (`writer=`) |
| `Backend` | `Wrapper` |
| `hawk.emit.BACKENDS` | `hawk.targets.TARGETS` |
| `hawk.compile.HOST`, `hawk.compile.DEVICE` | the target ids `"host"` and `"cuda"` |
| `CompileOptions(backend=...)` | `CompileOptions(target=...)` |
