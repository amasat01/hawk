# API reference

hawk's public Python surface, by subpackage. The authoring names
(`hawk.kernel`, the declarations, `hawk.steps`) live at the top level and
the math vocabulary, including the quaternion helpers and the random
distributions, at `hawk.math`. `hawk.raw_device` / `hawk.RawBlock` are the
escape hatch: splice a block of device code the vocabulary cannot spell,
with its access class declared by hand.

The compiled `hawk._core` extension and the IR internals are implementation detail and not documented here; they may
change between releases. The compiled backend libraries are reached only through `hawk.ospreys`, `hawk.targets` and
`hawk.artifact`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk
   hawk.math
   hawk.geometry
   hawk.ext
   hawk.diff
   hawk.diff.event
   hawk.ospreys
   hawk.targets
   hawk.runtime
   hawk.plans
   hawk.artifact
```

## The authoring names

The declarations a kernel is written in, and the top-level functions. Each has its own page.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk.kernel
   hawk.Kernel
   hawk.raw_device
   hawk.RawBlock
   hawk.steps
   hawk.chain
   hawk.scatter
   hawk.sample_position
   hawk.primitive
   hawk.resolve_dispatch
   hawk.samples_first
   hawk.samples_last
   hawk.Value
   hawk.Quantity
   hawk.Scalar
   hawk.Index
   hawk.Int32
   hawk.UInt32
   hawk.Quat
   hawk.Vector
   hawk.Matrix
   hawk.Param
   hawk.Meta
   hawk.Mutable
   hawk.Table
   hawk.Grid
   hawk.Sorted
   hawk.Staged
   hawk.Wide
   hawk.WideOut
   hawk.Terminated
   hawk.HawkError
```

## Running a kernel on the host

`hawk.load` opens a built kernel (a `LoadedKernel`), `hawk.run` runs it once, and the verbs below plan it, bind its planes and launch it as often as needed. `hawk.deploy(kernel)` builds a kernel into a plan, and calling a plan (`plan(x, a=3)`, which is what `kernel(x, a=3)` does) binds the arrays, launches once and returns what the kernel returns. Every one of these forms binds through `Plan.bound`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk.plan
   hawk.deploy
   hawk.HostTeam
   hawk.HostSerial
   hawk.DeviceKernel
   hawk.Partition
   hawk.asarray
   hawk.to_numpy
   hawk.synchronize
   hawk.banded_pack
   hawk.banded_unpack
```

## Running a model until it is done

`hawk.until_done` binds a plan, a kernel or a list of them run as one step and makes the loop that runs every sample to done; `hawk.simulate` runs it and returns the state. `hawk.ActiveSet` states how a runner compacts a kernel built over an active set, or makes the map and count that a plain launch of such a kernel runs. With `record=True` (and optionally `window=`, `memory_budget=` and `steps_reverse=`) a run keeps what its reverse pass needs, for a kernel built with `hawk.deploy(kernel, record=True)`, and `hawk.reverse` takes the `record` of the runner or of the result and returns the gradients of the cotangents you give, by plane name.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk.until_done
   hawk.simulate
   hawk.ActiveSet
   hawk.reverse
```

## Recording launches into a graph

A graph records bound plans once and replays them. `hawk.reverse` over a recorded graph is refused by name.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk.Graph
   hawk.GraphPipeline
```
