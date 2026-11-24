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
   hawk.device_props
   hawk.DeviceProps
   hawk.host_threads
   hawk.set_host_threads
   hawk.synchronize
   hawk.Stream
   hawk.Event
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

A graph records launches, conditionals and forks once and replays them as one; Python is recorded only as a `hawk.host_step`. Write one by hand (`g.add(bound)`, `with g.parallel() as p`, `with p.branch() as b`) or with `@hawk.Graph.builder`, which runs a function once: a kernel call inside it is recorded and returns its output arrays, and `g.result` declares what `g(n)` gives back after replaying `n` times. In a builder, `if`, `elif`, `else`, `while`, `for _ in range(cap)` that opens with `if not c: break`, and `match` on a one-element `int32` or `uint32` array (or a `hawk.SkipGuard`) become conditionals of the graph, read at every replay; any other condition is a Python value, decided once at build. `hawk.skippable(step, guard)` and `hawk.repeat_while(step, guard, max_iters)` are the same conditionals over a step you add by hand. Leaving a `with g.parallel()` is the join: no edge is ever declared, the library orders the branches by the planes they touch and refuses two that race, and `b.wait(array)` orders a branch after the earlier branches that write the array. `match` on a cell takes integer cases and an optional `case _`; `g.update(a=0.3)` or `g.update({"label.a": 0.3})` changes a value a launch binds by value, between replays, with no new record (not on a graph made with `record=True`). `g.bind(**arrays)` takes new arrays of the same shape and dtype for a builder's parameters with no new record.

`hawk.reverse(graph.record, cotangents, wrt=None)` over a graph made with `record=True` (a builder's kernel calls are built with their vjp; a plan added by hand is made with `hawk.deploy(kernel, reverse=True)`) returns the gradients of the cotangents, by plane name, through forks, conditionals and `while` loops as they ran at the last replay. `wrt` selects inputs; a plane bound by a Python number has no gradient and is refused by name (pass a one-element array). A built graph is drawn from what the library describes: `str(g)` and `g.show()`, `g.to_mermaid()`, `g.to_dot()`, `g.edges()`, `g.schedule()`, `g.node_labels()` and `g.num_nodes()`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk.Graph
   hawk.GraphPipeline
   hawk.SkipGuard
   hawk.skippable
   hawk.repeat_while
   hawk.host_step
```
