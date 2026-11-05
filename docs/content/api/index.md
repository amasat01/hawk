# API reference

hawk's public Python surface, by subpackage. The authoring names
(`hawk.kernel`, the declarations, `hawk.steps`) live at the top level and
the math vocabulary, including the quaternion helpers and the random
distributions, at `hawk.math`. `hawk.raw_device` / `hawk.RawBlock` are the
escape hatch: splice a block of device code the vocabulary cannot spell,
with its access class declared by hand.

The compiled `hawk._core` extension and the IR/emitter internals
(`hawk.ir`, `hawk.emit`) are implementation detail and not documented here.
Their `__all__` is a small "IR access" surface for a downstream package
that builds or renders sink sets by hand (`hawk.ir.canonical`,
`hawk.ir.recognize`, `hawk.emit.render_source`, …); everything else in
them may change between releases.

```{eval-rst}
.. autosummary::
   :toctree: generated

   hawk
   hawk.math
   hawk.geometry
   hawk.ext
   hawk.diff
   hawk.diff.event
   hawk.compile
   hawk.runtime
   hawk.artifact
   hawk.embedded
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
   hawk.scatter
   hawk.lane_position
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
