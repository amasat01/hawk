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
   hawk.ext
   hawk.diff
   hawk.compile
   hawk.runtime
   hawk.artifact
```
