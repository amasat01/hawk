# Interoperability

hawk itself crosses exactly one boundary: binding a plane on the **host**.
Everything on the **device** crosses through
[eagle](https://amasat01.github.io/eagle/) instead — hawk compiles a device
artifact but never launches one (see
[Stop when you're done](tutorials/03_stop_when_youre_done)) — so this page
covers the host crossing in full and then hands off.

## DLPack is the contract

The family-wide rule, stated once by
[raptor's interoperability certification matrix](https://amasat01.github.io/raptor/content/interop_protocols.html):
every device crossing is, mechanically, a DLPack exchange, and a capability
is claimed here only where a certification row backs it. hawk's own
public claims are narrower than the family's, and are listed below with
nothing implied beyond them.

## The host crossing: the buffer protocol, zero-copy

This page is explicitly about the runtime/binding layer itself, so its
examples spell out `hawk.runtime`/`rt.load`/`rt.run` rather than hawk's
top-level `hawk.load`/`hawk.run` doors (the same functions, re-exported).

`hawk.runtime` binds a plane by **address**, through the Python buffer
protocol (`ctypes` over `memoryview`) — not through numpy specifically.
Any C-contiguous, writable object that exports a buffer binds the same way:
a numpy array, or the numpy view of a CPU `torch` tensor (`tensor.numpy()`,
which shares the tensor's own memory rather than copying it).

```python
import pathlib
import tempfile

import hawk
from hawk import Mutable, Param, Scalar
from hawk.artifact import build
import hawk.runtime as rt

@hawk.kernel
def scale(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b

work_dir = pathlib.Path(tempfile.mkdtemp())
build(scale, work_dir, targets=("host",))
k = rt.load(work_dir, "scale")
```

```python
import torch

x = torch.arange(8, dtype=torch.float64)
y = torch.zeros(8, dtype=torch.float64)
rt.run(k, x=x.numpy(), a=2.0, b=1.0, y=y.numpy())
print(y)   # written in place through the shared buffer -- no copy back
# tensor([ 1.,  3.,  5.,  7.,  9., 11., 13., 15.], dtype=torch.float64)
```

Two things this binding does **not** do, both deliberate:

- It never allocates: every output plane is the caller's own, written in
  place. hawk scratch-allocates nothing a kernel writes into.
- It refuses a read-only or non-contiguous buffer outright, naming the slot
  — binding by address means a copy would silently drop a kernel's writes
  or freeze an input's later updates, so hawk never falls back to one.

## Layouts: component-major planes, sample-major views

A per-sample vector plane is **component-major**: a `Vector[3]` plane over
`N` samples is shaped `(3, N)`, one contiguous row per component. Most array
code stores per-sample vectors **sample-major**, `(N, 3)`. `hawk.runtime`
accepts a sample-major plane only when no copy is needed:

- An `(N, 3)` array whose transpose is C-contiguous (for example `x.T` of a
  contiguous `(3, N)` array) already holds the `(3, N)` bytes. It binds
  zero-copy, input or output, and the kernel's writes land in the caller's
  storage. The sample count is read from its leading axis.
- Any other `(N, 3)` array, such as a plain C-contiguous one, would need a
  copy, so it is refused, naming the argument, both shapes and the fix:
  pass `np.ascontiguousarray(x.T)`, allocate the plane as `(3, N)`, or launch
  through eagle, which copies it for you with an `eagle.LayoutWarning`.
- A square `(3, 3)` plane at `N == 3` reads both ways, as component-major
  `(3, N)` and as sample-major `(N, 3)`, so hawk will not guess: it is refused,
  naming the argument, its shape, both readings and the two fixes below. A
  square matrix head with `N == R == C` is refused the same way.

**Saying which axis holds the samples.** Both fixes are zero-copy, and neither
changes a shape that is not ambiguous:

- Per call: `hawk.run(k, ..., layout="samples_first")` (planes are `(N, 3)`) or
  `layout="samples_last"` (planes are `(3, N)`, native) resolves every ambiguous
  plane of that call. `HostKernel.bind_all` takes the same `layout=`.
- Per array: `hawk.samples_first(x)` or `hawk.samples_last(x)` wraps one
  argument. It is honoured for any shape, a marker that contradicts the shape
  is refused naming the argument, and it wins over the call's `layout=`.

`samples_first` still needs an array hawk can bind by address, so the `(N, 3)`
plane must be a free transposed view as above. The markers are shared with
eagle (`eagle.samples_first` / `eagle.samples_last`): each package accepts the
other's, since a marker is any object carrying `__raptor_samples_axis__`
(`"first"` or `"last"`) and the wrapped array as `.array`.

Scalar planes are one-dimensional and are not affected.

**One sample** is a plane whose shape is its per-sample head: a 0-d array for
a `Scalar` plane, `(3,)` for a `Vector[3]`, `(R, C)` (or the flat `(R*C,)`) for
a `Matrix[R, C]`. Its bytes are the `(3, 1)` plane, so `hawk.runtime` binds it
at its own address as a batch of one, and the kernel writes land in the
caller's 0-d or `(3,)` array (`plane_layout` answers `"single"`). A shape that
is also a batch stays a batch: `(1,)` on a scalar plane, `(3, 1)`, `(1, 3)` and
`(3, 3)`. A call that mixes one sample with a batch is refused, naming both,
and so is a Python number bound to a plane (there is no address to bind:
pass `np.array(x)`). eagle's own entry points apply the same rule and also accept
numbers for inputs and return head-shaped results; see
[Stop when you're done](tutorials/03_stop_when_youre_done.ipynb).

## The device crossing: eagle

A compiled device artifact (`.ptx`/`.cubin`, from
`hawk.artifact.build(..., targets=("cuda",))` or the pip-only
`hawk.compile.cubin`) is loaded and launched by eagle, which is where
`cupy` and CUDA `torch` tensors cross — through the same DLPack contract
raptor's matrix certifies. See:

- [eagle's interoperability pages](https://amasat01.github.io/eagle/) for
  the device-side numpy/cupy/torch crossing eagle performs on hawk's
  behalf.
- [raptor's interoperability certification matrix](https://amasat01.github.io/raptor/content/interop_protocols.html)
  for the family-wide, row-cited statement of what is certified — the
  authority this page defers to for anything about the device side.

:::{admonition} Roadmap
:class: note

`jax` and `tensorflow`/`keras` are both DLPack-capable and both foreseen for
the family. Neither is certified today anywhere in the stack hawk compiles
for — treat a claim that either works now as a documentation bug, here or
upstream.
:::
