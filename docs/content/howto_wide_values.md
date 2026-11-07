# Work with wide values

A wide value is a vector with hundreds of components per lane. hawk places it for you; `Wide` and `WideOut` are the planes for wide data that is bigger than a lane.

## A wide vector in a kernel

Write the loop the natural way:

```python
import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Table, Vector

W = 512

def zeros(n):                                   # a comprehension belongs in a plain helper
    return m.vec([0.0 for _ in range(n)])

def column(table, n):
    return m.vec([table.at(i) for i in range(n)])

@hawk.kernel
def fill_mask(a: Scalar, mask: Table["i": W], bf: Mutable[Vector[W]]):
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * column(mask, W)
```

A vector used in a loop (`bf[j]`) is read from memory where it is indexed, not built in registers. A value used whole is placed by its size against the device's registers and local memory, so a wide kernel no longer spills because it was written naturally. There is nothing to set: the placement follows the device a kernel is built for, and the [spill report](emit_form) shows the result.

`hawk.ir.recognize(kernel)` reads a sum of products over one loop (a matrix-vector product, a masked sum, several outputs sharing an operand, with `bf[j]` or table reads as operands) as a plain record of operands, reduce axis and outputs, for tools that route such a sum elsewhere. It only reads: it never changes the code a kernel generates.

A very large source is usually a wide kernel unrolled; see [the compile guard](compiling).

## A wide input: `Wide[W]`

`Wide[W]` is a per-edge table of `W` components whose number of edges is set when you run, not when you build. Read it with `at(edge=, component=)`; the bound array holds the edges one after another, `W` numbers each.

```python
from hawk import Index, Wide

@hawk.kernel
def gather(e: Index, c: Index, edges: Wide[3], y: Mutable[Scalar]):
    y = edges.at(edge=e, component=c)
```

```python
import numpy as np

edges = np.arange(12, dtype=float)               # 4 edges of 3 components
y = np.zeros(3)
hawk.run(hawk.load(directory, "gather"), n_samples=3, edges=edges, y=y,
         e=np.array([0, 2, 3], dtype=np.int32), c=np.array([1, 2, 0], dtype=np.int32))
# y == [1., 8., 9.]   (edges[e * 3 + c])
```

`directory` is where `hawk.artifact.build` put the kernel. Because the length is not part of the build, one build serves any number of edges. A `Table` is the better choice when the data has named axes; `Wide` is for a flat `edge x component` block.

## A wide output: `WideOut`

`WideOut[Scalar]` is an output plane whose length is also set at run time. Assign it as you would a `Mutable`:

```python
from hawk import WideOut

@hawk.kernel
def dbl(x: Scalar, w: WideOut[Scalar]):
    w = 2.0 * x
```

Each lane writes its own element. To write an element chosen at run time instead, store it by key:

```python
from hawk import Index

@hawk.kernel
def put(x: Scalar, k: Index, w: WideOut[Scalar]):
    hawk.scatter(w)[k] = 2.0 * x
```

Two lanes storing to the same key leave one of the two values. The payload is a scalar: a vector payload is refused when the kernel is built, with the reason. For an output of several components per lane, use `Mutable[Vector[W]]`.

Which form a wide per-lane value takes on a device is chosen from a table of measured rows, `wide.json`; where a chain of kernels is cut is chosen from `fusion.json`, beside it. Both are plain JSON, with a row per device.

The names are on the [API reference](api/index).
