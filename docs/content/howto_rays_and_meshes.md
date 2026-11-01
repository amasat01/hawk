# Cast rays against shapes and meshes

Three ray tests are `hawk.math` names, and `hawk.geometry` turns a list of triangles into a tree you can search for the nearest hit. Both run in a kernel on the CPU and on the GPU.

## One ray, one shape

Each test returns `[hit, a, b, c]`, read with `r[k]`. `hit` is `1.0` or `0.0`; the other entries are still computed on a miss, so you can select between candidates by `hit`.

| Call | Result |
|---|---|
| `m.ray_sphere(o, d, centre, r, tmax=None)` | `[hit, t0, t1, 0]`, the two roots ascending |
| `m.ray_aabb(o, d, lo, hi, tmax=None)` | `[hit, t_enter, t_exit, 0]`; `t_enter` is negative inside the box |
| `m.ray_tri(o, d, a, b, c, tmax=None)` | `[hit, t, u, v]`, two-sided; `(u, v)` are the weights of `b` and `c` |

```python
import hawk
import hawk.math as m
from hawk import Mutable, Vector

@hawk.kernel
def ball(o: Vector[3], d: Vector[3], y: Mutable[Vector[4]]):
    y = m.ray_sphere(o, d, m.vec(0.0, 0.0, 3.0), 1.0)
```

`o`, `d`, and the corners are length-3 values (or lists of three); `tmax` limits the ray. A degenerate input (a ray parallel to the triangle, a zero-area triangle, a negative discriminant) gives finite values and a derivative of exactly zero. The jump of `t` at a triangle edge or a rim is not modelled.

## Many triangles

`build_bvh` builds the tree on the host from an `(n, 3, 3)` array of vertices. `closest_hit` walks it inside a kernel. The tree is bound as seven tables, named in `hawk.geometry.PLANES`.

```python
import numpy as np
from hawk import Index, Scalar, Table
from hawk.geometry import build_bvh, closest_hit

tris = np.array([[[0, 0, 5], [4, 0, 5], [0, 4, 5]],
                 [[0, 0, 9], [4, 0, 9], [0, 4, 9]]], dtype=float)
bvh = build_bvh(tris)                       # leaf=4 triangles per leaf by default

@hawk.kernel
def nearest(o: Vector[3], d: Vector[3], tmax: Scalar, y: Mutable[Vector[5]],
            lo: Table[Scalar], hi: Table[Scalar], left: Table[Index], right: Table[Index],
            first: Table[Index], size: Table[Index], tri: Table[Scalar]):
    y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), o, d, tmax)
```

Bind `**bvh.planes()` next to your own planes (`bvh.planes(np.float32)` for a single-precision kernel):

```python
hawk.run(kernel, o=o, d=d, tmax=tmax, y=y, **bvh.planes())
```

The result `y` is `[hit, t, u, v, leaf_position]`, with zeros when nothing is hit within `tmax`. `bvh.order[leaf_position]` is the triangle's index in your input array.

- **Depth.** The traversal stack holds `depth=32` levels. A deeper tree (`bvh.depth`) is refused with the depth to pass, never walked with a short stack.
- **Same tree, same answer.** `build_bvh` is deterministic. The hit equals a brute-force test of every triangle.

The names and signatures are on the [API reference](api/index).
