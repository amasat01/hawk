"""044_nearest: nearest (docs/content/howto_rays_and_meshes.md: line 31 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import numpy as np
from hawk import Index, Mutable, Scalar, Table, Vector
from hawk.geometry import build_bvh, closest_hit

tris = np.array([[[0, 0, 5], [4, 0, 5], [0, 4, 5]],
                 [[0, 0, 9], [4, 0, 9], [0, 4, 9]]], dtype=float)

bvh = build_bvh(tris)                       # leaf=4 triangles per leaf by default


@hawk.kernel
def nearest(o: Vector[3], d: Vector[3], tmax: Scalar, y: Mutable[Vector[5]],
            lo: Table[Scalar], hi: Table[Scalar], left: Table[Index], right: Table[Index],
            first: Table[Index], size: Table[Index], tri: Table[Scalar]):
    y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), o, d, tmax)


CASE = {
    "id": "044_nearest",
    "source": [
        "docs/content/howto_rays_and_meshes.md: line 31 (markdown)",
    ],
    "kernels": [
        "nearest",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
