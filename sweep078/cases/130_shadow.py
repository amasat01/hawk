"""130_shadow: shadow (docs/content/vocabulary/24_line_of_sight.ipynb: cell 7 (notebook))."""
# ruff: noqa: B023, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import numpy as np
from hawk import Index, Mutable, Scalar, Table, Vector
from hawk.geometry import build_bvh, closest_hit


def icosphere(levels):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(levels):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    v = np.array(v)
    return v[np.array(f)]


earth = icosphere(4)

bvh = build_bvh(earth, leaf=4)


@hawk.kernel
def shadow(pos: Vector[3], sun: Vector[3], y: Mutable[Vector[5]],
           lo: Table[Scalar], hi: Table[Scalar], left: Table[Index], right: Table[Index],
           first: Table[Index], size: Table[Index], tri: Table[Scalar]):
    y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), pos, sun)


@hawk.kernel
def visible(p1: Vector[3], p2: Vector[3], y: Mutable[Vector[5]],
            lo: Table[Scalar], hi: Table[Scalar], left: Table[Index], right: Table[Index],
            first: Table[Index], size: Table[Index], tri: Table[Scalar]):
    y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), p1, p2 - p1, 1.0)


CASE = {
    "id": "130_shadow",
    "source": [
        "docs/content/vocabulary/24_line_of_sight.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "shadow",
        "visible",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
