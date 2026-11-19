"""013_reverse: reverse (docs/content/across_lanes.ipynb: cell 18 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Grid, Index, Mutable, Param, Scalar

N = 6


@hawk.kernel
def reverse(x: Scalar, i: Index, y: Mutable[Scalar]):
    y = x.at(i) * 2.0


@hawk.kernel
def laplacian(lane: Grid["row", "col":N], u: Scalar, h2inv: Param) -> Scalar:
    i, j = lane.row, lane.col
    if m.land(m.land(i > 0, i < N - 1), m.land(j > 0, j < N - 1)):
        out = (u.at(row=i - 1, col=j) + u.at(row=i + 1, col=j)
               + u.at(row=i, col=j - 1) + u.at(row=i, col=j + 1) - 4.0 * u) * h2inv
    else:
        out = 0.0
    return out


CASE = {
    "id": "013_reverse",
    "source": [
        "docs/content/across_lanes.ipynb: cell 18 (notebook)",
    ],
    "kernels": [
        "reverse",
        "laplacian",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
