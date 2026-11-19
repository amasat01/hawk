"""060_relax: relax (docs/content/tutorials/04_gradients_backward.ipynb: cell 15 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Kernel, Mutable, Param, Scalar
from hawk.diff import param_planes, vjp


@hawk.kernel
def relax(x: Scalar, g: Param, y: Mutable[Scalar]):
    s = x
    for _k in range(5):
        s = m.tanh(g * s + x)
    y = s


cells = param_planes(relax, per_cell=True)

cells_vjp = Kernel("relax_cells_vjp", tuple(vjp(cells, wrt=("g",))))


CASE = {
    "id": "060_relax",
    "source": [
        "docs/content/tutorials/04_gradients_backward.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "relax",
        "cells_vjp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "cells_vjp",
        ],
        [
            "relax",
        ],
    ],
}
