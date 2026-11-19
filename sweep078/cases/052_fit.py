"""052_fit: fit (docs/content/linear_algebra.md: line 94 (markdown))."""
# ruff: noqa: RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Int32, Matrix, Mutable, Vector


@hawk.kernel
def fit(J: Matrix[71, 71], r: Vector[71], x: Mutable[Vector[71]], info: Mutable[Int32]):
    x, info = m.solve(J, r)            # 71 by 71 per lane: a team per matrix


CASE = {
    "id": "052_fit",
    "source": [
        "docs/content/linear_algebra.md: line 94 (markdown)",
    ],
    "kernels": [
        "fit",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
