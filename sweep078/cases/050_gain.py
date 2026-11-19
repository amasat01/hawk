"""050_gain: gain (docs/content/linear_algebra.md: line 52 (markdown))."""
# ruff: noqa: F841, RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector


@hawk.kernel
def gain(L: Matrix[3, 3], r: Vector[3], k: Mutable[Vector[3]], info: Mutable[hawk.Int32]):
    factor, info = m.chol_from(L)
    k = m.cho_solve(factor, r)


CASE = {
    "id": "050_gain",
    "source": [
        "docs/content/linear_algebra.md: line 52 (markdown)",
    ],
    "kernels": [
        "gain",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
