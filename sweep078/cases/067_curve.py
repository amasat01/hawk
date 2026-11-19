"""067_curve: curve (docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 22 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
import hawk.math as m
from hawk import Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp


@hawk.kernel
def curve(x: Scalar, theta: Vector[16], y: Mutable[Scalar]):
    u = x * 15.0
    f = m.floor(u)
    i = m.index(f, 15)                 # the slot to the left of x, a runtime index
    w = u - f
    y = theta[i] * (1.0 - w) + theta[i + 1] * w


curve_vjp = Kernel("curve_vjp", vjp(curve, wrt=("theta",)))


CASE = {
    "id": "067_curve",
    "source": [
        "docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 22 (notebook)",
    ],
    "kernels": [
        "curve",
        "curve_vjp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "curve",
            "curve_vjp",
        ],
    ],
}
