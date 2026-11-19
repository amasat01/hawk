"""006_curve: curve (README.md: line 196 (readme))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar, Vector


@hawk.kernel
def curve(x: Scalar, theta: Vector[16], y: Mutable[Scalar]):
    u = x * 15.0
    f = hawk.math.floor(u)
    i = hawk.math.index(f, 15)          # a runtime index
    y = theta[i] * (1.0 - (u - f)) + theta[i + 1] * (u - f)


CASE = {
    "id": "006_curve",
    "source": [
        "README.md: line 196 (readme)",
    ],
    "kernels": [
        "curve",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
