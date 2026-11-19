"""040_newton_step: newton_step (docs/content/howto_linear_algebra.md: line 5 (markdown))."""
# ruff: noqa: F841, RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Int32, Matrix, Mutable, Vector


@hawk.kernel
def newton_step(J: Matrix[3, 3], f: Vector[3], x: Vector[3],
                y: Mutable[Vector[3]], info: Mutable[Int32]):
    dx, info = m.solve(J, f)
    y = x - dx


CASE = {
    "id": "040_newton_step",
    "source": [
        "docs/content/howto_linear_algebra.md: line 5 (markdown)",
        "docs/content/linear_algebra.md: line 5 (markdown)",
    ],
    "kernels": [
        "newton_step",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
