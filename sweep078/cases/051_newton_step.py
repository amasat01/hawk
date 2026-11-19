"""051_newton_step: newton_step (docs/content/linear_algebra.md: line 76 (markdown))."""
# ruff: noqa: RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector


def jac(C, x, n):
    return m.mat([[3.0 * x[i] * x[i] + C[i, j] if i == j else C[i, j] for j in range(n)]
                  for i in range(n)])


@hawk.kernel
def newton_step(C: Matrix[22, 22], x: Vector[22], f: Vector[22], dx: Mutable[Vector[22]]):
    dx, info = m.solve(jac(C, x, 22), f)


CASE = {
    "id": "051_newton_step",
    "source": [
        "docs/content/linear_algebra.md: line 76 (markdown)",
    ],
    "kernels": [
        "newton_step",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
