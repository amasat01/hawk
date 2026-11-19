"""132_kick: kick (docs/content/vocabulary.md: line 13 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def kick(x: Scalar, y: Scalar, vx: Scalar, vy: Scalar, dt: Param, mu: Param,
         x_next: Mutable[Scalar], y_next: Mutable[Scalar],
         vx_next: Mutable[Scalar], vy_next: Mutable[Scalar]):
    d3 = (x * x + y * y) ** 1.5
    vx_next = vx - dt * mu * x / d3
    vy_next = vy - dt * mu * y / d3
    x_next = x + dt * vx_next
    y_next = y + dt * vy_next


CASE = {
    "id": "132_kick",
    "source": [
        "docs/content/vocabulary.md: line 13 (markdown)",
    ],
    "kernels": [
        "kick",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
