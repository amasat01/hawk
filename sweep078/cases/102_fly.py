"""102_fly: fly (docs/content/vocabulary/12_monte_carlo_ensemble.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Mutable, Param, Scalar, Terminated, Vector
from hawk.ext import KernelKind


class Projectile(KernelKind, slug="projectile"):
    drag: Param
    g: Param
    dt: Param
    xmax: Param
    terminated: Terminated
    pos: Mutable[Vector[2]]
    vel: Mutable[Vector[2]]


@Projectile
def fly(drag, g, dt, xmax, terminated, pos, vel,
        x_land: Mutable[Scalar]):
    before = pos
    vel = vel + dt * (m.vec(0.0, -g) - drag * vel)
    pos = before + dt * vel                                   # the new velocity: symplectic Euler
    y0, y1 = before.components[1], pos.components[1]
    x_hit = before.components[0] + (y0 / (y0 - y1)) * (pos.components[0] - before.components[0])
    x_land = m.where(y1 < 0.0, x_hit, x_land)                 # where the path crossed y = 0
    terminated = m.lor(y1 < 0.0, pos.components[0] > xmax)    # landed, or left the field


CASE = {
    "id": "102_fly",
    "source": [
        "docs/content/vocabulary/12_monte_carlo_ensemble.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "fly",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
