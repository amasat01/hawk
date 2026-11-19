"""080_tethered: tethered (docs/content/vocabulary/01_a_family_of_kernels.ipynb: cell 11 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Vector
from hawk.ext import DATA_ONLY, KernelKind
from hawk.math import vec


class Orbit(KernelKind, slug="orbit"):
    r: Vector[3]
    v: Vector[3]
    dt: Param
    g: Param
    guard = DATA_ONLY


@Orbit
def tethered(r, v, dt, g, k: Param,
             r_next: Mutable[Vector[3]], v_next: Mutable[Vector[3]]):
    a = vec(0.0, 0.0, -g) - k * r              # a spring pulling back to the origin
    v_new = v + dt * a
    v_next = v_new
    r_next = r + dt * v_new


CASE = {
    "id": "080_tethered",
    "source": [
        "docs/content/vocabulary/01_a_family_of_kernels.ipynb: cell 11 (notebook)",
    ],
    "kernels": [
        "tethered",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
