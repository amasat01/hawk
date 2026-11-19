"""133_kick: kick (docs/content/vocabulary.md: line 28 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Vector
from hawk.ext import DATA_ONLY, KernelKind
from hawk.math import norm


class TwoBody(KernelKind, slug="two_body"):
    r: Vector[2]
    v: Vector[2]
    dt: Param
    mu: Param
    guard = DATA_ONLY


@TwoBody
def kick(r, v, dt, mu, r_next: Mutable[Vector[2]], v_next: Mutable[Vector[2]]):
    v_next = v - dt * mu * r / norm(r) ** 3
    r_next = r + dt * v_next


CASE = {
    "id": "133_kick",
    "source": [
        "docs/content/vocabulary.md: line 28 (markdown)",
    ],
    "kernels": [
        "kick",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
