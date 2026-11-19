"""004_drag: drag (README.md: line 157 (readme))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Scalar, Vector
from hawk.ext import Kind, Output
from hawk.math import norm

ACCEL = Kind("drag", output=Output.returned(Vector[3], slot="acc"))


@ACCEL
def drag(velocity: Vector[3], k: Param, speed: Mutable[Scalar]):
    speed = norm(velocity)          # an ordinary named sink
    return (-k * norm(velocity)) * velocity   # the returned slot, "acc"


CASE = {
    "id": "004_drag",
    "source": [
        "README.md: line 157 (readme)",
    ],
    "kernels": [
        "drag",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
