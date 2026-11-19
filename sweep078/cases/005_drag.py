"""005_drag: drag (README.md: line 175 (readme))."""

from hawk import Param, Vector
from hawk.ext import KernelKind, Output
from hawk.math import norm


class Accel(KernelKind, slug="drag"):
    output = Output.returned(Vector[3], slot="acc")


@Accel
def drag(velocity: Vector[3], k: Param):
    return (-k * norm(velocity)) * velocity


CASE = {
    "id": "005_drag",
    "source": [
        "README.md: line 175 (readme)",
    ],
    "kernels": [
        "drag",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
