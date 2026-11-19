"""025_drag: drag (docs/content/examples/01_ensemble_drag_throughput.ipynb: cell 3 (notebook))."""

from hawk import Mutable, Param, Scalar, Vector
from hawk.ext import Kind, Output
from hawk.math import norm

ACCEL = Kind("drag", output=Output.returned(Vector[3], slot="acc"))


@ACCEL
def drag(velocity: Vector[3], k: Param, speed: Mutable[Scalar]):
    speed = norm(velocity)
    return (-k * speed) * velocity


CASE = {
    "id": "025_drag",
    "source": [
        "docs/content/examples/01_ensemble_drag_throughput.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "drag",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
