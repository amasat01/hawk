"""026_drag_mc: drag_mc (docs/content/examples/01_ensemble_drag_throughput.ipynb: cell 9 (notebook))."""

import hawk
from hawk import Mutable, Param, Scalar, Vector
from hawk.ext import Kind, Output
from hawk.math import norm, random_normal, vec
from hawk.trace import decl

Seed = decl.Plane("uniform", hawk.Index)   # one integer seed for the whole launch

MC_DRAG = Kind("drag_mc", output=Output.returned(Vector[3], slot="acc"))


@MC_DRAG
def drag_mc(seed: Seed, k: Param, speed: Mutable[Scalar]):
    velocity = vec(random_normal(seed, 0), random_normal(seed, 1), random_normal(seed, 2))
    speed = norm(velocity)
    return (-k * speed) * velocity


CASE = {
    "id": "026_drag_mc",
    "source": [
        "docs/content/examples/01_ensemble_drag_throughput.ipynb: cell 9 (notebook)",
    ],
    "kernels": [
        "drag_mc",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
