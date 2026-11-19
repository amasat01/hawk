"""029_quadratic_drag: quadratic_drag (docs/content/examples/03_kernelkind_reuse.ipynb: cell 6 (notebook))."""

from hawk import Param, Vector
from hawk.ext import KernelKind, Output
from hawk.math import norm


class Accel(KernelKind, slug="accel"):
    output = Output.returned(Vector[3], slot="acc")


@Accel
def quadratic_drag(velocity: Vector[3], k: Param):
    return (-k * norm(velocity)) * velocity


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    velocity = np.array([[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    acc_quad = np.zeros((3, 2))
    return {"velocity": velocity, "k": 0.5, "acc": acc_quad}


CASE = {
    "id": "029_quadratic_drag",
    "source": [
        "docs/content/examples/03_kernelkind_reuse.ipynb: cell 6 (notebook)",
    ],
    "kernels": [
        "quadratic_drag",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "quadratic_drag",
}
