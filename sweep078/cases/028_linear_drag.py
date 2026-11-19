"""028_linear_drag: linear_drag (docs/content/examples/03_kernelkind_reuse.ipynb: cell 4 (notebook))."""

from hawk import Param, Vector
from hawk.ext import KernelKind, Output


class Accel(KernelKind, slug="accel"):
    output = Output.returned(Vector[3], slot="acc")


@Accel
def linear_drag(velocity: Vector[3], k: Param):
    return (-k) * velocity


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    velocity = np.array([[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    acc_linear = np.zeros((3, 2))
    return {"velocity": velocity, "k": 0.5, "acc": acc_linear}


CASE = {
    "id": "028_linear_drag",
    "source": [
        "docs/content/examples/03_kernelkind_reuse.ipynb: cell 4 (notebook)",
    ],
    "kernels": [
        "linear_drag",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "linear_drag",
}
