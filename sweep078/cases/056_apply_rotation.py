"""056_apply_rotation: apply_rotation (docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 17 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Quat, Vector
from hawk.math import quat_rotate


@hawk.kernel
def apply_rotation(q: Quat, v: Vector[3], out: Mutable[Vector[3]]):
    out = quat_rotate(q, v)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    q = np.array([[np.cos(np.pi / 4)], [0.0], [0.0], [np.sin(np.pi / 4)]])
    v = np.array([[1.0], [0.0], [0.0]])
    return {"q": q, "v": v}


CASE = {
    "id": "056_apply_rotation",
    "source": [
        "docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 17 (notebook)",
    ],
    "kernels": [
        "apply_rotation",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "apply_rotation",
    "call_form": True,
}
