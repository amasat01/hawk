"""054_rotate_by_matrix: rotate_by_matrix (docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 9 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import numpy as np
from hawk import Matrix, Mutable, Vector


@hawk.kernel
def rotate_by_matrix(spin: Matrix[3, 3], a: Vector[3], out: Mutable[Vector[3]]):
    out = spin @ a


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])   # one column per sample
    spin = np.tile((2.0 * np.eye(3)).reshape(9, 1), (1, 2))   # see below: why (9, 2)
    return {"spin": spin, "a": a}


def oracle(planes):
    """Expected outputs from the formula the source states."""
    spin = planes["spin"].reshape(3, 3, -1)
    return {"result": np.einsum("ijn,jn->in", spin, planes["a"])}


CASE = {
    "id": "054_rotate_by_matrix",
    "source": [
        "docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 9 (notebook)",
    ],
    "kernels": [
        "rotate_by_matrix",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "rotate_by_matrix",
    "call_form": True,
}
