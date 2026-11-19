"""055_triad: triad (docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Matrix, Mutable, Scalar, Vector
from hawk.math import cross, dot


@hawk.kernel
def triad(a: Vector[3], b: Vector[3], spin: Matrix[3, 3],
          perp: Mutable[Vector[3]], along: Mutable[Scalar],
          spun: Mutable[Vector[3]]):
    perp = cross(a, b)
    along = dot(a, b)
    spun = spin @ a


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])   # one column per sample
    b = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    spin = np.tile((2.0 * np.eye(3)).reshape(9, 1), (1, 2))   # see below: why (9, 2)
    return {"a": a, "b": b, "spin": spin}


CASE = {
    "id": "055_triad",
    "source": [
        "docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "triad",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "triad",
    "call_form": True,
}
