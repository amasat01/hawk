"""053_cross_product: cross_product (docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 5 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import numpy as np
from hawk import Mutable, Vector
from hawk.math import cross


@hawk.kernel
def cross_product(a: Vector[3], b: Vector[3], out: Mutable[Vector[3]]):
    out = cross(a, b)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])   # one column per sample
    b = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    return {"a": a, "b": b}


def oracle(planes):
    """Expected outputs from the formula the source states."""
    return {"result": np.cross(planes["a"].T, planes["b"].T).T}


CASE = {
    "id": "053_cross_product",
    "source": [
        "docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 5 (notebook)",
    ],
    "kernels": [
        "cross_product",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "cross_product",
    "call_form": True,
}
