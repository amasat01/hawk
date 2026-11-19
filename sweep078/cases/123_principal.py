"""123_principal: principal (docs/content/vocabulary/19_solid_mechanics.ipynb: cell 19 (notebook))."""
# ruff: noqa: F821, F841, RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Int32, Matrix, Mutable, Vector


@hawk.kernel
def principal(S: Matrix[3, 3], w: Mutable[Vector[3]], info: Mutable[Int32]):
    values, vectors, info = m.eigh(S)
    w = values


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    bar_steps = np.array([20.0, 40.0, 80.0, 160.0, 320.0])
    nb_lanes = len(bar_steps)
    return {"S": np.ascontiguousarray(tensors.reshape(nb_lanes, 9).T), "returns": 'numpy'}


CASE = {
    "id": "123_principal",
    "source": [
        "docs/content/vocabulary/19_solid_mechanics.ipynb: cell 19 (notebook)",
    ],
    "kernels": [
        "principal",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "needs": [
        "eagle",
    ],
}
