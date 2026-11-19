"""057_speed: speed (docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 21 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import numpy as np
from hawk import Mutable, Scalar, Vector
from hawk.math import norm


@hawk.kernel
def speed(v: Vector[3], out: Mutable[Scalar]):
    out = norm(v)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 4
    v_component_major = np.random.default_rng(0).normal(size=(3, n))   # (3, n)
    return {"v": v_component_major}


def oracle(planes):
    """Expected outputs from the formula the source states."""
    return {"result": np.sqrt((planes["v"] ** 2).sum(axis=0))}


CASE = {
    "id": "057_speed",
    "source": [
        "docs/content/tutorials/02_arguments_and_shapes.ipynb: cell 21 (notebook)",
    ],
    "kernels": [
        "speed",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "speed",
    "call_form": True,
}
