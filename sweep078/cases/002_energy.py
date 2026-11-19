"""002_energy: energy (README.md: line 63 (readme))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp
from hawk.math import dot


@hawk.kernel
def energy(v: Vector[3], out: Mutable[Scalar]):
    out = 0.5 * dot(v, v)


energy_vjp = Kernel("energy_vjp", vjp(energy, wrt=("v",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    v = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    e = np.zeros(4)
    return {"v": v, "out": e}


def oracle(planes):
    """Expected outputs from the formula the source states."""
    v = planes["v"]
    return {"out": 0.5 * (v * v).sum(axis=0)}


CASE = {
    "id": "002_energy",
    "source": [
        "README.md: line 63 (readme)",
        "docs/content/examples/02_autodiff_finite_difference_check.ipynb: cell 6 (notebook)",
        "docs/content/tutorials/04_gradients_backward.ipynb: cell 6 (notebook)",
    ],
    "kernels": [
        "energy",
        "energy_vjp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "energy",
    "build_groups": [
        [
            "energy",
            "energy_vjp",
        ],
    ],
}
