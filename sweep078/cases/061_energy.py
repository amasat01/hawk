"""061_energy: energy (docs/content/tutorials/05_gradients_forward.ipynb: cell 6 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Kernel, Mutable, Scalar, Vector
from hawk.diff import jvp
from hawk.math import dot


@hawk.kernel
def energy(v: Vector[3], out: Mutable[Scalar]):
    out = 0.5 * dot(v, v)


energy_jvp = Kernel("energy_jvp", jvp(energy, wrt=("v",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 500
    rng = np.random.default_rng(3)
    v = rng.normal(size=(3, n))
    dot_v = rng.normal(size=(3, n))   # the tangent direction to differentiate along
    return {"v": v, "dot_v": dot_v}


def oracle(planes):
    """Expected outputs from the formula the source states."""
    return {"result": (planes["v"] * planes["dot_v"]).sum(axis=0)}


CASE = {
    "id": "061_energy",
    "source": [
        "docs/content/tutorials/05_gradients_forward.ipynb: cell 6 (notebook)",
    ],
    "kernels": [
        "energy",
        "energy_jvp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "energy_jvp",
    "call_form": True,
    "build_groups": [
        [
            "energy",
            "energy_jvp",
        ],
    ],
}
