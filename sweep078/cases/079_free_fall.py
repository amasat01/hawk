"""079_free_fall: free_fall (docs/content/vocabulary/01_a_family_of_kernels.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Vector
from hawk.ext import DATA_ONLY, KernelKind
from hawk.math import vec


class Orbit(KernelKind, slug="orbit"):
    r: Vector[3]
    v: Vector[3]
    dt: Param
    g: Param
    guard = DATA_ONLY


@Orbit
def free_fall(r, v, dt, g, r_next: Mutable[Vector[3]], v_next: Mutable[Vector[3]]):
    v_new = v + dt * vec(0.0, 0.0, -g)
    v_next = v_new
    r_next = r + dt * v_new


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 200
    r0 = np.zeros((3, n)); r0[2] = 2.0
    r0 = np.zeros((3, n)); r0[2] = 2.0
    rng = np.random.default_rng(0)
    v0 = np.zeros((3, n)); v0[0] = rng.uniform(1.0, 4.0, n)
    v0 = np.zeros((3, n)); v0[0] = rng.uniform(1.0, 4.0, n)
    r_next, v_next = np.zeros((3, n)), np.zeros((3, n))
    return {"r": r0, "v": v0, "dt": 0.2, "g": 9.81, "r_next": r_next, "v_next": v_next}


CASE = {
    "id": "079_free_fall",
    "source": [
        "docs/content/vocabulary/01_a_family_of_kernels.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "free_fall",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "free_fall",
}
