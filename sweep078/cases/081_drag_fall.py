"""081_drag_fall: drag_fall (docs/content/vocabulary/02_extend_by_inheritance.ipynb: cell 8 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Scalar, Vector
from hawk.ext import DATA_ONLY, KernelKind
from hawk.math import norm, vec


class Orbit(KernelKind, slug="orbit"):
    r: Vector[3]
    v: Vector[3]
    dt: Param
    g: Param
    guard = DATA_ONLY


class Perturbed(Orbit, slug="perturbed"):
    """Adds ONE entry, `cd` -- a per-sample drag coefficient."""
    cd: Scalar


@Perturbed
def drag_fall(r, v, dt, g, cd, r_next: Mutable[Vector[3]], v_next: Mutable[Vector[3]]):
    a = vec(0.0, 0.0, -g) - cd * norm(v) * v
    v_new = v + dt * a
    v_next = v_new
    r_next = r + dt * v_new


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 4
    r0 = np.zeros((3, n)); r0[2] = 2.0
    r0 = np.zeros((3, n)); r0[2] = 2.0
    v0 = np.zeros((3, n)); v0[0] = 3.0
    v0 = np.zeros((3, n)); v0[0] = 3.0
    return {"r": r0, "v": v0, "dt": 0.1, "g": 9.81, "cd": np.array([0.0, 0.01, 0.05, 0.1])}


CASE = {
    "id": "081_drag_fall",
    "source": [
        "docs/content/vocabulary/02_extend_by_inheritance.ipynb: cell 8 (notebook)",
    ],
    "kernels": [
        "drag_fall",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "drag_fall",
    "call_form": True,
}
