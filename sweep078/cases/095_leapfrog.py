"""095_leapfrog: leapfrog (docs/content/vocabulary/06_orbits.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Vector
from hawk.ext import DATA_ONLY, KernelKind
from hawk.math import norm


class Orbit(KernelKind, slug="orbit_body"):
    r: Vector[3]
    v: Vector[3]
    dt: Param
    mu: Param
    guard = DATA_ONLY


@Orbit
def leapfrog(r, v, dt, mu, r_next: Mutable[Vector[3]], v_next: Mutable[Vector[3]]):
    v_half = v + 0.5 * dt * (-mu * r / norm(r) ** 3)
    r_new = r + dt * v_half
    r_next = r_new
    v_next = v_half + 0.5 * dt * (-mu * r_new / norm(r_new) ** 3)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    mu = 1.0
    ecc = np.array([0.0, 0.2, 0.5, 0.7])
    n = len(ecc)
    r = np.zeros((3, n)); r[0] = 1.0                       # periapsis on the x axis
    r = np.zeros((3, n)); r[0] = 1.0                       # periapsis on the x axis
    v = np.zeros((3, n)); v[1] = np.sqrt(mu * (1 + ecc))   # periapsis speed
    v = np.zeros((3, n)); v[1] = np.sqrt(mu * (1 + ecc))   # periapsis speed
    dt = 2 * np.pi / 1000
    return {"r": r, "v": v, "dt": dt, "mu": mu}


CASE = {
    "id": "095_leapfrog",
    "source": [
        "docs/content/vocabulary/06_orbits.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "leapfrog",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "leapfrog",
    "call_form": True,
}
