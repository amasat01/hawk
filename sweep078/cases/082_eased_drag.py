"""082_eased_drag: eased_drag (docs/content/vocabulary/03_your_own_derivative.ipynb: cell 9 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Scalar, Vector
from hawk.ext import DATA_ONLY, KernelKind, primitive
from hawk.math import exp, log, norm


class Orbit(KernelKind, slug="orbit"):
    r: Vector[3]
    v: Vector[3]
    dt: Param
    g: Param
    guard = DATA_ONLY


class Perturbed(Orbit, slug="perturbed"):
    cd: Scalar


@primitive("speed_ramp",
           vjp=lambda s, bar: bar / (1.0 + exp(-s)),
           jvp=lambda s, ds: ds / (1.0 + exp(-s)))
def speed_ramp(s):
    return log(1.0 + exp(s))


@Perturbed
def eased_drag(v, cd, v_crit: Param, a_mag: Mutable[Scalar]):
    a_mag = cd * speed_ramp(norm(v) - v_crit)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    speeds = np.linspace(0.0, 40.0, 6)
    v = np.zeros((3, 6)); v[0] = speeds
    v = np.zeros((3, 6)); v[0] = speeds
    return {"v": v, "cd": np.full(6, 0.004), "v_crit": 25.0}


CASE = {
    "id": "082_eased_drag",
    "source": [
        "docs/content/vocabulary/03_your_own_derivative.ipynb: cell 9 (notebook)",
    ],
    "kernels": [
        "eased_drag",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "eased_drag",
    "call_form": True,
}
