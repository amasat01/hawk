"""087_flight_step: flight_step (docs/content/vocabulary/04_capstone.ipynb: cell 12 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

from hawk import Mutable, Param, Scalar, Terminated, Vector
from hawk.ext import KernelKind, primitive
from hawk.math import exp, log, norm, vec


class Orbit(KernelKind, slug="orbit_capstone"):
    terminated: Terminated
    dt: Param
    g: Param


class Perturbed(Orbit, slug="perturbed_capstone"):
    cd: Scalar


@primitive("capstone_speed_ramp",
           vjp=lambda s, bar: bar / (1.0 + exp(-s)),
           jvp=lambda s, ds: ds / (1.0 + exp(-s)))
def speed_ramp(s):
    return log(1.0 + exp(s))


@Perturbed
def flight_step(r: Vector[3], v: Vector[3], v_crit: Param, dt, g, cd, terminated,
                 r_next: Mutable[Vector[3]], v_next: Mutable[Vector[3]]):
    ramp = speed_ramp(norm(v) - v_crit)
    a = vec(0.0, 0.0, -g) - cd * ramp * v
    v_next = v + dt * a
    r_next = r + dt * v_next   # sequential: uses NEW v_next (tutorial 03's note)


CASE = {
    "id": "087_flight_step",
    "source": [
        "docs/content/vocabulary/04_capstone.ipynb: cell 12 (notebook)",
    ],
    "kernels": [
        "flight_step",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
