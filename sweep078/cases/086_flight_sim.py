"""086_flight_sim: flight_sim (docs/content/vocabulary/04_capstone.ipynb: cell 10 (notebook))."""
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
def flight_sim(terminated, dt, g, cd, v_crit: Param,
               r: Mutable[Vector[3]], v: Mutable[Vector[3]]):
    ramp = speed_ramp(norm(v) - v_crit)
    a = vec(0.0, 0.0, -g) - cd * ramp * v
    v = v + dt * a
    r = r + dt * v   # sequential: uses the NEW v (tutorial 03's note)
    terminated = r[2] < 0.0


CASE = {
    "id": "086_flight_sim",
    "source": [
        "docs/content/vocabulary/04_capstone.ipynb: cell 10 (notebook)",
    ],
    "kernels": [
        "flight_sim",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
