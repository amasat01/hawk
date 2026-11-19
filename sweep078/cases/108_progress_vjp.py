"""108_progress_vjp: progress_vjp (docs/content/vocabulary/15_biochemistry.ipynb: cell 13 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Grid, Kernel, Mutable, Scalar
from hawk.diff import vjp
from hawk.ext import DATA_ONLY, KernelKind


class Enzyme(KernelKind, slug="enzyme"):
    S0: Scalar          # starting substrate (µM)
    Vmax: Scalar        # top rate (µM / min)
    Km: Scalar          # half-saturation substrate (µM)
    t_obs: Scalar       # time of this measurement (min)
    guard = DATA_ONLY


M = 12


def rk4(f, x, dt, *args):                         # one classical Runge-Kutta step
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rhs(x, Vmax, Km):                             # x = (substrate, product)
    v = Vmax * x[0] / (Km + x[0])
    return m.vec(-v, v)


def make_progress(nsteps):
    @Enzyme
    def progress(lane: Grid["v", "t":M], S0, Vmax, Km, t_obs, S: Mutable[Scalar], P: Mutable[Scalar]):
        h = t_obs / nsteps
        x = m.vec(S0, 0.0)
        for _ in range(nsteps):
            x = rk4(rhs, x, h, Vmax, Km)
        S, P = x[0], x[1]
    return progress


progress = make_progress(32)

progress_vjp = Kernel("progress_vjp", vjp(progress, wrt=("Vmax", "Km")))


CASE = {
    "id": "108_progress_vjp",
    "source": [
        "docs/content/vocabulary/15_biochemistry.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "progress",
        "progress_vjp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "progress",
            "progress_vjp",
        ],
    ],
}
