"""126_cost_vjp: cost_vjp (docs/content/vocabulary/21_control.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Kernel, Mutable, Param, Scalar
from hawk.diff import vjp
from hawk.ext import DATA_ONLY, KernelKind


def rk4(f, x, dt, *args):                         # one classical Runge-Kutta step
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def plant_rhs(x, kp, kd, c, inertia, w):          # x = (e, v, cost so far, energy dissipated so far)
    e, v = x[0], x[1]
    u = kp * e - kd * v                           # the control effort
    return m.vec(-v, (kp * e - c * v) / inertia, e * e + w * u * u, c * v * v)


class Plant(KernelKind, slug="plant"):
    kp: Scalar          # proportional gain, one per lane
    kd: Scalar          # derivative gain, one per lane
    inertia: Param      # J (kg m^2)
    friction: Param     # b (N m s / rad)
    guard = DATA_ONLY


DT, HORIZON, N_GAINS = 0.01, 30.0, 48

T_COST, W_EFFORT = 15.0, 0.01


def make_cost(dt, horizon):
    steps = round(horizon / dt)

    @Plant
    def plant_cost(kp, kd, inertia, friction, w: Param, cost: Mutable[Scalar], lost: Mutable[Scalar], e_end: Mutable[Scalar]):
        c = kd + friction
        x = m.vec(1.0, 0.0, 0.0, 0.0)                                   # e, v, the cost so far, the energy dissipated so far
        for _ in range(steps):
            x = rk4(plant_rhs, x, dt, kp, kd, c, inertia, w)
        cost, lost = x[2], x[3]
        e_end = 0.5 * kp * x[0] * x[0] + 0.5 * inertia * x[1] * x[1]
    return plant_cost


plant_cost = make_cost(DT, T_COST)

cost_vjp = Kernel("plant_cost_vjp", vjp(plant_cost, wrt=("kp", "kd")))


CASE = {
    "id": "126_cost_vjp",
    "source": [
        "docs/content/vocabulary/21_control.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "plant_cost",
        "cost_vjp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "plant_cost",
            "cost_vjp",
        ],
    ],
}
