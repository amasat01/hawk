"""124_sir_vjp: sir_vjp (docs/content/vocabulary/20_epidemiology.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Kernel, Mutable, Param, Scalar
from hawk.diff import vjp
from hawk.ext import DATA_ONLY, KernelKind


class Epidemic(KernelKind, slug="epidemic"):
    beta1: Scalar       # transmission rate before the measure (per day); one per lane, so a fit is one call
    beta2: Scalar       # transmission rate after it (per day)
    day: Scalar         # the reported day of this lane (1 = the first day)
    gamma: Param        # recovery rate (per day)
    i0: Param           # infectious fraction at day 0
    rho: Param          # reported share of the new infections
    t_switch: Param     # the day the measure starts
    guard = DATA_ONLY


PER_DAY, DAYS_MAX = 8, 120


def rk4(f, x, dt, *args):                         # one classical Runge-Kutta step
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def sir_rhs(x, beta, gamma):
    a = beta * x[0] * x[1]
    return m.vec(-a, a - gamma * x[1], gamma * x[1])


def make_sir(per_day):
    steps = per_day * DAYS_MAX

    @Epidemic
    def sir(beta1, beta2, day, gamma, i0, rho, t_switch,
            cases: Mutable[Scalar], s_out: Mutable[Scalar], r_out: Mutable[Scalar], r_sw: Mutable[Scalar], total: Mutable[Scalar]):
        h = 1.0 / per_day
        x = m.vec(1.0 - i0, i0, 0.0)                                      # s, i, r
        s = x[0]
        i = x[1]
        r = x[2]
        s_prev = s
        r_sw = 0.0
        beta = beta1
        for k in range(steps):
            if k + 0.5 > per_day * day:                                  # this lane's day is done
                break
            if m.abs(k - per_day * (day - 1.0)) < 0.5:                   # the start of the lane's day
                s_prev = s
            if m.abs(k - per_day * t_switch) < 0.5:                      # the measure starts
                r_sw = r
            if k + 0.5 > per_day * t_switch:
                beta = beta2
            x = rk4(sir_rhs, x, h, beta, gamma)
            s, i, r = x[0], x[1], x[2]
        cases = rho * (s_prev - s)
        s_out = s
        r_out = r
        total = s + i + r
    return sir


sir = make_sir(PER_DAY)

sir_vjp = Kernel("sir_vjp", vjp(sir, wrt=("beta1", "beta2")))


CASE = {
    "id": "124_sir_vjp",
    "source": [
        "docs/content/vocabulary/20_epidemiology.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "sir",
        "sir_vjp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "sir",
            "sir_vjp",
        ],
    ],
}
