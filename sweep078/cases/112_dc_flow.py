"""112_dc_flow: dc_flow (docs/content/vocabulary/16_energy.ipynb: cell 23 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Index, Kernel, Mutable, Param, Scalar, Vector
from hawk.diff import jvp
from hawk.ext import DATA_ONLY, KernelKind

LINES = [
    (1, 2, 0.01938, 0.05917, 0.0528, 1.0), (1, 5, 0.05403, 0.22304, 0.0492, 1.0), (2, 3, 0.04699, 0.19797, 0.0438, 1.0),
    (2, 4, 0.05811, 0.17632, 0.0374, 1.0), (2, 5, 0.05695, 0.17388, 0.0340, 1.0), (3, 4, 0.06701, 0.17103, 0.0346, 1.0),
    (4, 5, 0.01335, 0.04211, 0.0128, 1.0), (4, 7, 0.0, 0.20912, 0.0, 0.978), (4, 9, 0.0, 0.55618, 0.0, 0.969),
    (5, 6, 0.0, 0.25202, 0.0, 0.932), (6, 11, 0.09498, 0.19890, 0.0, 1.0), (6, 12, 0.12291, 0.25581, 0.0, 1.0),
    (6, 13, 0.06615, 0.13027, 0.0, 1.0), (7, 8, 0.0, 0.17615, 0.0, 1.0), (7, 9, 0.0, 0.11001, 0.0, 1.0),
    (9, 10, 0.03181, 0.08450, 0.0, 1.0), (9, 14, 0.12711, 0.27038, 0.0, 1.0), (10, 11, 0.08205, 0.19207, 0.0, 1.0),
    (12, 13, 0.22092, 0.19988, 0.0, 1.0), (13, 14, 0.17093, 0.34802, 0.0, 1.0),
]

LOAD_MW   = [0.0, 21.7, 94.2, 47.8, 7.6, 11.2, 0.0, 0.0, 29.5, 9.0, 3.5, 6.1, 13.5, 14.9]

GEN_MW    = [0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]       # bus 1 is the slack and finds its own

BASE_MVA = 100.0

NB, NL = len(LOAD_MW), len(LINES)


class Network(KernelKind, slug="power_grid"):
    outage: Index       # the line taken out of service (0..19), -1 for none
    kick: Index         # the bus whose injection is nudged (0..13), -1 for none
    nudge: Scalar       # the size of that nudge (MW)
    stress: Scalar      # the load level: 1 is the published load
    omega: Param        # over-relaxation of the DC sweep further down
    tol: Param          # a case has settled when no power or voltage mismatch is above this (per unit)
    guard = DATA_ONLY


def dc_sweep(th, b, kick, nudge, stress, omega):
    th = [th[i] for i in range(NB)]
    moved = 0.0
    for i in range(1, NB):
        pull, diag = 0.0, 0.0
        for k, (f, t, *_rest) in enumerate(LINES):
            if f - 1 == i: pull = pull + b[k] * th[t - 1]; diag = diag + b[k]
            if t - 1 == i: pull = pull + b[k] * th[f - 1]; diag = diag + b[k]
        p = (GEN_MW[i] - stress * LOAD_MW[i] + m.where(kick == i, nudge, 0.0)) / BASE_MVA
        live = diag > 1e-9
        new = th[i] + omega * (m.where(live, (p + pull) / m.where(live, diag, 1.0), th[i]) - th[i])
        moved = m.where(m.abs(new - th[i]) > moved, m.abs(new - th[i]), moved)
        th[i] = new
    return m.vec(*th), moved


def dc_setup(outage):
    return [m.where(outage == k, 0.0, 1.0) / (LINES[k][3] * LINES[k][5]) for k in range(NL)]    # in-service susceptance 1 / (x a)


def dc_flows(th, b):
    return m.vec(*[BASE_MVA * b[k] * (th[LINES[k][0] - 1] - th[LINES[k][1] - 1]) for k in range(NL)])


def zero_angles():
    return m.vec(*[0.0] * NB)


@Network
def dc_flow(outage, kick, nudge, stress, omega, tol, pf: Mutable[Vector[NL]], sweeps: Mutable[Scalar]):
    b = dc_setup(outage)
    th = zero_angles()
    done = 0.0
    n = 0.0
    for _ in range(3000):
        if done > 0.5:
            break
        n = n + 1.0
        th, moved = dc_sweep(th, b, kick, nudge, stress, omega)
        done = m.where(moved < tol, 1.0, 0.0)
    pf = dc_flows(th, b)
    sweeps = n


dc_flow_jvp = Kernel("dc_flow_jvp", jvp(dc_flow, wrt=("nudge",)))


CASE = {
    "id": "112_dc_flow",
    "source": [
        "docs/content/vocabulary/16_energy.ipynb: cell 23 (notebook)",
    ],
    "kernels": [
        "dc_flow",
        "dc_flow_jvp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "dc_flow",
            "dc_flow_jvp",
        ],
    ],
}
