"""110_power_flow: power_flow (docs/content/vocabulary/16_energy.ipynb: cell 8 (notebook))."""
# ruff: noqa: F841, RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Index, Mutable, Param, Scalar, Vector
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

LOAD_MVAR = [0.0, 12.7, 19.0, -3.9, 1.6, 7.5, 0.0, 0.0, 16.6, 5.8, 1.8, 1.6, 5.8, 5.0]

GEN_MW    = [0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]       # bus 1 is the slack and finds its own

HELD_V    = [1.06, 1.045, 1.01, None, None, 1.07, None, 1.09, None, None, None, None, None, None]   # voltage-held buses; None = load bus

SHUNT_B   = [0.0] * 8 + [0.19] + [0.0] * 5                                                # capacitor at bus 9 (per unit)

QLIM      = [None, (-40.0, 50.0), (0.0, 40.0), None, None, (-6.0, 24.0), None, (-6.0, 24.0)] + [None] * 6   # machine Qmin, Qmax (MVAr)

BASE_MVA = 100.0

NB, NL = len(LOAD_MW), len(LINES)

GENS = [i for i in range(NB) if QLIM[i]]                    # the machine buses 2, 3, 6, 8 (counted from 0 here)

NG, NB1 = len(GENS), NB - 1

NX = 2 * NB1                                                # unknowns: an angle and a magnitude for every bus but the slack

MAXIT = 60                                                  # Newton iterations a lane may take, switches included


class Network(KernelKind, slug="power_grid"):
    outage: Index       # the line taken out of service (0..19), -1 for none
    kick: Index         # the bus whose injection is nudged (0..13), -1 for none
    nudge: Scalar       # the size of that nudge (MW)
    stress: Scalar      # the load level: 1 is the published load
    omega: Param        # over-relaxation of the DC sweep further down
    tol: Param          # a case has settled when no power or voltage mismatch is above this (per unit)
    guard = DATA_ONLY


class Cx:
    """A complex number as two traced reals. Plain Python: it runs once, when the kernel is traced."""
    def __init__(self, re, im=0.0): self.re, self.im = re, im
    def __add__(self, o): o = o if isinstance(o, Cx) else Cx(o); return Cx(self.re + o.re, self.im + o.im)
    def __sub__(self, o): o = o if isinstance(o, Cx) else Cx(o); return Cx(self.re - o.re, self.im - o.im)
    def __mul__(self, o):
        if not isinstance(o, Cx): return Cx(self.re * o, self.im * o)
        return Cx(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    def conj(self): return Cx(self.re, -self.im)
    def abs2(self): return self.re * self.re + self.im * self.im


def line_switches(outage):
    return [m.where(outage == k, 0.0, 1.0) for k in range(NL)]             # 1 = in service, 0 = this lane's outage


def admittances(on):
    # the bus admittance matrix, kept as its diagonal and one (from, to, ...) record per line; an outaged line contributes zero
    ydiag = [Cx(0.0, SHUNT_B[i]) for i in range(NB)]
    ends = []
    for k, (f, t, r, x, b, a) in enumerate(LINES):
        f, t = f - 1, t - 1
        y = 1.0 / complex(r, x)
        ff, tt, ft = (y + 0.5j * b) / a**2, y + 0.5j * b, -y / a
        ff, tt, ft = (Cx(c.real * on[k], c.imag * on[k]) for c in (ff, tt, ft))
        ydiag[f] = ydiag[f] + ff
        ydiag[t] = ydiag[t] + tt
        ends.append((f, t, ff, tt, ft))
    return ydiag, ends


def live_buses(ydiag):
    return [m.where(ydiag[i].abs2() > 1e-9, 1.0, 0.0) for i in range(NB)]    # a bus cut off from every line stays where it is


def flat_start(stress):
    # angles, magnitudes, the machines that still hold (all), their reactive set points once switched, and the mismatch record
    return (m.vec(*[0.0] * NB), m.vec(*[v or 1.0 for v in HELD_V]), m.vec(*[1.0] * NG),
            m.vec(*[-stress * LOAD_MVAR[g] / BASE_MVA for g in GENS]), m.vec(*[0.0] * 8))


def injections(th, vm, ydiag, ends):
    # the real and reactive power each bus injects into the network, from the polar voltages
    P = [ydiag[i].re * vm[i] * vm[i] for i in range(NB)]
    Q = [-ydiag[i].im * vm[i] * vm[i] for i in range(NB)]
    for f, t, ff, tt, ft in ends:
        c, s, vv = m.cos(th[f] - th[t]), m.sin(th[f] - th[t]), vm[f] * vm[t]
        P[f] = P[f] + vv * (ft.re * c + ft.im * s)
        Q[f] = Q[f] + vv * (ft.re * s - ft.im * c)
        P[t] = P[t] + vv * (ft.re * c - ft.im * s)
        Q[t] = Q[t] - vv * (ft.re * s + ft.im * c)
    return P, Q


def specified_power(kick, nudge, stress):
    p = [(GEN_MW[i] - stress * LOAD_MW[i] + m.where(kick == i, nudge, 0.0)) / BASE_MVA for i in range(NB)]
    q = [-stress * LOAD_MVAR[i] / BASE_MVA for i in range(NB)]
    return p, q


def mismatch(vm, P, Q, p_set, q_spec, q_set, pv, live):
    # the 26 equations: real power at every bus but the slack, then reactive power at the load buses and the switched machines,
    # or the held magnitude at a machine that still holds its voltage. A bus with no line gets the equation "do not move"
    f = [live[i] * (P[i] - p_set[i]) for i in range(1, NB)]
    held = {g: k for k, g in enumerate(GENS)}
    for i in range(1, NB):
        if i in held:
            k = held[i]
            f.append(live[i] * (pv[k] * (vm[i] - HELD_V[i]) + (1.0 - pv[k]) * (Q[i] - q_set[k])))
        else:
            f.append(live[i] * (Q[i] - q_spec[i]))
    return f


def jacobian(th, vm, ydiag, ends, P, Q, pv, live):
    # d(mismatch) / d(angles, magnitudes): the polar Newton Jacobian, written entry by entry
    J = [[0.0] * NX for _ in range(NX)]

    def add(r, c, v):
        J[r][c] = J[r][c] + v

    for i in range(1, NB):
        ri, qi = i - 1, NB1 + i - 1
        add(ri, ri, -Q[i] - ydiag[i].im * vm[i] * vm[i])
        add(ri, qi, P[i] / vm[i] + ydiag[i].re * vm[i])
        add(qi, ri, P[i] - ydiag[i].re * vm[i] * vm[i])
        add(qi, qi, Q[i] / vm[i] - ydiag[i].im * vm[i])
    for f, t, ff, tt, ft in ends:
        for i, j in ((f, t), (t, f)):
            if i == 0 or j == 0:                                           # the slack's angle and magnitude are fixed
                continue
            c, s, G, B = m.cos(th[i] - th[j]), m.sin(th[i] - th[j]), ft.re, ft.im
            ri, rj, qi, qj = i - 1, j - 1, NB1 + i - 1, NB1 + j - 1
            add(ri, rj, vm[i] * vm[j] * (G * s - B * c))
            add(ri, qj, vm[i] * (G * c + B * s))
            add(qi, rj, -vm[i] * vm[j] * (G * c + B * s))
            add(qi, qj, vm[i] * (G * s - B * c))
    held = {g: k for k, g in enumerate(GENS)}
    for i in range(1, NB):
        ri, qi = i - 1, NB1 + i - 1
        keep_q = live[i] * (1.0 - pv[held[i]]) if i in held else live[i]   # a machine holding its voltage has no reactive row
        for c in range(NX):
            J[ri][c] = J[ri][c] * live[i]
            J[qi][c] = J[qi][c] * keep_q
        J[ri][ri] = J[ri][ri] + (1.0 - live[i])
        J[qi][qi] = J[qi][qi] + (1.0 - keep_q)
    return m.mat(J)


def newton_step(th, vm, pv, q_set, ydiag, ends, live, kick, nudge, stress, tol):
    # one iteration of the whole rule: a Newton step, or a switch of the machines that broke a limit
    p_spec, q_spec = specified_power(kick, nudge, stress)
    P, Q = injections(th, vm, ydiag, ends)
    f = mismatch(vm, P, Q, p_spec, q_spec, q_set, pv, live)
    worst = 0.0
    for v in f:
        worst = m.where(m.abs(v) > worst, m.abs(v), worst)
    close = m.where(worst < tol, 1.0, 0.0)
    switched, q_new = [], []
    for k, g in enumerate(GENS):
        lo, hi = QLIM[g]
        q_gen = Q[g] * BASE_MVA + stress * LOAD_MVAR[g]                    # the machine's reactive output, MVAr
        above, below = m.where(q_gen > hi + 1e-6, 1.0, 0.0), m.where(q_gen < lo - 1e-6, 1.0, 0.0)
        switched.append(close * pv[k] * live[g] * m.where(above + below > 0.5, 1.0, 0.0))     # only a converged machine that still holds
        limit = m.where(above > 0.5, hi, lo)
        q_new.append(m.where(switched[k] > 0.5, (limit - stress * LOAD_MVAR[g]) / BASE_MVA, q_set[k]))
    any_switch = 0.0
    for s in switched:
        any_switch = m.where(s > any_switch, s, any_switch)
    step = 1.0 - any_switch                                                 # a switch changes the equations: no step this time
    dx, _ = m.solve(jacobian(th, vm, ydiag, ends, P, Q, pv, live), m.vec(*[-v for v in f]))
    th_new = m.vec(*[th[i] + (step * dx[i - 1] if i > 0 else 0.0) for i in range(NB)])
    vm_new = m.vec(*[vm[i] + (step * dx[NB1 + i - 1] if i > 0 else 0.0) for i in range(NB)])
    pv_new = m.vec(*[pv[k] * (1.0 - switched[k]) for k in range(NG)])
    return th_new, vm_new, pv_new, m.vec(*q_new), close * (1.0 - any_switch), worst


def record(trace, n, worst):
    return m.vec(*[m.where(m.abs(n - (j + 1.0)) < 0.5, worst, trace[j]) for j in range(8)])


def rectangular(th, vm):
    return m.vec(*[vm[i] * m.cos(th[i]) for i in range(NB)]), m.vec(*[vm[i] * m.sin(th[i]) for i in range(NB)])


def machine_q(th, vm, ydiag, ends, stress):
    P, Q = injections(th, vm, ydiag, ends)
    return m.vec(*[Q[g] * BASE_MVA + stress * LOAD_MVAR[g] for g in GENS])


def line_flows(vr, vi, ends):
    # real power at the from end and the larger apparent power of the two ends, per line, in MW and MVA; the slack's output
    V = [Cx(vr[i], vi[i]) for i in range(NB)]
    pf, sf, slack = [], [], 0.0
    for f, t, ff, tt, ft in ends:
        s_from = V[f] * (ff * V[f] + ft * V[t]).conj()
        s_to = V[t] * (tt * V[t] + ft * V[f]).conj()
        pf.append(s_from.re * BASE_MVA)
        sf.append(m.sqrt(m.where(s_from.abs2() > s_to.abs2(), s_from.abs2(), s_to.abs2())) * BASE_MVA)
        if f == 0: slack = slack + s_from.re * BASE_MVA
        if t == 0: slack = slack + s_to.re * BASE_MVA
    return m.vec(*pf), m.vec(*sf), slack


@Network
def power_flow(outage, kick, nudge, stress, omega, tol,
               vre: Mutable[Vector[NB]], vim: Mutable[Vector[NB]], pf: Mutable[Vector[NL]], sf: Mutable[Vector[NL]],
               p_slack: Mutable[Scalar], iterations: Mutable[Scalar], settled: Mutable[Scalar],
               q_machine: Mutable[Vector[NG]], holding: Mutable[Vector[NG]], trace: Mutable[Vector[8]]):
    ydiag, ends = admittances(line_switches(outage))
    live = live_buses(ydiag)
    th, vm, pv, q_set, seen = flat_start(stress)                         # pv: 1 = this machine still holds its voltage
    done = 0.0
    n = 0.0
    for _ in range(MAXIT):
        if done > 0.5:
            break
        n = n + 1.0
        th, vm, pv, q_set, done, worst = newton_step(th, vm, pv, q_set, ydiag, ends, live, kick, nudge, stress, tol)
        seen = record(seen, n, worst)
    vre, vim = rectangular(th, vm)
    pf, sf, p_slack = line_flows(vre, vim, ends)
    q_machine = machine_q(th, vm, ydiag, ends, stress)
    holding = pv
    trace = seen
    iterations = n
    settled = done


CASE = {
    "id": "110_power_flow",
    "source": [
        "docs/content/vocabulary/16_energy.ipynb: cell 8 (notebook)",
    ],
    "kernels": [
        "power_flow",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
