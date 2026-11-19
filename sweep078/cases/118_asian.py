"""118_asian: asian (docs/content/vocabulary/18_finance.ipynb: cell 17 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
import numpy as np
from hawk import Index, Kernel, Mutable, Param, Scalar, Vector
from hawk.diff import jvp
from hawk.ext import DATA_ONLY, KernelKind


class Market(KernelKind, slug="market"):
    seed: Index[np.int64]       # the path number: with a draw counter, it fixes every random number
    strike: Scalar      # strike of this option (one per lane, so a sweep is one call)
    maturity: Scalar    # time to expiry (years)
    spot: Param         # today's price of the underlying
    vol: Param          # volatility (per square-root year)
    rate: Param         # risk-free rate (per year)
    guard = DATA_ONLY


STEPS = 12                                                 # monthly fixings for a one-year option


def no_fixings():
    return m.vec(*[0.0] * STEPS)


@Market
def asian(seed, strike, maturity, spot, vol, rate, arithmetic: Mutable[Scalar], geometric: Mutable[Scalar],
          fixings: Mutable[Vector[STEPS]]):
    dt = maturity / STEPS
    drift = (rate - 0.5 * vol * vol) * dt
    spread = vol * m.sqrt(dt)
    s = spot
    total = 0.0
    log_total = 0.0
    path = no_fixings()
    for i in range(STEPS):
        s = s * m.exp(drift + spread * m.random.normal(seed, i))
        total = total + s
        log_total = log_total + m.log(s)
        path[i] = s
    disc = m.exp(-rate * maturity)
    arithmetic = disc * m.maximum(total / STEPS - strike, 0.0)
    geometric = disc * m.maximum(m.exp(log_total / STEPS) - strike, 0.0)
    fixings = path


asian_jvp = Kernel("asian_jvp", jvp(asian, wrt=("spot", "vol")))


CASE = {
    "id": "118_asian",
    "source": [
        "docs/content/vocabulary/18_finance.ipynb: cell 17 (notebook)",
    ],
    "kernels": [
        "asian",
        "asian_jvp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "asian",
            "asian_jvp",
        ],
    ],
}
