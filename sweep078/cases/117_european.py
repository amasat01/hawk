"""117_european: european (docs/content/vocabulary/18_finance.ipynb: cell 15 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
import numpy as np
from hawk import Index, Kernel, Mutable, Param, Scalar
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


@Market
def european(seed, strike, maturity, spot, vol, rate,
             call: Mutable[Scalar], put: Mutable[Scalar], digital: Mutable[Scalar], digital_lr: Mutable[Scalar]):
    z = m.random.normal(seed, 0)                                                   # the one random number of the path
    s_end = spot * m.exp((rate - 0.5 * vol * vol) * maturity + vol * m.sqrt(maturity) * z)
    disc = m.exp(-rate * maturity)
    in_the_money = s_end > strike
    call = disc * m.maximum(s_end - strike, 0.0)
    put = disc * m.maximum(strike - s_end, 0.0)
    digital = disc * m.where(in_the_money, 1.0, 0.0)
    digital_lr = digital * z / (spot * vol * m.sqrt(maturity))                    # the likelihood-ratio estimate of the digital's delta


european_jvp = Kernel("european_jvp", jvp(european, wrt=("spot", "vol", "rate")))


CASE = {
    "id": "117_european",
    "source": [
        "docs/content/vocabulary/18_finance.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "european",
        "european_jvp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "european",
            "european_jvp",
        ],
    ],
}
