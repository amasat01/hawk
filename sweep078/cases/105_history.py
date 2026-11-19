"""105_history: history (docs/content/vocabulary/14_nuclear.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
import numpy as np
from hawk import Index, Mutable, Param, Scalar
from hawk.ext import DATA_ONLY, KernelKind


class Slab(KernelKind, slug="slab"):
    seed: Index[np.int64]       # the history number: with a draw counter, it fixes every random number
    width: Scalar       # slab thickness (cm); one per lane, so a sweep is one call
    c: Param            # scattering ratio sigma_s / sigma_t
    sigma_t: Param      # total cross-section the tally is evaluated at (1/cm)
    sigma_ref: Param    # total cross-section the flights are drawn from (1/cm)
    guard = DATA_ONLY


@Slab
def history(seed, width, c, sigma_t, sigma_ref,
            through: Mutable[Scalar], back: Mutable[Scalar], absorbed: Mutable[Scalar], unfinished: Mutable[Scalar]):
    x = 0.0
    mu = 1.0
    alive = 1.0
    log_w = 0.0
    through = 0.0
    back = 0.0
    absorbed = 0.0
    s = 0.0
    x_new = 0.0
    dist = 0.0
    for i in range(120):
        if alive < 0.5:                                                 # a finished history leaves the loop
            break
        s = m.random.exponential(seed, 3 * i, sigma_ref)            # free flight
        x_new = x + mu * s
        if x_new > width:                                           # leaves through the far face
            dist = (width - x) / mu
            log_w = log_w - (sigma_t - sigma_ref) * dist
            through = m.exp(log_w)
            alive = 0.0
        elif x_new < 0.0:                                           # leaves back through the near face
            dist = x / (-mu)
            log_w = log_w - (sigma_t - sigma_ref) * dist
            back = m.exp(log_w)
            alive = 0.0
        else:                                                       # collides at x_new
            x = x_new
            log_w = log_w + m.log(sigma_t / sigma_ref) - (sigma_t - sigma_ref) * s
            if m.random.uniform(seed, 3 * i + 1) < c:               # scatters: a new direction
                mu = m.random.uniform(seed, 3 * i + 2, -1.0, 1.0)
            else:                                                   # absorbed
                absorbed = m.exp(log_w)
                alive = 0.0
    unfinished = alive


CASE = {
    "id": "105_history",
    "source": [
        "docs/content/vocabulary/14_nuclear.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "history",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
