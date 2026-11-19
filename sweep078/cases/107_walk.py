"""107_walk: walk (docs/content/vocabulary/14_nuclear.ipynb: cell 21 (notebook))."""
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
def walk(seed, width, c, sigma_t, sigma_ref, flight: Scalar,
         x_out: Mutable[Scalar], mu_out: Mutable[Scalar], fate_out: Mutable[Scalar]):
    x = 0.0
    mu = 1.0
    alive = 1.0
    x_out = 0.0
    mu_out = 1.0
    fate_out = 9.0                                                      # 9: this history had already ended
    s = 0.0
    x_new = 0.0
    mu_flown = 1.0
    fate = 0.0
    for i in range(60):
        if alive < 0.5:
            break
        s = m.random.exponential(seed, 3 * i, sigma_ref)
        x_new = x + mu * s
        mu_flown = mu
        fate = 0.0                                                      # 0: collided and flies on
        if x_new > width:                                           # 1: out through the far face
            x_new = width
            fate = 1.0
            alive = 0.0
        elif x_new < 0.0:                                           # 2: out through the near face
            x_new = 0.0
            fate = 2.0
            alive = 0.0
        elif m.random.uniform(seed, 3 * i + 1) < c:
            mu = m.random.uniform(seed, 3 * i + 2, -1.0, 1.0)
        else:                                                       # 3: absorbed
            fate = 3.0
            alive = 0.0
        if m.abs(flight - i) < 0.5:
            x_out = x_new
            mu_out = mu_flown
            fate_out = fate
        x = x_new


CASE = {
    "id": "107_walk",
    "source": [
        "docs/content/vocabulary/14_nuclear.ipynb: cell 21 (notebook)",
    ],
    "kernels": [
        "walk",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
