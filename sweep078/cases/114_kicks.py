"""114_kicks: kicks (docs/content/vocabulary/17_particle_physics.ipynb: cell 13 (notebook))."""
# ruff: noqa: RUF059
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
import numpy as np
from hawk import Index, Mutable, Param, Scalar
from hawk.ext import DATA_ONLY, KernelKind

NLAY, NPHI = 10, 64                                       # layers, and azimuth cells per layer


class Tracker(KernelKind, slug="tracker"):
    seed: Index[np.int64]       # the track number: with a draw counter, it fixes every random number
    pt: Scalar          # transverse momentum (GeV)
    charge: Scalar      # +1 or -1
    phi0: Scalar        # azimuth the track leaves the beam line at (rad)
    kick: Index         # the layer whose hit is moved (0..9), -1 for none
    shift: Scalar       # by how much (m): the alignment error
    sigma: Param        # hit resolution (m)
    bfield: Param       # solenoid field (T)
    thickness: Param    # thickness of each layer's material (m); 0 switches multiple scattering off
    x_rad: Param        # radiation length of that material (m)
    guard = DATA_ONLY


def scatter_kick(seed, i, pt, thickness, x_rad):
    # one slab's random kick: the sideways drift and the angle, drawn together from Highland's covariance
    ratio = m.where(thickness > 0.0, thickness / x_rad, 1.0)
    theta0 = m.where(thickness > 0.0, 0.0136 / pt * m.sqrt(ratio) * (1.0 + 0.038 * m.log(ratio)), 0.0)
    cov = m.mat([[theta0 * theta0 / 3.0, theta0 * theta0 / 2.0], [theta0 * theta0 / 2.0, theta0 * theta0]])      # in units of (y / t, theta)
    z = m.mvn_sample(seed, 2 * NLAY + i, m.vec(0.0, 0.0), cov)
    return thickness * z[0], z[1]


@Tracker
def kicks(seed, pt, charge, phi0, kick, shift, sigma, bfield, thickness, x_rad, drift: Mutable[Scalar], kink: Mutable[Scalar]):
    drift, kink = scatter_kick(seed, 0, pt, thickness, x_rad)


CASE = {
    "id": "114_kicks",
    "source": [
        "docs/content/vocabulary/17_particle_physics.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "kicks",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
