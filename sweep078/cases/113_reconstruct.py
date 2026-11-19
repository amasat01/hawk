"""113_reconstruct: reconstruct (docs/content/vocabulary/17_particle_physics.ipynb: cell 7 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
import numpy as np
from hawk import Index, Mutable, Param, Scalar, Table, Vector
from hawk.ext import DATA_ONLY, KernelKind

NLAY, NPHI = 10, 64                                       # layers, and azimuth cells per layer

NC = NLAY * NPHI

TWO_PI = 2 * np.pi


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


def unwrapped_zeros():
    return m.vec(*[0.0] * NLAY)


def scatter_kick(seed, i, pt, thickness, x_rad):
    # one slab's random kick: the sideways drift and the angle, drawn together from Highland's covariance
    ratio = m.where(thickness > 0.0, thickness / x_rad, 1.0)
    theta0 = m.where(thickness > 0.0, 0.0136 / pt * m.sqrt(ratio) * (1.0 + 0.038 * m.log(ratio)), 0.0)
    cov = m.mat([[theta0 * theta0 / 3.0, theta0 * theta0 / 2.0], [theta0 * theta0 / 2.0, theta0 * theta0]])      # in units of (y / t, theta)
    z = m.mvn_sample(seed, 2 * NLAY + i, m.vec(0.0, 0.0), cov)
    return thickness * z[0], z[1]


def hit_angle(seed, i, radius, phi0, charge, kappa, sigma, kick, shift, scatter):
    # azimuth of the measured hit in layer i: the helix crosses radius r at phi0 - q asin(kappa r / 2); `scatter` is the sideways offset
    r = radius.at(i)
    miss = sigma * m.random.normal(seed, 2 * i) + m.where(kick == i, shift, 0.0) + scatter
    return phi0 - charge * m.asin(0.5 * kappa * r) + miss / r


@Tracker
def reconstruct(seed, pt, charge, phi0, kick, shift, sigma, bfield, thickness, x_rad, radius: Table[Scalar], weight: Table[Scalar],
                pt_fit: Mutable[Scalar], e_track: Mutable[Scalar], phi_hit: Mutable[Vector[NLAY]],
                occupancy: Mutable["cell":NC], energy: Mutable["cell":NC]):
    kappa = 0.3 * bfield / pt                              # curvature (1/m): the radius is pt / (0.3 B)
    curve = 0.0
    e_sum = 0.0
    hits = unwrapped_zeros()
    offset = 0.0
    angle = 0.0
    behind = 0.0
    for i in range(NLAY):
        r = radius.at(i)
        offset = offset + angle * (r - behind)             # the earlier kicks carry the track sideways
        behind = r
        drift, kink = scatter_kick(seed, i, pt, thickness, x_rad)
        offset = offset + drift
        angle = angle + kink
        phi = hit_angle(seed, i, radius, phi0, charge, kappa, sigma, kick, shift, offset)
        curve = curve + weight.at(i) * r * phi             # the fit, one hit at a time
        e = m.random.exponential(seed, 2 * i + 1, 1.0)     # energy the hit leaves in the layer (arbitrary units)
        u = phi / TWO_PI
        cell = i * NPHI + m.index((u - m.floor(u)) * NPHI, NPHI)      # the azimuth wraps; the cell key must too
        hawk.scatter(occupancy)[cell] += 1.0
        hawk.scatter(energy)[cell] += e
        e_sum = e_sum + e
        hits[i] = phi
    pt_fit = 0.15 * bfield / m.abs(curve)                  # c = -q kappa / 2, so pt = 0.3 B / (2 |c|)
    e_track = e_sum
    phi_hit = hits


CASE = {
    "id": "113_reconstruct",
    "source": [
        "docs/content/vocabulary/17_particle_physics.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "reconstruct",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
