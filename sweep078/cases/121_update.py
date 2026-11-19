"""121_update: update (docs/content/vocabulary/19_solid_mechanics.ipynb: cell 15 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
import numpy as np
from hawk import Index, Kernel, Mutable, Param, Scalar, Table, Vector
from hawk.diff import jvp
from hawk.ext import DATA_ONLY, KernelKind

NSTEP = 40                                                  # strain steps per history

SQ23 = float(np.sqrt(2.0 / 3.0))


class Material(KernelKind, slug="material"):
    path: Index         # which strain history this integration point follows (a row of the strain table)
    d_eps: Vector[6]    # an extra strain added at the last step: the probe for the tangent
    E: Param            # Young's modulus (MPa)
    nu: Param           # Poisson's ratio
    sigma_y: Param      # initial yield stress (MPa)
    hard: Param         # linear isotropic hardening modulus (MPa): yield stress = sigma_y + hard * alpha
    guard = DATA_ONLY


def zeros(n):
    return m.vec(*[0.0] * n)


def strain_at(strain, path, k, d_eps):
    # the six strain components e11 e22 e33 e12 e13 e23 of step k, plus the probe at the last step
    base = (path * NSTEP + k) * 6
    last = m.where(k == NSTEP - 1, 1.0, 0.0)
    return m.vec(*[strain.at(base + j) + last * d_eps[j] for j in range(6)])


def norm(s):                                                # the tensor norm of a symmetric tensor stored as 11 22 33 12 13 23
    return m.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))


def deviator(a):
    third = (a[0] + a[1] + a[2]) / 3.0
    return m.vec(a[0] - third, a[1] - third, a[2] - third, a[3], a[4], a[5])


def sphere(a, kappa):                                       # the pressure part of the stress: kappa tr(eps) on the diagonal
    p = kappa * (a[0] + a[1] + a[2])
    return m.vec(p, p, p, 0.0, 0.0, 0.0)


@Material
def update(path, d_eps, E, nu, sigma_y, hard, strain: Table[Scalar],
           stress: Mutable[Vector[6]], eps_p: Mutable[Vector[6]], alpha_out: Mutable[Scalar],
           axial: Mutable[Vector[NSTEP]], shear: Mutable[Vector[NSTEP]],
           s_trial_last: Mutable[Vector[6]], dgamma_last: Mutable[Scalar], off_surface: Mutable[Scalar], plastic_steps: Mutable[Scalar]):
    mu = E / (2.0 * (1.0 + nu))
    kappa = E / (3.0 * (1.0 - 2.0 * nu))
    ep = zeros(6)
    alpha = 0.0
    axial = zeros(NSTEP)
    shear = zeros(NSTEP)
    s_tr = zeros(6)
    dg = 0.0
    sigma = zeros(6)
    worst = 0.0
    count = 0.0
    for k in range(NSTEP):
        eps = strain_at(strain, path, k, d_eps)
        s_tr = 2.0 * mu * deviator(eps - ep)                              # trial: the step is elastic
        f_tr = norm(s_tr) - SQ23 * (sigma_y + hard * alpha)               # trial yield function
        if f_tr > 0.0:                                                    # plastic: return radially to the surface
            dg = f_tr / (2.0 * mu + 2.0 * hard / 3.0)
            count = count + 1.0
        else:                                                             # elastic: keep the trial stress
            dg = 0.0
        flow = s_tr * (1.0 / m.sqrt(norm(s_tr) * norm(s_tr) + 1e-300))   # unit direction of the trial deviator
        ep = ep + dg * flow
        alpha = alpha + SQ23 * dg
        s = s_tr - 2.0 * mu * dg * flow
        sigma = s + sphere(eps, kappa)
        gap = m.abs(norm(s) - SQ23 * (sigma_y + hard * alpha)) / sigma_y  # distance from the yield surface after the return
        worst = m.where(dg > 0.0, m.where(gap > worst, gap, worst), worst)
        axial[k] = sigma[0]
        shear[k] = sigma[3]
    stress = sigma
    eps_p = ep
    alpha_out = alpha
    s_trial_last = s_tr
    dgamma_last = dg
    off_surface = worst
    plastic_steps = count


update_jvp = Kernel("update_jvp", jvp(update, wrt=("d_eps",)))


CASE = {
    "id": "121_update",
    "source": [
        "docs/content/vocabulary/19_solid_mechanics.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "update",
        "update_jvp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "update",
            "update_jvp",
        ],
    ],
}
