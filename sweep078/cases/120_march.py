"""120_march: march (docs/content/vocabulary/19_solid_mechanics.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
import numpy as np
from hawk import Index, Mutable, Param, Scalar, Table, Vector
from hawk.ext import DATA_ONLY, KernelKind

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


def norm(s):                                                # the tensor norm of a symmetric tensor stored as 11 22 33 12 13 23
    return m.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))


def deviator(a):
    third = (a[0] + a[1] + a[2]) / 3.0
    return m.vec(a[0] - third, a[1] - third, a[2] - third, a[3], a[4], a[5])


def sphere(a, kappa):                                       # the pressure part of the stress: kappa tr(eps) on the diagonal
    p = kappa * (a[0] + a[1] + a[2])
    return m.vec(p, p, p, 0.0, 0.0, 0.0)


MAX_STEPS = 40_960

AMP_NP = 0.004                                              # the final e11, then the final e12 at fixed e11


@Material
def march(path, d_eps, E, nu, sigma_y, hard, steps: Table[Scalar], final: Mutable[Vector[6]]):
    mu = E / (2.0 * (1.0 + nu))
    kappa = E / (3.0 * (1.0 - 2.0 * nu))
    n = steps.at(path)                                      # this lane's number of steps
    half = n / 2.0
    ep = zeros(6)
    alpha = 0.0
    sigma = zeros(6)
    for k in range(MAX_STEPS):
        t = k + 1.0
        if t > n:
            break
        up = m.where(t <= half, t / half, 1.0)                           # first leg: e11 from 0 to AMP_NP
        across = m.where(t <= half, 0.0, (t - half) / half)              # second leg: e12 from 0 to AMP_NP at fixed e11
        last = m.where(t > n - 0.5, 1.0, 0.0)                            # the probe d_eps acts on the last step, as in `update`
        eps = m.vec(AMP_NP * up, 0.0, 0.0, AMP_NP * across, 0.0, 0.0) + last * d_eps
        s_tr = 2.0 * mu * deviator(eps - ep)
        f_tr = norm(s_tr) - SQ23 * (sigma_y + hard * alpha)
        dg = m.where(f_tr > 0.0, f_tr / (2.0 * mu + 2.0 * hard / 3.0), 0.0)
        flow = s_tr * (1.0 / m.sqrt(norm(s_tr) * norm(s_tr) + 1e-300))
        ep = ep + dg * flow
        alpha = alpha + SQ23 * dg
        sigma = s_tr - 2.0 * mu * dg * flow + sphere(eps, kappa)
    final = sigma


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    E_MOD, NU, SIGMA_Y, HARD = 200_000.0, 0.3, 250.0, 2_000.0   # MPa, -, MPa, MPa
    STEP_COUNTS = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, float(MAX_STEPS)])
    return {"path": np.arange(len(STEP_COUNTS), dtype=np.int32), "d_eps": np.zeros((6, len(STEP_COUNTS))), "E": E_MOD, "nu": NU, "sigma_y": SIGMA_Y, "hard": HARD, "steps": STEP_COUNTS, "returns": 'numpy'}


CASE = {
    "id": "120_march",
    "source": [
        "docs/content/vocabulary/19_solid_mechanics.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "march",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "march",
    "call_form": True,
}
