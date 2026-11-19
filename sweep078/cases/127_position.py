"""127_position: position (docs/content/vocabulary/21_control.ipynb: cell 17 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Mutable, Param, Scalar
from hawk.ext import DATA_ONLY, KernelKind


def rk4(f, x, dt, *args):                         # one classical Runge-Kutta step
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def settle_rhs(x, kp, c, inertia):                # x = (e, v)
    return m.vec(-x[1], (kp * x[0] - c * x[1]) / inertia)


class Plant(KernelKind, slug="plant"):
    kp: Scalar          # proportional gain, one per lane
    kd: Scalar          # derivative gain, one per lane
    inertia: Param      # J (kg m^2)
    friction: Param     # b (N m s / rad)
    guard = DATA_ONLY


N_STEPS = 200


@Plant
def position(kp, kd, inertia, friction, t_obs: Scalar, theta: Mutable[Scalar]):
    c = kd + friction
    dt = t_obs / N_STEPS
    x = m.vec(1.0, 0.0)
    for _ in range(N_STEPS):
        x = rk4(settle_rhs, x, dt, kp, c, inertia)
    theta = 1.0 - x[0]


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    INERTIA, FRICTION, BAND = 1.0, 0.1, 0.02
    T_COST, W_EFFORT = 15.0, 0.01
    def closed_cost(kp, kd, w=W_EFFORT):
        # x0^T P x0, with A^T P + P A = -Q, to infinite time; x = (e, v)
        kp, kd = np.broadcast_arrays(np.asarray(kp, dtype=float), np.asarray(kd, dtype=float))
        a12, a21, a22 = -1.0, kp / INERTIA, -(kd + FRICTION) / INERTIA
        q11, q12, q22 = 1.0 + w * kp**2, -w * kp * kd, w * kd**2
        M = np.zeros(kp.shape + (3, 3))
        M[..., 0, 0], M[..., 0, 1] = 0.0, 2 * a21                      # row for p11: 2 (a11 p11 + a21 p12), a11 = 0
        M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = a12, a22, a21        # row for p12: a12 p11 + (a11 + a22) p12 + a21 p22
        M[..., 2, 1], M[..., 2, 2] = 2 * a12, 2 * a22                  # row for p22: 2 (a12 p12 + a22 p22)
        rhs = -np.stack([q11, q12, q22], axis=-1)
        return np.linalg.solve(M, rhs[..., None])[..., 0][..., 0]
    fine_p, fine_d = np.geomspace(0.5, 40, 400), np.geomspace(0.2, 12, 400)
    surface = closed_cost(fine_p[:, None], fine_d[None, :])
    a, b = np.unravel_index(surface.argmin(), surface.shape)
    for _ in range(4):                                              # refine around the smallest value
        fine_p = np.geomspace(fine_p[max(a - 2, 0)], fine_p[min(a + 2, len(fine_p) - 1)], 201)
        fine_d = np.geomspace(fine_d[max(b - 2, 0)], fine_d[min(b + 2, len(fine_d) - 1)], 201)
        surface = closed_cost(fine_p[:, None], fine_d[None, :])
        a, b = np.unravel_index(surface.argmin(), surface.shape)
    opt_p, opt_d = fine_p[a], fine_d[b]
    shown = {"too soft": (1.0, 1.0), "too lively": (20.0, 0.6), "over-damped": (6.0, 12.0), "tuned": (float(opt_p), float(opt_d))}
    times = np.linspace(0.0, 8.0, 160)
    return {"kp": np.repeat([k for k, _ in shown.values()], len(times)), "kd": np.repeat([d for _, d in shown.values()], len(times)), "inertia": INERTIA, "friction": FRICTION, "t_obs": np.tile(times, len(shown))}


CASE = {
    "id": "127_position",
    "source": [
        "docs/content/vocabulary/21_control.ipynb: cell 17 (notebook)",
    ],
    "kernels": [
        "position",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "position",
    "call_form": True,
}
