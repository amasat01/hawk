"""129_monitor: monitor (docs/content/vocabulary/23_signal_processing.ipynb: cell 13 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Index, Int32, Kernel, Matrix, Mutable, Param, Scalar, Table
from hawk.diff import vjp
from hawk.ext import DATA_ONLY, KernelKind

DT, T, K0 = 0.1, 256, 40                                         # sample time (s), samples, the bin of the tone


class Channel(KernelKind, slug="channel"):
    chan: Index         # which row of the data this lane reads
    bin_k: Scalar       # the spectral bin this lane reports
    t_stop: Scalar      # the lane uses the first t_stop samples of its channel
    guard = DATA_ONLY


@Channel
def monitor(chan, bin_k, t_stop, q: Scalar, F: Param[Matrix[2, 2]], Q0: Param[Matrix[2, 2]], P0: Param[Matrix[2, 2]], r1: Param, r2: Param,
            z: Table["row", "t":T, "s":2],
            power: Mutable[Scalar], loglik: Mutable[Scalar], nis: Mutable[Scalar], info: Mutable[Int32]):
    R = m.diag(m.vec(r1, r2))
    x = m.vec(0.0, 0.0)
    P = P0
    coeff = 2.0 * m.cos(6.283185307179586 * bin_k / 256.0)
    g1 = 0.0
    g2 = 0.0
    ll = 0.0
    chi = 0.0
    info = m.int32(0)                                                      # the sum of every factorisation's status: zero when all succeed
    for t in range(256):
        xp = F @ x                                                    # predict
        Pp = F @ P @ m.transpose(F) + q * Q0
        S = Pp + R
        L, bad = m.cholesky(S)                                        # the factor of the innovation covariance
        info = info + bad
        y = m.vec(z[chan, t, 0] - xp[0], z[chan, t, 1] - xp[1])       # the innovation
        a = m.cho_solve(L, y)                                         # S^-1 y
        quad = m.dot(y, a)
        ll = ll - 0.5 * (quad + m.log(m.det(S)) + 3.6757541328186907)
        chi = chi + quad
        u = m.where(t < t_stop, y[0] / m.sqrt(m.diagonal(S)[0]), 0.0)  # the normalised innovation, zero after the record ends
        g1, g2 = u + coeff * g1 - g2, g1                              # Goertzel
        x = xp + Pp @ a                                               # correct: K y = P S^-1 y
        P = Pp - Pp @ m.cho_solve(L, Pp)
    power = g1 * g1 + g2 * g2 - coeff * g1 * g2
    loglik = ll
    nis = chi / 512.0


monitor_vjp = Kernel("monitor_vjp", vjp(monitor, wrt=("q",)))


CASE = {
    "id": "129_monitor",
    "source": [
        "docs/content/vocabulary/23_signal_processing.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "monitor",
        "monitor_vjp",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "monitor",
            "monitor_vjp",
        ],
    ],
}
