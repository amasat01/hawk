"""103_ignite: ignite (docs/content/vocabulary/13_combustion.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math as m
from hawk import Mutable, Param, Scalar
from hawk.ext import DATA_ONLY, KernelKind


class Reactor(KernelKind, slug="reactor"):
    T0: Scalar          # start temperature (K)
    Y0: Scalar          # start fuel mass fraction
    A: Param            # pre-exponential factor (1/s)
    Ta: Param           # activation temperature Ea / R (K)
    heat: Param         # temperature gained per unit fuel mass fraction burned (K)
    rise: Param         # ignition: the temperature has risen by this much (K)
    dT_step: Param      # each step warms the gas by about this much (K): it sets the time step
    guard = DATA_ONLY


@Reactor
def ignite(T0, Y0, A, Ta, heat, rise, dT_step,
           tau: Mutable[Scalar], T_end: Mutable[Scalar], Y_end: Mutable[Scalar]):
    T = T0
    Y = Y0
    t = 0.0
    tau = -1.0
    for _ in range(4000):
        k = A * m.exp(-Ta / T)
        dt = dT_step / (heat * k * Y + 1e-300)                  # warms the gas by about dT_step
        dk = k * Ta / (T * T)                                   # dk / dT
        dY = -dt * k * Y / (1.0 + dt * k - dt * heat * dk * Y)
        T_next = T - heat * dY
        if T_next > T0 + rise:                                  # ignition: the crossing time, interpolated in the step
            tau = t + dt * (T0 + rise - T) / (T_next - T)
            break
        T = T_next
        Y = Y + dY
        t = t + dt
    T_end = T
    Y_end = Y


CASE = {
    "id": "103_ignite",
    "source": [
        "docs/content/vocabulary/13_combustion.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "ignite",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
