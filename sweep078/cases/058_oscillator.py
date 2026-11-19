"""058_oscillator: oscillator (docs/content/tutorials/03_stop_when_youre_done.ipynb: cell 3 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel
def oscillator(omega: Scalar, t_end: Scalar, dt: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    x, v = x + dt * v, v - dt * omega * omega * x   # tuple: reads step-start x, v
    t = t + dt
    terminated = t >= t_end


CASE = {
    "id": "058_oscillator",
    "source": [
        "docs/content/tutorials/03_stop_when_youre_done.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "oscillator",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
