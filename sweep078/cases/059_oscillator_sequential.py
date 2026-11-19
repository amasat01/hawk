"""059_oscillator_sequential: oscillator_sequential (docs/content/tutorials/03_stop_when_youre_done.ipynb: cell 5 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel
def oscillator_sequential(omega: Scalar, t_end: Scalar, dt: Param,
                           terminated: Terminated, x: Mutable[Scalar],
                           v: Mutable[Scalar], t: Mutable[Scalar]):
    x = x + dt * v                        # old v
    v = v - dt * omega * omega * x        # NEW x, just written above
    t = t + dt
    terminated = t >= t_end


CASE = {
    "id": "059_oscillator_sequential",
    "source": [
        "docs/content/tutorials/03_stop_when_youre_done.ipynb: cell 5 (notebook)",
    ],
    "kernels": [
        "oscillator_sequential",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
