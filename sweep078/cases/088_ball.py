"""088_ball: ball (docs/content/vocabulary/05_tapes_and_events.ipynb: cell 3 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel
def ball(dt: Param, terminated: Terminated,
         y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - 9.81 * dt
    y = y + dt * v_start
    tn = t + dt
    t = tn
    terminated = y < 0.0      # it hit the ground


CASE = {
    "id": "088_ball",
    "source": [
        "docs/content/vocabulary/05_tapes_and_events.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "ball",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
