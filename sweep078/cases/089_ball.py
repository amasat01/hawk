"""089_ball: ball (docs/content/vocabulary/05_tapes_and_events.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math
from hawk import Mutable, Param, Scalar, Terminated
from hawk.diff import jvp
from hawk.diff.event import guard_kernel


@hawk.kernel
def ball(dt: Param, terminated: Terminated,
         y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - 9.81 * dt
    y = y + dt * v_start
    tn = t + dt
    t = tn
    terminated = y < 0.0      # it hit the ground


ball_guard = guard_kernel(ball)

dg = hawk.Kernel("ball_guard_jvp", jvp(ball_guard, wrt=("y", "v")))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    dot_g = np.zeros(2)
    return {"dt": 0.05, "terminated": np.zeros(2, dtype=bool), "dot_y": np.zeros(2), "dot_v": np.ones(2), "dot_event_g0": dot_g}


CASE = {
    "id": "089_ball",
    "source": [
        "docs/content/vocabulary/05_tapes_and_events.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "ball",
        "ball_guard",
        "dg",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "ball_guard_jvp",
    "build_groups": [
        [
            "ball",
        ],
        [
            "ball_guard",
            "dg",
        ],
    ],
}
