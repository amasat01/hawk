"""092_target: target (docs/content/vocabulary/05_tapes_and_events.ipynb: cell 19 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math
from hawk import Mutable, Param, Scalar, Terminated
from hawk.diff import jvp, refine_entry


@hawk.kernel(time="t", time_step="dt")
def target(dt: Param, a: Param, terminated: Terminated,
           y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - 9.81 * dt
    y = y + dt * v_start
    tn = t + dt
    t = tn
    terminated = y <= a * tn      # a surface rising at speed a


refine = refine_entry(target)

slope = hawk.Kernel("target_refine_jvp", jvp(refine, wrt=("dt_tau",)))


CASE = {
    "id": "092_target",
    "source": [
        "docs/content/vocabulary/05_tapes_and_events.ipynb: cell 19 (notebook)",
    ],
    "kernels": [
        "target",
        "refine",
        "slope",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_groups": [
        [
            "target",
        ],
        [
            "refine",
            "slope",
        ],
    ],
}
