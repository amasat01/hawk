"""093_horizon: horizon (docs/content/vocabulary/05_tapes_and_events.ipynb: cell 21 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel(time="t", time_step="dt")
def horizon(dt: Param, T: Scalar, terminated: Terminated, t: Mutable[Scalar]):
    tn = t + dt
    t = tn
    terminated = tn > T


CASE = {
    "id": "093_horizon",
    "source": [
        "docs/content/vocabulary/05_tapes_and_events.ipynb: cell 21 (notebook)",
    ],
    "kernels": [
        "horizon",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
