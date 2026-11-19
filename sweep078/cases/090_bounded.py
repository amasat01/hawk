"""090_bounded: bounded (docs/content/vocabulary/05_tapes_and_events.ipynb: cell 15 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel
def bounded(dt: Param, terminated: Terminated, y: Mutable[Scalar]):
    y = y - dt
    terminated = hawk.math.land(y < 0.0, y > -1.0)


CASE = {
    "id": "090_bounded",
    "source": [
        "docs/content/vocabulary/05_tapes_and_events.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "bounded",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
