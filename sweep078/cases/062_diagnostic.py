"""062_diagnostic: diagnostic (docs/content/tutorials/05_gradients_forward.ipynb: cell 11 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Kernel, Mutable, Scalar, Terminated
from hawk.diff import jvp


@hawk.kernel
def diagnostic(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = x + 1.0


diagnostic_jvp = Kernel("diagnostic_jvp", jvp(diagnostic, wrt=("x",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    terminated = np.zeros(8, dtype=bool)
    terminated[4:] = True                 # the second half is already done
    dot_x, dot_y = np.full(8, 0.4), np.zeros(8)
    return {"terminated": terminated, "dot_x": dot_x, "dot_y": dot_y}


CASE = {
    "id": "062_diagnostic",
    "source": [
        "docs/content/tutorials/05_gradients_forward.ipynb: cell 11 (notebook)",
    ],
    "kernels": [
        "diagnostic",
        "diagnostic_jvp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "diagnostic_jvp",
    "build_groups": [
        [
            "diagnostic",
            "diagnostic_jvp",
        ],
    ],
}
