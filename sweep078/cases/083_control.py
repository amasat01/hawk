"""083_control: control (docs/content/vocabulary/03_your_own_derivative.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Kernel, Mutable, Scalar
from hawk.diff import vjp
from hawk.ext import primitive


@primitive("deliberately_wrong", vjp=lambda s, bar: 3.0 * bar)
def deliberately_wrong(s):
    return s + 0.0          # the identity -- its TRUE gradient is 1, not 3


@hawk.kernel
def control(x: Scalar, y: Mutable[Scalar]):
    y = deliberately_wrong(x)


control_vjp = Kernel("control_vjp", vjp(control, wrt=("x",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    return {"bar_y": np.array([1.0, 1.0])}


CASE = {
    "id": "083_control",
    "source": [
        "docs/content/vocabulary/03_your_own_derivative.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "control",
        "control_vjp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "control_vjp",
    "call_form": True,
    "build_groups": [
        [
            "control",
            "control_vjp",
        ],
    ],
}
