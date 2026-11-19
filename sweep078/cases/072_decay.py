"""072_decay: decay (docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 18 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Kernel, Mutable, Scalar
from hawk.diff import vjp


@hawk.kernel
def decay(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    t = x
    for _k in range(20):
        s = s * 1.1 + g * t
        if s > lim:
            break                      # in the middle: some work is done before it
        t = m.tanh(t) + 0.1
    y = s + t


decay_vjp = Kernel("decay_vjp", vjp(decay, wrt=("x", "g")))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    dx = np.array([0.1, 0.5, 1.0, 2.0])
    dg, dlim = np.full(4, 0.3), np.full(4, 3.0)
    return {"x": dx, "g": dg, "lim": dlim, "bar_y": np.ones(4)}


CASE = {
    "id": "072_decay",
    "source": [
        "docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 18 (notebook)",
    ],
    "kernels": [
        "decay",
        "decay_vjp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "decay_vjp",
    "call_form": True,
    "build_groups": [
        [
            "decay",
            "decay_vjp",
        ],
    ],
}
