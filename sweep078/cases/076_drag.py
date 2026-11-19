"""076_drag: drag (docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 16 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as M
from hawk import Index, Kernel, Mutable, Param, Scalar
from hawk.diff import vjp


@hawk.kernel
def drag(dt: Param, regime: Index, v: Scalar, v_new: Mutable[Scalar]):
    a = M.dispatch(regime, [-0.1 * v,                 # laminar
                            -0.4 * v * abs(v),        # turbulent
                            -2.0 * v * M.exp(-v * v)], # a costly third law
                   policy="auto")
    v_new = v + dt * a


grad_switch = Kernel("drag_vjp", vjp(drag, wrt=("v",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 6
    regime = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    v = np.linspace(-1.0, 1.0, n)
    bar_v = np.zeros(n)
    return {"dt": 0.1, "regime": regime, "v": v, "bar_v_new": np.ones(n), "bar_v": bar_v}


CASE = {
    "id": "076_drag",
    "source": [
        "docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 16 (notebook)",
    ],
    "kernels": [
        "drag",
        "grad_switch",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "drag_vjp",
    "build_groups": [
        [
            "drag",
            "grad_switch",
        ],
    ],
}
