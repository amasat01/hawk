"""075_drag: drag (docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 14 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as M
from hawk import Index, Mutable, Param, Scalar


@hawk.kernel
def drag(dt: Param, regime: Index, v: Scalar, v_new: Mutable[Scalar]):
    a = M.dispatch(regime, [-0.1 * v,                 # laminar
                            -0.4 * v * abs(v),        # turbulent
                            -2.0 * v * M.exp(-v * v)], # a costly third law
                   policy="auto")
    v_new = v + dt * a


drag_split = hawk.resolve_dispatch(drag, {"dispatch0": "segmented"})


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 6
    regime = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    v = np.linspace(-1.0, 1.0, n)
    return {"dt": 0.1, "regime": regime, "v": v, "v_new": out}


CASE = {
    "id": "075_drag",
    "source": [
        "docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 14 (notebook)",
    ],
    "kernels": [
        "drag",
        "drag_split",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
