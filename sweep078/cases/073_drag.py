"""073_drag: drag (docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 3 (notebook))."""
# ruff: noqa: F841
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


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 9
    regime = np.array([0, 0, 1, 2, 2, 2, 1, 0, 5], dtype=np.int32)   # 5 clamps to 2
    mix = np.zeros(3, dtype=np.uint32)       # one counter per branch
    return {"dt": 0.1, "regime": regime, "v": np.ones(n), "v_new": np.zeros(n), "dispatch0_mix": mix}


CASE = {
    "id": "073_drag",
    "source": [
        "docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "drag",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "drag",
}
