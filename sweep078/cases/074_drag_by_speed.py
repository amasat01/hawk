"""074_drag_by_speed: drag_by_speed (docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 8 (notebook))."""

import hawk
import hawk.math as M
from hawk import Index, Mutable, Param, Scalar


@hawk.kernel
def drag_by_speed(dt: Param, cut: Param, slow: Index, fast: Index,
                  v: Mutable[Scalar], regime: Mutable[Index]):
    v = v + dt * M.dispatch(regime, [-0.1 * v, -0.4 * v * abs(v)], policy="auto")
    regime = M.select(abs(v) > cut, fast, slow)   # the step rewrites the class


CASE = {
    "id": "074_drag_by_speed",
    "source": [
        "docs/content/tutorials/08_choosing_a_dispatch_policy.ipynb: cell 8 (notebook)",
    ],
    "kernels": [
        "drag_by_speed",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
