"""037_hawk_step: hawk_step (docs/content/examples/nvidia_warp_side_by_side.ipynb: cell 10 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk.math import maximum


@hawk.kernel
def hawk_step(x: Scalar, y: Scalar, vx: Scalar, vy: Scalar, g: Param, dt: Param,
              terminated: Terminated, x_out: Mutable[Scalar], y_out: Mutable[Scalar],
              vx_out: Mutable[Scalar], vy_out: Mutable[Scalar]):
    new_vy = vy - g * dt
    new_y = maximum(y + new_vy * dt, 0.0)   # clamp at the ground, Warp's guard too
    x_out = x + vx * dt
    y_out = new_y
    vx_out = vx
    vy_out = new_vy


CASE = {
    "id": "037_hawk_step",
    "source": [
        "docs/content/examples/nvidia_warp_side_by_side.ipynb: cell 10 (notebook)",
    ],
    "kernels": [
        "hawk_step",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
