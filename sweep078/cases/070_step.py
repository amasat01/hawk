"""070_step: step (docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 13 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Kernel, Mutable, Param, Scalar
from hawk.diff import vjp


@hawk.kernel(jump=0.05)
def step(x: Scalar, t: Param, y: Mutable[Scalar]):
    if x < t:
        y = 1.0
    else:
        y = 0.0


step_vjp = Kernel("step_vjp", vjp(step, wrt=("t",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 400
    xs = np.random.default_rng(0).uniform(0.0, 1.0, n)
    t = 0.55
    return {"x": xs, "t": t}


CASE = {
    "id": "070_step",
    "source": [
        "docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "step",
        "step_vjp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "step",
    "call_form": True,
    "build_groups": [
        [
            "step",
            "step_vjp",
        ],
    ],
}
