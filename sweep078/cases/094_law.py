"""094_law: law (docs/content/vocabulary/05_tapes_and_events.ipynb: cell 23 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.math
from hawk import Mutable, Param, Scalar
from hawk.diff import param_planes, vjp


@hawk.kernel
def law(x: Scalar, k: Param, y: Mutable[Scalar]):
    y = k * x * x


law_params = param_planes(law)

law_grad = hawk.Kernel("law_grad", vjp(law_params, wrt=("x", "k")))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    x = np.array([1.0, 2.0, 3.0])
    k = np.array([2.0])          # one number, held in a plane of length 1
    k[0] = 5.0                   # same built kernel, new value
    bar_k, bar_k_comp, bar_x = np.zeros(1), np.zeros(1), np.zeros(3)
    return {"x": x, "k": k, "bar_y": np.ones(3), "bar_k": bar_k, "bar_k_comp": bar_k_comp, "bar_x": bar_x}


CASE = {
    "id": "094_law",
    "source": [
        "docs/content/vocabulary/05_tapes_and_events.ipynb: cell 23 (notebook)",
    ],
    "kernels": [
        "law",
        "law_params",
        "law_grad",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "law_grad",
    "build_groups": [
        [
            "law_params",
            "law_grad",
        ],
        [
            "law",
        ],
    ],
}
