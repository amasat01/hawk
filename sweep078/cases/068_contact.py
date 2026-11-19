"""068_contact: contact (docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 3 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar, Terminated


@hawk.kernel
def contact(x: Scalar, v: Scalar, k: Scalar, terminated: Terminated,
            v_new: Mutable[Scalar]):
    if x < 0.0:                    # touching: a stiff spring pushes back
        a = -9.81 - k * x
    else:                          # airborne: free flight
        a = -9.81
    v_new = v + 0.01 * a           # one explicit step of 10 ms
    terminated = x < -0.5          # pressed in too deep: stop this sample


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    return {"x": x, "v": v, "k": k, "terminated": flags, "v_new": out}


CASE = {
    "id": "068_contact",
    "source": [
        "docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "contact",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
