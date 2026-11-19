"""027_energy: energy (docs/content/examples/02_autodiff_finite_difference_check.ipynb: cell 4 (notebook))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar, Vector
from hawk.math import dot


@hawk.kernel
def energy(v: Vector[3], out: Mutable[Scalar]):
    out = 0.5 * dot(v, v)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    return {"v": v}


CASE = {
    "id": "027_energy",
    "source": [
        "docs/content/examples/02_autodiff_finite_difference_check.ipynb: cell 4 (notebook)",
        "docs/content/tutorials/04_gradients_backward.ipynb: cell 3 (notebook)",
        "docs/content/tutorials/05_gradients_forward.ipynb: cell 4 (notebook)",
    ],
    "kernels": [
        "energy",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
