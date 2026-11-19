"""065_energy: energy (docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 14 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Scalar, Vector
from hawk.math import dot


@hawk.kernel
def energy(v: Vector[3], total: Mutable[Scalar]):
    total = hawk.sum(dot(v, v))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 8
    v = rng.normal(size=(3, n))
    total = np.zeros(1)                      # the cell: one value
    return {"v": v, "total": total}


CASE = {
    "id": "065_energy",
    "source": [
        "docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 14 (notebook)",
    ],
    "kernels": [
        "energy",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "energy",
}
