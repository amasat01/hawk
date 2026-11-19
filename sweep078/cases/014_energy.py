"""014_energy: energy (docs/content/across_lanes.ipynb: cell 20 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Scalar


@hawk.kernel
def energy(x: Scalar, total: Mutable[Scalar]):
    total = hawk.sum(x * x)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    xs = np.random.default_rng(0).normal(size=1000)
    cell = np.zeros(1)
    return {"x": xs, "total": cell}


CASE = {
    "id": "014_energy",
    "source": [
        "docs/content/across_lanes.ipynb: cell 20 (notebook)",
    ],
    "kernels": [
        "energy",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "energy",
}
