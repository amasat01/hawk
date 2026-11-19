"""066_running_max: running_max (docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 18 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Scalar, Terminated
from hawk.math import maximum


@hawk.kernel
def running_max(value: Scalar, terminated: Terminated, best: Mutable[Scalar]):
    best = maximum(best, value)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 8
    best = np.zeros(n)
    terminated = np.zeros(n, dtype=bool)
    return {"value": np.full(n, value), "terminated": terminated, "best": best}


CASE = {
    "id": "066_running_max",
    "source": [
        "docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 18 (notebook)",
    ],
    "kernels": [
        "running_max",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
