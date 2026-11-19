"""015_share: share (docs/content/across_lanes.ipynb: cell 22 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Scalar


@hawk.kernel
def share(x: Scalar, z: Mutable[Scalar]):
    z = x / hawk.sum(x)                    # reads the total: two stages


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    xs = np.random.default_rng(0).uniform(1.0, 2.0, 10)
    z = np.zeros(10)
    return {"x": xs, "z": z}


def oracle(planes):
    """Expected value of the checked plane, as the source computes it."""
    xs = planes["x"]
    return {"z": xs / xs.sum()}


CASE = {
    "id": "015_share",
    "source": [
        "docs/content/across_lanes.ipynb: cell 22 (notebook)",
    ],
    "kernels": [
        "share",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "share",
}
