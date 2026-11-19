"""078_block_sums: block_sums (docs/content/tutorials/09_per_sample_control.ipynb: cell 10 (notebook))."""

import hawk
from hawk import Mutable, Scalar
from hawk.math import split_index


@hawk.kernel
def block_sums(x: Scalar, totals: Mutable[Scalar]):
    block, _place = split_index(4)
    hawk.scatter(totals)[block] += x


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 48
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0.0, 2.0 * np.pi, n)) + 0.3 * rng.normal(size=n)
    totals = np.zeros(n // 4)
    return {"x": x, "totals": totals}


def oracle(planes):
    """Expected value of the checked plane, as the source computes it."""
    x = planes["x"]
    return {"totals": x.reshape(-1, 4).sum(axis=1)}


CASE = {
    "id": "078_block_sums",
    "source": [
        "docs/content/tutorials/09_per_sample_control.ipynb: cell 10 (notebook)",
    ],
    "kernels": [
        "block_sums",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "block_sums",
    "call_form": True,
}
