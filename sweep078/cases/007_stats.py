"""007_stats: stats (docs/content/across_lanes.ipynb: cell 4 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar


@hawk.kernel
def stats(x: Scalar, total: Mutable[Scalar], mean: Mutable[Scalar], positive: Mutable[Scalar]):
    total = hawk.sum(x)
    mean = hawk.mean(x)
    positive = hawk.sum(x, where=x > 0.0)


@hawk.kernel
def peak(x: Scalar, hi: Mutable[Scalar]):
    hi = hawk.max(x)


CASE = {
    "id": "007_stats",
    "source": [
        "docs/content/across_lanes.ipynb: cell 4 (notebook)",
    ],
    "kernels": [
        "stats",
        "peak",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
