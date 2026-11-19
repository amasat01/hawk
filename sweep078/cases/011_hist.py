"""011_hist: hist (docs/content/across_lanes.ipynb: cell 14 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Index, Mutable, Scalar


@hawk.kernel
def hist(x: Scalar, bins: Mutable["bin":8]):
    hawk.scatter(bins)[m.index(x * 8.0, 8)] += 1.0


@hawk.kernel
def put(key: Index, x: Scalar, buf: Mutable["slot":6]):
    hawk.scatter(buf)[key] = x


CASE = {
    "id": "011_hist",
    "source": [
        "docs/content/across_lanes.ipynb: cell 14 (notebook)",
    ],
    "kernels": [
        "hist",
        "put",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
