"""012_by_segment: by_segment (docs/content/across_lanes.ipynb: cell 16 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Index, Mutable, Scalar, Sorted


@hawk.kernel
def by_segment(seg: Sorted[Index], x: Scalar, total: Mutable["segment":4]):
    hawk.scatter(total)[seg] += x


CASE = {
    "id": "012_by_segment",
    "source": [
        "docs/content/across_lanes.ipynb: cell 16 (notebook)",
    ],
    "kernels": [
        "by_segment",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
