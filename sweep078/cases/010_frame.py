"""010_frame: frame (docs/content/across_lanes.ipynb: cell 12 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Matrix, Mutable, Vector


@hawk.kernel
def frame(mm: Matrix[2, 2], gram: Mutable[Matrix[2, 2]], top: Mutable[Vector[2]], colsum: Mutable[Vector[2]]):
    gram = hawk.sum(mm)                                  # a 2x2 cell
    top = hawk.sum(mm.rows[0])                           # one row of it
    colsum = hawk.sum(mm.sum(axis="rows"))               # the value's own axis first, then the lanes


CASE = {
    "id": "010_frame",
    "source": [
        "docs/content/across_lanes.ipynb: cell 12 (notebook)",
    ],
    "kernels": [
        "frame",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
