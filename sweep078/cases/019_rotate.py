"""019_rotate: rotate (docs/content/broadcasting.md: line 15 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Matrix, Mutable, Param, Vector


@hawk.kernel
def rotate(x: Vector[2], A: Param[Matrix[2, 2]], y: Mutable[Vector[2]]):
    y = A @ x                      # one A for every lane


CASE = {
    "id": "019_rotate",
    "source": [
        "docs/content/broadcasting.md: line 15 (markdown)",
    ],
    "kernels": [
        "rotate",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
