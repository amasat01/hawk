"""032_root: root (docs/content/examples/05_embedded_c_target.ipynb: cell 32 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Scalar
from hawk.math import sqrt


@hawk.kernel
def root(x: Scalar, y: Mutable[Scalar]):
    y = sqrt(x)


CASE = {
    "id": "032_root",
    "source": [
        "docs/content/examples/05_embedded_c_target.ipynb: cell 32 (notebook)",
    ],
    "kernels": [
        "root",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
