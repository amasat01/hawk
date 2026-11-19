"""049_scale: scale (docs/content/interop.md: line 29 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def scale(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


CASE = {
    "id": "049_scale",
    "source": [
        "docs/content/interop.md: line 29 (markdown)",
        "docs/content/tutorials/01_your_first_kernel.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "scale",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
