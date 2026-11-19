"""016_scale: scale (docs/content/best_practices.md: line 11 (markdown))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Index, Mutable, Scalar, Table


@hawk.kernel
def scale(x: Scalar, k: Index, gain: Table["i": 8], y: Mutable[Scalar]):
    y = gain.at(k) * x


CASE = {
    "id": "016_scale",
    "source": [
        "docs/content/best_practices.md: line 11 (markdown)",
    ],
    "kernels": [
        "scale",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
