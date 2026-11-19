"""023_drag: drag (docs/content/contributing/add_a_target.md: line 35 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def drag(v: Scalar, k: Param, a: Mutable[Scalar]):
    a = -k * v


CASE = {
    "id": "023_drag",
    "source": [
        "docs/content/contributing/add_a_target.md: line 35 (markdown)",
    ],
    "kernels": [
        "drag",
    ],
    "targets": ('toy',),
    "inputs": None,
    "oracle": None,
}
