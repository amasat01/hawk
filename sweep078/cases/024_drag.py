"""024_drag: drag (docs/content/embedded.md: line 19 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Vector
from hawk.math import norm


@hawk.kernel
def drag(state: Vector[3], rho: Param, out: Mutable[Vector[3]]):
    out = -rho * state * norm(state)


CASE = {
    "id": "024_drag",
    "source": [
        "docs/content/embedded.md: line 19 (markdown)",
        "docs/content/examples/05_embedded_c_target.ipynb: cell 4 (notebook)",
    ],
    "kernels": [
        "drag",
    ],
    "targets": ('c', 'host'),
    "inputs": None,
    "oracle": None,
}
