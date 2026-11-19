"""033_draw: draw (docs/content/examples/05_embedded_c_target.ipynb: cell 34 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
import hawk.math as hm
from hawk import Index, Mutable, Scalar


@hawk.kernel
def draw(seed: Index, counter: Index, y: Mutable[Scalar]):
    y = hm.random.uniform(seed, counter)


CASE = {
    "id": "033_draw",
    "source": [
        "docs/content/examples/05_embedded_c_target.ipynb: cell 34 (notebook)",
    ],
    "kernels": [
        "draw",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
