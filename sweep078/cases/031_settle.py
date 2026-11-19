"""031_settle: settle (docs/content/examples/05_embedded_c_target.ipynb: cell 30 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Scalar


@hawk.kernel
def settle(x: Scalar, y: Mutable[Scalar]):
    s = x
    for _k in range(5000):
        s = s * 0.5 + 1.0
    y = s


CASE = {
    "id": "031_settle",
    "source": [
        "docs/content/examples/05_embedded_c_target.ipynb: cell 30 (notebook)",
    ],
    "kernels": [
        "settle",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
