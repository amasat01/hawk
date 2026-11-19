"""047_dbl: dbl (docs/content/howto_wide_values.md: line 66 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Scalar, WideOut


@hawk.kernel
def dbl(x: Scalar, w: WideOut[Scalar]):
    w = 2.0 * x


CASE = {
    "id": "047_dbl",
    "source": [
        "docs/content/howto_wide_values.md: line 66 (markdown)",
    ],
    "kernels": [
        "dbl",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
