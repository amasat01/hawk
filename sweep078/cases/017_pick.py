"""017_pick: pick (docs/content/best_practices.md: line 27 (markdown))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Index, Mutable, Scalar, Table


@hawk.kernel
def pick(i: Index, tab: Table["j": 8], y: Mutable[Scalar]):
    # tab.at(i + 3) is read for every sample, taken or not; the clamp keeps it in the table
    y = tab.at(i + 3) if i < 5 else 0.0


CASE = {
    "id": "017_pick",
    "source": [
        "docs/content/best_practices.md: line 27 (markdown)",
    ],
    "kernels": [
        "pick",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
