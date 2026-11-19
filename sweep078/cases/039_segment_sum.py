"""039_segment_sum: segment_sum (docs/content/howto_index_set.md: line 68 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Index, Mutable, Scalar, Table


@hawk.kernel
def segment_sum(lo: Index, hi: Index, t: Table[Scalar], y: Mutable[Scalar]):
    s = 0.0
    for k in range(hi - lo, max=8):
        s = s + t.at(lo + k)
    y = s


CASE = {
    "id": "039_segment_sum",
    "source": [
        "docs/content/howto_index_set.md: line 68 (markdown)",
    ],
    "kernels": [
        "segment_sum",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
