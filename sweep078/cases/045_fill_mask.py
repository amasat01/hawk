"""045_fill_mask: fill_mask (docs/content/howto_wide_values.md: line 9 (markdown))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Table, Vector

W = 512


def zeros(n):                                   # a comprehension belongs in a plain helper
    return m.vec([0.0 for _ in range(n)])


def column(table, n):
    return m.vec([table.at(i) for i in range(n)])


@hawk.kernel
def fill_mask(a: Scalar, mask: Table["i": W], bf: Mutable[Vector[W]]):
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * column(mask, W)


CASE = {
    "id": "045_fill_mask",
    "source": [
        "docs/content/howto_wide_values.md: line 9 (markdown)",
    ],
    "kernels": [
        "fill_mask",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
