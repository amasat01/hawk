"""099_matvec: matvec (docs/content/vocabulary/11_batched_contraction.ipynb: cell 9 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Grid, Scalar, Table
from hawk.ext import DATA_ONLY, KernelKind

N = 3


class Batch(KernelKind, slug="batch"):
    A: Table[Scalar]        # the matrices, one after the other, row by row
    x: Table[Scalar]        # the vectors, one after the other
    guard = DATA_ONLY


@Batch
def matvec(lane: Grid["batch", "i":N, "j":N], A, x) -> Scalar:
    a = A.at(lane.flat)
    v = x.at(lane.batch * N + lane.j)
    return hawk.sum(a * v, over="j")


CASE = {
    "id": "099_matvec",
    "source": [
        "docs/content/vocabulary/11_batched_contraction.ipynb: cell 9 (notebook)",
    ],
    "kernels": [
        "matvec",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
