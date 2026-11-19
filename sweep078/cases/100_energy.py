"""100_energy: energy (docs/content/vocabulary/11_batched_contraction.ipynb: cell 13 (notebook))."""
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
def energy(lane: Grid["batch", "i":N, "j":N], A, x) -> Scalar:
    xi = x.at(lane.batch * N + lane.i)
    xj = x.at(lane.batch * N + lane.j)
    return hawk.sum(xi * A.at(lane.flat) * xj, over=("i", "j"))


CASE = {
    "id": "100_energy",
    "source": [
        "docs/content/vocabulary/11_batched_contraction.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "energy",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
