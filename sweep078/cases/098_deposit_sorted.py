"""098_deposit_sorted: deposit_sorted (docs/content/vocabulary/08_particle_binning.ipynb: cell 13 (notebook))."""

import hawk
from hawk import Index, Mutable, Scalar, Sorted
from hawk.ext import DATA_ONLY, KernelKind

NB = 16                                            # cells


class Particle(KernelKind, slug="particle"):
    x: Scalar               # position in [0, 1)
    w: Scalar               # weight
    cell: Sorted[Index]     # the cell number, for particles kept sorted by cell
    guard = DATA_ONLY


@Particle
def deposit_sorted(cell, w, mass: Mutable["bin":NB]):
    hawk.scatter(mass)[cell] += w


CASE = {
    "id": "098_deposit_sorted",
    "source": [
        "docs/content/vocabulary/08_particle_binning.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "deposit_sorted",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
