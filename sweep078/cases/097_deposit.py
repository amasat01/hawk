"""097_deposit: deposit (docs/content/vocabulary/08_particle_binning.ipynb: cell 9 (notebook))."""

import hawk
import hawk.math as m
from hawk import Index, Mutable, Scalar, Sorted
from hawk.ext import DATA_ONLY, KernelKind

NB = 16                                            # cells


class Particle(KernelKind, slug="particle"):
    x: Scalar               # position in [0, 1)
    w: Scalar               # weight
    cell: Sorted[Index]     # the cell number, for particles kept sorted by cell
    guard = DATA_ONLY


@Particle
def deposit(x, w, mass: Mutable["bin":NB], count: Mutable["bin":NB]):
    cell = m.index(x * NB, NB)
    hawk.scatter(mass, policy="compensated")[cell] += w
    hawk.scatter(count)[cell] += 1.0


CASE = {
    "id": "097_deposit",
    "source": [
        "docs/content/vocabulary/08_particle_binning.ipynb: cell 9 (notebook)",
    ],
    "kernels": [
        "deposit",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
