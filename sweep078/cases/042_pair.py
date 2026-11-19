"""042_pair: pair (docs/content/howto_random_draws.md: line 27 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Index, Matrix, Mutable, Vector


@hawk.kernel
def pair(seed: Index, step: Index, L: Matrix[2, 2], w: Mutable[Vector[2]]):
    w = m.random.multivariate_normal(seed, step, m.vec(1.0, 2.0), L)


CASE = {
    "id": "042_pair",
    "source": [
        "docs/content/howto_random_draws.md: line 27 (markdown)",
    ],
    "kernels": [
        "pair",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
