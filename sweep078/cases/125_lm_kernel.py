"""125_lm_kernel: lm_kernel (docs/content/vocabulary/20_epidemiology.ipynb: cell 15 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Scalar, Vector


@hawk.kernel
def lm_kernel(J: Matrix[35, 2], r: Vector[35], mu: Scalar, step: Mutable[Vector[2]]):
    step = m.lm_step(J, r, mu)                                    # -(J^T J + mu I)^-1 J^T r, one lane per data set


CASE = {
    "id": "125_lm_kernel",
    "source": [
        "docs/content/vocabulary/20_epidemiology.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "lm_kernel",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
