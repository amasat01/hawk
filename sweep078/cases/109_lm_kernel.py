"""109_lm_kernel: lm_kernel (docs/content/vocabulary/15_biochemistry.ipynb: cell 15 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Scalar, Vector


@hawk.kernel
def lm_kernel(J: Matrix[12, 2], r: Vector[12], mu: Scalar, step: Mutable[Vector[2]]):
    step = m.lm_step(J, r, mu)                                       # -(J^T J + mu I)^-1 J^T r, one lane per experiment


CASE = {
    "id": "109_lm_kernel",
    "source": [
        "docs/content/vocabulary/15_biochemistry.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "lm_kernel",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
