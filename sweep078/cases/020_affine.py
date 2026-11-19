"""020_affine: affine (docs/content/compile_cache.ipynb: cell 3 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Param, Scalar


@hawk.kernel
def affine(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


CASE = {
    "id": "020_affine",
    "source": [
        "docs/content/compile_cache.ipynb: cell 3 (notebook)",
    ],
    "kernels": [
        "affine",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
