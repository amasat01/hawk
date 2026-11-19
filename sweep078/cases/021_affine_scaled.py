"""021_affine_scaled: affine_scaled (docs/content/compile_cache.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Mutable, Param, Scalar


@hawk.kernel
def affine_scaled(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = 2.0 * a * x   # one constant added: a different kernel body


CASE = {
    "id": "021_affine_scaled",
    "source": [
        "docs/content/compile_cache.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "affine_scaled",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
