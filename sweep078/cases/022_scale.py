"""022_scale: scale (docs/content/compile_options.ipynb: cell 7 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def scale(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


CASE = {
    "id": "022_scale",
    "source": [
        "docs/content/compile_options.ipynb: cell 7 (notebook)",
        "docs/content/compile_options.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "scale",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
    "build_options": {
        "opt_level": "O0",
        "host_profile": "portable",
    },
}
