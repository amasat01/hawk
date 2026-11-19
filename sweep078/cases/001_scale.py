"""001_scale: scale (README.md: line 39 (readme))."""

import hawk
import numpy as np
from hawk import Param, Scalar


@hawk.kernel
def scale(x: Scalar, a: Param, b: Param) -> Scalar:
    return a * x + b


def oracle(planes):
    """Expected outputs from the formula the source states."""
    return {"result": planes["a"] * planes["x"] + planes["b"]}


def inputs():
    """Planes of the run the source shows."""
    x = np.linspace(0.0, 7.0, 8)
    return {"x": x, "a": 2.0, "b": 1.0}


CASE = {
    "id": "001_scale",
    "source": [
        "README.md: line 39 (readme)",
        "docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 5 (notebook)",
        "docs/content/quickstart.md: line 12 (markdown)",
    ],
    "kernels": [
        "scale",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "scale",
    "call_form": True,
}
