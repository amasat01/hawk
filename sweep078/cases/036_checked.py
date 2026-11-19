"""036_checked: checked (docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 13 (notebook))."""

import hawk
from hawk import Scalar


@hawk.kernel
def checked(x: Scalar) -> tuple[Scalar, Scalar]:      # says: two values
    return x + 1.0, x * 3.0


CASE = {
    "id": "036_checked",
    "source": [
        "docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 13 (notebook)",
    ],
    "kernels": [
        "checked",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
