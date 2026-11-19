"""063_running_total: running_total (docs/content/tutorials/05_gradients_forward.ipynb: cell 14 (notebook))."""

import hawk
from hawk import Mutable, Scalar, Terminated


@hawk.kernel
def running_total(x: Scalar, terminated: Terminated, acc: Mutable[Scalar]):
    acc = acc + x   # reads acc's launch-start value -- a recurrence across launches


CASE = {
    "id": "063_running_total",
    "source": [
        "docs/content/tutorials/05_gradients_forward.ipynb: cell 14 (notebook)",
    ],
    "kernels": [
        "running_total",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
