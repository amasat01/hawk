"""101_matvec_in_lane: matvec_in_lane (docs/content/vocabulary/11_batched_contraction.ipynb: cell 15 (notebook))."""

import hawk
from hawk import Matrix, Vector


@hawk.kernel
def matvec_in_lane(A: Matrix[3, 3], x: Vector[3]) -> Vector[3]:
    return A @ x


CASE = {
    "id": "101_matvec_in_lane",
    "source": [
        "docs/content/vocabulary/11_batched_contraction.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "matvec_in_lane",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
