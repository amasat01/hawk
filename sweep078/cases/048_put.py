"""048_put: put (docs/content/howto_wide_values.md: line 76 (markdown))."""

import hawk
from hawk import Index, Scalar, WideOut


@hawk.kernel
def put(x: Scalar, k: Index, w: WideOut[Scalar]):
    hawk.scatter(w)[k] = 2.0 * x


CASE = {
    "id": "048_put",
    "source": [
        "docs/content/howto_wide_values.md: line 76 (markdown)",
    ],
    "kernels": [
        "put",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
