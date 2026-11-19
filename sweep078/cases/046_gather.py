"""046_gather: gather (docs/content/howto_wide_values.md: line 42 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Index, Mutable, Scalar, Wide


@hawk.kernel
def gather(e: Index, c: Index, edges: Wide[3], y: Mutable[Scalar]):
    y = edges.at(edge=e, component=c)


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    edges = np.arange(12, dtype=float)               # 4 edges of 3 components
    y = np.zeros(3)
    return {"n_samples": 3, "edges": edges, "y": y, "e": np.array([0, 2, 3], dtype=np.int32), "c": np.array([1, 2, 0], dtype=np.int32)}


CASE = {
    "id": "046_gather",
    "source": [
        "docs/content/howto_wide_values.md: line 42 (markdown)",
    ],
    "kernels": [
        "gather",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "gather",
}
