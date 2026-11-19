"""096_where_am_i: where_am_i (docs/content/vocabulary/07_fem_assembly.ipynb: cell 15 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Grid, Index


@hawk.kernel
def where_am_i(lane: Grid["e", "a":3]) -> Index:
    return lane.e * 10 + lane[-1]


@hawk.kernel
def flat_number(lane: Grid["e", "a":3]) -> Index:
    return lane.flat


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    return {"e": 2, "returns": 'numpy'}


CASE = {
    "id": "096_where_am_i",
    "source": [
        "docs/content/vocabulary/07_fem_assembly.ipynb: cell 15 (notebook)",
    ],
    "kernels": [
        "where_am_i",
        "flat_number",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "where_am_i",
    "call_form": True,
}
