"""008_per_channel: per_channel (docs/content/across_lanes.ipynb: cell 8 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Grid, Scalar


@hawk.kernel
def per_channel(lane: Grid["channel", "sample":50], x: Scalar) -> Scalar:
    return hawk.sum(x * x, over="sample")


CASE = {
    "id": "008_per_channel",
    "source": [
        "docs/content/across_lanes.ipynb: cell 8 (notebook)",
    ],
    "kernels": [
        "per_channel",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
