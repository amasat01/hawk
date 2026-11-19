"""147_refuse_in_a_branch: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar


def in_a_branch():
    @hawk.kernel
    def k(x: Scalar, y: Mutable[Scalar]):
        if x > 0.0:
            y = hawk.sum(x)


CASE = {
    'id': '147_refuse_in_a_branch',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['in_a_branch'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
