"""144_refuse_own_sum: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar


def own_sum():
    @hawk.kernel
    def k(x: Scalar, y: Mutable[Scalar]):
        y = x.sum()


CASE = {
    'id': '144_refuse_own_sum',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['own_sum'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
