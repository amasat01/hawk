"""145_refuse_samples_axis: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar, Vector


def samples_axis():
    @hawk.kernel
    def k(v: Vector[3], y: Mutable[Scalar]):
        y = v.sum(axis="samples")


CASE = {
    'id': '145_refuse_samples_axis',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['samples_axis'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
