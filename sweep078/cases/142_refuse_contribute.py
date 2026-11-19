"""142_refuse_contribute: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""

import hawk
from hawk import Mutable, Scalar


def contribute():
    @hawk.kernel
    def k(x: Scalar, y: Mutable[Scalar]):
        y.contribute(x)


CASE = {
    'id': '142_refuse_contribute',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['contribute'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
