"""143_refuse_add_at: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""

import hawk
from hawk import Index, Mutable, Scalar


def add_at():
    @hawk.kernel
    def k(x: Scalar, key: Index, y: Mutable[Scalar]):
        y.add(x, at=key)


CASE = {
    'id': '143_refuse_add_at',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['add_at'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
