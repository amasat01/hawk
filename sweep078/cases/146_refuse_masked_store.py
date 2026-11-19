"""146_refuse_masked_store: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""

import hawk
from hawk import Mutable, Scalar


def masked_store():
    @hawk.kernel
    def k(x: Scalar, y: Mutable[Scalar]):
        y = x
        hawk.all_samples(y)[x > 0.0] = x


CASE = {
    'id': '146_refuse_masked_store',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['masked_store'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
