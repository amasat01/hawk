"""136_refuse_annotation_mismatch: a kernel shape the source says is refused (06_every_form_of_a_kernel.ipynb: cell 15)."""

import hawk
from hawk import Scalar


def annotation_mismatch():
    @hawk.kernel
    def bad(x: Scalar) -> tuple[Scalar, Scalar]:       # promises two values ...
        return x + 1.0                                 # ... returns one


CASE = {
    'id': '136_refuse_annotation_mismatch',
    'source': ['docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 15'],
    'kernels': [],
    'definers': ['annotation_mismatch'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
