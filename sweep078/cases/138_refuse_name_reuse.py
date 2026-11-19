"""138_refuse_name_reuse: a kernel shape the source says is refused (06_every_form_of_a_kernel.ipynb: cell 15)."""

import hawk
from hawk import Scalar


def name_reuse():
    @hawk.kernel
    def bad(x: Scalar):
        return x                                       # returning a parameter under its own name


CASE = {
    'id': '138_refuse_name_reuse',
    'source': ['docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 15'],
    'kernels': [],
    'definers': ['name_reuse'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
