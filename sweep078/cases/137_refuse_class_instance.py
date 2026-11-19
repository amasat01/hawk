"""137_refuse_class_instance: a kernel shape the source says is refused (06_every_form_of_a_kernel.ipynb: cell 15)."""

import hawk
from hawk import Scalar


def class_instance():
    class P:
        def __repr__(self): return "P()"
    @hawk.kernel
    def bad(x: Scalar):
        return P()                                     # only values, tuples, lists and dicts


CASE = {
    'id': '137_refuse_class_instance',
    'source': ['docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 15'],
    'kernels': [],
    'definers': ['class_instance'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
