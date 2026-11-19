"""139_refuse_buffer_parameter: a kernel shape the source says is refused (06_every_form_of_a_kernel.ipynb: cell 15)."""

import hawk
from hawk import Scalar


def buffer_parameter():
    @hawk.kernel
    def bad(x: Scalar, buffer: Scalar):                # `buffer` is the graph builder's keyword
        return x + 1.0


CASE = {
    'id': '139_refuse_buffer_parameter',
    'source': ['docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 15'],
    'kernels': [],
    'definers': ['buffer_parameter'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
