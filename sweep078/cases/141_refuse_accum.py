"""141_refuse_accum: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""
# ruff: noqa: B018
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk


def accum():
    hawk.Accum


CASE = {
    'id': '141_refuse_accum',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['accum'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
