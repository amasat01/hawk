"""140_refuse_all_samples_name: a kernel shape the source says is refused (across_lanes.ipynb: cell 24)."""
# ruff: noqa: B018
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk


def all_samples_name():
    hawk.all_samples


CASE = {
    'id': '140_refuse_all_samples_name',
    'source': ['docs/content/across_lanes.ipynb: cell 24'],
    'kernels': [],
    'definers': ['all_samples_name'],
    'targets': ('host',),
    'inputs': None,
    'oracle': None,
}
