"""030_gather: gather (docs/content/examples/05_embedded_c_target.ipynb: cell 26 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Index, Mutable, Scalar, Table


@hawk.kernel
def gather(table: Table[Scalar], where: Index, y: Mutable[Scalar]):
    y = table.at(where)


CASE = {
    "id": "030_gather",
    "source": [
        "docs/content/examples/05_embedded_c_target.ipynb: cell 26 (notebook)",
        "docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 5 (notebook)",
    ],
    "kernels": [
        "gather",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
