"""038_miss: miss (docs/content/howto_index_set.md: line 5 (markdown))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
import numpy as np
from hawk import Grid, Mutable, Table
from hawk.ext import IndexSet, Kind

SCREEN = Kind("screen", lanes=IndexSet())


@SCREEN
def miss(lane: Grid["pair"], pos: Table["node"], d: Mutable[hawk.Scalar], rs: Mutable["node"]):
    i, j = m.pair_index()
    t = (pos.at(i) - pos.at(j)) ** 2
    d = t                              # written at the pair's own id
    hawk.scatter(rs)[j] += t           # a keyed sum over the listed pairs only


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    return {"pos": pos, "d": d, "rs": rs, "pair": n_pairs, "active_map": np.array([3, 17, 44], dtype=np.int32), "active_count": np.array([3], dtype=np.uint32)}


CASE = {
    "id": "038_miss",
    "source": [
        "docs/content/howto_index_set.md: line 5 (markdown)",
    ],
    "kernels": [
        "miss",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
