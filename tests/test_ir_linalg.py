# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The linear-algebra op kinds: one factorisation node, consumers and
projections over it, the type rules, the cost formulas (operations and bytes),
and the general-shape descriptor a sidecar yields.

The formulas are the ones ``aether::linalg`` documents: ``lu`` n^3/3, ``chol``
n^3/6, ``eigh`` S(n) n(n-1)/2 (6n+10), a solve n^2 k.
"""

from __future__ import annotations

import pytest

from hawk import runtime
from hawk.errors import HawkError
































# --- cost ---------------------------------------------------------------------



















# --- the sidecar descriptor reads declared shapes ----------------------------------

def _sidecar(**kw):
    base = {"scalar_type": "float64", "mutables": [], "params": [], "arg_dtypes": {}}
    base.update(kw)
    # the slots these rows describe are the mirror-shaped ones: outputs, vectors and matrices
    base["arg_mirrors"] = {name: "GRefMirror" for _role, name in base.get("arg_spec", ())}
    return base


@pytest.mark.parametrize("shape", [(3, 3), (2, 5), (7, 7), (1, 4), (32, 32)])
def test_a_matrix_of_any_extent_gets_its_declared_kind(shape):
    sidecar = _sidecar(arg_spec=[["out", "W"], ["mat_in", "A"], ["vec_in", "v"]],
                       mat_shapes={"A": list(shape)}, vec_widths={"v": 6},
                       mutables=[{"name": "W", "dtype": "matrix", "width": shape[0] * shape[1],
                                  "shape": list(shape), "default": None}])
    assert runtime.descriptor_for_sidecar(sidecar) == ["gref", "gref", "gref"]


def test_a_vector_output_reads_its_width_and_a_missing_extent_is_refused():
    ok = _sidecar(arg_spec=[["out", "y"]],
                  mutables=[{"name": "y", "dtype": "vector", "width": 5, "default": None}])
    assert runtime.descriptor_for_sidecar(ok) == ["gref"]
    for sidecar, what in (
            (_sidecar(arg_spec=[["mat_in", "A"]]), "mat_shapes"),
            (_sidecar(arg_spec=[["vec_in", "v"]]), "vec_widths"),
            (_sidecar(arg_spec=[["out", "W"]],
                      mutables=[{"name": "W", "dtype": "matrix", "width": 6, "default": None}]),
             "mutables.shape")):
        with pytest.raises(HawkError, match=what):
            runtime.descriptor_for_sidecar(sidecar)


# --- a factorisation is never an arm ---------------------------------------------------

