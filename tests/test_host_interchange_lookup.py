# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel that only reads lookup planes runs its loops interchanged on the host, its wide carries in place.

A table read is a ``cross_sample_read`` of the walk, but a lookup plane is read-only and the same for every lane, so a tile of
lanes may run a loop trip by trip beside it (:func:`hawk.ir.vectorise.lane_form_access`); the access class itself is
unchanged, which is what a rank partition reads. What is pinned:

* **The predicate.** The wide-value corpus kernels that read a table interchange, with the reason string
  (:func:`hawk.ir.vectorise.vectorise_blocker`) saying why a kernel that does not does not; a loop that folds a long vector
  stays its lane's own (the fold is the vector work already).
* **The wide carry stays in its plane.** Between the tile loop's head and its tail the text holds no ``Item`` of more than
  the placement threshold: a carry is a row of the tile, written element by element, never loaded and stored whole per trip.
* **No bit moves.** The same TU built both ways (``HAWK_HOST_INTERCHANGE=0`` is the per-sample loop) returns the same bits, and
  both equal NumPy on exact data.

Every build pins the EXACT ``native`` host profile.
"""

from __future__ import annotations

import re

import _oracle as O
import _wide_kernels as W
import numpy as np
import pytest
from conftest import sidecar_of

from hawk.artifact import build_bundle
from hawk.ir.vectorise import interchanged_loops, vectorise_blocker, vectorises

ROWS = ["fill_mask", "matvec_computed", "masked_sum", "wide51"]
PLAIN = ("HAWK_HOST_INTERCHANGE=0",)


def _host_text(name):
    from hawk.artifact import bundle as B
    from hawk.emit import scalar_mode

    k = W.CORPUS[name].kernel
    out = []
    for kk, n, kd in zip(*B._expand_split([k], [k.name], [k.kind], smode=scalar_mode("float64"), device=False)[:3]):
        src = B._sources(B._active_unit(kk, n, kd), name=n, smode=scalar_mode("float64"), kind=kd, targets=("host",),
                         layout_sizes_override=None)
        out.append(src["host"].text)
    return "".join(out)


@pytest.mark.parametrize("name", ["fill_mask", "matvec_computed", "masked_sum"])
def test_a_kernel_that_only_reads_a_table_interchanges(name):
    walk = W.CORPUS[name].kernel.walk
    assert walk.access.cls == "cross_sample_read"             # the class placement legality reads is unchanged
    assert interchanged_loops(walk)


def test_a_loop_that_folds_a_long_vector_stays_its_lanes_own_and_the_gate_says_why():
    walk = W.matvec_in.walk
    assert walk.access.cls == "cross_sample_read" and interchanged_loops(walk) == ()
    assert not vectorises(walk)
    assert "no loop runs interchanged" in vectorise_blocker(walk)


@pytest.mark.parametrize("name", ROWS)
def test_the_tile_loop_holds_no_whole_wide_value(name):
    text = _host_text(name)
    lanes = text[text.index("#if HAWK_HOST_INTERCHANGE"):text.index("#else", text.index("#if HAWK_HOST_INTERCHANGE"))]
    wide = [int(w) for w in re.findall(r"Item<Real, (\d+)>", lanes)]
    assert not [w for w in wide if w > 64], f"{name}: an Item of {max(wide)} elements in the tile loop"
    if name != "masked_sum" and name != "wide51":
        assert "hawk_abi::tile_lane" in lanes                  # the wide carry is a row of the tile


@pytest.mark.parametrize("name", ["fill_mask", "matvec_computed", "wide51", "masked_sum"])
def test_both_loop_shapes_return_the_same_bits_and_numpy_s(name, tmp_path, cache_dir):
    entry = W.CORPUS[name]
    n = W.LANES_HOST
    planes = entry.planes(np.random.default_rng(3), np.float64, n, True)
    out = entry.shape(n)
    got = []
    for defines in ((), PLAIN):
        planes_run = {k: v.copy() for k, v in planes.items()}
        planes_run[entry.out] = np.zeros(out)
        k = entry.kernel
        b = build_bundle([k], tmp_path / f"d{len(defines)}", targets=("host",), cache_dir=cache_dir, defines=defines,
                         host_profile="native")
        kernel = O.load(b.directory, k.name, sidecar_of(b, k.name))
        for _role, nm in kernel.arg_spec:           # a scratch plane of a placement: a host text never touches it
            if nm.startswith("hwd"):
                planes_run[nm] = np.zeros(1)
        got.append(np.array(O.run_kernel(kernel, n, **planes_run)))
    assert np.array_equal(got[0], got[1]), "the two loop shapes differ"
    assert np.array_equal(got[0], entry.want(np.float64, planes, n)), "the result differs from NumPy's"
