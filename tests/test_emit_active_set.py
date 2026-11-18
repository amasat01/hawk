# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An active-set index map: ``Guard(active_set=True)``.

A kernel whose guard carries ``active_set=True`` reads two more ``lookup``
planes, ``active_map`` (the ascending live sample indices) and ``active_count``
(their number), and its index prologue maps launch position ``t`` to sample
``active_map[t]``, exiting at or past the device-read count. eagle's
``ActiveSet`` fills both (its tests run the compiled kernel against the
map-free one, bit for bit, host and device).

Rows here, on the emitted source:

* **Only the prologue changes.** The body string is the map-free kernel's,
  character for character, on both backends; the map read and the count exit
  sit in the prologue, the one geometry seam; the two planes are declared in
  the entry. A kind without the map renders its kernel unchanged, and its
  guard's repr (part of a unit's key) is the pre-map spelling.
* **Derived kernels** built under the map's kind carry the same prologue.
* **Refusals**: a map without a mask (at ``Guard``), and at build a kernel
  accumulating across samples, a kernel binding none of the guard's masks, a
  kernel using a reserved plane name and a segmented unit.

RED: render the map kernel with the ordinary prologue and the prologue rows
fail; render the map read into the body and the body-identity row fails.
"""

from __future__ import annotations

import importlib

import pytest

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk.ext import Guard, Kind
from hawk.errors import HawkError

B = importlib.import_module("hawk.artifact.bundle")

MAP = Kind("active_rows", guard=Guard(active_set=True))
MAP_READ = "reinterpret_cast<const std::int32_t*>(lut_active_map.data())[hawk_t]"
COUNT_READ = "reinterpret_cast<const std::int32_t*>(lut_active_count.data())[0]"


def _step(omega: Scalar, dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt * omega


# --------------------------------------------------------------------------- #
# -- the guard.
# --------------------------------------------------------------------------- #
def test_guard_repr_is_the_pre_map_spelling_without_the_map():
    assert repr(Guard()) == "Guard(mask='terminated', masks=())"
    assert Guard() == Guard(active_set=False)
    assert Guard(active_set=True) != Guard()
    assert "active_set=True" in repr(Guard(active_set=True))


def test_a_map_needs_a_mask():
    with pytest.raises(HawkError, match="derived from a mask"):
        Guard(mask=None, active_set=True)


# --------------------------------------------------------------------------- #
# -- the prologue.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# -- refusals at build.
# --------------------------------------------------------------------------- #


def test_refuses_a_segmented_unit(tmp_path):
    from hawk import math as M

    def seg(k: hawk.Index, bs: hawk.Vector[2], terminated: Terminated,
            y: Mutable[Scalar]):
        y = M.dispatch(k, [bs[0], bs[1]], strategy="segmented")

    kern = hawk.kernel(seg, kind=MAP)
    with pytest.raises(HawkError, match="index map in a segmented"):
        B.build_bundle([kern], tmp_path, targets=("host",), kind=MAP)


def test_a_plain_member_of_an_active_bundle_reads_no_map(tmp_path):
    """Each member keeps its own guard: a kernel traced without a kind, bundled
    with an active-set kernel, takes no ``active_map``/``active_count`` planes
    and keeps the ordinary prologue. RED: render every member under the
    bundle's kind and the plain member acquires both lookups."""
    def plain(omega: Scalar, terminated: Terminated, y: Mutable[Scalar]):
        y = omega * 2.0

    active, other = hawk.kernel(_step, kind=MAP), hawk.kernel(plain)
    bundle = B.build_bundle([active, other], tmp_path, targets=("host",))
    lookups = {a.name: {n for role, n in a.sidecar["arg_spec"] if role == "lookup"}
               for a in bundle.artifacts}
    assert lookups[active.name] == {"active_map", "active_count"}
    assert lookups["plain"] == set()
