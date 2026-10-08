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
from hawk import Accum, Index, Kernel, Mutable, Param, Scalar, Terminated
from hawk.diff import jvp
from hawk.emit import scalar_mode
from hawk.emit.aether import geometry_hits
from hawk.ext import DEFAULT_KIND, Guard, Kind
from hawk.ir import HawkError

B = importlib.import_module("hawk.artifact.bundle")

MAP = Kind("active_rows", guard=Guard(active_set=True))
MAP_READ = "reinterpret_cast<const std::int32_t*>(lut_active_map.data())[hawk_t]"
COUNT_READ = "reinterpret_cast<const std::int32_t*>(lut_active_count.data())[0]"


def _step(omega: Scalar, dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt * omega


def _sources(k, kind, name=None):
    name = name or k.name
    unit = B._active_unit(k, name, kind)
    return B._sources(unit, name=name, smode=scalar_mode("float64"), kind=kind,
                      targets=("cuda", "host"), layout_sizes_override=None)


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
def test_map_free_kind_returns_the_kernel_itself():
    k = hawk.kernel(_step)
    assert B._active_unit(k, k.name, DEFAULT_KIND) is k
    assert B._active_spec(DEFAULT_KIND) is None


@pytest.mark.parametrize("target", ["cuda", "host"])
def test_only_the_prologue_changes(target):
    plain = _sources(hawk.kernel(_step), DEFAULT_KIND, name="step")[target]
    mapped = _sources(hawk.kernel(_step, kind=MAP), MAP, name="step")[target]
    assert mapped.body == plain.body, "the map must not reach the body"
    assert geometry_hits(mapped.body) == []
    for text in (MAP_READ, COUNT_READ):
        assert text in mapped.text and text not in plain.text
        assert text not in mapped.body
    assert "lut_active_map" in mapped.text and "lut_active_count" in mapped.text
    assert "hawk_t" not in plain.text


def test_device_prologue_exits_on_the_count_before_reading_the_map():
    src = _sources(hawk.kernel(_step, kind=MAP), MAP, name="step")["cuda"].text
    exit_at = src.index(f"if (hawk_t >= static_cast<long long>({COUNT_READ})) return;")
    read_at = src.index(MAP_READ)
    assert exit_at < read_at
    assert "if (hawk_flat >= count) return;" in src


def test_host_loop_stops_at_the_live_count():
    src = _sources(hawk.kernel(_step, kind=MAP), MAP, name="step")["host"].text
    assert f"const std::int64_t hawk_live = static_cast<std::int64_t>({COUNT_READ});" in src
    # the device's exit at the count is the host loop's hoisted bound, no `break`
    assert "hawk_stop = hawk_end < hawk_live ? hawk_end : hawk_live;" in src
    assert "for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t)" in src
    assert "break;" not in src
    assert src.index("hawk_live =") < src.index(MAP_READ)


def test_the_map_planes_are_declared_lookup_slots():
    unit = B._active_unit(hawk.kernel(_step, kind=MAP), "step", MAP)
    assert ("lookup", "active_map") in unit.walk.slot_of
    assert ("lookup", "active_count") in unit.walk.slot_of
    assert unit.walk.digest != hawk.kernel(_step).walk.digest


def _term(omega: Scalar, x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = omega * x * x


def test_a_derived_kernel_carries_the_map_prologue():
    primal = hawk.kernel(_term, kind=MAP)
    derived = Kernel("step_jvp", jvp(primal, wrt=("omega",)), {}, kind=MAP)
    src = _sources(derived, MAP)
    for target in ("cuda", "host"):
        assert MAP_READ in src[target].text
        assert MAP_READ not in src[target].body


# --------------------------------------------------------------------------- #
# -- refusals at build.
# --------------------------------------------------------------------------- #
def test_refuses_an_accumulating_kernel():
    def scatter(x: Scalar, lane: Index, terminated: Terminated,
                acc: Accum[Scalar]):
        acc.add(x, at=lane)

    k = hawk.kernel(scatter, kind=MAP)
    with pytest.raises(HawkError, match="accum_out"):
        B._active_unit(k, k.name, MAP)


def test_refuses_a_kernel_binding_no_mask():
    def bare(omega: Scalar, x: Mutable[Scalar]):
        x = x * omega

    k = hawk.kernel(bare, kind=MAP)
    with pytest.raises(HawkError, match="binds none"):
        B._active_unit(k, k.name, MAP)


def test_refuses_a_reserved_plane_name():
    def clash(active_map: Scalar, terminated: Terminated, x: Mutable[Scalar]):
        x = x + active_map

    k = hawk.kernel(clash, kind=MAP)
    with pytest.raises(HawkError, match="reserved"):
        B._active_unit(k, k.name, MAP)


def test_refuses_a_segmented_unit(tmp_path):
    from hawk import math as M

    def seg(k: hawk.Index, bs: hawk.Vector[2], terminated: Terminated,
            y: Mutable[Scalar]):
        y = M.dispatch(k, [bs[0], bs[1]], policy="segmented")

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
