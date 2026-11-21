# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An active-set automatic kernel's fast-path entries agree with the map-free ones, on the device.

The device unit of an automatic kernel traced under an active-set kind carries, beside the map entry, the two entries
the map-free automatic kernel has, indexing samples directly: ``<name>_range`` and ``<name>_persist``. Both compile only
under ``HAWK_FAST_ENTRIES`` (a build's ``defines``). This file keeps the compiled agreement row: on the RK4 oscillator
and the RK7(8) attempt kernel (N = 1 ... 10k; a budget below the stops, and every sample terminated at launch) the
``_range`` and ``_persist`` entries leave the step counts, the mask and the counter equal to the map-free kernel's entries
and to a run of the map entry over the identity map, and every float plane within the math libraries' rounding of them.
Every launch reports its exact step count in ``hawk_steps``. The entries run with a decoy map bound, so a map read shows.
The codegen rows of the old file (entry names and parameter lists in the text) were rows of the deleted renderer."""

from __future__ import annotations

import ctypes

import _deploy as L
import numpy as np
import pytest
from _team_stages import needs_hawk
from conftest import sidecar_of
from test_host_interchange import _mixed, _planes, oscillator, rkf78_attempt
from test_persist import _close

import hawk
from hawk.artifact import build_bundle
from hawk.ext import Guard, Kind

DT = 0.01
KERNELS = {"oscillator": oscillator, "rkf78_attempt": rkf78_attempt}
ACTIVE = Kind("fast_entries_active", guard=Guard(active_set=True))
#: the macro that compiles the two fast entries, and the suffix of the range entry (the library writes both).
FAST_ENTRIES_MACRO = "HAWK_FAST_ENTRIES"
RANGE_SUFFIX = "_range"
FAST = (f"{FAST_ENTRIES_MACRO}=1",)


# --------------------------------------------------------------------------- #
# -- compiled: within rounding (counts exact).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("fast_entries")
    plain = build_bundle(list(KERNELS.values()), root / "plain", targets=("cuda", "host"),
                         cache_dir=cache_dir, defines=FAST)
    active = build_bundle(list(KERNELS.values()), root / "active", targets=("cuda", "host"),
                          kind=ACTIVE, cache_dir=cache_dir, defines=FAST)
    return plain, active


def _bind(bundle, name, planes, *, mapped):
    """``(bound plan, device planes, counter, word, n)``; ``mapped``
    binds the identity map (every sample, ascending) and its count, or
    (``"decoy"``) a map of sample 0 alone, which an entry that read it would
    show."""
    import cupy as cp
    sidecar = sidecar_of(bundle, name)
    loaded = L.device_loaded(bundle.directory, name, sidecar)
    dev = {key: cp.asarray(value) for key, value in planes.items()}
    n = planes["terminated"].shape[0]
    counter = cp.zeros(1, dtype=cp.uint32)
    counter[0] = int(np.count_nonzero(planes["terminated"]))
    word = cp.zeros(1, dtype=cp.int64)
    kw = {"dt": DT} if "dt" in dict((n_, r) for r, n_ in sidecar["arg_spec"]) else {}
    if mapped == "decoy":
        # a map and count no fast entry may read: only sample 0, once
        kw["active_map"] = cp.zeros(n, dtype=cp.int32)
        kw["active_count"] = cp.ones(1, dtype=cp.uint32)
    elif mapped:
        kw["active_map"] = cp.arange(n, dtype=cp.int32)
        kw["active_count"] = cp.full(1, n, dtype=cp.uint32)
    bound = hawk.plan(loaded, hawk.DeviceKernel).bind(finished_count=counter, fused_steps=word, **kw, **dev)
    return bound, dev, counter, word, n


def _snap(dev, counter):
    return {**{key: value.get() for key, value in dev.items()}, "counter": counter.get()}


def _launch(bound, entry, n, grid, extra=()):
    """Launch the named entry of ``bound``'s unit over ``grid`` blocks, with the range ``n`` and the ``extra`` arguments."""
    needs_hawk("bound.launch_entry(entry: str, *, n: int, grid: int, extra: tuple = ()) -> None, "
               "which launches a named entry of the unit the plan is bound over")


def _own(bundle, name, planes, budget, *, mapped=False, k_max=None):
    """The kernel's own entry: ONE launch of word ``budget``, or (``k_max``)
    launches of ``min(k_max, left)`` until every sample finished."""
    bound, dev, counter, word, n = _bind(bundle, name, planes, mapped=mapped)
    left = budget
    while left > 0 and int(counter[0]) != n:
        word[0] = left if k_max is None else min(k_max, left)
        bound.launch()
        bound.wait()
        left -= int(word[0])
        if k_max is None:
            break
    return _snap(dev, counter)


def _steps_of(name, planes):
    """Each sample's step count so far, off its own planes."""
    if name == "oscillator":
        return np.asarray(planes["k"], dtype=np.int64)
    return np.asarray(planes["n_acc"] + planes["n_rej"], dtype=np.int64)


def _reported(name, before, after, cell, total):
    """The launch's ``hawk_steps`` and ``hawk_stepsum`` reports, checked
    against the largest and the summed per-sample step counts the planes
    show this launch took."""
    taken = _steps_of(name, after) - _steps_of(name, before)
    want = int(taken.max()) if taken.size else 0
    got = int(cell.get()[0])
    assert got == want, f"hawk_steps {got} != the longest sample's {want}"
    assert int(total.get()[0]) == int(taken.sum()), "hawk_stepsum != the steps taken"
    return got


def _range(bundle, name, planes, budget, *, mapped="decoy"):
    import cupy as cp

    bound, dev, counter, word, n = _bind(bundle, name, planes, mapped=mapped)
    counter[0] = 0  # the entry counts the samples finished on entry itself
    word[0] = budget
    steps = cp.zeros(1, dtype=cp.uint32)
    total = cp.zeros(1, dtype=cp.uint64)
    _launch(bound, f"{name}{RANGE_SUFFIX}", n, (n + 255) // 256,
            (ctypes.c_void_p(steps.data.ptr), ctypes.c_void_p(total.data.ptr)))
    out = _snap(dev, counter)
    _reported(name, planes, out, steps, total)
    return out


def _persist(bundle, name, planes, budget, grid, *, mapped):
    import cupy as cp

    bound, dev, counter, word, n = _bind(bundle, name, planes, mapped=mapped)
    counter[0] = 0  # the entry counts the samples finished on entry itself
    word[0] = budget
    nxt = cp.zeros(1, dtype=cp.uint32)
    steps = cp.zeros(1, dtype=cp.uint32)
    total = cp.zeros(1, dtype=cp.uint64)
    _launch(bound, f"{name}_persist", n, grid,
            (ctypes.c_void_p(nxt.data.ptr), ctypes.c_void_p(0),
             ctypes.c_void_p(steps.data.ptr), ctypes.c_void_p(total.data.ptr),
             ctypes.c_uint(32)))
    assert int(nxt[0]) >= n, "every sample was fetched"
    out = _snap(dev, counter)
    _reported(name, planes, out, steps, total)
    return out


def _cases(name, n, case, plain=None):
    if case == "continuing":
        # a batch a first launch of 37 steps left partly run
        first = _cases(name, n, "mixed" if name != "oscillator" else "spread")
        return {key: value for key, value in _own(plain, name, first, 37).items()
                if key != "counter"}
    if case == "mixed":
        return _mixed(name, n, seed=20261006)
    planes = _planes(name, n, "spread", seed=20261006)
    if case == "uniform":
        planes["nstop"][:] = 250.0
    if case == "terminated_at_0":
        planes["terminated"][:] = True
    return planes


def _same(got, want, what):
    for key in want:
        assert got[key].tobytes() == want[key].tobytes(), f"{what}: {key}"


CASES = [("oscillator", "spread", 2), ("rkf78_attempt", "mixed", 2),  # 2: the eps bound
         ("oscillator", "uniform", 1000), ("oscillator", "spread", 1000),
         ("oscillator", "spread", 37), ("oscillator", "terminated_at_0", 1000),
         ("rkf78_attempt", "mixed", 1000), ("rkf78_attempt", "mixed", 37),
         ("rkf78_attempt", "terminated_at_0", 1000), ("oscillator", "continuing", 1000),
         ("rkf78_attempt", "continuing", 1000)]


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 33, 1000, 2049, 10_000])
@pytest.mark.parametrize("name,case,budget", CASES,
                         ids=[f"{k}-{c}-{b}" for k, c, b in CASES])
def test_the_fast_entries_are_the_map_free_entries_within_rounding(bundles, name, case,
                                                                   budget, n):
    plain, active = bundles
    planes = _cases(name, n, case, plain)
    want = _own(plain, name, planes, budget)
    _close(_persist(plain, name, planes, budget, 64, mapped=False), want,
          "map-free persist", budget, planes)
    _close(_range(plain, name, planes, budget, mapped=False), want, "map-free range", budget, planes)
    _close(_own(active, name, planes, budget, mapped=True), want, "identity map", budget, planes)
    _close(_own(active, name, planes, budget, mapped=True, k_max=64), want,
          "identity map, 64-step launches", budget, planes)
    _close(_range(active, name, planes, budget), want, "range entry", budget, planes)
    for grid in (1, 64):
        _close(_persist(active, name, planes, budget, grid, mapped="decoy"), want,
               f"active persist grid={grid}", budget, planes)
    steps = want["k"] if name == "oscillator" else want["n_acc"] + want["n_rej"]
    start = planes["k"] if name == "oscillator" else planes["n_acc"] + planes["n_rej"]
    if case == "terminated_at_0":
        assert (steps == start).all(), "nothing moves"
    elif budget == 37 and n >= 33:
        assert (steps - start).max() == 37 and not want["terminated"].all(), \
            "the budget cut"
