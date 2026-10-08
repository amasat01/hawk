# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An active-set automatic kernel's fast-path entries move no bit.

An automatic kernel traced under an active-set kind (``Guard(active_set=True)``)
reads its samples through the index map. Its device unit also carries the two
entries the map-free automatic kernel of the same body has, indexing samples
directly (``base + flat``): ``<name>_range`` (the two-path fused entry over a
contiguous range) and ``<name>_persist`` (:func:`hawk.emit.cuda.persist_entry`).
Both take the map entry's parameters (the map and count planes among them,
unread) and compile only under ``HAWK_FAST_ENTRIES`` (a build's
``defines``), so the default build compiles the map entry alone. A map-free
automatic kernel has the same ``_range`` entry under the macro beside its
always-built ``_persist``. Both entries report the launch's exact step count:
each sample's steps taken, maxed into a trailing ``hawk_steps`` cell. What is
pinned:

* **Codegen.** The two entries exist for an active-set automatic kernel on the
  device only, behind the macro; a map-free automatic kernel's text has no
  ``_range`` entry, and a single-step active-set kernel has neither.
* **Bit identity.** The RK4 oscillator (uniform and spread stops) and the
  RK7(8) attempt kernel (mixed eccentricities), N = 1, 33, 1000 and 10k, with
  a budget below the stops and with every sample terminated at launch: the
  ``_range`` and ``_persist`` entries leave every plane, the mask and the
  counter byte-equal to the map-free kernel's fused and persistent entries
  and to a run of the map entry over the identity map (one launch of the
  budget, and launches of at most 64 steps). Step counts are planes (``k``,
  ``n_acc``/``n_rej``), compared too. The fast entries run with a decoy map
  (sample 0 alone) bound, so a map read would show.
* **Exact steps.** Every range and persist launch (map-free and active-set,
  N up to 10k, all finish / budget hit / terminated at launch / a second
  launch continuing a partly-run batch) reports in ``hawk_steps`` the largest
  per-sample step count of that launch, read off the step planes.
"""

from __future__ import annotations

import ctypes
import importlib
import re

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of
from test_host_interchange import _mixed, _planes, oscillator, rkf78_attempt

from hawk.artifact import build_bundle
from hawk.emit import FAST_ENTRIES_MACRO, RANGE_SUFFIX, scalar_mode
from hawk.ext import Guard, Kind

B = importlib.import_module("hawk.artifact.bundle")

DT = 0.01
KERNELS = {"oscillator": oscillator, "rkf78_attempt": rkf78_attempt}
ACTIVE = Kind("fast_entries_active", guard=Guard(active_set=True))
FAST = (f"{FAST_ENTRIES_MACRO}=1",)


def _sources(kernel, kind):
    unit = B._active_unit(kernel, kernel.name, kind)
    return B._sources(unit, name=kernel.name, smode=scalar_mode("float64"), kind=kind,
                      targets=("cuda", "host"), layout_sizes_override=None)


def _entries(text):
    return set(re.findall(r'extern "C" __global__ void (\w+)\(', text))


# --------------------------------------------------------------------------- #
# -- codegen.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(KERNELS))
def test_an_active_set_automatic_kernel_has_the_fast_entries_on_cuda_only(name):
    kernel = KERNELS[name]
    active = _sources(kernel, ACTIVE)
    cuda, host = active["cuda"].text, active["host"].text
    fast = {f"{name}{RANGE_SUFFIX}", f"{name}_persist"}
    assert _entries(cuda) == {name} | fast
    assert not any(f"void {e}(" in host for e in fast)
    # behind the macro, default off: the default build compiles the map entry alone
    gate = cuda.index(f"#if {FAST_ENTRIES_MACRO}")
    assert f"#define {FAST_ENTRIES_MACRO} 0" in cuda[:gate]
    assert all(cuda.index(f"void {e}(") > gate for e in fast)
    assert cuda.index(f"void {name}(") < gate
    # the fast entries index samples directly: the map is read once, by the map entry
    assert cuda.count("lut_active_map.data())[hawk_t]") == 2      # its two paths
    assert cuda[gate:].count("lut_active_map.data())") == 0
    # the same parameter list as the map entry, and the persist entry's extras
    sig = {e: cuda[cuda.index(f"void {e}("):].split(")\n{", 1)[0].split("(", 1)[1]
           for e in (name, *fast)}
    assert sig[f"{name}{RANGE_SUFFIX}"] == (sig[name] + ",\n    unsigned int* hawk_steps"
                                         ",\n    unsigned long long* hawk_stepsum")
    assert sig[f"{name}_persist"] == (sig[name] + ",\n    unsigned int* hawk_next"
                                      ",\n    unsigned long long* hawk_util"
                                      ",\n    unsigned int* hawk_steps"
                                      ",\n    unsigned long long* hawk_stepsum")


@pytest.mark.parametrize("name", list(KERNELS))
def test_a_map_free_kernel_gates_only_its_range_entry(name):
    plain = _sources(KERNELS[name], KERNELS[name].kind)["cuda"].text
    assert _entries(plain) == {name, f"{name}_persist", f"{name}{RANGE_SUFFIX}"}
    gate = plain.index(f"#if {FAST_ENTRIES_MACRO}")
    assert plain.index(f"void {name}_persist(") < gate
    assert plain.index(f"void {name}{RANGE_SUFFIX}(") > gate


def test_a_single_step_active_set_kernel_has_neither():
    one = oscillator.step
    text = _sources(one, ACTIVE)["cuda"].text
    assert _entries(text) == {one.name}
    assert FAST_ENTRIES_MACRO not in text


# --------------------------------------------------------------------------- #
# -- compiled: bit for bit.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("fast_entries")
    plain = build_bundle(list(KERNELS.values()), root / "plain", targets=("cuda",),
                         cache_dir=cache_dir, defines=FAST)
    active = build_bundle(list(KERNELS.values()), root / "active", targets=("cuda",),
                          kind=ACTIVE, cache_dir=cache_dir, defines=FAST)
    return plain, active


def _bind(bundle, name, planes, *, mapped):
    """``(bound plan, device planes, counter, word, module, n)``; ``mapped``
    binds the identity map (every sample, ascending) and its count, or
    (``"decoy"``) a map of sample 0 alone, which an entry that read it would
    show."""
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    plugin = L.device_plugin(bundle.directory, name, sidecar)
    dev = {key: cp.asarray(value) for key, value in planes.items()}
    n = planes["terminated"].shape[0]
    counter = cp.zeros(1, dtype=cp.uint32)
    counter[0] = int(np.count_nonzero(planes["terminated"]))
    word = cp.zeros(1, dtype=cp.int64)
    kw = {"dt": DT} if "dt" in dict((n_, r) for r, n_ in sidecar["arg_spec"]) else {}
    if mapped == "decoy":
        # a map and count no fast entry may read: only sample 0, once
        kw["active_map"] = cp.zeros(n, dtype=cp.int32)
        kw["active_count"] = cp.ones(1, dtype=cp.int32)
    elif mapped:
        kw["active_map"] = cp.arange(n, dtype=cp.int32)
        kw["active_count"] = cp.full(1, n, dtype=cp.int32)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        finished_count=counter.view(cp.int32), fused_steps=word, **kw, **dev)
    return bound, dev, counter, word, plugin._keepalive[1].module, n


def _snap(dev, counter):
    return {**{key: value.get() for key, value in dev.items()}, "counter": counter.get()}


def _launch(bound, module, entry, n, grid, extra=()):
    import cupy as cp

    boxes = [ctypes.c_longlong(0), ctypes.c_longlong(n), ctypes.c_longlong(n), *extra]
    addrs = list(bound._addrs) + [ctypes.addressof(box) for box in boxes]
    params = (ctypes.c_void_p * len(addrs))(*addrs)
    fn = module.get_function(entry)
    cp.cuda.driver.launchKernel(fn.kernel.ptr, grid, 1, 1, 256, 1, 1, 0, 0,
                                ctypes.addressof(params), 0)
    cp.cuda.runtime.deviceSynchronize()


def _own(bundle, name, planes, budget, *, mapped=False, k_max=None):
    """The kernel's own entry: ONE launch of word ``budget``, or (``k_max``)
    launches of ``min(k_max, left)`` until every sample finished."""
    bound, dev, counter, word, _, n = _bind(bundle, name, planes, mapped=mapped)
    left = budget
    while left > 0 and int(counter[0]) != n:
        word[0] = left if k_max is None else min(k_max, left)
        bound.launch()
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

    bound, dev, counter, word, module, n = _bind(bundle, name, planes, mapped=mapped)
    counter[0] = 0  # the entry counts the samples finished on entry itself
    word[0] = budget
    steps = cp.zeros(1, dtype=cp.uint32)
    total = cp.zeros(1, dtype=cp.uint64)
    _launch(bound, module, f"{name}{RANGE_SUFFIX}", n, (n + 255) // 256,
            (ctypes.c_void_p(steps.data.ptr), ctypes.c_void_p(total.data.ptr)))
    out = _snap(dev, counter)
    _reported(name, planes, out, steps, total)
    return out


def _persist(bundle, name, planes, budget, grid, *, mapped):
    import cupy as cp

    bound, dev, counter, word, module, n = _bind(bundle, name, planes, mapped=mapped)
    counter[0] = 0  # the entry counts the samples finished on entry itself
    word[0] = budget
    nxt = cp.zeros(1, dtype=cp.uint32)
    steps = cp.zeros(1, dtype=cp.uint32)
    total = cp.zeros(1, dtype=cp.uint64)
    _launch(bound, module, f"{name}_persist", n, grid,
            (ctypes.c_void_p(nxt.data.ptr), ctypes.c_void_p(0),
             ctypes.c_void_p(steps.data.ptr), ctypes.c_void_p(total.data.ptr)))
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


CASES = [("oscillator", "uniform", 1000), ("oscillator", "spread", 1000),
         ("oscillator", "spread", 37), ("oscillator", "terminated_at_0", 1000),
         ("rkf78_attempt", "mixed", 1000), ("rkf78_attempt", "mixed", 37),
         ("rkf78_attempt", "terminated_at_0", 1000), ("oscillator", "continuing", 1000),
         ("rkf78_attempt", "continuing", 1000)]


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 33, 1000, 2049, 10_000])
@pytest.mark.parametrize("name,case,budget", CASES,
                         ids=[f"{k}-{c}-{b}" for k, c, b in CASES])
def test_the_fast_entries_are_the_map_free_entries_bit_for_bit(bundles, name, case,
                                                               budget, n):
    plain, active = bundles
    planes = _cases(name, n, case, plain)
    want = _own(plain, name, planes, budget)
    _same(_persist(plain, name, planes, budget, 64, mapped=False), want,
          "map-free persist")
    _same(_range(plain, name, planes, budget, mapped=False), want, "map-free range")
    _same(_own(active, name, planes, budget, mapped=True), want, "identity map")
    _same(_own(active, name, planes, budget, mapped=True, k_max=64), want,
          "identity map, 64-step launches")
    _same(_range(active, name, planes, budget), want, "range entry")
    for grid in (1, 64):
        _same(_persist(active, name, planes, budget, grid, mapped="decoy"), want,
              f"active persist grid={grid}")
    steps = want["k"] if name == "oscillator" else want["n_acc"] + want["n_rej"]
    start = planes["k"] if name == "oscillator" else planes["n_acc"] + planes["n_rej"]
    if case == "terminated_at_0":
        assert (steps == start).all(), "nothing moves"
    elif budget == 37 and n >= 33:
        assert (steps - start).max() == 37 and not want["terminated"].all(), \
            "the budget cut"
