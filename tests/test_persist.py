# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The persistent device entry of an automatic kernel moves no bit.

An automatic (``steps="auto"``) kernel's device unit carries, beside its
fused entry, ``<name>_persist`` (:func:`hawk.emit.cuda.persist_entry`):
lanes fetch samples off a counter, run each to its finish or to the step
budget the ``fused_steps`` word holds, commit it and fetch the next. What
is pinned:

* **Bit identity.** The RK4 oscillator (uniform and spread stops, some
  samples finished before the launch, a budget below the stops) and the
  RK7(8) attempt kernel (a ``Vector[6]`` carry, mixed eccentricities, a
  budget hit), for N = 1, 31, 33, 1000 and 10k, launched persistent on a
  small and a large grid, leave every plane, the mask and the counter
  byte-equal to ONE fused launch with the same word, and to the fused
  launches of at most 64 steps each a runner makes. Step counts per sample
  are planes (``k``, ``n_acc``/``n_rej``), so they are compared too.
* **The word is a runtime bound.** A word above 64 runs that many steps in
  one fused launch (the rows above use it).
* **Lane utilisation** is counted when asked: active lanes over
  warp-iterations, in ``(0, 1]``.
* **Only automatic kernels** carry the entry, and only on the device.
"""

from __future__ import annotations

import ctypes

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of
from test_host_interchange import (
    _mixed,
    _planes,
    oscillator,
    oscillator_x16,
    rkf78_attempt,
)

from hawk.artifact import build_bundle
from hawk.emit import BACKENDS, render_source
from hawk.emit.aether import render_lane_split
from hawk.emit.cuda import PERSIST_CHUNK, SHORT_DEVICE_STEP_OPS

DT = 0.01
KERNELS = {"oscillator": oscillator, "rkf78_attempt": rkf78_attempt}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle(list(KERNELS.values()), tmp_path_factory.mktemp("persist"),
                        targets=("cuda",), cache_dir=cache_dir)


def _bind(bundle, name, planes):
    """``(bound plan, device planes, counter, word, module)``."""
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    plugin = L.device_plugin(bundle.directory, name, sidecar)
    dev = {key: cp.asarray(value) for key, value in planes.items()}
    counter = cp.zeros(1, dtype=cp.uint32)
    counter[0] = int(np.count_nonzero(planes["terminated"]))
    word = cp.zeros(1, dtype=cp.int64)
    kw = {"dt": DT} if "dt" in dict((n, r) for r, n in sidecar["arg_spec"]) else {}
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        finished_count=counter.view(cp.int32), fused_steps=word, **kw, **dev)
    return bound, dev, counter, word, plugin._keepalive[1].module


def _snap(dev, counter):
    return {**{key: value.get() for key, value in dev.items()}, "counter": counter.get()}


def _fused(bundle, name, planes, budget, k_max=None):
    """Fused launches: ONE with word ``budget``, or (``k_max``) a runner's
    launches of ``min(k_max, left)`` until every sample finished."""
    bound, dev, counter, word, _ = _bind(bundle, name, planes)
    n = planes["terminated"].shape[0]
    left = budget
    while left > 0 and int(counter[0]) != n:
        word[0] = left if k_max is None else min(k_max, left)
        bound.launch()
        left -= int(word[0])
        if k_max is None:
            break
    return _snap(dev, counter)


def _persist(bundle, name, planes, budget, grid, util=None, steps=None):
    """One launch of ``<name>_persist`` on ``grid`` blocks of 256 (``steps``:
    the ``hawk_steps`` cell, or null)."""
    import cupy as cp

    bound, dev, counter, word, module = _bind(bundle, name, planes)
    counter[0] = 0  # the entry counts the samples finished on entry itself
    n = planes["terminated"].shape[0]
    word[0] = budget
    fn = module.get_function(f"{name}_persist")
    nxt = cp.zeros(1, dtype=cp.uint32)
    extra = [ctypes.c_longlong(0), ctypes.c_longlong(n), ctypes.c_longlong(n),
             ctypes.c_void_p(nxt.data.ptr),
             ctypes.c_void_p(0 if util is None else util.data.ptr),
             ctypes.c_void_p(0 if steps is None else steps.data.ptr), ctypes.c_void_p(0)]
    addrs = list(bound._addrs) + [ctypes.addressof(box) for box in extra]
    params = (ctypes.c_void_p * len(addrs))(*addrs)
    cp.cuda.driver.launchKernel(fn.kernel.ptr, grid, 1, 1, 256, 1, 1, 0, 0,
                                ctypes.addressof(params), 0)
    cp.cuda.runtime.deviceSynchronize()
    assert int(nxt[0]) >= n, "every sample was fetched"
    return _snap(dev, counter)


def _cases(name, n, case):
    if case == "mixed":
        return _mixed(name, n, seed=20261007)
    planes = _planes(name, n, "spread", seed=20261007)
    if case == "uniform" and name == "oscillator":
        planes["nstop"][:] = 250.0
    if case == "all_finished":
        planes["terminated"][:] = True
    return planes


def _same(got, want, what):
    for key in want:
        assert got[key].tobytes() == want[key].tobytes(), f"{what}: {key}"


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 31, 33, 1000, 10_000])
@pytest.mark.parametrize("case,budget", [("spread", 1000), ("uniform", 1000),
                                         ("mixed", 1000), ("mixed", 37),
                                         ("all_finished", 1000)])
@pytest.mark.parametrize("name", list(KERNELS))
def test_persist_is_the_fused_launch_bit_for_bit(bundle, name, case, budget, n):
    if case == "uniform" and name != "oscillator":
        pytest.skip("the RK7(8) batch has its own spread of attempts")
    planes = _cases(name, n, case)
    one = _fused(bundle, name, planes, budget)
    _same(_fused(bundle, name, planes, budget, k_max=64), one, "64-step launches")
    for grid in (1, 64):
        _same(_persist(bundle, name, planes, budget, grid), one, f"persist grid={grid}")
    if case == "all_finished":
        _same(one, {**planes, "counter": np.array([n], dtype=np.uint32)},
              "nothing moves")
    if case == "mixed" and budget == 37 and n >= 31:
        steps = one["k"] if name == "oscillator" else one["n_acc"] + one["n_rej"]
        assert steps.max() == 37 and not one["terminated"].all(), "the budget cut"


@pytest.mark.gpu
def test_persist_counts_its_lane_utilisation(bundle):
    import cupy as cp

    util = cp.zeros(2, dtype=cp.uint64)
    _persist(bundle, "oscillator", _cases("oscillator", 10_000, "spread"), 1000, 8,
             util=util)
    active, slots = (int(v) for v in util.get())
    assert 0 < active <= slots and slots % 32 == 0



@pytest.mark.gpu
@pytest.mark.parametrize("budget", [1000, 2 * PERSIST_CHUNK])
def test_persist_chunks_with_lanes_finishing_anywhere_are_bit_identical(bundle, budget):
    """Lanes of one warp finishing just before, at and after a chunk boundary,
    at and past the budget, with finished samples mixed in, on few lanes (so
    each refills several times inside chunks): the fused launch's bits."""
    k = PERSIST_CHUNK
    n = 3 * 32 + 5
    planes = _planes("oscillator", n, "spread", seed=20261007)
    stops = [1, k - 1, k, k + 1, 2 * k + 1, budget - 1, budget, budget + 7]
    planes["nstop"][:] = [float(stops[j % len(stops)]) for j in range(n)]
    planes["terminated"][:] = np.arange(n) % 7 == 3
    one = _fused(bundle, "oscillator", planes, budget)
    for grid in (1, 2):
        _same(_persist(bundle, "oscillator", planes, budget, grid), one, f"persist grid={grid}")


@pytest.mark.gpu
def test_persist_counts_every_step_once_on_a_uniform_batch(bundle):
    """Every step counted once, and no slot lost beyond each lane's last,
    partial chunk and each warp's final head (the one that finds the batch
    exhausted): no lane idles inside a chunk."""
    import cupy as cp

    util = cp.zeros(2, dtype=cp.uint64)
    n, grid = 4096, 8
    planes = _cases("oscillator", n, "uniform")
    _persist(bundle, "oscillator", planes, 1000, grid, util=util)
    active, slots = (int(v) for v in util.get())
    lanes = grid * 256
    assert active == n * int(planes["nstop"][0])
    assert slots - active <= lanes * (PERSIST_CHUNK - 1) + (lanes // 32) * 32 * PERSIST_CHUNK


def test_the_persist_loop_has_no_warp_collective():
    """Correct however independent thread scheduling splits a warp: no
    synchronising or exchanging warp intrinsic in the persistent loop, only
    the one ``__activemask`` naming the lanes that count capacity."""
    text = render_source(oscillator.name, oscillator.sinks, oscillator.walk, BACKENDS["cuda"],
                         kind=oscillator.kind, one_step=oscillator.one_step).text
    body = text.split(f"void {oscillator.name}_persist(", 1)[1].split("\n}\n", 1)[0]
    for token in ("__syncwarp", "__shfl", "__ballot", "__reduce", "__any_sync", "__all_sync"):
        assert token not in body, token
    assert body.count("__activemask()") == 1
    assert f"hawk_j < {PERSIST_CHUNK}" in body


@pytest.mark.parametrize("kernel, light", [(oscillator, True), (rkf78_attempt, False)],
                         ids=lambda k: getattr(k, "name", str(k)))
def test_only_a_light_step_unrolls_its_device_loops(kernel, light):
    """A step at or under ``SHORT_DEVICE_STEP_OPS`` operations unrolls the
    ``_persist`` chunk loop and the ``_range`` step loop by 4; a heavier step
    unrolls neither."""
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kernel.kind)
    assert (split.op_count <= SHORT_DEVICE_STEP_OPS) == light, split.op_count
    text = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    for suffix in ("_persist", "_range"):
        body = _entry_body(text, kernel.name, suffix)
        assert body.count("#pragma unroll 4") == (1 if light else 0), (kernel.name, suffix)
        assert body.count("#pragma unroll") == body.count("#pragma unroll 4")


def test_a_stepping_kernels_sidecar_carries_its_step_op_count(bundle):
    """``step_ops`` is the lane split's own operation count, for the runtime
    that launches the kernel; a kernel that does not step has none."""
    from hawk.artifact.bundle import _step_ops

    for name, kernel in KERNELS.items():
        split = render_lane_split(kernel.sinks, kernel.walk, kind=kernel.kind)
        assert sidecar_of(bundle, name)["step_ops"] == split.op_count > 0, name
        assert sidecar_of(bundle, name)["entry_counts_finished"] is True, name
    assert _step_ops(oscillator.step, oscillator.step.kind) is None


def test_only_an_automatic_kernel_has_a_persist_entry():
    one = oscillator.step          # the single step: no fused_steps word
    for kernel, want in ((oscillator, True), (rkf78_attempt, True),
                         (oscillator_x16, False), (one, False)):
        for backend in ("cuda", "host"):
            text = render_source(kernel.name, kernel.sinks, kernel.walk,
                                 BACKENDS[backend], kind=kernel.kind,
                                 one_step=getattr(kernel, "one_step", None)).text
            has = f"void {kernel.name}_persist(" in text
            assert has == (want and backend == "cuda"), (kernel.name, backend)


def _entry_body(text, name, suffix):
    head = f"void {name}{suffix}(" if suffix else f"void {name}("
    return text.split(head, 1)[1].split("\n}\n", 1)[0]


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_fast_entries_count_trips_in_32_bits(kernel):
    """The ``_persist`` and ``_range`` trip/count and the fused body's runtime
    trip loop are 32-bit ``int``, never ``Int``/``long long``."""
    import re

    text = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    for suffix in ("_persist", "_range"):
        body = _entry_body(text, kernel.name, suffix)
        for bad in ("Int hawk_trip", "Int hawk_cnt", "long long hawk_trip",
                    "long long hawk_cnt"):
            assert bad not in body, f"{kernel.name}{suffix}: 64-bit counter `{bad}`"
        assert "int hawk_trip" in body, f"{kernel.name}{suffix}: no `int hawk_trip`"
        assert "int hawk_cnt" in body, f"{kernel.name}{suffix}: no `int hawk_cnt`"
    fused = _entry_body(text, kernel.name, "")
    assert "lut_fused_steps" in fused
    loops = re.findall(r"for \((\w+(?: \w+)*) hawk_\w+_step = 0;", fused)
    assert loops, f"{kernel.name}: no runtime-count loop in the fused body"
    for decl in loops:
        assert decl == "int", f"{kernel.name}: fused loop index is `{decl}`, want `int`"


@pytest.mark.gpu
def test_a_budget_word_past_int_range_still_runs_to_the_data_exit(bundle):
    """A word of 2**33 (an int64 word) is clamped, not wrapped: every sample
    stops by its data long before, the bits and step counts are those of word 1000."""
    n = 1000
    planes = _cases("oscillator", n, "spread")
    one = _fused(bundle, "oscillator", planes, 1000)
    assert one["terminated"].all(), "every sample stops by data"
    big = 2 ** 33
    got = _fused(bundle, "oscillator", planes, big)
    _same(got, one, "fused word 2**33")
    assert got["k"].tobytes() == one["k"].tobytes(), "steps match"
    for grid in (1, 8):
        got = _persist(bundle, "oscillator", planes, big, grid)
        _same(got, one, f"persist word 2**33 grid={grid}")
        assert got["k"].tobytes() == one["k"].tobytes(), f"steps match grid={grid}"


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [37, 2 * PERSIST_CHUNK])
def test_persist_budget_hits_inside_and_at_chunk_ends(bundle, budget):
    """A budget off and on a chunk multiple, stops cycling around it, some
    samples finished on entry, on one block (many refills, some landing
    mid-chunk on a sample whose budget is shorter than the chunk's rest)."""
    n = 3 * 32 + 5
    planes = _planes("oscillator", n, "spread", seed=20261007)
    stops = [1, 5, 31, 33, budget - 1, budget, budget + 3]
    planes["nstop"][:] = [float(stops[j % len(stops)]) for j in range(n)]
    planes["terminated"][:] = np.arange(n) % 7 == 3
    one = _fused(bundle, "oscillator", planes, budget)
    assert one["k"].max() == budget and not one["terminated"].all(), "the budget cut"
    _same(_persist(bundle, "oscillator", planes, budget, 1), one, "persist grid=1")


def test_the_fast_entries_count_samples_finished_on_entry():
    """Both fast entries add a sample finished on entry to the finish counter,
    as the step adds one finishing in the run, so the counter -- zeroed before
    the launch -- holds the batch's finished count, and the sidecar says so."""
    from hawk.emit.cuda import entry_counts_finished
    from hawk.emit.aether import render_lane_split

    k = KERNELS["oscillator"]
    text = render_source(k.name, k.sinks, k.walk, BACKENDS["cuda"],
                         kind=k.kind, one_step=k.one_step).text
    for entry in ("_persist", "_range"):
        body = text.split(f"void {k.name}{entry}(", 1)[1].split("\n}\n", 1)[0]
        on_entry = "} else { ++hawk_fin; }" if entry == "_persist" else \
            "} else { hawk_abi::finish_count("
        assert on_entry in body, entry
    assert entry_counts_finished(k.walk, render_lane_split(k.sinks, k.walk, kind=k.kind))


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_the_persist_loop_reports_its_cells_once_at_exit(kernel):
    """No atomic on a shared cell per sample inside the persistent loop: a
    lane keeps its longest committed trip count and its finish count (the
    step's finish epilogue and the on-entry count) in registers and reports
    each ONCE after the loop, warp-reduced. The sample's global index is
    computed once per fetched sample, not once per step."""
    text = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    body = _entry_body(text, kernel.name, "_persist")
    loop, exit_ = body.rsplit("    if (hawk_util != nullptr) atomicAdd(&hawk_util[0]", 1)
    assert "atomicMax(" not in body
    assert "hawk_abi::finish_count(" not in body
    assert "hawk_steps" not in loop.split(f"hawk_j < {PERSIST_CHUNK}", 1)[1]
    assert loop.count("++hawk_fin;") == 3       # on entry (twice: both fetches), the epilogue
    assert exit_.count("hawk_steps_max(hawk_steps, hawk_maxtrip);") == 1
    assert exit_.count("hawk_finish_sum(lut_finished_count, hawk_fin);") == 1
    chunk = loop.split(f"hawk_j < {PERSIST_CHUNK}", 1)[1].split("hawk_have = false;", 1)[0]
    assert "base + hawk_s" not in chunk and "long long" not in chunk


#: sha256 of the text the persist entry's change must leave alone: each
#: kernel's own (fused) entry and ``_range`` entry, and its whole host TU.
_FROZEN = {
    ("oscillator", ""): "d98cd506d5f9f9be9d998aa3501f5cee6ff12a75e1e67d64acb7467e13f51606",
    ("oscillator", "_range"): "68d0e37410029b9ba535d1d9db09b06f5fe561e364fc32c9760585cdd7e02f65",
    ("oscillator", "host"): "21c9bc02e697a53ada5da2e2c506304a727472fc59595e0d295fa03b7b7cbc8c",
    ("rkf78_attempt", ""): "322144bf209f87053cfc18ddac7f263d4269793061bc231e9b4b7f0e96748de8",
    ("rkf78_attempt", "_range"): "f302e17c6f68be322f68fc3d37cdd101c6c5732f512f42e49e65c6547ea36a40",
    ("rkf78_attempt", "host"): "6398950aa5ef257ff7e6ae00cea2fd3c43c79de35082960918bd14557e7d6017",
}


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_the_other_entries_are_unchanged_by_the_persist_report(kernel, monkeypatch):
    import hashlib

    import hawk.emit.host as host_emit

    # The host TU's tile is sized to (and names) the machine's caches; pin the
    # ones the digests were frozen with, so they describe the emitter, not the
    # CPU the test runs on.
    pinned = host_emit.CacheSizes(32 * 1024, 1024 * 1024, "/sys/devices/system/cpu/cpu0/cache")
    monkeypatch.setattr(host_emit, "machine_caches", lambda: pinned)

    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    cuda = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    host = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["host"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    for suffix in ("", "_range"):
        assert digest(_entry_body(cuda, kernel.name, suffix)) == \
            _FROZEN[(kernel.name, suffix)], (kernel.name, suffix)
    assert digest(host) == _FROZEN[(kernel.name, "host")], kernel.name


@pytest.mark.gpu
@pytest.mark.parametrize("name,case,budget", [("oscillator", "spread", 1000),
                                              ("oscillator", "mixed", 37),
                                              ("rkf78_attempt", "mixed", 1000)])
def test_persist_reports_the_longest_trip_and_the_finished_count(bundle, name, case, budget):
    """The ``hawk_steps`` cell holds the launch's longest per-sample trip
    count (the samples' own step planes moved by exactly that much at most,
    and one by exactly that much), and the finish counter the batch's
    finished count, on a small and a large grid."""
    import cupy as cp

    n = 10_000
    planes = _cases(name, n, case)
    for grid in (1, 64):
        steps = cp.zeros(1, dtype=cp.uint32)
        got = _persist(bundle, name, planes, budget, grid, steps=steps)
        if name == "oscillator":
            trips = got["k"] - planes["k"]
        else:
            trips = (got["n_acc"] + got["n_rej"]) - (planes["n_acc"] + planes["n_rej"])
        assert int(steps.get()[0]) == int(trips.max()) > 0, (name, case, grid)
        assert int(got["counter"][0]) == int(np.count_nonzero(got["terminated"])), grid
