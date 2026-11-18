# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The persistent device entry of an automatic kernel agrees with the fused one within rounding.

An automatic (``steps="auto"``) kernel's device unit carries, beside its
fused entry, ``<name>_persist`` (:func:`ospreys.cuda.emit.persist_entry`):
lanes fetch samples off a counter, run each to its finish or to the step
budget the ``fused_steps`` word holds, commit it and fetch the next. What
is pinned:

* **Agreement.** The RK4 oscillator (uniform and spread stops, some
  samples finished before the launch, a budget below the stops) and the
  RK7(8) attempt kernel (a ``Vector[6]`` carry, mixed eccentricities, a
  budget hit), for N = 1, 31, 33, 1000 and 10k, launched persistent on a
  small and a large grid, leave the step counts, the mask and the counter
  equal to ONE fused launch with the same word, and every float plane within
  the math libraries' rounding of it (the two entries are compiled apart); the fused
  launches of at most 64 steps each a runner makes are byte-equal to it. Step
  counts per sample are planes (``k``, ``n_acc``/``n_rej``), so they are compared too.
* **The word is a runtime bound.** A word above 64 runs that many steps in
  one fused launch (the rows above use it).
* **Warp-converged.** One warp-collective claim per chunk head (one
  ``atomicAdd(hawk_next`` in the entry), nothing collective or atomic in the
  step body, and the chunk length a launch argument: any length leaves the
  same bits (one entry, one compile). A sample ending in a chunk is written back at the next head,
  with the warp's other ending lanes.
* **Slot fill** is counted when asked: the steps run over the warps' chunk
  slots (32 lanes times the chunk length per head), a lane masked after its
  sample commits counting as an empty slot.
* **Only automatic kernels** carry the entry, and only on the device.
"""

from __future__ import annotations

import ctypes
import os

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of
from test_host_interchange import (
    _mixed,
    _planes,
    oscillator,
    EXP_REJECT,
    RK_A,
    RK_B,
    RK_BE,
    TOL,
    rkf78_attempt,
)

from hawk.artifact import build_bundle

DT = 0.01
#: The chunk length these rows launch with unless they vary it.
CHUNK = 32
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
        finished_count=counter, fused_steps=word, **kw, **dev)
    return bound, dev, counter, word, L.module_of(plugin._keepalive[1])


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


def _persist(bundle, name, planes, budget, grid, util=None, steps=None, chunk=CHUNK):
    """One launch of ``<name>_persist`` on ``grid`` blocks of 256 (``steps``:
    the ``hawk_steps`` cell, or null) with chunks of ``chunk`` steps."""
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
             ctypes.c_void_p(0 if steps is None else steps.data.ptr), ctypes.c_void_p(0),
             ctypes.c_uint(chunk)]
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


#: Planes that count or flag: exact in every comparison.
EXACT = ("k", "n_acc", "n_rej", "terminated", "counter", "nstop")
EPS = np.finfo(float).eps


def _kappa(start):
    """Per sample, the conditioning of the RK7(8) error sum (sum |e_i k_i| / |sum e_i k_i|)
    at the component that sets the step size, at the launch's input state: what a rounding
    in a stage costs the error estimate, and so ``h``. Components within their own
    rounding of the largest error ratio (where a rounding could switch which one sets
    the step) count too: the worst of those."""
    s0 = start["s"]
    h0 = np.minimum(start["h"], np.maximum(start["t_final"] - start["t"], 1e-300))

    def rhs(v):
        ir3 = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** -1.5
        return np.array([v[3], v[4], v[5], -v[0] * ir3, -v[1] * ir3, -v[2] * ir3])

    ks = [rhs(s0)]
    for row in RK_A:
        ks.append(rhs(s0 + h0 * sum(a * k for a, k in zip(row, ks))))
    terms = np.array([e * k for e, k in zip(RK_BE, ks)])
    s8 = s0 + h0 * sum(b * k for b, k in zip(RK_B, ks))
    with np.errstate(all="ignore"):
        kappa = np.nan_to_num(np.abs(terms).sum(0) / np.abs(terms.sum(0)))  # 0/0: no error
        ratio = np.abs(terms.sum(0)) * h0 / (TOL + TOL * np.maximum(np.abs(s8), np.abs(s0)))
    top = ratio.max(0)
    tied = ratio >= top * (1 - 8 * EPS * kappa)
    return np.where(tied, kappa, 0.0).max(0)


def _close(got, want, what, budget, start):
    """Two compiled entries agree within the math libraries' rounding. Counts and flags
    are exact. After a budget of 2 steps the state is within 8 eps a step, and the RK7(8)
    step size ``h`` per sample within 4 * (1/8) * kappa * 8 eps; after a long budget
    only ``t`` of a finished sample is compared (``t_final``): different step-size
    sequences give equally valid trajectories, so their divergence is integrator error,
    not rounding."""
    short = budget <= 2
    kernel = "oscillator" if "x" in want else "rkf78_attempt"
    state = ("x", "v") if "x" in want else ("s", "t")
    for key in want:
        if key in EXACT or want[key].dtype.kind != "f" or (key not in state + ("h",)):
            assert got[key].tobytes() == want[key].tobytes(), f"{what}: {key}"
        elif key == "h":
            if short:
                kappa = _kappa(start)
                with np.errstate(all="ignore"):  # an unbounded kappa gives an infinite bound, and a 0/0 rel
                    bound = MARGIN * abs(EXP_REJECT) * kappa * 8 * EPS
                    rel = np.abs(got[key] - want[key]) / np.abs(want[key])
                _report(kernel, what, budget, key, np.nanmax(rel / bound, initial=0.0),
                        rel.max(initial=0.0), bound[rel.argmax()] if rel.size else 0.0)
                assert ((rel <= bound) | (rel == 0)).all(), f"{what}: {key}"
        elif short:
            top = max(float(np.abs(want[key]).max(initial=0.0)), np.finfo(float).tiny)
            diff = float(np.abs(got[key] - want[key]).max(initial=0.0))
            _report(kernel, what, budget, key, diff / top / (8 * EPS * budget),
                    diff / top, 8 * EPS * budget)
            assert diff <= 8 * EPS * budget * top, f"{what}: {key}"
        elif key == "t":
            done = want["terminated"].astype(bool)
            assert got[key][done].tobytes() == want[key][done].tobytes(), f"{what}: {key}"


MARGIN = 4.0


def _report(*row):
    """Append ``kernel entry budget plane observed/bound observed bound`` to the file
    ``HAWK_TEST_ULP_REPORT`` names, if set."""
    path = os.environ.get("HAWK_TEST_ULP_REPORT")
    if path:
        with open(path, "a") as f:
            f.write(" ".join(str(v).replace(" ", "_") for v in row) + "\n")


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 31, 33, 1000, 10_000])
@pytest.mark.parametrize("case,budget", [("spread", 1000), ("uniform", 1000),
                                         ("mixed", 1000), ("mixed", 37),
                                         ("all_finished", 1000),
                                         ("spread", 2)])  # 2: the eps-bound case
@pytest.mark.parametrize("name", list(KERNELS))
def test_persist_is_the_fused_launch_within_rounding(bundle, name, case, budget, n):
    if case == "uniform" and name != "oscillator":
        pytest.skip("the RK7(8) batch has its own spread of attempts")
    planes = _cases(name, n, case)
    one = _fused(bundle, name, planes, budget)
    _same(_fused(bundle, name, planes, budget, k_max=64), one, "64-step launches")
    for grid in (1, 64):
        _close(_persist(bundle, name, planes, budget, grid), one, f"persist grid={grid}",
               budget, planes)
    if case == "all_finished":
        _same(one, {**planes, "counter": np.array([n], dtype=np.uint32)},
              "nothing moves")
    if case == "mixed" and budget == 37 and n >= 31:
        steps = one["k"] if name == "oscillator" else one["n_acc"] + one["n_rej"]
        assert steps.max() == 37 and not one["terminated"].all(), "the budget cut"


@pytest.mark.gpu
def test_persist_counts_its_slot_fill(bundle):
    import cupy as cp

    util = cp.zeros(2, dtype=cp.uint64)
    _persist(bundle, "oscillator", _cases("oscillator", 10_000, "spread"), 1000, 8,
             util=util)
    active, slots = (int(v) for v in util.get())
    assert 0 < active <= slots and slots % (32 * CHUNK) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("chunk", [CHUNK, 8, 1, 0, 6])
@pytest.mark.parametrize("budget", [1000, 2 * CHUNK])
def test_persist_chunks_with_lanes_finishing_anywhere_are_bit_identical(bundle, budget, chunk):
    """Lanes of one warp finishing just before, at and after a chunk boundary,
    at and past the budget, with finished samples mixed in, on few lanes (so
    each claims several times), for chunks of 32, 8 and 1 steps, 0 (each
    sample whole) and 6 (not a power of two: shorter chunks, the same work):
    the bits of the default chunk, which agree with the fused launch within rounding."""
    k = CHUNK
    n = 3 * 32 + 5
    planes = _planes("oscillator", n, "spread", seed=20261007)
    stops = [1, k - 1, k, k + 1, 2 * k + 1, budget - 1, budget, budget + 7]
    planes["nstop"][:] = [float(stops[j % len(stops)]) for j in range(n)]
    planes["terminated"][:] = np.arange(n) % 7 == 3
    one = _fused(bundle, "oscillator", planes, budget)
    for grid in (1, 2):
        ref = _persist(bundle, "oscillator", planes, budget, grid)
        _close(ref, one, f"persist grid={grid}", budget, planes)
        _same(_persist(bundle, "oscillator", planes, budget, grid, chunk=chunk), ref,
              f"persist grid={grid} chunk={chunk}")


@pytest.mark.gpu
def test_persist_counts_every_step_once_on_a_uniform_batch(bundle):
    """Every step counted once, and on a uniform batch the warps' chunks run
    nearly full: only each sample's last, partial chunk leaves slots empty."""
    import cupy as cp

    util = cp.zeros(2, dtype=cp.uint64)
    n, grid = 4096, 8
    planes = _cases("oscillator", n, "uniform")
    _persist(bundle, "oscillator", planes, 1000, grid, util=util)
    active, slots = (int(v) for v in util.get())
    stop = int(planes["nstop"][0])
    assert active == n * stop
    assert slots == n * -(-stop // CHUNK) * CHUNK, "one sample per lane per chunk run"


def test_the_plan_view_reads_the_fp64_op_count_under_either_key(bundle):
    """``plan_view`` carries ``fp64_ops_ir``, and reads a sidecar that still
    spells it ``ops_per_step`` the same."""
    from hawk.artifact import plan_view

    sidecar = sidecar_of(bundle, "oscillator")
    want = sidecar["fp64_ops_ir"]
    assert plan_view(sidecar)["fp64_ops_ir"] == want
    old = {k: v for k, v in sidecar.items() if k != "fp64_ops_ir"}
    assert plan_view(dict(old, ops_per_step=want))["fp64_ops_ir"] == want
    assert "fp64_ops_ir" not in plan_view(old)


def _entry_body(text, name, suffix):
    head = f"void {name}{suffix}(" if suffix else f"void {name}("
    return text.split(head, 1)[1].split("\n}\n", 1)[0]


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
        _close(got, one, f"persist word 2**33 grid={grid}", 1000, planes)
        assert got["k"].tobytes() == one["k"].tobytes(), f"steps match grid={grid}"


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [37, 2 * CHUNK])
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
    _close(_persist(bundle, "oscillator", planes, budget, 1), one, "persist grid=1", budget, planes)


#: sha256 of the text the persist entry's change must leave alone: each
#: kernel's own (fused) entry and ``_range`` entry, and its whole host TU.
# The host digests moved when a host unit lost the device arms of its helpers; the CUDA entry digests moved when a device unit lost its path-marker guard.
_FROZEN = {
    ("oscillator", ""): "1114cb4bf6b517d5cafb8f3891ffbb56f4a3d9ced1634e0e77364d87264e3633",
    ("oscillator", "_range"): "97944ac29526e49ff59cc64a13c40d68be07597e505bc5bb402289454d00912e",
    ("oscillator", "host"): "8eb5f01f7c10b35be6f5452c7d90c9a5b2e781bc1a7d2f614680b253f74ee648",
    ("rkf78_attempt", ""): "c2f52ee4f0ccc2033802da27ee954e34efd1bf54b3490c23e117c6c0d48e86bb",
    ("rkf78_attempt", "_range"): "b6c7c041d665d592eac195e543d53d727a00f494261c6769b94160db987027a4",
    ("rkf78_attempt", "host"): "0f8bf1629800333d48258a2d7caeb90bfa3f809e9a1f487dbd3faff4a800d1d4",
}


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


@pytest.mark.gpu
def test_slot_fill_counts_the_masked_lanes_of_a_fragmented_warp(bundle):
    """Every warp split down the middle: half its lanes run a 1-step
    sample, half a 1000-step one, and nothing is left to claim. The 1-step
    lanes sit masked while the others run, and the slot fill says so:
    every warp runs ceil(1000 / chunk) full chunks, so the slots are exact
    and the fill is about one half."""
    import cupy as cp

    lanes = 256
    planes = _planes("oscillator", lanes, "spread", seed=20261011)
    planes["nstop"][:] = np.where(np.arange(lanes) % 32 < 16, 1.0, 1000.0)
    util = cp.zeros(2, dtype=cp.uint64)
    _persist(bundle, "oscillator", planes, 1000, 1, util=util)
    active, slots = (int(v) for v in util.get())
    assert active == int(planes["nstop"].sum())
    assert slots == (lanes // 32) * -(-1000 // CHUNK) * 32 * CHUNK
    assert 0.45 < active / slots < 0.55
