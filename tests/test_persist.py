# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The persistent device entry of an automatic kernel agrees with the fused one within rounding.

An automatic (``steps="auto"``) kernel's device unit carries, beside its
fused entry, ``<name>_persist`` (:func:`hawk.emit.cuda.persist_entry`):
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
    oscillator_x16,
    EXP_REJECT,
    RK_A,
    RK_B,
    RK_BE,
    TOL,
    rkf78_attempt,
)

from hawk.artifact import build_bundle
from hawk.emit import render_source
from hawk.emit.render import render_lane_split
from hawk.emit.cuda import SHORT_DEVICE_STEP_OPS
from hawk.targets import TARGETS

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
                bound = MARGIN * abs(EXP_REJECT) * _kappa(start) * 8 * EPS
                with np.errstate(all="ignore"):
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


def _loop_parts(text, name):
    """``(head, step body, rest)`` of a persist entry: the text up to the
    chunk loop, the unrolled chunk loop, and what follows it."""
    body = _entry_body(text, name, "_persist")
    head, rest = body.split("        do {", 1)
    chunk, tail = rest.split("        } while (", 1)
    return head, chunk, tail


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_the_persist_loop_is_warp_converged(kernel):
    """The warp claims its samples together at the chunk head -- one ballot,
    ONE ``atomicAdd(hawk_next`` by the leader, the base shuffled out -- and
    the step body holds no warp collective and no atomic: a lane whose sample
    commits is masked until the next head. The chunk length is the launch's
    ``hawk_chunk``, never a constant in the code."""
    text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    body = _entry_body(text, kernel.name, "_persist")
    head, chunk, tail = _loop_parts(text, kernel.name)
    assert body.count("atomicAdd(hawk_next") == head.count("atomicAdd(hawk_next") == 1
    for token in ("__syncwarp", "__shfl", "__ballot", "__any_sync", "__all_sync",
                  "__activemask", "atomic"):
        assert token not in chunk, token
    for token in ("__ballot_sync(", "__shfl_sync(", "__any_sync("):
        assert token in head, token
    # the ballot is the warp's meeting point at the head: no separate barrier
    assert "__syncwarp" not in head
    assert tail.startswith("(hawk_trip & (hawk_chunk - 1u)) != 0u);")
    assert "% 32u" not in body and "unsigned int hawk_chunk)" in text


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_a_sample_is_written_back_at_the_next_head(kernel):
    """A lane whose sample ends in the chunk only marks it (the ended bit in
    its trip count, the trips run below it when the step exited, 0 when the
    budget ran out) and is written back at the next chunk head, with the
    warp's other ending lanes, before the lane claims again: the chunk holds
    no plane store and no report, and the head holds the commit once, ahead
    of the claim's sample index. The loop's finish flag is rebuilt from the
    trip count, not held."""
    import re

    text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    head, chunk, _ = _loop_parts(text, kernel.name)
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kernel.kind)
    stores = re.findall(r"^\s*(\w+)\[i\] = ", split.post, re.M)
    assert stores
    for plane in stores:
        assert f"{plane}[i] = " not in chunk, plane
        assert head.count(f"{plane}[i] = ") == 1, plane
    assert "hawk_maxtrip" not in chunk and "hawk_ran" not in chunk
    ended = chunk.split(f"if ({split.exit} || hawk_trip >= hawk_budget) {{", 1)[1]
    assert ended.split("break;", 1)[0].strip() == \
        f"hawk_trip = {split.exit} ? (hawk_trip | 0x80000000u) : 0x80000000u;"
    commit = head.index("if (hawk_trip > hawk_budget) {")
    assert commit < head.index("hawk_s = hawk_base + ") and "finished = (hawk_trip != 0x80000000u ?" in head


def test_a_value_the_trip_count_cannot_rebuild_keeps_the_commit_in_the_chunk():
    """A committed plane the step never reads (``energy`` here) is neither the
    exit flag nor carried, so the head could not write it back: such a kernel
    keeps the commit where the sample ends, in the chunk."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated

    @hawk.kernel
    def decay(rate: Scalar, nstop: Scalar, dt: Param, terminated: Terminated,
              x: Mutable[Scalar], energy: Mutable[Scalar], k: Mutable[Scalar]):
        x = x - dt * rate * x
        energy = x * x
        kn = k + 1.0
        k = kn
        terminated = kn >= nstop

    text = render_source(decay.name, decay.sinks, decay.walk, TARGETS["cuda"],
                         kind=decay.kind, one_step=decay.one_step).text
    head, chunk, _ = _loop_parts(text, decay.name)
    assert "mut_energy[i] = " in chunk and "mut_energy[i] = " not in head
    assert "hawk_maxtrip = max(hawk_maxtrip, hawk_trip);" in chunk
    assert "if (hawk_trip > hawk_budget) {" not in head


@pytest.mark.parametrize("kernel, light", [(oscillator, True), (rkf78_attempt, False)],
                         ids=lambda k: getattr(k, "name", str(k)))
def test_only_a_light_step_unrolls_its_device_loops(kernel, light):
    """A step at or under ``SHORT_DEVICE_STEP_OPS`` operations unrolls the
    ``_persist`` chunk loop and the ``_range`` step loop by 4; a heavier step
    unrolls neither."""
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kernel.kind)
    assert (split.op_count <= SHORT_DEVICE_STEP_OPS) == light, split.op_count
    text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    for suffix in ("_persist", "_range"):
        body = _entry_body(text, kernel.name, suffix)
        assert body.count("#pragma unroll 4") == (1 if light else 0), (kernel.name, suffix)
        assert body.count("#pragma unroll") == body.count("#pragma unroll 4")


def test_a_stepping_kernels_sidecar_carries_its_step_op_count(bundle):
    """``step_ops`` is the lane split's own operation count, for the runtime
    that launches the kernel, and ``fp64_ops_ir`` its 64-bit floating-point
    operations in the IR; a kernel that does not step has neither."""
    from hawk.artifact.bundle import _step_ops

    for name, kernel in KERNELS.items():
        split = render_lane_split(kernel.sinks, kernel.walk, kind=kernel.kind)
        assert sidecar_of(bundle, name)["step_ops"] == split.op_count > 0, name
        assert sidecar_of(bundle, name)["fp64_ops_ir"] == split.fp64_ops > 0, name
        assert sidecar_of(bundle, name)["entry_counts_finished"] is True, name
        assert "ops_per_step" not in sidecar_of(bundle, name), name
    assert _step_ops(oscillator.step, oscillator.step.kind) is None


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


def test_only_an_automatic_kernel_has_a_persist_entry():
    one = oscillator.step          # the single step: no fused_steps word
    for kernel, want in ((oscillator, True), (rkf78_attempt, True),
                         (oscillator_x16, False), (one, False)):
        for target in ("cuda", "host"):
            text = render_source(kernel.name, kernel.sinks, kernel.walk,
                                 TARGETS[target], kind=kernel.kind,
                                 one_step=getattr(kernel, "one_step", None)).text
            has = f"void {kernel.name}_persist(" in text
            assert has == (want and target == "cuda"), (kernel.name, target)


def _entry_body(text, name, suffix):
    head = f"void {name}{suffix}(" if suffix else f"void {name}("
    return text.split(head, 1)[1].split("\n}\n", 1)[0]


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_fast_entries_count_trips_in_32_bits(kernel):
    """The ``_persist`` and ``_range`` trip/count and the fused body's runtime
    trip loop are 32-bit ``int``, never ``Int``/``long long``."""
    import re

    text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    # the persist entry's trip budget is hawk_budget (the launch's one word),
    # the range entry's hawk_cnt
    for suffix, count in (("_persist", "hawk_budget"), ("_range", "hawk_cnt")):
        body = _entry_body(text, kernel.name, suffix)
        for bad in ("Int hawk_trip", f"Int {count}", "long long hawk_trip",
                    f"long long {count}"):
            assert bad not in body, f"{kernel.name}{suffix}: 64-bit counter `{bad}`"
        assert "int hawk_trip" in body, f"{kernel.name}{suffix}: no `int hawk_trip`"
        assert f"int {count}" in body, f"{kernel.name}{suffix}: no `int {count}`"
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


def test_the_fast_entries_count_samples_finished_on_entry():
    """Both fast entries add a sample finished on entry to the finish counter,
    as the step adds one finishing in the run, so the counter -- zeroed before
    the launch -- holds the batch's finished count, and the sidecar says so."""
    from hawk.emit.cuda import entry_counts_finished
    from hawk.emit.render import render_lane_split

    k = KERNELS["oscillator"]
    text = render_source(k.name, k.sinks, k.walk, TARGETS["cuda"],
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
    each ONCE after the loop, warp-reduced. Nothing 64-bit is held across
    a step: the sample is a 32-bit offset from ``base``."""
    text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    body = _entry_body(text, kernel.name, "_persist")
    loop, exit_ = body.rsplit("    hawk_persist_util(hawk_util, hawk_ran);", 1)
    _, chunk, _ = _loop_parts(text, kernel.name)
    assert "atomicMax(" not in body
    assert "hawk_abi::finish_count(" not in body
    assert "hawk_steps" not in chunk and "hawk_util" not in chunk
    assert loop.count("++hawk_fin;") == 2       # on entry (at the claim), the epilogue
    assert exit_.count("hawk_steps_max(hawk_steps, hawk_maxtrip);") == 1
    assert exit_.count("hawk_finish_sum(lut_finished_count, hawk_fin);") == 1
    assert "long long" not in chunk


#: sha256 of the text the persist entry's change must leave alone: each
#: kernel's own (fused) entry and ``_range`` entry, and its whole host TU.
_FROZEN = {
    ("oscillator", ""): "c18c3f46b3d30faa40849a4847be5dea5a14e33540fd60ec71ba51a36cd0dc35",
    ("oscillator", "_range"): "97944ac29526e49ff59cc64a13c40d68be07597e505bc5bb402289454d00912e",
    ("oscillator", "host"): "3f65be80ef5276adb804c99b6e4ebf16cd10a2f6a6cdcc83b9f89ee75890b205",
    ("rkf78_attempt", ""): "f7663ab3a0401183de32192038abcaf32453b5f5a18c2b209ac84bf60f296bf2",
    ("rkf78_attempt", "_range"): "b6c7c041d665d592eac195e543d53d727a00f494261c6769b94160db987027a4",
    ("rkf78_attempt", "host"): "5f6540717598e66361225f99cd55ed14996d5cc5f79c9138adb335db3f5212b5",
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

    cuda = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    host = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["host"],
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


@pytest.mark.parametrize("kernel", [oscillator, rkf78_attempt], ids=lambda k: k.name)
def test_the_persist_entry_holds_only_what_a_step_needs(kernel):
    """The entry's residency is its register count, so it carries across a
    step only what the next step reads: a value computed from ``Param`` inputs
    alone is evaluated once at entry (not a per-lane copy), a carry the step
    never reads (the loop's exit flag) is rebuilt by the commit from the trip
    count, the trip count also ends the chunk (no chunk counter), and the
    step budget left says whether the lane holds a sample (no flag). The same
    rules hold for any kernel; these two are witnesses."""
    text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["cuda"],
                         kind=kernel.kind, one_step=kernel.one_step).text
    body = _entry_body(text, kernel.name, "_persist")
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kernel.kind)
    stores = [line.split("{}")[0].split()[-1] for line in body.splitlines()
              if line.startswith("    ") and line.endswith("{};")]
    read = [c for c in split.carries if c[0] in split.step]
    assert len([s for s in stores if s.startswith("hawk_pc")]) == len(read) < len(split.carries)
    free = len(split.sample_free and [d for d in split.sample_free
                                       if any(d.split(" = ")[0].endswith(" " + n)
                                              for n, _ in split.live_ins)])
    assert len([s for s in stores if s.startswith("hawk_pl")]) == len(split.live_ins) - free
    head = body.split("    while (true) {", 1)[0]
    assert all(f"    {d}" in head for d in split.sample_free)
    assert "unsigned hawk_ran = 0u;" in head and "long long hawk_" not in head
    assert "bool hawk_" not in head and "hawk_chunks" not in body
    if kernel is oscillator:
        assert free == 2, "dt/6 and dt/2 read no sample"


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
