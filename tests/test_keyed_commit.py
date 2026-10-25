# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A keyed accumulate whose key moves in runs of lanes commits once per run.

The lane facts (:func:`hawk.ir.lanes.key_run`) name the run: ``proven`` (the
key is equal over every aligned run of W lanes), ``tested`` (no static
proof; a warp vote decides) or ``sorted`` (a ``Sorted[Index]`` key, runs
found at run time). The emitter spells ``aether::accum::run``'s call for it
(``add<W>``, ``addTested``, ``addSorted``, each with a compensated twin for
the compensated target); any other key keeps the per-lane atomic. Pinned:

* **Text.** The call per form on both targets from ONE body, the run state
  declared and closed by the entry, the header included only when called;
  an ``Index`` key keeps ``hawk_abi::accum_add``; a step kernel (two-path
  entry) never holds a keyed commit.
* **Commits at run time**, counted through aether's debug counters (the
  build's own ``AETHER_RUN_COUNT_COMMITS`` define; the default text has no
  counter): on the device one per aligned warp (n/32), per half warp for a
  run of 48 (n/16), one per uniform warp for the tested form, one per run
  inside each warp for the sorted form; on the host no lane commits.
* **The sum contract** against ``math.fsum`` per slot: the error over
  ``eps * sum|t|`` within 1 compensated, the default policy included, whose
  companion the runtime owns (a plain bound grows with the terms per slot; 8 at
  these run lengths; 8 for runs shorter than
  a warp, whose in-warp tree is plain), float32 and float64, host and device. The error in ULP of the exact sum is printed as a card
  line, never asserted.
* **The reverse** of a keyed read is the same lowering: ``vjp`` of
  ``th.at(seg)`` scatters ``bar_th`` through the same call; under a
  compensated kind it declares and binds the companion ``bar_th_c``.
"""

from __future__ import annotations

import ctypes
import math

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import HawkError, Index, Mutable, Scalar, Sorted, Table, Terminated
from hawk.artifact import build_bundle
from hawk.diff import vjp
from hawk.emit import FLOAT64, render_source
from hawk.ext import Kind, sink_policy
from hawk.targets import TARGETS
from hawk.trace.kernel import Kernel

NS = 8192
RUN = "aether::accum::run::"


def g(x):
    return x * x + 0.25 * x


@hawk.kernel
def kuni(x: Scalar, acc: Mutable[Scalar]):
    j, _ = m.split_index(NS)
    hawk.scatter(acc, policy="plain")[j] += x


@hawk.kernel
def kuni_c(x: Scalar, acc: Mutable[Scalar]):
    j, _ = m.split_index(NS)
    hawk.scatter(acc, policy="compensated")[j] += x


@hawk.kernel
def kfb(x: Scalar, acc: Mutable[Scalar]):
    j, _ = m.split_index(48)
    hawk.scatter(acc, policy="plain")[j] += x


@hawk.kernel
def kfb_c(x: Scalar, acc: Mutable[Scalar]):
    j, _ = m.split_index(48)
    hawk.scatter(acc, policy="compensated")[j] += x


# The second divisor (6) makes the lane axes a non-chain, so no run is proven:
# the key moves in runs of 64 that a warp vote finds.
@hawk.kernel
def ktst(x: Scalar, acc: Mutable[Scalar]):
    lane = m.sample_index()
    hawk.scatter(acc, policy="plain")[lane / 64] += x + 0.0 * (lane - (lane / 6) * 6)


@hawk.kernel
def ktst_c(x: Scalar, acc: Mutable[Scalar]):
    lane = m.sample_index()
    hawk.scatter(acc, policy="compensated")[lane / 64] += x + 0.0 * (lane - (lane / 6) * 6)


@hawk.kernel
def ksrt(x: Scalar, seg: Sorted[Index], acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="plain")[seg] += x


@hawk.kernel
def ksrt_c(x: Scalar, seg: Sorted[Index], acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="compensated")[seg] += x


@hawk.kernel
def knon(x: Scalar, key: Index, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="plain")[key] += x


# The default policy: compensated, with the companion owned by the runtime.
@hawk.kernel
def kuni_a(x: Scalar, acc: Mutable[Scalar]):
    j, _ = m.split_index(NS)
    hawk.scatter(acc)[j] += x


@hawk.kernel
def kfb_a(x: Scalar, acc: Mutable[Scalar]):
    j, _ = m.split_index(48)
    hawk.scatter(acc)[j] += x


@hawk.kernel
def ktst_a(x: Scalar, acc: Mutable[Scalar]):
    lane = m.sample_index()
    hawk.scatter(acc)[lane / 64] += x + 0.0 * (lane - (lane / 6) * 6)


@hawk.kernel
def ksrt_a(x: Scalar, seg: Sorted[Index], acc: Mutable[Scalar]):
    hawk.scatter(acc)[seg] += x


@hawk.kernel
def kread(x: Scalar, seg: Sorted[Index], th: Table[Scalar], y: Mutable[Scalar]):
    y = th.at(seg) * x


@hawk.kernel
def kread_axis(x: Scalar, th: Table[Scalar], y: Mutable[Scalar]):
    j, s = m.split_index(NS)
    y = th.at(j) * g(x)


@hawk.kernel
def kfem(w: Scalar, conn: Table[Index], f: Table[Scalar], y: Mutable[Scalar]):
    e, v = m.split_index(4)
    y = f.at(conn.at(e * 4 + v)) * w


@hawk.kernel
def kspline(x: Scalar, th: Table[Scalar], y: Mutable[Scalar]):
    j, s = m.split_index(NS)
    y = th.at(j * 12 + m.index(x, 9) + 2) * g(x)


def _vjp(primal, name):
    return Kernel(name, vjp(primal, wrt=("f",) if primal is kfem else ("th",)))


VREAD, VAXIS, VFEM, VSPLINE = (_vjp(kread, "kread_vjp"), _vjp(kread_axis, "kread_axis_vjp"),
                               _vjp(kfem, "kfem_vjp"), _vjp(kspline, "kspline_vjp"))
#: The compensated reverse: the kernel declares and binds the companion ``bar_th_c``.
VREAD_C = Kernel("kread_vjp_c", vjp(kread, wrt=("th",)),
                 kind=Kind("keyed_reverse_compensated",
                           sink=sink_policy("compensated", into="bar_th_c")))

#: kernel -> the call its keyed commit spells.
CALLS = {kuni: "add<32>(aout_acc, ", kfb: "add<16>(aout_acc, ",
         ktst: "addTested(aout_acc, ", ksrt: "addSorted(aout_acc, ",
         kuni_c: "addCompensated<32>(aout_acc, aout_acc_c, ",
         kfb_c: "addCompensated<16>(aout_acc, aout_acc_c, ",
         ktst_c: "addTestedCompensated(aout_acc, aout_acc_c, ",
         ksrt_c: "addSortedCompensated(aout_acc, aout_acc_c, ",
         kuni_a: "addCompensated<32>(aout_acc, aout_acc_c, ",
         kfb_a: "addCompensated<16>(aout_acc, aout_acc_c, ",
         ktst_a: "addTestedCompensated(aout_acc, aout_acc_c, ",
         ksrt_a: "addSortedCompensated(aout_acc, aout_acc_c, ",
         VREAD: "addSorted(aout_bar_th, ", VAXIS: "add<32>(aout_bar_th, ",
         VREAD_C: "addSortedCompensated(aout_bar_th, aout_bar_th_c, "}


def _text(kernel, target):
    return render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                         mode=FLOAT64, kind=kernel.kind)


@pytest.mark.parametrize("kernel", list(CALLS), ids=lambda k: k.name)
def test_a_run_key_spells_one_run_commit_on_both_targets(kernel):
    cuda, host = _text(kernel, "cuda"), _text(kernel, "host")
    assert cuda.body == host.body
    state = "hawk_run_aout_bar_th" if kernel in (VREAD, VAXIS, VREAD_C) else "hawk_run_aout_acc"
    target = state[len("hawk_run_"):]
    for src in (cuda, host):
        assert src.body.count(RUN + CALLS[kernel]) == 1
        assert src.body.rstrip().endswith(f"{state});")
        assert "hawk_abi::accum_add" not in src.body
        assert src.text.count("#include <aether/accum/run.h>\n") == 1
        assert f"    auto {state} = {RUN}open({target});" in src.text
        assert f"    {RUN}close({state});" in src.text
        assert "AETHER_RUN_COUNT_COMMITS" not in src.text


@pytest.mark.parametrize("kernel", [knon, VFEM, VSPLINE], ids=lambda k: k.name)
def test_an_index_or_data_key_keeps_the_per_lane_commit(kernel):
    for target in ("cuda", "host"):
        src = _text(kernel, target)
        assert "hawk_abi::accum_add(" in src.body
        assert RUN not in src.text and "hawk_run_" not in src.text


def test_a_step_kernel_never_holds_a_keyed_commit():
    """The run state is entry-local to one loop; a step kernel's two-path entry
    never needs it, because a keyed accumulate is not a step's commit."""
    @hawk.kernel
    def stepped(x: Scalar, seg: Sorted[Index], terminated: Terminated,
                k: Mutable[Scalar], acc: Mutable[Scalar]):
        hawk.scatter(acc)[seg] += x
        kn = k + 1.0
        k = kn
        terminated = kn >= 3.0

    assert stepped.one_step is None
    with pytest.raises(HawkError, match="accum_out sink"):
        hawk.steps(stepped, "auto")


# ---- run time: commits and the sum contract ----------------------------------
N = 64 * NS
COUNT = ("AETHER_RUN_COUNT_COMMITS=1",)
TARGET_ARMS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


def _mixed_runs(n):
    """Sorted segment ids whose runs are 1..40 lanes long."""
    lengths = np.random.default_rng(7).integers(1, 41, n)
    return np.repeat(np.arange(n), lengths)[:n].astype(np.int64)


def _warp_runs(key):
    w = key.reshape(-1, 32)
    return int(w.shape[0] + np.count_nonzero(w[:, 1:] != w[:, :-1]))


#: The default-policy twin of each plain kernel.
AUTO = {kuni: kuni_a, kfb: kfb_a, ktst: ktst_a, ksrt: ksrt_a}

#: case -> (plain kernel, compensated kernel, n, key(l), the device's run commits)
CASES = {
    "proven_8192": (kuni, kuni_c, N, np.arange(N) // NS, N // 32),
    "proven_48": (kfb, kfb_c, 1024 * 48, np.arange(1024 * 48) // 48,
                  1024 * 48 // 16),
    "tested_64": (ktst, ktst_c, N, np.arange(N) // 64, N // 32),
    "sorted_8192": (ksrt, ksrt_c, N, np.arange(N) // NS, N // 32),
    "sorted_mixed": (ksrt, ksrt_c, N, _mixed_runs(N),
                     _warp_runs(_mixed_runs(N))),
}


@pytest.fixture(scope="module", params=["float32", "float64"])
def bundles(request, tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp(f"keyed_{request.param}")
    kernels = {k for c in CASES.values() for k in c[:2]} | set(AUTO.values())
    out = {k.name: build_bundle([k], root / k.name, mode=request.param,
                                targets=("cuda", "host"), cache_dir=cache_dir,
                                defines=COUNT)
           for k in kernels}
    return out, request.param


def _commits(plugin, target):
    """(get, reset) of aether's two run-commit counters in the loaded entry."""
    if target == "host":
        cells = (ctypes.c_ulonglong * 2).in_dll(plugin._keepalive, "aether_run_commits")

        def reset():
            cells[0] = cells[1] = 0
        return (lambda: (int(cells[0]), int(cells[1]))), reset
    import cupy as cp

    arr = cp.ndarray((2,), dtype=cp.uint64,
                     memptr=plugin._keepalive[1].module.get_global("aether_run_commits"))
    return (lambda: tuple(int(v) for v in arr.get())), (lambda: arr.fill(0))


def _launch(bundle, name, target, planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    get, reset = _commits(plugin, target)
    reset()
    eplan.plan(plugin, structure=structure).bind(**planes).launch()
    return get()


def _bound(compensated, whole_warps):
    """The contract, over ``eps * sum|t|``: plain grows with the terms per slot
    (8 at these sizes); 1 compensated when every
    run fills whole warps. A run shorter than a warp is summed by the plain
    in-warp tree before its one compensated commit, so it keeps the plain 8."""
    return 1.0 if compensated and whole_warps else 8.0


def _error(got, terms, key, slots, mode, what):
    """The worst slot's error over ``eps(mode) * sum|t|`` (the contract), and
    in ULP of the exact sum (a card line, printed, never asserted)."""
    order = np.argsort(key, kind="stable")
    bounds = np.searchsorted(key[order], np.arange(slots + 1))
    ordered = terms[order]
    runs = [ordered[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    ref = np.array([math.fsum(r) for r in runs])
    scale = np.finfo(mode).eps * np.array([math.fsum(np.abs(r)) for r in runs])
    err = np.abs(got - ref)
    scaled = float(np.max(err / scale))
    ulp = float(np.max(err / np.spacing(np.abs(ref).astype(mode)).astype(np.float64)))
    print(f"card {what} {mode}: err/(eps*sum|t|) {scaled:.3g}, ULP {ulp:.3g}")
    return scaled


@pytest.mark.parametrize("policy", ["plain", "compensated", "auto"])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("target", TARGET_ARMS)
def test_one_commit_per_run_within_the_sum_contract(bundles, target, case, policy):
    """``auto`` binds no companion: the runtime owns it, and the target alone
    holds the compensated sum."""
    built, mode = bundles
    plain, comp, n, key, device_runs = CASES[case]
    kernel = {"plain": plain, "compensated": comp, "auto": AUTO.get(plain)}[policy]
    if kernel is None:
        pytest.skip("no default-policy twin for this case")
    compensated = policy != "plain"
    xp = np if target == "host" else __import__("cupy")
    x = np.random.default_rng(1234).uniform(-1, 1, n).astype(mode)
    slots = int(key.max()) + 1
    planes = {"x": xp.asarray(x), "acc": xp.zeros(slots, mode)}
    if policy == "compensated":
        planes["acc_c"] = xp.zeros(slots, mode)
    if any(name == "seg" for _, name in kernel.walk.arg_spec):
        planes["seg"] = xp.asarray(key)
    run, lane = _launch(built[kernel.name], kernel.name, target, planes)
    if target == "cuda":
        xp.cuda.Device().synchronize()
    got = sum(np.asarray(planes[p].get() if target == "cuda" else planes[p]).astype(np.float64)
              for p in (("acc", "acc_c") if policy == "compensated" else ("acc",)))
    terms = x.astype(np.float64)        # the plane's values: exact terms on every arm
    what = f"{case} {policy} {target}"
    assert _error(got, terms, key, slots, mode, what) <= _bound(compensated, case != "sorted_mixed")
    assert lane == 0
    if target == "cuda":
        assert run == device_runs
    else:
        assert slots <= run < n // 8


@hawk.kernel
def knon_c(x: Scalar, key: Index, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="compensated")[key] += x


@hawk.kernel
def kone_c(x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="compensated")[0] += x


@hawk.kernel
def knon_a(x: Scalar, key: Index, acc: Mutable[Scalar]):
    hawk.scatter(acc)[key] += x


@hawk.kernel
def kone_a(x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc)[0] += x


#: 200k terms into 16 slots: unsorted keys commit per lane, sorted keys and
#: one constant key once per run.
NB, NBIG = 16, 200_000
BIG = {"unsorted": knon_c, "sorted": ksrt_c, "uniform": kone_c}
BIG_AUTO = {"unsorted": knon_a, "sorted": ksrt_a, "uniform": kone_a}


def test_a_compensated_commit_is_one_two_sum_without_a_fence():
    """The hot path: the compensated commit is the atomic two-sum into the
    target and its companion, nothing more; the unit's second entry
    ``<kernel>__fold_<target>`` folds the pair after the launch. The plain
    commit has neither; the default policy spells the compensated one."""
    for target in ("cuda", "host"):
        plain = _text(knon, target).text
        assert "accum_add_compensated" not in plain and "__fold_" not in plain
        text = _text(knon_c, target).text
        assert "hawk_abi::accum_add_compensated(" in text
        assert "__threadfence" not in text and "atomic_thread_fence" not in text
        assert text.count("hawk_abi::fold_companion(aout_acc, aout_acc_c, i);") == 1
        assert ("knon_c__fold_acc_host(" if target == "host" else "knon_c__fold_acc(") in text
        assert _text(knon_a, target).body == _text(knon_c, target).body.replace(
            "knon_c", "knon_a")


def _plan(kernel, target, mode, tmp_path, cache_dir):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = build_bundle([kernel], tmp_path, mode=mode, targets=("cuda", "host"),
                          cache_dir=cache_dir)
    sidecar = sidecar_of(bundle, kernel.name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, kernel.name, sidecar), eexec.HostTeam
    else:
        plugin, structure = (L.device_plugin(bundle.directory, kernel.name, sidecar),
                             eexec.DeviceKernel)
    return eplan.plan(plugin, structure=structure)


def _big_planes(keys, mode, xp):
    rng = np.random.default_rng(4)
    key = np.minimum((rng.beta(2, 5, NBIG) * NB).astype(np.int64), NB - 1)
    if keys == "sorted":
        key = np.sort(key)
    elif keys == "uniform":
        key = np.zeros(NBIG, np.int64)
    x = rng.uniform(0.5, 1.5, NBIG).astype(mode)
    return key, x, {"x": xp.asarray(x), "key": xp.asarray(key), "seg": xp.asarray(key),
                    "acc": xp.zeros(NB, mode)}


@pytest.mark.parametrize("mode", ["float32", "float64"])
@pytest.mark.parametrize("keys", list(BIG_AUTO))
@pytest.mark.parametrize("target", TARGET_ARMS)
def test_a_default_keyed_sum_is_delivered_in_the_target_alone(target, keys, mode,
                                                                tmp_path, cache_dir):
    """No companion is bound: the plan owns one, and the target alone holds the
    compensated sum, within 1 of ``eps * sum|t|`` per slot."""
    kernel = BIG_AUTO[keys]
    xp = np if target == "host" else __import__("cupy")
    key, x, planes = _big_planes(keys, mode, xp)
    names = [nm for _, nm in kernel.walk.arg_spec if nm != "acc_c"]
    assert ("accum_out", "acc_c") in kernel.walk.arg_spec
    plan = _plan(kernel, target, mode, tmp_path, cache_dir)
    plan.bind(**{k: planes[k] for k in names}).launch()
    if target == "cuda":
        xp.cuda.Device().synchronize()
    slots = int(key.max()) + 1
    got = np.asarray(planes["acc"].get() if target == "cuda" else planes["acc"])[:slots]
    what = f"{keys} {NBIG} into {slots} default {target}"
    assert _error(got.astype(np.float64), x.astype(np.float64), key, slots, mode, what) <= 1.0


@pytest.mark.parametrize("kernel", [knon_a, knon_c], ids=["auto", "compensated"])
@pytest.mark.parametrize("target", TARGET_ARMS)
def test_replays_into_one_sink_keep_the_companion_across_launches(target, kernel, tmp_path,
                                                                   cache_dir):
    """The companion is zeroed once, at bind, and keeps the residual of the
    fold after each launch: the target alone stays within 1 of
    ``eps * sum|w|`` after 100 replays into the same sink."""
    import eagle

    R = 100
    xp = np if target == "host" else __import__("cupy")
    key, x, planes = _big_planes("unsorted", "float64", xp)
    plan = _plan(kernel, target, "float64", tmp_path, cache_dir)
    bound = plan.bind(x=planes["x"], key=planes["key"], acc=planes["acc"])
    graph = eagle.Graph().add(bound).build()
    graph.launch(R)
    if target == "cuda":
        xp.cuda.Device().synchronize()
    got = np.asarray(planes["acc"].get() if target == "cuda" else planes["acc"])
    assert _error(got, np.tile(x.astype(np.float64), R), np.tile(key, R), NB, "float64",
                  f"{kernel.name} {R} replays {target}") <= 1.0


@pytest.mark.gpu
def test_a_captured_replay_reuses_the_owned_companion(tmp_path, cache_dir):
    """Three replays of one captured launch add the same terms three times:
    the target alone holds the compensated total, so the companion the capture
    recorded is the one every replay adds into, and the fold node rides
    in the capture."""
    import cupy as cp
    import eagle

    key, x, planes = _big_planes("unsorted", "float64", cp)
    plan = _plan(knon_a, "cuda", "float64", tmp_path, cache_dir)
    bound = plan.bind(x=planes["x"], key=planes["key"], acc=planes["acc"])
    graph = eagle.Graph().add(bound).build()
    for _ in range(3):
        graph.launch()
    cp.cuda.Device().synchronize()
    got = planes["acc"].get()
    terms = np.tile(x.astype(np.float64), 3)
    assert _error(got, terms, np.tile(key, 3), NB, "float64", "replay x3 default") <= 1.0


@pytest.mark.parametrize("target", TARGET_ARMS)
def test_the_owned_companion_is_not_returned_and_not_asked_for(target, tmp_path, cache_dir):
    xp = np if target == "host" else __import__("cupy")
    key, x, planes = _big_planes("sorted", "float64", xp)
    plan = _plan(ksrt_a, target, "float64", tmp_path, cache_dir)
    got = plan.run(x=planes["x"], seg=planes["seg"], acc=planes["acc"])
    assert not isinstance(got, dict) and got.shape == (NB,)
    assert ("accum_out", "acc_c") in plan.plugin.arg_spec
    made = plan.run(x=planes["x"], seg=planes["seg"])      # the run makes the target too
    for out in (got, made):
        out = np.asarray(out.get() if hasattr(out, "get") else out)[:NB]
        assert _error(out, x, key, NB, "float64", f"run sorted default {target}") <= 1.0


def test_hawks_own_host_door_folds_the_owned_companion(tmp_path, cache_dir):
    """``hawk.runtime.run`` owns the companion it is not given and folds it
    into the target after the launch: the target alone holds the sum."""
    from hawk import runtime

    key, x, planes = _big_planes("unsorted", "float64", np)
    bundle = build_bundle([knon_a], tmp_path, targets=("host",), cache_dir=cache_dir)
    runtime.run(runtime.load(bundle.directory, knon_a.name), x=x, key=key, acc=planes["acc"])
    assert _error(planes["acc"], x, key, NB, "float64", "hawk door default host") <= 1.0


def test_a_launch_allocates_nothing_for_the_owned_companion(tmp_path, cache_dir):
    """The companion is made at bind: a hundred launches of the bound plan
    hold the same memory (the target is big enough to show one more plane)."""
    import tracemalloc

    slots = 1 << 18
    plan = _plan(ksrt_a, "host", "float64", tmp_path, cache_dir)
    seg = np.sort(np.random.default_rng(1).integers(0, slots, 4096)).astype(np.int64)
    bound = plan.bind(x=np.ones(4096), seg=seg, acc=np.zeros(slots))
    bound.launch()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(100):
        bound.launch()
    grown = sum(s.size_diff for s in tracemalloc.take_snapshot().compare_to(before, "filename"))
    tracemalloc.stop()
    assert grown < slots, f"a hundred launches grew memory by {grown} bytes"


@pytest.mark.parametrize("mode", ["float32", "float64"])
@pytest.mark.parametrize("keys", list(BIG))
@pytest.mark.parametrize("target", TARGET_ARMS)
def test_a_companion_the_caller_binds_keeps_target_plus_companion(target, keys, mode,
                                                                   tmp_path, cache_dir):
    """A companion the caller binds is the caller's: never folded, and target
    plus companion is the compensated sum, within 1 of ``eps * sum|t|`` per
    slot."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    kernel = BIG[keys]
    bundle = build_bundle([kernel], tmp_path, mode=mode, targets=("cuda", "host"),
                          cache_dir=cache_dir)
    rng = np.random.default_rng(4)
    key = np.minimum((rng.beta(2, 5, NBIG) * NB).astype(np.int64), NB - 1)
    if keys == "sorted":
        key = np.sort(key)
    elif keys == "uniform":
        key = np.zeros(NBIG, np.int64)
    x = rng.uniform(0.5, 1.5, NBIG).astype(mode)
    xp = np if target == "host" else __import__("cupy")
    planes = {"x": xp.asarray(x), "key": xp.asarray(key), "seg": xp.asarray(key),
              "acc": xp.zeros(NB, mode), "acc_c": xp.zeros(NB, mode)}
    sidecar = sidecar_of(bundle, kernel.name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, kernel.name, sidecar), eexec.HostTeam
    else:
        plugin, structure = (L.device_plugin(bundle.directory, kernel.name, sidecar),
                             eexec.DeviceKernel)
    names = [nm for _, nm in kernel.walk.arg_spec]
    eplan.plan(plugin, structure=structure).bind(**{k: planes[k] for k in names}).launch()
    if target == "cuda":
        xp.cuda.Device().synchronize()
    slots = int(key.max()) + 1
    got = {p: np.asarray(planes[p].get() if target == "cuda" else planes[p])[:slots]
           .astype(np.float64) for p in ("acc", "acc_c")}
    terms = x.astype(np.float64)
    what = f"{keys} {NBIG} into {slots} compensated {target} pair"
    assert _error(got["acc"] + got["acc_c"], terms, key, slots, mode, what) <= 1.0


def test_a_compensated_reverse_declares_its_companion():
    assert ("accum_out", "bar_th_c") in VREAD_C.walk.arg_spec
    assert ("accum_out", "bar_th_c") not in VREAD.walk.arg_spec
    assert type(VREAD_C.sinks).__name__ == "Derived" and VREAD_C.sinks.wrt == ("th",)


@pytest.mark.parametrize("compensated", [False, True], ids=["plain", "compensated"])
@pytest.mark.parametrize("target", TARGET_ARMS)
def test_the_reverse_of_a_sorted_read_commits_once_per_run(target, compensated, tmp_path,
                                                           cache_dir):
    mode = "float64"
    kernel = VREAD_C if compensated else VREAD
    kw = dict(mode=mode, targets=("cuda", "host"), cache_dir=cache_dir, defines=COUNT)
    primal = build_bundle([kread], tmp_path / "p", **kw)
    bundle = build_bundle([kernel], tmp_path / "v",
                          derivative={kernel.name: (kread.name, primal.digest)}, **kw)
    xp = np if target == "host" else __import__("cupy")
    rng = np.random.default_rng(99)
    key = _mixed_runs(N)
    slots = int(key.max()) + 1
    x, by = rng.uniform(0.5, 1.5, N), rng.uniform(0.5, 1.5, N)   # positive terms: no cancellation
    planes = {"x": xp.asarray(x), "seg": xp.asarray(key), "bar_y": xp.asarray(by),
              "bar_th": xp.zeros(slots), "bar_th_c": xp.zeros(slots)}
    names = [nm for _, nm in kernel.walk.arg_spec]
    run, lane = _launch(bundle, kernel.name, target, {k: planes[k] for k in names})
    if target == "cuda":
        xp.cuda.Device().synchronize()
    got = sum(np.asarray(planes[p].get() if target == "cuda" else planes[p])
              for p in (("bar_th", "bar_th_c") if compensated else ("bar_th",)))
    what = f"reverse {'compensated' if compensated else 'plain'} {target}"
    assert _error(got, by * x, key, slots, mode, what) <= _bound(compensated, False)
    assert lane == 0
    if target == "cuda":
        assert run == _warp_runs(key)
    else:
        assert slots <= run < N // 8


@pytest.mark.gpu
def test_a_run_commit_compiles_through_the_sealed_payload(monkeypatch, tmp_path):
    """The pip-only device route: NVRTC serving the sealed payload's in-memory
    headers, which carry ``aether/accum/run.h``."""
    from hawk.compile import CompileOptions, compile_source, cubin_available
    from hawk.compile.payload import current_payload, reset_payload_memo

    if not cubin_available():
        pytest.skip("no NVRTC binding or no CUDA device")
    monkeypatch.setenv("HAWK_DEVICE_COMPILER", "nvrtc")
    reset_payload_memo()
    assert "aether/accum/run.h" in current_payload().device_headers
    src = render_source(ksrt_c.name, ksrt_c.sinks, ksrt_c.walk, TARGETS["cuda"],
                        mode=FLOAT64, kind=ksrt_c.kind)
    result = compile_source(src.text, ksrt_c.name,
                            CompileOptions(target="cuda", cache_dir=str(tmp_path / "cache")))
    assert result.artifact.is_file()
