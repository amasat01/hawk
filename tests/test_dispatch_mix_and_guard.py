# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The branch-mix counters of an ``"auto"`` dispatch, the selector guard of a
``segmented`` unit, and ``hawk.resolve_dispatch``.

Rows:

* **mix counters** — an ``"auto"`` dispatch counts, per branch, the samples
  that picked it into ``<group>_mix`` (K uint32 cells): after a launch the
  cells equal ``np.bincount`` of the clamped selector over the live samples
  exactly (terminated samples are not counted), on the host (``cuda`` arm
  marked ``gpu``: one warp ballot per branch); a forced ``"switch"`` binds no
  counter and its body text is the ``"auto"`` body minus the one count line;
* **selector guard** — the segmented units run on a sort that has gone
  stale (the selector drifted after the class sort): every drifted sample is
  marked stale and keeps its old value, the class plane holds the new
  classes, and a switch pass over just the stale samples completes the step
  to 0 differences from a switch over all samples;
* **resolve_dispatch** — an ``"auto"`` group resolves to the named policy
  (the same values as the forced kernel), a declared group or an unknown
  group or policy is refused, and an automatic kernel is resolved on its
  single step under the same name.
"""

from __future__ import annotations

import json
import pathlib

import _oracle as O
import numpy as np
import pytest

import hawk
import hawk.math as M
from hawk import Index, Mutable, Scalar, Terminated, Vector
from hawk.artifact import build_bundle
from hawk.emit import render_body
from hawk.ir import HawkError

N = 61
K = 3


def _branches(bs):
    return [bs[j] * (j + 1.0) for j in range(K)]


@hawk.kernel(steps=1)
def mix_auto(k: Index, bs: Vector[K], terminated: Terminated, y: Mutable[Scalar]):
    y = M.dispatch(k, _branches(bs), policy="auto")


@hawk.kernel(steps=1)
def mix_switch(k: Index, bs: Vector[K], terminated: Terminated, y: Mutable[Scalar]):
    y = M.dispatch(k, _branches(bs), policy="switch")


@hawk.kernel(steps=1)
def mix_seg(k: Index, bs: Vector[K], y: Mutable[Scalar]):
    y = M.dispatch(k, _branches(bs), policy="segmented")


@hawk.kernel(steps=1)
def plain_switch(k: Index, bs: Vector[K], y: Mutable[Scalar]):
    y = M.dispatch(k, _branches(bs), policy="switch")


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    root = pathlib.Path(tmp_path_factory.mktemp("mix"))
    return {
        "local": build_bundle([mix_auto, mix_switch, plain_switch], root / "local",
                              targets=("cuda", "host"), cache_dir=cache_dir),
        "seg": build_bundle([mix_seg], root / "seg", targets=("cuda", "host"),
                            cache_dir=cache_dir),
    }


def _sidecar(bundle, name):
    return json.loads((bundle.directory / f"{name}.json").read_text())


def _inputs(rng):
    k = rng.integers(-2, K + 2, N).astype(np.int64)          # out of range clamps
    bs = rng.uniform(0.5, 2.0, (K, N))
    stopped = rng.random(N) < 0.2
    return k, bs, stopped


def test_auto_counts_the_mix_and_a_forced_policy_counts_nothing(built):
    sc = _sidecar(built["local"], "mix_auto")
    assert sc["dispatch"][0]["mix"] == "dispatch0_mix"
    assert ["lookup", "dispatch0_mix"] in sc["arg_spec"]
    forced = _sidecar(built["local"], "mix_switch")
    assert "mix" not in forced["dispatch"][0]
    assert all(n != "dispatch0_mix" for _, n in forced["arg_spec"])
    auto_text = render_body(mix_auto.sinks, mix_auto.walk).text
    switch_text = render_body(mix_switch.sinks, mix_switch.walk).text
    lines = [ln for ln in auto_text.split("\n") if "hawk_abi::mix_count(" not in ln]
    assert "\n".join(lines) == switch_text


def _host_mix(built):
    rng = np.random.default_rng(3)
    k, bs, stopped = _inputs(rng)
    # The counters are K uint32 cells at the start of the plane (the
    # finished_count precedent); the host runtime binds integer planes as
    # int64, so the cells are read back through a uint32 view.
    mix = np.zeros(K, dtype=np.int64)
    y = np.zeros(N)
    hawk.run(hawk.load(built["local"].directory, "mix_auto"), k=k, bs=bs,
             terminated=stopped.copy(), y=y, dispatch0_mix=mix)
    return mix.view(np.uint32)[:K], np.bincount(np.clip(k, 0, K - 1)[~stopped],
                                                 minlength=K)


def test_the_mix_is_the_bincount_of_the_live_classes_on_the_host(built):
    got, want = _host_mix(built)
    np.testing.assert_array_equal(got, want)
    assert want.min() > 0 and want.sum() < N           # every class, some stopped


@pytest.mark.gpu
def test_the_mix_is_the_bincount_of_the_live_classes_on_the_device(built):
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    rng = np.random.default_rng(3)
    k, bs, stopped = _inputs(rng)
    mix = cp.zeros(K, dtype=cp.uint32)
    plugin = L.device_plugin(built["local"].directory, "mix_auto",
                             _sidecar(built["local"], "mix_auto"))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        k=cp.asarray(k), bs=cp.asarray(bs), terminated=cp.asarray(stopped),
        y=cp.zeros(N), dispatch0_mix=mix.view(cp.int32)).launch()
    want = np.bincount(np.clip(k, 0, K - 1)[~stopped], minlength=K)
    np.testing.assert_array_equal(cp.asnumpy(mix), want)


def test_a_stale_sort_is_guarded_and_a_switch_sweep_completes_it(built):
    """The class sort ran on ``k_sorted``; then the selector drifted for a few
    samples. The units still run over the OLD runs."""
    rng = np.random.default_rng(9)
    counts = [20, 25, 16]
    k_sorted = np.concatenate([np.full(c, j, dtype=np.int64) for j, c in enumerate(counts)])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    k_now = k_sorted.copy()
    drift = rng.choice(N, 9, replace=False)
    k_now[drift] = (k_sorted[drift] + rng.integers(1, K, 9)) % K
    bs = rng.uniform(0.5, 2.0, (K, N))
    y0 = rng.uniform(-1.0, 1.0, N)
    y = y0.copy()
    cls = np.full(N, -1, dtype=np.int64)
    stale = np.zeros(N, dtype=bool)
    seg = built["seg"]
    for j in range(K):
        name = f"mix_seg__k{j}"
        kern = O.load(seg.directory, name, _sidecar(seg, name))
        O.run_kernel(kern, N, base=0, count=counts[j], k=k_now, bs=bs, y=y,
                     dispatch0_offsets=offsets, dispatch0_class=cls,
                     dispatch0_stale=stale)
    np.testing.assert_array_equal(np.flatnonzero(stale), np.sort(drift))
    np.testing.assert_array_equal(y[drift], y0[drift])       # kept, not committed
    np.testing.assert_array_equal(cls, k_now)
    # the sweep: the switch kernel over just the stale samples
    swept = y.copy()
    full = np.zeros(N)
    hawk.run(hawk.load(built["local"].directory, "plain_switch"), k=k_now, bs=bs, y=full)
    swept[stale] = full[stale]
    assert int(np.sum(swept != full)) == 0


def test_resolve_dispatch_lowers_an_auto_group_and_refuses_the_rest():
    resolved = hawk.resolve_dispatch(mix_auto, {"dispatch0": "switch"})
    assert resolved.name == "mix_auto"
    assert [d.policy for d in resolved.walk.dispatches] == ["switch"]
    assert render_body(resolved.sinks, resolved.walk).text == render_body(
        mix_switch.sinks, mix_switch.walk).text
    seg = hawk.resolve_dispatch(mix_auto, {"dispatch0": "segmented"})
    assert [d.policy for d in seg.walk.dispatches] == ["segmented"]
    with pytest.raises(HawkError, match=r"declared policy='switch'; only an 'auto'"):
        hawk.resolve_dispatch(mix_switch, {"dispatch0": "segmented"})
    with pytest.raises(HawkError, match=r"'dispatch4' is not a dispatch group"):
        hawk.resolve_dispatch(mix_auto, {"dispatch4": "switch"})
    with pytest.raises(HawkError, match=r"resolves to one of 'switch', 'segmented'"):
        hawk.resolve_dispatch(mix_auto, {"dispatch0": "auto"})


def test_an_automatic_kernel_resolves_on_its_single_step():
    @hawk.kernel
    def walker(k: Index, bs: Vector[K], terminated: Terminated, x: Mutable[Scalar]):
        x = x + M.dispatch(k, _branches(bs), policy="auto")
        terminated = x > 10.0

    assert walker.step is not None
    resolved = hawk.resolve_dispatch(walker, {"dispatch0": "predicated"})
    assert resolved.name == "walker" and resolved.step is not None
    assert [d.policy for d in resolved.step.walk.dispatches] == ["predicated"]


def test_the_host_runtime_counts_into_scratch_when_the_mix_is_left_out(built):
    rng = np.random.default_rng(4)
    k, bs, _ = _inputs(rng)
    y = np.zeros(N)
    hawk.run(hawk.load(built["local"].directory, "mix_auto"), k=k, bs=bs,
             terminated=np.zeros(N, dtype=bool), y=y)
    np.testing.assert_array_equal(y, bs[np.clip(k, 0, K - 1), np.arange(N)]
                                  * (np.clip(k, 0, K - 1) + 1.0))


# -- a dispatch inside the fused automatic loop ---------------------------- #
def _scaled_branches(bs, scale):
    return [b * scale for b in _branches(bs)]


@hawk.kernel
def walker(k: Index, bs: Vector[K], terminated: Terminated, x: Mutable[Scalar]):
    # the branches read the carried x, so the dispatch is evaluated every trip
    # (a dispatch of loop-invariant values is evaluated, and counted, once
    # per launch, before the loop)
    x = x + M.dispatch(k, _scaled_branches(bs, 1.0 + 1e-3 * x), policy="auto")
    terminated = x > 1e9


@pytest.fixture(scope="module")
def fused(tmp_path_factory, cache_dir):
    root = pathlib.Path(tmp_path_factory.mktemp("fused_mix"))
    return build_bundle([walker], root, targets=("cuda", "host"), cache_dir=cache_dir)


def _fused_run(bundle, target, words):
    """``walker`` launched once per word of ``words`` (trips per launch);
    returns the branch counts after the last launch, and the classes."""
    import _deploy as L
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = _sidecar(bundle, "walker")
    if target == "host":
        plugin, structure, xp = (L.host_plugin(bundle.directory, "walker", sidecar),
                                 eexec.HostTeam, np)
    else:
        import cupy as xp

        plugin, structure = (L.device_plugin(bundle.directory, "walker", sidecar),
                             eexec.DeviceKernel)
    rng = np.random.default_rng(12)
    k = rng.integers(0, K, N).astype(np.int64)
    mix = xp.zeros(K, dtype=xp.uint32)
    word = xp.zeros(1, dtype=xp.int64)
    bound = eplan.plan(plugin, structure=structure).bind(
        k=xp.asarray(k), bs=xp.asarray(rng.uniform(0.5, 1.0, (K, N))),
        terminated=xp.zeros(N, dtype=bool), x=xp.zeros(N),
        finished_count=xp.zeros(1, dtype=xp.uint32).view(xp.int32),
        fused_steps=word, dispatch0_mix=mix.view(xp.int32))
    for w in words:
        word[0] = w
        bound.launch()
    got = mix.get() if hasattr(mix, "get") else mix
    return np.asarray(got), np.bincount(k, minlength=K)


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fused_loop_counts_mix(fused, target):
    """The group inside the fused loop is named and counted once per trip:
    a launch of ``w`` trips counts ``w`` times the class histogram, the same
    as ``w`` one-step launches."""
    (entry,) = _sidecar(fused, "walker")["dispatch"]
    assert entry["mix"] == "dispatch0_mix" and entry["static_selector"] is True
    trips, hist = _fused_run(fused, target, [7])
    np.testing.assert_array_equal(trips, 7 * hist)
    singles, _ = _fused_run(fused, target, [1] * 7)
    np.testing.assert_array_equal(singles, trips)


def test_a_carried_selector_inside_the_fused_loop_is_not_static():
    @hawk.kernel
    def drifter(bs: Vector[K], terminated: Terminated, x: Mutable[Scalar],
                c: Mutable[Index]):
        x = x + M.dispatch(c, _branches(bs), policy="auto")
        c = M.select(x > 1.0, c, c + 1)
        terminated = x > 1e9

    (entry,) = drifter.walk.dispatches
    assert ("mutable", "c") in entry.selector_reads
    assert entry.static_selector is False


def test_auto_segmented_runs_unfused(tmp_path, cache_dir):
    """Resolved to ``segmented``, an automatic kernel yields its SINGLE step
    split into units: K=1 launches of the units over a sorted class equal the
    switch step, step after step."""
    split = hawk.resolve_dispatch(walker, {"dispatch0": "segmented"})
    assert split.step is None and split.walk.finish == ("terminated", 1)
    ref = hawk.resolve_dispatch(walker, {"dispatch0": "switch"}).step
    # exact profile: the unit and the switch step contract no multiply-add
    # differently, so they agree bit for bit
    bundle = build_bundle([split], tmp_path / "split", targets=("host",),
                          cache_dir=cache_dir, host_profile="native")
    ref_bundle = build_bundle([ref.__class__("walker_ref", ref.sinks, ref.planes,
                                             kind=ref.kind)],
                              tmp_path / "ref", targets=("host",), cache_dir=cache_dir,
                              host_profile="native")
    names = {a.name for a in bundle.artifacts}
    assert {f"walker__k{j}" for j in range(K)} <= names
    rng = np.random.default_rng(13)
    counts = [17, 30, 14]
    k = np.concatenate([np.full(c, j, dtype=np.int64) for j, c in enumerate(counts)])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    bs = rng.uniform(0.5, 1.0, (K, N))
    x_split, x_ref = np.zeros(N), np.zeros(N)
    stopped = np.zeros(N, dtype=bool)
    cls, stale = np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)
    for _ in range(3):
        for j in range(K):
            name = f"walker__k{j}"
            unit = O.load(bundle.directory, name, _sidecar(bundle, name))
            O.run_kernel(unit, N, base=0, count=counts[j], k=k, bs=bs, x=x_split,
                         terminated=stopped, dispatch0_offsets=offsets,
                         dispatch0_class=cls, dispatch0_stale=stale,
                         finished_count=np.zeros(1, dtype=np.int64))
        hawk.run(hawk.load(ref_bundle.directory, "walker_ref"), k=k, bs=bs, x=x_ref,
                 terminated=np.zeros(N, dtype=bool))
        np.testing.assert_array_equal(x_split, x_ref)
    assert not stale.any() and (x_ref > 0).all()
