# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A sample whose mask is set when the launch reaches it does NO work.

The guard used to wrap the COMMITS only: a terminated sample skipped its stores
but still paid every load and every arithmetic operation of the body. The guard
now opens before the first load and closes after the last commit, so the whole
body — loads, arithmetic, loops, stores, scattered accumulates — runs for a
live sample only. The masks are ``terminated``-role planes the body only reads,
so nothing the commits store changes.

Three observations:

* **Structure.** A guarded body is EXACTLY the mask-free body (``MASK_FREE``,
  the data-only kind that gates nothing) indented one level inside
  ``if (!mask) { ... }`` — the guard is the first line, so it precedes every
  load, and no second test of the mask survives inside (a reverse loop's own
  scatter used to re-test it every iteration). Derived kernels (vjp/jvp, #30)
  inherit the same wrap for the primal's mask. A kernel whose guard binds no
  mask renders unchanged.
* **Semantics, compiled, host AND device.** On a mixed batch a live sample's
  outputs equal the mask-free kind's, and a terminated sample keeps its prior
  value and adds nothing to a scattered accumulate — primal and derived.
* **No work.** The timing half of the evidence is GPU-load-dependent and so is
  not a test row: on the eagle performance card's oscillator at N = 1e6 with
  the batch's stop steps sorted (whole warps finish together), the 22 %-active
  batch's time over the all-active batch's fell from 0.56 to 0.30. With the
  card's own random interleaving the GPU ratio moves only from 0.90 to 0.86:
  88 % of 32-lane warps still hold at least one live sample on an average
  step, and a warp runs the body if any lane does.

RED: run against the commit-only guard, the four wrap rows and the reverse-loop
row fail (the first body line is a load, not the guard; the loop re-tests the
mask); the two-mask and guard-free rows pass there too (their bodies hoist
nothing, so there is nothing to move), and so do the compiled rows — the
outputs are the same before and after, which is the point.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import _kernels as K
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Accum, Index, Kernel, Mutable, Scalar, Table, Terminated, kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ext import Guard, Kind
from hawk.ir import canonical

N = 16
GUARD = "if (!trm_terminated[i].eval()) {"


@kernel
def _busy(x: Scalar, lane: Index, terminated: Terminated, y: Mutable[Scalar],
          acc: Accum[Scalar]):
    s = x * x + 0.25
    for _ in range(5):
        s = s * 0.5 + x
    acc.add(s, at=lane)
    y = y + s * x


@kernel
def _busy_d(x: Scalar, lane: Index, terminated: Terminated, y: Mutable[Scalar],
            acc: Accum[Scalar]):
    """:func:`_busy` without a launch-start read a derivative refuses."""
    s = x * x + 0.25
    for _ in range(5):
        s = s * 0.5 + x
    acc.add(s, at=lane)
    y = s * x


@kernel
def _smooth(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    """The compiled derivative subject: own-column only, so the primal and
    both derivatives share one bundle's execution axis."""
    s = x * x + 0.25
    for _ in range(5):
        s = s * 0.5 + x
    y = s * x


@kernel
def _loop_gather(x: Table[Scalar], theta: Table[Scalar], terminated: Terminated,
                 y: Mutable[Scalar]):
    total = 0.0
    for e in range(4):
        total = total + x.at(e) * theta.at(e)
    y = total


def _derived(primal, transform, wrt, name):
    return Kernel(name, transform(primal, wrt=wrt))


def _subjects():
    return {
        "busy": (_busy.sinks, _busy.walk),
        "busy_d_vjp": _sinks_walk(vjp(_busy_d, wrt=("x",))),
        "busy_d_jvp": _sinks_walk(jvp(_busy_d, wrt=("x",))),
        "loop_gather_vjp": _sinks_walk(vjp(_loop_gather, wrt=("theta",))),
    }


def _sinks_walk(sinks):
    return sinks, canonical(sinks)


def _unindent(lines):
    assert all(line.startswith("    ") for line in lines if line.strip()), lines
    return [line[4:] for line in lines]


# --------------------------------------------------------------------------- #
# -- structure.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_subjects()))
def test_the_guard_wraps_the_whole_body_before_the_first_load(name):
    sinks, walk = _subjects()[name]
    guarded = render_body(sinks, walk).text.splitlines()
    free = render_body(sinks, walk, kind=D.MASK_FREE).text.splitlines()
    assert guarded[0].strip() == GUARD, (
        f"{name}: the mask test must be the body's FIRST line, ahead of every "
        "load:\n" + "\n".join(guarded))
    assert guarded[-1].strip() == "}"
    assert _unindent(guarded[1:-1]) == free, (
        f"{name}: inside the guard the body must be exactly the mask-free body")
    assert sum(GUARD in line for line in guarded) == 1, (
        f"{name}: the mask is tested once per sample, never again inside")


def test_a_reverse_loop_scatter_is_not_re_guarded_per_iteration():
    """The shape that used to re-test the mask on EVERY iteration: a reverse
    loop's own scatter commit (``hawk.diff.transform._masked_sink``'s Loop)."""
    sinks, walk = _subjects()["loop_gather_vjp"]
    text = render_body(sinks, walk).text
    assert "for (" in text, "the fixture must lower a loop or the row tests nothing"
    assert text.count("trm_terminated[i].eval()") >= 1
    loop_at = text.index("for (")
    assert GUARD not in text[loop_at:], text


def test_a_two_mask_guard_wraps_the_whole_body():
    two = Kind("two_mask", guard=Guard(masks=("terminated", "rejected")))
    walk = D.diagnostic_two.walk
    lines = render_body(D.diagnostic_two.sinks, walk, kind=two).text.splitlines()
    free = render_body(D.diagnostic_two.sinks, walk, kind=D.MASK_FREE).text
    assert lines[0].strip() == (
        "if (!(trm_terminated[i].eval() || trm_rejected[i].eval())) {")
    assert _unindent(lines[1:-1]) == free.splitlines()


def test_a_guard_free_kernel_renders_exactly_as_the_mask_free_kind():
    """No bound mask, no scope: not one character of the body moves."""
    checked = 0
    for name, k in K.CORPUS.items():
        if any(role == "terminated" for role, _ in k.walk.arg_spec):
            continue
        assert render_body(k.sinks, k.walk).text == render_body(
            k.sinks, k.walk, kind=D.MASK_FREE).text, name
        checked += 1
    assert checked, "the corpus must hold guard-free kernels or the row is empty"


# --------------------------------------------------------------------------- #
# -- semantics, compiled: host and device.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    units = {"primal": [_busy],
             "derived": [_smooth, _derived(_smooth, vjp, ("x",), "smooth_vjp"),
                         _derived(_smooth, jvp, ("x",), "smooth_jvp")]}
    root = tmp_path_factory.mktemp("h36")
    return {(unit, tag): build_bundle(kernels, root / f"{unit}_{tag}",
                                      targets=("cuda", "host"),
                                      cache_dir=cache_dir, **kw)
            for unit, kernels in units.items()
            for tag, kw in (("guarded", {}), ("free", {"kind": D.MASK_FREE}))}


def _inputs():
    rng = np.random.default_rng(36)
    x = rng.uniform(-1.0, 1.0, N)
    lane = (np.arange(N) % 4).astype(np.int64)
    terminated = np.zeros(N, dtype=bool)
    terminated[[1, 2, 5, 8, 9, 10, 15]] = True
    return x, lane, terminated


def _host(bundle, name, **kw):
    import eagle.exec as eexec
    from eagle import plan as eplan

    assert eexec.HostTeam.tile_count(eexec.Partition.whole(N), 0) == 1, (
        "one tile: the scattered accumulate shares lanes across samples")
    plugin = L.host_plugin(bundle.directory, name, sidecar_of(bundle, name))
    return eplan.plan(plugin, structure=eexec.HostTeam).run(**kw)


def _device(bundle, name, **kw):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.device_plugin(bundle.directory, name, sidecar_of(bundle, name))
    return eplan.plan(plugin, structure=eexec.DeviceKernel).run(**kw)


def _np(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _outputs(got):
    # eagle.plan.Plan.run returns a dict keyed by plane name for several
    # outputs; dict insertion order follows arg_spec's own order, the same
    # order this file's positional unpacking already expects.
    values = (got.values() if isinstance(got, dict)
             else got if isinstance(got, tuple) else (got,))
    return [_np(a) for a in values]


def _check(run, bundles):
    x, lane, terminated = _inputs()
    live = ~terminated
    prior = np.linspace(3.0, 4.0, N)

    y_g, acc_g = _outputs(run(bundles["primal", "guarded"], "_busy", x=x, lane=lane,
                              terminated=terminated, y=prior.copy()))
    y_f, acc_f = _outputs(run(bundles["primal", "free"], "_busy", x=x, lane=lane,
                              terminated=terminated, y=prior.copy()))
    np.testing.assert_array_equal(y_g[live], y_f[live])
    np.testing.assert_array_equal(y_g[terminated], prior[terminated])
    # what the mask-free kind adds per lane, re-derived from its live samples
    # alone: y = prior + s*x, so s = (y - prior) / x for the free run.
    s = (y_f - prior) / x
    want = np.zeros_like(acc_g)
    np.add.at(want, lane[live], s[live])
    np.testing.assert_allclose(acc_g[:4], want[:4], rtol=1e-12, atol=0)
    assert not np.array_equal(acc_g[:4], acc_f[:4]), (
        "the mask-free run must differ, or the mask selects nothing")

    seed = np.linspace(0.5, 1.5, N)
    for name, kw in (("smooth_vjp", {"bar_y": seed}),
                     ("smooth_jvp", {"dot_x": seed})):
        got_g = _outputs(run(bundles["derived", "guarded"], name, x=x,
                             terminated=terminated, **kw))
        got_f = _outputs(run(bundles["derived", "free"], name, x=x,
                             terminated=terminated, **kw))
        # the derived kernel's own Select already zeroes a terminated sample's
        # contribution under the mask-free kind; the guard must agree.
        assert len(got_g) == len(got_f)
        for g, f in zip(got_g, got_f):
            np.testing.assert_allclose(g, f, rtol=1e-12, atol=0)


def test_compiled_host_skips_terminated_samples_and_keeps_live_results(bundles):
    _check(_host, bundles)


@pytest.mark.gpu
def test_compiled_device_skips_terminated_samples_and_keeps_live_results(bundles):
    _check(_device, bundles)
