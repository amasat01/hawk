# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A derived (``vjp``/``jvp``) kernel honours the PRIMAL's own ``terminated``
guard: a sample the primal's guard skips contributes EXACTLY zero to
every derived output — an own-column gradient/tangent, and a scattered
Accum/Param sum alike — and a LIVE sample's derivative is unchanged.

Two tiers: an IR-level check (:mod:`_eval`, the same scratch interpreter
``test_diff_finite_differences.py`` uses) covering the own-column AND the
scattered-accumulate shape cheaply, then a COMPILED host+device run of a
published bundle (``_deployable.diagnostic``'s vjp/jvp, the kind every other
terminated-mask row in ``test_ext_seams.py`` already exercises) — the body
text is shared between HAWK's two backends (``hawk/emit/host.py``'s own
docstring), so one compiled check pinned on both targets is what "host and
device emitters both" means in practice.
"""

from __future__ import annotations

from types import SimpleNamespace

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

from hawk import Index, Kernel, Mutable, Scalar, Table, Terminated, kernel
from hawk.diff import jvp, vjp
from hawk.ir import Leaf, Loop, canonical_nodes

H = 1e-5
N = 8


# --------------------------------------------------------------------------- #
# -- IR-level: own-column (D.diagnostic: y = x + 1.0) and a scattered accum.
# --------------------------------------------------------------------------- #
def test_vjp_zeroes_a_terminated_own_column_gradient_and_matches_fd_live():
    derived = vjp(D.diagnostic, wrt=("x",))
    x0, seed = 1.7, 3.0

    live = evaluate(derived, {"x": x0, "terminated": False, "bar_y": seed})
    plus = evaluate(D.diagnostic.sinks, {"x": x0 + H, "terminated": False})["y"]
    minus = evaluate(D.diagnostic.sinks, {"x": x0 - H, "terminated": False})["y"]
    expect = (plus - minus) / (2 * H) * seed
    assert live["bar_x"] == pytest.approx(expect, rel=1e-6, abs=1e-7)

    terminated = evaluate(derived, {"x": x0, "terminated": True, "bar_y": seed})
    assert terminated["bar_x"] == 0.0, (
        f"a terminated sample's gradient must be EXACTLY zero, got {terminated['bar_x']}")


def test_jvp_zeroes_a_terminated_own_column_tangent_and_matches_fd_live():
    derived = jvp(D.diagnostic, wrt=("x",))
    x0, tangent = 1.7, 0.6

    live = evaluate(derived, {"x": x0, "terminated": False, "dot_x": tangent})
    plus = evaluate(D.diagnostic.sinks, {"x": x0 + H * tangent, "terminated": False})["y"]
    minus = evaluate(D.diagnostic.sinks, {"x": x0 - H * tangent, "terminated": False})["y"]
    expect = (plus - minus) / (2 * H)
    assert live["dot_y"] == pytest.approx(expect, rel=1e-6, abs=1e-7)

    terminated = evaluate(derived, {"x": x0, "terminated": True, "dot_x": tangent})
    assert terminated["dot_y"] == 0.0, (
        f"a terminated sample's tangent must be EXACTLY zero, got {terminated['dot_y']}")


@kernel
def _gather_masked(table: Table[Scalar], where: Index, terminated: Terminated,
                   y: Mutable[Scalar]):
    y = table.at(where)


def test_vjp_scatter_adds_nothing_from_a_terminated_sample():
    """The gather/scatter transpose (module docstring, ``hawk/diff/transform.py``):
    ``table.at(where)``'s reverse is an ``AccumWrite`` scattered at ``where`` —
    exactly the "Param/Accum sums included" shape the gap names. A terminated
    sample must add exactly 0 at its own target lane; a live one adds its
    seeded adjoint there."""
    derived = vjp(_gather_masked, wrt=("table",))
    lanes = 4

    live = evaluate(derived, {"table": 0.0, "where": 2, "terminated": False,
                              "bar_y": 5.0}, lanes=lanes)
    assert live["bar_table"][2] == pytest.approx(5.0)

    gone = evaluate(derived, {"table": 0.0, "where": 2, "terminated": True,
                              "bar_y": 5.0}, lanes=lanes)
    assert gone["bar_table"][2] == 0.0, (
        f"a terminated sample must scatter-add nothing; got {gone['bar_table'][2]}")


# --------------------------------------------------------------------------- #
# -- compiled: the SAME body text on host AND device (shared emission).
# --------------------------------------------------------------------------- #
def _diagnostic_bundle(tmp_path, cache_dir, *, targets):
    vjp_kernel = Kernel("diagnostic_vjp", vjp(D.diagnostic, wrt=("x",)))
    jvp_kernel = Kernel("diagnostic_jvp", jvp(D.diagnostic, wrt=("x",)))
    from hawk.artifact import build_bundle

    return build_bundle([D.diagnostic, vjp_kernel, jvp_kernel], tmp_path,
                        targets=targets, cache_dir=cache_dir)


def _mask():
    x = np.arange(N, dtype=float) + 1.0
    terminated = np.zeros(N, dtype=bool)
    terminated[N // 2:] = True
    return x, terminated


def test_compiled_host_vjp_and_jvp_zero_the_terminated_half(tmp_path, cache_dir):
    bundle = _diagnostic_bundle(tmp_path, cache_dir, targets=("host",))
    x, terminated = _mask()

    plugin = L.host_plugin(bundle.directory, "diagnostic_vjp",
                           sidecar_of(bundle, "diagnostic_vjp"))
    import eagle.exec as eexec
    from eagle import plan as eplan

    bar_y = np.full(N, 2.0)
    bar_x = eplan.plan(plugin, structure=eexec.HostTeam).run(
        x=x, terminated=terminated, bar_y=bar_y)
    np.testing.assert_array_equal(bar_x[:N // 2], bar_y[:N // 2])
    np.testing.assert_array_equal(bar_x[N // 2:], np.zeros(N - N // 2))

    plugin = L.host_plugin(bundle.directory, "diagnostic_jvp",
                           sidecar_of(bundle, "diagnostic_jvp"))
    dot_x = np.full(N, 0.4)
    dot_y = eplan.plan(plugin, structure=eexec.HostTeam).run(
        x=x, terminated=terminated, dot_x=dot_x)
    np.testing.assert_array_equal(dot_y[:N // 2], dot_x[:N // 2])
    np.testing.assert_array_equal(dot_y[N // 2:], np.zeros(N - N // 2))


@pytest.mark.gpu
def test_compiled_device_vjp_and_jvp_zero_the_terminated_half(tmp_path, cache_dir):
    bundle = _diagnostic_bundle(tmp_path, cache_dir, targets=("cuda", "host"))
    x, terminated = _mask()

    import eagle.exec as eexec
    from eagle import plan as eplan

    def _get(a):
        return np.asarray(a.get() if hasattr(a, "get") else a)

    plugin = L.device_plugin(bundle.directory, "diagnostic_vjp",
                             sidecar_of(bundle, "diagnostic_vjp"))
    bar_y = np.full(N, 2.0)
    bar_x = _get(eplan.plan(plugin, structure=eexec.DeviceKernel).run(
        x=x, terminated=terminated, bar_y=bar_y))
    np.testing.assert_array_equal(bar_x[:N // 2], bar_y[:N // 2])
    np.testing.assert_array_equal(bar_x[N // 2:], np.zeros(N - N // 2))

    plugin = L.device_plugin(bundle.directory, "diagnostic_jvp",
                             sidecar_of(bundle, "diagnostic_jvp"))
    dot_x = np.full(N, 0.4)
    dot_y = _get(eplan.plan(plugin, structure=eexec.DeviceKernel).run(
        x=x, terminated=terminated, dot_x=dot_x))
    np.testing.assert_array_equal(dot_y[:N // 2], dot_x[:N // 2])
    np.testing.assert_array_equal(dot_y[N // 2:], np.zeros(N - N // 2))


# --------------------------------------------------------------------------- #
# -- a FINISHING primal (``terminated = cond``): termination is discrete,
# -- so the derived kernels drop the finish -- no counter, no mask write, no
# -- sidecar ``finish`` -- and keep zeroing the terminated samples.
# --------------------------------------------------------------------------- #
@kernel
def _finishing(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = x + 1.0
    terminated = x > 100.0


def _finishing_bundle(tmp_path, cache_dir, targets):
    from hawk.artifact import build_bundle

    return build_bundle(
        [_finishing,
         Kernel("finishing_vjp", vjp(_finishing, wrt=("x",))),
         Kernel("finishing_jvp", jvp(_finishing, wrt=("x",)))],
        tmp_path, targets=targets, cache_dir=cache_dir)


def _check_finishing(bundle, target):
    import eagle.exec as eexec
    from eagle import plan as eplan

    x, terminated = _mask()
    structure = eexec.HostTeam if target == "host" else eexec.DeviceKernel
    plugin_of = L.host_plugin if target == "host" else L.device_plugin
    for name, seed_name, out_seed in (("finishing_vjp", "bar_y", 2.0),
                                      ("finishing_jvp", "dot_x", 0.4)):
        sidecar = sidecar_of(bundle, name)
        assert "finish" not in sidecar and sidecar["terminated_readonly"] is True
        assert ["lookup", "finished_count"] not in sidecar["arg_spec"]
        mask = terminated.copy()
        seed = np.full(N, out_seed)
        got = eplan.plan(plugin_of(bundle.directory, name, sidecar),
                         structure=structure).run(
            x=x, terminated=mask, **{seed_name: seed})
        got = np.asarray(got.get() if hasattr(got, "get") else got)
        np.testing.assert_array_equal(got[:N // 2], seed[:N // 2])
        np.testing.assert_array_equal(got[N // 2:], np.zeros(N - N // 2))
        np.testing.assert_array_equal(mask, terminated)   # no mask write


def test_a_finishing_primals_derivatives_drop_the_finish_on_host(tmp_path, cache_dir):
    _check_finishing(_finishing_bundle(tmp_path, cache_dir, ("host",)), "host")


@pytest.mark.gpu
def test_a_finishing_primals_derivatives_drop_the_finish_on_device(tmp_path,
                                                                   cache_dir):
    _check_finishing(_finishing_bundle(tmp_path, cache_dir, ("cuda", "host")),
                     "cuda")


# --------------------------------------------------------------------------- #
# -- regression (post-#30): a guard-free primal must come out BYTE-FOR-
# -- STRUCTURE identical to before #30 -- no Select inserted at all -- and a
# -- kernel-LIKE object with no readable `.kind` (a downstream package's own
# -- kernel wrapper) must be read the SAME way, never crash.
# --------------------------------------------------------------------------- #
def _terminated_leaves(derived) -> list:
    seq, _ = canonical_nodes(derived)
    return [n for n in seq if isinstance(n, Leaf) and n.kind == "terminated"]


def test_a_guard_free_kernel_gets_no_termination_select():
    """D.axpb declares no Terminated plane at all: its vjp/jvp must read NO
    terminated-kind leaf anywhere -- the structural signature of "nothing was
    masked" that holds regardless of what the primal's OWN arithmetic does
    (an unrelated Select from some other rule would not be a terminated-kind
    leaf's own Select, so this is not fooled by one)."""
    assert not _terminated_leaves(vjp(D.axpb)), (
        "a guard-free primal's vjp must read no terminated plane")
    assert not _terminated_leaves(jvp(D.axpb)), (
        "a guard-free primal's jvp must read no terminated plane")


def test_a_kernel_like_object_with_no_kind_is_treated_as_guard_free():
    """The regression this message reports:
    ``AttributeError: 'HawkKernel' object has no attribute 'kind'`` --
    a downstream package's own kernel wrapper carries ``.walk``/``.sinks``
    but not ``.kind``. A primal this function cannot read a guard off of is
    read as having none, never an error, and the derived IR must therefore carry NO
    termination masking either -- same structural check as the row above,
    against a primal (D.diagnostic) that DOES declare Terminated, so this
    also proves the "no .kind" case is not a coincidental "it happens to
    have no terminated plane" pass."""
    bare = SimpleNamespace(sinks=D.diagnostic.sinks, walk=D.diagnostic.walk)
    derived = vjp(bare, wrt=("x",))          # must not raise
    assert not _terminated_leaves(derived), (
        "a primal with no readable .kind must be treated as guard-free")
    derived = jvp(bare, wrt=("x",))          # must not raise
    assert not _terminated_leaves(derived)


# --------------------------------------------------------------------------- #
# -- regression (post-#30): a reverse loop's OWN scatter commit (the
# -- gather/scatter transpose of a `.at()` read inside a lowered `for`) is a
# -- Loop, not a value-bearing Sink -- a downstream package's KAN edge
# -- kernels are built on exactly this shape (_kan.py's own docstring:
# -- copied from a downstream package's KAN edge cell). 609 AttributeErrors
# -- on 'Loop' object has no attribute 'value' is what an un-recursing
# -- masker does to it.
# --------------------------------------------------------------------------- #
@kernel
def _loop_gather_masked(x: Table[Scalar], theta: Table[Scalar],
                        terminated: Terminated, y: Mutable[Scalar]):
    total = 0.0
    for e in range(4):
        total = total + x.at(e) * theta.at(e)
    y = total


def test_vjp_of_a_scatter_inside_a_loop_does_not_crash_and_masks_it():
    derived = vjp(_loop_gather_masked, wrt=("theta",))
    loops = [s for s in derived if isinstance(s, Loop)]
    assert loops, (
        "this fixture's own reverse pass must produce a Loop root (the "
        "gather/scatter transpose) or the row is not exercising #30's "
        "regression at all")
    assert _terminated_leaves(derived), (
        "a Loop-rooted derivative of a terminated-guarded primal must still "
        "read the terminated plane -- inside the loop's own boundary")
    # canonical() (the same walk a build goes through) must succeed and bind
    # the terminated slot -- the ACTUAL crash site this message reports.
    from hawk.ir import canonical

    walk = canonical(derived)
    assert ("terminated", "terminated") in walk.slot_of

    zeros, one_hot = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]
    live = evaluate(derived, {"x": zeros, "theta": zeros, "terminated": False,
                              "bar_y": 5.0}, lanes=4)
    assert live["bar_theta"][2] == pytest.approx(0.0), (
        "d(total)/d(theta[2]) = x.at(edge=2) = 0.0 at this probe point")

    live2 = evaluate(derived, {"x": one_hot, "theta": zeros, "terminated": False,
                               "bar_y": 5.0}, lanes=4)
    assert live2["bar_theta"][2] == pytest.approx(5.0)

    gone = evaluate(derived, {"x": one_hot, "theta": zeros, "terminated": True,
                              "bar_y": 5.0}, lanes=4)
    assert gone["bar_theta"][2] == 0.0, (
        f"a terminated sample's loop-scattered gradient must be EXACTLY "
        f"zero; got {gone['bar_theta'][2]}")


# --------------------------------------------------------------------------- #
# -- Kernel's `planes` argument: optional, defaulting to `{}`.
# --------------------------------------------------------------------------- #
def test_kernel_without_a_planes_argument_matches_one_given_an_empty_dict():
    """A derived (vjp/jvp) kernel is normally wrapped with no declarations of
    its own: ``planes`` defaults to ``None`` and is translated to an empty
    dict internally, so the old, now-redundant explicit-``{}`` spelling and
    the defaulted one produce the identical walk."""
    sinks = vjp(D.diagnostic, wrt=("x",))
    no_planes = {}
    explicit = Kernel("diagnostic_vjp", sinks, no_planes)
    defaulted = Kernel("diagnostic_vjp", sinks)
    assert defaulted.planes == {}
    assert defaulted.walk.digest == explicit.walk.digest
    assert defaulted.arg_spec == explicit.arg_spec
