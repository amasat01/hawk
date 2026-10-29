# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The named net axis: ``Grid["net", "sample"]`` with per-net weights, per-net losses
(``hawk.sum(..., over="sample")`` keeps the net label) and per-net gradients (``vjp``).

Each row equals a numpy loop over the nets, on the host and (``gpu`` mark) the device. Nets of
different width are padded to ``H`` and a per-net mask row zeroes the unused units: their gradient
is exactly zero. A gradient is summed over its own net's samples only, so perturbing net i moves
no value or gradient of net j (bitwise). The device row runs K = 16384 nets and replays bitwise."""

from __future__ import annotations

import _deploy as L
import _net_axis_kernels as N
import numpy as np
import pytest
from conftest import _xp, sidecar_of

from hawk import Kernel
from hawk.artifact import build_bundle
from hawk.diff import vjp

H, P = N.H, N.P
S = 40
WIDTHS = (8, 3, 5, 1, 6)
TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    """``{kernel name: bundle}``: the loss under both policies at S = 40, 48, 64, and the
    compensated gradient at S = 40 and 64."""
    root = tmp_path_factory.mktemp("net_axis")
    out = {}
    for s, policy in ((40, "compensated"), (40, "ordered"), (48, "ordered"), (48, "compensated"),
                      (64, "compensated")):
        loss = N.net_loss_for(s, policy)
        home = build_bundle([loss], root / loss.name, targets=("cuda", "host"),
                            cache_dir=cache_dir, host_profile="native")
        out[loss.name] = home
        if s != 48 or policy == "ordered":
            grad = Kernel(f"net_grad_{s}_{policy}", vjp(loss, wrt=("W",), primal_unit=home.digest))
            out[grad.name] = build_bundle([grad], root / grad.name, targets=("cuda", "host"),
                                          cache_dir=cache_dir, host_profile="native")
    return out


def _problem(k, widths, seed=0, S=S):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(k, P)) * 0.7
    x, y = rng.normal(size=(k, S)), rng.normal(size=(k, S))
    wide = np.resize(np.asarray(widths), k)
    return w, x, y, (np.arange(H)[None, :] < wide[:, None]).astype(np.float64)


def _oracle(w, x, y, mask):
    S = x.shape[1]
    """Per-net loss and dloss/dW by a loop over the nets."""
    k = w.shape[0]
    loss, grad = np.zeros(k), np.zeros((k, P))
    for i in range(k):
        a, b, c = w[i, 0::3], w[i, 1::3], w[i, 2::3]
        t = np.tanh(a[:, None] * x[i] + b[:, None])
        r = (mask[i] * c) @ t - y[i]
        loss[i] = r @ r
        grad[i, 0::3] = (2 * mask[i] * c)[:, None] * (1 - t * t) * x[i] * r @ np.ones(S)
        grad[i, 1::3] = ((2 * mask[i] * c)[:, None] * (1 - t * t) * r).sum(axis=1)
        grad[i, 2::3] = (2 * mask[i])[:, None] * t * r @ np.ones(S)
    return loss, grad


def _plan(bundles, name, target, planes, k):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = bundles[name]
    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    return eplan.plan(plugin, structure=structure).bind(net=k, **planes)


def _planes(target, w, x, y, mask):
    xp = _xp(target)
    k = w.shape[0]
    return {"W": xp.asarray(w.ravel()), "mask": xp.asarray(mask.ravel()),
            "x": xp.asarray(x.ravel()), "y": xp.asarray(y.ravel()),
            "out": xp.zeros(k), "bar_out": xp.ones(k), "bar_W": xp.zeros(k * P)}


def _host(a):
    return np.array(a.get() if hasattr(a, "get") else a)


def _both(bundles, target, w, x, y, mask):
    """(loss, gradient) from the compiled kernels."""
    k, s = w.shape[0], x.shape[1]
    planes = _planes(target, w, x, y, mask)
    fwd = {n: planes[n] for n in ("W", "mask", "x", "y", "out")}
    _plan(bundles, f"net_loss_{s}_compensated", target, fwd, k).launch()
    back = {n: planes[n] for n in ("W", "mask", "x", "y", "bar_out", "bar_W")}
    _plan(bundles, f"net_grad_{s}_compensated", target, back, k).launch()
    return _host(planes["out"]), _host(planes["bar_W"]).reshape(k, P)


@pytest.mark.parametrize("target", TARGETS)
def test_values_and_gradients_equal_the_loop_over_nets(bundles, target):
    w, x, y, mask = _problem(5, WIDTHS)
    loss, grad = _both(bundles, target, w, x, y, mask)
    want_loss, want_grad = _oracle(w, x, y, mask)
    np.testing.assert_allclose(loss, want_loss, rtol=1e-12)
    np.testing.assert_allclose(grad, want_grad, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("target", TARGETS)
def test_masked_width_has_exactly_zero_gradient(bundles, target):
    w, x, y, mask = _problem(5, WIDTHS)
    _, grad = _both(bundles, target, w, x, y, mask)
    dead = np.repeat(mask == 0.0, 3, axis=1)
    assert dead.any() and (grad[dead] == 0.0).all() and (grad[~dead] != 0.0).all()


@pytest.mark.parametrize("target", TARGETS)
def test_perturbing_one_net_moves_no_other_net(bundles, target):
    w, x, y, mask = _problem(5, WIDTHS)
    base = _both(bundles, target, w, x, y, mask)
    w2, x2, y2 = w.copy(), x.copy(), y.copy()
    w2[2] += 0.5
    x2[2] *= 1.5
    y2[2] -= 1.0
    moved = _both(bundles, target, w2, x2, y2, mask)
    keep = [0, 1, 3, 4]
    assert np.array_equal(moved[0][keep], base[0][keep])
    assert np.array_equal(moved[1][keep], base[1][keep])
    assert not np.array_equal(moved[1][2], base[1][2])


def _grad(bundles, target, w, x, y, mask, policy, launches=1):
    """The gradient from ``launches`` launches. The ordered kernel has a fold stage, which the host
    runtime runs and the device driver of ``_team_stages`` launches."""
    from hawk import runtime as rt
    k, s = w.shape[0], x.shape[1]
    name = f"net_grad_{s}_{policy}"
    seen = []
    for _ in range(launches):
        planes = _planes(target, w, x, y, mask)
        back = {n: planes[n] for n in ("W", "mask", "x", "y", "bar_out", "bar_W")}
        if policy == "compensated":
            _plan(bundles, name, target, back, k).launch()
        elif target == "host":
            rt.run(rt.load(bundles[name].directory, name, sidecar_of(bundles[name], name)), **back)
        else:
            from _team_stages import run_device
            run_device(bundles[name], name, back, k * s)
        seen.append(_host(planes["bar_W"]).reshape(k, P))
    return seen


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("s", (40, 48))
def test_ordered_gradient_equals_the_loop_and_is_bitwise(bundles, target, s):
    k = 5 if target == "host" else 16384
    w, x, y, mask = _problem(k, WIDTHS, seed=3, S=s)
    seen = _grad(bundles, target, w, x, y, mask, "ordered", 3)
    assert seen[0].tobytes() == seen[1].tobytes() == seen[2].tobytes()
    pick = np.random.default_rng(1).choice(k, min(k, 256), replace=False)
    _, want = _oracle(w[pick], x[pick], y[pick], mask[pick])
    np.testing.assert_allclose(seen[0][pick], want, rtol=1e-10, atol=1e-10)
    dead = np.repeat(mask[pick] == 0.0, 3, axis=1)
    assert (seen[0][pick][dead] == 0.0).all()
    comp = _grad(bundles, target, w, x, y, mask, "compensated")[0] if s == 40 else seen[0]
    np.testing.assert_allclose(seen[0], comp, rtol=1e-9, atol=1e-9 * np.abs(comp).max())


@pytest.mark.gpu
@pytest.mark.parametrize("s", (40, 64))
def test_sixteen_thousand_nets_compensated_match_the_loop_and_replay_when_aligned(bundles, s):
    """The default (compensated) gradient: equal to the loop; bitwise only when a net fills at most
    two warps (S = 64), since its commits are atomic."""
    w, x, y, mask = _problem(16384, WIDTHS, seed=3, S=s)
    seen = _grad(bundles, "cuda", w, x, y, mask, "compensated", 3)
    pick = np.random.default_rng(1).choice(16384, 256, replace=False)
    _, want = _oracle(w[pick], x[pick], y[pick], mask[pick])
    np.testing.assert_allclose(seen[0][pick], want, rtol=1e-10, atol=1e-10)
    if s == 64:
        assert seen[0].tobytes() == seen[1].tobytes() == seen[2].tobytes()


def _loss(bundles, target, w, x, y, mask, policy):
    """The loss from 3 launches; the ordered one has team stages, which the host runtime runs and
    the device driver of ``_team_stages`` launches."""
    from hawk import runtime as rt
    k, s = w.shape[0], x.shape[1]
    name = f"net_loss_{s}_{policy}"
    seen = []
    for _ in range(3):
        if policy == "compensated":
            planes = _planes(target, w, x, y, mask)
            _plan(bundles, name, target, {n: planes[n] for n in ("W", "mask", "x", "y", "out")}, k).launch()
            seen.append(_host(planes["out"]))
        elif target == "host":
            out = np.zeros(k)
            rt.run(rt.load(bundles[name].directory, name, sidecar_of(bundles[name], name)),
                   W=w.ravel(), mask=mask.ravel(), x=x, y=y, out=out)
            seen.append(out)
        else:
            from _team_stages import run_device
            planes = _planes(target, w, x, y, mask)
            run_device(bundles[name], name, {n: planes[n] for n in ("W", "mask", "x", "y", "out")}, k * s)
            seen.append(_host(planes["out"]))
    return seen


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("s", (40, 48))
def test_ordered_loss_equals_the_loop_and_the_compensated_one(bundles, target, s):
    """Few nets on the host, 16384 on the device: the ordered sum is bitwise across launches
    (no atomics), equals the loop, and agrees with the compensated one to the sum bound."""
    k = 5 if target == "host" else 16384
    w, x, y, mask = _problem(k, WIDTHS, seed=3, S=s)
    ordered = _loss(bundles, target, w, x, y, mask, "ordered")
    assert ordered[0].tobytes() == ordered[1].tobytes() == ordered[2].tobytes()
    want, _ = _oracle(w, x, y, mask)
    np.testing.assert_allclose(ordered[0], want, rtol=1e-12)
    comp = _loss(bundles, target, w, x, y, mask, "compensated")[0]
    np.testing.assert_allclose(ordered[0], comp, rtol=8 * s * np.finfo(np.float64).eps)


def test_ordered_gradient_agrees_with_central_differences_at_five_nets(bundles):
    w, x, y, mask = _problem(5, WIDTHS, seed=7)
    grad = _grad(bundles, "host", w, x, y, mask, "ordered")[0]
    rng = np.random.default_rng(2)
    for net, col in zip(rng.integers(0, 5, 8), rng.integers(0, P, 8)):
        up, dn = w.copy(), w.copy()
        up[net, col] += 1e-6
        dn[net, col] -= 1e-6
        slope = (_loss(bundles, "host", up, x, y, mask, "ordered")[0][net]
                 - _loss(bundles, "host", dn, x, y, mask, "ordered")[0][net]) / 2e-6
        assert abs(grad[net, col] - slope) <= 1e-6 * max(1.0, abs(slope))


def test_stage_kernels_of_a_derivative_inherit_the_primal_its_parent_names(bundles, tmp_path, cache_dir):
    """No ``primal_unit=`` on the vjp: ``derivative=`` on the parent covers every stage."""
    loss = N.net_loss_for(40, "ordered")
    grad = Kernel("net_grad_inherit", vjp(loss, wrt=("W",)))
    home = bundles[loss.name]
    built = build_bundle([grad], tmp_path / "g", targets=("host",), cache_dir=cache_dir,
                         host_profile="native", derivative={grad.name: (loss.name, home.digest)})
    stages = sidecar_of(built, grad.name)["stages"]
    assert len(stages) == 2
    assert {sidecar_of(built, st["kernel"])["derivative"]["primal"] for st in stages} == {loss.name}
