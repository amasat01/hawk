# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A reversed ``for`` that reads a wide plane at a runtime index and holds a
``dispatch``: the loop rule for the switch, and the per-iteration commit of the
runtime-indexed gradient (``bar[i].at(k) += g`` inside the loop). Three rows: the
interpreter twin against the unrolled layer and against finite differences, the
emitted text, and the compiled host kernel."""

from __future__ import annotations

import _spline_forms as S
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Kernel
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ir import HawkError, canonical

WRT = ("x", "theta")


def _env(n, form, rng):
    theta = rng.normal(0.0, 0.3, size=n * S.W)
    theta[S.W - 2::S.W], theta[S.W - 1::S.W] = 0.0, 1.0
    env = {"x": rng.uniform(-0.9, 0.9, size=n * S.ROWS), "theta": theta,
           "terminated": False, "bar_y": np.array([1.0, 0.0])}
    if form == "switch":
        env["kinds"] = rng.integers(0, 3, size=n).astype(float)
    return env


def _layers(n, form):
    rolled = S.kan_rolled(form, n)
    unrolled = S.kan_layer("local", n) if form == "plain" else S.kan_switch(n)
    return rolled, unrolled


@pytest.mark.parametrize("n", [4, 8, 32])
@pytest.mark.parametrize("form", ["plain", "switch"])
def test_the_rolled_vjp_equals_the_unrolled_one_on_the_twin(n, form):
    """Weights and coefficients (one contribution each) bit for bit; the lanes of
    ``u`` (``x0``, ``xs``, ``bar_x``: sums of several terms) to rounding of the largest term (the order of a longer sum)."""
    rolled, unrolled = _layers(n, form)
    env = _env(n, form, np.random.default_rng(10 + n))
    a = evaluate(vjp(rolled, wrt=WRT), env)
    b = evaluate(vjp(unrolled, wrt=WRT), env)
    lanes = np.arange(n * S.W) % S.W
    exact = lanes < S.W - 2
    assert np.array_equal(np.asarray(a["bar_theta"])[exact], np.asarray(b["bar_theta"])[exact])
    for got, ref in ((np.asarray(a["bar_theta"])[~exact], np.asarray(b["bar_theta"])[~exact]),
                     (a["bar_x"], b["bar_x"])):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())


def _central(f, v, h=1e-6):
    out = np.zeros_like(v)
    for i in range(v.size):
        d = np.zeros_like(v)
        d[i] = h
        out[i] = (f(v + d) - f(v - d)) / (2 * h)
    return out


def test_the_rolled_switch_vjp_and_jvp_match_finite_differences():
    n = 4
    rolled, _ = _layers(n, "switch")
    env = _env(n, "switch", np.random.default_rng(7))
    env["kinds"] = np.array([0.0, 1.0, 2.0, 0.0])
    got = evaluate(vjp(rolled, wrt=WRT), env)

    def value(x=None, theta=None):
        e = {**env, "x": env["x"] if x is None else x,
             "theta": env["theta"] if theta is None else theta}
        return float(evaluate(rolled.sinks, e)["y"][0])

    want_theta = _central(lambda t: value(theta=t), env["theta"])
    want_x = _central(lambda v: value(x=v), env["x"])
    for g, w in ((got["bar_theta"], want_theta), (got["bar_x"], want_x)):
        assert np.max(np.abs(np.asarray(g) - w) / np.maximum(1.0, np.abs(w))) <= 1e-9
    # the forward rule: a directional derivative against the same difference
    rng = np.random.default_rng(1)
    dth, dx = rng.normal(size=n * S.W), rng.normal(size=n * S.ROWS)
    tan = evaluate(jvp(rolled, wrt=WRT), {**env, "dot_x": dx, "dot_theta": dth})
    dot = float(np.asarray(tan[next(k for k in tan if k.startswith("dot_y"))])[0])
    h = 1e-6
    fd = (value(env["x"] + h * dx, env["theta"] + h * dth)
          - value(env["x"] - h * dx, env["theta"] - h * dth)) / (2 * h)
    assert abs(dot - fd) / max(1.0, abs(fd)) <= 1e-9


@hawk.kernel
def _recurrence(x: hawk.Vector[4], kinds: hawk.Vector[4], g: hawk.Scalar,
                y: hawk.Mutable[hawk.Scalar]):
    """A recurrence whose dispatch branches read the carry (a tape row inside a
    branch)."""
    acc = 0.1
    for e in range(4):
        acc = m.dispatch(m.int32(kinds[e]),
                         [m.tanh(g * acc + x[e]), acc * 0.5 + x[e] * x[e]],
                         policy="switch")
    y = acc


def test_a_recurrence_whose_dispatch_reads_the_carry_matches_finite_differences():
    rng = np.random.default_rng(2)
    env = {"x": rng.normal(size=4), "kinds": np.array([0.0, 1.0, 0.0, 1.0]), "g": 0.7}
    got = evaluate(vjp(_recurrence, wrt=("x", "g")), {**env, "bar_y": 1.0})

    def value(x, g):
        return float(evaluate(_recurrence.sinks, {**env, "x": x, "g": g})["y"])

    h = 1e-6
    want_x = _central(lambda v: value(v, env["g"]), env["x"])
    want_g = (value(env["x"], 0.7 + h) - value(env["x"], 0.7 - h)) / (2 * h)
    assert np.max(np.abs(np.asarray(got["bar_x"]) - want_x)) <= 1e-9
    assert abs(float(got["bar_g"]) - want_g) <= 1e-9


# --------------------------------------------------------------------------
# the emitted text


@pytest.mark.parametrize("form", ["plain", "switch"])
def test_the_rolled_vjp_commits_inside_the_loop_and_holds_no_row(form):
    sinks = vjp(S.kan_rolled(form, 8), wrt=WRT)
    lines = render_body(tuple(sinks), canonical(tuple(sinks))).text.splitlines()
    text = "\n".join(lines)
    # no carried or copied row (the dead zero base the loop boundary names is
    # dropped by the compiler: a 0.6.5 item)
    assert "hawk_sv" not in text and "hawk_it" not in text
    for plane in ("mut_bar_theta", "mut_bar_x"):         # one prologue per plane
        assert sum("for (Int hk" in ln and plane in ln for ln in lines) == 1, plane
    loop = next(n for n, ln in enumerate(lines) if "for (Int hawk_k" in ln)
    links = [n for n, ln in enumerate(lines) if "mut_bar_theta[i].at(" in ln and "+=" in ln]
    depth = len(lines[loop]) - len(lines[loop].lstrip())
    assert links and all(n > loop and len(lines[n]) - len(lines[n].lstrip()) > depth
                         for n in links), links
    assert not any(ln.strip().startswith("mut_bar_theta[i] =") for ln in lines)


# --------------------------------------------------------------------------
# the compiled host kernel


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("rolled_reverse")

    def get(name, primal, sinks):
        d = root / name
        d.mkdir()
        bundle = build_bundle([primal, Kernel(name, sinks)], d, targets=("host",),
                              cache_dir=cache_dir, host_profile="native", mode="float64")
        return rt.load(bundle.directory, name, sidecar_of(bundle, name))

    return get


def test_the_rolled_switch_vjp_host_kernel_equals_the_twin(host):
    """64 samples, one in nine terminated, planes pre-filled with 7.5: every live
    lane agrees with the interpreter to host contraction, a terminated sample's
    planes are untouched, and a slot no edge reads is zeroed by the prologue."""
    n, count = 8, 64
    rolled = S.kan_rolled("switch", n)
    sinks = vjp(rolled, wrt=WRT)
    art = host("rolled_switch_vjp", rolled, sinks)
    rng = np.random.default_rng(5)
    theta = rng.normal(0.0, 0.3, size=(n * S.W, count))
    theta[S.W - 2::S.W], theta[S.W - 1::S.W] = 0.0, 1.0
    x = rng.uniform(-0.95, 0.95, size=(n * S.ROWS, count))
    kinds = rng.integers(0, 3, size=(n, count)).astype(float)
    term = np.zeros(count, dtype=bool)
    term[::9] = True
    bar_y = np.zeros((2, count))
    bar_y[0] = rng.normal(size=count)
    planes = {"bar_theta": np.full((n * S.W, count), 7.5),
              "bar_x": np.full((n * S.ROWS, count), 7.5)}
    rt.run(art, x=x, theta=theta, kinds=kinds, terminated=term, bar_y=bar_y, **planes,
           layout="samples_last")
    for s in range(count):
        if term[s]:
            assert np.all(planes["bar_theta"][:, s] == 7.5)
            assert np.all(planes["bar_x"][:, s] == 7.5)
            continue
        want = evaluate(sinks, {"x": x[:, s], "theta": theta[:, s], "kinds": kinds[:, s],
                                "terminated": False, "bar_y": bar_y[:, s]})
        for key in ("bar_theta", "bar_x"):
            ref = np.asarray(want[key])
            np.testing.assert_allclose(planes[key][:, s], ref, rtol=1e-12,
                                       atol=1e-13 * np.abs(ref).max())


@hawk.kernel
def _folded(x: hawk.Vector[4], theta: hawk.Vector[6], y: hawk.Mutable[hawk.Scalar]):
    """Two iterations of the loop read ONE slot (``index(0.5 e, 6)``: 0, 0, 1, 1):
    the per-iteration commit must ADD."""
    total = 0.0
    for e in range(4):
        total = total + theta[m.index(0.5 * m.real(e), 6)] * x[e]
    y = total


def test_two_iterations_hitting_one_slot_accumulate(host):
    sinks = vjp(_folded, wrt=("theta",))
    art = host("folded_vjp", _folded, sinks)
    rng = np.random.default_rng(4)
    count = 5
    x, theta = rng.normal(size=(4, count)), rng.normal(size=(6, count))
    seed = rng.normal(size=count)
    bar = np.full((6, count), 7.5)
    rt.run(art, x=x, theta=theta, bar_y=seed, bar_theta=bar, layout="samples_last")
    for s in range(count):
        want = np.zeros(6)
        want[0], want[1] = seed[s] * (x[0, s] + x[1, s]), seed[s] * (x[2, s] + x[3, s])
        assert np.allclose(bar[:, s], want, rtol=1e-14, atol=1e-15), s
        twin = evaluate(sinks, {"x": x[:, s], "theta": theta[:, s], "bar_y": seed[s]})
        assert np.allclose(np.asarray(twin["bar_theta"]), want, rtol=1e-14, atol=1e-15), s


def test_second_order_through_a_rolled_vjp_refuses_by_name():
    with pytest.raises(HawkError, match="once per iteration"):
        jvp(vjp(S.kan_rolled("plain", 4), wrt=WRT), wrt=WRT)


# --------------------------------------------------------------------------
# the refusals: each names what, why, and what to write instead


@hawk.kernel
def _gather_in_a_case(x: hawk.Vector[4], kinds: hawk.Vector[4],
                      theta: hawk.Table[hawk.Scalar], y: hawk.Mutable[hawk.Scalar]):
    t = 0.0
    for e in range(4):
        t = m.dispatch(m.int32(kinds[e]),
                       [theta[m.index(t + 0.5 * m.real(e), 6)] * x[e], t * 0.5 + x[e] * x[e]],
                       policy="switch")
    y = t


def test_a_gather_inside_a_case_inside_a_loop_refuses_by_name():
    with pytest.raises(HawkError) as e:
        vjp(_gather_in_a_case, wrt=("theta",))
    text = str(e.value)
    assert "at() inside a `dispatch` branch inside a loop body" in text      # what
    assert "committed under the branch's case" in text                        # why
    assert "Read the plane outside the dispatch" in text                      # instead


@hawk.kernel(jump=0.04)
def _jump_with_a_case(x: hawk.Vector[4], kinds: hawk.Vector[4], s: hawk.Scalar,
                      y: hawk.Mutable[hawk.Scalar]):
    t = 0.0
    for e in range(4):
        t = m.dispatch(m.int32(kinds[e]), [m.tanh(t + x[e]), x[e] * x[e]], policy="switch")
        if t < s:
            t = t * 2.0
    y = t


def test_a_dispatch_in_a_reversed_loop_of_a_jump_kernel_refuses_by_name():
    with pytest.raises(HawkError) as e:
        vjp(_jump_with_a_case, wrt=("x", "s"))
    text = str(e.value)
    assert "`dispatch` inside a loop body of a `jump=` kernel" in text       # what
    assert "switching comparisons of the plan are not applied" in text        # why
    assert "not built" in text


@hawk.kernel
def _promoted_and_gathered(x: hawk.Vector[4], theta: hawk.Vector[6], g: hawk.Param,
                           y: hawk.Mutable[hawk.Scalar]):
    t = 0.0
    for e in range(4):
        t = t + g * theta[m.index(0.5 * m.real(e), 6)] * x[e]
    y = t


def test_a_promoted_parameter_beside_a_per_iteration_gradient_refuses_by_name():
    from hawk.diff import param_planes

    with pytest.raises(HawkError) as e:
        vjp(param_planes(_promoted_and_gathered, ["g"]), wrt=("theta", "g"))
    text = str(e.value)
    assert "reads a parameter cell" in text                                   # what
    assert "per-32-sample sum of the parameter is not built" in text          # why
    assert "plain `Param` (not through param_planes)" in text                 # instead
