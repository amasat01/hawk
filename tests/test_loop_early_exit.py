# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Loops a sample can leave early: ``if cond: break`` inside a capped ``for``.

The form: ONE ``if <traced bool>: break`` as a top-level statement of a
``for`` over a compile-time ``range`` (the cap). It lowers to a real per-sample
``break`` on host and device; a sample keeps the values its carried names held
at the break (Python's semantics). Rows:

* **Surface.** Every refusal names its construct (``while``, ``continue``, a
  nested or doubled ``break``, a static or non-bool condition).
* **Semantics.** The scratch interpreter, and the COMPILED kernel on host and
  device, agree with plain Python for a break at the head, the middle and the
  tail of the body — including exit on the first iteration, a sample that
  never exits (runs to the cap), a sample masked on entry, an active-set map
  and a sample-major plane.
* **Emission.** The break is really emitted; a loop without one renders
  exactly as before (no counter, no test).
* **Derivatives.** jvp through every form and vjp through a head or tail break
  match central differences, compiled derivatives match the interpreter, the
  second-order routes match a difference of the gradient, and a middle-break
  vjp is refused rather than guessed.
"""

from __future__ import annotations

import linecache
import math

import _deploy as L
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Kernel, Mutable, Param, Scalar, Terminated, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ext import Guard, Kind
from hawk.ir import HawkError, Loop, LoopCount, canonical, canonical_nodes

CAP = 12
LIM = 3.0
N = 64
H = 1e-6


# --------------------------------------------------------------------------- #
# subjects and their plain-python references
# --------------------------------------------------------------------------- #
@hawk.kernel
def head(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    for _k in range(CAP):
        if s > lim:
            break
        s = m.sin(g * s) + s * 1.2
    y = s


def py_head(x, g, lim):
    s = x
    for _k in range(CAP):
        if s > lim:
            break
        s = math.sin(g * s) + s * 1.2
    return s


@hawk.kernel
def middle(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    t = x
    for _k in range(CAP):
        s = s * 1.1 + g * t
        if s > lim:
            break
        t = m.tanh(t) + 0.1
    y = s + t


def py_middle(x, g, lim):
    s = t = x
    for _k in range(CAP):
        s = s * 1.1 + g * t
        if s > lim:
            break
        t = math.tanh(t) + 0.1
    return s + t


@hawk.kernel
def tail(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    for _k in range(CAP):
        s = m.tanh(g * s) + s * 1.1 + 0.05
        if s > lim:
            break
    y = s


def py_tail(x, g, lim):
    s = x
    for _k in range(CAP):
        s = math.tanh(g * s) + s * 1.1 + 0.05
        if s > lim:
            break
    return s


@hawk.kernel
def gated_sum(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    """An ACCUMULATOR whose exit reads the index only: its reverse is a forward
    loop bounded by the primal's per-sample count."""
    acc = 0.0
    for k in range(CAP):
        if x * k > lim:
            break
        acc = acc + m.sin(g * x * k)
    y = acc


def py_gated_sum(x, g, lim):
    acc = 0.0
    for k in range(CAP):
        if x * k > lim:
            break
        acc += math.sin(g * x * k)
    return acc


@hawk.kernel
def trajectory(omega: Scalar, nstop: Scalar, dt: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], k: Mutable[Scalar]):
    """The CPU card's whole-trajectory shape: each sample stops at its own
    step, and a sample masked on entry keeps its prior state."""
    xs = x
    vs = v
    ks = k
    w2 = omega * omega
    for _s in range(CAP):
        if ks >= nstop:
            break
        a = -w2 * xs
        xs = xs + dt * vs
        vs = vs + dt * a
        ks = ks + 1.0
    x = xs
    v = vs
    k = ks


def py_trajectory(omega, nstop, dt, x, v, k):
    for _s in range(CAP):
        if k >= nstop:
            break
        a = -omega * omega * x
        x, v, k = x + dt * v, v + dt * a, k + 1.0
    return x, v, k


@hawk.kernel
def vec_head(w0: Vector[3], lim: Scalar, out: Mutable[Vector[3]]):
    """A rank-1 carry: the break keeps the whole vector of its iteration."""
    w = w0
    for _k in range(CAP):
        if w[0] > lim:
            break
        w = w * 1.5 + 0.1
    out = w


SUBJECTS = {"head": (head, py_head), "middle": (middle, py_middle),
            "tail": (tail, py_tail), "gated_sum": (gated_sum, py_gated_sum)}

#: (x, g) points: exit on the FIRST iteration, a mid-range exit, and a sample
#: that never reaches the limit (runs to the cap). None sits on a threshold
#: crossing, where the exit step (and so the derivative) jumps.
POINTS = ((5.0, 0.7), (0.31, 0.7), (0.05, -0.9))


def _loop(k) -> Loop:
    loops = [n for n in canonical_nodes(k.sinks)[0] if isinstance(n, Loop)]
    assert len(loops) == 1
    return loops[0]


# --------------------------------------------------------------------------- #
# surface
# --------------------------------------------------------------------------- #
def _traced(source: str):
    filename = "<early-exit>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict = {"kernel": hawk.kernel, "Scalar": Scalar, "Mutable": Mutable,
                   "m": m}
    exec(compile(source, filename, "exec"), scope)
    return scope["k"]


REFUSALS = [
    ("while", "    s = x\n    while s < 3.0:\n        s = s * 2.0\n", "`While`"),
    ("continue", "    s = x\n    for j in range(4):\n        if s > 1.0:\n"
     "            continue\n        s = s * 2.0\n", "`continue`"),
    ("break inside a nested if", "    s = x\n    for j in range(4):\n"
     "        if s > 1.0:\n            if s > 2.0:\n                break\n"
     "        s = s * 2.0\n", "TOP-LEVEL"),
    ("break with another statement", "    s = x\n    for j in range(4):\n"
     "        if s > 1.0:\n            s = s + 1.0\n            break\n"
     "        s = s * 2.0\n", "TOP-LEVEL"),
    ("break with an else", "    s = x\n    for j in range(4):\n"
     "        if s > 1.0:\n            break\n        else:\n"
     "            s = s * 2.0\n", "TOP-LEVEL"),
    ("bare break", "    s = x\n    for j in range(4):\n        s = s * 2.0\n"
     "        break\n", "TOP-LEVEL"),
    ("two breaks", "    s = x\n    for j in range(4):\n        if s > 1.0:\n"
     "            break\n        s = s * 2.0\n        if s > 2.0:\n"
     "            break\n", "ONE `if cond: break`"),
    ("break outside a loop", "    s = x\n    if s > 1.0:\n        s = s + 1.0\n"
     "    break\n", None),
    ("a static condition", "    s = x\n    for j in range(4):\n"
     "        if 1 > 2:\n            break\n        s = s * 2.0\n",
     "not a data-dependent exit"),
    ("a non-bool condition", "    s = x\n    for j in range(4):\n"
     "        if s:\n            break\n        s = s * 2.0\n", "rank-0 bool"),
]


@pytest.mark.parametrize("case", REFUSALS, ids=[c[0] for c in REFUSALS])
def test_every_early_exit_refusal_names_its_construct(case):
    _label, body, needle = case
    source = "@kernel\ndef k(x: Scalar, out: Mutable[Scalar]):\n" + body + \
        "    out = s\n"
    with pytest.raises((HawkError, SyntaxError)) as excinfo:
        _traced(source)
    if needle is not None:
        assert needle in str(excinfo.value), str(excinfo.value)


def test_the_break_position_is_read_from_the_values():
    assert _loop(head).exit_form == "head"
    assert _loop(middle).exit_form == "middle"
    assert _loop(tail).exit_form == "tail"


# --------------------------------------------------------------------------- #
# semantics: the interpreter against plain python
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(SUBJECTS))
@pytest.mark.parametrize("point", POINTS)
def test_the_interpreter_matches_python(name, point):
    k, ref = SUBJECTS[name]
    x, g = point
    got = float(evaluate(k.sinks, {"x": x, "g": g, "lim": LIM})["y"])
    assert got == pytest.approx(ref(x, g, LIM), rel=1e-14, abs=1e-14)


# --------------------------------------------------------------------------- #
# emission
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["head", "middle"])
def test_a_head_or_middle_break_is_tested_before_the_rest_of_the_body(name):
    text = render_body(SUBJECTS[name][0].sinks, SUBJECTS[name][0].walk).text
    body = text[text.index("for ("):]
    assert "break;" in body, text
    # the rest of the iteration follows the test: the update is after it
    assert body.index("break;") < body.index("hawk_n"), text


def test_a_tail_break_tests_the_condition_before_the_carries_move():
    text = render_body(tail.sinks, tail.walk).text
    lines = [ln.strip() for ln in text.splitlines()]
    x_at = next(i for i, ln in enumerate(lines) if ln.startswith("const bool hawk_x"))
    move_at = next(i for i, ln in enumerate(lines)
                   if ln.startswith("hawk_v") and "= hawk_n" in ln)
    brk_at = next(i for i, ln in enumerate(lines) if ln.startswith("if (hawk_x"))
    assert x_at < move_at < brk_at, text
    assert lines[brk_at].endswith("break;")


def test_a_loop_without_a_break_renders_as_before():
    @hawk.kernel
    def plain(x: Scalar, y: Mutable[Scalar]):
        s = x
        for _k in range(CAP):
            s = m.sin(s) + s * 1.2
        y = s

    loop = _loop(plain)
    assert loop.exit_cond is None and loop.count is None and loop.exit_form is None
    text = render_body(plain.sinks, plain.walk).text
    for token in ("break", "hawk_c", "hawk_x", "hawk_e"):
        assert token not in text, text
    # the header is the literal-bound header, untouched
    assert "for (Int hawk_k" in text and f"< {CAP}; ++" in text


def test_the_digest_tells_a_break_apart():
    @hawk.kernel
    def plain(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
        s = x
        for _k in range(CAP):
            s = m.sin(g * s) + s * 1.2
        y = s

    assert plain.walk.digest != head.walk.digest


def test_a_vjp_keeps_the_count_only_where_a_derived_loop_reads_it():
    for k in (head, tail, gated_sum):
        d = vjp(k, wrt=("x",))
        text = render_body(d, canonical(d)).text
        assert "Int hawk_c" in text and "++hawk_c" in text, text
        assert any(isinstance(n, LoopCount) for n in canonical_nodes(d)[0])
    text = render_body(head.sinks, head.walk).text
    assert "hawk_c" not in text, "a primal loop keeps no counter"


# --------------------------------------------------------------------------- #
# derivatives against central differences (interpreter)
# --------------------------------------------------------------------------- #
def _central(name, point, held):
    _k, ref = SUBJECTS[name]
    x, g = point
    args = {"x": x, "g": g}
    up = dict(args)
    up[held] += H
    down = dict(args)
    down[held] -= H
    return (ref(up["x"], up["g"], LIM) - ref(down["x"], down["g"], LIM)) / (2 * H)


@pytest.mark.parametrize("name", sorted(SUBJECTS))
@pytest.mark.parametrize("point", POINTS[1:])
@pytest.mark.parametrize("held", ["x", "g"])
def test_jvp_matches_a_central_difference(name, point, held):
    k, _ref = SUBJECTS[name]
    x, g = point
    d = jvp(k, wrt=("x", "g"))
    got = evaluate(d, {"x": x, "g": g, "lim": LIM, "dot_x": float(held == "x"),
                       "dot_g": float(held == "g")})
    assert float(got["dot_y"]) == pytest.approx(_central(name, point, held),
                                                rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("name", ["head", "tail", "gated_sum"])
@pytest.mark.parametrize("point", POINTS[1:])
def test_vjp_matches_a_central_difference(name, point):
    k, _ref = SUBJECTS[name]
    x, g = point
    got = evaluate(vjp(k, wrt=("x", "g")),
                   {"x": x, "g": g, "lim": LIM, "bar_y": 1.0})
    for held in ("x", "g"):
        assert float(got[f"bar_{held}"]) == pytest.approx(
            _central(name, point, held), rel=1e-6, abs=1e-8), held


def test_a_middle_break_vjp_is_refused_by_name():
    with pytest.raises(HawkError, match="MIDDLE"):
        vjp(middle, wrt=("x",))


def _gradient(k, env):
    got = evaluate(vjp(k, wrt=("x", "g")), {**env, "bar_y": 1.0})
    return np.array([float(got["bar_x"]), float(got["bar_g"])])


@pytest.mark.parametrize("name", ["head", "tail", "gated_sum"])
def test_second_order_matches_a_difference_of_the_gradient(name):
    k, _ref = SUBJECTS[name]
    env = {"x": 0.31, "g": 0.7, "lim": LIM}
    v = {"x": 0.5, "g": -1.25}
    hh = 1e-5
    up = _gradient(k, {**env, "x": env["x"] + hh * v["x"], "g": env["g"] + hh * v["g"]})
    dn = _gradient(k, {**env, "x": env["x"] - hh * v["x"], "g": env["g"] - hh * v["g"]})
    want = (up - dn) / (2 * hh)
    fo = evaluate(jvp(vjp(k, wrt=("x", "g")), wrt=("x", "g")),
                  {**env, "bar_y": 1.0, "dot_x": v["x"], "dot_g": v["g"]})
    np.testing.assert_allclose([float(fo["dot_bar_x"]), float(fo["dot_bar_g"])],
                               want, rtol=1e-6, atol=1e-7)
    ro = evaluate(vjp(jvp(k, wrt=("x", "g"))),
                  {**env, "bar_dot_y": 1.0, "dot_x": v["x"], "dot_g": v["g"]})
    np.testing.assert_allclose([float(ro["bar_x"]), float(ro["bar_g"])],
                               want, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------- #
# compiled: host and device
# --------------------------------------------------------------------------- #
MAP = Kind("early_exit_rows", guard=Guard(active_set=True))


def _derived(primal, transform, name):
    return Kernel(name, transform(primal, wrt=("x", "g")))


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("early_exit")
    kernels = [head, middle, tail, gated_sum, trajectory, vec_head,
               _derived(head, vjp, "head_vjp"), _derived(tail, vjp, "tail_vjp"),
               _derived(gated_sum, vjp, "gated_sum_vjp"),
               _derived(middle, jvp, "middle_jvp"), _derived(head, jvp, "head_jvp")]
    out = {"main": build_bundle(kernels, root / "main", targets=("cuda", "host"),
                                cache_dir=cache_dir)}
    out["map"] = build_bundle([trajectory], root / "map", targets=("host",),
                              cache_dir=cache_dir, kind=MAP)
    return out


def _host(bundle, name, **kw):
    art = rt.load(bundle.directory, name, sidecar_of(bundle, name))
    rt.run(art, **kw)


def _device(bundle, name, outs, **kw):
    """Run on the device; eagle returns the outputs in a dict keyed by plane
    name (several outputs) or as the bare plane (one), so ``outs`` is read
    back by name, never by a guessed slot order."""
    import eagle.exec as eexec
    from eagle import plan as eplan
    plugin = L.device_plugin(bundle.directory, name, sidecar_of(bundle, name))
    got = eplan.plan(plugin, structure=eexec.DeviceKernel).run(**kw)
    by_name = got if isinstance(got, dict) else {outs[0]: got}
    return [np.asarray(by_name[o].get() if hasattr(by_name[o], "get") else by_name[o])
            for o in outs]


def _xg():
    rng = np.random.default_rng(43)
    x = rng.uniform(0.02, 0.6, N)
    x[:4] = (5.0, 4.0, 0.03, 0.02)      # exit at once x2, never-exit candidates
    g = rng.uniform(-0.9, 0.9, N)
    return x, g


def _scalar_cases():
    x, g = _xg()
    lim = np.full(N, LIM)
    return x, g, lim


def _run_scalar(runner, bundle, name, outs, **inputs):
    if runner == "host":
        planes = {o: np.zeros(N) for o in outs}
        _host(bundle, name, **inputs, **planes)
        return [planes[o] for o in outs]
    return _device(bundle, name, outs, **inputs)


RUNNERS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_compiled_primal_matches_python(bundles, runner, name):
    _k, ref = SUBJECTS[name]
    x, g, lim = _scalar_cases()
    (y,) = _run_scalar(runner, bundles["main"], name, ["y"], x=x, g=g, lim=lim)
    want = np.array([ref(a, b, LIM) for a, b in zip(x, g)])
    np.testing.assert_allclose(y, want, rtol=1e-12, atol=1e-13)


DERIVED = [("head_vjp", head, {"bar_y"}, ("bar_x", "bar_g")),
           ("tail_vjp", tail, {"bar_y"}, ("bar_x", "bar_g")),
           ("gated_sum_vjp", gated_sum, {"bar_y"}, ("bar_x", "bar_g")),
           ("head_jvp", head, {"dot_x", "dot_g"}, ("dot_y",)),
           ("middle_jvp", middle, {"dot_x", "dot_g"}, ("dot_y",))]


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("case", DERIVED, ids=[c[0] for c in DERIVED])
def test_compiled_derivative_matches_the_interpreter(bundles, runner, case):
    name, primal, seeds, outs = case
    x, g, lim = _scalar_cases()
    seed = {s: np.linspace(0.5, 1.5, N) * (1 + i) for i, s in enumerate(sorted(seeds))}
    got = _run_scalar(runner, bundles["main"], name, list(outs), x=x, g=g, lim=lim,
                      **seed)
    transform = vjp if name.endswith("vjp") else jvp
    derived = transform(primal, wrt=("x", "g"))
    for j in range(N):
        env = {"x": x[j], "g": g[j], "lim": LIM, **{s: v[j] for s, v in seed.items()}}
        want = evaluate(derived, env)
        for o, arr in zip(outs, got):
            assert arr[j] == pytest.approx(float(want[o]), rel=1e-11, abs=1e-12), (
                name, o, j)


def _trajectory_inputs():
    rng = np.random.default_rng(7)
    omega = rng.uniform(0.5, 2.0, N)
    nstop = rng.integers(0, CAP + 4, N).astype(float)   # 0 = exit at once; > CAP = cap
    nstop[:3] = (0.0, CAP + 3.0, 5.0)
    terminated = np.zeros(N, dtype=bool)
    terminated[[3, 10, 17]] = True
    x0 = np.linspace(1.0, 2.0, N)
    v0 = np.zeros(N)
    return omega, nstop, terminated, x0, v0


def _trajectory_want(omega, nstop, terminated, x0, v0, dt):
    want = [np.array(c) for c in zip(*[
        py_trajectory(o, s, dt, a, b, 0.0) for o, s, a, b in zip(omega, nstop, x0, v0)])]
    for arr, start in zip(want, (x0, v0, np.zeros(N))):
        arr[terminated] = start[terminated]
    return want


@pytest.mark.parametrize("runner", RUNNERS)
def test_compiled_trajectory_stops_each_sample_at_its_own_step(bundles, runner):
    omega, nstop, terminated, x0, v0 = _trajectory_inputs()
    dt = 0.05
    if runner == "host":
        x, v, k = x0.copy(), v0.copy(), np.zeros(N)
        _host(bundles["main"], "trajectory", omega=omega, nstop=nstop, dt=dt,
              terminated=terminated, x=x, v=v, k=k)
        got = [x, v, k]
    else:
        got = _device(bundles["main"], "trajectory", ["x", "v", "k"], omega=omega,
                      nstop=nstop, dt=dt, terminated=terminated, x=x0.copy(),
                      v=v0.copy(), k=np.zeros(N))
    want = _trajectory_want(omega, nstop, terminated, x0, v0, dt)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)
    live = ~terminated
    np.testing.assert_array_equal(got[2][live], np.minimum(nstop, CAP)[live])


def test_compiled_trajectory_under_an_active_set_map(bundles):
    """The early exit inside a compacted launch: eagle's ActiveSet maps launch
    positions to the live samples, the break happens inside each one."""
    import eagle.exec as eexec
    from eagle import ActiveSet
    from eagle import plan as eplan

    omega, nstop, terminated, x0, v0 = _trajectory_inputs()
    dt = 0.05
    bundle = bundles["map"]
    plugin = L.host_plugin(bundle.directory, "trajectory",
                           sidecar_of(bundle, "trajectory"))
    x, v, k = x0.copy(), v0.copy(), np.zeros(N)
    aset = ActiveSet(terminated)
    aset.compact()
    assert aset.live == N - int(terminated.sum())
    eplan.plan(plugin, structure=eexec.HostTeam).bind(
        omega=omega, nstop=nstop, dt=dt, terminated=terminated, x=x, v=v, k=k,
        **aset.planes()).launch()
    want = _trajectory_want(omega, nstop, terminated, x0, v0, dt)
    for a, b in zip((x, v, k), want):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)


def test_compiled_rank_one_carry_through_a_sample_major_view(bundles):
    rng = np.random.default_rng(3)
    w0 = rng.uniform(0.0, 2.0, (N, 3))          # sample-major, C-contiguous
    w0[0, 0] = 9.0                              # exits at once
    native = np.ascontiguousarray(w0.T)         # (3, N)
    out = np.zeros((3, N))
    _host(bundles["main"], "vec_head", w0=native, lim=np.full(N, LIM), out=out)
    # the same storage handed over as an (N, 3) view: binds zero-copy
    out_view = np.zeros((3, N))
    _host(bundles["main"], "vec_head", w0=native.T, lim=np.full(N, LIM),
          out=out_view.T)
    want = np.empty((N, 3))
    for j in range(N):
        w = w0[j].copy()
        for _k in range(CAP):
            if w[0] > LIM:
                break
            w = w * 1.5 + 0.1
        want[j] = w
    np.testing.assert_allclose(out.T, want, rtol=1e-13)
    np.testing.assert_allclose(out_view.T, want, rtol=1e-13)
