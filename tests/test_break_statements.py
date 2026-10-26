# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``if cond: S; break`` inside a capped ``for``: assignments that run on the
exiting trip, then the exit.

It is front-end sugar for a snapshot of ``cond``, ``if snapshot: S`` and the
bare ``if snapshot: break``, so every row compares it with that form written
by hand (``m.where`` on the condition, then the bare break). Rows:

* **Primal** (B1-B3): the sugar and the hand-written kernel agree bit for bit
  on host, lanes leaving at different trips keep their own recorded values, a
  vector assigned in ``S`` is whole.
* **Derivatives** (B4): jvp and vjp match central differences and the
  hand-written kernel's derivatives.
* **Surface** (B5): what ``S`` may not hold, and a second exit, are refused by name.
* **jump=** (B6): refused inside the existing refusals, the hand-written form too.
* **Targets** (B7, B8): the C unit compiles strict c99, passes MISRA and
  computes the host's bits; the CUDA source compiles; the device values are a
  gpu-marked row.
* **Render baseline** (B9) is ``tests/test_render_baseline.py``, unchanged.
"""

from __future__ import annotations

import linecache
import math

import _embedded as E
import numpy as np
import pytest
from _eval import evaluate
from _toolchain import compile_check
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.embedded import EmbeddedProfile, check, export, refuse_audit_hits, render
from hawk.emit import FLOAT64, render_body, render_source
from hawk.ir import HawkError
from hawk.targets import TARGETS

CAP = 12
LIM = 3.0
N = 64
H = 1e-6


# --------------------------------------------------------------------------- #
# subjects: each sugar kernel beside its hand-written twin
# --------------------------------------------------------------------------- #
@hawk.kernel
def ignite(x: Scalar, g: Scalar, lim: Scalar, tau: Mutable[Scalar],
           y: Mutable[Scalar]):
    s = x
    t = 0.0
    r = -1.0
    for _k in range(CAP):
        s = m.sin(g * s) + s * 1.2
        t = t + 0.1
        if s > lim:
            r = t + 0.25 * s
            break
    tau = r
    y = s


@hawk.kernel
def ignite_hand(x: Scalar, g: Scalar, lim: Scalar, tau: Mutable[Scalar],
                y: Mutable[Scalar]):
    s = x
    t = 0.0
    r = -1.0
    for _k in range(CAP):
        s = m.sin(g * s) + s * 1.2
        t = t + 0.1
        c = s > lim
        r = m.where(c, t + 0.25 * s, r)
        if c:
            break
    tau = r
    y = s


def py_ignite(x, g, lim):
    s, t, r = x, 0.0, -1.0
    for _k in range(CAP):
        s = math.sin(g * s) + s * 1.2
        t = t + 0.1
        if s > lim:
            r = t + 0.25 * s
            break
    return r, s


@hawk.kernel
def trips(x: Scalar, g: Scalar, tol: Scalar, n: Mutable[Scalar],
          e_out: Mutable[Scalar]):
    """A recorded trip count: each lane exits at its own trip."""
    e = x
    count = -1.0
    for k in range(CAP):
        e = e * g
        if e < tol:
            count = 1.0 * k
            break
    n = count
    e_out = e


@hawk.kernel
def trips_hand(x: Scalar, g: Scalar, tol: Scalar, n: Mutable[Scalar],
               e_out: Mutable[Scalar]):
    e = x
    count = -1.0
    for k in range(CAP):
        e = e * g
        c = e < tol
        count = m.where(c, 1.0 * k, count)
        if c:
            break
    n = count
    e_out = e


def py_trips(x, g, tol):
    e, count = x, -1.0
    for k in range(CAP):
        e = e * g
        if e < tol:
            count = float(k)
            break
    return count, e


@hawk.kernel
def vec_exit(w0: Vector[3], lim: Scalar, out: Mutable[Vector[3]],
             best: Mutable[Vector[3]]):
    """A vector assigned, and rescaled in place, on the exiting trip."""
    w = w0
    keep = w0
    for _k in range(CAP):
        w = w * 1.5 + 0.1
        if w[0] > lim:
            keep = w + 1.0
            w = w * 0.5
            break
    out = w
    best = keep


@hawk.kernel
def vec_exit_hand(w0: Vector[3], lim: Scalar, out: Mutable[Vector[3]],
                  best: Mutable[Vector[3]]):
    w = w0
    keep = w0
    for _k in range(CAP):
        w = w * 1.5 + 0.1
        c = w[0] > lim
        keep = m.where(c, w + 1.0, keep)
        w = m.where(c, w * 0.5, w)
        if c:
            break
    out = w
    best = keep


def py_vec_exit(w0, lim):
    w, keep = np.array(w0, float), np.array(w0, float)
    for _k in range(CAP):
        w = w * 1.5 + 0.1
        if w[0] > lim:
            keep = w + 1.0
            w = w * 0.5
            break
    return w, keep


@hawk.kernel
def accumulate(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    """An augmented assignment in S."""
    s = x
    acc = 0.0
    for _k in range(CAP):
        s = m.sin(g * s) + s * 1.2
        if s > lim:
            acc += s * 2.0
            break
    y = acc + s


def py_accumulate(x, g, lim):
    s, acc = x, 0.0
    for _k in range(CAP):
        s = math.sin(g * s) + s * 1.2
        if s > lim:
            acc += s * 2.0
            break
    return acc + s


#: (x, g): leaves on the first trip, leaves mid-range, never leaves.
POINTS = ((5.0, 0.7), (0.31, 0.7), (0.05, -0.9))


def _xg():
    rng = np.random.default_rng(43)
    x = rng.uniform(0.02, 0.6, N)
    x[:4] = (5.0, 4.0, 0.03, 0.02)
    g = rng.uniform(-0.9, 0.9, N)
    return x, g


def _trips_in():
    rng = np.random.default_rng(5)
    x = rng.uniform(0.5, 4.0, N)
    g = rng.uniform(0.2, 0.95, N)
    x[:3] = (1e-9, 3.0, 4.0)
    g[:3] = (0.5, 0.45, 0.99)       # leaves at once, late, never
    return x, g, np.full(N, 1e-3)


# --------------------------------------------------------------------------- #
# compiled bundle (host, cuda) and the host unit the C comparison reads
# --------------------------------------------------------------------------- #
KERNELS = [ignite, ignite_hand, trips, trips_hand, vec_exit, vec_exit_hand,
           accumulate]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("break_statements")
    return build_bundle(KERNELS, root, targets=("cuda", "host"), host_profile="native",
                        cache_dir=cache_dir)


def _host(bundle, name, **kw):
    rt.run(rt.load(bundle.directory, name, sidecar_of(bundle, name)), **kw)


def _run_ignite(bundle, name):
    x, g = _xg()
    tau, y = np.zeros(N), np.zeros(N)
    _host(bundle, name, x=x, g=g, lim=np.full(N, LIM), tau=tau, y=y)
    return tau, y


def _run_trips(bundle, name):
    x, g, tol = _trips_in()
    n, e = np.zeros(N), np.zeros(N)
    _host(bundle, name, x=x, g=g, tol=tol, n=n, e_out=e)
    return n, e


def _run_vec(bundle, name):
    rng = np.random.default_rng(11)
    w0 = rng.uniform(-0.5, 1.0, (3, N))
    out, best = np.zeros((3, N)), np.zeros((3, N))
    _host(bundle, name, w0=w0, lim=np.full(N, LIM), out=out, best=best)
    return w0, out, best


# B1 ------------------------------------------------------------------------ #
def test_the_sugar_is_bit_identical_to_the_hand_written_kernel(bundle):
    for got, want in zip(_run_ignite(bundle, "ignite"), _run_ignite(bundle, "ignite_hand")):
        assert got.tobytes() == want.tobytes()
    tau, y = _run_ignite(bundle, "ignite")
    x, g = _xg()
    ref = np.array([py_ignite(a, b, LIM) for a, b in zip(x, g)])
    np.testing.assert_allclose(tau, ref[:, 0], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(y, ref[:, 1], rtol=1e-12, atol=1e-13)
    assert (tau < 0).any() and (tau > 0).any(), "the draw must hold both outcomes"


def test_the_exit_trip_applies_s_to_that_trips_values():
    """The interpreter, against plain python: the condition is read before S."""
    for point in POINTS:
        got = evaluate(ignite.sinks, {"x": point[0], "g": point[1], "lim": LIM})
        want = py_ignite(*point, LIM)
        assert float(got["tau"]) == pytest.approx(want[0], rel=1e-14, abs=1e-14)
        assert float(got["y"]) == pytest.approx(want[1], rel=1e-14, abs=1e-14)


# B2 ------------------------------------------------------------------------ #
def test_lanes_leave_at_their_own_trip_and_record_it(bundle):
    n, e = _run_trips(bundle, "trips")
    x, g, tol = _trips_in()
    ref = np.array([py_trips(a, b, t) for a, b, t in zip(x, g, tol)])
    np.testing.assert_array_equal(n, ref[:, 0])
    np.testing.assert_allclose(e, ref[:, 1], rtol=1e-13, atol=0)
    assert len(set(n)) > 3 and n[0] == 0 and n[1] > 5 and n[2] == -1.0
    for got, want in zip((n, e), _run_trips(bundle, "trips_hand")):
        assert got.tobytes() == want.tobytes()


# B3 ------------------------------------------------------------------------ #
def test_a_vector_assigned_in_s_is_whole(bundle):
    w0, out, best = _run_vec(bundle, "vec_exit")
    for j in range(N):
        w, keep = py_vec_exit(w0[:, j], LIM)
        np.testing.assert_allclose(out[:, j], w, rtol=1e-13)
        np.testing.assert_allclose(best[:, j], keep, rtol=1e-13)
    _, out_h, best_h = _run_vec(bundle, "vec_exit_hand")
    assert out.tobytes() == out_h.tobytes() and best.tobytes() == best_h.tobytes()


def test_an_augmented_assignment_in_s(bundle):
    x, g = _xg()
    y = np.zeros(N)
    _host(bundle, "accumulate", x=x, g=g, lim=np.full(N, LIM), y=y)
    np.testing.assert_allclose(y, [py_accumulate(a, b, LIM) for a, b in zip(x, g)],
                               rtol=1e-12, atol=1e-13)


# B4 ------------------------------------------------------------------------ #
def _central(ref, point, held, out):
    up, dn = dict(zip("xg", point)), dict(zip("xg", point))
    up[held] += H
    dn[held] -= H
    return (ref(up["x"], up["g"], LIM)[out] - ref(dn["x"], dn["g"], LIM)[out]) / (2 * H)


@pytest.mark.parametrize("point", POINTS[1:])
@pytest.mark.parametrize("held", ["x", "g"])
def test_jvp_matches_central_differences_and_the_hand_written_one(point, held):
    env = {"x": point[0], "g": point[1], "lim": LIM,
           "dot_x": float(held == "x"), "dot_g": float(held == "g")}
    got = evaluate(jvp(ignite, wrt=("x", "g")), env)
    twin = evaluate(jvp(ignite_hand, wrt=("x", "g")), env)
    for out, name in ((0, "dot_tau"), (1, "dot_y")):
        assert float(got[name]) == pytest.approx(
            _central(py_ignite, point, held, out), rel=1e-6, abs=1e-8)
        assert float(got[name]) == pytest.approx(float(twin[name]), rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("point", POINTS[1:])
def test_vjp_matches_central_differences_and_the_hand_written_one(point):
    env = {"x": point[0], "g": point[1], "lim": LIM, "bar_tau": 1.0, "bar_y": 0.5}
    got = evaluate(vjp(ignite, wrt=("x", "g")), env)
    twin = evaluate(vjp(ignite_hand, wrt=("x", "g")), env)
    for held in ("x", "g"):
        want = (_central(py_ignite, point, held, 0)
                + 0.5 * _central(py_ignite, point, held, 1))
        assert float(got[f"bar_{held}"]) == pytest.approx(want, rel=1e-6, abs=1e-8)
        assert float(got[f"bar_{held}"]) == pytest.approx(
            float(twin[f"bar_{held}"]), rel=1e-13, abs=1e-14)


# B5 ------------------------------------------------------------------------ #
def _traced(source: str):
    filename = "<break-statements>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict = {"kernel": hawk.kernel, "Scalar": Scalar, "Mutable": Mutable,
                   "m": m, "hawk": hawk}
    exec(compile(source, filename, "exec"), scope)
    return scope["k"]


_HEAD = "    s = x\n    r = 0.0\n    for j in range(4):\n        s = s * 2.0\n"
REFUSALS = [
    ("a nested if", _HEAD + "        if s > 1.0:\n            if s > 2.0:\n"
     "                r = s\n            break\n", "takes only assignments"),
    ("a nested for", _HEAD + "        if s > 1.0:\n            for i in range(2):\n"
     "                r = r + s\n            break\n", "takes only assignments"),
    ("a sink call", "    s = x\n    r = 0.0\n    for j in range(4):\n"
     "        s = s * 2.0\n        if s > 1.0:\n            out.contribute(s)\n"
     "            break\n", "takes only assignments"),
    ("a reduction", _HEAD + "        if s > 1.0:\n            r = hawk.sum(s)\n"
     "            break\n", "reduces over the lanes"),
    ("a statement before the break that is no assignment",
     _HEAD + "        if s > 1.0:\n            pass\n            break\n",
     "takes only assignments"),
    ("two exits", _HEAD + "        if s > 1.0:\n            r = s\n            break\n"
     "        if s > 2.0:\n            r = s + 1.0\n            break\n",
     "ONE `if cond: break`"),
    ("an exit and a bare exit", _HEAD + "        if s > 1.0:\n            r = s\n"
     "            break\n        if s > 2.0:\n            break\n",
     "ONE `if cond: break`"),
    ("an exit inside an if", _HEAD + "        if s > 1.0:\n            if s > 3.0:\n"
     "                r = s\n                break\n", "TOP-LEVEL"),
    ("an else", _HEAD + "        if s > 1.0:\n            r = s\n            break\n"
     "        else:\n            r = 2.0\n", "TOP-LEVEL"),
]


@pytest.mark.parametrize("case", REFUSALS, ids=[c[0] for c in REFUSALS])
def test_what_s_may_not_hold_is_refused_by_name(case):
    _label, body, needle = case
    source = ("@kernel\ndef k(x: Scalar, out: Mutable[Scalar]):\n" + body
              + "    out = s + r\n")
    with pytest.raises(HawkError) as excinfo:
        _traced(source)
    assert needle in str(excinfo.value), str(excinfo.value)


def test_a_kernel_with_only_bare_exits_is_unchanged_by_the_lowering():
    @hawk.kernel
    def bare(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
        s = x
        for _k in range(CAP):
            s = m.sin(g * s) + s * 1.2
            if s > lim:
                break
        y = s

    assert "brk" not in render_body(bare.sinks, bare.walk).text


# B6 ------------------------------------------------------------------------ #
def _jump_kernel(sugar: bool):
    body = ("        if s > lim:\n            r = s * 0.5\n            break\n" if sugar
            else "        c = s > lim\n        r = m.where(c, s * 0.5, r)\n"
                 "        if c:\n            break\n")
    return _traced(
        "@kernel(jump=0.01)\ndef k(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):\n"
        "    s = x\n    r = 0.0\n    for j in range(8):\n"
        "        s = m.sin(g * s) + s * 1.2\n" + body + "    y = s + r\n")


@pytest.mark.parametrize("sugar", [True, False], ids=["sugar", "hand"])
@pytest.mark.parametrize("transform", [jvp, vjp], ids=["jvp", "vjp"])
def test_jump_refuses_the_form_as_it_refuses_the_hand_written_one(sugar, transform):
    k = _jump_kernel(sugar)
    with pytest.raises(HawkError, match="event, not a jump|`break` in the middle"):
        transform(k, wrt=("x",))


def test_jump_gives_the_form_and_the_hand_written_kernel_the_same_refusal():
    messages = []
    for sugar in (True, False):
        with pytest.raises(HawkError) as excinfo:
            jvp(_jump_kernel(sugar), wrt=("x",))
        messages.append(str(excinfo.value).split(":", 1)[-1])
    assert messages[0] == messages[1]


# B7 ------------------------------------------------------------------------ #
STRICT = EmbeddedProfile(require_cppcheck=True)
C_CASES = [("ignite", ignite, _run_ignite, ("tau", "y")),
           ("trips", trips, _run_trips, ("n", "e_out"))]


@pytest.mark.parametrize("case", C_CASES, ids=[c[0] for c in C_CASES])
def test_the_c_unit_is_strict_c99_misra_clean_and_audited(case):
    _name, kernel, _run, _outs = case
    source = render(kernel, mode="float64")
    assert source.text.count("break;") == 1
    refuse_audit_hits(source)           # audit.break accepts the lowered shape
    checks = check(source, profile=STRICT)
    assert [run["std"] for run in checks["cc"]] == ["c99", "c11"]
    assert all(run["rc"] == 0 for run in checks["cc"])
    assert checks["cppcheck"]["findings"] == []


@pytest.mark.parametrize("case", C_CASES, ids=[c[0] for c in C_CASES])
def test_the_c_unit_computes_the_hosts_bits(bundle, tmp_path, case):
    name, kernel, run_host, outs = case
    art = export(kernel, tmp_path, mode="float64")
    so = E.cc_shared(art.c_path, tmp_path)
    if name == "ignite":
        x, g = _xg()
        inputs = {"x": x, "g": g, "lim": np.full(N, LIM)}
    else:
        x, g, tol = _trips_in()
        inputs = {"x": x, "g": g, "tol": tol}
    c_side = {o: np.zeros(N) for o in outs}
    E.run_c(so, name, art.report, N, mode="float64", **inputs, **c_side)
    for got, want in zip((c_side[o] for o in outs), run_host(bundle, name)):
        assert got.tobytes() == want.tobytes()


# B8 ------------------------------------------------------------------------ #
@pytest.mark.parametrize("kernel", [ignite, trips, vec_exit], ids=lambda k: k.name)
def test_the_emitted_sources_compile(tmp_path, kernel):
    failures = []
    for target in ("cuda", "host"):
        src = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                            mode=FLOAT64, kind=kernel.kind)
        path = tmp_path / f"{kernel.name}_{target}.{'cu' if target == 'cuda' else 'cpp'}"
        path.write_text(src.text)
        rc, stderr, argv = compile_check(path, target)
        if rc != 0:
            failures.append(f"--- {kernel.name}/{target}\n$ {' '.join(argv)}\n{stderr[:2000]}")
        assert "break;" in src.text
    assert not failures, "\n".join(failures)


@pytest.mark.gpu
def test_the_device_values_match_the_host(bundle):
    from test_loop_early_exit import _device

    x, g = _xg()
    want = _run_ignite(bundle, "ignite")
    got = _device(bundle, "ignite", ["tau", "y"], x=x, g=g, lim=np.full(N, LIM))
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
