# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The autodiff exactness matrix: every derivative is exact, or refused by name.

Rows are the features a kernel can lean on (broadcast input, run-time first axis,
a switch, a per-lane stop, ``if c: S; break``, keyed sums, an element fill, a
stepped kernel, ``jump=``, per-lane linear algebra, a ``Grid`` with ``a=``);
columns are the modes:

* **jvp** against a central difference (f64, ``h`` and tolerance per row, inputs
  kept off switch boundaries and exit thresholds so the difference is valid);
* **vjp** against a central difference of ``<u, f(x)>``, element by element;
* **duality** ``<u, jvp(v)> == <vjp(u), v>`` to 1e-12 relative;
* **f32 smoke**: the f32 build runs, is finite and is within ``F32_RTOL`` of the
  f64 answer (rows with a compiled f32 path).

A cell that does not pass must be named in ``REFUSED`` (the library refuses it by
design, with that message) or in ``BUGS`` (``xfail(strict=True)``: a silent wrong
answer, with the mechanism). ``jump=`` rows have a piecewise-constant primal, so a
central difference is meaningless: they are judged by the mean tangent against the
exact derivative of the counting game (four standard errors) and by duality.

Three doors run a row: a built host bundle (``runtime.run``), the kernel called
directly (the documented door for a broadcast input), and the interpreter
(``tests/_eval.py``) for per-lane linear algebra and the stepped kernel.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest
from _autodiff_kernels import (
    JLANES,
    LANE,
    NC,
    S_JUMP,
    TRUTH,
    A,
    bcast,
    exit_s_break,
    fill_curve,
    grid_kw,
    jchain,
    jgate,
    keyed_sum,
    keyed_sum_c,
    la_chol,
    la_chol_from,
    la_chol_from_rows,
    la_det,
    la_eigh,
    la_inv,
    la_lstsq,
    la_mat_newton,
    la_mat_rows,
    la_mvn,
    la_solve,
    node_gather,
    node_scatter,
    pend,
    readback,
    stop_break,
    switch_dispatch,
    switch_if,
    vbcast,
)
from _eval import evaluate
from conftest import sidecar_of

from hawk import HawkError, Kernel
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, step_jvp, step_vjp, vjp

N = 12
F32_RTOL = 2e-3     # f32 smoke: relative to the largest |value| of the f64 answer
DUALITY_RTOL = 1e-12
STEPS_K = 6


# --------------------------------------------------------------------------- #
# the rows: a Case names the kernel, the point, what is differentiated, the shapes
# --------------------------------------------------------------------------- #
class Case:
    def __init__(self, primal, planes, wrt, shapes, door="bundle", ext=None, h=1e-6,
                 rtol=1e-6, f32=None, extra=None):
        self.P, self.planes, self.wrt, self.shapes = primal, planes, tuple(wrt), shapes
        self.door, self.ext, self.h, self.rtol = door, ext or {}, h, rtol
        self.f32 = door == "bundle" if f32 is None else f32
        self.extra = extra or {}      # extra: planes only the derivatives take
        self.outs = list(shapes)
        self.name = primal.name

    def plane(self, name):
        return self.planes[name]


def _u(seed, *shape, lo=0.5, hi=1.5):
    return np.random.default_rng(seed).uniform(lo, hi, shape)


def _scalar(v):
    return float(v) if np.ndim(v) == 0 else np.asarray(v)


def _lane_case(primal, wrt, shapes, **planes):
    return Case(primal, planes, wrt, shapes)


def _bcast(name, primal, x, wrt):
    return Case(primal, {wrt: _scalar(x), "w": _u(3, *LANE, lo=-1, hi=1)}, (wrt,),
                {"out": LANE}, door="direct", ext={"a": A}, f32=False)


def _spd(seed, n=3):
    g = np.random.default_rng(seed).standard_normal((n, n))
    return g @ g.T + n * np.eye(n)


def _mat(seed, n=3):
    return np.random.default_rng(seed).standard_normal((n, n)) + 3.0 * np.eye(n)


def _factor(seed, n=3):
    """A lower triangular factor with a positive diagonal and garbage above it."""
    g = np.random.default_rng(seed).standard_normal((n, n))
    return np.tril(g, -1) + np.diag(1.0 + np.abs(np.diag(g))) + 0.5 * np.triu(g, 1)


def _vec(seed, n=3):
    return np.random.default_rng(seed).standard_normal(n)


_SW = np.array([0.1, 0.2, 0.45, 0.55, 0.8, 0.9] * 2)         # clear of 0.3, 0.7, 1/3, 2/3
_G = _u(7, N)
_TR = np.random.default_rng(5)
_STOP_X, _STOP_G = _TR.uniform(0.5, 4.0, N), _TR.uniform(0.2, 0.95, N)
_STOP_X[:3], _STOP_G[:3] = (1e-9, 3.0, 4.0), (0.5, 0.45, 0.99)     # leaves at once, late, never
_BR = np.random.default_rng(43)
_BR_X, _BR_G = _BR.uniform(0.02, 0.6, N), _BR.uniform(-0.9, 0.9, N)
_BR_X[:4] = (5.0, 4.0, 0.03, 0.02)
_NODE_E = 6

CASES: dict = {
    "bcast_a1c": _bcast("a1c", bcast, _u(1, A, 1, NC), "x"),
    "bcast_c": _bcast("c", bcast, _u(1, NC), "x"),
    "bcast_scalar": _bcast("s", bcast, 0.7, "x"),
    "bcast_unit": _bcast("u", bcast, _u(1, 1, 1, 1), "x"),
    "bcast_vec_bc": _bcast("vbc", vbcast, _u(1, 3, 1, NC), "v"),
    "bcast_vec_c": _bcast("vc", vbcast, _u(1, 3, NC), "v"),
    "bcast_vec_a1c": _bcast("va", vbcast, _u(1, 3, A, 1, NC), "v"),
    "grid_kw": Case(grid_kw, {"p": _u(2, A, 1, NC), "x": _u(4, *LANE), "k": 0.8}, ("p", "x", "k"),
                    {"y": LANE}, door="direct", ext={"a": A}, f32=False),
    "node_gather": Case(node_gather, {"w": _u(2, _NODE_E, 3), "x": _u(3, _NODE_E)}, ("x", "w"),
                        {"out": (_NODE_E,)}, ext={"e": _NODE_E}),
    "node_scatter": Case(node_scatter, {"x": _u(3, _NODE_E)}, ("x",), {"b": (4,)},
                         ext={"e": _NODE_E, "node": 4}),
    "switch_if": _lane_case(switch_if, ("x", "g"), {"y": (N,)}, x=_SW, g=_G),
    "switch_dispatch": _lane_case(switch_dispatch, ("x", "g"), {"y": (N,)}, x=_SW, g=_G),
    "stop_break": _lane_case(stop_break, ("x", "g"), {"n": (N,), "e_out": (N,)}, x=_STOP_X,
                             g=_STOP_G, tol=np.full(N, 1e-3),
                             terminated=np.arange(N) % 5 == 4),
    "exit_s_break": _lane_case(exit_s_break, ("x", "g"), {"tau": (N,), "y": (N,)}, x=_BR_X,
                               g=_BR_G, lim=np.full(N, 3.0)),
    "keyed_sum": Case(keyed_sum, {"x": _u(5, 96)}, ("x",), {"tb": (2, 3)}),
    "keyed_sum_comp": Case(keyed_sum_c, {"x": _u(5, 96)}, ("x",), {"tb": (2, 3)}),
    "keyed_readback": Case(readback, {"x": _u(6, 48)}, ("x",), {"y": (48,)}, ext={"row": 3}),
    "fill_curve": _lane_case(fill_curve, ("a", "b"), {"curve": (4, N)}, a=_u(8, N, lo=0.3, hi=1),
                             b=_u(9, N)),
    "steps_pend": Case(pend, {"dt": 0.05, "c": 0.3, "x": 0.7, "v": -0.4}, ("x", "v", "c", "dt"),
                       {"x": (), "v": ()}, door="steps", f32=False),
    "la_solve": Case(la_solve, {"A": _mat(1), "b": _vec(2)}, ("A", "b"), {"x": (3,)}, door="interp"),
    "la_cholesky_cho_solve": Case(la_chol, {"S": _spd(3), "b": _vec(4)}, ("S", "b"),
                                  {"x": (3,), "L": (3, 3)}, door="interp"),
    "la_chol_from": Case(la_chol_from, {"L": _factor(12), "b": _vec(13)}, ("L", "b"),
                         {"x": (3,), "f": (3, 3)}, door="interp"),
    "la_chol_from_rows": Case(la_chol_from_rows,
                              {"u": _mat(21)[0], "v": _mat(21)[1], "w": _mat(21)[2], "b": _vec(22)},
                              ("u", "v", "w", "b"), {"x": (3,)}, door="interp"),
    "la_det": Case(la_det, {"A": _mat(5)}, ("A",), {"d": ()}, door="interp"),
    "la_inv": Case(la_inv, {"A": _mat(6)}, ("A",), {"Ainv": (3, 3)}, door="interp"),
    "la_eigh_w": Case(la_eigh, {"S": _spd(7)}, ("S",), {"w": (3,)}, door="interp"),
    "la_mat_newton": Case(la_mat_newton, {"x0": np.array([1.2, 0.8]), "b": np.array([2.0, 1.1])},
                          ("x0", "b"), {"x": (2,)}, door="interp"),
    "la_mat_rows": Case(la_mat_rows, {"u": _mat(21)[0], "v": _mat(21)[1], "w": _mat(21)[2], "b": _vec(22)},
                        ("u", "v", "w", "b"), {"x": (3,)}, door="interp"),
    "la_lstsq": Case(la_lstsq, {"J": np.random.default_rng(8).standard_normal((4, 3)),
                                "r": _vec(9, 4)}, ("J", "r"), {"x": (3,)}, door="interp"),
    "la_mvn_pathwise": Case(la_mvn, {"seed": 7, "counter": 2, "mu": _vec(10), "S": _spd(11)},
                            ("mu", "S"), {"y": (3,)}, door="interp"),
}


# --------------------------------------------------------------------------- #
# doors: each gives prim(planes), jvp(planes, tangents), vjp(planes, cotangents) -> dicts
# --------------------------------------------------------------------------- #
def _named(result, names):
    return result if isinstance(result, dict) else {names[0]: result}


def _derived(case, make, **kw):
    return Kernel(f"{case.name}_{make.__name__}", make(case.P, wrt=case.wrt, **kw))


class Rig:
    """Builds each (row, mode) once; a bundle door loads its three artifacts."""

    def __init__(self, root, cache_dir):
        self.root, self.cache_dir, self.memo = pathlib.Path(root), cache_dir, {}

    def runners(self, key, mode="float64"):
        if (key, mode) not in self.memo:
            case = {**CASES, **JCASES}[key]
            make = {"bundle": self._bundle, "direct": self._direct, "interp": self._interp,
                    "steps": self._steps}[case.door]
            self.memo[key, mode] = make(case, mode, key)
        return self.memo[key, mode]

    def _direct(self, case, mode, key):
        jk, vk = _derived(case, jvp), _derived(case, vjp)

        def prim(p):
            return _named(case.P(**p, **case.ext), case.outs)

        def j(p, t):
            return _named(jk(**p, **{f"dot_{k}": v for k, v in t.items()}, **case.ext),
                          [f"dot_{o}" for o in case.outs])

        def v(p, u):
            return _named(vk(**p, **{f"bar_{k}": x for k, x in u.items()}, **case.ext),
                          [f"bar_{w}" for w in case.wrt])
        return prim, j, v

    def _interp(self, case, mode, key):
        def prim(p):
            return evaluate(case.P.sinks, p)

        def j(p, t):
            return evaluate(jvp(case.P, wrt=case.wrt), {**p, **{f"dot_{k}": x for k, x in t.items()}})

        def v(p, u):
            return evaluate(vjp(case.P, wrt=case.wrt), {**p, **{f"bar_{k}": x for k, x in u.items()}})
        return prim, j, v

    def _steps(self, case, mode, key):
        """K launches of ``pend``: the primal, the forward tangent carried launch to
        launch, and the reverse sweep over the stored launches."""
        fwd, rev = step_jvp(pend), step_vjp(pend)
        fwd, rev = getattr(fwd, "sinks", fwd), getattr(rev, "sinks", rev)

        def env(p, x, v):
            return {"dt": p["dt"], "c": p["c"], "t_final": 1e9, "terminated": False, "x": x,
                    "v": v, "t": 0.0}

        def states(p):
            x, v, path = p["x"], p["v"], []
            for _ in range(STEPS_K):
                path.append((x, v))
                out = evaluate(pend.sinks, env(p, x, v))
                x, v = float(out["x"]), float(out["v"])
            return path, (x, v)

        def prim(p):
            return dict(zip(("x", "v"), states(p)[1]))

        def j(p, t):
            dx, dv = t["x"], t["v"]
            for x, v in states(p)[0]:
                out = evaluate(fwd, {**env(p, x, v), "dot_dt": t["dt"], "dot_c": t["c"],
                                     "dot_x": dx, "dot_v": dv, "dot_t": 0.0})
                dx, dv = float(out["dot_x"]), float(out["dot_v"])
            return {"dot_x": dx, "dot_v": dv}

        def vj(p, u):
            bx, bv, bc, bdt = u["x"], u["v"], 0.0, 0.0
            for x, v in reversed(states(p)[0]):
                out = evaluate(rev, {**env(p, x, v), "bar_x_next": bx, "bar_v_next": bv,
                                     "bar_t_next": 0.0})
                bx, bv = float(out["bar_x"]), float(out["bar_v"])
                bc, bdt = bc + float(out["bar_c"]), bdt + float(out["bar_dt"])
            return {"bar_x": bx, "bar_v": bv, "bar_c": bc, "bar_dt": bdt}
        return prim, j, vj

    def _bundle(self, case, mode, key):
        dt = np.float32 if mode == "float32" else np.float64
        kw = dict(targets=("host",), cache_dir=self.cache_dir, host_profile="native", mode=mode)
        home = build_bundle([case.P], self.root / f"{key}_{mode}" / "p", **kw)
        arts = {"p": rt.load(home.directory, case.P.name, sidecar_of(home, case.P.name))}
        for tag, make in (("jvp", jvp), ("vjp", vjp)):
            d = _derived(case, make, primal_unit=home.digest)
            b = build_bundle([d], self.root / f"{key}_{mode}" / tag, **kw)
            arts[tag] = rt.load(b.directory, d.name, sidecar_of(b, d.name))

        def cast(a):
            return a.astype(dt) if isinstance(a, np.ndarray) and a.dtype.kind == "f" else a

        def run(art, planes, prefix, shapes):
            outs = {prefix + o: np.zeros(sh, dt) for o, sh in shapes.items()}
            own = set(art.sidecar.get("companions", {})) | set(art.sidecar.get("keyed", {}))
            for role, name in art.arg_spec:           # a compensation plane the caller supplies
                base = name.removeprefix(prefix).removesuffix("_c")
                if role == "accum_out" and name not in outs and name not in own and base in shapes:
                    outs[name] = np.zeros(shapes[base], dt)
            rt.run(art, **{k: cast(v) for k, v in planes.items()}, **outs, **case.ext)
            return {k: v.astype(np.float64) for k, v in outs.items()}

        in_shapes = {w: np.shape(case.planes[w]) for w in case.wrt}

        def prim(p):
            return run(arts["p"], p, "", case.shapes)

        def j(p, t):
            return run(arts["jvp"], {**p, **case.extra, **{f"dot_{k}": x for k, x in t.items()}},
                       "dot_", case.shapes)

        def v(p, u):
            return run(arts["vjp"], {**p, **case.extra, **{f"bar_{k}": x for k, x in u.items()}},
                       "bar_", in_shapes)
        return prim, j, v


@pytest.fixture(scope="module")
def rig(tmp_path_factory, cache_dir):
    return Rig(tmp_path_factory.mktemp("admx"), cache_dir)


# --------------------------------------------------------------------------- #
# the checks (door-blind)
# --------------------------------------------------------------------------- #
def _float(a):
    return np.asarray(a, dtype=np.float64)


def _draw(seed, shapes):
    rng = np.random.default_rng(seed)
    return {k: (float(rng.normal()) if s == () else rng.normal(size=s)) for k, s in shapes.items()}


def _shapes_in(case):
    return {w: np.shape(case.planes[w]) for w in case.wrt}


def _bumped(planes, w, index, delta):
    out = dict(planes)
    arr = np.array(planes[w], dtype=np.float64)
    arr[index] += delta
    out[w] = float(arr) if arr.ndim == 0 else arr
    return out


def _scale(a):
    return max(float(np.abs(_float(a)).max()), 1e-300)


def _flat(a, like):
    return _float(a).reshape(np.shape(like))


def _bar(got, w, case):
    """The cotangent of input ``w`` in its own shape; a ``Param``'s comes as one cell per lane."""
    bar = _float(got[f"bar_{w}"])
    return bar.sum() if np.ndim(case.planes[w]) == 0 else bar.reshape(np.shape(case.planes[w]))


def _sample(size, seed):
    return np.arange(size) if size <= 40 else np.sort(np.random.default_rng(seed).choice(size, 12, False))


def check_jvp(rig, key):
    case, (prim, j, _) = CASES[key], rig.runners(key)
    t = _draw(1, _shapes_in(case))
    got = j(case.planes, t)
    side = {s: {w: _scalar(_float(case.planes[w]) + s * case.h * t[w]) for w in case.wrt}
            for s in (1, -1)}
    hi, lo = (prim({**case.planes, **side[s]}) for s in (1, -1))
    for o in case.outs:
        fd = (_float(hi[o]) - _float(lo[o])).ravel() / (2 * case.h)
        np.testing.assert_allclose(_float(got[f"dot_{o}"]).ravel(), fd, rtol=case.rtol,
                                   atol=case.rtol * _scale(fd) + 1e-12, err_msg=f"jvp {key} {o}")


def check_vjp(rig, key):
    case, (prim, _, v) = CASES[key], rig.runners(key)
    u = _draw(2, case.shapes)
    got = v(case.planes, u)

    def loss(p):
        r = prim(p)
        return sum(float(np.sum(u[o] * _flat(r[o], u[o]))) for o in case.outs)

    for w in case.wrt:
        bar = np.atleast_1d(_bar(got, w, case)).reshape(-1)
        idx = _sample(bar.size, 3)
        fd = np.array([(loss(_bumped(case.planes, w, np.unravel_index(i, np.shape(case.planes[w])), case.h))
                        - loss(_bumped(case.planes, w, np.unravel_index(i, np.shape(case.planes[w])), -case.h)))
                       / (2 * case.h) for i in idx])
        np.testing.assert_allclose(bar[idx], fd, rtol=case.rtol, atol=case.rtol * _scale(fd) + 1e-12,
                                   err_msg=f"vjp {key} wrt {w}")


def check_duality(rig, key):
    case, (_, j, v) = CASES[key], rig.runners(key)
    t, u = _draw(4, _shapes_in(case)), _draw(5, case.shapes)
    dot, bar = j(case.planes, t), v(case.planes, u)
    lhs = sum(float(np.sum(u[o] * _flat(dot[f"dot_{o}"], u[o]))) for o in case.outs)
    rhs = sum(float(np.sum(_bar(bar, w, case) * t[w])) for w in case.wrt)
    assert lhs == pytest.approx(rhs, rel=DUALITY_RTOL, abs=1e-14 * (1 + abs(lhs) + abs(rhs))), (lhs, rhs)


def check_f32(rig, key):
    case = CASES[key]
    (_, j64, v64), (_, j32, v32) = rig.runners(key), rig.runners(key, "float32")
    t, u = _draw(6, _shapes_in(case)), _draw(7, case.shapes)
    for ref, low in ((j64(case.planes, t), j32(case.planes, t)), (v64(case.planes, u), v32(case.planes, u))):
        for name, want in ref.items():
            assert np.isfinite(low[name]).all(), name
            np.testing.assert_allclose(low[name], _float(want), rtol=0,
                                       atol=F32_RTOL * _scale(want), err_msg=f"f32 {key} {name}")


#: Cells the library refuses by design: ``(row, mode) -> the message``.
REFUSED: dict = {
    # a Vector given broadcast along a run-time axis is refused at bind, so every mode is
    (k := "bcast_vec_a1c", m_): "'v' is a Vector given broadcast but varying along the runtime axis 'a'"
    for m_ in ("jvp", "vjp", "duality")
}
#: Cells that are silently wrong: ``(row, mode) -> the mechanism``; strict xfail.
BUGS: dict = {}


def _cell(rig, key, mode, fn):
    if (key, mode) in REFUSED:
        with pytest.raises(HawkError, match=REFUSED[key, mode]):
            fn(rig, key)
        return
    fn(rig, key)


def _rows(mode, keys=CASES):
    return [pytest.param(k, marks=[pytest.mark.xfail(strict=True, reason=BUGS[k, mode])]
                         if (k, mode) in BUGS else []) for k in keys]


@pytest.mark.parametrize("key", _rows("jvp"))
def test_jvp_matches_central_differences(rig, key):
    _cell(rig, key, "jvp", check_jvp)


@pytest.mark.parametrize("key", _rows("vjp"))
def test_vjp_matches_central_differences(rig, key):
    _cell(rig, key, "vjp", check_vjp)


@pytest.mark.parametrize("key", _rows("duality"))
def test_duality_jvp_vjp(rig, key):
    _cell(rig, key, "duality", check_duality)


@pytest.mark.parametrize("key", _rows("f32", [k for k, c in CASES.items() if c.f32]))
def test_f32_smoke(rig, key):
    _cell(rig, key, "f32", check_f32)


# --------------------------------------------------------------------------- #
# jump=: the primal is piecewise constant; the mean tangent meets the exact game
# --------------------------------------------------------------------------- #
Z = 4.0
_U = np.ascontiguousarray(np.random.default_rng(1).uniform(0.0, 1.0, (6, JLANES)))
JCASES = {k: Case(p, {"u": _U, "s": np.full(JLANES, S_JUMP)}, ("s",),
                  {"tally": (JLANES,), "alive_out": (JLANES,)}, extra={"jump_seed": 3}, f32=False)
          for k, p in (("jgate", jgate), ("jchain", jchain))}


def _within(values, truth):
    se = values.std() / np.sqrt(values.size)
    assert abs(values.mean() - truth) < Z * se, (values.mean(), truth, se)


@pytest.mark.parametrize("key", _rows("jvp", JCASES))
def test_jump_mean_tangent_is_the_derivative_of_the_game(rig, key):
    case = JCASES[key]
    out = rig.runners(key)[1](case.planes, {"s": np.ones(JLANES)})
    _within(out["dot_tally"], TRUTH[key][0])
    _within(out["dot_alive_out"], TRUTH[key][1])


@pytest.mark.parametrize("key", _rows("vjp", JCASES))
def test_jump_mean_cotangent_is_the_derivative_of_the_game(rig, key):
    case, ones, zeros = JCASES[key], np.ones(JLANES), np.zeros(JLANES)
    v = rig.runners(key)[2]
    _within(v(case.planes, {"tally": ones, "alive_out": zeros})["bar_s"], TRUTH[key][0])
    _within(v(case.planes, {"tally": zeros, "alive_out": ones})["bar_s"], TRUTH[key][1])


@pytest.mark.parametrize("key", _rows("duality", JCASES))
def test_jump_reverse_is_the_transpose_of_the_forward(rig, key):
    case = JCASES[key]
    _, j, v = rig.runners(key)
    rng = np.random.default_rng(5)
    vt, va, ds = (rng.normal(size=JLANES) for _ in range(3))
    dot = j(case.planes, {"s": ds})
    bar = v(case.planes, {"tally": vt, "alive_out": va})["bar_s"]
    want = vt * dot["dot_tally"] + va * dot["dot_alive_out"]
    np.testing.assert_allclose(bar * ds, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


# --------------------------------------------------------------------------- #
# PRODUCT-BUG (row 2, vjp, the direct door): run in a child, it segfaults the process
# --------------------------------------------------------------------------- #
_DIRECT_TABLE_VJP = """
import sys
import numpy as np
sys.path.insert(0, {tests!r})
from _autodiff_kernels import node_gather
from hawk import Kernel
from hawk.diff import vjp
got = Kernel("ng_direct_vjp", vjp(node_gather, wrt=("x", "w")))(
    w=np.full((6, 3), 1.5), x=np.full(6, 0.7), bar_out=np.ones(6), e=6)
assert np.shape(got["bar_w"]) == (6, 3), np.shape(got["bar_w"])
for _ in range(20):                       # a heap overflow shows on a repeat
    Kernel("ng_direct_vjp", vjp(node_gather, wrt=("x", "w")))(
        w=np.full((6, 3), 1.5), x=np.full(6, 0.7), bar_out=np.ones(6), e=6)
"""


def test_direct_door_vjp_of_a_runtime_axis_table_returns_the_whole_gradient():
    done = subprocess.run([sys.executable, "-c", _DIRECT_TABLE_VJP.format(tests=str(pathlib.Path(__file__).parent))],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-300:]


def test_a_gradient_plane_smaller_than_its_lookup_plane_is_refused_before_launch(tmp_path, cache_dir):
    from hawk import runtime as rt
    from hawk.artifact import build_bundle

    kw = dict(targets=("host",), cache_dir=cache_dir, host_profile="native")
    home = build_bundle([node_gather], tmp_path / "p", **kw)
    derived = Kernel("ng_vjp", vjp(node_gather, wrt=("x", "w"), primal_unit=home.digest))
    bundle = build_bundle([derived], tmp_path / "d", **kw)
    run = rt.load(bundle.directory, "ng_vjp", next(a.sidecar for a in bundle.artifacts))
    assert run.sidecar["adjoint_like"] == {"bar_w": "w"}
    args = dict(w=np.full((6, 3), 1.5), x=np.full(6, 0.7), bar_out=np.ones(6), bar_x=np.zeros(6), e=6)
    with pytest.raises(HawkError, match="gradient of the lookup plane 'w'"):
        rt.run(run, bar_w=np.zeros(6), **args)
    rt.run(run, bar_w=np.zeros((6, 3)), **args)


def test_the_jvp_of_a_compensated_scatter_owns_its_companion(tmp_path, cache_dir):
    from hawk.artifact import build_bundle

    kw = dict(targets=("host",), cache_dir=cache_dir, host_profile="native")
    home = build_bundle([keyed_sum_c], tmp_path / "p", **kw)
    derived = Kernel("kc_jvp", jvp(keyed_sum_c, wrt=("x",), primal_unit=home.digest))
    assert derived.walk.companions == {"dot_tb_c": "dot_tb"}
    bundle = build_bundle([derived], tmp_path / "d", **kw)
    assert next(a.sidecar for a in bundle.artifacts)["companions"] == {"dot_tb_c": "dot_tb"}


def test_the_direct_door_runs_a_keyed_read_back_kernel_and_its_derivatives():
    x = np.random.default_rng(0).random(48)
    v, u = np.random.default_rng(1).random((3, 16)), np.random.default_rng(2).random((3, 16))
    xx = x.reshape(3, 16)
    s = xx.sum(1, keepdims=True)
    got = readback(x=x, row=3)
    np.testing.assert_allclose(got, xx / s, rtol=1e-14)
    dot = Kernel("rb_jvp", jvp(readback, wrt=("x",)))(x=x, dot_x=v.ravel(), row=3)
    np.testing.assert_allclose(dot, v / s - xx * v.sum(1, keepdims=True) / s ** 2,
                               rtol=1e-12, atol=1e-14)
    bar = Kernel("rb_vjp", vjp(readback, wrt=("x",)))(x=x, bar_y=u.ravel(), row=3)
    np.testing.assert_allclose(bar, u / s - (u * xx).sum(1, keepdims=True) / s ** 2,
                               rtol=1e-12, atol=1e-14)
