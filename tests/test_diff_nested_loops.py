# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A ``for`` nested in a ``for`` differentiates in BOTH directions.

Reverse recomputes the inner forward loop inside each reversed outer iteration, its
history in a local tape of the inner trip count; forward carries the inner tangents
in the inner loop. Rows: central finite differences, the forward/reverse duality
``<bar, J t> == <J^T bar, t>``, a vector carry, a triple nest, an inner gather of an
input that is not differentiated, an outer ``break``; and the shapes that stay
refused BY NAME: an inner ``break``, an inner gather of a differentiated plane in
reverse, ``jump=``."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ir import HawkError, canonical

S = hawk.Scalar
OUT = hawk.Mutable[hawk.Scalar]


@hawk.kernel
def sum_of_inner(x: S, g: S, out: OUT):
    total = 0.0
    for r in range(3):
        inner = 0.0
        for _c in range(4):
            inner = m.tanh(g * inner + x * (r + 1.0))
        total = total + inner
    out = total


@hawk.kernel
def carried_into_inner(x: S, g: S, out: OUT):
    acc = x
    for r in range(3):
        inner = acc
        for _c in range(4):
            inner = m.tanh(g * inner + 0.3 * (r + 1.0))
        acc = 0.5 * acc + inner
    out = acc


@hawk.kernel
def triple(x: S, g: S, out: OUT):
    acc = x
    for a in range(2):
        for b in range(2):
            mid = acc
            for _c in range(3):
                mid = m.sin(g * mid + 0.2 * (a + 1.0) + 0.1 * (b + 1.0))
            acc = acc * 0.7 + mid
    out = acc


@hawk.kernel
def vector_carry(v: hawk.Vector[3], g: S, out: hawk.Mutable[hawk.Vector[3]]):
    acc = v
    for r in range(3):
        w = acc
        for _c in range(3):
            w = m.vec(m.tanh(g * w[0] + w[1] * 0.3), m.sin(w[2] + g * 0.1), w[0] * w[1])
        acc = acc * 0.6 + w * (0.2 + 0.1 * r)
    out = acc


@hawk.kernel
def stop_outer(x: S, g: S, out: OUT, n: hawk.Mutable[S]):
    acc = x
    count = -1.0
    for r in range(5):
        inner = acc
        for _c in range(3):
            inner = m.tanh(g * inner + 0.2)
        acc = acc * 1.4 + inner
        if acc > 3.0:
            count = 1.0 * r
            break
    out = acc
    n = count


def _prim(kernel, **env):
    got = evaluate(kernel.sinks, {**env, "terminated": False}, lanes=4)
    return {k: np.asarray(v, dtype=float) for k, v in got.items()}


def _fd(kernel, name, env, wrt, sink="out", h=1e-6):
    hi, lo = dict(env), dict(env)
    hi[wrt] = np.asarray(env[wrt], dtype=float) + h
    lo[wrt] = np.asarray(env[wrt], dtype=float) - h
    return (_prim(kernel, **hi)[sink] - _prim(kernel, **lo)[sink]) / (2 * h)


@hawk.kernel
def bare_outer_values(x: S, g: S, out: OUT):
    """The enclosing loop's index and carry read BARE inside the inner body."""
    acc = x
    for r in range(3):
        inner = 0.3
        for _c in range(3):
            inner = m.tanh(inner * g + r + 0.2 * acc)
        acc = acc * 0.8 + inner
    out = acc


SCALAR_CASES = [("sum_of_inner", sum_of_inner), ("carried_into_inner", carried_into_inner),
                ("triple", triple), ("bare_outer_values", bare_outer_values)]


@pytest.mark.parametrize("label,kernel", SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_reverse_matches_finite_differences(label, kernel):
    env = {"x": 0.37, "g": 0.9}
    got = evaluate(vjp(kernel, wrt=("x", "g")), {**env, "terminated": False, "bar_out": 1.0})
    for wrt in ("x", "g"):
        np.testing.assert_allclose(got["bar_" + wrt], _fd(kernel, label, env, wrt),
                                   rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("label,kernel", SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_forward_matches_finite_differences(label, kernel):
    env = {"x": 0.37, "g": 0.9}
    for wrt in ("x", "g"):
        got = evaluate(jvp(kernel, wrt=(wrt,)),
                       {**env, "terminated": False, "dot_" + wrt: 1.0})
        np.testing.assert_allclose(got["dot_out"], _fd(kernel, label, env, wrt),
                                   rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("label,kernel", SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_forward_and_reverse_are_each_others_transpose(label, kernel):
    env = {"x": -0.21, "g": 1.3, "terminated": False}
    rev = evaluate(vjp(kernel, wrt=("x", "g")), {**env, "bar_out": 0.8})
    fwd = evaluate(jvp(kernel, wrt=("x", "g")), {**env, "dot_x": 0.6, "dot_g": -0.4})
    lhs = 0.8 * float(fwd["dot_out"])
    rhs = float(rev["bar_x"]) * 0.6 + float(rev["bar_g"]) * -0.4
    assert abs(lhs - rhs) < 1e-12 * max(1.0, abs(lhs))


def test_a_vector_carry_through_the_inner_loop():
    v = np.array([0.3, -0.2, 0.5])
    env = {"v": v, "g": 0.8, "terminated": False}
    bar = np.array([1.0, -0.5, 0.25])
    rev = evaluate(vjp(vector_carry, wrt=("v", "g")), {**env, "bar_out": bar})
    for i in range(3):
        hi, lo = v.copy(), v.copy()
        hi[i] += 1e-6
        lo[i] -= 1e-6
        d = (_prim(vector_carry, **{**env, "v": hi})["out"]
             - _prim(vector_carry, **{**env, "v": lo})["out"]) / 2e-6
        np.testing.assert_allclose(float(np.dot(bar, d)), float(rev["bar_v"][i]), rtol=1e-6, atol=1e-8)
    fwd = evaluate(jvp(vector_carry, wrt=("v",)), {**env, "dot_v": np.array([0.4, 0.1, -0.3])})
    dv = np.array([0.4, 0.1, -0.3])
    want = sum(float(rev["bar_v"][i]) * dv[i] for i in range(3))
    np.testing.assert_allclose(float(np.dot(bar, fwd["dot_out"])), want, rtol=1e-9)


def test_an_outer_break_around_a_nested_loop_is_exact():
    env = {"x": 0.9, "g": 0.7, "terminated": False}
    assert _prim(stop_outer, **env)["n"] >= 0           # it does leave early
    rev = evaluate(vjp(stop_outer, wrt=("x", "g")), {**env, "bar_out": 1.0, "bar_n": 0.0})
    for wrt in ("x", "g"):
        np.testing.assert_allclose(rev["bar_" + wrt], _fd(stop_outer, "stop", env, wrt),
                                   rtol=1e-6, atol=1e-8)
    fwd = evaluate(jvp(stop_outer, wrt=("x",)), {**env, "dot_x": 1.0})
    np.testing.assert_allclose(fwd["dot_out"], _fd(stop_outer, "stop", env, "x"), rtol=1e-6, atol=1e-8)


def test_the_emitted_inner_tape_is_local_to_the_reversed_iteration():
    d = vjp(carried_into_inner, wrt=("x", "g"))
    text = render_body(tuple(d), canonical(tuple(d))).text
    # the inner forward loop, its local tape and the inner reverse sit INSIDE the
    # reversed outer loop (the outer tape is declared before it)
    outer_reverse = text.index("--hawk_k")
    assert text.index("Real hawk_tp", outer_reverse) > outer_reverse


# -- gathers ------------------------------------------------------------------

N = 4


@hawk.kernel
def inner_gather(x: S, w: hawk.Table["row":N], out: OUT):
    acc = x
    for r in range(2):
        for c in range(N):
            acc = m.tanh(acc * 0.8 + w.at(c) * 0.3)
    out = acc


def test_an_inner_gather_of_an_undifferentiated_plane_is_exact_both_ways():
    w = np.array([0.4, -0.3, 0.2, 0.7])
    env = {"x": 0.5, "w": w, "terminated": False}
    rev = evaluate(vjp(inner_gather, wrt=("x",)), {**env, "bar_out": 1.0}, lanes=N)
    want = _fd(inner_gather, "ig", env, "x")
    np.testing.assert_allclose(rev["bar_x"], want, rtol=1e-7, atol=1e-9)
    fwd = evaluate(jvp(inner_gather, wrt=("x", "w")), {**env, "dot_x": 0.0, "dot_w": np.eye(N)[1]}, lanes=N)
    hi, lo = w.copy(), w.copy()
    hi[1] += 1e-6
    lo[1] -= 1e-6
    d = (_prim(inner_gather, **{**env, "w": hi})["out"] - _prim(inner_gather, **{**env, "w": lo})["out"]) / 2e-6
    np.testing.assert_allclose(fwd["dot_out"], d, rtol=1e-7, atol=1e-9)


def test_an_inner_gather_of_a_differentiated_plane_is_refused_in_reverse_by_name():
    with pytest.raises(HawkError, match="gathers a plane being differentiated"):
        vjp(inner_gather, wrt=("x", "w"))


# -- the shapes that stay refused ----------------------------------------------


@hawk.kernel
def inner_break(x: S, out: OUT):
    acc = x
    for _r in range(3):
        for _c in range(3):
            acc = acc * 1.3
            if acc > 2.0:
                break
    out = acc


def test_an_inner_break_is_refused_by_name_in_both_directions():
    for f in (vjp, jvp):
        with pytest.raises(HawkError, match="nested `for` over '_c' with a `break`"):
            f(inner_break, wrt=("x",))


@hawk.kernel(jump=0.04)
def jump_nested(x: S, out: OUT):
    acc = 0.0
    for r in range(2):
        inner = x
        for _c in range(2):
            inner = inner * 0.9 + 0.1 * r
        acc = acc + m.where(inner > 0.3, inner, 0.0)
    out = acc


def test_jump_through_a_nested_loop_is_refused_by_name():
    with pytest.raises(HawkError, match="nested"):
        vjp(jump_nested, wrt=("x",))


# -- second order -----------------------------------------------------------------


@hawk.kernel
def second(x: S, out: OUT):
    total = 0.0
    for r in range(3):
        inner = 0.0
        for _c in range(2):
            inner = m.tanh(inner * 0.7 + x * r)
        total = total + inner * x
    out = total


def test_forward_over_reverse_through_a_nested_loop_is_exact():
    """jvp(vjp(k)) of a nested loop: the second derivative against a difference
    of the gradient."""
    d = jvp(vjp(second, wrt=("x",)), wrt=("x",))

    def grad(x):
        return float(evaluate(vjp(second, wrt=("x",)),
                              {"x": x, "terminated": False, "bar_out": 1.0})["bar_x"])
    got = evaluate(d, {"x": 0.4, "terminated": False, "bar_out": 1.0, "dot_x": 1.0})
    want = (grad(0.4 + 1e-5) - grad(0.4 - 1e-5)) / 2e-5
    np.testing.assert_allclose(float(got["dot_bar_x"]), want, rtol=1e-6)


# -- the compiled host kernel ---------------------------------------------------------


def test_the_compiled_host_kernels_equal_the_twin(tmp_path_factory, cache_dir):
    """Reverse and forward of ``carried_into_inner`` compiled for the host: every
    lane of 16 equals the interpreter to host contraction."""
    from conftest import sidecar_of

    from hawk import runtime as rt
    from hawk.artifact import build_bundle
    from hawk.trace.kernel import Kernel

    rev = vjp(carried_into_inner, wrt=("x", "g"))
    fwd = jvp(carried_into_inner, wrt=("x", "g"))
    root = tmp_path_factory.mktemp("nested_loops")
    bundle = build_bundle([carried_into_inner, Kernel("nested_rev", rev), Kernel("nested_fwd", fwd)],
                          root, targets=("host",), cache_dir=cache_dir, host_profile="native",
                          mode="float64")
    rng = np.random.default_rng(3)
    n = 16
    x, g = rng.uniform(-0.6, 0.6, n), rng.uniform(0.4, 1.4, n)
    bar = rng.normal(size=n)
    bx, bg = np.zeros(n), np.zeros(n)
    rt.run(rt.load(bundle.directory, "nested_rev", sidecar_of(bundle, "nested_rev")),
           x=x, g=g, bar_out=bar, bar_x=bx, bar_g=bg, terminated=np.zeros(n, dtype=bool))
    dx, dg, dout = rng.normal(size=n), rng.normal(size=n), np.zeros(n)
    rt.run(rt.load(bundle.directory, "nested_fwd", sidecar_of(bundle, "nested_fwd")),
           x=x, g=g, dot_x=dx, dot_g=dg, dot_out=dout, terminated=np.zeros(n, dtype=bool))
    for s in range(n):
        env = {"x": x[s], "g": g[s], "terminated": False}
        want = evaluate(rev, {**env, "bar_out": bar[s]})
        np.testing.assert_allclose([bx[s], bg[s]], [want["bar_x"], want["bar_g"]], rtol=1e-12, atol=1e-13)
        wf = evaluate(fwd, {**env, "dot_x": dx[s], "dot_g": dg[s]})
        np.testing.assert_allclose(dout[s], wf["dot_out"], rtol=1e-12, atol=1e-13)
