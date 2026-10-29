# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivatives of the ray ops, against central differences and on degenerate input.

``hit`` has zero derivative; ``t``, ``u``, ``v`` (the slab's enter and exit, the
sphere's two roots) follow the computed values, also on a miss. A degenerate input
(a ray parallel to the triangle, a zero-area triangle, a zero direction, a
tangent sphere, a slab with a zero direction component) gives finite
gradients, exactly zero where the primitive's own output is the masked zero."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate
from test_ir_rays import ball, box, tri

from hawk.diff import jvp, vjp

H = 1e-6
KINDS = {
    "tri": (tri, ("o", "d", "a", "b", "c"), ()),
    "box": (box, ("o", "d", "lo", "hi"), ()),
    "ball": (ball, ("o", "d", "c", "r"), ()),
}
RNG = np.random.default_rng(17)


def _regular(kind):
    """One scene per call: a ray that hits, and one that misses."""
    if kind == "tri":
        base = dict(o=[0.2, 0.1, -2.0], d=[0.05, -0.02, 1.0], a=[-1.0, -1.0, 0.0], b=[1.0, -1.0, 0.2],
                    c=[0.0, 1.5, -0.1])
        miss = dict(base, o=[3.0, 0.1, -2.0])
    elif kind == "box":
        base = dict(o=[-3.0, 0.2, 0.1], d=[1.0, 0.1, 0.05], lo=[-1.0, -1.0, -1.0], hi=[1.2, 1.0, 0.9])
        miss = dict(base, o=[-3.0, 4.0, 0.1])
    else:
        base = dict(o=[-3.0, 0.2, 0.1], d=[1.0, 0.1, 0.05], c=[0.1, 0.0, 0.0], r=1.3)
        miss = dict(base, o=[-3.0, 4.0, 0.1])
    return {"hit": base, "miss": miss}


def _env(scene):
    out = {k: np.array(v, dtype=float) if isinstance(v, list) else float(v) for k, v in scene.items()}
    out["tmax"] = 100.0
    return out


def _value(kernel, env):
    return np.asarray(evaluate(kernel.sinks, env)["y"])


def _bump(env, name, k, delta):
    out = dict(env)
    if k is None:
        out[name] = env[name] + delta
    else:
        v = env[name].copy()
        v[k] += delta
        out[name] = v
    return out


def _slots(env, names):
    """Every scalar input, as ``(name, component or None)``."""
    return [(n, k) for n in names for k in ([None] if np.ndim(env[n]) == 0 else range(3))]


@pytest.mark.parametrize("which", ["hit", "miss"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_vjp_and_jvp_match_central_differences(kind, which):
    kernel, names, _ = KINDS[kind]
    env = _env(_regular(kind)[which])
    assert (_value(kernel, env)[0] == 1.0) == (which == "hit")
    slots = _slots(env, names)
    fd = np.zeros((4, len(slots)))
    for j, (n, k) in enumerate(slots):
        fd[:, j] = (_value(kernel, _bump(env, n, k, H)) - _value(kernel, _bump(env, n, k, -H))) / (2 * H)
    assert np.all(np.isfinite(fd))
    rev = vjp(kernel.sinks)
    fwd = jvp(kernel.sinks)
    for out in (1, 2, 3):
        if kind != "tri" and out == 3:
            continue
        bar = np.eye(4)[out]
        got = evaluate(rev, {**env, "bar_y": bar})
        row = np.concatenate([np.atleast_1d(got[f"bar_{n}"]) for n in names])
        np.testing.assert_allclose(row, fd[out], rtol=2e-5, atol=2e-7)
    for j, (n, k) in enumerate(slots):
        dot = {f"dot_{m}": (np.zeros(3) if np.ndim(env[m]) else 0.0) for m in names}
        if k is None:
            dot[f"dot_{n}"] = 1.0
        else:
            dot[f"dot_{n}"][k] = 1.0
        got = np.asarray(evaluate(fwd, {**env, **dot})["dot_y"])
        np.testing.assert_allclose(got, fd[:, j], rtol=2e-5, atol=2e-7)
    assert evaluate(fwd, {**env, **{f"dot_{n}": (np.ones(3) if np.ndim(env[n]) else 1.0) for n in names}})["dot_y"][0] == 0.0
    assert evaluate(rev, {**env, "bar_y": np.array([1.0, 0, 0, 0])})[f"bar_{names[0]}"].tolist() == [0.0] * 3


DEGENERATE = {
    "tri": [
        dict(o=[0, 0, -1], d=[1, 0, 0], a=[0, 0, 0], b=[1, 0, 0], c=[0, 1, 0]),         # parallel
        dict(o=[0, 0, -1], d=[0, 0, 1], a=[0, 0, 0], b=[1, 0, 0], c=[2, 0, 0]),         # zero area
        dict(o=[0, 0, -1], d=[0, 0, 0], a=[0, 0, 0], b=[1, 0, 0], c=[0, 1, 0]),         # zero direction
    ],
    "box": [
        dict(o=[0.5, 0.5, -1], d=[0, 0, 1], lo=[0, 0, 0], hi=[1, 1, 1]),               # d_x = d_y = 0, inside
        dict(o=[2.5, 0.5, -1], d=[0, 0, 1], lo=[0, 0, 0], hi=[1, 1, 1]),               # d_x = 0, outside
        dict(o=[0.5, 0.5, 0.5], d=[0, 0, 0], lo=[0, 0, 0], hi=[1, 1, 1]),              # zero direction
    ],
    "ball": [
        dict(o=[0, 0, 5], d=[0, 0, 0], c=[0, 0, 0], r=1.0),                            # zero direction
        dict(o=[0, 2, 0], d=[1, 0, 0], c=[0, 0, 0], r=1.0),                            # no real root
        dict(o=[0, 1, 0], d=[1, 0, 0], c=[0, 0, 0], r=1.0),                            # tangent
    ],
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_degenerate_input_has_finite_gradients_and_zero_where_the_output_is_masked(kind):
    kernel, names, _ = KINDS[kind]
    rev, fwd = vjp(kernel.sinks), jvp(kernel.sinks)
    for scene in DEGENERATE[kind]:
        env = _env(scene)
        want = _value(kernel, env)
        assert np.all(np.isfinite(want))
        for out in range(4):
            got = evaluate(rev, {**env, "bar_y": np.eye(4)[out]})
            for n in names:
                g = np.atleast_1d(got[f"bar_{n}"])
                assert np.all(np.isfinite(g)), (kind, scene, out, n)
        dot = {f"dot_{n}": (np.ones(3) if np.ndim(env[n]) else 1.0) for n in names}
        tangent = np.asarray(evaluate(fwd, {**env, **dot})["dot_y"])
        assert np.all(np.isfinite(tangent)), (kind, scene)
        masked = kind != "box" and not (kind == "ball" and scene["o"] == [0, 1, 0])
        if masked:
            # a zero-denominator triangle, a zero direction, a sphere with no real root
            assert np.all(tangent == 0.0), (kind, scene, tangent)
            for out in range(4):
                got = evaluate(rev, {**env, "bar_y": np.eye(4)[out]})
                assert all(not np.any(np.atleast_1d(got[f"bar_{n}"])) for n in names), (kind, scene, out)
