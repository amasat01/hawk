# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-case scoped hoisting for ``dispatch(..., strategy="switch")``.

``_Renderer._dispatch`` renders each ``case j: { ... }`` fresh, scoped to
that block, instead of hoisting every branch's temporaries ahead of the
switch regardless of which case needs them — a switch that only selects one
branch must not pay, every element, for computing all K. ``predicated`` is
unaffected: it is the evaluate-all strategy by definition, so there is no
per-branch scope to gain. These rows are text-level (no compile) except the
one numeric row at the bottom, which builds and runs the host artifact only
(no GPU)."""

from __future__ import annotations
from raptor.wings.base.dispatch import dispatch as _dispatch
from _specs import HOST, default_kind
from _specs import aether_writer

import re

import numpy as np
import pytest

import hawk
import hawk.math as M
from raptor.ir import Assign, Dispatch, Leaf, canonical
from raptor.ir.types import TensorType

S = TensorType((), "f64")


def _leaf(name: str, t: TensorType, role: str | None = None) -> Leaf:
    role = role or {0: "per_sample"}[len(t.shape)]
    return Leaf("vocab_read", role, name, t)


@hawk.kernel
def _dk_switch(k: hawk.Index, a: hawk.Scalar, x: hawk.Scalar, b: hawk.Scalar,
               y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [a * x, M.exp(x), M.log(x * x + 1.0) * b],
                        strategy="switch")


@hawk.kernel
def _dk_predicated(k: hawk.Index, a: hawk.Scalar, x: hawk.Scalar, b: hawk.Scalar,
                   y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [a * x, M.exp(x), M.log(x * x + 1.0) * b],
                        strategy="predicated")


# --------------------------------------------------------------------------- #
# 1. Row own shape: no exp/log before the switch, shared reads before,
#    one branch body per case.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 2. predicated is untouched -- same shape, and textually unaffected by the
# switch-only scoping logic.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 3. A value EVERY non-elided branch reads is computed once, above the switch;
#    one only some branches read stays in its case.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 4. the elision is unaffected by the scoping change.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 5. Numeric identity (Part B): the scoped switch still matches a plain
#    numpy oracle -- HOST target only, no GPU.
# --------------------------------------------------------------------------- #
def test_switch_numeric_matches_numpy_on_the_host(tmp_path_factory, cache_dir):
    import _oracle as O
    from conftest import sidecar_of

    from hawk.artifact import build_bundle

    root = tmp_path_factory.mktemp("hawk_disp1_numeric")
    bundle = build_bundle([_dk_switch], root, targets=("host",), cache_dir=cache_dir)
    n = 64
    rng = np.random.default_rng(20260910)
    kind = rng.integers(-2, 5, size=n).astype(np.int32)
    a = rng.standard_normal(n)
    x = rng.standard_normal(n)
    b = rng.standard_normal(n)
    got = O.run(bundle.directory, "_dk_switch", n, sidecar_of(bundle, "_dk_switch"),
               k=kind, a=a, x=x, b=b, y=np.zeros(n, dtype=np.float64))
    got = np.asarray(got)
    clamped = np.clip(kind, 0, 2)
    branch0 = a * x
    branch1 = np.exp(x)
    branch2 = np.log(x * x + 1.0) * b
    want = np.where(clamped == 0, branch0, np.where(clamped == 1, branch1, branch2))
    worst = float(np.max(np.abs(got - want)))
    limit = 8 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(want))))
    assert worst <= limit, f"host dispatch differs from numpy by {worst}, band {limit}"


# --------------------------------------------------------------------------- #
# 6. Sibling switches on one selector fuse into ONE switch.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 7. The derived dispatch seeds a node every branch reads ONCE.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 8. The all-but-one hoist is speculative, so it refuses what could fault.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 9. A switch inside a loop body is scoped like one outside it.
# --------------------------------------------------------------------------- #
def _twice(v, x):
    """``v`` read twice, so the renderer names it."""
    return v * x + v


@hawk.kernel
def _dk_switch_in_loop(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    acc = 0.0
    for t in range(4):
        xv = x + 0.25 * t
        acc = acc + M.dispatch(k, [_twice(M.exp(xv), xv), _twice(M.log(xv * xv + 1.0), xv), xv],
                               strategy="switch")
    y = acc


def test_a_switch_in_a_loop_body_matches_numpy_on_the_host(tmp_path_factory, cache_dir):
    import _oracle as O
    from conftest import sidecar_of

    from hawk.artifact import build_bundle

    root = tmp_path_factory.mktemp("hawk_disp_loop")
    bundle = build_bundle([_dk_switch_in_loop], root, targets=("host",), cache_dir=cache_dir)
    n = 48
    rng = np.random.default_rng(20261025)
    kind = rng.integers(0, 3, size=n).astype(np.int32)
    x = rng.uniform(0.1, 0.9, n)
    got = np.asarray(O.run(bundle.directory, "_dk_switch_in_loop", n,
                           sidecar_of(bundle, "_dk_switch_in_loop"),
                           k=kind, x=x, y=np.zeros(n, dtype=np.float64)))
    want = np.zeros(n)
    for t in range(4):
        xv = x + 0.25 * t
        want += np.choose(kind, [np.exp(xv) * xv + np.exp(xv),
                                 np.log(xv * xv + 1.0) * xv + np.log(xv * xv + 1.0), xv])
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)


# --------------------------------------------------------------------------- #
# 10. A traced if/elif chain that would run every arm's costly work lowers as a
#     switch; one whose costly work inlines into `?:` stays a select.
# --------------------------------------------------------------------------- #
@hawk.kernel
def _dk_ladder(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar],
               z: hawk.Mutable[hawk.Scalar], w: hawk.Mutable[hawk.Scalar]):
    v = 0.0
    if k == 0:
        v = _twice(M.exp(x), x)
    elif k == 1:
        v = _twice(M.log(x * x + 1.0), x)
    else:
        v = _twice(M.tanh(x), x)
    y = v
    u = 0.0                                  # no `else`: k == 0 keeps the zero
    if k == 1:
        u = _twice(M.sin(x), x)
    elif k == 2:
        u = _twice(M.cos(x), x)
    z = u
    if x > 0.5:                              # not an equality on a kind
        s = _twice(M.exp(-x), x)
    else:
        s = _twice(M.sqrt(x) / (x + 1.0), x)
    w = s


@hawk.kernel
def _dk_ladder_explicit(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar],
                        z: hawk.Mutable[hawk.Scalar], w: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [_twice(M.exp(x), x), _twice(M.log(x * x + 1.0), x),
                       _twice(M.tanh(x), x)], strategy="switch")
    z = M.dispatch(k, [0.0, _twice(M.sin(x), x), _twice(M.cos(x), x)], strategy="switch")
    w = M.dispatch(M.int32(M.where(x > 0.5, 0.0, 1.0)),
                   [_twice(M.exp(-x), x), _twice(M.sqrt(x) / (x + 1.0), x)], strategy="switch")


@hawk.kernel
def _dk_ladder_rewritten(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    v = 0.0
    if k == 0:
        v = M.exp(x) * M.exp(x) + M.exp(x)            # one value, written three times
    elif k == 1:
        v = x * 3.0
    else:
        v = x
    y = v


def test_a_scalar_chain_switch_keeps_its_value_and_its_derivatives():
    from _eval import evaluate

    from hawk.diff import jvp, vjp

    h = 1e-6
    for kind in range(3):
        for x in (0.3, 0.7):
            want = [np.exp(x) ** 2 + np.exp(x), 3.0 * x, x][kind]
            assert evaluate(_dk_ladder_rewritten.sinks, {"k": kind, "x": x})["y"] == pytest.approx(
                want, rel=1e-14)
            slope = (evaluate(_dk_ladder_rewritten.sinks, {"k": kind, "x": x + h})["y"]
                     - evaluate(_dk_ladder_rewritten.sinks, {"k": kind, "x": x - h})["y"]) / (2 * h)
            rev = evaluate(vjp(_dk_ladder_rewritten.sinks), {"k": kind, "x": x, "bar_y": 1.0})["bar_x"]
            assert rev == pytest.approx(slope, rel=1e-6)
            fwd = evaluate(jvp(_dk_ladder_rewritten.sinks), {"k": kind, "x": x, "dot_x": 1.0})["dot_y"]
            assert fwd == pytest.approx(slope, rel=1e-6)


def _ladder_inputs(n: int):
    rng = np.random.default_rng(20261026)
    return rng.integers(0, 3, size=n).astype(np.int32), rng.uniform(0.05, 0.95, n)


def test_a_ladder_equals_its_explicit_dispatch_on_the_host(tmp_path_factory, cache_dir):
    import _oracle as O
    from conftest import sidecar_of

    from hawk.artifact import build_bundle

    root = tmp_path_factory.mktemp("hawk_ladder_host")
    bundle = build_bundle([_dk_ladder, _dk_ladder_explicit], root, targets=("host",),
                          cache_dir=cache_dir)
    n = 64
    kind, x = _ladder_inputs(n)
    outs = []
    for name in ("_dk_ladder", "_dk_ladder_explicit"):
        planes = {p: np.zeros(n) for p in ("y", "z", "w")}
        O.run(bundle.directory, name, n, sidecar_of(bundle, name), k=kind, x=x, **planes)
        outs.append(planes)
    for p in ("y", "z", "w"):
        np.testing.assert_array_equal(outs[0][p], outs[1][p])
    assert np.all(outs[0]["z"][kind == 0] == 0.0)


def test_the_reverse_of_a_ladder_is_scoped_and_matches_a_finite_difference():
    from types import SimpleNamespace

    from _eval import evaluate

    from hawk.artifact import emit_source
    from hawk.diff import vjp

    @hawk.kernel
    def ladder(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        v = 0.0
        if k == 0:
            v = _twice(M.exp(x), x)
        elif k == 1:
            v = _twice(M.log(x * x + 1.0), x)
        else:
            v = _twice(M.tanh(x), x)
        y = v

    derived = vjp(ladder.sinks)
    kernel = SimpleNamespace(name="ladder_vjp", sinks=tuple(derived), walk=canonical(derived))
    text = emit_source(kernel, targets=("host",)).sources["host"].text
    assert "aether::math::" not in text[:text.index("switch (")]
    h = 1e-6
    for kind in range(3):
        for x in (0.3, 0.7):
            got = evaluate(derived, {"k": kind, "x": x, "bar_y": 1.0})["bar_x"]
            plus = evaluate(ladder.sinks, {"k": kind, "x": x + h})["y"]
            minus = evaluate(ladder.sinks, {"k": kind, "x": x - h})["y"]
            assert abs(got - (plus - minus) / (2 * h)) <= 1e-7 * max(1.0, abs(got))


@pytest.mark.gpu
def test_a_ladder_equals_its_explicit_dispatch_on_the_device(tmp_path_factory, cache_dir):
    import cupy as cp
    import eagle.exec as eexec
    from _deploy import device_plugin
    from conftest import sidecar_of
    from eagle import plan as eplan

    from hawk.artifact import build_bundle

    root = tmp_path_factory.mktemp("hawk_ladder_cuda")
    bundle = build_bundle([_dk_ladder, _dk_ladder_explicit], root, targets=("cuda",),
                          cache_dir=cache_dir)
    n = 256
    kind, x = _ladder_inputs(n)
    outs = []
    for name in ("_dk_ladder", "_dk_ladder_explicit"):
        plugin = device_plugin(bundle.directory, name, sidecar_of(bundle, name))
        planes = {p: cp.zeros(n) for p in ("y", "z", "w")}
        eplan.plan(plugin, structure=eexec.DeviceKernel, npartitions=1).run(
            k=cp.asarray(kind), x=cp.asarray(x), **planes)
        outs.append({p: v.get() for p, v in planes.items()})
    for p in ("y", "z", "w"):
        np.testing.assert_array_equal(outs[0][p], outs[1][p])


