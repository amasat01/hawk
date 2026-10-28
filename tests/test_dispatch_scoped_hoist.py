# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-case scoped hoisting for ``dispatch(..., policy="switch")``.

``_Renderer._dispatch`` renders each ``case j: { ... }`` fresh, scoped to
that block, instead of hoisting every branch's temporaries ahead of the
switch regardless of which case needs them — a switch that only selects one
branch must not pay, every element, for computing all K. ``predicated`` is
unaffected: it is the evaluate-all policy by definition, so there is no
per-branch scope to gain. These rows are text-level (no compile) except the
one numeric row at the bottom, which builds and runs the host artifact only
(no GPU)."""

from __future__ import annotations

import re

import numpy as np
import pytest

import hawk
import hawk.math as M
from hawk.emit import render_body
from hawk.ir import Assign, Dispatch, Leaf, canonical
from hawk.types import TensorType

S = TensorType((), "f64")


def _leaf(name: str, t: TensorType, role: str | None = None) -> Leaf:
    role = role or {0: "per_sample"}[len(t.shape)]
    return Leaf("vocab_read", role, name, t)


@hawk.kernel
def _dk_switch(k: hawk.Index, a: hawk.Scalar, x: hawk.Scalar, b: hawk.Scalar,
               y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [a * x, M.exp(x), M.log(x * x + 1.0) * b],
                        policy="switch")


@hawk.kernel
def _dk_predicated(k: hawk.Index, a: hawk.Scalar, x: hawk.Scalar, b: hawk.Scalar,
                   y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [a * x, M.exp(x), M.log(x * x + 1.0) * b],
                        policy="predicated")


def _switch_text() -> str:
    sinks = _dk_switch.sinks
    return render_body(sinks, canonical(sinks)).text


def _predicated_text() -> str:
    sinks = _dk_predicated.sinks
    return render_body(sinks, canonical(sinks)).text


def _split_before_after_switch(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    switch_i = next(i for i, l in enumerate(lines) if "switch (" in l)
    return lines[:switch_i], lines[switch_i:]


# --------------------------------------------------------------------------- #
# 1. Row own shape: no exp/log before the switch, shared reads before,
#    one branch body per case.
# --------------------------------------------------------------------------- #
def test_switch_keeps_no_math_call_before_the_switch():
    before, after = _split_before_after_switch(_switch_text())
    before_text = "\n".join(before)
    assert "aether::math::exp(" not in before_text
    assert "aether::math::log(" not in before_text
    after_text = "\n".join(after)
    assert "aether::math::exp(" in after_text
    assert "aether::math::log(" in after_text


def test_switch_emits_one_case_block_per_branch():
    text = _switch_text()
    assert "case 0: {" in text
    assert "case 1: {" in text
    assert "default: {" in text
    assert "switch (" in text


def test_switch_hoists_only_what_every_case_reads():
    """``x`` is read by the default case twice (``x * x``) and by cases 0 and 1
    once: every case reads it, so it is computed ONCE above the switch (and named,
    being read more than once). ``a``/``b`` are read by one case each and stay in
    their case."""
    before, after = _split_before_after_switch(_switch_text())
    before_text = "\n".join(before)
    assert before_text.count("psc_x[i].eval()") == 1
    assert "psc_a" not in before_text and "psc_b" not in before_text
    assert "psc_x[i].eval()" not in "\n".join(after)


# --------------------------------------------------------------------------- #
# 2. predicated is untouched -- same shape, and textually unaffected by the
# switch-only scoping logic.
# --------------------------------------------------------------------------- #
def test_predicated_still_hoists_the_shared_read_before_the_ternary():
    """``x`` is read by all three predicated branches (via the CHAIN, since
    every candidate is always live) -- the OLD, un-scoped behaviour: a single
    shared ``const auto`` local ahead of the ``?:`` chain, never per-branch
    duplication (predicated is the evaluate-ALL policy, so there is
    nothing to gain by scoping it)."""
    text = _predicated_text()
    assert "switch (" not in text
    assert " ? " in text
    assert re.search(r"const auto \w+ = psc_x\[i\]\.eval\(\);", text)
    # exactly ONE x-read local -- not duplicated per branch.
    assert len(re.findall(r"psc_x\[i\]\.eval\(\)", text)) == 1


# --------------------------------------------------------------------------- #
# 3. A value EVERY non-elided branch reads is computed once, above the switch;
#    one only some branches read stays in its case.
# --------------------------------------------------------------------------- #
@hawk.kernel
def _dk_shared_across_branches(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [x * x + 1.0, x * x + 2.0, x * x + 3.0],
                        policy="switch")


@hawk.kernel
def _dk_shared_by_two_of_three(k: hawk.Index, x: hawk.Scalar, w: hawk.Scalar,
                               y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [x * x + 1.0, x * x + 2.0, w + 3.0], policy="switch")


def test_a_value_every_branch_shares_is_hoisted_once_above_the_switch():
    sinks = _dk_shared_across_branches.sinks
    text = render_body(sinks, canonical(sinks)).text
    before, after = _split_before_after_switch(text)
    assert "\n".join(before).count("psc_x[i].eval()") == 1
    assert "psc_x[i].eval()" not in "\n".join(after)
    # `x * x` is the common sub-expression and is named once, before the switch.
    assert len(re.findall(r"const auto \w+ = \(?tb?\w+ \* ", "\n".join(before))) <= 1


@hawk.kernel
def _dk_shared_by_two_of_four(k: hawk.Index, x: hawk.Scalar, w: hawk.Scalar,
                              y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [x * x + 1.0, x * x + 2.0, w + 3.0, w * 2.0], policy="switch")


def test_a_value_all_but_one_branch_reads_is_hoisted():
    """With three or more cases, what all but one read is computed once above
    the switch (the extra case computes it unused)."""
    sinks = _dk_shared_by_two_of_three.sinks
    before, after = _split_before_after_switch(render_body(sinks, canonical(sinks)).text)
    assert "\n".join(before).count("psc_x[i].eval()") == 1
    assert "psc_x[i].eval()" not in "\n".join(after)


def test_a_value_fewer_branches_read_is_not_hoisted():
    """``x * x`` is read by two of four cases: the other two must not pay for it."""
    sinks = _dk_shared_by_two_of_four.sinks
    before, after = _split_before_after_switch(render_body(sinks, canonical(sinks)).text)
    assert "psc_x" not in "\n".join(before)
    assert "psc_x[i].eval()" in "\n".join(after)


# --------------------------------------------------------------------------- #
# 4. the elision is unaffected by the scoping change.
# --------------------------------------------------------------------------- #
def test_elision_still_works_under_the_scoped_renderer():
    from hawk.diff.rules import zero_like

    k = _leaf("k", TensorType((), "i32"))
    b0, b2 = _leaf("b0", S), _leaf("b2", S)
    z = zero_like(S)
    d = Dispatch(k, [b0, z, b2], "switch", S)
    sinks = (Assign("out", d, S),)
    text = render_body(sinks, canonical(sinks)).text
    assert "case 1:" not in text
    assert "case 0: {" in text
    # an ELIDED branch's default writes the typed zero DIRECTLY (no
    # branch scope needed there, since there is no branch left to render).
    assert "default:" in text
    assert "static_cast<Real>(0)" in text


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
    kind = rng.integers(-2, 5, size=n).astype(np.int64)
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
@hawk.kernel
def _dk_siblings(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar],
                 z: hawk.Mutable[hawk.Scalar]):
    s = M.exp(x * x)
    g = M.log(s + 2.0)
    y = M.dispatch(k, [s + 1.0, g], policy="switch")
    z = M.dispatch(k, [s * 3.0, g * 4.0], policy="switch")


def test_sibling_switches_on_one_selector_fuse_into_one():
    sinks = _dk_siblings.sinks
    text = render_body(sinks, canonical(sinks)).text
    assert text.count("switch (") == 1
    assert text.count("const std::int32_t hawk_disp") == 1       # one clamp
    assert text.count("aether::math::log(") == 1                  # shared by both members
    assert len(re.findall(r"hawk_disp\d+ = ", text)) == 4         # each case assigns both


# --------------------------------------------------------------------------- #
# 7. The derived dispatch seeds a node every branch reads ONCE.
# --------------------------------------------------------------------------- #
@hawk.kernel
def _dk_shared_operand(k: hawk.Index, a: hawk.Scalar, b: hawk.Scalar, x: hawk.Scalar,
                       y: hawk.Mutable[hawk.Scalar]):
    s = M.sin(x)
    y = M.dispatch(k, [s * a, s * b], policy="switch")


def test_vjp_runs_a_shared_operands_chain_once_with_a_combined_seed():
    from hawk.diff import vjp

    derived = vjp(_dk_shared_operand.sinks)
    text = render_body(derived, canonical(derived)).text
    assert text.count("aether::math::cos(") == 1          # d sin(x), once, not per case


# --------------------------------------------------------------------------- #
# 8. The all-but-one hoist is speculative, so it refuses what could fault.
# --------------------------------------------------------------------------- #
@hawk.kernel
def _dk_unsafe_shared(k: hawk.Index, m: hawk.Index, x: hawk.Scalar,
                      y: hawk.Mutable[hawk.Scalar]):
    q = M.real(m // 3) + x          # integer division: SIGFPE on the host if run unneeded
    y = M.dispatch(k, [q * 1.0, q * 2.0, x + 3.0], policy="switch")


def test_the_all_but_one_hoist_refuses_an_integer_division():
    sinks = _dk_unsafe_shared.sinks
    before, after = _split_before_after_switch(render_body(sinks, canonical(sinks)).text)
    assert "int_floordiv" not in "\n".join(before)
    assert "int_floordiv" in "\n".join(after)


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
                               policy="switch")
    y = acc


def _loop_body(text: str) -> str:
    return text[text.index("for ("):]


def test_a_switch_in_a_loop_body_keeps_its_math_in_its_cases():
    sinks = _dk_switch_in_loop.sinks
    body = _loop_body(render_body(sinks, canonical(sinks)).text)
    before, after = body.split("switch (", 1)
    assert "aether::math::exp(" not in before and "aether::math::log(" not in before
    assert "aether::math::exp(" in after and "aether::math::log(" in after


def test_a_switch_in_a_loop_body_matches_numpy_on_the_host(tmp_path_factory, cache_dir):
    import _oracle as O
    from conftest import sidecar_of

    from hawk.artifact import build_bundle

    root = tmp_path_factory.mktemp("hawk_disp_loop")
    bundle = build_bundle([_dk_switch_in_loop], root, targets=("host",), cache_dir=cache_dir)
    n = 48
    rng = np.random.default_rng(20261025)
    kind = rng.integers(0, 3, size=n).astype(np.int64)
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
                       _twice(M.tanh(x), x)], policy="switch")
    z = M.dispatch(k, [0.0, _twice(M.sin(x), x), _twice(M.cos(x), x)], policy="switch")
    w = M.dispatch(M.int32(M.where(x > 0.5, 0.0, 1.0)),
                   [_twice(M.exp(-x), x), _twice(M.sqrt(x) / (x + 1.0), x)], policy="switch")


@hawk.kernel
def _dk_ladder_vector(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Vector[2]]):
    v = M.vec(0.0, 0.0)
    if k == 0:
        v = M.vec(M.exp(x), 1.0)
    elif k == 1:
        v = M.vec(M.log(x + 1.0), 2.0)
    else:
        v = M.vec(M.tanh(x), 3.0)
    y = v


@hawk.kernel
def _dk_ladder_lazy(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    v = 0.0
    if k == 0:
        v = M.exp(x) * x
    elif k == 1:
        v = M.log(x * x + 1.0) + x
    else:
        v = M.tanh(x)
    y = v


def _text(kernel) -> str:
    return render_body(kernel.sinks, canonical(kernel.sinks)).text


def test_a_costly_ladder_renders_one_switch_per_chain():
    text = _text(_dk_ladder)
    assert text.count("switch (") == 3
    head = text[:text.index("switch (")]
    assert "aether::math::" not in head
    assert [d.policy for d in _dk_ladder.walk.dispatches] == ["switch"] * 3


def test_a_vector_ladder_renders_one_switch():
    text = _text(_dk_ladder_vector)
    assert text.count("switch (") == 1
    assert "aether::select(" not in text
    assert "aether::math::" not in text[:text.index("switch (")]


def test_a_lazy_ladder_stays_a_select():
    text = _text(_dk_ladder_lazy)
    assert "switch (" not in text and " ? " in text
    assert not _dk_ladder_lazy.walk.dispatches


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


def test_a_scalar_chain_whose_arm_writes_a_costly_value_twice_is_a_switch():
    text = _text(_dk_ladder_rewritten)
    assert text.count("switch (") == 1
    assert text.count("aether::math::exp(") == 1
    assert text.index("aether::math::exp(") > text.index("case 0:")
    assert text.index("aether::math::exp(") < text.index("case 1:")


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
    return rng.integers(0, 3, size=n).astype(np.int64), rng.uniform(0.05, 0.95, n)


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


def test_the_reverse_of_a_ladder_is_scoped_and_matches_a_finite_difference():
    from _eval import evaluate

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
    text = render_body(derived, canonical(derived)).text
    assert "aether::math::" not in text[:text.index("switch (")]
    h = 1e-6
    for kind in range(3):
        for x in (0.3, 0.7):
            got = evaluate(derived, {"k": kind, "x": x, "bar_y": 1.0})["bar_x"]
            plus = evaluate(ladder.sinks, {"k": kind, "x": x + h})["y"]
            minus = evaluate(ladder.sinks, {"k": kind, "x": x - h})["y"]
            assert abs(got - (plus - minus) / (2 * h)) <= 1e-7 * max(1.0, abs(got))
