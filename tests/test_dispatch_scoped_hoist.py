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


def test_switch_has_no_const_auto_temporary_before_the_switch():
    """Every value this micro-kernel's branches touch (``a``, ``x``, ``b``)
    is used at most once WITHIN any single branch except ``x`` inside the
    default branch (``x * x``) -- none of that is shared with anything
    OUTSIDE a branch, so nothing should be materialised before the switch at
    all (the selector's own clamp local is the one line before it, and it is
    not a ``const auto``)."""
    before, _ = _split_before_after_switch(_switch_text())
    before_text = "\n".join(before)
    assert "const auto" not in before_text


def test_the_default_branch_names_x_only_inside_its_own_case():
    """``x`` is read twice inside the default branch (``x * x``) -- HAWK's
    own rule fires there (local fan-out 2), but ONLY there: case 0 and case 1
    each read ``x`` once and must inline it, never sharing a name with the
    default case's own materialisation."""
    text = _switch_text()
    lines = text.splitlines()
    default_i = next(i for i, l in enumerate(lines) if "default: {" in l)
    close_i = next(i for i in range(default_i, len(lines)) if lines[i].strip() == "}")
    default_block = "\n".join(lines[default_i:close_i])
    assert re.search(r"const auto \w+ = psc_x\[i\]\.eval\(\);", default_block)
    case0_i = next(i for i, l in enumerate(lines) if "case 0: {" in l)
    case1_i = next(i for i, l in enumerate(lines) if "case 1: {" in l)
    case0_block = "\n".join(lines[case0_i:case1_i])
    assert "const auto" not in case0_block


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
# 3. A value two DIFFERENT switch branches both need is DUPLICATED, not
#    hoisted back before the switch (Part A's own design choice).
# --------------------------------------------------------------------------- #
@hawk.kernel
def _dk_shared_across_branches(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    # branch 0 and branch 1 both compute the IDENTICAL `x * x` subexpression
    # (structurally deduped to ONE IR node) as part of a DIFFERENT
    # outer expression each.
    y = M.dispatch(k, [x * x + 1.0, x * x + 2.0, x * x + 3.0],
                        policy="switch")


def test_a_value_two_branches_share_is_duplicated_not_hoisted():
    """``x`` is read TWICE within each branch (``x * x``), so HAWK's own
    materialisation rule names it inside EVERY case that needs it (local
    fan-out 2) -- never once, shared, before the switch: no ``psc_x[i].eval()``
    read may appear before the switch at all, and it must appear at least
    TWICE after it (once per case's own, independently-named local)."""
    sinks = _dk_shared_across_branches.sinks
    text = render_body(sinks, canonical(sinks)).text
    before, after = _split_before_after_switch(text)
    before_text = "\n".join(before)
    assert "psc_x[i].eval()" not in before_text
    after_text = "\n".join(after)
    assert after_text.count("psc_x[i].eval()") >= 2
    # each occurrence is its OWN `const auto` local (never one name shared
    # by two cases, which would mean the value was hoisted back outside).
    names = set(re.findall(r"const auto (\w+) = psc_x\[i\]\.eval\(\);", after_text))
    assert len(names) >= 2


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
