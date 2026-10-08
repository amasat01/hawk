# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""/: SECOND ORDER through a lowered ``for`` — forward-over-reverse and
reverse-over-forward, checked against a central difference of the gradient.

WHAT A SECOND-ORDER DERIVATIVE OF A LOOP IS. ``jvp(vjp(k))`` and ``vjp(jvp(k))``
both compute a HESSIAN-VECTOR PRODUCT: the directional derivative, along a
tangent ``v``, of the gradient of ``k``. The rows below check it the only
way a second derivative can honestly be checked — by differencing the FIRST
derivative, which is itself already checked against a difference of the
primal in the loop-lowering primal rows.

WHY IT NEEDED NOTHING NEW EXCEPT ONE RULE. A VJP-derived IR's loops are
ordinary loops: the reverse of an accumulator is a forward accumulator
loop that stores nothing ((a)), and the reverse of a recurrence is a
backward loop that reads the forward loop's tape ((b)). Forward mode carries
tangents through either of those exactly as it carries them through a primal —
so the refusal that used to sit in ``hawk/diff/loops.forward_loop`` was never a
statement about the mathematics, only about the ONE node with no tangent rule:
:class:`~hawk.ir.loop_nodes.TapeRead`. Its rule is that the tangent of
``TapeRead(F, j, k)`` is ``TapeRead(F', m + j, k)`` — the same iteration of the
AUGMENTED forward loop, whose slot ``m + j`` already carries the tangent of slot
``j``. In the emitted code the tape becomes a tape of PAIRS, which
:func:`test_the_tape_becomes_a_tape_of_pairs` reads off the generated body.

Both orders are checked on four subjects: two accumulators, whose reverse is
a forward loop that stores nothing, and two recurrences, whose reverse reads
the forward loop's tape. The recurrence case is the more serious of the two
kinds, and was a SEPARATE
defect, found by this file's finite-difference row rather than by the refusal:
``hawk/diff/loops._reverse_recurrence`` pulled back only the carried slots that
arrived with a SEED. A recurrence is exactly the shape in which an unseeded
slot's adjoint stops being zero — it accumulates from every other slot whose
body result reads it — so the terms owed after the first backward iteration were
dropped. A first-order VJP never noticed (a primal's sinks usually seed every
slot its output reads); reverse-over-forward did, because the augmented loop's
seeds cover only its TANGENT half. That is the class of failure a derivative
gate exists for: a graph that walks, hashes, emits and compiles perfectly, and
returns a plausible wrong number.

WHAT STAYS REFUSED, AND WHY IT IS NOT A GAP. Reverse-OVER-reverse still refuses,
by name: the adjoint of a tape read is a scatter INTO the tape — a write to
storage the forward loop owns — and HAWK carries no rule for it. The refusal
names ``jvp(vjp(kernel))`` as the route that computes the same object, which is
the one this file certifies. A loop that COMMITS once per iteration also stays
refused in both directions: that is a derived loop's scatter, and
differentiating it would be differentiating a derivative's WRITE rather than its
value.
"""

from __future__ import annotations

import pathlib

import _toolchain as TC
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.artifact.layout import exports
from hawk.diff import jvp, vjp
from hawk.emit import BACKENDS, CUDA, FLOAT64, render_body, render_source
from hawk.ir import HawkError, Loop, canonical

#: The central-difference step for the SECOND derivative row. It differences the
#: analytic FIRST derivative (not the primal twice), so the usual h**2 error of a
#: central difference applies once, not twice, and 1e-5 keeps truncation and
#: round-off both near 1e-8 on these subjects.
H = 1e-5

ROWS, COLS = 4, 5


# --------------------------------------------------------------------------- #
# The four subjects, re-used verbatim: one per loop class, plus the second
# shape inside each class that behaves differently.
# --------------------------------------------------------------------------- #
@hawk.kernel
def additive(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
    """ACCUMULATOR, one carry."""
    acc = 0.0
    for k in range(8):
        acc = acc + g * m.sin(x + k)
    y = acc


@hawk.kernel
def two_accumulators(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
    """ACCUMULATOR, two carries — a sum and a sum of squares in one pass."""
    total = 0.0
    squares = 0.0
    for k in range(6):
        term = g * m.cos(x + k)
        total = total + term
        squares = squares + term * term
    y = total + squares


@hawk.kernel
def chebyshev(z: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """RECURRENCE, two carries — ``T_{n+1} = 2 z T_n - T_{n-1}``."""
    prev = 1.0
    cur = z
    for _n in range(5):
        prev, cur = cur, 2.0 * z * cur - prev
    y = cur + prev


@hawk.kernel
def saturating(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
    """RECURRENCE, one carry — ``acc = tanh(g*acc + x)``."""
    acc = 0.0
    for _k in range(6):
        acc = m.tanh(g * acc + x)
    y = acc


@hawk.kernel
def gathering(t: hawk.Table["row":ROWS, "col":COLS], y: hawk.Mutable[hawk.Scalar]):
    """An ACCUMULATOR over a GATHER — the one subject whose VJP is a loop that
    COMMITS, and therefore the one whose second derivative stays refused."""
    total = 0.0
    for r in range(ROWS):
        total = total + t.at(row=r, col=2) * 1.5
    y = total


#: ``(id, kernel, env, wrt, tangent)`` — the tangent is the direction ``v`` the
#: Hessian is contracted against, deliberately NOT a basis vector for the
#: two-input subjects: a mixed direction exercises the off-diagonal second
#: derivatives, which a one-hot seed would leave unchecked.
SECOND_ORDER_CASES = [
    ("additive", additive, {"x": 0.7, "g": 1.3}, ("x", "g"), {"x": 1.0, "g": 0.5}),
    ("two_accumulators", two_accumulators, {"x": 0.4, "g": 0.9}, ("x", "g"),
     {"x": 0.75, "g": -1.25}),
    ("chebyshev", chebyshev, {"z": 0.6}, ("z",), {"z": 1.0}),
    ("saturating", saturating, {"x": 0.35, "g": 0.8}, ("x", "g"),
     {"x": -0.4, "g": 1.1}),
]


def _gradient(kernel, env: dict, wrt: tuple, sink: str) -> dict:
    """The analytic FIRST derivative at ``env`` — the thing the second-order row
    differences. It is itself gated against a difference of the primal by
    the loop-lowering primal rows, so this row never leans on an
    unchecked quantity."""
    derived = vjp(kernel, wrt=wrt)
    got = evaluate(derived, {**env, f"bar_{sink}": 1.0})
    return {n: float(got[f"bar_{n}"]) for n in wrt}


def _hessian_vector_reference(kernel, env: dict, wrt: tuple, sink: str,
                              tangent: dict) -> dict:
    """``H v`` by a central difference of the gradient along ``v``."""
    up = _gradient(kernel, {n: env[n] + H * tangent[n] for n in env}, wrt, sink)
    down = _gradient(kernel, {n: env[n] - H * tangent[n] for n in env}, wrt, sink)
    return {n: (up[n] - down[n]) / (2 * H) for n in wrt}


@pytest.mark.parametrize("case", SECOND_ORDER_CASES,
                         ids=[c[0] for c in SECOND_ORDER_CASES])
def test_forward_over_reverse_matches_a_difference_of_the_gradient(case):
    """``jvp(vjp(k))``: seeding the reverse pass with ``bar_y = 1`` and the
    forward pass with the tangent ``v`` makes ``dot_bar_<input>`` the
    Hessian-vector product, which must reproduce a central difference of the
    analytic gradient, for both accumulator and recurrence subjects."""
    _id, kernel, env, wrt, tangent = case
    sink = kernel.sinks[0].name
    want = _hessian_vector_reference(kernel, env, wrt, sink, tangent)
    derived = jvp(vjp(kernel, wrt=wrt), wrt=wrt)
    got = evaluate(derived, {**env, f"bar_{sink}": 1.0,
                             **{f"dot_{n}": tangent[n] for n in wrt}})
    for held in wrt:
        assert float(got[f"dot_bar_{held}"]) == pytest.approx(
            want[held], rel=1e-6, abs=1e-7), (
            f"forward-over-reverse disagrees with a difference of the gradient "
            f"for {held}: {got[f'dot_bar_{held}']} vs {want[held]}")


@pytest.mark.parametrize("case", SECOND_ORDER_CASES,
                         ids=[c[0] for c in SECOND_ORDER_CASES])
def test_reverse_over_forward_matches_a_difference_of_the_gradient(case):
    """``vjp(jvp(k))`` — the same second-order object from the other side.

    The JVP's own output ``dot_y`` is ``grad(y) . v``; its reverse w.r.t. the
    PRIMAL inputs is therefore ``H v`` again, and its reverse w.r.t. the TANGENT
    inputs is the plain gradient — so this row checks both halves and, in doing
    so, checks that the two second-order routes agree with each other through a
    quantity neither of them computed.

    RED, and not by refusal: both recurrences RAN here and returned the wrong
    number (2.108e+01 and 1.467e+00 against this same reference) because the
    reverse of a recurrence pulled back only its SEEDED slots."""
    _id, kernel, env, wrt, tangent = case
    sink = kernel.sinks[0].name
    want = _hessian_vector_reference(kernel, env, wrt, sink, tangent)
    plain = _gradient(kernel, env, wrt, sink)
    derived = vjp(jvp(kernel, wrt=wrt))
    got = evaluate(derived, {**env, **{f"dot_{n}": tangent[n] for n in wrt},
                             f"bar_dot_{sink}": 1.0})
    for held in wrt:
        assert float(got[f"bar_{held}"]) == pytest.approx(
            want[held], rel=1e-6, abs=1e-7), (
            f"reverse-over-forward disagrees with a difference of the gradient "
            f"for {held}: {got[f'bar_{held}']} vs {want[held]}")
        assert float(got[f"bar_dot_{held}"]) == pytest.approx(
            plain[held], rel=1e-9, abs=1e-12), (
            "the reverse of a JVP w.r.t. its TANGENT input is the plain "
            f"gradient; for {held} it is not")


def test_the_two_second_order_routes_agree_with_each_other():
    """Forward-over-reverse and reverse-over-forward compute the same object by
    different machinery — one augments a loop, the other tapes one — so a
    disagreement between them is a defect in exactly one of the two, and the
    finite-difference rows above would not always say which. Cheap, and it is
    the only row here whose reference is analytic on both sides."""
    for _id, kernel, env, wrt, tangent in SECOND_ORDER_CASES:
        sink = kernel.sinks[0].name
        fwd_rev = evaluate(jvp(vjp(kernel, wrt=wrt), wrt=wrt),
                           {**env, f"bar_{sink}": 1.0,
                            **{f"dot_{n}": tangent[n] for n in wrt}})
        rev_fwd = evaluate(vjp(jvp(kernel, wrt=wrt)),
                           {**env, **{f"dot_{n}": tangent[n] for n in wrt},
                            f"bar_dot_{sink}": 1.0})
        for held in wrt:
            assert float(fwd_rev[f"dot_bar_{held}"]) == pytest.approx(
                float(rev_fwd[f"bar_{held}"]), rel=1e-12), (_id, held)


# --------------------------------------------------------------------------- #
# The emitted shape.
# --------------------------------------------------------------------------- #
def test_the_tape_becomes_a_tape_of_pairs():
    """(b) at second order, read off the generated body.

    Forward-over-reverse of a RECURRENCE emits three loops — the primal (taped),
    its AUGMENTED twin (which tapes the tangent slot), and the reverse loop that
    reads a row of each per iteration. That is what "the tape becomes a tape of
    pairs" means concretely, and it is the whole reason the tangent of a tape
    read needs no new storage mechanism: the augmented loop's own tape already
    holds the tangent, at the same iteration, in the twin slot."""
    derived = jvp(vjp(chebyshev, wrt=("z",)), wrt=("z",))
    walk = canonical(derived)
    text = render_body(derived, walk).text
    assert text.count("for (") == 3, text
    tapes = sorted({ln.strip().split()[1].split("[")[0]
                    for ln in text.splitlines() if "hawk_tp" in ln
                    and ln.strip().startswith("Real hawk_tp")})
    assert len(tapes) == 2, (tapes, text)
    # the reverse loop reads BOTH tapes, at its own (backwards) index
    reverse = text.split("for (")[-1]
    for tape in tapes:
        assert f"{tape}[hawk_k" in reverse, (tape, reverse)
    assert "> -1; --" in text, "the reverse loop must run the range backwards"


def test_both_backends_wrap_the_same_second_order_body():
    """ at second order: ONE renderer produces the body and a BACKEND
    supplies only the wrapper, so nothing about a tape of pairs is a host
    construct the device path re-derives."""
    derived = jvp(vjp(saturating, wrt=("x", "g")), wrt=("x", "g"))
    walk = canonical(derived)
    host = render_source("hvp", derived, walk, BACKENDS["host"])
    device = render_source("hvp", derived, walk, CUDA)
    assert host.body == device.body
    assert host.body.count("for (") == 3


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_a_second_order_loop_kernel_compiles(backend, tmp_path):
    """The emitted second-order TU is real C++ / CUDA, checked by the compiler
    rather than by eye — a body full of correctly-named tape arrays that does
    not compile certifies nothing. Never skipped: an unresolvable toolchain
    FAILS and names the variable that would fix it (``tests/_toolchain.py``)."""
    derived = jvp(vjp(saturating, wrt=("x", "g")), wrt=("x", "g"))
    walk = canonical(derived)
    src = render_source("hvp", derived, walk, BACKENDS[backend], mode=FLOAT64,
                        exports=exports(backend))
    path = pathlib.Path(tmp_path) / ("hvp.cpp" if backend == "host" else "hvp.cu")
    path.write_text(src.text)
    rc, err, argv = TC.compile_check(path, backend)
    assert rc == 0, f"$ {' '.join(argv)}\n{err[-4000:]}"


# --------------------------------------------------------------------------- #
# The BUGFIX, pinned structurally as well as numerically.
# --------------------------------------------------------------------------- #
def test_an_unseeded_carry_still_gets_its_adjoint_recurrence():
    """The structural half of the ``_reverse_recurrence`` bugfix, read off the
    emitted body rather than off a number.

    ``vjp(jvp(chebyshev))`` reverses an augmented loop with four carried slots
    whose seeds cover only the two TANGENT ones, so the two PRIMAL slots arrive
    unseeded. Their adjoints are nevertheless live recurrences — the tangent
    body results read the primal carries — so the derived loop must both READ
    those carried locals inside its body and update them with real expressions.

    With the pre-fix pullback the same body came out as::

        const Real hawk_n10_1 = static_cast<Real>(0.0);

    with ``hawk_v10_0_bar_cur`` and ``hawk_v10_1_bar_prev`` never read at all —
    two carried locals written every iteration and consulted by nothing, which
    is what a dropped adjoint recurrence looks like in generated code."""
    derived = vjp(jvp(chebyshev, wrt=("z",)))
    text = render_body(derived, canonical(derived)).text
    reverse = text.split("for (")[-1]
    results = [ln.strip() for ln in reverse.splitlines()
               if ln.strip().startswith("const Real hawk_n")]
    assert len(results) == 6, text
    zeros = [ln for ln in results if ln.endswith("static_cast<Real>(0.0);")]
    assert not zeros, (
        "a carried slot of the reverse loop is updated with a literal zero: its "
        f"adjoint recurrence was dropped.\n{text}")
    # the reverse loop's OWN carried locals: every emitted loop identifier
    # carries its loop's canonical position (`hawk_v<pos>_…`, `hawk_k<pos>_…`),
    # so the header just split off names the prefix without guessing.
    pos = reverse.split("Int hawk_k")[1].split("_")[0]
    carried = sorted({ln.strip().split()[1] for ln in text.splitlines()
                      if ln.strip().startswith(f"Real hawk_v{pos}_")})
    assert len(carried) == 6, carried
    for local in carried:
        stores = sum(1 for ln in reverse.splitlines()
                     if ln.strip().startswith(f"{local} = "))
        assert reverse.count(local) > stores, (
            f"the carried adjoint {local} is written every iteration and read "
            f"by nothing: its recurrence was dropped.\n{text}")


# --------------------------------------------------------------------------- #
# What stays refused.
# --------------------------------------------------------------------------- #
def test_reverse_over_reverse_is_refused_and_names_the_route_that_works():
    """The one direction that is genuinely not built. The adjoint of a tape read
    is a scatter INTO the tape, and rather than guess at it the refusal names
    ``jvp(vjp(kernel))`` — which this file certifies against a finite
    difference — as the way to the same number."""
    with pytest.raises(HawkError, match="reverse-over-reverse") as excinfo:
        vjp(vjp(chebyshev, wrt=("z",)), wrt=("z",))
    assert "jvp(vjp(kernel))" in str(excinfo.value), str(excinfo.value)


def test_differentiating_a_loop_that_COMMITS_is_still_refused_in_both_directions():
    """A derived loop that scatters once per iteration is a derivative's WRITE,
    not its value; differentiating it again is not the second derivative of
    anything the author asked for, and stays refused in both directions."""
    derived = vjp(gathering)
    assert len(derived) == 1 and isinstance(derived[0], Loop)
    with pytest.raises(HawkError, match="second-order"):
        vjp(derived)
    with pytest.raises(HawkError, match="second-order"):
        jvp(derived)


def test_a_nested_loop_is_still_refused_by_the_derivative():
    """The other standing refusal: reverse mode over a nested ``for`` would have
    to keep the inner loop's carried history for every outer iteration — a tape
    of a tape, which is not what a tape of PAIRS is and is not built."""
    @hawk.kernel
    def grid(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for r in range(3):
            inner = 0.0
            for _c in range(2):
                inner = inner + x * r
            total = total + inner
        out = total

    with pytest.raises(HawkError, match="nested `for`"):
        jvp(vjp(grid))
