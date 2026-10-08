# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A bounded ``for`` LOWERS to a real loop — the node, the emission, the
classes, and every refusal, enumerated.

WHY THIS FILE EXISTS AT ALL. HAWK used to lower a bounded ``for`` by
EXECUTING it at trace time: the tracer ran the Python loop and the IR carried
the unrolled chain. That is a transpiler's answer, and it has a hard failure
mode measured in — a 64-edge x 23-basis KAN cell reached ptxas as ONE
function and took 25-27 GB of RSS on a 31 GB box. The fix is deliberately
minimal and goes no further: the code is only lowered to a
CUDA/C++ loop — the compiler then does the next step. This is not a
transpiler; it is a code generator. So the body is traced ONCE
against a symbolic index, the loop-carried values are explicit, and the emitter
writes a plain ``for`` with a literal bound; whether it is unrolled is the C++ /
CUDA compiler's decision, made with register pressure in view — a fact HAWK does
not have.

THE UNROLL-BY-EXECUTION MECHANISM IS GONE. A bounded ``for`` is no longer
unrolled by running the Python loop at trace time; it lowers to a real Loop
node instead. Rows elsewhere that used to depend on execution-time unrolling
are re-owned onto the Loop node, not retired, and this file adds what they
do not reach: the node's own shape, the emission, the two autodiff classes
and the refusals.

ENUMERATION, NOT SAMPLING. A loop is one of exactly two differentiable classes
(``accumulator``, ``recurrence``) or it is REFUSED by name; :data:`LOOP_CASES`
carries one kernel per class and :func:`test_every_loop_class_is_covered` fails
if a class in :data:`hawk.ir.loops.LOOP_CLASSES` has no case. A refusal without
a row is the same hole one class further out, so :data:`REFUSALS` enumerates
those too.
"""

from __future__ import annotations

import linecache

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.diff import jvp, vjp
from hawk.emit import BACKENDS, CUDA, render_body, render_source
from hawk.ir import HawkError, Loop, LoopValue, canonical
from hawk.ir.loops import ACCUMULATOR, LOOP_CLASSES, RECURRENCE, classify_loop

H = 1e-5
ROWS, COLS = 4, 5


# --------------------------------------------------------------------------- #
# The subjects: one kernel per loop CLASS (enumerated, see the module note).
# --------------------------------------------------------------------------- #
@hawk.kernel
def additive(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
    """ACCUMULATOR, one carry — the additive-loop rows' own shape, and the one every
    contraction has: the body only adds into the carry."""
    acc = 0.0
    for k in range(8):
        acc = acc + g * m.sin(x + k)
    y = acc


@hawk.kernel
def two_accumulators(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
    """ACCUMULATOR, two carries — a sum and a sum of squares in one pass, which
    is what makes "no carry may be read outside its own additive slot" a rule
    with something to say rather than a restatement of "one carry"."""
    total = 0.0
    squares = 0.0
    for k in range(6):
        term = g * m.cos(x + k)
        total = total + term
        squares = squares + term * term
    y = total + squares


@hawk.kernel
def chebyshev(z: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """RECURRENCE, two carries — ``T_{n+1} = 2 z T_n - T_{n-1}``.

    The exemplar names, and the reason the recurrence class needs the
    forward loop's per-iteration values at all: the adjoint at iteration ``k``
    reads ``T_k``, which the forward loop has already overwritten."""
    prev = 1.0
    cur = z
    for _n in range(5):
        prev, cur = cur, 2.0 * z * cur - prev
    y = cur + prev


@hawk.kernel
def saturating(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
    """RECURRENCE, one carry — ``acc = tanh(g*acc + x)``, the recurrence benchmark's own
    ``edge64`` body at a size a finite difference can check."""
    acc = 0.0
    for _k in range(6):
        acc = m.tanh(g * acc + x)
    y = acc


@hawk.kernel
def max_fold(t: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Scalar]):
    """RECURRENCE by classification, because a ``max`` fold is not additive —
    the carry is READ by the body's own arithmetic, so its adjoint evolves."""
    best = t.at(row=0)
    for r in range(ROWS):
        best = m.max(best, t.at(row=r))
    y = best


@hawk.kernel
def gathering(t: hawk.Table["row":ROWS, "col":COLS], y: hawk.Mutable[hawk.Scalar]):
    """An ACCUMULATOR over a GATHER: its reverse is a scatter at the row the
    gather read, once per iteration, in the loop's own scope (the transpose,
     scope)."""
    total = 0.0
    for r in range(ROWS):
        total = total + t.at(row=r, col=2) * 1.5
    y = total


#: ``(id, kernel, class, env, wrt)`` — ONE case per loop class, plus the two
#: shapes inside each class that behave differently (a second carry; a fold
#: whose carry is read by arithmetic rather than by an addition).
LOOP_CASES = [
    ("additive", additive, ACCUMULATOR, {"x": 0.7, "g": 1.3}, ("x", "g")),
    ("two_accumulators", two_accumulators, ACCUMULATOR, {"x": 0.4, "g": 0.9},
     ("x", "g")),
    ("chebyshev", chebyshev, RECURRENCE, {"z": 0.6}, ("z",)),
    ("saturating", saturating, RECURRENCE, {"x": 0.35, "g": 0.8}, ("x", "g")),
]


def _loops(kernel) -> list:
    return [n for n in kernel.walk.order if isinstance(n, Loop)]


# --------------------------------------------------------------------------- #
# The node.
# --------------------------------------------------------------------------- #
def test_a_bounded_for_is_ONE_node_whose_size_does_not_track_its_bound():
    """The whole point of, stated as a number: an unrolled ``for`` grows the
    IR with its trip count and a LOWERED one does not. Two kernels whose only
    difference is the bound must have the same node count — under the old
    unroll-by-execution the 64-bound one had eight times the 8-bound one's
    chain, which is what reached ptxas as a 25 GB function."""
    def cell(bound: int):
        @hawk.kernel
        def k(x: hawk.Scalar, g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
            acc = 0.0
            for j in range(bound):
                acc = acc + g * m.sin(x + j)
            y = acc
        return k

    small, large = cell(8), cell(64)
    assert len(small.walk.order) == len(large.walk.order)
    assert len(_loops(small)) == len(_loops(large)) == 1
    assert (_loops(small)[0].trip, _loops(large)[0].trip) == (8, 64)
    assert isinstance(small.sinks[0].value, LoopValue)


def test_the_walk_visits_the_loop_once_and_the_bodys_leaves_are_its_leaves():
    """The walk visits the loop once, and the body's leaves are its leaves.
    The body is a SCOPE, so its nodes are not in the flat order — but
    every plane it reads must still bind a slot, or the entry signature would
    not carry the table the loop reads and the kernel could not be called."""
    assert ("lookup", "t") in gathering.arg_spec
    order = gathering.walk.order
    assert sum(1 for n in order if isinstance(n, Loop)) == 1
    body = _loops(gathering)[0].body_nodes()
    assert body and not any(n in order for n in body), (
        "a loop body's own nodes are the emitter's to render inside the scope; "
        "hoisting them into the flat order would be the unroll again")


def test_the_digest_covers_the_body_and_the_bound():
    """The walk digest is a content hash, so two bodies that differ
    only INSIDE the loop must not hash alike — and a second trace of the same
    source must hash identically (no object counter may leak into the key)."""
    def cell(bound: int, scale: float):
        @hawk.kernel
        def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            acc = 0.0
            for j in range(bound):
                acc = acc + scale * m.sin(x + j)
            y = acc
        return k

    assert cell(8, 1.0).walk.digest == cell(8, 1.0).walk.digest
    assert cell(8, 1.0).walk.digest != cell(9, 1.0).walk.digest
    assert cell(8, 1.0).walk.digest != cell(8, 2.0).walk.digest


def test_a_gather_inside_a_loop_is_still_cross_sample_read():
    """ soundness direction. The classifier must descend into the
    scope: a kernel declared ``sample_local`` may be placed under a
    ``RankPartition`` that splits the plane it reads, which is a wrong answer
    rather than a placement cost."""
    assert gathering.walk.access.cls == "cross_sample_read"


# --------------------------------------------------------------------------- #
# The emission ( whole deliverable).
# --------------------------------------------------------------------------- #
def test_the_body_is_a_plain_for_with_a_literal_bound_and_no_pragma():
    """The emission's negative half: no unroll policy of HAWK's own. Every trace-time constant reaches
    the TU as a LITERAL — the bound, the declared stride — so the compiler can
    see the trip count; and there is no ``#pragma unroll``, no threshold and no
    policy, because whether to unroll is the compiler's decision."""
    text = render_body(gathering.sinks, gathering.walk).text
    assert text.count("for (") == 1, text
    assert f"< {ROWS};" in text, text
    assert "pragma" not in text.lower(), text
    assert "unroll" not in text.lower(), text
    # the declared row stride is a literal in the subscript, not a lookup
    assert f"* static_cast<Int>({COLS})" in text, text


def test_both_backends_wrap_the_same_body_string():
    """ for a lowered loop: ONE renderer produces the body and a
    BACKEND supplies only the wrapper, so the loop is not a host construct that
    the device path re-derives."""
    host = render_source("g", gathering.sinks, gathering.walk, BACKENDS["host"])
    device = render_source("g", gathering.sinks, gathering.walk, CUDA)
    assert host.body == device.body
    assert "for (" in host.body


def test_the_carried_locals_are_updated_simultaneously():
    """A two-carry recurrence swaps its carries (``prev, cur = cur, …``). If the
    emitted body assigned each carry as its result became available, the second
    expression would read the FIRST's new value — a different recurrence and a
    silently wrong number. The generated code must name both results before
    assigning either."""
    text = render_body(chebyshev.sinks, chebyshev.walk).text
    lines = [ln.strip() for ln in text.splitlines()]
    results = [i for i, ln in enumerate(lines) if ln.startswith("const Real hawk_n")]
    stores = [i for i, ln in enumerate(lines)
              if ln.startswith("hawk_v") and " = hawk_n" in ln]
    assert len(results) == len(stores) == 2, text
    assert max(results) < min(stores), text


def test_a_zero_trip_for_collapses_to_the_values_it_started_with():
    """``range(0)`` runs the body no times. Emitting a loop no iteration of which
    runs would also size a reverse pass's tape at zero, which C++ cannot
    declare; collapsing is the same answer Python's own ``for`` gives."""
    @hawk.kernel
    def never(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for _k in range(0):
            acc = acc * 2.0
        y = acc

    assert not _loops(never)
    assert float(evaluate(never.sinks, {"x": 3.0})["y"]) == pytest.approx(3.0)


def test_a_nested_for_emits_a_nested_scope():
    """Nesting is structural: the body renderer is the same renderer, so an
    inner loop is a scope inside a scope and neither is unrolled."""
    @hawk.kernel
    def grid(t: hawk.Table["row":ROWS, "col":COLS], y: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for r in range(ROWS):
            row = 0.0
            for c in range(COLS):
                row = row + t.at(row=r, col=c)
            total = total + row * row
        y = total

    text = render_body(grid.sinks, grid.walk).text
    assert text.count("for (") == 2, text
    table = np.arange(ROWS * COLS, dtype=float) * 0.25 + 1.0
    got = float(evaluate(grid.sinks, {"t": table})["y"])
    want = sum(sum(table[r * COLS + c] for c in range(COLS)) ** 2
               for r in range(ROWS))
    assert got == pytest.approx(want)


# --------------------------------------------------------------------------- #
# The classes and their derivatives.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", LOOP_CASES, ids=[c[0] for c in LOOP_CASES])
def test_the_classifier_names_the_class_from_the_bodys_structure(case):
    _id, kernel, expected, _env, _wrt = case
    loop = _loops(kernel)[0]
    assert classify_loop(loop).name == expected


def test_every_loop_class_is_covered():
    """Enumeration, not sampling: a class with no case is a class nothing
    checks, which is the hole this whole file exists to close."""
    covered = {expected for _id, _k, expected, _e, _w in LOOP_CASES}
    assert covered == set(LOOP_CLASSES), (
        f"loop classes with no case: {set(LOOP_CLASSES) - covered}")


def test_an_accumulator_reverse_stores_nothing_and_a_recurrence_taped_body_does():
    """The classes differ in ONE observable, and it is the one that matters for
    a device kernel's local memory: the accumulator's adjoint never reads the
    carried value, so its emitted reverse declares no array; the recurrence's
    does, sized by the (compile-time) trip count."""
    acc_text = render_body(*_derived(additive))
    rec_text = render_body(*_derived(saturating))
    assert "hawk_tp" not in acc_text.text, acc_text.text
    assert "hawk_tp" in rec_text.text, rec_text.text
    assert "[6];" in rec_text.text, rec_text.text


def _derived(kernel):
    sinks = vjp(kernel)
    return sinks, canonical(sinks)


@pytest.mark.parametrize("case", LOOP_CASES, ids=[c[0] for c in LOOP_CASES])
def test_the_reverse_of_a_loop_matches_a_central_difference(case):
    """The numeric gate. A mirrored loop that walks, hashes and emits perfectly
    can still compute the wrong number — a reversed range off by one, an adjoint
    that forgot the carry — so the rules are EVALUATED against the primal, not
    inspected."""
    _id, kernel, _cls, env, wrt = case
    sink = kernel.sinks[0].name
    derived = vjp(kernel, wrt=wrt)
    got = evaluate(derived, {**env, f"bar_{sink}": 1.0})
    for held in wrt:
        want = _central(kernel, sink, env, held)
        assert float(got[f"bar_{held}"]) == pytest.approx(want, rel=1e-6), held


@pytest.mark.parametrize("case", LOOP_CASES, ids=[c[0] for c in LOOP_CASES])
def test_the_forward_of_a_loop_matches_a_central_difference(case):
    """Forward mode carries the tangent alongside the primal in ONE loop, so a
    directional derivative along each input in turn must reproduce the same
    central difference the reverse row checks."""
    _id, kernel, _cls, env, wrt = case
    sink = kernel.sinks[0].name
    derived = jvp(kernel, wrt=wrt)
    for held in wrt:
        seeds = {f"dot_{n}": (1.0 if n == held else 0.0) for n in wrt}
        got = evaluate(derived, {**env, **seeds})
        want = _central(kernel, sink, env, held)
        assert float(got[f"dot_{sink}"]) == pytest.approx(want, rel=1e-6), held


def _central(kernel, sink: str, env: dict, held: str) -> float:
    up = evaluate(kernel.sinks, {**env, held: env[held] + H})[sink]
    down = evaluate(kernel.sinks, {**env, held: env[held] - H})[sink]
    return float(up - down) / (2 * H)


def test_a_gather_loops_reverse_scatters_once_per_iteration_in_the_same_scope():
    """The transpose inside scope. The reverse of ``t.at(e(k))`` summed
    over ``k`` is ``bar_t[e(k)] += g`` — once per iteration, at that iteration's
    own row. There is no way to say that with a sink at the top level, so the
    derived loop COMMITS, and the walk admits it as a statement root."""
    derived = vjp(gathering)
    assert len(derived) == 1 and isinstance(derived[0], Loop)
    assert [s.name for s in derived[0].body_sinks] == ["bar_t"]
    walk = canonical(derived)
    assert ("accum_out", "bar_t") in walk.arg_spec
    assert walk.access.cls == "cross_sample_write"
    table = np.arange(ROWS * COLS, dtype=float) * 0.5 + 1.0
    got = evaluate(derived, {"t": table, "bar_y": 1.0}, lanes=ROWS * COLS)["bar_t"]
    want = np.zeros(ROWS * COLS)
    for r in range(ROWS):
        want[r * COLS + 2] += 1.5
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)
    text = render_body(derived, walk).text
    assert text.count("for (") == 1 and "hawk_abi::accum_add(aout_bar_t, " in text, text


def test_the_reverse_of_a_max_fold_is_the_branch_the_fold_took():
    """A ``max`` fold classifies as a recurrence — its carry is read by
    arithmetic, not by an addition — and its adjoint must land on the row that
    actually won, which is what the taped carry is for."""
    derived = vjp(max_fold)
    table = np.array([1.0, 7.0, 3.0, 2.0])
    got = evaluate(derived, {"t": table, "bar_y": 1.0}, lanes=ROWS)["bar_t"]
    want = np.zeros(ROWS)
    want[1] = 1.0
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)


# --------------------------------------------------------------------------- #
# The refusals — enumerated, each naming its construct.
# --------------------------------------------------------------------------- #
def _traced(source: str, name: str = "k", **extra):
    filename = "<authored>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict = {"kernel": hawk.kernel, "Scalar": hawk.Scalar, "Param": hawk.Param,
                   "Mutable": hawk.Mutable, "Vector": hawk.Vector, "Accum": hawk.Accum,
                   "Table": hawk.Table, "m": m, **extra}
    exec(compile(source, filename, "exec"), scope)
    return scope[name]


REFUSALS = [
    ("a loop-local name read after the loop",
     "    acc = 0.0\n    for j in range(3):\n        term = acc + 1.0\n"
     "        acc = term\n    out = term\n",
     "loop-LOCAL"),
    ("the loop variable read after the loop",
     "    acc = 0.0\n    for j in range(3):\n        acc = acc + 1.0\n"
     "    out = acc + j\n",
     "loop VARIABLE"),
    ("a sink commit inside the loop",
     "    acc = 0.0\n    for j in range(3):\n        a.add(acc)\n"
     "        acc = acc + 1.0\n    out = acc\n",
     "may not sit inside a `for`"),
    ("a loop that carries nothing",
     "    for j in range(3):\n        z = x + 1.0\n    out = x\n",
     "computes nothing observable"),
    ("a runtime trip count",
     "    acc = 0.0\n    for j in range(3):\n        acc = acc + x\n"
     "    out = acc\n",
     None),
]


@pytest.mark.parametrize("case", REFUSALS[:-1], ids=[c[0] for c in REFUSALS[:-1]])
def test_every_loop_refusal_names_its_construct(case):
    _label, body, needle = case
    source = ("@kernel\ndef k(x: Scalar, a: Accum, out: Mutable[Scalar]):\n"
              + body)
    with pytest.raises(HawkError) as excinfo:
        _traced(source, Accum=hawk.Accum[hawk.Scalar])
    assert needle in str(excinfo.value), str(excinfo.value)


def test_a_mutable_committed_only_inside_the_loop_is_carried_not_refused():
    """A Mutable's own name is an ordinary local now, so writing it only
    inside a `for` no longer needs a separate local seeded before the loop
    (the refusal this used to be a row of, above): the loop rewrite sees the
    SAME shape it always has for any other carried local, since a read of
    `out` before the loop (its launch-start value) is seeded automatically
    the first time control flow touches its name. The committed value is
    the LAST iteration's -- checked end to end against a numpy reference in
    this file's own host/device oracle rows."""
    @hawk.kernel
    def k(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        for j in range(3):
            out = x + j
    assert sorted(k.walk.prior_reads) == ["out"]
    assert k.arg_spec == (("mutable", "out"), ("per_sample", "x"))


def test_a_rank_one_carry_is_declared_as_an_aether_Item():
    """A carried slot may be rank>=1 — the emitted local is an ``aether::Item``.

    This row was a REFUSAL when the Loop node landed, and the refusal's reason
    was a real fact about C++ read one step short: a rank>=1 aether value is an
    expression TREE whose type differs between the initial value and the body
    result, so ``const auto`` cannot declare one local for both — and worse, an
    expression node stores its operands BY REFERENCE, so such a local would
    dangle across iterations. The step that was missing is that aether HAS a
    storage type with all-static extents which converts from and assigns from
    any matching expression: ``aether::Item``. Declaring the carry as that type
    is both correct and the only correct choice, and it is what makes a vector
    adjoint accumulated across iterations expressible at all (/ —
    ``tests/test_trace_loop_vector_index.py`` is where that is used in anger).

    The type-AGREEMENT check is untouched and still refuses: one carried slot
    has one type for the whole loop."""
    @hawk.kernel
    def carried(v: hawk.Vector[3], out: hawk.Mutable[hawk.Vector[3]]):
        acc = v
        for _k in range(3):
            acc = acc + v
        out = acc

    text = render_body(carried.sinks, carried.walk).text
    assert "aether::Item<Real, 3> hawk_v" in text, text
    got = evaluate(carried.sinks, {"v": np.array([1.0, 2.0, 3.0])})["out"]
    np.testing.assert_allclose(got, np.array([4.0, 8.0, 12.0]))


def test_a_nested_loop_is_refused_by_the_DERIVATIVE_and_not_by_the_tracer():
    """Nesting emits fine (see the emission row); what is refused is
    DIFFERENTIATING it, because reverse mode would have to keep the inner
    loop's carried history for every outer iteration — a tape of a tape."""
    @hawk.kernel
    def grid(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for r in range(3):
            inner = 0.0
            for _c in range(2):
                inner = inner + x * r
            total = total + inner
        out = total

    render_body(grid.sinks, grid.walk)          # emission is fine
    with pytest.raises(HawkError, match="nested `for`"):
        vjp(grid)


def test_second_order_through_a_loop_is_refused_rather_than_guessed():
    """A derived loop COMMITS once per iteration, which no primal does.
    Differentiating it again would be differentiating a derivative's scatter;
    refuse and name the way round it."""
    derived = vjp(gathering)
    with pytest.raises(HawkError, match="second-order"):
        vjp(derived)
    with pytest.raises(HawkError, match="second-order"):
        jvp(derived)
