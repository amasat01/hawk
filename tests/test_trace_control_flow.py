# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``ast`` pass-30): the accepted control-flow forms become
``select`` nodes, and everything else is refused by name and line.

 permits ``if`` / ``elif`` / ``else``, a bounded ``for`` and an early
``return``, "a validated, default-deny surface". The first group asserts the
accepted forms are traced CORRECTLY -- the merged value is evaluated on both
sides, so a rewrite that silently kept one branch would be caught by the number,
not by the node count. The second group asserts the refusals: every one names
the construct and carries the author's own line, since a default-deny surface
whose message does not locate the offending line is a surface authors cannot
use.
"""

from __future__ import annotations

import linecache

import pytest
from _eval import evaluate

from hawk import Mutable, Param, Scalar, Value, Vector, kernel
from hawk.ir import HawkError, Select
from hawk.math import norm, select


def _traced(source: str, name: str = "k", **extra):
    """Trace a kernel written as source text (so a refusal's line is checkable).

    The source is registered with ``linecache`` under a synthetic filename so the
    tracer's own ``inspect.getsourcelines`` can read it back, which is exactly
    what the pass does for a real file."""
    filename = "<authored>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict = {"kernel": kernel, "Scalar": Scalar, "Param": Param,
                   "Mutable": Mutable, "Vector": Vector, "norm": norm,
                   "select": select, **extra}
    exec(compile(source, filename, "exec"), scope)
    return scope[name]


@kernel
def branchy(x: Scalar, limit: Param, out: Mutable[Scalar]):
    y = x * 2.0
    if x > limit:
        y = x * 3.0
        z = y + 1.0
    else:
        z = y - 1.0
    out = y + z


def test_if_else_merges_both_branches_into_select_and_computes_both_sides():
    assert any(isinstance(n, Select) for n in branchy.walk.order)
    for x, limit, expect in ((5.0, 1.0, 5 * 3.0 + (5 * 3.0 + 1.0)),
                             (0.5, 1.0, 0.5 * 2.0 + (0.5 * 2.0 - 1.0))):
        got = evaluate(branchy.sinks, {"x": x, "limit": limit})["out"]
        assert float(got) == pytest.approx(expect)


def test_a_bounded_for_lowers_to_a_loop_that_computes_the_same_number():
    """The body is traced ONCE against a symbolic index and the IR carries
    a :class:`~hawk.ir.loop_nodes.Loop`, not the unrolled chain the pass used to
    build by executing the Python loop. The NUMBER is the gate — a lowering that
    dropped an iteration, or ran the body once, would pass a node-count check
    and fail here."""
    @kernel
    def rolled(x: Scalar, out: Mutable[Scalar]):
        acc = x
        for _ in range(3):
            acc = acc * x
        out = acc

    from hawk.ir import Loop
    assert sum(1 for n in rolled.walk.order if isinstance(n, Loop)) == 1
    assert float(evaluate(rolled.sinks, {"x": 2.0})["out"]) == pytest.approx(2.0 ** 4)


def test_an_early_return_guard_becomes_the_else_branch():
    @kernel
    def guarded(x: Scalar, out: Mutable[Scalar]):
        if x > 0.0:
            out = x
            return
        out = x * -1.0

    assert float(evaluate(guarded.sinks, {"x": 3.0})["out"]) == pytest.approx(3.0)
    assert float(evaluate(guarded.sinks, {"x": -3.0})["out"]) == pytest.approx(3.0)


def test_a_conditional_expression_is_a_select():
    @kernel
    def ternary(x: Scalar, out: Mutable[Scalar]):
        out = (x * 2.0) if x > 1.0 else (x * 5.0)

    assert float(evaluate(ternary.sinks, {"x": 4.0})["out"]) == pytest.approx(8.0)
    assert float(evaluate(ternary.sinks, {"x": 0.5})["out"]) == pytest.approx(2.5)


def test_nested_and_elif_branches_merge():
    @kernel
    def nested(x: Scalar, out: Mutable[Scalar]):
        y = x
        if x > 2.0:
            y = x * 10.0
        elif x > 1.0:
            y = x * 100.0
        else:
            if x > 0.0:
                y = x * 1000.0
        out = y

    for x, expect in ((3.0, 30.0), (1.5, 150.0), (0.5, 500.0), (-1.0, -1.0)):
        assert float(evaluate(nested.sinks, {"x": x})["out"]) == pytest.approx(expect)


REFUSALS = [
    ("while", "    while x > 0.0:\n        x = x - 1.0\n", "While"),
    ("try", "    try:\n        y = x\n    except Exception:\n        y = x\n", "Try"),
    ("with", "    with x:\n        y = x\n", "With"),
    ("import", "    import math\n    y = x\n", "Import"),
    ("assert", "    assert x\n    y = x\n", "Assert"),
    ("lambda", "    f = lambda a: a\n    y = x\n", "lambda"),
    ("comprehension", "    y = [a for a in range(3)]\n", "comprehension"),
    ("boolop", "    if x > 0.0 and x < 1.0:\n        y = x\n    else:\n        y = x\n",
     "land / lor"),
    ("chained", "    if 0.0 < x < 1.0:\n        y = x\n    else:\n        y = x\n",
     "chained comparison"),
    ("membership", "    if x in x:\n        y = x\n    else:\n        y = x\n",
     "identity / membership"),
    ("walrus", "    y = (z := x)\n", "walrus"),
    # widened the bounded-`for` rule: the trip count must RESOLVE to a
    # compile-time int, which is not the same as being spelled as a literal —
    # every real kernel factory writes `range(rows)` with `rows` a local of the
    # function that builds the declaration. So the
    # refusal moved from "not a literal" to "not an int", asked of the VALUE at
    # trace time; these three cases keep both halves of it under test.
    ("runtime range bound", "    for i in range(x):\n        y = x\n",
     "compile-time extent"),
    ("range over a tuple target",
     "    for i, j in range(3):\n        y = x\n", "single name"),
    ("range by keyword", "    for i in range(stop=3):\n        y = x\n",
     "single name"),
    ("value return", "    return x\n", "returns nothing"),
    ("reserved prefix", "    __hawk_c0 = x\n    y = x\n", "reserved"),
]


@pytest.mark.parametrize("case", REFUSALS, ids=[c[0] for c in REFUSALS])
def test_the_default_deny_surface_refuses_by_name_and_line(case):
    label, body, needle = case
    source = ("@kernel\ndef k(x: Scalar, out: Mutable[Scalar]):\n" + body
              + "    out = x\n")
    with pytest.raises(HawkError) as excinfo:
        _traced(source)
    message = str(excinfo.value)
    assert needle in message, message
    assert "<authored>:" in message, f"a refusal must locate the line: {message}"


def test_not_is_refused_with_its_own_line():
    source = ("@kernel\ndef k(x: Scalar, out: Mutable[Scalar]):\n"
              "    if not x:\n        y = x\n    else:\n        y = x\n"
              "    out = y\n")
    with pytest.raises(HawkError, match="lnot"):
        _traced(source)


def test_a_name_assigned_in_only_one_branch_must_be_bound_before_the_if():
    source = ("@kernel\ndef k(x: Scalar, out: Mutable[Scalar]):\n"
              "    if x > 0.0:\n        y = x\n"
              "    out = x\n")
    with pytest.raises(HawkError, match="assigned in only one branch"):
        _traced(source)


def test_a_sink_commit_inside_a_branch_is_refused():
    source = ("@kernel\ndef k(x: Scalar, acc: Accum, out: Mutable[Scalar]):\n"
              "    if x > 0.0:\n        acc.add(x)\n    else:\n        y = x\n"
              "    out = x\n")
    from hawk import Accum
    with pytest.raises(HawkError, match="inside a branch"):
        _traced(source, Accum=Accum[Scalar])


def test_a_traced_value_has_no_truth_value_outside_the_pass():
    @kernel
    def k(x: Scalar, out: Mutable[Scalar]):
        out = x

    with pytest.raises(HawkError, match="no truth value"):
        bool(Value(k.sinks[0].value))


def test_an_undeclared_parameter_and_an_uncommitted_sink_are_refused():
    with pytest.raises(HawkError, match="carry no declaration"):
        _traced("@kernel\ndef k(x, out: Mutable[Scalar]):\n    out = x\n")
    with pytest.raises(HawkError, match="never committed"):
        _traced("@kernel\ndef k(x: Scalar, out: Mutable[Scalar]):\n    y = x\n")
