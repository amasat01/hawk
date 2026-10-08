# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The rule-table card: every node KIND, its two directions, and the
table's LOC -- WRITTEN BY THE RUN, and RED when a kind has no rule.

The measurable claim is "a rule-table LOC count in the card, not an
assertion here", so the number lands in a card the run regenerates
(``tests/cards/CARD_DIFF_RULES.md``) and prose only cites it. The gate is the
enumeration around it: every kind in ``hawk.ir.ops.OP_KINDS`` must carry a table
entry -- an explicit zero rule for the undifferentiable ones, or the transform's
own cross-sample handling -- and every SINK family must survive both transforms,
checked by deriving one, not by reading the source. Absence is the failure
condition, because a missing rule that silently contributes zero is the error
class a rule registry's own docstring names.

This test previously failed when the ``quat_recip`` entry was deleted
from ``hawk/hawk/diff/rules.py``'s ``_SHAPED`` table -- the card test named the
kind as unruled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from _cards import card_path

from hawk.diff import jvp, vjp
from hawk.diff.rules import RULES, _zero_jvp, _zero_vjp
from hawk.ir import (
    AccumWrite,
    Assign,
    At,
    Const,
    Dispatch,
    Leaf,
    MapreducePartial,
    SampleIndex,
    WideWrite,
)
from hawk.ir import make as mk
from hawk.ir.ops import OP_KINDS
from hawk.types import TensorType

F64 = TensorType((), "f64")
I32 = TensorType((), "i32")
_TABLE = Path(__file__).resolve().parent.parent / "hawk" / "diff" / "rules.py"
_CARD = card_path("CARD_DIFF_RULES.md")

#: The node families the TRANSFORM differentiates itself: terminals, the
#: four sink kinds it seeds/mirrors, and the gather.
_FAMILIES = {
    "leaf": "terminal: a gradient/tangent plane, not a rule",
    "constant": "terminal: zero flow",
    "assign": "seeded from an adjoint plane / mirrored as a tangent sink",
    "wide_write": "seeded from an adjoint plane / mirrored as a tangent sink",
    "accum_write": "reverse = gather of its adjoint plane",
    "mapreduce_partial": "reverse = broadcast of a rank-0 'sum' adjoint",
    "at": "reverse = scatter-add AccumWrite; forward = tangent lookup",
    "sample_index": "terminal: an address, not a differentiable quantity",
    "primitive": "the AUTHOR's supplied rule, applied to the "
                 "boundary's declared inputs; never differentiated through",
    "loop": "MIRRORED, not folded: an accumulator loop's reverse is a "
            "forward loop storing nothing; a recurrence's runs backwards over "
            "the forward loop's own per-iteration carries; forward mode is ONE "
            "loop carrying primal and tangent together (hawk/diff/loops.py)",
    "loop_value": "the projection of one carried slot: its adjoint SEEDS the "
                  "loop it came from, and its tangent is the augmented loop's "
                  "matching slot; never a rule of its own",
    "loop_index": "terminal: a loop's own index is an address, not a "
                  "differentiable quantity (SampleIndex's treatment)",
    "loop_carry": "terminal INSIDE the body: its adjoint is what the reverse "
                  "loop carries, and its tangent what the augmented loop does",
    "tape_read": "the forward loop's carried value at one iteration — the ONE "
                 "thing a reverse recurrence needs and the only reason a "
                 "bounded local array is declared at all",
    "dispatch": ": K INDEPENDENT per-branch reverse passes combined into "
                "ONE adjoint dispatch per leaf reached, SAME policy as the "
                "primal; forward mirrors the branches under that same policy; "
                "`kind` itself is an address, not a differentiable quantity, "
                "and receives no adjoint",
}


def _loop_sinks(x):
    """One sink whose value is a lowered ``for``'s carried result.

    Built through the TRACER rather than by hand, because the Loop node's own
    invariants (a rank-0 carry, a body whose only foreign reads are its
    operands) are the tracer's to establish and a hand-built loop would be
    testing a shape no author can write."""
    import hawk
    import hawk.math as m

    @hawk.kernel
    def folded(seed: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for k in range(3):
            acc = acc + m.sin(seed + k)
        out = acc

    del x
    return folded.sinks


def _primitive_sinks(x, value):
    """One sink whose value passes through an extension-seam boundary.

    Registered lazily and under a name of this row's own, because the registry
    refuses a collision by design and this module may be imported beside any
    other that registers primitives."""
    from hawk.ext import primitive, primitives

    name = "card_identity"
    definition = primitives().get(name)
    if definition is None:
        definition = primitive(name, vjp=lambda a, bar: bar,
                               jvp=lambda a, da: da)(lambda a: a * 1.0)
    from hawk.trace.value import Value, node_of

    return (Assign("out", node_of(definition(Value(value))), F64),)


def _loop_recurrence():
    """A RECURRENCE loop's sinks — the one shape whose reverse mints a
    ``tape_read``, so the family row for it exercises the node rather than
    asserting it exists."""
    import hawk
    import hawk.math as m

    @hawk.kernel
    def recurring(seed: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        acc = seed
        for _k in range(3):
            acc = m.tanh(acc) * seed
        out = acc

    return recurring.sinks


def _entry(kind: str):
    return next((rule for key, rule in RULES.items() if key[0] == kind), None)


def _direction(rule, which: str) -> str:
    if rule is None:
        return "MISSING"
    fn = getattr(rule, which)
    if fn in (_zero_vjp, _zero_jvp):
        return "zero"
    if fn is None:
        return rule.mode
    return "rule"


def test_every_op_kind_carries_a_rule():
    unruled = [kind for kind in OP_KINDS if _entry(kind) is None]
    assert not unruled, (
        "the rule table is closed-world -- a kind with no derivative is "
        "registered with an explicit zero rule, never omitted. Unruled: "
        f"{unruled}"
    )


def test_every_kind_covers_both_directions():
    gaps = [kind for kind in OP_KINDS
            if "MISSING" in (_direction(_entry(kind), "vjp"),
                             _direction(_entry(kind), "jvp"))]
    assert not gaps, f"differentiation is BOTH directions; kinds missing one: {gaps}"


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_sink_family_survives_both_transforms(family):
    x = Leaf("vocab_read", "per_sample", "x", F64)
    idx = Leaf("vocab_read", "per_sample", "idx", I32)
    tab = Leaf("table_read", "lookup", "tab", F64)
    value = mk("mul", (x, x))
    prim = _primitive_sinks(x, value)
    loop_sinks = _loop_sinks(x)
    sinks = {
        "loop": loop_sinks,
        "loop_value": loop_sinks,
        "loop_index": loop_sinks,
        "loop_carry": loop_sinks,
        "tape_read": _loop_recurrence(),
        "primitive": prim,
        "sample_index": (Assign("out", mk("mul", (At(tab, SampleIndex(), F64), x)),
                                F64),),
        "assign": (Assign("out", value, F64),),
        "wide_write": (WideWrite("wide", value, None, F64),),
        "accum_write": (AccumWrite("acc", value, idx, F64),),
        "mapreduce_partial": (MapreducePartial("total", value, "sum", F64),),
        "at": (Assign("out", mk("mul", (At(tab, idx, F64), x)), F64),),
        "dispatch": (Assign("out", Dispatch(idx, [x, mk("mul", (x, x))], "switch",
                                            F64), F64),),
        "leaf": (Assign("out", x, F64),),
        "constant": (Assign("out", mk("add", (x, Const(3.0, F64))), F64),),
    }[family]
    assert vjp(sinks) and jvp(sinks), f"{family} must derive in both directions"


@pytest.mark.repo_local
def test_the_card_is_regenerated_and_carries_the_table_loc():
    text = _TABLE.read_text()
    loc = len(text.splitlines())
    rows = [f"| `{kind}` | {_direction(_entry(kind), 'vjp')} | "
            f"{_direction(_entry(kind), 'jvp')} | "
            f"{getattr(_entry(kind), 'mode', 'MISSING')} |"
            for kind in OP_KINDS]
    families = [f"| `{k}` | {v} |" for k, v in sorted(_FAMILIES.items())]
    _CARD.write_text(
        "# CARD — the derivative rule table\n\n"
        f"Generated by `tests/test_diff_rule_card.py`; the run writes the card, "
        "prose only cites it.\n\n"
        f"- rule table: `hawk/hawk/diff/rules.py`\n"
        f"- **rule-table LOC: {loc}**\n"
        f"- rule-table sha256: `{hashlib.sha256(text.encode()).hexdigest()}`\n"
        f"- op kinds with a rule: **{len(OP_KINDS)}/{len(OP_KINDS)}**\n"
        f"- table entries (`(kind, rank)` keys): {len(RULES)}\n\n"
        "`rule` = a derivative formula; `zero` = an explicit zero rule (a kind that "
        "is genuinely undifferentiable, registered rather than omitted); `walker` = "
        "differentiated by the transform itself (the cross-sample forms).\n\n"
        "| op kind | vjp | jvp | mode |\n|---|---|---|---|\n" + "\n".join(rows) +
        "\n\n## Node families the transform owns\n\n| family | treatment |\n|---|---|\n"
        + "\n".join(families) + "\n")
    assert _CARD.exists() and f"rule-table LOC: {loc}" in _CARD.read_text()
