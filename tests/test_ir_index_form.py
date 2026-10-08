# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The index-form classifier: :func:`hawk.ir.access.index_form`.

One analysis answers "how does this index expression move with the sample (or a
loop) index?" for every rule that asks it — the contraction recogniser's
reduce-axis stride today, a clustered-scatter rule later. The rows pin the
ACCEPTED shapes by their exact ``(coefficient, offset)`` and the REJECTED shapes
by ``None``, and one row plants a wrong coefficient into the classifier and
requires the accepted table to notice: a table that passes under a broken
classifier proves nothing.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import hawk
import hawk.math as m
from hawk.ir import access
from hawk.ir.access import IndexForm, index_form
from hawk.ir.loop_nodes import LoopIndex
from hawk.ir.nodes import At, Const, Op, RoleConst, SampleIndex
from hawk.ir.ops import TensorType

I32 = TensorType((), "i32")
F64 = TensorType((), "f64")


def c(value: int) -> Const:
    return Const(value, I32)


def op(kind: str, *operands, ttype: TensorType = I32) -> Op:
    return Op(kind, operands, ttype)


S = SampleIndex()
R = LoopIndex(0, "r")
RUNTIME = Const(1.5, F64)     # a sample-free value that is not a compile-time int


def _accepted():
    """``(label, node, var, coefficient, offset)`` — every accepted shape."""
    third = Fraction(1, 6)
    return [
        ("lane", S, None, 1, 0),
        ("constant", c(7), None, 0, 7),
        ("runtime base", RUNTIME, None, 0, None),
        ("loop index is sample-free", R, None, 0, None),
        ("s*3+2", op("add", op("mul", S, c(3)), c(2)), None, 3, 2),
        ("3*s-1", op("sub", op("mul", c(3), S), c(1)), None, 3, -1),
        ("s-s", op("sub", S, S), None, 0, 0),
        ("s+runtime", op("add", S, RUNTIME), None, 1, None),
        ("(s*2)*5", op("mul", op("mul", S, c(2)), c(5)), None, 10, 0),
        ("s*0", op("mul", S, c(0)), None, 0, 0),
        ("major half s/6", op("div", S, c(6)), None, third, 0),
        ("(s+4)/6", op("div", op("add", S, c(4)), c(6)), None, third,
         Fraction(4, 6)),
        ("row index (s/6)*6", op("mul", op("div", S, c(6)), c(6)), None, third, 0),
        ("tile*k+kc", op("add", op("mul", op("div", S, c(6)), c(4)), R), None,
         third, 0),
        ("q-b", op("sub", op("div", S, c(6)), c(2)), None, third, 0),
        ("q*0", op("mul", op("div", S, c(6)), c(0)), None, 0, 0),
        ("(2s+3)/2 folds exact", op("div", op("add", op("mul", S, c(2)), c(3)),
                                    c(2)), None, 1, 1),
        ("((2s)/2)*3", op("mul", op("div", op("mul", S, c(2)), c(2)), c(3)),
         None, 3, 0),
        ("loop var r*4+s", op("add", op("mul", R, c(4)), S), R, 4, None),
    ]


def _rejected():
    """``(label, node, var, quotients)`` — every shape that must be ``None``."""
    q = op("div", S, c(6))
    return [
        ("s*s", op("mul", S, S), None, True),
        ("s*runtime", op("mul", S, RUNTIME), None, True),
        ("minor half s-(s/6)*6", op("sub", S, op("mul", q, c(6))), None, True),
        ("b-q", op("sub", c(5), q), None, True),
        ("q+q", op("add", q, q), None, True),
        ("nested quotient", op("div", q, c(2)), None, True),
        ("divide by zero", op("div", S, c(0)), None, True),
        ("negative divisor", op("div", S, c(-2)), None, True),
        ("negative coefficient", op("div", op("mul", S, c(-1)), c(2)), None, True),
        ("negative offset", op("div", op("sub", S, c(1)), c(2)), None, True),
        ("runtime offset", op("div", op("add", S, RUNTIME), c(2)), None, True),
        ("runtime divisor", op("div", S, RUNTIME), None, True),
        ("lane divisor", op("div", c(64), S), None, True),
        ("real division", op("div", S, c(2), ttype=F64), None, True),
        ("neg", op("neg", S), None, True),
        ("max", op("max", S, c(3)), None, True),
        ("quotient off", q, None, False),
        ("scaled quotient off", op("mul", q, c(6)), None, False),
        ("loop var r/2 off", op("div", R, c(2)), R, False),
    ]


def _mismatches() -> list:
    out = []
    for label, node, var, coefficient, offset in _accepted():
        form = index_form(node, var)
        if form is None or (form.coefficient, form.offset) != (coefficient, offset):
            out.append((label, form))
    return out


def test_every_accepted_shape_classifies_exactly():
    assert _mismatches() == []


@pytest.mark.parametrize("label,node,var,quotients", _rejected(),
                         ids=[row[0] for row in _rejected()])
def test_every_rejected_shape_is_none(label, node, var, quotients):
    assert index_form(node, var, quotients=quotients) is None, label


def test_a_planted_wrong_coefficient_is_caught(monkeypatch):
    """Non-vacuity: an off-by-one integer factor must break the accepted table,
    on the row it touches and only through the coefficient it scales."""
    real = access._int_factor

    def planted(node):
        value, label = real(node)
        return (None, None) if value is None else (value + 1, label)

    monkeypatch.setattr(access, "_int_factor", planted)
    broken = dict(_mismatches())
    assert "s*3+2" in broken
    assert broken["s*3+2"].coefficient == 4
    assert "major half s/6" in broken
    assert "lane" not in broken


def test_the_coefficient_carries_its_provenance():
    named = index_form(op("mul", S, RoleConst(6, "t_row_stride")))
    assert named == IndexForm(Fraction(6), Fraction(0), ("t_row_stride",), False)
    anonymous = index_form(op("mul", S, c(6)))
    assert anonymous.labels == () and anonymous.anonymous
    free = index_form(op("mul", R, c(6)))       # sample-free: no provenance
    assert free == IndexForm(Fraction(0), None)


def test_var_is_matched_by_identity():
    """Against a loop index, the lane and a DIFFERENT loop index are bases."""
    other = LoopIndex(0, "r")
    assert index_form(op("add", R, S), R) == IndexForm(Fraction(1), None)
    assert index_form(other, R) == IndexForm(Fraction(0), None)


@hawk.kernel
def split_reads(t: hawk.Table["row":4, "col":6], y: hawk.Mutable[hawk.Scalar]):
    row, col = m.split_index(6)
    y = t.at(row=row, col=0) + t.at(row=row, col=col)


def test_traced_split_index_reads():
    """The forms the tracer actually mints: a read through the major half alone
    is a quotient form (six lanes per row), the full ``(major, minor)`` read is
    not an index form at all."""
    reads = [n for n in split_reads.sinks[0].value.operands if isinstance(n, At)]
    forms = [index_form(r.index) for r in reads]
    assert len(forms) == 2 and None in forms
    (major,) = [f for f in forms if f is not None]
    assert (major.coefficient, major.offset) == (Fraction(1, 6), 0)
