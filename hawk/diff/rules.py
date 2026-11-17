# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative rule of a node kind, as the derivative builder asks for it.

The library owns the rule templates and runs them (:mod:`hawk._ir`); this module only names which
table answers which kind, and hands the builder one record per kind with its two directions. A
direction takes the node and the incoming adjoint (``vjp``) or a function giving the tangent of an
operand (``jvp``), exactly as the builder calls a rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .. import _ir
from .._errors import HawkError

#: The kinds with no derivative (piecewise constant, integer-valued or random) and the kinds the
#: transform handles itself (``at``, ``dispatch``).
ZERO_KINDS = _ir.rule_modes("zero")
WALKER_KINDS = _ir.rule_modes("walker")


@dataclass(frozen=True)
class DiffRule:
    """One node kind's two derivative directions and how it is answered (``rule``, ``zero``.

    ``walker``).
    """

    vjp: Callable[..., Any] | None = None
    jvp: Callable[..., Any] | None = None
    mode: str = "rule"


def _tangents(node: Any, tangent_of: Callable[[Any], Any]) -> dict:
    """Map each operand that has a tangent to it (an operand with none is left out)."""
    found = {operand: tangent_of(operand) for operand in node.operands}
    return {operand: tangent for operand, tangent in found.items() if tangent is not None}


def _sum(terms: tuple) -> Any:
    """Return the sum of the live terms, or ``None`` when there are none."""
    if not terms:
        return None
    total = terms[0]
    for term in terms[1:]:
        total = _ir.make("add", [total, term])
    return total


def _binary(kind: str) -> DiffRule:
    """Make the rule of an elementwise binary: factor templates for both directions."""

    def vjp(node: Any, adjoint: Any) -> tuple:
        a, b = node.operands
        flat = _ir.rule_factor("bin_vjp", kind, {"n": node, "g": adjoint, "a": a, "b": b})
        return tuple(zip(flat[::2], flat[1::2]))

    def jvp(node: Any, tangent_of: Callable[[Any], Any]) -> Any:
        a, b = node.operands
        bound = {"n": node, "ta": tangent_of(a), "tb": tangent_of(b), "a": a, "b": b}
        return _sum(_ir.rule_factor("bin_jvp", kind, bound))

    return DiffRule(vjp, jvp)


def _unary(kind: str) -> DiffRule:
    """Make the rule of an elementwise unary: its derivative factor times the adjoint or tangent."""

    def factor(node: Any, x: Any) -> Any:
        (slope,) = _ir.rule_factor("unary_d", kind, {"n": node, "x": x})
        return slope

    def vjp(node: Any, adjoint: Any) -> tuple:
        x = node.operands[0]
        return ((x, _ir.make("mul", [adjoint, factor(node, x)])),)

    def jvp(node: Any, tangent_of: Callable[[Any], Any]) -> Any:
        x = node.operands[0]
        return _ir.make("mul", [tangent_of(x), factor(node, x)])

    return DiffRule(vjp, jvp)


def _program(table: str, kind: str) -> DiffRule:
    """Make the rule of a kind whose derivative is a library program (``core``, ``linalg``.

    ``rays``).
    """
    return DiffRule(
        lambda node, adjoint: _ir.rule_vjp(table, kind, node, adjoint),
        lambda node, tangent_of: _ir.rule_jvp(table, kind, node, _tangents(node, tangent_of)),
    )


_REDUCE = DiffRule(
    lambda node, adjoint: _ir.reduce_vjp(node, adjoint),
    lambda node, tangent_of: _ir.reduce_jvp(node, _tangents(node, tangent_of)),
)
_ZERO = DiffRule(lambda node, adjoint: (), lambda node, tangent_of: None, mode="zero")
_WALKER = DiffRule(mode="walker")


def rule_for(kind: str, rank: int) -> DiffRule:
    """Return the rule of a node kind (``rank`` is passed as the builder does, no rule uses it).

    A kind no table knows is refused by name: the table is closed-world, so this is an
    undifferentiable node, not a silent zero. When two tables name a kind the later one wins, in
    the order walker, reduction, core, rays, linalg, zero, unary, binary.
    """
    if kind in WALKER_KINDS:
        return _WALKER
    if kind == _ir.REDUCE_KIND:
        return _REDUCE
    for table in ("core", "rays", "linalg"):
        if kind in _ir.rule_kinds(table):
            return _program(table, kind)
    if kind in ZERO_KINDS:
        return _ZERO
    if kind in _ir.rule_kinds("unary_d"):
        return _unary(kind)
    if kind in _ir.rule_kinds("bin_vjp"):
        return _binary(kind)
    raise HawkError(
        f"no derivative rule for node kind {kind!r} (rank {rank}): HAWK's rule table is "
        "closed-world, a kind with no derivative is registered with an explicit zero rule, never "
        "omitted, so this is a genuinely undifferentiable node, not a silent zero"
    )
