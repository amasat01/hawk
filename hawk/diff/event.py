# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A step kernel's stop rule as a DIFFERENTIABLE guard.

A kernel finishing its sample on ``terminated = cond`` stops on a boolean,
whose derivative is zero everywhere it exists. Event refinement needs the
smooth function that CHANGES SIGN where the boolean flips: :func:`event_guard`
reads it out of ``cond`` as an ordinary IR node, never a mask, under ONE
sign convention — the sample stops where ``g <= 0``:

* ``a < b`` / ``a <= b``  ->  ``g = a - b``;
* ``a > b`` / ``a >= b``  ->  ``g = b - a``;
* ``c1 or c2 ...``        ->  one guard per comparison (each its own stop rule).

Everything else is refused at the guard's construction, naming the kernel
and the rule: ``and``/``not``, ``==``/``!=``, a bare flag, an integer
comparison (a count), and a comparison whose two sides carry no smooth
dependence on a floating-point input (a rounding, a sign, a class test or a
comparison on the path). The guard is a function of the step's PRE-step
inputs — the same body the step evaluates — so evaluating it at a step size
``tau`` is the guard at the end of a partial step.

:func:`guard_kernel` packages the guards as a kernel committing one
``event_g<i>`` plane each, with every launch-start read of a Mutable turned
into a plain read of the same plane: a pure function of the state, so the
ordinary :func:`~hawk.diff.jvp`/:func:`~hawk.diff.vjp` differentiate it with
respect to the state AND the parameters.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from ..ir import Const, HawkError, Leaf, Node
from ..ir import make as _mk
from ..ir.loop_nodes import LoopIndex
from ..ir.nodes import Assign, Dispatch, Finish, SampleIndex, Select
from ..ir.ops import COMPARE, LOGICAL, PREDICATE
from ..ir.walk import substitute
from .transform import _READ_ROLE

__all__ = ["EventGuard", "event_guard", "guard_kernel"]

#: Comparisons a guard is read from, with the side ``g`` keeps positive
#: while the sample runs: ``"lhs"`` -> ``g = lhs - rhs``, else ``rhs - lhs``.
_SIGN = {"lt": "lhs", "le": "lhs", "gt": "rhs", "ge": "rhs"}

#: Piecewise-constant float ops: a smooth guard never passes through one.
_STEPWISE = ("floor", "ceil", "trunc", "round", "rint", "sign")

_FLOAT = ("f64", "f32")

#: The guard plane names :func:`guard_kernel` commits.
GUARD_PREFIX = "event_g"

#: The spelling of a comparison kind in a refusal.
_OPS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=",
        "land": "and", "lor": "or", "lnot": "not",
        "add": "+", "sub": "-", "mul": "*", "div": "/"}


class EventGuard(NamedTuple):
    """A kernel's stop rule as smooth guards: ``guards[i] <= 0`` is stop
    rule ``i`` firing; ``rules[i]`` spells the comparison it came from."""

    kernel: str
    mask: str
    guards: tuple
    rules: tuple


def _single_step(kernel: Any) -> Any:
    from ..trace.steps import default_step

    return default_step(kernel)


def _refuse(kernel: str, cond: Node) -> HawkError:
    return HawkError(
        f'events="refine": the stop rule of {kernel!r} is not a smooth crossing '
        f"({spell(cond)}); refine needs g(x) that changes sign, not a count or a "
        "flag")


def spell(node: Node, depth: int = 3) -> str:
    """A short, readable spelling of ``node`` for a refusal."""
    if isinstance(node, Leaf):
        return node.name
    if isinstance(node, Const):
        return repr(node.literal)
    if depth <= 0:
        return "..."
    ops = node.operands
    if node.kind in _OPS and len(ops) == 2:
        lhs, rhs = spell(ops[0], depth - 1), spell(ops[1], depth - 1)
        return f"({lhs} {_OPS[node.kind]} {rhs})"
    if node.kind == "neg" and ops:
        return f"(-{spell(ops[0], depth - 1)})"
    if node.kind == "lnot" and ops:
        return f"(not {spell(ops[0], depth - 1)})"
    inner = ", ".join(spell(o, depth - 1) for o in ops)
    return f"{node.kind}({inner})"


def _smooth(node: Node, memo: dict) -> bool:
    """Whether ``node`` depends SMOOTHLY on a floating-point input; ``None``
    when a piecewise-constant op sits on a floating-point path (refused)."""
    got = memo.get(id(node))
    if got is not None or id(node) in memo:
        return got
    if isinstance(node, Leaf):
        out = node.ttype.dtype in _FLOAT
    elif isinstance(node, (Const, SampleIndex, LoopIndex)):
        out = False
    elif node.ttype.dtype not in _FLOAT:
        out = False
    elif isinstance(node, Select):
        a, b = _smooth(node.operands[1], memo), _smooth(node.operands[2], memo)
        out = None if a is None or b is None else (a or b)
    elif isinstance(node, Dispatch):
        parts = [_smooth(b, memo) for b in node.branches]
        out = None if None in parts else any(parts)
    elif node.kind in _STEPWISE:
        out = None if _smooth(node.operands[0], memo) else False
    elif node.kind in COMPARE + LOGICAL + PREDICATE:
        out = False
    else:
        parts = [_smooth(o, memo) for o in node.operands]
        out = None if None in parts else any(parts)
    memo[id(node)] = out
    return out


def _comparisons(cond: Node) -> list:
    """The comparisons an ``or`` chain is built of, in operand order;
    anything else (an ``and``, a ``not``, a flag) as itself."""
    if getattr(cond, "kind", None) == "lor":
        out: list = []
        for o in cond.operands:
            out.extend(_comparisons(o))
        return out
    return [cond]


def event_guard(kernel: Any) -> EventGuard:
    """``kernel``'s stop rule as :class:`EventGuard` (see the module
    docstring for the sign convention and what is refused). A default
    automatic kernel is read through its single step."""
    kernel = _single_step(kernel)
    name = getattr(kernel, "name", "?")
    finishes = [s for s in getattr(kernel, "sinks", ()) if isinstance(s, Finish)]
    if not finishes:
        raise HawkError(
            f'events="refine": {name!r} never finishes its sample; there is no '
            "stop rule to refine — finish with `terminated = cond`")
    finish = finishes[0]
    if finish.steps != 1:
        raise HawkError(
            f'events="refine": {name!r} takes {finish.steps} steps per launch; '
            "the guard is read from the single step")
    guards, rules = _read(name, finish.value)
    return EventGuard(name, finish.name, guards, rules)


def guards_of(name: str, cond: Node) -> tuple:
    """The guards of the stop condition ``cond`` (kernel ``name`` names a
    refusal), under :func:`event_guard`'s convention and refusals."""
    return _read(name, cond)[0]


def _read(name: str, cond: Node) -> tuple:
    memo: dict = {}
    guards, rules = [], []
    for cmp in _comparisons(cond):
        side = _SIGN.get(getattr(cmp, "kind", None))
        if side is None:
            raise _refuse(name, cond)
        lhs, rhs = cmp.operands
        smooth = (_smooth(lhs, memo), _smooth(rhs, memo))
        if None in smooth or not any(smooth):
            raise _refuse(name, cond)
        g = _mk("sub", (lhs, rhs) if side == "lhs" else (rhs, lhs))
        guards.append(g)
        rules.append(spell(cmp))
    return tuple(guards), tuple(rules)


def guard_kernel(kernel: Any) -> Any:
    """A kernel committing each guard of ``kernel`` into ``event_g<i>``
    (rank 0), every launch-start Mutable read turned into a plain read of
    the same plane, named ``<name>_guard``; same kind as ``kernel``."""
    from ..trace.kernel import Kernel

    eg = event_guard(kernel)
    kernel = _single_step(kernel)
    rebuilt = as_plain_reads(eg.guards)
    sinks = tuple(Assign(f"{GUARD_PREFIX}{i}", g, g.ttype)
                  for i, g in enumerate(rebuilt))
    planes = {k: v for k, v in kernel.planes.items()
              if getattr(v, "form", None) == "terminated"}
    return Kernel(f"{kernel.name}_guard", sinks, planes, kind=kernel.kind)


def as_plain_reads(roots: tuple) -> tuple:
    """``roots`` with every launch-start read of a Mutable turned into a
    plain read of the same plane: a pure function of its inputs, which is
    what ``jvp``/``vjp`` differentiate (they refuse a launch-start read)."""
    mapping: dict = {}
    stack, seen = list(roots), set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Leaf) and node.kind == "prior_read":
            role = _READ_ROLE[len(node.ttype.shape)]
            mapping[id(node)] = Leaf("vocab_read", role, node.name, node.ttype)
        stack.extend(node.operands)
    return substitute(roots, mapping) if mapping else tuple(roots)
