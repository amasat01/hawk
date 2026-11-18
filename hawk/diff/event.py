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

from .. import _ir
from .._errors import HawkError
from .transform import _READ_ROLE, _single_step

__all__ = ["EventGuard", "event_guard", "guard_kernel", "state_guard"]

#: Comparisons a guard is read from, with the side ``g`` keeps positive
#: while the sample runs: ``"lhs"`` -> ``g = lhs - rhs``, else ``rhs - lhs``.
_SIGN = {"lt": "lhs", "le": "lhs", "gt": "rhs", "ge": "rhs"}

#: Piecewise-constant float ops: a smooth guard never passes through one.
_STEPWISE = ("floor", "ceil", "trunc", "round", "rint", "sign")

_FLOAT = ("f64", "f32")

#: The guard plane names :func:`guard_kernel` commits.
GUARD_PREFIX = "event_g"

#: The spelling of a comparison kind in a refusal.
_OPS = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "land": "and",
    "lor": "or",
    "lnot": "not",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


class EventGuard(NamedTuple):
    """A kernel's stop rule as smooth guards: ``guards[i] <= 0`` is stop
    rule ``i`` firing; ``rules[i]`` spells the comparison it came from."""

    kernel: str
    mask: str
    guards: tuple
    rules: tuple


def _literal(node: Any) -> Any:
    """The plain value a constant holds, read as the kind of field it is stored in."""
    reader = {1: node.field_text, 2: node.field_int, 3: node.field_real, 4: node.field_flag}
    kind = node.field_kind("literal")
    return reader[kind]("literal") if kind in reader else None


def _refuse(kernel: str, cond: Any) -> HawkError:
    return HawkError(
        f'events="refine": the stop rule of {kernel!r} is not a smooth crossing '
        f"({spell(cond)}); refine needs g(x) that changes sign, not a count or a "
        "flag"
    )


def spell(node: Any, depth: int = 3) -> str:
    """A short, readable spelling of ``node`` for a refusal."""
    if _ir.is_a(node, "leaf"):
        return node.field_text("name")
    if _ir.is_a(node, "const"):
        return repr(_literal(node))
    if depth <= 0:
        return "..."
    ops = node.operands
    kind = node.field_text("kind")
    if kind in _OPS and len(ops) == 2:
        lhs, rhs = spell(ops[0], depth - 1), spell(ops[1], depth - 1)
        return f"({lhs} {_OPS[kind]} {rhs})"
    if kind == "neg" and ops:
        return f"(-{spell(ops[0], depth - 1)})"
    if kind == "lnot" and ops:
        return f"(not {spell(ops[0], depth - 1)})"
    inner = ", ".join(spell(o, depth - 1) for o in ops)
    return f"{kind}({inner})"


def _smooth(node: Any, memo: dict) -> bool:
    """Whether ``node`` depends SMOOTHLY on a floating-point input; ``None``
    when a piecewise-constant op sits on a floating-point path (refused)."""
    got = memo.get(node)
    if got is not None or node in memo:
        return got
    if _ir.is_a(node, "leaf"):
        out = _ir.type_of(node).dtype in _FLOAT
    elif _ir.is_a(node, "const", "sample_index", "plane_extent", "loop_index"):
        out = False
    elif _ir.type_of(node).dtype not in _FLOAT:
        out = False
    elif _ir.is_a(node, "select"):
        a, b = _smooth(node.operands[1], memo), _smooth(node.operands[2], memo)
        out = None if a is None or b is None else (a or b)
    elif _ir.is_a(node, "dispatch"):
        parts = [_smooth(b, memo) for b in node.operands[1:]]
        out = None if None in parts else any(parts)
    elif node.field_text("kind") in _STEPWISE:
        out = None if _smooth(node.operands[0], memo) else False
    elif node.field_text("kind") in _ir.COMPARE + _ir.LOGICAL + _ir.PREDICATE:
        out = False
    else:
        parts = [_smooth(o, memo) for o in node.operands]
        out = None if None in parts else any(parts)
    memo[node] = out
    return out


def _comparisons(cond: Any) -> list:
    """The comparisons an ``or`` chain is built of, in operand order;
    anything else (an ``and``, a ``not``, a flag) as itself."""
    if cond.field_text("kind") == "lor":
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
    finishes = [s for s in getattr(kernel, "sinks", ()) if _ir.is_a(s, "finish")]
    if not finishes:
        raise HawkError(
            f'events="refine": {name!r} never finishes its sample; there is no '
            "stop rule to refine — finish with `terminated = cond`"
        )
    finish = finishes[0]
    if kernel.steps != 1:
        raise HawkError(
            f'events="refine": {name!r} takes {kernel.steps} steps per launch; '
            "the guard is read from the single step"
        )
    guards, rules = _read(name, finish.operands[0])
    return EventGuard(name, finish.field_text("name"), guards, rules)


def guards_of(name: str, cond: Any) -> tuple:
    """The guards of the stop condition ``cond`` (kernel ``name`` names a
    refusal), under :func:`event_guard`'s convention and refusals."""
    return _read(name, cond)[0]


def _read(name: str, cond: Any) -> tuple:
    memo: dict = {}
    guards, rules = [], []
    for cmp in _comparisons(cond):
        side = _SIGN.get(cmp.field_text("kind"))
        if side is None:
            raise _refuse(name, cond)
        lhs, rhs = cmp.operands
        smooth = (_smooth(lhs, memo), _smooth(rhs, memo))
        if None in smooth or not any(smooth):
            raise _refuse(name, cond)
        g = _ir.make("sub", (lhs, rhs) if side == "lhs" else (rhs, lhs))
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
    sinks = tuple(_guard_sink(i, g) for i, g in enumerate(rebuilt))
    planes = {k: v for k, v in kernel.planes.items() if getattr(v, "form", None) == "terminated"}
    return Kernel(f"{kernel.name}_guard", sinks, planes, kind=kernel.kind)


def state_guard(kernel: Any) -> Any:
    """A kernel committing each guard of ``kernel`` into ``event_g<i>`` as a
    function of the state AFTER the step, named ``<name>_state_guard``: every
    value the step commits is read back from its own plane (the time plane
    included), so its :func:`~hawk.diff.vjp` is ``g_x`` over the extended
    state — what a graze test measures at a located event. A guard that also
    reads a value the step does not commit reads it as a plain plane.

    Example::

        @hawk.kernel(steps=1, time="t", time_step="dt")
        def osc(dt: Param, a: Param, terminated: Terminated,
                x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
            v_start = v
            v = v_start - dt * x
            x = x + dt * v_start
            t = t + dt
            terminated = x <= a * t

        state_guard(osc)   # event_g0 = a*t - x... read from the planes x, t
    """
    from ..trace.kernel import Kernel

    eg = event_guard(kernel)
    step = _single_step(kernel)
    mapping = {s.operands[0]: _plane_read(s) for s in step.sinks if _ir.is_a(s, "assign")}
    guards = as_plain_reads(_ir.substitute(eg.guards, mapping))
    sinks = tuple(_guard_sink(i, g) for i, g in enumerate(guards))
    return Kernel(f"{step.name}_state_guard", sinks, {}, kind=step.kind)


def _guard_sink(i: int, guard: Any) -> Any:
    """The sink committing guard ``i`` into its ``event_g<i>`` plane."""
    return _ir.make_node("Assign", _ir.type_of(guard), [guard], name=f"{GUARD_PREFIX}{i}")


def _plane_read(node: Any) -> Any:
    """A plain read of the plane ``node`` (a launch-start read or a sink) stands for."""
    ttype = _ir.type_of(node)
    role = _READ_ROLE[len(ttype.shape)]
    return _ir.leaf(node.ctx, "vocab_read", role, node.field_text("name"), ttype)


def as_plain_reads(roots: tuple) -> tuple:
    """``roots`` with every launch-start read of a Mutable turned into a
    plain read of the same plane: a pure function of its inputs, which is
    what ``jvp``/``vjp`` differentiate (they refuse a launch-start read)."""
    mapping: dict = {}
    for node in _ir.post_order(roots):
        if _ir.is_a(node, "leaf") and node.field_text("kind") == "prior_read":
            mapping[node] = _plane_read(node)
    return _ir.substitute(roots, mapping) if mapping else tuple(roots)
