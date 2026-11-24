# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivatives of a ``Call`` of a function: the rule functions a call of the derivative kernel calls.

A function has ONE forward rule (the reads and a tangent per read, giving the tangent of the result) and ONE reverse rule per
read (the reads and the adjoint of the result, giving that read's gradient). The author's rules are held by the function's
own node, and the call answers that node. A traced function without them gets functions derived from its body, written once
per derivative run however many calls it makes; a native function without them is refused by name, since its text is opaque.

The derived functions are ordinary functions of the IR: the body is rewritten as a small kernel over placeholder leaves, run
through the kernel transform of :mod:`hawk.diff.transform`, and its result is put back over the parameters of the new function.
"""

from __future__ import annotations

from typing import Any

from .. import _ir
from ..errors import HawkError
from ..trace.function import param_nodes, rule_signature

#: The element types that carry a derivative.
FLOATS = ("f64", "f32")

#: The planes of the small kernel a body is differentiated as: the ``i``-th read, and the one result.
_READ = "$in{}"
_RESULT = "$out"


def _leaves_by_name(root: Any) -> dict:
    """The leaves of the graph under ``root``, loop bodies included, by name."""
    nodes = list(_ir.post_order([root]))
    for loop in [n for n in nodes if _ir.is_a(n, "loop")]:
        nodes += _ir.body_nodes(loop)
    return {n.field_text("name"): n for n in nodes if _ir.is_a(n, "leaf")}


def _derive(function: Any, direction: str, index: int) -> Any:
    """The Function node, in the context of ``function``, that differentiates the traced ``function`` in ``direction``.

    ``index`` is the read a reverse function gives the gradient of."""
    from . import transform  # the transform calls this module for every call it meets

    context = function.ctx
    reads = function.field_nodes("params")
    (body,) = function.field_nodes("body")
    result = _ir.type_of(function)
    label, signature, returns = rule_signature(
        function.field_text("name"),
        tuple((p.field_text("name"), _ir.type_of(p)) for p in reads),
        result,
        direction,
        index,
    )
    params = param_nodes(context, signature)
    placeholders = {p: _ir.leaf(context, "uniform", "uniform", _READ.format(i), _ir.type_of(p))
                    for i, p in enumerate(reads)}  # fmt: skip
    value = _ir.substitute([body], placeholders)[0]
    out = _ir.make_node("Assign", result, [value], name=_RESULT)
    present = _leaves_by_name(value)
    bound = {_READ.format(i): params[i] for i in range(len(reads))}
    floats = result.dtype in FLOATS
    if direction == "jvp":
        if not (
            floats
            and any(_ir.type_of(leaf).dtype in FLOATS for leaf in present.values())
        ):
            return _ir.function(
                context, label, params, _ir.zero_like(context, (), returns), returns
            )
        derived = transform.jvp((out,))
        bound |= {
            f"{transform.TANGENT_PREFIX}_{_READ.format(i)}": params[len(reads) + i]
            for i in range(len(reads))
        }
    else:
        wanted = _READ.format(index)
        if not (
            floats
            and wanted in present
            and _ir.type_of(present[wanted]).dtype in FLOATS
        ):
            return _ir.function(
                context, label, params, _ir.zero_like(context, (), returns), returns
            )
        derived = transform.vjp((out,), wrt=[wanted])
        bound[f"{transform.ADJOINT_PREFIX}_{_RESULT}"] = params[len(reads)]
    root = derived[0].operands[0]
    found = _leaves_by_name(root)
    swapped = _ir.substitute(
        [root], {found[n]: p for n, p in bound.items() if n in found}
    )[0]
    return _ir.function(context, label, params, swapped, (), returns)


def _rules(call: Any, direction: str, memo: dict) -> tuple:
    """The Function nodes that differentiate the function ``call`` names in ``direction``: one for ``jvp``, one per read for
    ``vjp``. The function's own rules win; a traced function without them is derived, once per ``memo``; a native function
    without them is refused by name.

    ``memo`` belongs to one derivative run and holds what that run derived."""
    function = call.field_nodes("callee")[0]
    given = function.field_nodes(direction)
    if given:
        return given
    name = function.field_text("name")
    if call.field_flag("native"):
        raise HawkError(
            f"native function {name!r} has no {direction} rule; pass {direction}="
        )
    key = (function.field_text("digest"), direction)
    if key not in memo:
        count = len(function.field_nodes("params")) if direction == "vjp" else 1
        memo[key] = tuple(_derive(function, direction, i) for i in range(count))
    return memo[key]


def _call_of(function: Any, args: list) -> Any:
    """The ``Call`` node of the Function node ``function`` over the nodes ``args``."""
    result = _ir.type_of(function)
    return _ir.make_node(
        "Call",
        (result.shape, result.dtype, result.tag),
        args,
        context=function.ctx,
        function=(function.field_text("name"), function.field_text("digest")),
    )


def call_tangent(call: Any, tangents: list, memo: dict) -> Any:
    """The tangent of the ``Call`` node ``call``: a call of its forward rule on the reads and their tangents.

    ``tangents`` holds the tangent of each operand, ``None`` for one without; that one enters as a structural zero."""
    (rule,) = _rules(call, "jvp", memo)
    dots = [
        t if t is not None else _ir.zero_like(call.ctx, _ir.type_of(c))
        for t, c in zip(tangents, call.operands)
    ]
    return _call_of(rule, [*call.operands, *dots])


def call_adjoints(call: Any, adjoint: Any, memo: dict) -> tuple:
    """``((operand, adjoint), ...)`` of the ``Call`` node ``call``: for each operand that carries a derivative, a call of the
    reverse rule of that read on the reads and the adjoint of the result."""
    rules = _rules(call, "vjp", memo)
    return tuple(
        (child, _call_of(rule, [*call.operands, adjoint]))
        for child, rule in zip(call.operands, rules)
        if _ir.type_of(child).dtype in FLOATS
    )
