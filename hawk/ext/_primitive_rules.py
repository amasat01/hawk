# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A user primitive's derivative rules, kept as templates: ordinary IR over placeholder leaves.

A rule is registered per input signature; instantiating it substitutes a boundary node's operands
for the placeholders. The registry is a plain object the caller owns; nothing callable is stored.
It lives beside :mod:`hawk.ext.primitive` because a caller registers its rules from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Sequence

from .. import _ir
from .._errors import HawkError

#: A placeholder leaf's name starts with this; no plane or uniform may.
PREFIX = "$"
_WORDS = MappingProxyType({"vjp": "reverse (vjp)", "jvp": "forward (jvp)"})


@dataclass(frozen=True)
class PrimitiveDecl:
    """What registration declares of a primitive: its name.

    Its extension id and its rule directions.
    """

    name: str
    ext: str
    directions: tuple


@dataclass
class PrimitiveRegistry:
    """The primitives of one owner: ``name -> (declaration, {signature: {direction: (key.

    Templates)}})``.
    """

    primitives: dict = field(default_factory=dict)


def placeholder(context: Any, tag: str, ttype: Sequence) -> Any:
    """Make the free leaf a template reads: ``in<i>``, ``out``, ``cot`` or ``dot<i>``."""
    return _ir.leaf(context, "uniform", "uniform", PREFIX + tag, ttype)


def signature_of(node: Any) -> tuple:
    """Return the input types and the output type of a primitive node.

    The key of its rule instance.
    """
    return tuple(_ir.type_of(x) for x in node.operands[1:]), _ir.type_of(node)


def declare_primitive(
    registry: PrimitiveRegistry, context: Any, name: str, *, ext: str, directions: tuple
) -> PrimitiveDecl:
    """Register a primitive's name; an equal re-registration returns the held one.

    A different one refuses.
    """
    if not isinstance(name, str) or not name:
        raise HawkError(
            "a primitive is registered by NAME: the name is the boundary node's identity in the "
            "walk's digest"
        )
    if _ir.kind_reserved(context, name):
        raise HawkError(f"primitive {name!r}: the name is reserved (a core or declared kind)")
    decl = PrimitiveDecl(name, ext, tuple(directions))
    held = registry.primitives.get(name)
    if held is not None:
        if held[0] == decl:
            return decl
        raise HawkError(
            f"primitive {name!r} is already registered (by extension {held[0].ext!r}, rules "
            f"{held[0].directions}); a second registration under the same name is refused because "
            "the name is what identifies the boundary node in Walk.digest"
        )
    registry.primitives[name] = (decl, {})
    return decl


def _allowed(direction: str, count: int) -> set:
    """Return the placeholder names a template of ``direction`` may read freely."""
    names = {"out"} | {f"in{i}" for i in range(count)}
    if direction == "vjp":
        names.add("cot")
    else:
        names |= {f"dot{i}" for i in range(count)}
    return {PREFIX + name for name in names}


def _loops(template: Any) -> list:
    """Return the loops among the operand graph of ``template``."""
    return [n for n in _ir.post_order([template]) if _ir.is_a(n, "loop")]


def _walk(template: Any) -> list:
    """Return every node of ``template``, loop bodies included."""
    nodes = _ir.post_order([template])
    for lp in _loops(template):
        nodes += _ir.body_nodes(lp)
    return nodes


def register_rule(
    registry: PrimitiveRegistry, name: str, signature: tuple, direction: str, templates: Any
) -> None:
    """Register a rule instance for ``signature``: vjp one template per input, jvp one template.

    Checks the arity, the type each template owes and its free leaves; an equal re-registration is
    a no-op and a different one is refused.
    """
    held = registry.primitives.get(name)
    if held is None:
        raise HawkError(f"primitive {name!r} is not registered")
    inputs, output = signature
    if direction == "vjp":
        found, owed = tuple(templates), tuple(inputs)
        if len(found) != len(inputs):
            raise HawkError(
                f"primitive {name!r}: its vjp returned {len(found)} gradient(s) for "
                f"{len(inputs)} input(s). A reverse rule returns ONE adjoint per declared "
                "input, in call order \u2014 a short tuple would silently leave an input's "
                "gradient at zero"
            )
    elif direction == "jvp":
        found, owed = (templates,), (output,)
    else:
        raise HawkError(f"primitive {name!r}: unknown direction {direction!r}")
    allowed = _allowed(direction, len(inputs))
    for template, want in zip(found, owed):
        have = _ir.type_of(template)
        if have != want:
            raise HawkError(
                f"primitive {name!r}: its {direction} produced a value typed {have!r} where "
                f"{want!r} was owed. A supplied rule is checked against the type it stands in for, "
                "because a rank or dtype mismatch here reaches the emitter as a shape error far "
                "from the declaration that caused it"
            )
        _ir.rule_template_check(name, [template], sorted(allowed))
    key = tuple(template.ctx.dump([template]) for template in found)
    instance = held[1].setdefault(signature, {})
    old = instance.get(direction)
    if old is not None and old[0] != key:
        raise HawkError(
            f"primitive {name!r} is already registered with a different {direction} rule for this "
            "signature"
        )
    instance[direction] = (key, found)


def has_rule(registry: PrimitiveRegistry, node: Any, direction: str) -> bool:
    """Whether a rule instance of ``direction`` is registered for the node's signature."""
    held = registry.primitives.get(node.field_text("name"))
    return held is not None and direction in held[1].get(signature_of(node), {})


def substitute(template: Any, mapping: dict) -> Any:
    """Return ``template`` with each placeholder leaf replaced by ``mapping[name]``.

    The substitution goes through operands and loop bodies in one library call; a placeholder the
    mapping does not name is left as it is.
    """
    if mapping:
        # a rule is registered in its own context; the nodes substituted in belong to the kernel's
        template = _ir.adopt(next(iter(mapping.values())).ctx, [template])[0]
    leaves = {
        node.field_text("name"): node
        for node in _walk(template)
        if _ir.is_a(node, "leaf") and node.field_text("name").startswith(PREFIX)
    }
    by_node = {leaves[name]: value for name, value in mapping.items() if name in leaves}
    return _ir.substitute([template], by_node)[0]


def _supplied(registry: PrimitiveRegistry, node: Any, direction: str) -> tuple:
    """Return the registered templates of ``direction`` for ``node``, or refuse by name."""
    name = node.field_text("name")
    held = registry.primitives.get(name)
    if held is None or direction not in held[0].directions:
        raise HawkError(
            f"primitive {name!r} supplies no {_WORDS[direction]} rule, so this direction cannot be "
            "taken. HAWK does not fall back to differentiating THROUGH a primitive's forward: the "
            "forward is inlined and the table would happily produce a number, which is exactly the "
            f"number {name!r} was declared to replace. Register a {direction} rule for it"
        )
    got = held[1].get(signature_of(node), {}).get(direction)
    if got is None:
        raise HawkError(
            f"primitive {name!r}: no {direction} rule instance for this signature is registered"
        )
    return got[1]


def vjp_contributions(registry: PrimitiveRegistry, node: Any, adjoint: Any) -> tuple:
    """Return ``((input, adjoint), ...)`` for one boundary node.

    From the registered rule instance.
    """
    templates = _supplied(registry, node, "vjp")
    inputs = node.operands[1:]
    bound = {PREFIX + f"in{i}": x for i, x in enumerate(inputs)}
    bound[PREFIX + "cot"], bound[PREFIX + "out"] = adjoint, node
    return tuple((child, substitute(t, bound)) for child, t in zip(inputs, templates))


def jvp_tangent(registry: PrimitiveRegistry, node: Any, supplied: Sequence) -> Any:
    """Return the output's tangent for one boundary node; an input with no tangent is a zero."""
    (template,) = _supplied(registry, node, "jvp")
    inputs = node.operands[1:]
    bound = {PREFIX + f"in{i}": x for i, x in enumerate(inputs)}
    for i, (tangent, x) in enumerate(zip(supplied, inputs)):
        zero = _ir.zero_like(node.ctx, _ir.type_of(x))
        bound[PREFIX + f"dot{i}"] = tangent if tangent is not None else zero
    bound[PREFIX + "out"] = node
    return substitute(template, bound)
