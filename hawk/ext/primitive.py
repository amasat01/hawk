# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The custom-primitive seam: a custom primitive with a SUPPLIED derivative.

The generated-rule path (:mod:`hawk.diff`) stays the default. This seam
is for a forward whose derivative the author knows and the rule table
would only approximate or compute expensively — a declaration consumed
by the transform, never a post-hoc edit of a derived graph.

**The forward is inlined, not called.** ``@primitive`` decorates a plain
Python function of traced values; a call in a body runs it, building the
same DAG the author would have built by hand, since HAWK's codegen
contract is one body string per kernel.

**The boundary is recorded anyway.** The inlined subgraph is wrapped in
a :class:`hawk.ir.nodes.Primitive` node whose operands are that subgraph
plus the declared inputs. Reverse and forward mode apply the supplied
rule to those inputs and never push into the subgraph, so the author's
derivative replaces the table's. A shared subexpression still earns its
own adjoint via the walk's structural dedup.

**A missing direction is a refusal, not a fallback.** An unsupplied
direction raises, naming the primitive and direction, rather than
silently producing the table's unwanted number — the same reason a name
collision refuses: two primitives sharing a name would make the derived
IR depend on import order.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..diff.rules import zero_like
from ..ir import HawkError, Node, Primitive
from ..trace.value import Value, node_of

#: Every registered primitive, by name: the name is the node's dedup
#: identity and therefore part of ``Walk.digest``.
_REGISTRY: dict = {}

#: Makes the check-and-claim of a name one step: of N threads registering one
#: name, exactly one wins and the others raise (a bare get-then-set would let
#: two both pass the check).
_REGISTRY_LOCK = threading.Lock()


def primitives() -> dict:
    """A copy of the primitive registry, by name (introspection only)."""
    return dict(_REGISTRY)


@dataclass(frozen=True)
class PrimitiveDef:
    """One registered primitive: its forward and the rules that replace
    the table. ``vjp`` is a traced function of ``(inputs..., bar_out)``
    returning one gradient per input; ``jvp`` is a traced function of
    ``(inputs..., dot_inputs...)`` returning the output's tangent. Both
    build ordinary HAWK IR, so a derived kernel goes through the same
    walk, emitter and cache as any primal."""

    name: str
    forward: Callable
    vjp: Callable | None = None
    jvp: Callable | None = None

    def __call__(self, *args: Any) -> Value:
        """Trace the forward INLINE over ``args`` and record the boundary."""
        inputs = tuple(node_of(a) for a in args)
        traced = self.forward(*(Value(x) for x in inputs))
        return Value(Primitive(self.name, self, node_of(traced), inputs))

    def vjp_contributions(self, node: Primitive, g: Node) -> tuple:
        """``((input, adjoint), …)`` for one boundary node, from the supplied rule."""
        rule = self._rule("vjp", "reverse (vjp)")
        inputs = node.inputs
        got = rule(*(Value(x) for x in inputs), Value(g))
        got = tuple(got) if isinstance(got, (tuple, list)) else (got,)
        if len(got) != len(inputs):
            raise HawkError(
                f"primitive {self.name!r}: its vjp returned {len(got)} gradient(s) "
                f"for {len(inputs)} input(s). A reverse rule returns ONE adjoint per "
                "declared input, in call order — a short tuple would silently leave "
                "an input's gradient at zero"
            )
        return tuple((child, self._typed(child, node_of(value), "vjp"))
                     for child, value in zip(inputs, got))

    def jvp_tangent(self, node: Primitive, supplied: Sequence) -> Node:
        """The output's tangent for one boundary node, from the supplied
        rule. An input with no incoming tangent is passed a structural
        zero: the rule is a function of a fixed argument list."""
        rule = self._rule("jvp", "forward (jvp)")
        inputs = node.inputs
        dots = [t if t is not None else zero_like(x.ttype)
                for t, x in zip(supplied, inputs)]
        got = rule(*(Value(x) for x in inputs), *(Value(d) for d in dots))
        return self._typed(node, node_of(got), "jvp")

    def _rule(self, which: str, direction: str) -> Callable:
        rule = getattr(self, which)
        if rule is None:
            raise HawkError(
                f"primitive {self.name!r} supplies no {direction} rule, so this "
                f"direction cannot be taken. HAWK does not fall "
                "back to differentiating THROUGH a primitive's forward: the forward "
                "is inlined and the table would happily produce a number, which is "
                f"exactly the number {self.name!r} was declared to replace. Register "
                f"it as primitive({self.name!r}, {which}=...)"
            )
        return rule

    def _typed(self, target: Node, value: Node, which: str) -> Node:
        if value.ttype != target.ttype:
            raise HawkError(
                f"primitive {self.name!r}: its {which} produced a value typed "
                f"{value.ttype!r} where {target.ttype!r} was owed. A supplied rule "
                "is checked against the type it stands in for, because a rank or "
                "dtype mismatch here reaches the emitter as a shape error far from "
                "the declaration that caused it"
            )
        return value


def primitive(name: str, *, vjp: Callable | None = None,
              jvp: Callable | None = None) -> Callable:
    """Register a custom primitive named ``name``.

    Used as a decorator on the forward::

        @primitive("softplus", vjp=lambda x, bar: bar / (1.0 + exp(-x)))
        def softplus(x):
            return log(1.0 + exp(x))

    The decorated object is a :class:`PrimitiveDef`, callable from a
    kernel body like a free function. A second primitive under a taken
    name is refused: a collision would make two kernels' walk digests
    agree while their derivatives disagreed."""
    def register(fn: Callable) -> PrimitiveDef:
        if not name:
            raise HawkError(
                "primitive(): a primitive is registered by NAME — the name is the "
                "boundary node's identity in the walk's digest")
        definition = PrimitiveDef(name, fn, vjp, jvp)
        with _REGISTRY_LOCK:
            held = _REGISTRY.setdefault(name, definition)
        if held is not definition:
            raise HawkError(
                f"primitive {name!r} is already registered (its forward is "
                f"{held.forward!r}); a second registration under the same name is "
                "refused because the name is what identifies the boundary node in "
                "Walk.digest — two primitives sharing it would give two different "
                "kernels the same content hash, and the cache would serve one for "
                "the other"
            )
        return definition
    return register
