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

from raptor.rules import primitive as rules
from ..dialects import KINDS
from ..ir import Node, Primitive
from .._errors import HawkError
from ..trace.helpers import call, call_lowered
from ..trace.value import Value, node_of

#: Every registered primitive's FRONT END, by name: the Python callables live here, in hawk, only so a rule can be TRACED once
#: per input signature into a template. What raptor keeps (in :data:`hawk.dialects.KINDS`) is data: the name and the templates.
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
    """One registered primitive's front end: its forward and the functions that are TRACED into the rule templates raptor
    stores. ``vjp`` is a traced function of ``(inputs..., bar_out)`` returning one gradient per input; ``jvp`` a traced
    function of ``(inputs..., dot_inputs...)`` returning the output's tangent. Both build ordinary IR."""

    name: str
    forward: Callable
    vjp: Callable | None = None
    jvp: Callable | None = None

    def __call__(self, *args: Any) -> Value:
        """Trace the forward INLINE over ``args`` and record the boundary."""
        inputs = tuple(node_of(a) for a in args)
        traced = call(self.forward, *(Value(x) for x in inputs))
        if self.vjp is None and self.jvp is None:
            return traced                # no rule to attach: the table differentiates it
        return Value(Primitive(self.name, node_of(traced), inputs))

    def ensure(self, node: Primitive, direction: str) -> None:
        """Trace the rule of ``direction`` for ``node``'s signature into a template, once, and register it as data."""
        fn = getattr(self, direction)
        if fn is None or rules.has_rule(KINDS, node, direction):
            return                       # a missing direction is refused by raptor, by name
        ins = [rules.placeholder(f"in{i}", x.ttype) for i, x in enumerate(node.inputs)]
        if direction == "vjp":
            got = call_lowered(fn, *(Value(x) for x in ins), Value(rules.placeholder("cot", node.ttype)))
            got = tuple(got) if isinstance(got, (tuple, list)) else (got,)
            templates = tuple(node_of(v) for v in got)
        else:
            dots = [rules.placeholder(f"dot{i}", x.ttype) for i, x in enumerate(node.inputs)]
            templates = node_of(call_lowered(fn, *(Value(x) for x in ins), *(Value(d) for d in dots)))
        rules.register_rule(KINDS, self.name, rules.signature_of(node), direction, templates)


def vjp_contributions(node: Primitive, g: Node) -> tuple:
    """``((input, adjoint), …)`` for one boundary node, from the registered rule."""
    _REGISTRY[node.name].ensure(node, "vjp") if node.name in _REGISTRY else None
    return rules.vjp_contributions(KINDS, node, g)


def jvp_tangent(node: Primitive, supplied: Sequence) -> Node:
    """The output's tangent for one boundary node, from the registered rule."""
    _REGISTRY[node.name].ensure(node, "jvp") if node.name in _REGISTRY else None
    return rules.jvp_tangent(KINDS, node, supplied)


def primitive(name: Callable | str | None = None, *, vjp: Callable | None = None,
              jvp: Callable | None = None) -> Callable:
    """Register a custom primitive, named ``name`` or after the function.

    Used as a decorator on the forward::

        @hawk.primitive("softplus", vjp=lambda x, bar: bar / (1.0 + exp(-x)))
        def softplus(x):
            return log(1.0 + exp(x))

    ``@hawk.primitive`` alone names it ``softplus`` after the function. The
    decorated object is a :class:`PrimitiveDef`, callable from a kernel body
    like a free function. With neither rule it is a plain inline helper whose
    derivative goes through its forward.

    Registering the same function again (a notebook cell run twice) is a
    no-op that gives back the first registration. A different function under a
    taken name is refused: a collision would make two kernels' walk digests
    agree while their derivatives disagreed."""
    def register(fn: Callable, label: str | None) -> PrimitiveDef:
        label = label or getattr(fn, "__name__", "")
        if not label:
            raise HawkError(
                "primitive(): a primitive is registered by NAME — the name is the "
                "boundary node's identity in the walk's digest")
        definition = PrimitiveDef(label, fn, vjp, jvp)
        with _REGISTRY_LOCK:
            held = _REGISTRY.setdefault(label, definition)
            if held is definition:
                try:
                    rules.declare_primitive(KINDS, label, ext="user", directions=tuple(
                        d for d, f in (("vjp", vjp), ("jvp", jvp)) if f is not None))
                except HawkError:
                    del _REGISTRY[label]
                    raise
        if held is not definition and not _same_function(held.forward, fn):
            raise HawkError(
                f"primitive {label!r} is already registered (its forward is "
                f"{held.forward!r}); a second registration under the same name is "
                "refused because the name is what identifies the boundary node in "
                "Walk.digest — two primitives sharing it would give two different "
                "kernels the same content hash, and the cache would serve one for "
                "the other"
            )
        return held
    if callable(name):                                   # bare `@primitive`
        return register(name, None)
    return lambda fn: register(fn, name)


def _same_function(a: Callable, b: Callable) -> bool:
    """Whether ``b`` is ``a`` again: the same function, or one defined again from
    the same text (a notebook cell run twice)."""
    if a is b:
        return True
    ca, cb = getattr(a, "__code__", None), getattr(b, "__code__", None)
    if ca is None or cb is None or a.__qualname__ != b.__qualname__:
        return False
    try:
        held = ([c.cell_contents for c in a.__closure__ or ()]
                == [c.cell_contents for c in b.__closure__ or ()])
    except ValueError:                           # an unbound cell
        held = False
    return held and ca.co_code == cb.co_code and ca.co_consts == cb.co_consts \
        and ca.co_names == cb.co_names
