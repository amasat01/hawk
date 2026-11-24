# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@hawk.function`` and :func:`native`: typed functions that are written once in the generated text and called.

A plain ``def`` helper is inlined at every use (:mod:`hawk.trace.helpers`). A ``def`` decorated with
:func:`function` is recorded instead as ONE ``Function`` of the kernel's function table, with a ``Call`` per use, so a large
helper used many times costs its text once. The parameters are annotated with payload types; the body is traced once
per kernel from placeholders of those types and may read nothing but its parameters. :func:`native` declares the same kind
of function with the author's own text per target instead of a traced body.

A function has two derivative directions. Each is a function of the same table: the author's rule, traced once per kernel
into ordinary IR, or, for a traced function without one, the function :mod:`hawk.diff.functions` derives from the body. A
forward rule is ONE function of the reads and a tangent per read; the reverse rules are one function per read, each of
the reads and the adjoint of the result."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .. import _ir
from ..errors import HawkError
from .helpers import call_lowered
from .lane_axes import current_context, trace_context
from .value import Value, node_of

#: How many contexts a function remembers its recording for.
_RECORDED_LIMIT = 256

#: The functions whose bodies are being traced right now, innermost last (a function that reaches itself has no end).
_ACTIVE = threading.local()

#: The two derivative directions, with the word an author passes for each.
DIRECTIONS = ("jvp", "vjp")


class FunctionDef:
    """A definition decorated with :func:`function`; calling it in a kernel body records a ``Call``."""

    def __init__(
        self, fn: Callable, jvp: Callable | None = None, vjp: Callable | None = None
    ) -> None:
        self.fn = fn
        self.__name__ = self.__qualname__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.params = self._declared_params()
        self.result = self._declared_result()
        self._recorded: dict = {}
        self._install_rules(jvp, vjp)

    def _install_rules(self, jvp: Callable | None, vjp: Callable | None) -> None:
        """Keep the author's rules as the derivative functions they trace into; a direction without one stays empty."""
        for direction, rule in (("jvp", jvp), ("vjp", vjp)):
            if rule is not None and not callable(rule):
                raise HawkError(
                    f"function {self.__name__!r}: {direction} must be a callable of the reads and "
                    f"{'a tangent per read' if direction == 'jvp' else 'the adjoint of the result'}, got {rule!r}"
                )
        #: The author's derivative functions by direction: one for ``jvp``, one per read for ``vjp``, none when not given.
        self.author = {
            "jvp": () if jvp is None else (RuleDef(self, "jvp", jvp, 0),),
            "vjp": ()
            if vjp is None
            else tuple(RuleDef(self, "vjp", vjp, i) for i in range(len(self.params))),
        }

    def _declared_params(self) -> tuple:
        """``((name, type), ...)`` from the annotations; a parameter without a payload type is refused by name."""
        out = []
        for name, p in inspect.signature(self.fn, eval_str=True).parameters.items():
            if p.kind is not p.POSITIONAL_OR_KEYWORD or p.default is not p.empty:
                raise HawkError(
                    f"function {self.__name__!r}: parameter {name!r} must be a plain positional "
                    "parameter with no default"
                )
            if not isinstance(p.annotation, _ir.TensorType):
                raise HawkError(
                    f"function {self.__name__!r}: parameter {name!r} needs a payload type annotation "
                    "(Scalar, Int32, Vector[W], Matrix[R, C], ...), got "
                    f"{'none' if p.annotation is p.empty else repr(p.annotation)}"
                )
            out.append((name, p.annotation))
        return tuple(out)

    def _declared_result(self) -> _ir.TensorType | None:
        """The type of the ``->`` annotation, which the traced result must equal; none when it is not written."""
        returns = inspect.signature(self.fn, eval_str=True).return_annotation
        if returns is inspect.Signature.empty:
            return None
        if not isinstance(returns, _ir.TensorType):
            raise HawkError(
                f"function {self.__name__!r}: the return annotation must be a payload type, got "
                f"{returns!r}"
            )
        return returns

    def __call__(self, *args: Any) -> Value:
        """Record a ``Call`` of this function over ``args`` (a body traced once per context)."""
        if len(args) != len(self.params):
            raise HawkError(
                f"function {self.__name__!r} takes {len(self.params)} argument(s), got {len(args)}"
            )
        operands = [node_of(a) for a in args]
        context = operands[0].ctx if operands else current_context()
        _, node, name, digest = self._held(context)
        result = _ir.type_of(node)
        return Value(
            _ir.make_node(
                "Call",
                (result.shape, result.dtype, result.tag),
                operands,
                context=context,
                function=(name, digest),
            )
        )

    def _held(self, context: Any) -> tuple:
        """``(context, Function node, name, digest)`` of this function in ``context``, declaring it on the first use."""
        held = self._recorded.get(id(context))
        if held is None:
            node = self._declare(context)
            held = (context, node, self.__name__, node.field_text("digest"))
            if len(self._recorded) >= _RECORDED_LIMIT:
                self._recorded.clear()
            # the entry holds the context, so its identity is never reused while it is listed
            self._recorded[id(context)] = held
        return held

    def _param_nodes(self, context: Any) -> list:
        """One ``Param`` node per declared parameter, the leaves of the function's declaration."""
        return param_nodes(context, self.params)

    def _rule_nodes(self, context: Any) -> dict:
        """The Function nodes of the author's rules in ``context``, as the keywords ``_ir.function`` takes them."""
        forward, reverse = (
            tuple(d._held(context)[1] for d in self.author[k]) for k in DIRECTIONS
        )
        return {"jvp": forward[0] if forward else None, "vjp": reverse}

    def _body(self, context: Any, values: list) -> Any:
        """The value this function's body computes from its parameters ``values``."""
        return call_lowered(self.fn, *values)

    def _declare(self, context: Any) -> Any:
        """The ``Function`` node of this function in ``context``: its body traced from the parameters, and its rules."""
        active = _active()
        if self in active:
            raise HawkError(
                f"function {self.__name__!r} calls itself: a function is traced once from its "
                "parameters, so recursion has no end — write the repetition as a bounded `for`"
            )
        active.append(self)
        try:
            params = self._param_nodes(context)
            with trace_context(context):
                body = node_of(self._body(context, [Value(p) for p in params]))
            rules = self._rule_nodes(context)
        finally:
            active.pop()
        result = _ir.type_of(body)
        if self.result is not None and self.result != result:
            raise HawkError(
                f"function {self.__name__!r} is declared to return {self.result!r} but its body "
                f"computes {result!r}"
            )
        return _ir.function(context, self.__name__, params, body, (), result, **rules)


class NativeFunctionDef(FunctionDef):
    """A function whose body is the author's text per target; calling it in a kernel body records a ``Call``."""

    def __init__(
        self,
        name: str,
        reads: Mapping[str, _ir.TensorType],
        returns: _ir.TensorType,
        text: Mapping[str, str],
        jvp: Callable | None = None,
        vjp: Callable | None = None,
    ) -> None:
        self.__name__ = self.__qualname__ = name
        self.params = tuple(reads.items())
        self.result = _ir.TensorType(returns.shape, returns.dtype, returns.tag)
        #: The author's body text keyed by target name; the key is carried as data and read only by the library that owns it.
        self.text = tuple(sorted(text.items()))
        self._recorded = {}
        self._install_rules(jvp, vjp)

    def _declare(self, context: Any) -> Any:
        """The ``Function`` node of this function in ``context``: its texts and its rules, and no traced body."""
        return _ir.function(
            context,
            self.__name__,
            self._param_nodes(context),
            None,
            self.text,
            self.result,
            **self._rule_nodes(context),
        )


class RuleDef(FunctionDef):
    """One derivative function of ``owner``: the forward rule, or the reverse rule of read ``index``.

    It takes the owner's reads and then, for the forward rule, one tangent per read and, for a reverse rule, the adjoint of the
    result; it returns the tangent of the result, or the gradient of read ``index``. The author's rule, one callable, is traced
    into it, once per context, when the owner is declared; a traced owner without rules is derived by
    :mod:`hawk.diff.functions` from its Function node."""

    def __init__(
        self, owner: FunctionDef, direction: str, rule: Callable | None, index: int
    ) -> None:
        label, params, self.result = rule_signature(
            owner.__name__, owner.params, owner.result, direction, index
        )
        self.__name__ = self.__qualname__ = label
        self.params = params
        self.owner, self.direction, self.index, self.rule = (
            owner,
            direction,
            index,
            rule,
        )
        self._recorded = {}
        self._install_rules(None, None)

    def _body(self, context: Any, values: list) -> Any:
        """The author's rule traced over the parameters; a reverse rule's tuple of gradients is cut to this read's."""
        got = call_lowered(self.rule, *values)
        if self.direction == "jvp":
            return got
        grads = tuple(got) if isinstance(got, (tuple, list)) else (got,)
        if len(grads) != len(self.owner.params):
            raise HawkError(
                f"function {self.owner.__name__!r}: its vjp rule returned {len(grads)} gradient(s) for "
                f"{len(self.owner.params)} read(s); a reverse rule returns one gradient per read, in parameter order"
            )
        return grads[self.index]


def param_nodes(context: Any, params: Sequence) -> list:
    """One ``Param`` node per ``(name, type)`` of ``params``, the leaves of a function's declaration."""
    return [
        _ir.make_node(
            "Param",
            (t.shape, t.dtype, t.tag),
            context=context,
            name=n,
            access="read",
        )
        for n, t in params
    ]


def rule_signature(
    name: str, reads: Sequence, result: _ir.TensorType, direction: str, index: int
) -> tuple:
    """``(label, params, result type)`` of the derivative function of ``direction`` of the function ``name``.

    The parameters are the reads, then a tangent per read (forward) or the adjoint of the result (reverse); the forward rule
    returns the tangent of the result and reverse rule ``index`` the gradient of read ``index``. A tail name is led by
    underscores until it is none of the reads' names."""
    taken = {n for n, _ in reads}

    def fresh(stem: str) -> str:
        while stem in taken:
            stem = "_" + stem
        return stem

    if direction == "jvp":
        tail = tuple((fresh(f"dot_{n}"), t) for n, t in reads)
        return f"{name}_jvp", (*reads, *tail), result
    tail = ((fresh("bar_result"), result),)
    return f"{name}_vjp_{reads[index][0]}", (*reads, *tail), reads[index][1]


def _active() -> list:
    try:
        return _ACTIVE.stack
    except AttributeError:
        _ACTIVE.stack = []
        return _ACTIVE.stack


def function(
    fn: Callable | None = None,
    *,
    jvp: Callable | None = None,
    vjp: Callable | None = None,
) -> FunctionDef | Callable:
    """Declare ``fn`` a function: written once in the generated text and called, not inlined at each use.

    Used as a decorator on a ``def`` whose parameters carry payload types and whose body reads nothing but them::

        @hawk.function
        def drag(v: hawk.Vector[3], rho: hawk.Scalar) -> hawk.Vector[3]:
            return v * (rho * m.norm(v))

    A call in a kernel body, ``a = drag(v, rho)``, records one call; a plain helper without the decorator stays
    inlined. A wrong number of arguments or a wrong argument type is refused by the function's name.

    A traced function is differentiated through its body, in both directions, with no more from you. ``jvp=`` and
    ``vjp=`` replace that with your own rules: ``jvp(reads..., dot_reads...)`` returns the tangent of the result and
    ``vjp(reads..., bar_result)`` returns one gradient per read (a tuple when there are several); each is traced once into
    ordinary functions that the derivative kernels call."""
    if fn is None:
        return lambda f: FunctionDef(f, jvp, vjp)
    return FunctionDef(fn, jvp, vjp)


def native(
    name: str,
    *,
    reads: Mapping[str, _ir.TensorType],
    returns: _ir.TensorType,
    text: Mapping[str, str],
    jvp: Callable | None = None,
    vjp: Callable | None = None,
) -> NativeFunctionDef:
    """Declare a function whose body is your own text for each target, written once in the generated text and called.

    ``reads`` names the parameters and their payload types, ``returns`` is the type of the one result, and ``text`` maps
    a target name to the BODY of the function: the statements between its braces, which see the parameters and nothing else::

        fma_sq = hawk.native(
            "fma_sq",
            reads={"x": hawk.Scalar, "y": hawk.Scalar},
            returns=hawk.Scalar,
            text={"host": "return std::fma(x, x, y);"},
        )

    A call in a kernel body, ``out = fma_sq(a, b)``, records one call. A target with no text is refused by the function's
    and the target's names when the kernel is built for it.

    The text is opaque, so a derivative of a call needs your rule: ``jvp(reads..., dot_reads...)`` returns the tangent of
    the result and ``vjp(reads..., bar_result)`` one gradient per read (a tuple when there are several). A rule is an
    ordinary traced callable, and may be another native function; a direction asked of a call without its rule is refused
    by the function's name."""
    if not name.isidentifier():
        raise HawkError(f"native function {name!r}: the name must be an identifier")
    for param, declared in reads.items():
        if not param.isidentifier() or not isinstance(declared, _ir.TensorType):
            raise HawkError(
                f"native function {name!r}: parameter {param!r} needs an identifier name and a payload "
                f"type (Scalar, Int32, Vector[W], Matrix[R, C], ...), got {declared!r}"
            )
    if not isinstance(returns, _ir.TensorType):
        raise HawkError(
            f"native function {name!r}: returns needs a payload type, got {returns!r}"
        )
    if not text or not all(
        isinstance(k, str) and k and isinstance(v, str) for k, v in text.items()
    ):
        raise HawkError(
            f"native function {name!r}: text maps a target name to the body text, with at least one target"
        )
    return NativeFunctionDef(name, reads, returns, text, jvp, vjp)
