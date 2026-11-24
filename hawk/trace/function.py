# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@hawk.function`` and :func:`native`: typed functions that are written once in the generated text and called.

A plain ``def`` helper is inlined at every use (:mod:`hawk.trace.helpers`). A ``def`` decorated with
:func:`function` is recorded instead as ONE ``Function`` of the kernel's function table, with a ``Call`` per use, so a large
helper used many times costs its text once. The parameters are annotated with payload types; the body is traced once
per kernel from placeholders of those types and may read nothing but its parameters. :func:`native` declares the same kind
of function with the author's own text per target instead of a traced body."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Mapping
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


class FunctionDef:
    """A definition decorated with :func:`function`; calling it in a kernel body records a ``Call``."""

    def __init__(self, fn: Callable) -> None:
        self.fn = fn
        self.__name__ = self.__qualname__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.params = self._declared_params()
        self.result = self._declared_result()
        #: ``(context, name, digest, result type)`` of this function in each context it was recorded in, by the context's
        #: identity; the entry holds the context, so an identity is never reused while it is listed.
        self._recorded: dict = {}

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
        name, digest, result = self._record(context)
        return Value(
            _ir.make_node(
                "Call", result, operands, context=context, function=(name, digest)
            )
        )

    def _record(self, context: Any) -> tuple:
        """The ``(name, digest, result type)`` of this function in ``context``, declaring it on the first use."""
        held = self._recorded.get(id(context))
        if held is not None:
            return held[1:]
        node = self._declare(context)
        result = _ir.type_of(node)
        if len(self._recorded) >= _RECORDED_LIMIT:
            self._recorded.clear()
        held = (
            context,
            self.__name__,
            node.field_text("digest"),
            (result.shape, result.dtype, result.tag),
        )
        self._recorded[id(context)] = held
        return held[1:]

    def _param_nodes(self, context: Any) -> list:
        """One ``Param`` node per declared parameter, the leaves of the function's declaration."""
        return [
            _ir.make_node(
                "Param",
                (t.shape, t.dtype, t.tag),
                context=context,
                name=n,
                access="read",
            )
            for n, t in self.params
        ]

    def _declare(self, context: Any) -> Any:
        """The ``Function`` node of this function in ``context``: its body traced from the parameters."""
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
                body = node_of(call_lowered(self.fn, *(Value(p) for p in params)))
        finally:
            active.pop()
        result = _ir.type_of(body)
        if self.result is not None and self.result != result:
            raise HawkError(
                f"function {self.__name__!r} is declared to return {self.result!r} but its body "
                f"computes {result!r}"
            )
        return _ir.function(context, self.__name__, params, body, (), result)


class NativeFunctionDef(FunctionDef):
    """A function whose body is the author's text per target; calling it in a kernel body records a ``Call``."""

    def __init__(
        self,
        name: str,
        reads: Mapping[str, _ir.TensorType],
        returns: _ir.TensorType,
        text: Mapping[str, str],
    ) -> None:
        self.__name__ = self.__qualname__ = name
        self.params = tuple(reads.items())
        self.result = _ir.TensorType(returns.shape, returns.dtype, returns.tag)
        #: The author's body text keyed by target name; the key is carried as data and read only by the library that owns it.
        self.text = tuple(sorted(text.items()))
        self._recorded = {}

    def _declare(self, context: Any) -> Any:
        """The ``Function`` node of this function in ``context``: its texts, and no traced body."""
        return _ir.function(
            context,
            self.__name__,
            self._param_nodes(context),
            None,
            self.text,
            self.result,
        )


def _active() -> list:
    try:
        return _ACTIVE.stack
    except AttributeError:
        _ACTIVE.stack = []
        return _ACTIVE.stack


def function(fn: Callable) -> FunctionDef:
    """Declare ``fn`` a function: written once in the generated text and called, not inlined at each use.

    Used as a decorator on a ``def`` whose parameters carry payload types and whose body reads nothing but them::

        @hawk.function
        def drag(v: hawk.Vector[3], rho: hawk.Scalar) -> hawk.Vector[3]:
            return v * (rho * m.norm(v))

    A call in a kernel body, ``a = drag(v, rho)``, records one call; a plain helper without the decorator stays
    inlined. A wrong number of arguments or a wrong argument type is refused by the function's name."""
    return FunctionDef(fn)


def native(
    name: str,
    *,
    reads: Mapping[str, _ir.TensorType],
    returns: _ir.TensorType,
    text: Mapping[str, str],
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
    and the target's names when the kernel is built for it."""
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
    return NativeFunctionDef(name, reads, returns, text)
