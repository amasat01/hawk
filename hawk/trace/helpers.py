# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Plain ``def`` helpers called from a kernel.

A user function called with traced values gets the kernel body's own
control-flow rewrite (``if``/``elif``/``else``, ``for``, ``break``, nested
helpers), once per function, and is then inlined: derivatives go through its
body like any other traced code. Library functions (hawk, installed packages,
the standard library) are never rewritten, and neither is a call whose
arguments are all untraced.

Every call to a bare name in a kernel body reaches :func:`call` (the rewrite
puts it there); the check below decides whether the callee is rewritten."""

from __future__ import annotations

import ast
import builtins
import inspect
import sysconfig
import textwrap
import threading
import types
from typing import Any

from ..ir import HawkError
from .value import Value

#: The rewritten form of each helper, by code object.
_REWRITTEN: dict = {}
_REWRITTEN_LIMIT = 4096

#: Helpers that did not trace as plain Python.
_NEEDS_REWRITE: set = set()

#: The code objects of the helpers being traced right now, innermost last.
_ACTIVE = threading.local()

_LIBRARY_ROOTS = ("hawk", "eagle", "raptor", "aether")
_LIBRARY_DIRS = tuple({sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")})
#: The name a rewritten helper is defined under, so a call by its own name reaches the original.
_BODY = "__hawk_helper__"
_CONTROL_FLOW = (ast.If, ast.IfExp, ast.For)


def call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)``. A user helper called with a traced value runs as
    written first: a helper that traces as plain Python (a static ``for`` over a
    list, arithmetic) keeps its text and its IR. One that reaches a traced
    ``if``, ``break`` or ``and`` fails there, and is rewritten like a kernel body
    (once per function) and run again, inlined."""
    if type(fn) is not types.FunctionType or not _traced(args, kwargs) \
            or not _user_function(fn):
        return fn(*args, **kwargs)
    code = fn.__code__
    if code not in _NEEDS_REWRITE:
        try:
            return fn(*args, **kwargs)
        except HawkError:
            if _rewrite(fn) is None:
                raise
            _NEEDS_REWRITE.add(code)
    rewritten = _rewrite(fn)
    if isinstance(rewritten, _Refused):
        rewritten()
    stack = _active()
    if code in stack:
        raise HawkError(
            f"helper {fn.__name__!r} calls itself: a helper is inlined while the "
            "kernel is traced, so recursion has no end to unroll — write the "
            "repetition as a bounded `for`")
    stack.append(code)
    try:
        return rewritten(*args, **kwargs)
    finally:
        stack.pop()


class _Refused:
    """The rewrite's refusal, raised when the helper needs the rewrite."""

    def __init__(self, refusal: HawkError) -> None:
        self.refusal = refusal

    def __call__(self) -> None:
        raise self.refusal


def _active() -> list:
    try:
        return _ACTIVE.stack
    except AttributeError:
        _ACTIVE.stack = []
        return _ACTIVE.stack


def _traced(args: tuple, kwargs: dict) -> bool:
    return any(isinstance(v, Value) or (isinstance(v, (tuple, list))
                                        and any(isinstance(w, Value) for w in v))
               for v in (*args, *kwargs.values()))


def traced(fn: types.FunctionType) -> types.FunctionType:
    """Mark a function of this package whose body uses control flow on traced
    values, so :func:`call` rewrites it like a user helper."""
    fn.__hawk_traced__ = True
    return fn


def _user_function(fn: types.FunctionType) -> bool:
    code = fn.__code__
    if fn.__name__ == "<lambda>" or hasattr(fn, "__wrapped__"):
        return False
    if getattr(fn, "__hawk_traced__", False):
        return True                      # a hawk function written to be traced like a kernel body
    if (fn.__module__ or "").split(".")[0] in _LIBRARY_ROOTS:
        return False
    return not any(code.co_filename.startswith(d) for d in _LIBRARY_DIRS) \
        and "site-packages" not in code.co_filename


def _rewrite(fn: types.FunctionType) -> types.FunctionType | None:
    """The rewritten helper over ``fn``'s globals, closure and defaults, or
    ``None`` when it is left as it is: a library function, one with no
    control flow, one whose source is not available."""
    code = fn.__code__
    entry = _REWRITTEN.get(code, False)
    if entry is False:
        entry = _compile(fn) if _user_function(fn) else None
        if len(_REWRITTEN) >= _REWRITTEN_LIMIT:
            _REWRITTEN.clear()
        _REWRITTEN[code] = entry
    if entry is None:
        return None
    if isinstance(entry, HawkError):
        return _Refused(entry)
    from .kernel import _scope

    scope = _scope(fn, {})
    install(scope)
    exec(entry, scope)  # the author's rewritten helper
    out = scope[_BODY]
    out.__name__ = out.__qualname__ = fn.__name__
    out.__defaults__, out.__kwdefaults__ = fn.__defaults__, fn.__kwdefaults__
    return out


def _compile(fn: types.FunctionType):
    """The compiled rewrite of ``fn``, a :class:`HawkError` when the kernel
    rewrite refuses it, ``None`` when it needs none."""
    from . import astpass

    try:
        source, first = inspect.getsourcelines(fn)
        tree = ast.parse(textwrap.dedent("".join(source)))
    except (OSError, TypeError, SyntaxError):
        return None
    fn_def = tree.body[0]
    if not isinstance(fn_def, ast.FunctionDef) or not any(
            isinstance(n, _CONTROL_FLOW) or _calls_a_name(n)
            for st in fn_def.body for n in ast.walk(st)):
        return None
    ast.increment_lineno(tree, first - 1)
    filename = inspect.getsourcefile(fn) or "<helper>"
    fn_def.decorator_list, fn_def.returns = [], None
    fn_def.name = _BODY               # the helper's own name still means the original
    args = fn_def.args
    for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
        if a is not None:
            a.annotation = None
    args.defaults, args.kw_defaults = [], [None] * len(args.kwonlyargs)
    try:
        last = fn_def.body[-1]
        rewritten = astpass.transform(fn_def, filename, frozenset(), frozenset(),
                                      isinstance(last, ast.Return) and last.value is not None)
    except HawkError as refusal:
        return refusal
    return compile(ast.Module(body=[rewritten], type_ignores=[]), filename, "exec")


def _calls_a_name(node: ast.AST) -> bool:
    """A call to a bare name that is not a builtin: possibly a nested helper."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and not hasattr(builtins, node.func.id))


def install(scope: dict) -> None:
    """Bind the names the rewritten code calls into ``scope``."""
    from . import astpass, reduce

    scope[astpass.SELECT_HELPER] = astpass.merge
    scope[astpass.LOOP_HELPER] = astpass.open_loop
    scope[astpass.SETITEM_HELPER] = astpass.set_component
    scope[astpass.MAXMIN_HELPER] = reduce.maxmin
    scope[astpass.SCATTER_HELPER] = reduce.scatter_store
    scope[astpass.RANGE_HELPER] = astpass.static_range
    scope[astpass.CALL_HELPER] = call
