# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The graph builder: ``@hawk.Graph.builder`` runs a Python function once, and what it means becomes a :class:`hawk.Graph`.

An AST pass rewrites the ``if`` / ``while`` / guarded ``for`` / ``match`` statements of the body into calls on a control object.
While a builder runs, a kernel call is recorded instead of launched (:func:`current` is what :meth:`hawk.plans.Plan.bound` asks),
and a condition that is a one-element ``int32`` or ``uint32`` array (or a :class:`hawk.SkipGuard`) becomes a conditional of the
graph, read at every replay; any other condition is a Python value, decided once, here. Nothing of the user's Python runs at a
replay.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import copy
import functools
import inspect
import itertools
import linecache
import textwrap
import warnings
from typing import NamedTuple

from .errors import HawkError

__all__ = ["Recorder", "attach", "current", "make_builder"]

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("hawk_graph_recorder", default=None)
_COUNT = itertools.count()
#: The ``match`` statement, which Python has from 3.10; an empty tuple before it, so no statement is one.
_MATCH = getattr(ast, "Match", ())
#: The dtypes a condition cell may have: the library tests one element of an int32 or uint32 plane.
_CELL_DTYPES = ("int32", "uint32")


def current():
    """The :class:`Recorder` of the builder or block running now, else ``None``."""
    return _ACTIVE.get()


@contextlib.contextmanager
def _recording(recorder):
    """Make ``recorder`` the one :func:`current` returns for the length of the block."""
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


class Recorder:
    """One run of a builder (or of a block over a hand-made graph): where kernel calls are recorded to."""

    def __init__(self, graph, kind: str, name: str) -> None:
        """``graph`` is the graph a block appends to, ``None`` for a builder, which makes its own; ``kind`` is ``"builder"`` or
        ``"block"``."""
        self.graph, self.kind, self.name = graph, kind, name
        self.block_depth = 0
        self._labels: dict = {}

    @property
    def reverse(self) -> bool:
        """Whether the graph being written is kept for :func:`hawk.reverse`, so its plans must carry their vjp."""
        return self.graph is not None and self.graph._records

    def record(self, bound, kernel: str) -> None:
        """Record the launch of ``bound`` (a call of kernel ``kernel``) into the scope being written, under a label that is the
        kernel's name, numbered from the second call."""
        graph = self._graph()
        count = self._labels[kernel] = self._labels.get(kernel, 0) + 1
        from .graph import _Launch  # graph imports this module first

        bound.join(graph._store_over(bound.door))
        graph._append(_Launch(bound, kernel if count == 1 else f"{kernel}#{count}"))

    def record_step(self, bound) -> None:
        """Record the call of the host step ``bound`` into the scope being written."""
        graph = self._graph()
        from .graph import _HostCall  # graph imports this module first

        graph._store_over(bound.door)
        graph._append(_HostCall(bound))

    def store(self, door):
        """The plane store of the graph being written, which the plans of its launches are bound through, over ``door``: the
        door of the memory the launch's planes live in, which becomes the graph's door at the first launch."""
        return self._graph()._store_over(door)

    def _graph(self):
        """The graph being written, or a refusal: a kernel call comes after ``g = hawk.Graph()``."""
        if self.graph is None:
            raise HawkError(f"{self.name}: make the graph (`g = hawk.Graph()`) before the first kernel call")
        return self.graph


def attach(graph) -> None:
    """Make ``graph``, just constructed, the one the running builder records into; refuse a second, or one inside a block."""
    recorder = current()
    if recorder is None:
        return
    if recorder.kind == "block" or recorder.block_depth:
        raise HawkError(f"{recorder.name}: a block appends to the graph it is given; a new graph is made in a builder")
    if recorder.graph is not None:
        raise HawkError(f"{recorder.name}: a builder makes ONE graph")
    recorder.graph = graph


# -- conditions ------------------------------------------------------------------------------------------------------------- #
class _Not:
    """``not c`` of a cell: the condition, negated."""

    def __init__(self, inner) -> None:
        """Hold the negated condition."""
        self.inner = inner


class _Test(NamedTuple):
    """What a conditional tests: ``cell[index]`` against ``baseline[baseline_index]`` (against 0 without a baseline)."""

    cell: object
    index: int
    baseline: object
    baseline_index: int
    negate: bool

    def negated(self) -> _Test:
        """The test that holds when this one does not."""
        return self._replace(negate=not self.negate)


def _test_of(value, where: str):
    """The :class:`_Test` a condition ``value`` stands for, or a Python ``bool`` when it is decided here, at build."""
    from .graph import SkipGuard

    if isinstance(value, _Not):
        inner = _test_of(value.inner, where)
        return (not inner) if isinstance(inner, bool) else inner.negated()
    if isinstance(value, SkipGuard):
        _check_cell(value.count_arr, where, value.count_idx)
        if value.baseline_arr is not None:
            _check_cell(value.baseline_arr, where, value.baseline_idx)
        return _Test(value.count_arr, value.count_idx, value.baseline_arr, value.baseline_idx, value.negate)
    if getattr(value, "ndim", 0) > 0:
        _check_cell(value, where, 0, one=True)
        return _Test(value, 0, None, 0, False)
    return bool(value)


def _check_cell(cell, where: str, index: int = 0, one: bool = False) -> None:
    """Refuse a condition array that is not an ``int32`` or ``uint32`` array holding the element ``index`` (a bare condition is
    a one-element array), by name."""
    shape, dtype = tuple(getattr(cell, "shape", ())), str(getattr(cell, "dtype", ""))
    sized = shape == (1,) if one else len(shape) == 1 and 0 <= index < shape[0]
    if not sized or dtype not in _CELL_DTYPES:
        raise HawkError(f"{where}: a graph condition is {'a one-element' if one else 'an element of an'} int32 or uint32 array "
                        f"(or a Python value); got shape {shape}, index {index} and dtype {dtype}")


class _Control:
    """What an ``if`` chain, a loop or a ``match`` becomes while a builder runs: a Python decision made once, or, when the
    condition is a cell, a conditional recorded into the graph."""

    device = False

    def __init__(self, recorder: Recorder, where: str, leaks: tuple) -> None:
        """``leaks`` are the names assigned from a call in the body and read after it."""
        self.recorder, self.where, self.leaks = recorder, where, leaks
        self.graph = recorder.graph
        self._items: list = []

    def _scope(self, items: list):
        """Record into ``items`` for the length of a ``with`` block."""
        if self.graph is None:
            raise HawkError(f"{self.where}: make the graph (`g = hawk.Graph()`) before control flow on a cell")

        @contextlib.contextmanager
        def scope():
            self.graph._scopes.append(items)
            try:
                yield
            finally:
                self.graph._scopes.pop()

        return scope()

    def refuse_exit(self) -> None:
        """Refuse a ``break``, ``continue`` or ``return`` in a body that is recorded."""
        if self.device:
            raise HawkError(f"{self.where}: `break`, `continue` and `return` cannot be recorded in a graph conditional or loop "
                            "body: leave the body by its condition (`for _ in range(cap): if not c: break` is the guarded form)")

    def _refuse_leaks(self) -> None:
        """Refuse a name a recorded arm assigns from a kernel call and the code after the construct reads."""
        if self.device and self.leaks:
            raise HawkError(f"{self.where}: {', '.join(self.leaks)} is assigned from a kernel call inside graph control flow and "
                            "read after it; the arm that runs is decided at replay. Make the array first and pass it to the "
                            "call (`k(x, y=y)`), which writes it in place")


class _IfControl(_Control):
    """An ``if`` / ``elif`` / ``else`` chain."""

    def __init__(self, recorder: Recorder, conditions: tuple, has_else: bool, where: str, leaks: tuple) -> None:
        """``conditions`` are functions, evaluated in order as Python evaluates them, or all of them when the first is a cell."""
        super().__init__(recorder, where, leaks)
        self._conditions, self._has_else = conditions, has_else
        first = _test_of(conditions[0](), where)
        self.device = not isinstance(first, bool)
        if self.device:
            tests = [first] + [_test_of(condition(), where) for condition in conditions[1:]]
            if any(isinstance(test, bool) for test in tests):
                raise HawkError(f"{where}: an if / elif chain mixes a cell and a Python value; make every condition a cell, or "
                                "none")
            self._tests = tests
            self._arms = [[] for _ in range(len(conditions) + has_else)]
        else:
            self._chosen = self._choose(first)

    def _choose(self, first: bool) -> int | None:
        """The arm a Python chain takes: the first whose condition holds, else the ``else`` arm if there is one."""
        if first:
            return 0
        for position, condition in enumerate(self._conditions[1:], 1):
            value = _test_of(condition(), self.where)
            if not isinstance(value, bool):
                raise HawkError(f"{self.where}: an if / elif chain mixes a cell and a Python value; make every condition a "
                                "cell, or none")
            if value:
                return position
        return len(self._conditions) if self._has_else else None

    def runs(self, arm: int) -> bool:
        """Whether the body of ``arm`` is to execute now: always when recording it, else only the chosen one."""
        return True if self.device else self._chosen == arm

    def arm(self, arm: int):
        """The ``with`` block of the body of ``arm``."""
        return self._scope(self._arms[arm]) if self.device else contextlib.nullcontext()

    def end(self) -> None:
        """Close the chain: a recorded one goes into the graph."""
        if not self.device:
            return
        self._refuse_leaks()
        item = _chain(self._tests, self._arms, self._has_else)
        if item is not None:
            self.graph._append(item)


def _chain(tests: list, arms: list, has_else: bool):
    """The conditional item of an ``if`` chain: IF when it has no ``else``, IF_ELSE nested one level per ``elif``; ``None`` when
    nothing is recorded in it."""
    from .graph import _IF, _IF_ELSE, _Cond

    test = tests[0]
    if len(tests) == 1:
        other = arms[1] if has_else else []
    else:
        nested = _chain(tests[1:], arms[1:], has_else)
        other = [] if nested is None else [nested]
    body = arms[0]
    if not body and not other:
        return None
    if not other:
        return _Cond(_IF, test.cell, [body], test.negate, 0, test.index, test.baseline, test.baseline_index)
    if not body:
        return _Cond(_IF, test.cell, [other], not test.negate, 0, test.index, test.baseline, test.baseline_index)
    return _Cond(_IF_ELSE, test.cell, [body, other], test.negate, 0, test.index, test.baseline, test.baseline_index)


class _MatchControl(_Control):
    """A ``match`` on an integer subject."""

    def __init__(self, recorder: Recorder, subject, labels: tuple, has_default: bool, where: str, leaks: tuple) -> None:
        """``labels`` are the integer literals of the cases, in order."""
        super().__init__(recorder, where, leaks)
        value = subject()
        self.device = getattr(value, "ndim", 0) > 0
        if self.device:
            _check_cell(value, where, 0, one=True)
            self._cell = value
            self._labels = tuple(labels)
            #: the cases 0, 1, 2, ... in order and no default are one SWITCH, which selects the arm by the cell's value
            self._switch = not has_default and self._labels == tuple(range(len(labels)))
            self._has_default = has_default
            self._arms = [[] for _ in range(len(labels) + (has_default and not self._switch))]
        else:
            self._chosen = next((position for position, label in enumerate(labels) if value == label),
                                len(labels) if has_default else None)

    def runs(self, arm: int) -> bool:
        """Whether the body of ``arm`` is to execute now."""
        return True if self.device else self._chosen == arm

    def arm(self, arm: int):
        """The ``with`` block of the body of ``arm``."""
        return self._scope(self._arms[arm]) if self.device else contextlib.nullcontext()

    def end(self) -> None:
        """Close the match: a recorded one is one SWITCH of the graph, or, with a default arm or other labels, a chain of
        conditionals each testing the cell for equality with a one-element plane that holds a label."""
        if not self.device:
            return
        from .graph import _SWITCH, _Cond

        self._refuse_leaks()
        if not any(self._arms):
            return
        if self._switch:
            self.graph._append(_Cond(_SWITCH, self._cell, self._arms))
            return
        tests = [_Test(self._cell, 0, self._label_plane(label), 0, True) for label in self._labels]
        item = _chain(tests, self._arms, self._has_default)
        if item is not None:
            self.graph._append(item)

    def _label_plane(self, label: int):
        """A one-element plane, in the memory and dtype of the cell, holding ``label``: what the library compares the cell with.
        The conditional keeps it alive."""
        import numpy as np

        dtype = np.dtype(str(self._cell.dtype))
        if not np.can_cast(np.min_scalar_type(label), dtype, casting="safe"):
            raise HawkError(f"{self.where}: the case {label} does not fit the {dtype} cell it is matched against")
        from . import _place

        return _place.Maker.at(_place.device_of(self._cell), self._cell).copy_of(np.array([label], dtype=dtype))


class _LoopControl(_Control):
    """A ``while`` loop, or a ``for _ in range(cap)`` that opens with ``if not c: break``."""

    def __init__(self, recorder: Recorder, condition, cap: int | None, where: str, leaks: tuple) -> None:
        """``condition`` is a function; ``cap`` is the range of a ``for`` (``None`` for a ``while``)."""
        super().__init__(recorder, where, leaks)
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise HawkError(f"{where}: the cap of a graph loop is a whole number of 1 or more, got {cap!r}")
        self.cap = cap
        self._first = _test_of(condition(), where)
        self._condition = condition
        self.device = not isinstance(self._first, bool)
        self._entered = False
        if self.device and cap is None:
            warnings.warn(f"{where}: a graph loop with no cap runs until its cell clears; cap it with `for _ in range(cap): if "
                          "not c: break`", stacklevel=3)

    def next(self) -> bool:
        """The test of a ``while``: a recorded body is entered once, to be recorded; a Python loop is evaluated as usual."""
        if self.device:
            entered, self._entered = self._entered, True
            return not entered
        value = self._first if not self._entered else bool(self._condition())
        self._entered = True
        return value

    def iterate(self):
        """The iterations of a ``for``: one, to record the body, or the Python range."""
        return iter((0,)) if self.device else iter(range(self.cap))

    def body(self):
        """The ``with`` block of the loop body."""
        return self._scope(self._items) if self.device else contextlib.nullcontext()

    def end(self) -> None:
        """Close the loop: a recorded one is one WHILE of the graph."""
        if not self.device:
            return
        from .graph import _WHILE, _Cond

        self._refuse_leaks()
        test = self._first
        if self._items:
            self.graph._append(_Cond(_WHILE, test.cell, [self._items], test.negate, self.cap or 0, test.index, test.baseline,
                                     test.baseline_index))


class _Runtime:
    """The object the rewritten code calls (``_hg``)."""

    neg = staticmethod(lambda value: _Not(value) if getattr(value, "ndim", 0) > 0 or _is_guard(value) else (not value))

    @staticmethod
    def if_(conditions, has_else, leaks, line):
        """An :class:`_IfControl` for the chain at ``line``."""
        recorder = _need()
        return _IfControl(recorder, conditions, has_else, recorder.name + f":{line}", leaks)

    @staticmethod
    def match(subject, labels, has_default, leaks, line):
        """A :class:`_MatchControl` for the ``match`` at ``line``."""
        recorder = _need()
        return _MatchControl(recorder, subject, labels, has_default, recorder.name + f":{line}", leaks)

    @staticmethod
    def loop(condition, cap, leaks, line):
        """A :class:`_LoopControl` for the loop at ``line``."""
        recorder = _need()
        return _LoopControl(recorder, condition, cap, recorder.name + f":{line}", leaks)


def _is_guard(value) -> bool:
    """Whether ``value`` is a :class:`hawk.SkipGuard`."""
    from .graph import SkipGuard

    return isinstance(value, SkipGuard)


def _need() -> Recorder:
    """The running recorder: rewritten code runs only under one."""
    recorder = current()
    if recorder is None:
        raise HawkError("a graph builder or block runs inside its decorated call")
    return recorder


_HG = _Runtime()


# -- the AST pass ----------------------------------------------------------------------------------------------------------- #
def _loads(statements) -> set:
    """The names the ``statements`` read."""
    return {node.id for stmt in statements for node in ast.walk(stmt) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _call_targets(statements) -> set:
    """The names the ``statements`` assign from a call (``name = f(...)``)."""
    return {node.targets[0].id for stmt in statements for node in ast.walk(stmt)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)}


def _parse(source: str) -> list:
    """The statements of ``source``."""
    return ast.parse(textwrap.dedent(source)).body


def _lambda(expression) -> ast.Lambda:
    """``lambda: expression``."""
    arguments = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    return ast.Lambda(arguments, expression)


def _exit_guards(statements: list, control: str, in_loop: bool = False) -> list:
    """``statements`` with a ``<control>.refuse_exit()`` before each ``break``, ``continue`` and ``return`` that leaves the
    construct: a recorded body runs once, so leaving it early would silently drop the rest of it."""
    out = []
    for stmt in statements:
        if isinstance(stmt, (ast.Break, ast.Continue)) and not in_loop or isinstance(stmt, ast.Return):
            out.append(ast.copy_location(_parse(f"{control}.refuse_exit()")[0], stmt))
        out.append(_exit_inside(stmt, control, in_loop))
    return out


def _exit_inside(stmt, control: str, in_loop: bool):
    """``stmt`` with the exit guards of its own nested blocks; a nested loop owns its ``break`` and ``continue``."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return stmt
    nested = in_loop or isinstance(stmt, (ast.For, ast.While))
    for field in ("body", "orelse", "finalbody"):
        block = getattr(stmt, field, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            setattr(stmt, field, _exit_guards(block, control, nested))
    for handler in getattr(stmt, "handlers", []):
        handler.body = _exit_guards(handler.body, control, nested)
    for case in getattr(stmt, "cases", []):
        case.body = _exit_guards(case.body, control, nested)
    return stmt


class _Rewriter:
    """Rewrites the control flow of one function body into calls on :class:`_Runtime`."""

    def body(self, statements: list, after: set) -> list:
        """``statements`` rewritten; ``after`` are the names read after them."""
        out = []
        for position, stmt in enumerate(statements):
            out += self.stmt(stmt, _loads(statements[position + 1:]) | after)
        return out

    def stmt(self, stmt, after: set) -> list:
        """The statements ``stmt`` becomes."""
        if isinstance(stmt, ast.If):
            return self.if_(stmt, after)
        if isinstance(stmt, ast.While):
            return self.while_(stmt, after)
        if isinstance(stmt, _MATCH):
            return self.match_(stmt, after)
        if isinstance(stmt, ast.For):
            return self.for_(stmt, after)
        if isinstance(stmt, (ast.With, ast.Try)):
            for field in ("body", "orelse", "finalbody"):
                if hasattr(stmt, field):
                    setattr(stmt, field, self.body(getattr(stmt, field), after))
            for handler in getattr(stmt, "handlers", []):
                handler.body = self.body(handler.body, after)
        return [stmt]

    @staticmethod
    def _condition(test):
        """``test`` with a leading ``not`` made the runtime's, which negates a cell and a guard and a Python value alike."""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return ast.Call(ast.Attribute(ast.Name("_hg", ast.Load()), "neg", ast.Load()), [test.operand], [])
        return test

    def _leaks(self, arms: list, after: set) -> str:
        """The source text of the tuple of names assigned from a call in ``arms`` and read after them."""
        return repr(tuple(sorted(_call_targets([s for arm in arms for s in arm]) & after)))

    def _arms(self, control: str, arms: list, after: set) -> list:
        """The guarded ``if <control>.runs(k): with <control>.arm(k): body`` of each arm."""
        out = []
        for position, arm in enumerate(arms):
            body = _exit_guards(self.body(arm, after), control)
            wrapper = _parse(f"if {control}.runs({position}):\n    with {control}.arm({position}):\n        pass\n")[0]
            wrapper.body[0].body = body
            out.append(wrapper)
        return out

    def if_(self, stmt: ast.If, after: set) -> list:
        """An ``if`` chain."""
        arms, conditions, node = [], [], stmt
        while True:
            conditions.append(self._condition(node.test))
            arms.append(node.body)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                node = node.orelse[0]
            else:
                break
        if node.orelse:
            arms.append(node.orelse)
        control = f"__hg{next(_COUNT)}"
        head = _parse(f"{control} = _hg.if_(CONDITIONS, {bool(node.orelse)}, {self._leaks(arms, after)}, {stmt.lineno})")
        head[0].value.args[0] = ast.Tuple([_lambda(c) for c in conditions], ast.Load())
        return head + self._arms(control, arms, after) + _parse(f"{control}.end()")

    def match_(self, stmt, after: set) -> list:
        """A ``match`` whose cases are integer literals and an optional ``case _``."""
        labels, arms, default = [], [], None
        for case in stmt.cases:
            pattern = case.pattern
            if case.guard is not None:
                raise HawkError(f"line {pattern.lineno}: a match case with a guard is not supported in a builder")
            if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant) \
                    and type(pattern.value.value) is int:
                labels.append(pattern.value.value)
                arms.append(case.body)
            elif isinstance(pattern, ast.MatchAs) and pattern.pattern is None and pattern.name is None:
                default = case.body
            else:
                raise HawkError(f"line {pattern.lineno}: a builder's match takes integer literals and `case _:`")
        if default is not None:
            arms.append(default)
        control = f"__hg{next(_COUNT)}"
        head = _parse(f"{control} = _hg.match(SUBJECT, {tuple(labels)!r}, {default is not None}, {self._leaks(arms, after)}, "
                      f"{stmt.lineno})")
        head[0].value.args[0] = _lambda(stmt.subject)
        return head + self._arms(control, arms, after) + _parse(f"{control}.end()")

    def while_(self, stmt: ast.While, after: set) -> list:
        """A ``while``."""
        if stmt.orelse:
            raise HawkError(f"line {stmt.lineno}: while / else is not supported in a builder")
        control = f"__hg{next(_COUNT)}"
        head = _parse(f"{control} = _hg.loop(CONDITION, None, {self._leaks([stmt.body], after)}, {stmt.lineno})")
        head[0].value.args[0] = _lambda(self._condition(stmt.test))
        loop = _parse(f"while {control}.next():\n    with {control}.body():\n        pass\n")[0]
        loop.body[0].body = _exit_guards(self.body(stmt.body, after | _loads(stmt.body)), control)
        return head + [loop] + _parse(f"{control}.end()")

    def for_(self, stmt: ast.For, after: set) -> list:
        """A ``for _ in range(cap)`` that opens with ``if not c: break``; any other ``for`` is Python."""
        first = stmt.body[0] if stmt.body else None
        guarded = (isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "range"
                   and len(stmt.iter.args) == 1 and not stmt.orelse and isinstance(first, ast.If) and not first.orelse
                   and len(first.body) == 1 and isinstance(first.body[0], ast.Break)
                   and isinstance(first.test, ast.UnaryOp) and isinstance(first.test.op, ast.Not))
        if not guarded:
            stmt.body = self.body(stmt.body, after | _loads(stmt.body))
            stmt.orelse = self.body(stmt.orelse, after)
            return [stmt]
        control = f"__hg{next(_COUNT)}"
        rest = stmt.body[1:]
        head = _parse(f"{control} = _hg.loop(CONDITION, CAP, {self._leaks([rest], after)}, {stmt.lineno})")
        head[0].value.args[0] = _lambda(first.test.operand)
        head[0].value.args[1] = stmt.iter.args[0]
        loop = _parse(f"for TARGET in {control}.iterate():\n    if not {control}.device:\n        pass\n"
                      f"    with {control}.body():\n        pass\n")[0]
        loop.target = stmt.target
        loop.body[0].body = [copy.deepcopy(first)]
        loop.body[1].body = _exit_guards(self.body(rest, after | _loads(rest)), control)
        return head + [loop] + _parse(f"{control}.end()")


_MEMO: dict = {}


def _definition(function) -> ast.FunctionDef:
    """The ``FunctionDef`` of ``function`` from its source. A source that is not the function that was loaded (a file edited under
    a running process) is refused by name, never rewritten."""
    for _attempt in range(2):
        definition = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        names = list(inspect.signature(function).parameters)
        if isinstance(definition, ast.FunctionDef) and definition.name == function.__name__ \
                and [a.arg for a in definition.args.args] == names[:len(definition.args.args)]:
            return definition
        linecache.checkcache(function.__code__.co_filename)
    raise HawkError(f"{function.__name__}: the source of {function.__code__.co_filename}:{function.__code__.co_firstlineno} is "
                    "not this function (the file changed since it was loaded): reload the module")


def _rewrite_function(function):
    """``function`` with its control flow lowered to calls on the runtime; the compiled code is kept per function, the closure
    rebuilt per call."""
    code = function.__code__
    key = (code.co_filename, code.co_firstlineno, code)
    if key not in _MEMO:
        definition = _definition(function)
        definition.decorator_list = []
        definition.body = _Rewriter().body(definition.body, set())
        arguments = ast.arguments(posonlyargs=[], args=[ast.arg(n) for n in (*code.co_freevars, "_hg")], kwonlyargs=[],
                                  kw_defaults=[], defaults=[])
        make = ast.FunctionDef("__hawk_make", arguments, [definition, ast.Return(ast.Name(definition.name, ast.Load()))], [], None)
        module = ast.Module([make], [])
        for node in (make, make.body[1]):
            node.lineno, node.col_offset, node.end_lineno, node.end_col_offset = definition.lineno, 0, definition.end_lineno, 0
        ast.fix_missing_locations(module)
        ast.increment_lineno(module, code.co_firstlineno - 1)
        _MEMO[key] = compile(module, code.co_filename, "exec")
    scope: dict = {}
    exec(_MEMO[key], function.__globals__, scope)
    rewritten = scope["__hawk_make"](*(cell.cell_contents for cell in (function.__closure__ or ())), _HG)
    rewritten.__defaults__, rewritten.__kwdefaults__ = function.__defaults__, function.__kwdefaults__
    return rewritten


# -- builders and blocks ---------------------------------------------------------------------------------------------------- #
def _is_graph_annotation(annotation) -> bool:
    """Whether ``annotation`` names :class:`hawk.Graph`."""
    if annotation is None or annotation is inspect.Parameter.empty:
        return False
    if isinstance(annotation, str):
        return annotation.replace(" ", "").endswith("Graph")
    from .graph import Graph

    return isinstance(annotation, type) and issubclass(annotation, Graph)


def make_builder(function):
    """The builder or block ``function`` stands for (:meth:`hawk.Graph.builder`)."""
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    block = bool(parameters) and _is_graph_annotation(parameters[0].annotation)
    returns = signature.return_annotation
    if not block and returns is not inspect.Signature.empty and not _is_graph_annotation(returns):
        raise HawkError(f"{function.__name__}: a builder returns its graph (`-> hawk.Graph`), not {returns}")
    held: dict = {}

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if "run" not in held:
            held["run"] = _rewrite_function(function)
        return (_run_block if block else _run_builder)(function, held["run"], signature, args, kwargs)

    #: Whether calling the wrapper makes and builds a graph, which is what :func:`hawk.diff.jvp` and :func:`hawk.diff.vjp` take.
    wrapper.makes_graph = not block
    return wrapper


def _run_block(function, rewritten, signature, args, kwargs):
    """Run a block: it appends to the graph (or branch) it is given."""
    from .graph import Graph

    if not args:
        raise HawkError(f"{function.__name__}: a block takes the graph first")
    handle, recorder = args[0], current()
    if recorder is None:
        if not isinstance(handle, Graph):
            raise HawkError(f"{function.__name__}: a block takes the open hawk.Graph it appends to")
        with _recording(Recorder(handle, "block", function.__name__)):
            return rewritten(*args, **kwargs)
    scope = recorder.graph._branches[-1] if recorder.graph is not None and recorder.graph._branches else recorder.graph
    if handle is not scope:
        raise HawkError(f"{function.__name__}: pass the handle of the scope you are in (the branch handle inside a branch)")
    recorder.block_depth += 1
    try:
        return rewritten(*args, **kwargs)
    finally:
        recorder.block_depth -= 1


def _run_builder(function, rewritten, signature, args, kwargs):
    """Run a builder: it makes, records into and returns its graph, which is then built."""
    if current() is not None:
        raise HawkError(f"{function.__name__}: a builder inside a builder; make it a block: give it the graph (`def "
                        f"{function.__name__}(g: hawk.Graph, ...)`)")
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    recorder = Recorder(None, "builder", function.__name__)
    with _recording(recorder):
        out = rewritten(*args, **kwargs)
    graph = recorder.graph
    if graph is None:
        raise HawkError(f"{function.__name__}: a builder makes its graph: `g = hawk.Graph()`")
    if out is not graph:
        raise HawkError(f"{function.__name__}: a builder returns its graph: `return g`; declare the arrays it gives back with "
                        "`g.result = ...`")
    graph._params = dict(bound.arguments)
    return graph.build()
