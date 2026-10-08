# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``ast`` pass: a default-deny body surface, rewritten into ``select``.

Three ordered steps over the decorated function's own source. (1) EARLY RETURN
normalisation: ``if guard: ...; return`` followed by the rest of the body
becomes that ``if``'s ``else``. (2) VALIDATION, default-deny: only assignment,
expression, ``if``/``elif``/``else`` and a bounded ``for`` over a literal
``range`` (holding at most one top-level ``if cond: break``) survive;
everything else — ``while``, ``continue``, any other ``break``, ``try``,
``with``, ``import``, comprehensions, lambdas, ``and``/``or``/``not``, chained
or identity comparisons — is refused by a :class:`~hawk.ir.HawkError` naming
the construct and its line. (3) The ``if`` and ``for`` REWRITES: each ``if``
branch runs under Python's own evaluator and the names it assigns are merged
afterwards by ``select``; a bounded ``for`` runs its body ONCE against a
symbolic index and the names it carries become a :class:`~hawk.ir.Loop`. A
loop's ``if cond: break`` becomes the loop's EXIT (:meth:`_LoopBuilder.exit`),
lowered to a real per-sample ``break``. A sink method call may not sit inside
a branch or a loop — it would commit unconditionally, or once instead of once
per iteration.

A ``for`` is lowered to a real C++/CUDA ``for`` rather than unrolled at trace
time: unrolling blew ptxas's memory past 25 GB on one KAN-sized kernel, so the
loop stays intact and the compiler decides whether to unroll it.
"""

from __future__ import annotations

import ast
from typing import Any

from ..ir import HawkError
from ..ir.loop_nodes import LoopCarry, LoopIndex, LoopValue
from ..ir.loops import build_loop
from .value import Value, index_node, node_of, own_column_rewrite_advice, select

#: The reserved prefix of every name this pass synthesises.
PREFIX = "__hawk_"
#: The name the rewritten body reaches :func:`merge` under.
SELECT_HELPER = "__hawk_select__"
#: The name the rewritten body reaches :func:`open_loop` under.
LOOP_HELPER = "__hawk_loop__"
#: The name the rewritten body reaches :func:`set_component` under.
SETITEM_HELPER = "__hawk_setitem__"

_ALLOWED_STMTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.If,
                  ast.For, ast.Pass)
_BANNED_EXPR = {
    ast.Lambda: "a lambda", ast.ListComp: "a list comprehension",
    ast.SetComp: "a set comprehension", ast.DictComp: "a dict comprehension",
    ast.GeneratorExp: "a generator expression", ast.NamedExpr: "the walrus operator",
    ast.Yield: "`yield`", ast.YieldFrom: "`yield from`", ast.Await: "`await`",
    ast.Starred: "argument unpacking (`*`)", ast.JoinedStr: "an f-string",
    ast.BoolOp: "`and` / `or` (use land / lor — a kernel has no short circuit)",
}

#: The specific advice a comprehension refusal gives below.
_COMPREHENSION_ADVICE = (
    " — a kernel body is traced ONCE, so a comprehension over a lowered loop's "
    "own index would build a single element. To fill a rank-1 value component "
    "by component, write the component: bind `v` before a bounded `for` and "
    "assign `v[k] = …` inside it. A comprehension over a "
    "COMPILE-TIME sequence belongs in a plain-python helper, whose body the "
    "ast pass never sees"
)

#: The banned expressions whose message carries that advice.
_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _err(node: ast.AST, filename: str, msg: str) -> HawkError:
    return HawkError(f"{filename}:{getattr(node, 'lineno', '?')}: {msg}")



class _LoopLocal:
    """A name a ``for`` body assigned that was NOT bound before the loop.

    Reading it after the loop would silently lift one iteration's expression
    out of a loop that ran symbolically ONCE, so every use refuses by NAME
    instead. A name bound before the loop is loop-CARRIED and comes back from
    :meth:`_LoopBuilder.close` as an ordinary traced value."""

    __slots__ = ("_name", "_where", "_what")

    def __init__(self, name: str, where: str, what: str) -> None:
        self._name = name
        self._where = where
        self._what = what

    def _hawk_refuse(self, *_args: Any, **_kwargs: Any):
        raise HawkError(
            f"{self._where}: {self._name!r} is {self._what} of the `for` above and "
            "has no value after it. The body was traced ONCE against a symbolic "
            "index (lowers a bounded `for` to a real `for`; it is not unrolled "
            "by execution), so this name holds one iteration's expression, not the "
            "loop's result. Bind it BEFORE the loop to make it loop-carried, or "
            "read it only inside the body"
        )

    def __repr__(self) -> str:                # pragma: no cover - diagnostics only
        return f"<loop-local {self._name!r} at {self._where}>"


for _dunder in ("add", "radd", "sub", "rsub", "mul", "rmul", "truediv",
                "rtruediv", "pow", "rpow", "neg", "abs", "matmul", "rmatmul",
                "lt", "le", "gt", "ge", "eq", "ne", "getitem", "setitem",
                "getattr", "call", "bool", "float", "int", "index", "iter"):
    setattr(_LoopLocal, f"__{_dunder}__", _LoopLocal._hawk_refuse)
_LoopLocal.__hash__ = object.__hash__
del _dunder


class _LoopBuilder:
    """The object the rewritten body drives ONE bounded ``for`` through.

    Three calls, in order: :attr:`index` binds the loop variable to a symbolic
    :class:`~hawk.ir.loop_nodes.LoopIndex`, :meth:`carries` swaps each
    loop-carried name for its :class:`~hawk.ir.loop_nodes.LoopCarry`
    placeholder, and :meth:`close` reads the body's final values and mints the
    :class:`~hawk.ir.Loop`. The body runs exactly once, under Python's own
    evaluator, like an ``if`` branch."""

    __slots__ = ("start", "stop", "step", "trip", "where", "names", "depth",
                 "index", "_inits", "_originals", "_carries", "_exit")

    def __init__(self, bounds: tuple, where: str, names: tuple, index_name: str,
                 depth: int) -> None:
        resolved = []
        for bound in bounds:
            if isinstance(bound, bool) or not isinstance(bound, int):
                raise HawkError(
                    f"{where}: a `for` bound must resolve to a compile-time int — "
                    "the trip count is a compile-time extent, not a runtime value, "
                    "because the emitted `for` carries it as a literal the compiler "
                    f"can see (as amended by); got {bound!r}"
                )
            resolved.append(int(bound))
        if len(resolved) == 1:
            self.start, self.stop, self.step = 0, resolved[0], 1
        elif len(resolved) == 2:
            (self.start, self.stop), self.step = resolved, 1
        else:
            self.start, self.stop, self.step = resolved
        if self.step == 0:
            raise HawkError(f"{where}: `range` step of zero never terminates")
        self.trip = len(range(self.start, self.stop, self.step))
        self.where = where
        self.names = tuple(names)
        self.depth = int(depth)
        self.index = Value(LoopIndex(depth, index_name))
        self._inits: tuple = ()
        self._originals: tuple = ()
        self._carries: tuple = ()
        self._exit: tuple | None = None

    def carries(self, *values: Any) -> tuple:
        """Swap each carried name for its placeholder, remembering its initial
        value. Returns a tuple even for one name, so the unpack shape is fixed."""
        if not self.names:
            raise HawkError(
                f"{self.where}: this `for` assigns no name that is bound before it "
                "and commits no sink, so it computes nothing observable. A HAWK "
                "loop's whole output is its carried values: bind the "
                "accumulator before the loop"
            )
        self._originals = values
        self._inits = tuple(node_of(v) for v in values)
        self._carries = tuple(
            LoopCarry(self.depth, j, name, init.ttype)
            for j, (name, init) in enumerate(zip(self.names, self._inits))
        )
        return tuple(Value(c) for c in self._carries)

    def exit(self, cond: Any, values: tuple) -> None:
        """The loop's one ``if cond: break``: the per-sample condition and the
        carried values it leaves AT the break (Python's own semantics)."""
        if isinstance(cond, bool) or not isinstance(cond, Value):
            raise HawkError(
                f"{self.where}: this `break`'s condition is not a traced value "
                f"(got {cond!r}), so it is the same for every sample and every "
                "iteration — it is not a data-dependent exit. Shorten the `range` "
                "instead")
        node = node_of(cond)
        if node.ttype.shape != () or node.ttype.dtype != "bool":
            raise HawkError(
                f"{self.where}: a `break` condition must be a rank-0 bool "
                f"(a comparison, land / lor / lnot), got {node.ttype!r}")
        self._exit = (node, tuple(node_of(v) for v in values))

    def close(self, values: tuple) -> tuple:
        """Mint the loop from the values the body left in the carried names."""
        if self.trip == 0:
            # trip == 0: minting a Loop would size a reverse-pass tape at
            # zero, which C++ can't declare. Collapse to the originals
            # instead — Python's own `for` gives the same answer.
            return self._originals
        nexts = tuple(node_of(v) for v in values)
        extra: dict = {}
        if self._exit is not None:
            extra = {"exit_cond": self._exit[0], "exit_values": self._exit[1]}
        loop = build_loop(self.index.node, self._carries, self._inits, nexts,
                          start=self.start, stop=self.stop, step=self.step,
                          **extra)
        return tuple(Value(LoopValue(loop, j)) for j in range(len(self.names)))

    def local(self, name: str, what: str = "a loop-local name") -> _LoopLocal:
        """The sentinel a loop-LOCAL name (and the loop variable) is left as."""
        return _LoopLocal(name, self.where, what)


def open_loop(*args: Any, where: str, names: tuple, index_name: str,
              depth: int) -> _LoopBuilder:
    """The rewrite's entry point: resolve one ``for``'s bounds and open it."""
    return _LoopBuilder(args, where, names, index_name, depth)


def set_component(vector: Any, index: Any, value: Any, name: str) -> Any:
    """``v[k] = e`` — one component of a rank>=1 value replaced.

    A HAWK value is IMMUTABLE, so this becomes a REBINDING:
    ``v = __hawk_setitem__(v, k, e, "v")``. That makes ``v`` an ordinary name,
    so a ``for`` whose body writes its components carries it like a scalar
    accumulator (:mod:`hawk.ir.loops`).

    The index may be static or symbolic; folding a static one is the C++
    compiler's decision, not the tracer's."""
    from ..ir import make as _make
    from .decl import Plane

    if isinstance(vector, Plane):
        vector[index] = value       # a DECLARATION assigned to: its own refusal
    node = node_of(vector)
    if not node.ttype.shape:
        raise HawkError(
            f"{name!r} is a rank-0 value typed {node.ttype!r}, so it has no "
            "components to write; `v[k] = …` writes one component of a rank>=1 "
            "value (a Vector, a matrix row is not a HAWK form)")
    return Value(_make("set_component_at",
                       (node, index_node(index), node_of(value))))


def merge(cond: Any, on_true: Any, on_false: Any, name: str) -> Any:
    """Merge one name's two branch values (the rewrite's ``select`` call)."""
    if isinstance(cond, bool):
        return on_true if cond else on_false
    if on_true is on_false:
        return on_true
    if isinstance(on_true, Value) or isinstance(on_false, Value):
        return select(cond, on_true, on_false)
    if on_true == on_false:
        return on_true
    numbers = (int, float)
    if isinstance(on_true, numbers) and isinstance(on_false, numbers) \
            and not isinstance(on_true, bool) and not isinstance(on_false, bool):
        return select(cond, on_true, on_false)
    raise HawkError(
        f"the branches of an `if` give {name!r} two different UNTRACED values "
        f"({on_true!r} vs {on_false!r}) — only traced values merge into a select "
        "node"
    )


#: The local a value-returning body's final `return` expression is bound to.
RESULT_NAME = "hawk_returned_value"


def transform(fn: ast.FunctionDef, filename: str, sinks: frozenset[str],
              mutables: frozenset[str], returns_value: bool = False, *,
              finishes: frozenset[str] = frozenset()) -> ast.FunctionDef:
    """Normalise, validate and rewrite one kernel body. With ``returns_value``
    the body is a traced EXPRESSION instead: it must end with ``return <expr>``,
    rewritten like any assignment and returned.

    ``mutables`` names every Mutable AND WideOut own-column parameter — both
    forms get the SAME plain-variable treatment
    (:func:`_rewrite_mutable_stores`): a read before the plane's first store
    is its launch-start value, lazily; the final value commits once at the
    end. ``finishes`` are the ``Terminated`` parameters the body may finish
    with ONE ``mask = cond`` statement each
    (:class:`hawk.trace.value.TerminatedRef`): its TARGET is renamed like a
    mutable store, seeded first with the mask's "does not finish" value so an
    ``if`` folds a one-sided finish into a ``select`` — but its READS stay
    the mask's own bare value, unaffected (see :func:`_rewrite_mutable_stores`
    for why finishes and mutables need different treatment here).

    A literal ``.prior`` on a Mutable/Terminated/WideOut own-column name is
    refused here, syntactically, on the author's OWN un-rewritten source
    (the ``[...] = e`` half of the same refusal is caught inside
    :func:`_rewrite_mutable_stores`, which also sees every other store) —
    before this function's own later `.prior` reads (the lazy seed) are
    even minted."""
    body = list(fn.body)
    if returns_value:
        last = body[-1] if body else fn
        if not (isinstance(last, ast.Return) and last.value is not None):
            raise _err(last, filename,
                       "a traced value's body must end with `return <expression>`")
        body[-1] = ast.copy_location(
            ast.Assign(targets=[ast.Name(RESULT_NAME, ast.Store())],
                       value=last.value), last)
    body = _normalize_returns(body, filename)
    _validate(body, filename, sinks, in_branch=False)
    _check_no_prior_attr(body, filename, mutables | finishes)
    body, written = _rewrite_mutable_stores(body, sinks, mutables, finishes,
                                            filename, fn.name)
    body = [_assign(_mutable_name(name), ast.Call(
        func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="unset",
                           ctx=ast.Load()), args=[], keywords=[]))
            for name in written if name in finishes] + body
    params = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    body = _Rewriter(filename).block(body, {a.arg for a in params})
    body += [_commit(name) for name in written]
    if returns_value:
        body.append(ast.Return(value=ast.Name(RESULT_NAME, ast.Load())))
    out = ast.FunctionDef(name=fn.name, args=fn.args, body=body, decorator_list=[],
                          returns=None, type_comment=None)
    out.type_params = []          # py312 field; harmless where absent
    return ast.fix_missing_locations(ast.copy_location(out, fn))


# --------------------------------------------------------------------------
# (1) early return


def _normalize_returns(stmts: list[ast.stmt], filename: str) -> list[ast.stmt]:
    """``if g: ...; return`` + rest -> ``if g: ... else: rest``."""
    for i, st in enumerate(stmts):
        if isinstance(st, ast.If) and not st.orelse and st.body\
                and isinstance(st.body[-1], ast.Return):
            if st.body[-1].value is not None:
                raise _err(st.body[-1], filename,
                           "a kernel returns nothing — it commits through its "
                           "Mutable / Accum / Reduce parameters")
            guard = ast.copy_location(
                ast.If(test=st.test, body=st.body[:-1] or [ast.Pass()],
                       orelse=_normalize_returns(stmts[i + 1:], filename)), st)
            return stmts[:i] + [guard]
        if isinstance(st, ast.Return) and i == len(stmts) - 1 and st.value is None:
            return stmts[:i]
    return stmts


# --------------------------------------------------------------------------
# (2) default-deny validation


def _validate(stmts: list[ast.stmt], filename: str, sinks: frozenset[str],
              in_branch: bool, in_loop: bool = False) -> None:
    for st in stmts:
        if isinstance(st, ast.Return):
            raise _err(st, filename,
                       "a kernel returns nothing — it commits through its Mutable / "
                       "Accum / Reduce parameters, and an early `return` is supported "
                       "only as a guard: `if cond: ...; return` at the top level of "
                       "the body")
        if isinstance(st, ast.Break):
            raise _err(st, filename,
                       "a `break` is a HAWK form only as `if cond: break` — an "
                       "`if` with no `else` whose whole body is the `break` — "
                       "written as a TOP-LEVEL statement of a `for` body (not "
                       "inside another `if`)")
        if isinstance(st, ast.Continue):
            raise _err(st, filename,
                       "`continue` is not a HAWK body form — guard the rest of "
                       "the iteration with an `if` instead")
        if isinstance(st, ast.While):
            raise _err(st, filename,
                       "`While` is not a HAWK body form: a GPU loop needs a "
                       "static cap and a `while` has nowhere to put one. Write "
                       "`for _ in range(CAP):` with `if lnot(cond): break` as its "
                       "first statement")
        if not isinstance(st, _ALLOWED_STMTS):
            raise _err(st, filename,
                       f"`{type(st).__name__}` is not a HAWK body form; the surface "
                       "is assignment, expression, if/elif/else and a bounded `for` "
                       "(default-deny)")
        for expr in _expressions(st):
            _check_expr(expr, filename)
        _check_targets(st, filename)
        if isinstance(st, ast.Expr) and in_branch and _is_sink_call(st, sinks):
            raise _err(st, filename,
                       "a sink commit may not sit inside a branch — it would commit "
                       "unconditionally; assign the value and commit once")
        if isinstance(st, ast.Expr) and in_loop and isinstance(st.value, ast.Call):
            if _is_sink_call(st, sinks):
                raise _err(st, filename,
                           "a sink commit may not sit inside a `for` — the body is "
                           "traced ONCE against a symbolic index, so the "
                           "commit would happen once instead of once per iteration; "
                           "accumulate into a loop-carried name and commit once "
                           "after the loop")
            raise _err(st, filename,
                       "a bare call statement may not sit inside a `for`: it "
                       "discards its value, so the only thing it can do is mutate "
                       "a python object, and the body is traced ONCE against a "
                       "symbolic index — a list appended to here would hold "
                       "ONE element. To fill a rank-1 value component by "
                       "component, write the component: `v[k] = …` inside the "
                       "loop, with `v` bound before it")
        if isinstance(st, ast.If):
            _validate(st.body, filename, sinks, True, in_loop)
            _validate(st.orelse, filename, sinks, True, in_loop)
        elif isinstance(st, ast.For):
            if st.orelse:
                raise _err(st, filename, "a `for ... else` clause is not a HAWK form")
            _check_range(st, filename)
            exits = [x for x in st.body if _is_break_if(x)]
            if len(exits) > 1:
                raise _err(exits[1], filename,
                           "a `for` may hold ONE `if cond: break`; combine the "
                           "conditions with lor(...)")
            for x in exits:
                _check_expr(x.test, filename)
            _validate([x for x in st.body if not _is_break_if(x)], filename,
                      sinks, in_branch, True)


def _is_break_if(st: ast.stmt) -> bool:
    """``if cond: break`` exactly: no ``else``, the ``break`` its whole body."""
    return (isinstance(st, ast.If) and not st.orelse and len(st.body) == 1
            and isinstance(st.body[0], ast.Break))


def _expressions(st: ast.stmt) -> list[ast.expr]:
    if isinstance(st, (ast.Assign, ast.AugAssign, ast.Expr)):
        return [st.value]
    if isinstance(st, ast.AnnAssign):
        return [st.value] if st.value is not None else []
    if isinstance(st, ast.If):
        return [st.test]
    if isinstance(st, ast.For):
        return [st.iter]
    return []


def _check_expr(expr: ast.expr, filename: str) -> None:
    for n in ast.walk(expr):
        banned = _BANNED_EXPR.get(type(n))
        if banned is not None:
            advice = (_COMPREHENSION_ADVICE
                      if isinstance(n, _COMPREHENSION_NODES) else "")
            raise _err(n, filename,
                       f"{banned} is not a HAWK body form{advice}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            raise _err(n, filename, "`not` is not a HAWK body form — use lnot(x)")
        if isinstance(n, ast.Compare):
            if len(n.ops) != 1:
                raise _err(n, filename, "a chained comparison is not a HAWK body form")
            if isinstance(n.ops[0], (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                raise _err(n, filename,
                           "identity / membership (`is`, `in`) is not a HAWK body form")
        if isinstance(n, ast.Name) and n.id.startswith(PREFIX):
            raise _err(n, filename, f"the name prefix {PREFIX!r} is reserved")


def _check_targets(st: ast.stmt, filename: str) -> None:
    targets = getattr(st, "targets", None) or (
        [st.target] if hasattr(st, "target") else [])
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Attribute):
                raise _err(n, filename, "an attribute store is not a HAWK body form")
            if isinstance(n, ast.Name) and n.id.startswith(PREFIX):
                raise _err(n, filename,
                           f"the name prefix {PREFIX!r} is reserved")


def _check_range(st: ast.For, filename: str) -> None:
    """The STRUCTURAL half of the bounded-``for`` rule: a ``range`` call over a
    single name, at most three arguments, no keywords. Whether each bound is a
    compile-time integer is the other half, checked on the VALUE by
    :func:`bounded_range` — the AST can't tell a factory-local bound from a
    traced one."""
    call = st.iter
    ok = (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
          and call.func.id == "range" and call.args and not call.keywords
          and len(call.args) <= 3)
    if not ok or not isinstance(st.target, ast.Name):
        raise _err(st, filename,
                   "a `for` must be bounded by a `range(...)` over a single name — "
                   "the trip count is a compile-time extent, not a runtime value "
                   "")


def _is_sink_call(st: ast.Expr, sinks: frozenset[str]) -> bool:
    call = st.value
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id in sinks)


# --------------------------------------------------------------------------
# (3) mutable stores, then the if -> select rewrite


#: Prefix for a rewritten mutable store, so a mutable written on both `if`
#: sides merges through select like an author's own name.
MUT_PREFIX = PREFIX + "m_"


def _mutable_name(name: str) -> str:
    return f"{MUT_PREFIX}{name}"


def _check_no_prior_attr(stmts: list[ast.stmt], filename: str,
                         own_column: frozenset[str]) -> None:
    """A literal ``x.prior`` on a Mutable/Terminated/WideOut own-column name
    is no longer a HAWK form — the plane's bare name is an ordinary local,
    and a read before its first store is the launch-start value
    automatically. Scans the AUTHOR's own, un-rewritten source; ``.prior``
    on any OTHER name (an Accum/Reduce plane, which has no own-column form
    and keeps `.prior`'s existing "scattered or reduced" refusal unchanged)
    is not this rule's concern."""
    for st in stmts:
        for n in ast.walk(st):
            if (isinstance(n, ast.Attribute) and n.attr == "prior"
                    and isinstance(n.value, ast.Name) and n.value.id in own_column):
                raise _err(n, filename,
                           "`.prior` is no longer a HAWK form — "
                           f"{own_column_rewrite_advice(n.value.id)}")


def _rewrite_mutable_stores(stmts: list[ast.stmt], sinks: frozenset[str],
                            mutables: frozenset[str],
                            finishes: frozenset[str] = frozenset(),
                            filename: str = "<kernel>", kernel: str = "?"
                            ) -> tuple[list[ast.stmt], list[str]]:
    """Every plain-variable commit in the body, rewritten so the ``if``/
    ``for`` pass downstream sees exactly the shape it already knows how to
    merge or carry — the SAME ``__hawk_m_x`` + ``x.write(...)`` machinery as
    before, keyed on a Name target instead of a Subscript.

    A Mutable/WideOut own-column name ``m`` (``mutables`` — merged by
    :mod:`hawk.trace.kernel` from both forms, which share this exact
    treatment) is renamed EVERYWHERE in the body — every read and every
    store alike, by :class:`_RenameOwnColumn` — to the reserved shadow
    ``__hawk_m_m``, so the body's bare ``m`` never changes meaning out from
    under it and the shadow is free to carry the evolving value through
    Python's OWN sequential / tuple / for-carry rules, unaided. Where the
    body's first access to ``m`` (:func:`_needs_prior`, a conservative,
    control-flow-aware scan) is a READ, the shadow is seeded from ``m.prior``
    first — the plane's launch-start value, lazily, so a write-only plane
    mints no leaf at all.

    A ``Terminated`` finish name is different: ONLY its one finish-statement
    TARGET is renamed here (never its reads, which keep reading the mask's
    own bare value, unaffected by a later finish — there is no "new" value a
    finish exposes back to the body, unlike a Mutable), so the existing
    ``.unset()`` select-merge machinery for a one-sided finish applies
    unchanged to a bare ``terminated = cond``.

    A component store ``v[k] = e`` on an ordinary local is a REBINDING
    (:func:`set_component`) — unaffected by any of this; it already made
    ``v``'s own bare name the carrier, before a Mutable/WideOut's own bare
    name needed one, and now applies uniformly to a Mutable/WideOut's own
    bare name too (no longer "committed whole" only).

    The ``[...] = e`` half of the ban on the old spelling is caught HERE, on
    every Subscript store/aug-store (the ``.prior`` half is caught
    separately, before this runs, by :func:`_check_no_prior_attr`, on the
    author's own un-rewritten source)."""
    written: list[str] = []
    finished: list[str] = []
    own_column = mutables | finishes

    class _T(ast.NodeTransformer):
        loops = 0

        def visit_For(self, node: ast.For):
            self.loops += 1
            self.generic_visit(node)
            self.loops -= 1
            return node

        def visit_Assign(self, node: ast.Assign):
            if len(node.targets) != 1:
                return node
            t = node.targets[0]
            if isinstance(t, ast.Name):
                if t.id in finishes:
                    _finish_store(node, t.id, self.loops, finished, filename, kernel)
                    if t.id not in written:
                        written.append(t.id)
                    node.targets = [ast.copy_location(
                        ast.Name(id=_mutable_name(t.id), ctx=ast.Store()), t)]
                return node          # a mutable/wide_out/ordinary local: untouched
                                     # here, bulk-renamed below if it is own-column
            if not isinstance(t, ast.Subscript) or not isinstance(t.value, ast.Name):
                return node
            name = t.value.id
            whole = (isinstance(t.slice, ast.Constant) and t.slice.value is Ellipsis)
            if whole and name in own_column:
                raise _err(node, filename,
                           f"kernel {kernel!r}: `{name}[...] = …` is no longer a "
                           f"HAWK form — {own_column_rewrite_advice(name)}")
            if name in finishes:
                raise _err(node, filename,
                           f"kernel {kernel!r}: a mask is finished whole — write "
                           f"`{name} = cond`")
            if name in mutables:
                return _component_store(node, t, node.value)
            if name in sinks:
                return node          # Accum/Reduce, or a WideOut scatter: own protocol
            if whole:
                raise _err(node, filename,
                           f"kernel {kernel!r}: `{name}[...] = …` commits "
                           "a plane the kernel did not declare — a plane is "
                           "written through its own parameter (a Mutable, or a "
                           "declared `Terminated` mask finished with "
                           f"`{name} = cond`)")
            # `v[k] = e`: a REBINDING (see `set_component`), rewritten here
            # because both subscript stores must become name assignments
            # before the loop rewrite decides which names a `for` carries.
            return _component_store(node, t, node.value)

        def visit_AugAssign(self, node: ast.AugAssign):
            """``v[k] += e`` — the accumulate spelling of the same store.

            Rewritten here because a traced value has no ``__setitem__``:
            Python's own augmented subscript store relies on
            ``__getitem__``/``__setitem__``, so this writes the read out
            explicitly and folds into the plain ``v[k] <op> e`` store. A
            bare-Name augmented store (``t += dt``) is NOT handled here —
            left untouched, it is bulk-renamed below like any other own-
            column occurrence, and Python's own ``+=`` already reads-then-
            stores correctly once the shadow carries the value."""
            t = node.target
            if not isinstance(t, ast.Subscript) or not isinstance(t.value, ast.Name):
                return node
            name = t.value.id
            whole = (isinstance(t.slice, ast.Constant) and t.slice.value is Ellipsis)
            if whole and name in own_column:
                raise _err(node, filename,
                           f"kernel {kernel!r}: `{name}[...] = …` is no longer a "
                           f"HAWK form — {own_column_rewrite_advice(name)}")
            if name in finishes:
                raise _err(node, filename,
                           f"kernel {kernel!r}: the mask {name!r} is finished "
                           f"with `{name} = cond`, never an augmented assignment")
            if name in mutables or name in sinks:
                return node                      # a plane's own commit protocol
            read = ast.copy_location(ast.Subscript(
                value=ast.Name(id=name, ctx=ast.Load()), slice=t.slice,
                ctx=ast.Load()), t)
            combined = ast.copy_location(
                ast.BinOp(left=read, op=node.op, right=node.value), node)
            return _component_store(node, t, combined)

    out = [_T().visit(st) for st in stmts]
    present = sorted(m for m in mutables if _mentions(out, m))
    seeds: list[ast.stmt] = []
    for name in present:
        if _needs_prior(out, name):
            seeds.append(_assign(_mutable_name(name), ast.Attribute(
                value=ast.Name(id=name, ctx=ast.Load()), attr="prior",
                ctx=ast.Load())))
        if _is_stored(out, name) and name not in written:
            written.append(name)
    if present:
        renamer = _RenameOwnColumn(set(present))
        out = [renamer.visit(st) for st in out]
    return seeds + out, written


def _mentions(stmts: list[ast.stmt], name: str) -> bool:
    """Whether ``name`` is referenced anywhere (read or write) in ``stmts``."""
    return any(isinstance(n, ast.Name) and n.id == name
              for st in stmts for n in ast.walk(st))


def _is_stored(stmts: list[ast.stmt], name: str) -> bool:
    """Whether ``name`` is EVER a bare-Name store target in ``stmts``."""
    return any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
              for st in stmts for n in ast.walk(st))


def _needs_prior(stmts: list[ast.stmt], name: str) -> bool:
    """True when ``name`` may be read before this body assigns it, i.e. the body
    needs the plane's launch-start value.

    A read before any assignment needs it; so does a branch that leaves the name
    unassigned (a one-sided ``if`` keeps the old value), and any mention inside a
    loop (a loop-carried value starts from the launch-start value). An ``if``
    whose branches all assign the name before reading it does not. Over-triggering is harmless for a Mutable (an unused seed
    never reaches a sink, so the walk digest is unchanged) but wrong for a WideOut
    own column, which has no launch-start value."""
    return _first_access(stmts, name) == "read"


def _mentions_as(node: ast.AST, name: str, ctx: type) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ctx)
               for n in ast.walk(node))


def _first_access(stmts: list[ast.stmt], name: str) -> str | None:
    """``"read"`` (the start value may be needed), ``"write"`` (assigned before
    any read on every path) or ``None`` (not touched)."""
    for st in stmts:
        if isinstance(st, ast.If):
            if _mentions_as(st.test, name, ast.Load):
                return "read"
            a, b = _first_access(st.body, name), _first_access(st.orelse, name)
            if "read" in (a, b) or (a is None) != (b is None):
                return "read"
            if a == "write":
                return "write"
            continue
        if isinstance(st, (ast.For, ast.While, ast.With, ast.Try)):
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(st)):
                return "read"
            continue
        if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name)\
                and st.target.id == name:
            return "read"
        if _mentions_as(st, name, ast.Load):
            return "read"
        if _mentions_as(st, name, ast.Store):
            return "write"
    return None


class _RenameOwnColumn(ast.NodeTransformer):
    """Every ``Name`` (read or write alike) for a plain-variable Mutable/
    WideOut name, renamed to its reserved shadow — see
    :func:`_rewrite_mutable_stores`."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.names:
            return ast.copy_location(
                ast.Name(id=_mutable_name(node.id), ctx=node.ctx), node)
        return node


def _finish_store(node: ast.Assign, name: str, loops: int,
                  finished: list[str], filename: str, kernel: str) -> None:
    """The static refusals of one ``mask = cond`` finish statement."""
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
        raise _err(node, filename,
                   f"kernel {kernel!r}: the mask {name!r} is finished with the "
                   f"Python literal {node.value.value!r}; `= True` is 'always "
                   "finish' — spell the condition (`= t >= t_final`) — and "
                   "`= False` is a no-op that reads as a bug")
    if name in finished:
        raise _err(node, filename,
                   f"kernel {kernel!r}: the mask {name!r} is finished twice; a "
                   f"mask takes exactly ONE `{name} = cond` statement — "
                   "combine the conditions with lor(...)")
    if loops:
        raise _err(node, filename,
                   f"kernel {kernel!r}: the mask {name!r} is finished inside a "
                   "`for`; the body is traced ONCE against a symbolic index — "
                   "compute the condition in the loop and finish once after it")
    finished.append(name)


def _component_store(node: ast.stmt, target: ast.Subscript,
                     value: ast.expr) -> ast.stmt:
    """``v[k] = value`` as the REBINDING the IR needs (see :func:`set_component`):
    ``v = __hawk_setitem__(v, k, value, "v")``."""
    name = target.value.id
    return ast.copy_location(ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id=SETITEM_HELPER, ctx=ast.Load()),
            args=[ast.Name(id=name, ctx=ast.Load()), target.slice, value,
                  ast.Constant(value=name)],
            keywords=[])), node)


def _commit(name: str) -> ast.stmt:
    call = ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()),
                                       attr="write", ctx=ast.Load()),
                    args=[ast.Name(id=_mutable_name(name), ctx=ast.Load())],
                    keywords=[])
    return ast.Expr(value=call)


class _Rewriter:
    """Turns every ``if``/``IfExp`` into ``select`` merges, and every bounded
    ``for`` into ONE symbolic pass that mints a :class:`~hawk.ir.Loop`."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.k = 0
        self.loops = 0
        self.depth = 0
        #: (holder, carried names) of each open ``for``, innermost last.
        self.open_loops: list[tuple[str, list[str]]] = []

    def block(self, stmts: list[ast.stmt], bound: set[str]) -> list[ast.stmt]:
        """Rewrite a statement list, threading the set of names bound so far."""
        out: list[ast.stmt] = []
        for st in stmts:
            if _is_break_if(st):
                holder, carried = self.open_loops[-1]
                out.append(ast.copy_location(ast.fix_missing_locations(ast.Expr(
                    value=ast.Call(func=ast.Attribute(
                        value=ast.Name(id=holder, ctx=ast.Load()), attr="exit",
                        ctx=ast.Load()),
                        args=[_IfExp(self).visit(st.test),
                              ast.Tuple(elts=_loads(carried), ctx=ast.Load())],
                        keywords=[]))), st))
            elif isinstance(st, ast.If):
                out.extend(self._if(st, bound))
            elif isinstance(st, ast.For):
                out.extend(self._for(st, bound))
            else:
                out.append(_IfExp(self).visit(st))
                bound |= _assigned([st])
        return out

    def _for(self, st: ast.For, bound: set[str]) -> list[ast.stmt]:
        """One bounded ``for``, rewritten into a SINGLE symbolic pass: open the
        loop, bind the index name, swap every loop-CARRIED name for its
        placeholder, run the body once, then close — minting the loop and
        returning the carried names' final values.

        A name is CARRIED iff the body assigns it AND it was already bound
        before the loop; otherwise it is loop-LOCAL and is left as a refusing
        sentinel rather than leaking one iteration's expression out (the loop
        VARIABLE is treated the same way)."""
        assigned = _assigned(st.body)
        carried = sorted(assigned & bound)
        locals_ = sorted(assigned - bound)
        for name in locals_:
            if name.startswith(MUT_PREFIX):
                raise _err(st, self.filename,
                           f"the mutable plane {name[len(MUT_PREFIX):]!r} is "
                           "committed only inside this `for`. The body is traced "
                           "ONCE against a symbolic index, so the commit would "
                           "carry one iteration's expression; commit it once after "
                           "the loop, or seed the plane before the loop so the write "
                           "is loop-carried")
        k, self.loops = self.loops, self.loops + 1
        holder = f"{PREFIX}L{k}"
        where = f"{self.filename}:{getattr(st, 'lineno', '?')}"
        index_name = st.target.id
        call = st.iter
        out: list[ast.stmt] = [
            _assign(holder, ast.Call(
                func=ast.Name(id=LOOP_HELPER, ctx=ast.Load()),
                args=list(call.args),
                keywords=[
                    ast.keyword(arg="where", value=ast.Constant(value=where)),
                    ast.keyword(arg="names", value=ast.Tuple(
                        elts=[ast.Constant(value=n) for n in carried],
                        ctx=ast.Load())),
                    ast.keyword(arg="index_name",
                                value=ast.Constant(value=index_name)),
                    ast.keyword(arg="depth", value=ast.Constant(value=self.depth)),
                ])),
            _assign(index_name, ast.Attribute(
                value=ast.Name(id=holder, ctx=ast.Load()), attr="index",
                ctx=ast.Load())),
            ast.Assign(
                targets=[ast.Tuple(
                    elts=[ast.Name(id=n, ctx=ast.Store()) for n in carried],
                    ctx=ast.Store())],
                value=ast.Call(func=ast.Attribute(
                    value=ast.Name(id=holder, ctx=ast.Load()), attr="carries",
                    ctx=ast.Load()), args=_loads(carried), keywords=[])),
        ]
        self.depth += 1
        inner = set(bound)
        inner.add(index_name)
        self.open_loops.append((holder, carried))
        out += self.block(st.body, inner)
        self.open_loops.pop()
        self.depth -= 1
        out.append(ast.Assign(
            targets=[ast.Tuple(
                elts=[ast.Name(id=n, ctx=ast.Store()) for n in carried],
                ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(
                value=ast.Name(id=holder, ctx=ast.Load()), attr="close",
                ctx=ast.Load()),
                args=[ast.Tuple(elts=_loads(carried), ctx=ast.Load())],
                keywords=[])))
        out.append(_assign(index_name, _local_call(holder, index_name,
                                                   "the loop VARIABLE")))
        for name in locals_:
            out.append(_assign(name, _local_call(holder, name,
                                                 "a loop-LOCAL name")))
        bound |= set(carried)
        return [ast.copy_location(ast.fix_missing_locations(s), st) for s in out]

    def _if(self, st: ast.If, bound: set[str]) -> list[ast.stmt]:
        k, self.k = self.k, self.k + 1
        cond, snap, then = f"{PREFIX}c{k}", f"{PREFIX}s{k}", f"{PREFIX}t{k}"
        in_body, in_else = _assigned(st.body), _assigned(st.orelse)
        merged = sorted(in_body | in_else)
        for name in merged:
            if name not in (in_body & in_else) and name not in bound:
                if name.startswith(MUT_PREFIX):
                    raise _err(st, self.filename,
                               f"the mutable plane {name[len(MUT_PREFIX):]!r} is "
                               "committed on only one side of this `if` — commit it on "
                               "both paths, or commit once after the branch")
                raise _err(st, self.filename,
                           f"{name!r} is assigned in only one branch and is not bound "
                           "before the `if` — it would have no value on the other path "
                           "")
        pre = sorted(set(merged) & bound)
        body = self.block(st.body, set(bound))
        orelse = self.block(st.orelse, set(bound))
        out = [_assign(cond, _IfExp(self).visit(st.test)),
               _assign(snap, _tuple(pre))] + body +\
              [_assign(then, _tuple(merged)), _unpack(pre, snap)] + orelse
        for i, name in enumerate(merged):
            out.append(_assign(name, ast.Call(
                func=ast.Name(id=SELECT_HELPER, ctx=ast.Load()),
                args=[ast.Name(id=cond, ctx=ast.Load()),
                      ast.Subscript(value=ast.Name(id=then, ctx=ast.Load()),
                                    slice=ast.Constant(value=i), ctx=ast.Load()),
                      ast.Name(id=name, ctx=ast.Load()),
                      ast.Constant(value=name)], keywords=[])))
        bound |= set(merged)
        return [ast.copy_location(ast.fix_missing_locations(s), st) for s in out]


class _IfExp(ast.NodeTransformer):
    """``a if c else b`` is a select — the expression form of the rewrite."""

    def __init__(self, owner: _Rewriter) -> None:
        self.owner = owner

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        self.generic_visit(node)
        return ast.copy_location(ast.Call(
            func=ast.Name(id=SELECT_HELPER, ctx=ast.Load()),
            args=[node.test, node.body, node.orelse,
                  ast.Constant(value="<conditional expression>")], keywords=[]), node)



def _assigned(stmts: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.add(n.id)
    return {n for n in names
            if not n.startswith(PREFIX) or n.startswith(MUT_PREFIX)}


def _assign(name: str, value: ast.expr) -> ast.stmt:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)


def _tuple(names: list[str]) -> ast.expr:
    return ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in names],
                     ctx=ast.Load())


def _loads(names: list[str]) -> list[ast.expr]:
    return [ast.Name(id=n, ctx=ast.Load()) for n in names]


def _local_call(holder: str, name: str, what: str) -> ast.expr:
    """``<holder>.local("<name>", "<what>")`` — the sentinel a loop-local and the
    loop variable are left as once the scope has closed."""
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id=holder, ctx=ast.Load()), attr="local",
                           ctx=ast.Load()),
        args=[ast.Constant(value=name), ast.Constant(value=what)], keywords=[])


def _unpack(names: list[str], source: str) -> ast.stmt:
    if not names:
        return ast.Pass()
    return ast.Assign(targets=[ast.Tuple(
        elts=[ast.Name(id=n, ctx=ast.Store()) for n in names], ctx=ast.Store())],
        value=ast.Name(id=source, ctx=ast.Load()))
