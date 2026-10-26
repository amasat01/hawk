# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Which elements of a Mutable plane a kernel body stores before it reads any.

A kernel that fills ``out[0]``, ``out[1]``, ... and reads no element it has not
yet stored never reads the launch-start value of ``out``: its derivative is well
defined, and the plane starts from zeros. A body that reads an element it has not
stored (``out[j+1]`` in a stencil, a partial fill that keeps the rest) needs the
launch-start value, a recurrence across launches that autodiff refuses by name.

The decision has two halves. The front end (:mod:`hawk.trace.astpass`) turns the
statements that mention the plane into a SPEC: a nested tuple whose leaves are
lambdas over the loop variables in scope, compiled in the kernel's own namespace,
so a bound such as ``range(DAYS)`` or an index ``3 * i + d`` resolves to the same
integers the trace sees. :func:`cover_of` then walks the spec with static ints,
loops UNROLLED in the analysis (their trip counts are static), and returns the
set of stored indices, or ``None`` the moment it cannot prove the answer.

Spec nodes: ``("seq", [node, ...])``; ``("for", var, args_fn, body)``;
``("if", then, else)``; ``("stmt", reads, stores)`` where ``reads`` is a list of
``fn`` (an element read) or ``None`` (the whole plane) and ``stores`` an index
``fn``, ``None`` (no element store) or ``"all"`` (a whole-plane write);
``("opaque",)`` for anything else that mentions the plane. Each ``fn`` is
``(fn, scope)``: ``fn(*values of scope)``.
"""

from __future__ import annotations

_BUDGET = 200_000


class _Unprovable(Exception):
    pass


def cover_of(spec, width: int) -> frozenset | None:
    """The indices stored before any read needed the launch-start value, as a
    frozenset; ``None`` when the body reads an element not yet stored, or the
    analysis cannot decide (a non-static index or bound, an unrolled trip count
    over the budget, a construct it does not model)."""
    state = {"covered": set(), "all": False, "steps": 0}
    try:
        _run(spec, {}, state, width)
    except (_Unprovable, TypeError, ValueError, NameError, ZeroDivisionError,
            IndexError, KeyError, AttributeError, OverflowError):
        return None
    return frozenset(range(width)) if state["all"] else frozenset(state["covered"])


def _static(fn_scope, env: dict) -> int:
    fn, scope = fn_scope
    value = fn(*[env[v] for v in scope])
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Unprovable
    return value


def _args(fn_scope, env: dict) -> tuple:
    fn, scope = fn_scope
    out = fn(*[env[v] for v in scope])
    if not isinstance(out, tuple) or not all(
            isinstance(a, int) and not isinstance(a, bool) for a in out):
        raise _Unprovable
    return out


def _run(node, env: dict, state: dict, width: int) -> None:
    state["steps"] += 1
    if state["steps"] > _BUDGET:
        raise _Unprovable
    kind = node[0]
    if kind == "seq":
        for child in node[1]:
            _run(child, env, state, width)
    elif kind == "for":
        _kind, var, args_fn, body = node
        for value in range(*_args(args_fn, env)):
            _run(body, {**env, var: value}, state, width)
    elif kind == "if":
        outer = (set(state["covered"]), state["all"])
        _run(node[1], env, state, width)
        then = (state["covered"], state["all"])
        state["covered"], state["all"] = set(outer[0]), outer[1]
        _run(node[2], env, state, width)
        other = (state["covered"], state["all"])
        state["covered"] = then[0] & other[0]
        state["all"] = then[1] and other[1]
    elif kind == "stmt":
        _kind, reads, store = node
        for read in reads:
            if state["all"]:
                continue
            if read is None:
                if set(range(width)) - state["covered"]:
                    raise _Unprovable
            elif _static(read, env) not in state["covered"]:
                raise _Unprovable
        if store == "all":
            state["all"] = True
        elif store is not None:
            index = _static(store, env)
            if not 0 <= index < width:
                raise _Unprovable
            state["covered"].add(index)
    else:
        raise _Unprovable
