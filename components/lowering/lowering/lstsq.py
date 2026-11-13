# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lowering of the least-squares kinds (``lstsq``, ``lstsq_gram_solve``, ``lstsq_info``), per target.

The shared IR holds the math; HERE, when a target's code is lowered, a node becomes the normal equations (a Cholesky of
``J^T J``) or Householder QR, by the ``linalg.max_dim`` of the variant the build is for. The derivative rules were taken on the
``lstsq`` nodes before this runs (:func:`hawk.diff.vjp` works on the traced kernel; lowering is the first step of a build)."""

from __future__ import annotations

from raptor.ir.linalg import LSTSQ_OPS

from raptor.ir import make as _make
from raptor.ir.nodes import Loop

from . import team as _team
from .cells import walk as _walk


def _lower_node(node, qr: bool):
    j = node.operands[0]
    if node.kind == "lstsq_info":
        if qr:
            return _make("qr_info", (_make("qr", (j,)),))
        return _make("chol_info", (_make("chol", (_gram(j),)),))
    b = node.operands[1]
    if qr:
        return _make("qr_solve" if node.kind == "lstsq" else "qr_gram_solve", (_make("qr", (j,)), b))
    rhs = _make("mv" if len(b.ttype.shape) == 1 else "mm", (_make("transpose", (j,)), b))
    return _make("cho_solve", (_make("chol", (_gram(j),)), rhs if node.kind == "lstsq" else b))


def _gram(j):
    """``J^T J``."""
    return _make("mm", (_make("transpose", (j,)), j))


def _children(n) -> list:
    """``n``'s operands, and for a loop also the roots of its body (the body is not among a loop's operands)."""
    if not isinstance(n, Loop):
        return list(n.operands)
    out = [*n.operands, *n.nexts, *n.exit_values, *n.body_sinks]
    if n.exit_cond is not None:
        out.append(n.exit_cond)
    return out


def _loop_over(n, memo):
    """loop ``n`` with its body (and its initial values) as lowered in ``memo``."""
    from raptor.ir.loops import build_loop, exit_kwargs

    kw = exit_kwargs(n)
    kw["exit_cond"] = None if n.exit_cond is None else memo[id(n.exit_cond)]
    kw["exit_values"] = tuple(memo[id(v)] for v in n.exit_values)
    if n.count is not None:
        kw["count"] = memo[id(n.count)]
    return build_loop(n.index, n.carries, tuple(memo[id(i)] for i in n.inits), tuple(memo[id(v)] for v in n.nexts),
                      start=n.start, stop=n.stop, step=n.step, body_sinks=tuple(memo[id(s)] for s in n.body_sinks), **kw)


def lowered_sinks(sinks, spec):
    """``sinks`` with every least-squares node lowered for the target ``spec`` is seen as (its variant's ``linalg.max_dim``, refused by
    name when it states none); the same tuple object when there is none. A node inside a loop body is lowered there: the loop is rebuilt over the lowered body."""
    memo: dict = {}
    max_dim = None
    stack = list(sinks)
    found = False
    while stack:
        n = stack[-1]
        if id(n) in memo:
            stack.pop()
            continue
        kids = _children(n)
        todo = [c for c in kids if id(c) not in memo]
        if todo:
            stack.extend(todo)
            continue
        stack.pop()
        if isinstance(n, Loop):
            memo[id(n)] = n if all(memo[id(c)] is c for c in kids) else _loop_over(n, memo)
            continue
        new = tuple(memo[id(c)] for c in n.operands)
        if n.kind in LSTSQ_OPS:
            found = True
            max_dim = _team.max_dim_of(spec, required=True) if max_dim is None else max_dim
            memo[id(n)] = _lower_node(n.rebuilt(new), new[0].ttype.shape[1] > max_dim)
        else:
            memo[id(n)] = n if all(a is b for a, b in zip(new, n.operands)) else n.rebuilt(new)
    return tuple(memo[id(s)] for s in sinks) if found else sinks


def lowered(sinks, walk, spec):
    """``(sinks, walk)`` ready for a writer: the least-squares nodes lowered for ``spec``'s target (:func:`lowered_sinks`) and the walk
    taken again, else both as given. The ONE call of the lowering, made by whoever prepares a kernel for a writer."""
    if not walk.kinds.intersection(LSTSQ_OPS):
        return sinks, walk
    sinks = lowered_sinks(tuple(sinks), spec)
    return sinks, _walk(sinks)
