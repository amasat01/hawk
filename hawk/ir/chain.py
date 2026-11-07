# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A sequence of kernels bound by name into one sink set (:func:`hawk.chain`).

Member ``j``'s read of ``n`` binds to the latest value an earlier member committed to ``n``, else ``n`` is an input
of the chain. A name no later member reads is an output; a name several members update in turn (each reading it
first) binds in order and its last value is the output. The binding is a substitution in the IR: the composed body
holds each intermediate as one value, read by any number of members, and stores none of them.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

from .nodes import At, HawkError, Leaf
from .walk import substitute

#: The roles a member may read an intermediate as: its own lane's element.
BOUND_LEAF_ROLES = ("per_sample", "vec_in", "mat_in", "mutable")
#: The roles an intermediate may be committed as.
BOUND_SINK_ROLES = ("mutable", "out")


@dataclass(frozen=True)
class ChainIR:
    """The composed sink set of a chain. ``planes`` holds the members' declarations of every input and output;
    ``crossing[b]`` the elements per sample of the values produced before boundary ``b`` (between members ``b`` and
    ``b + 1``) and read after it."""

    sinks: tuple
    planes: dict
    outputs: tuple
    crossing: tuple


def _walk_sinks(sinks) -> tuple[list, set]:
    """The leaves under ``sinks`` and the ids of the leaves an ``At`` reads at a computed index."""
    leaves, gathered, seen, stack = [], set(), set(), list(sinks)
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Leaf):
            leaves.append(n)
        elif isinstance(n, At):
            gathered.add(id(n.operands[0]))
        stack.extend(n.operands)
    return leaves, gathered


def _elements(ttype) -> int:
    count = 1
    for d in ttype.shape:
        count *= d
    return count


def check_members(members: Sequence) -> None:
    """The refusals that do not need the binding: R3 (the members' declarations disagree) and R4 (a stepped member)."""
    first = members[0]
    for k in members:
        if k.step is not None or k.one_step is not None:
            raise HawkError(f"chain: member {k.name!r} is a stepped kernel; chain the single step, then run "
                            "hawk.steps over the chain")
        for what in ("kind", "lane_space", "group", "jump"):
            if getattr(k, what) != getattr(first, what):
                raise HawkError(f"chain: members {first.name!r} and {k.name!r} disagree on {what} "
                                f"({getattr(first, what)!r} vs {getattr(k, what)!r}); a chain runs as one launch")


def bind(members: Sequence, keep: Sequence[str] = ()) -> ChainIR:
    """Compose ``members`` in order (see the module docstring); ``keep`` names intermediates to commit as outputs
    too. Refuses, naming both members and the name: R1 a name with two types or two input roles, R2 a second writer
    of a name that does not read it first, R5 an intermediate read at another sample or committed off its own lane."""
    check_members(members)
    latest: dict = {}           # name -> (member, rebuilt sink)
    inputs: dict = {}           # name -> (member, role, ttype) of an unbound read
    reads: list = []            # per member: the names it reads
    for k in members:
        leaves, gathered = _walk_sinks(k.sinks)
        reads.append({lf.name for lf in leaves})
        mapping = {}
        for lf in leaves:
            if lf.name not in latest:
                held = inputs.setdefault(lf.name, (k.name, lf.role, lf.ttype))
                if held[1:] != (lf.role, lf.ttype):
                    raise HawkError(f"chain: {held[0]!r} reads {lf.name!r} as (role={held[1]!r}, {held[2]!r}) but "
                                    f"{k.name!r} reads it as (role={lf.role!r}, {lf.ttype!r}); rename one")
                continue
            writer, sink = latest[lf.name]
            if lf.ttype != sink.ttype:
                raise HawkError(f"chain: {writer!r} commits {lf.name!r} as {sink.ttype!r} but {k.name!r} reads it as "
                                f"{lf.ttype!r}; one name is one value")
            if lf.role not in BOUND_LEAF_ROLES or id(lf) in gathered or sink.index is not None \
                    or sink.role not in BOUND_SINK_ROLES:
                raise HawkError(f"chain: {k.name!r} reads {lf.name!r}, which {writer!r} commits, at another sample "
                                f"(or {writer!r} commits it off its own lane); an intermediate is the lane's own "
                                "value: end the chain at that name and read the plane in a second chain")
            mapping[id(lf)] = sink.value
        for s in k.sinks:
            if s.name in latest and s.name not in reads[-1]:
                raise HawkError(f"chain: {latest[s.name][0]!r} and {k.name!r} both commit {s.name!r} and "
                                f"{k.name!r} does not read it first; the order would decide the value")
            if s.name in inputs and inputs[s.name][1] != "mutable":
                held = inputs[s.name]
                raise HawkError(f"chain: {held[0]!r} reads {s.name!r} as an input (role={held[1]!r}) but {k.name!r} "
                                f"commits it; rename one")
        flat = [o for s in k.sinks for o in s.operands]
        new = list(substitute(flat, mapping)) if mapping else flat
        for s in k.sinks:
            rebuilt = copy.copy(s)
            rebuilt.operands = tuple(new[:len(s.operands)])
            del new[:len(s.operands)]
            latest[s.name] = (k.name, rebuilt)
    writer_at = {}
    for j, k in enumerate(members):
        for s in k.sinks:
            writer_at[s.name] = j
    outputs = tuple(n for n, j in writer_at.items()
                    if n in keep or not any(n in r for r in reads[j + 1:]))
    consumed = set(writer_at) - set(outputs) - set(inputs)
    planes: dict = {}
    for j, k in enumerate(members):                   # an output's declaration is its last writer's
        for n, decl in k.planes.items():
            if n not in consumed and (n not in planes or writer_at.get(n) == j):
                planes[n] = decl
    produced: dict = {}
    crossing = []
    for b in range(1, len(members)):
        for s in members[b - 1].sinks:
            produced[s.name] = s.ttype
        later = set().union(*reads[b:])
        crossing.append(sum(_elements(t) for n, t in produced.items() if n in later))
    return ChainIR(tuple(latest[n][1] for n in outputs), planes, outputs, tuple(crossing))


#: The prefix of the plane a cut chain passes a value through from one segment to the next.
CROSS_PREFIX = "hwc_"


def _cross_names(name: str, ttype) -> list:
    """The scalar planes one crossing value of ``ttype`` travels in: one per element."""
    return [f"{CROSS_PREFIX}{name}"] if not ttype.shape else [f"{CROSS_PREFIX}{name}_{k}" for k in range(ttype.shape[0])]


def segments(members: Sequence, cuts: Sequence[int]) -> list:
    """``[(sinks, planes)]``, one per segment of ``members`` cut before each member index in ``cuts``. A value one
    segment computes and a later one reads crosses in scalar planes the runtime makes per lane (``planes``:
    ``(name, dtype)`` of those a segment writes): the writer stores each element, the reader assembles them back."""
    from ..types import TensorType
    from .nodes import Assign, Op

    bounds = [0, *cuts, len(members)]
    crossed: dict = {}                      # name -> ttype of a value an earlier segment stored
    out = []
    for a, b in zip(bounds, bounds[1:]):
        later = set().union(*({lf.name for lf in _walk_sinks(k.sinks)[0]} for k in members[b:]))
        sinks = bind(members[a:b], keep=tuple(later)).sinks
        mapping = {}
        for lf in _walk_sinks(sinks)[0]:
            if lf.name in crossed:
                scalar = TensorType((), lf.ttype.dtype)
                parts = [Leaf("vocab_read", "per_sample", nm, scalar) for nm in _cross_names(lf.name, lf.ttype)]
                mapping[id(lf)] = parts[0] if not lf.ttype.shape else Op("vec", tuple(parts), lf.ttype)
        if mapping:
            flat = [o for s in sinks for o in s.operands]
            new = list(substitute(flat, mapping))
            rebuilt = []
            for s in sinks:
                twin = copy.copy(s)
                twin.operands = tuple(new[:len(s.operands)])
                del new[:len(s.operands)]
                rebuilt.append(twin)
            sinks = rebuilt
        kept, planes = [], []
        for s in sinks:
            if s.name not in later:
                kept.append(s)
                continue
            if len(s.ttype.shape) > 1 or s.ttype.dtype not in ("f64", "f32", "i32"):
                raise HawkError(f"chain: {s.name!r} would cross a cut as {s.ttype!r}; only scalars and vectors of "
                                "reals or Index cross")
            scalar = TensorType((), s.ttype.dtype)
            names = _cross_names(s.name, s.ttype)
            for k, nm in enumerate(names):
                part = s.value if not s.ttype.shape else Op("component", (s.value,), scalar, k)
                kept.append(Assign(nm, part, scalar))
                planes.append((nm, "int32" if s.ttype.dtype == "i32" else "real"))
            crossed[s.name] = s.ttype
        out.append((tuple(kept), tuple(planes)))
    return out
