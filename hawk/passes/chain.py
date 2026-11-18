# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A sequence of kernels bound by name into one sink set (:func:`hawk.chain`).

Member ``j``'s read of ``n`` binds to the latest value an earlier member committed to ``n``, else ``n`` is an input
of the chain. A name no later member reads is an output; a name several members update in turn (each reading it
first) binds in order and its last value is the output. The binding is a substitution in the IR: the composed body
holds each intermediate as one value, read by any number of members, and stores none of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .. import _ir
from .._errors import HawkError

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
    """The leaves under ``sinks`` and the leaves an ``At`` reads at a computed index."""
    nodes = _ir.post_order(sinks)
    leaves = [n for n in nodes if _ir.is_a(n, "leaf")]
    gathered = {n.operands[0] for n in nodes if _ir.is_a(n, "at")}
    return leaves, gathered


def _rebuilt(sinks, mapping) -> list:
    """``sinks`` with each node that is a key of ``mapping`` replaced by its value in the values they read."""
    flat = [o for s in sinks for o in s.operands]
    new = list(_ir.substitute(flat, mapping))
    rebuilt = []
    for s in sinks:
        rebuilt.append(_ir.replace(s, operands=tuple(new[:len(s.operands)])))
        del new[:len(s.operands)]
    return rebuilt


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
    context = members[0].context     # members come from their own contexts; the chain is one graph
    latest: dict = {}           # name -> (member, rebuilt sink)
    inputs: dict = {}           # name -> (member, role, ttype) of an unbound read
    reads: list = []            # per member: the names it reads
    for k in members:
        sinks = _ir.adopt(context, k.sinks)
        leaves, gathered = _walk_sinks(sinks)
        reads.append({lf.field_text("name") for lf in leaves})
        mapping = {}
        for lf in leaves:
            name, role, ttype = lf.field_text("name"), lf.field_text("role"), _ir.type_of(lf)
            if name not in latest:
                held = inputs.setdefault(name, (k.name, role, ttype))
                if held[1:] != (role, ttype):
                    raise HawkError(f"chain: {held[0]!r} reads {name!r} as (role={held[1]!r}, {held[2]!r}) but "
                                    f"{k.name!r} reads it as (role={role!r}, {ttype!r}); rename one")
                continue
            writer, sink = latest[name]
            if ttype != _ir.type_of(sink):
                raise HawkError(f"chain: {writer!r} commits {name!r} as {_ir.type_of(sink)!r} but {k.name!r} reads it as "
                                f"{ttype!r}; one name is one value")
            if role not in BOUND_LEAF_ROLES or lf in gathered or len(sink.operands) > 1 \
                    or sink.field_text("role") not in BOUND_SINK_ROLES:
                raise HawkError(f"chain: {k.name!r} reads {name!r}, which {writer!r} commits, at another sample "
                                f"(or {writer!r} commits it off its own lane); an intermediate is the lane's own "
                                "value: end the chain at that name and read the plane in a second chain")
            mapping[lf] = sink.operands[0]
        for s in sinks:
            name = s.field_text("name")
            if name in latest and name not in reads[-1]:
                raise HawkError(f"chain: {latest[name][0]!r} and {k.name!r} both commit {name!r} and "
                                f"{k.name!r} does not read it first; the order would decide the value")
            if name in inputs and inputs[name][1] != "mutable":
                held = inputs[name]
                raise HawkError(f"chain: {held[0]!r} reads {name!r} as an input (role={held[1]!r}) but {k.name!r} "
                                f"commits it; rename one")
        for s in (_rebuilt(sinks, mapping) if mapping else sinks):
            latest[s.field_text("name")] = (k.name, s)
    writer_at = {}
    for j, k in enumerate(members):
        for s in k.sinks:
            writer_at[s.field_text("name")] = j
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
            produced[s.field_text("name")] = _ir.type_of(s)
        later = set().union(*reads[b:])
        crossing.append(sum(_ir.element_count(t) for n, t in produced.items() if n in later))
    return ChainIR(tuple(latest[n][1] for n in outputs), planes, outputs, tuple(crossing))


#: The prefix of the plane a cut chain passes a value through from one segment to the next.
CROSS_PREFIX = "hwc_"




def segments(members: Sequence, cuts: Sequence[int]) -> list:
    """``[(sinks, planes)]``, one per segment of ``members`` cut before each member index in ``cuts``. A value one
    segment computes and a later one reads crosses in a plane the runtime makes per lane, shaped as a caller's plane of
    its width (``planes``: ``(name, dtype, elements)`` of those a segment writes): the writer commits the value there and
    the reader reads its own lane's, as any per-lane input."""
    bounds = [0, *cuts, len(members)]
    crossed: dict = {}                      # name -> ttype of a value an earlier segment stored
    out = []
    for a, b in zip(bounds, bounds[1:]):
        later = set().union(*({lf.field_text("name") for lf in _walk_sinks(k.sinks)[0]} for k in members[b:]))
        sinks = bind(members[a:b], keep=tuple(later)).sinks
        mapping = {}
        for lf in _walk_sinks(sinks)[0]:
            name, ttype = lf.field_text("name"), _ir.type_of(lf)
            if name in crossed:
                mapping[lf] = _ir.leaf(lf.ctx, "vocab_read", "vec_in" if ttype.shape else "per_sample",
                                       CROSS_PREFIX + name, ttype)
        if mapping:
            sinks = _rebuilt(sinks, mapping)
        kept, planes = [], []
        for s in sinks:
            name, ttype = s.field_text("name"), _ir.type_of(s)
            if name not in later:
                kept.append(s)
                continue
            if len(ttype.shape) > 1 or ttype.dtype not in ("f64", "f32", "i32"):
                raise HawkError(f"chain: {name!r} would cross a cut as {ttype!r}; only scalars and vectors of "
                                "reals or Index cross")
            kept.append(_ir.make_node("Assign", ttype, (s.operands[0],), name=CROSS_PREFIX + name))
            planes.append((CROSS_PREFIX + name, "int32" if ttype.dtype == "i32" else "real", _ir.element_count(ttype)))
            crossed[name] = ttype
        out.append((tuple(kept), tuple(planes)))
    return out
