# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The forward derivative of a kernel's group of lanes.

The tangent kernel holds the same group (membership, body order, barriers). Beside each floating scratch plane it declares a
tangent scratch plane of the same extent, stored and read wherever the primal is; the primal scratch stays, because a
tangent reads the primal values it multiplies. A group-wide sum and a scan are linear, so their tangent is the sum or scan
of the tangents. A minimum or maximum takes the tangent of its winner, the first lane in rank order among equals; a lane
holding NaN never wins, so a group of NaN has a zero tangent. ``any`` and ``all`` are decisions and carry none.

The reverse derivative is one kernel with the same group. Its body first recomputes the primal scratch stores (primal scratch
is local to the group, so the forward sweep is redone inside it), then runs the adjoint of each phase in reverse order. An
adjoint scratch plane ``bar_<name>`` sits beside each floating scratch plane. A scratch slot is read by many lanes, and
lanes of a group cannot add into one slot together, so a read's adjoint is gathered: the reading lane stores its
contribution into a plane of its own, a barrier follows, and the slot's own lane sums what the lanes that read it sent,
found from the inverse of the read's edge rule (:func:`gather_contributions`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .. import _ir
from ..errors import HawkError

#: A rank is below the lane count of any group; a lane that does not hold the extreme gets this larger rank.
_NO_WINNER = float(2**40)
_RANK = _ir.TensorType((), "f64")


def groups_in(sinks: Iterable[Any]) -> tuple[Any, ...]:
    """The ``Group`` nodes among the root ``sinks``."""
    return tuple(s for s in sinks if _ir.is_a(s, "group"))


def flat_sinks(sinks: Iterable[Any]) -> tuple[Any, ...]:
    """The ``sinks`` with each group replaced by the scratch planes it declares and the sinks of its body."""
    flat: list = []
    for sink in sinks:
        if _ir.is_a(sink, "group"):
            flat.extend(sink.field_nodes("scratch"))
            flat.extend(_ir.group_body(sink))
        else:
            flat.append(sink)
    return tuple(flat)


def has_tangent(node: Any) -> bool:
    """Whether the plane or value ``node`` is a floating one (an integer or a bool carries no tangent)."""
    return _ir.type_of(node).dtype in ("f64", "f32")


def tangent_scratch(scratch: Any, derived: str) -> Any:
    """The scratch plane named ``derived`` beside ``scratch``, of its type and extent (a tangent or an adjoint plane)."""
    return _ir.scratch(
        scratch.ctx, derived, _ir.type_of(scratch), _ir.field_pairs(scratch, "axes")
    )


def reduce_tangent(node: Any, tangent: Any) -> Any:
    """The tangent of the group-wide reduce ``node`` given the tangent of its value, or ``None`` when it has none.

    A sum adds the tangents in the order the primal adds. A minimum or maximum sums the tangent of its winner lane over
    the group: the winner is the lowest rank among the lanes equal to the result, found with a minimum of ranks, so a
    lane holding NaN (never equal to anything) cannot win and an all-NaN group sums nothing.
    """
    op = node.field_text("op")
    if op == "sum":
        return _ir.group_reduce("sum", tangent, node.field_text("order"))
    if op not in ("min", "max"):
        return None
    ttype = _ir.type_of(tangent)
    won = _ir.make_node(
        "Select", ttype, [winner_lane(node), tangent, _ir.zero_like(node.ctx, ttype)]
    )
    return _ir.group_reduce("sum", won, "tree")


def winner_lane(node: Any) -> Any:
    """The test that this lane holds the result of the group minimum or maximum ``node``: the lowest rank among the lanes
    equal to the result, found with a minimum of ranks. A lane holding NaN is equal to nothing, so it never wins and a group
    of NaN has no winner."""
    context = node.ctx
    rank = _ir.make("cast", (_ir.group_element(context, "GroupRank"),), "f64")
    none = _ir.const(context, _NO_WINNER, _RANK)
    holds = _ir.make("eq", (node.operands[0], node))
    first = _ir.group_reduce(
        "min", _ir.make_node("Select", _RANK, [holds, rank, none]), "tree"
    )
    return _ir.make("eq", (rank, first))


def scan_tangent(node: Any, tangent: Any) -> Any:
    """The tangent of the group scan ``node``: the scan of the tangents (only a running sum is linear)."""
    op = node.field_text("op")
    if op != "sum":
        raise HawkError(
            f"a group scan of {op!r} has no forward derivative; only the running sum is linear"
        )
    return _ir.group_scan(op, tangent, node.field_flag("inclusive"))


def tangent_group(
    group: Any,
    tangent_scratches: dict,
    derive: Callable[[Any], Any | None],
    context: Any,
) -> Any:
    """The group of the tangent kernel built from the primal ``group``.

    ``tangent_scratches`` maps each floating primal scratch plane to its tangent plane. ``derive(sink)`` gives the tangent
    sink of one body sink (``None`` for a sink with none). The body keeps its order: a scratch store stays, its tangent
    store follows it, and a barrier is the same barrier.
    """
    body: list = []
    for sink in _ir.group_body(group):
        if _ir.is_a(sink, "update") and sink.field_text("role") == "scratch":
            body.append(sink)
        elif not _ir.is_a(sink, "sink"):  # a barrier
            body.append(sink)
            continue
        made = derive(sink)
        if made is not None:
            body.append(made)
    keys = group.field_nodes("key")
    return _ir.group(
        context,
        group.field_text("membership"),
        body,
        [*group.field_nodes("scratch"), *tangent_scratches.values()],
        key=keys[0] if keys else None,
        count=group.field_int("count"),
        grid=_ir.field_pairs(group, "grid"),
        axes=group.field_names("axes"),
        max_size=group.field_int("max_size"),
    )


# --------------------------------------------------------------------------
# reverse mode

_INDEX = _ir.TensorType((), "i32")


def reduce_adjoint(node: Any, adjoint: Any) -> list[tuple[Any, Any]]:
    """The ``(operand, adjoint)`` pairs the group-wide reduce or scan ``node`` sends back, given the adjoint of its result.

    The result of a sum is the same on every lane, so each lane's operand gets the group sum of the adjoints of all lanes. A
    minimum or maximum sends that sum to the winner lane alone (:func:`winner_lane`). A running sum sends lane ``s`` the sum
    of the adjoints of the lanes that read it: the total minus the inclusive running sum, plus lane ``s``'s own adjoint when
    the scan includes it (the host writes no exclusive scan, so only the inclusive one is used). ``any`` and ``all`` are decisions and send nothing.
    """
    (operand,) = node.operands
    op = node.field_text("op")
    if _ir.is_a(node, "group_scan"):
        if op != "sum":
            raise HawkError(
                f"a group scan of {op!r} has no reverse derivative; only the running sum is linear"
            )
        total = _ir.group_reduce("sum", adjoint, "lane")
        through = _ir.group_scan("sum", adjoint, True)
        # the sum of the adjoints of the lanes past ``s``; a scan that includes its own lane sends ``s`` the lane ``s`` too
        past = _ir.make("sub", (total, through))
        return [
            (
                operand,
                _ir.make("add", (past, adjoint))
                if node.field_flag("inclusive")
                else past,
            )
        ]
    if op == "sum":
        return [(operand, _ir.group_reduce("sum", adjoint, node.field_text("order")))]
    if op not in ("min", "max"):
        return []
    ttype = _ir.type_of(adjoint)
    summed = _ir.group_reduce("sum", adjoint, "lane")
    won = _ir.make_node(
        "Select", ttype, [winner_lane(node), summed, _ir.zero_like(node.ctx, ttype)]
    )
    return [(operand, won)]


def _is_scratch_store(sink: Any) -> bool:
    """Whether the body sink ``sink`` stores into a scratch plane."""
    return _ir.is_a(sink, "update") and sink.field_text("role") == "scratch"


def _is_barrier(sink: Any) -> bool:
    """Whether the body sink ``sink`` is a barrier: it is neither an output commit nor a scratch store."""
    return not _is_scratch_store(sink) and not _ir.is_a(sink, "sink")


def _phases(group: Any) -> list[list[Any]]:
    """The body sinks of ``group`` split at its barriers, in order."""
    phases: list[list[Any]] = [[]]
    for sink in _ir.group_body(group):
        if _is_barrier(sink):
            phases.append([])
        else:
            phases[-1].append(sink)
    return phases


def _rank_offset(index: Any) -> int | None:
    """``k`` when the node ``index`` is ``rank + k`` (``rank``, ``rank + k``, ``k + rank`` or ``rank - k``), else ``None``."""
    if _ir.is_a(index, "group_rank"):
        return 0
    if not (_ir.is_a(index, "op") and index.field_text("kind") in ("add", "sub")):
        return None
    left, right = index.operands
    sign = 1 if index.field_text("kind") == "add" else -1
    if _ir.is_a(left, "group_rank") and _ir.is_a(right, "const"):
        return sign * right.field_int("literal")
    if sign == 1 and _ir.is_a(right, "group_rank") and _ir.is_a(left, "const"):
        return left.field_int("literal")
    return None


def _shifted_rank(context: Any, shift: int) -> Any:
    """The node ``rank + shift`` (an integer)."""
    rank = _ir.group_element(context, "GroupRank")
    if shift == 0:
        return rank
    return _ir.make("add", (rank, _ir.const(context, shift, _INDEX)))


def _read(plane: Any, index: Any, edge: str) -> Any:
    """The read of the scratch plane ``plane`` at ``index`` under the edge rule ``edge``."""
    return _ir.make_node("At", _ir.type_of(plane), [plane, index], edge=edge)


def gather_contributions(sent: Any, offset: int, edge: str) -> Any:
    """What each slot receives, as the slot's own lane reads it, from the per-lane contributions ``sent`` (a scratch plane
    holding the contribution of the lane at its own rank) of the reads at ``rank + offset`` under the edge rule ``edge``.

    The slot ``r`` is read by the lanes ``s`` with ``edge(s + offset) = r``, the inverse of the read:

    * ``wrap``: exactly one lane, ``(r - offset) mod size``, which a wrapping read of ``sent`` at ``rank - offset`` finds;
    * ``zero``: the lane ``r - offset`` when it lies in the group, else none, which a zero read of ``sent`` finds;
    * ``clamp``: the lane ``r - offset`` as for ``zero``, plus every lane the clamp folded onto the end slot: the last slot
      also receives all lanes ``s >= size - 1 - offset`` when ``offset > 0``, and the first slot all lanes ``s <= -offset``
      when ``offset < 0``, summed in rank order with a group sum of the contributions masked to those lanes.
    """
    context = sent.ctx
    own = _read(sent, _shifted_rank(context, 0), "clamp")
    if offset == 0:
        return own
    if edge == "wrap":
        return _read(sent, _shifted_rank(context, -offset), "wrap")
    interior = _read(sent, _shifted_rank(context, -offset), "zero")
    if edge == "zero":
        return interior
    rank = _ir.group_element(context, "GroupRank")
    size = _ir.group_element(context, "GroupSize")
    zero = _ir.zero_like(context, _ir.type_of(own))
    if offset > 0:
        last = _ir.make("sub", (size, _ir.const(context, 1, _INDEX)))
        # the lanes that clamp onto the last slot are those from ``size - 1 - offset`` on
        folded = _ir.make(
            "ge", (rank, _ir.make("sub", (last, _ir.const(context, offset, _INDEX))))
        )
        end = _ir.make("eq", (rank, last))
    else:
        folded = _ir.make("le", (rank, _ir.const(context, -offset, _INDEX)))
        end = _ir.make("eq", (rank, _ir.const(context, 0, _INDEX)))
    masked = _ir.make_node("Select", _ir.type_of(own), [folded, own, zero])
    return _ir.make_node(
        "Select",
        _ir.type_of(own),
        [end, _ir.group_reduce("sum", masked, "lane"), interior],
    )


def _scratch_reads(sinks: Iterable[Any], scratches: dict) -> list[Any]:
    """The reads of the floating scratch planes ``scratches`` (by name) in the expressions of ``sinks``, each once."""
    return [
        node
        for node in _ir.post_order(sinks)
        if _ir.is_a(node, "at") and node.operands[0].field_text("name") in scratches
    ]


def _refuse_unsupported(group: Any, scratches: dict, phases: list[list[Any]]) -> None:
    """Refuse by name what the reverse of the group cannot hold: a scratch plane stored in two places or away from the
    lane's own rank, or read at an index that is not the lane's rank plus a constant."""
    stored: set[str] = set()
    for sink in (s for phase in phases for s in phase if _is_scratch_store(s)):
        name = sink.field_text("name")
        if name in scratches and name in stored:
            raise HawkError(
                f"the scratch plane {name!r} is stored in more than one place; the reverse derivative keeps one version of "
                "each scratch plane, so store it once"
            )
        stored.add(name)
        if name in scratches and _rank_offset(sink.operands[1]) != 0:
            raise HawkError(
                f"the scratch plane {name!r} is stored away from the lane's own rank; the reverse derivative needs "
                "`t[g.rank] = ...`"
            )
    for read in _scratch_reads((s for phase in phases for s in phase), scratches):
        if _rank_offset(read.operands[1]) is None:
            raise HawkError(
                f"the scratch plane {read.operands[0].field_text('name')!r} is read at an index that is not the lane's rank "
                "plus a constant; the reverse derivative needs `t[g.rank]` or `t.at(g.rank + k, edge=...)`"
            )


def reverse_group(
    group: Any,
    leaves: Sequence[Any],
    seed: Callable[[Any], Any],
    run_pass: Callable[[Sequence[tuple[Any, Any]]], Callable[[Any], Any | None]],
    finish: Callable[[dict], list],
    derived: Callable[[str], str],
    taken: set[str],
) -> Any:
    """The group of the reverse kernel built from the primal ``group``.

    The body recomputes every scratch store of the primal (barriers kept), then takes the phases last to first. A phase seeds
    the value of each output sink with ``seed(sink)`` and of each floating scratch store with the slot's adjoint, and
    ``run_pass(seeds)`` runs the reverse over the phase's expressions and returns the adjoint of any node, or ``None``. The
    adjoint of every ``leaves`` entry is summed over the phases and handed to ``finish``, whose sinks (the gradient
    commits) close the body: they are evaluated after the last barrier, when every adjoint scratch plane is complete, which
    is also what each phase's own read of its seeds finds. ``derived(name)`` names the adjoint plane of a scratch plane and
    ``taken`` holds the names in use.
    """
    context = group.ctx
    primal_scratch = {s.field_text("name"): s for s in group.field_nodes("scratch")}
    floating = {n: s for n, s in primal_scratch.items() if has_tangent(s)}
    phases = _phases(group)
    _refuse_unsupported(group, floating, phases)
    adjoint_plane = {n: tangent_scratch(s, derived(n)) for n, s in floating.items()}
    rank = _ir.group_element(context, "GroupRank")

    def barrier() -> Any:
        """A barrier node of its own: the library reads a barrier repeated as one node as a single barrier."""
        return _ir.group_element(context, "Barrier")

    def store(plane: Any, value: Any) -> Any:
        """The store of ``value`` into the lane's own slot of ``plane``."""
        return _ir.make_node(
            "Update",
            _ir.type_of(value),
            [value, rank],
            name=plane.field_text("name"),
            op="store",
            role="scratch",
            edge="unchecked",
            axes=_ir.field_pairs(plane, "axes"),
        )

    body: list = [
        store(p, _ir.zero_like(context, _ir.type_of(p))) for p in adjoint_plane.values()
    ]
    for phase in phases:
        body.extend(s for s in phase if _is_scratch_store(s))
        body.append(barrier())
    sent_planes: list = []
    totals: dict = {}
    for phase in reversed(phases):
        seeds = []
        for sink in phase:
            if not has_tangent(sink):
                continue
            if _is_scratch_store(sink):
                held = _read(adjoint_plane[sink.field_text("name")], rank, "clamp")
                seeds.append((sink.operands[0], held))
            else:
                seeds.append((sink.operands[0], seed(sink)))
        adjoint_of = run_pass(seeds)
        for leaf in leaves:
            got = adjoint_of(leaf)
            if got is not None:
                held = totals.get(leaf)
                totals[leaf] = got if held is None else _ir.make("add", (held, got))
        sent: dict[str, list] = {}
        met: set = set()
        for read in _scratch_reads(phase, floating):
            name = read.operands[0].field_text("name")
            # reads made apart that agree in plane, offset and edge are one read to the adjoint tables (they share one
            # accumulated adjoint), so each is sent once, never once per node
            form = (name, _rank_offset(read.operands[1]), read.field_text("edge"))
            got = adjoint_of(read)
            if got is None or form in met:
                continue
            met.add(form)
            plane = tangent_scratch(
                floating[name], _fresh(derived(name), len(sent_planes), taken)
            )
            sent_planes.append(plane)
            body.append(store(plane, got))
            gathered = gather_contributions(
                plane, _rank_offset(read.operands[1]), read.field_text("edge")
            )
            sent.setdefault(name, []).append(gathered)
        if not sent:
            continue
        body.append(barrier())
        for name, parts in sent.items():
            total = parts[0]
            for part in parts[1:]:
                total = _ir.make("add", (total, part))
            plane = adjoint_plane[name]
            body.append(
                store(plane, _ir.make("add", (_read(plane, rank, "clamp"), total)))
            )
        body.append(barrier())
    body.extend(finish(totals))
    keys = group.field_nodes("key")
    return _ir.group(
        context,
        group.field_text("membership"),
        body,
        [*primal_scratch.values(), *adjoint_plane.values(), *sent_planes],
        key=keys[0] if keys else None,
        count=group.field_int("count"),
        grid=_ir.field_pairs(group, "grid"),
        axes=group.field_names("axes"),
        max_size=group.field_int("max_size"),
    )


def _fresh(base: str, number: int, taken: set[str]) -> str:
    """A plane name ``<base>_in<number>`` that no plane in ``taken`` holds."""
    name = f"{base}_in{number}"
    while name in taken:
        name = f"{name}_"
    taken.add(name)
    return name
