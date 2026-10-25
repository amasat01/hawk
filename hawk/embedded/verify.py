# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C target's verify pass: everything checked on the hawk IR before a
line of C is rendered.

:func:`refuse_unsupported` refuses, by name, every feature the C target does
not build (:data:`UNSUPPORTED`). :func:`ir_checks` never refuses: it computes
the per-sample budgets, classifies every runtime index as ``proven`` or
``clamped``, reports each floating-point domain site, and describes the
interface one row per slot. :func:`apply_caps` is where a profile's opt-in
caps turn that report into a refusal. Every refusal has one shape::

    '<kernel>': the C target refuses node <pos> (<kind>) under <rule>: <why>; <fix>

A node position is its index in ``walk.order``, or ``loop<i>.<k>`` for the
``k``-th node of the body of the loop at ``i`` (nested: ``loop<i>.<k>.<j>``).
"""

from __future__ import annotations

from collections.abc import Iterator
from math import prod

from ..emit.spelling import binding_name
from ..ext import DEFAULT_KIND
from ..ir import HawkError
from ..ir.cost import fp64_ops, kernel_cost
from ..ir.loop_nodes import Loop, LoopIndex, TapeRead
from ..ir.nodes import (
    FINISHED_PLANE,
    AccumWrite,
    At,
    Const,
    Dispatch,
    Op,
    Sink,
    TapeWrite,
    WideWrite,
    is_param_cell,
)
from ..ir.walk import canonical_nodes
from ..targets import FEATURES, TARGETS
from ..targets.features import INDEX_INTS
from ..types import INT_PLANE_DTYPES

_FLOATS = ("f64", "f32")
#: Bytes one element of a dtype occupies in the C build (an index Int is 64-bit).
_SIZEOF = {"f64": 8, "f32": 4, "i32": 8, "i64": 8, "int32": 4, "uint32": 4, "bool": 1}
#: The named fields of ``BranchCost.row_tuple()``, the ``ops`` budget's keys.
_OPS = ("arith32", "arith64", "special32", "special64", "loads", "gathers",
        "chain32", "chain64", "chainSpecial32", "chainSpecial64", "stores")
_ARITH = ("arith32", "arith64", "special32", "special64")
#: Index kinds where the two builds part: the C++ spells them infix (signed
#: overflow, and a division by zero or ``MIN / -1``, is undefined), the C
#: defines them through ``hawk_i64_*``. An Int32/UInt32 operation and every
#: Index ``//``, ``%`` and shift go through the host's own helper too
#: (``hawk/emit/wrapper.py`` ``_INT_HELPERS``), so they are no site.
_INT_UB = ("add", "sub", "mul", "neg", "div")
_INPUT_ROLES = ("per_sample", "vec_in", "mat_in", "lookup", "wide_in", "uniform",
                "nsamples")
_LENGTH_ROLES = ("lookup", "wide_in", "wide_out", "accum_out")


def _refusal(name, pos, kind, rule, why, fix) -> HawkError:
    return HawkError(f"{name!r}: the C target refuses node {pos} ({kind}) under "
                     f"{rule}: {why}; {fix}")


def _top(sinks) -> tuple:
    """THE traversal's top-level node sequence for ``sinks``: the renderer's
    own (:func:`hawk.ir.walk.canonical_nodes`), so a position here is the
    position the C text names. Nothing here re-enumerates the walk."""
    return canonical_nodes(tuple(sinks))[0]


def _every_node(top) -> Iterator[tuple]:
    """``(pos, node, enclosing loops)`` for every node of the top-level
    sequence ``top`` (:func:`_top`) and, recursively, every loop body (its
    nodes, then its per-iteration sinks). Dispatch branches are operands, so
    they are already in ``top``."""
    seen: set = set()

    def visit(pos, node, loops):
        if id(node) in seen:
            return
        seen.add(id(node))
        yield pos, node, loops
        if isinstance(node, Loop):
            prefix = pos if isinstance(pos, str) else f"loop{pos}"
            for k, child in enumerate((*node.body_nodes(), *node.body_sinks)):
                yield from visit(f"{prefix}.{k}", child, (*loops, node))

    for i, node in enumerate(top):
        yield from visit(i, node, ())


def positions(sinks) -> dict[int, int | str]:
    """``id(node) -> position`` for every node the traversal of ``sinks``
    reaches, loop bodies included: the numbering every report site and
    refusal uses (an int at the top level, ``loop<i>.<k>`` inside a body)."""
    return {id(node): pos for pos, node, _ in _every_node(_top(sinks))}


#: The features the C target refuses, from the shared table (:data:`hawk.targets.FEATURES`).
UNSUPPORTED = {f: entry for f, entry in FEATURES.items()
               if f not in TARGETS["c"].features}


def refuse_unsupported(name: str, kernel, kind, walk, sinks) -> None:
    """Refuse the first :data:`UNSUPPORTED` feature ``kernel`` uses, by name."""
    if kind is None:
        kind = getattr(kernel, "kind", None)
    nodes = list(_every_node(_top(sinks)))
    for feature, (predicate, (why, fix)) in UNSUPPORTED.items():
        hit = predicate(kernel, kind, walk, nodes)
        if hit is not None:
            pos, what = hit
            raise _refusal(name, pos, what, f"unsupported.{feature}", why, fix)


# --------------------------------------------------------------------------
# a. bounds and budgets


def _trips(loop: Loop) -> int:
    inner = sum(_trips(n) for n in loop.body_nodes() if isinstance(n, Loop))
    return loop.trip * (1 + inner)


def _max_trips(top) -> tuple[int, list]:
    """Worst-case trips per sample: top-level loops summed (dispatch branches
    included, since their loops are top-level), nested loops multiplied; a
    runtime-count loop contributes its static trip, which the C clamps to."""
    loops = [{"node": f"loop{i}", "trips": _trips(n)}
             for i, n in enumerate(top) if isinstance(n, Loop)]
    return sum(entry["trips"] for entry in loops), loops


def _bytes(ttype) -> int:
    return prod(ttype.shape) * _SIZEOF[ttype.dtype]


def _static_bytes(nodes) -> dict:
    """Tapes (``trip`` x carry, per slot some ``TapeRead`` names) and carries
    (every loop's, nested included), mirroring the renderer's tape scan."""
    taped: dict = {}
    carries = 0
    for _, node, _ in nodes:
        if isinstance(node, TapeRead):
            taped[(id(node.loop), node.slot)] = (
                node.loop.trip * _bytes(node.loop.carries[node.slot].ttype))
        if isinstance(node, Loop):
            carries += sum(_bytes(c.ttype) for c in node.carries)
    tapes = sum(taped.values())
    return {"tapes": tapes, "carries": carries, "temporaries": None,
            "estimate": tapes + carries, "total": None}


# --------------------------------------------------------------------------
# b. index sites


def _loop_of(index, loops) -> Loop | None:
    for loop in reversed(loops):
        if index is loop.index:
            return loop
    for loop in reversed(loops):
        if (loop.index.depth, loop.index.name) == (index.depth, index.name):
            return loop
    return None


def _span(loop: Loop) -> tuple[int, int]:
    last = loop.start + (loop.trip - 1) * loop.step
    return min(loop.start, last), max(loop.start, last)


def _in_range(index, extent: int, loops) -> tuple[bool, str]:
    if isinstance(index, Const) and isinstance(index.literal, int):
        return (0 <= index.literal < extent,
                f"literal {index.literal} against extent {extent}")
    loop = _loop_of(index, loops) if isinstance(index, LoopIndex) else None
    if loop is None or loop.trip == 0:
        return False, f"a runtime index against extent {extent}"
    lo, hi = _span(loop)
    return (0 <= lo and hi < extent,
            f"loop index over [{lo}, {hi}] against extent {extent}")


def _tape_ordinal(node: TapeRead, loops) -> tuple[bool, str]:
    fwd = node.loop
    loop = _loop_of(node.index, loops) if isinstance(node.index, LoopIndex) else None
    if loop is None or loop.trip == 0:
        return False, "the tape ordinal is not a loop index"
    if loop.step % fwd.step or (loop.start - fwd.start) % fwd.step:
        return False, "the reading loop's range is not the taped loop's"
    lo, hi = ((v - fwd.start) // fwd.step for v in _span(loop))
    lo, hi = min(lo, hi), max(lo, hi)
    return (0 <= lo and hi < fwd.trip,
            f"ordinals [{lo}, {hi}] of a {fwd.trip}-row tape")


def _tape_row(sink: TapeWrite) -> tuple[bool, str]:
    row = sink.row
    if (isinstance(row, Op) and row.kind == "mod" and isinstance(row.operands[1], Const)
            and isinstance(row.operands[1].literal, int)):
        window = row.operands[1].literal
        return (window > 0 and window * sink.width <= sink.ttype.shape[0],
                f"row mod {window} of {sink.ttype.shape[0] // sink.width} rows")
    return False, "the row is not a floor modulo of a literal window"


def _index_site(node, loops) -> dict | None:
    if isinstance(node, At):
        if is_param_cell(node):
            return {"kind": "param_cell", "status": "proven",
                    "reason": "a parameter cell is read at index 0", "length": None}
        plane = node.plane
        return {"kind": "table_read", "status": "clamped",
                "reason": "a runtime index into a runtime-length plane",
                "length": binding_name(plane.role, plane.name) + "_len"}
    if isinstance(node, Op) and node.kind in ("component_at", "set_component_at"):
        extent = node.operands[0].ttype.shape[0]
        ok, why = _in_range(node.operands[1], extent, loops)
        return {"kind": node.kind, "status": "proven" if ok else "clamped",
                "reason": why, "length": extent}
    if isinstance(node, TapeRead):
        (ok, why), length = _tape_ordinal(node, loops), node.loop.trip
    elif isinstance(node, TapeWrite):
        (ok, why), length = _tape_row(node), node.ttype.shape[0] // node.width
    elif isinstance(node, (WideWrite, AccumWrite)) and node.index is not None:
        return {"kind": "scatter", "status": "clamped",
                "reason": "a scattered sink's runtime row",
                "length": binding_name(node.role, node.name) + "_len"}
    elif isinstance(node, Dispatch):
        return {"kind": "dispatch", "status": "clamped",
                "reason": f"the selector is clamped to [0, {node.K})",
                "length": node.K}
    else:
        return None
    return {"kind": node.kind, "status": "proven" if ok else "clamped",
            "reason": why, "length": length}


def _index_sites(nodes) -> list:
    sites = []
    for pos, node, loops in nodes:
        site = _index_site(node, loops)
        if site is not None:
            sites.append({"node": pos, **site})
    return sites


def index_status(report: dict, pos) -> str:
    """The ``proven``/``clamped`` status of the index site at ``pos``."""
    for site in report["indices"]:
        if site["node"] == pos:
            return site["status"]
    raise HawkError(f"node {pos} is not an index site of this report")


def site_status(report: dict, sinks, node) -> str:
    """The ``proven``/``clamped`` status of the index site ``node`` (a node
    reached from ``sinks``), looked up through :func:`positions`."""
    pos = positions(sinks).get(id(node))
    if pos is None:
        raise HawkError(f"{node!r} is not a node of this walk")
    try:
        return index_status(report, pos)
    except HawkError:
        raise HawkError(f"node {pos} ({node.kind}) is not an index site of this "
                        "report") from None


# --------------------------------------------------------------------------
# c. floating-point domain

_UNKNOWN, _NONNEG, _POSITIVE = "unknown", "nonneg", "positive"
_RANK = {_UNKNOWN: 0, _NONNEG: 1, _POSITIVE: 2}
_LOWER = {v: k for k, v in _RANK.items()}
#: Kinds whose sign is read off their operands' signs.
_FROM_OPERANDS = ("add", "mul", "div", "max", "min", "clip", "splat", "sqrt",
                  "select", "primitive")
#: Domain sites: kind -> (the operand index checked, the sign it needs).
_SITES = {"div": (1, _POSITIVE), "sqrt": (0, _NONNEG), "rsqrt": (0, _POSITIVE),
          "log": (0, _POSITIVE), "log2": (0, _POSITIVE), "log10": (0, _POSITIVE),
          "log1p": (0, _POSITIVE), "pow": (0, _POSITIVE), "acosh": (0, None),
          "acos": (0, "unit"), "asin": (0, "unit"), "atanh": (0, "unit")}


def _own_sign(node, signs: list) -> str:
    """The sign ``node`` is proven to have, for non-NaN operands: a NaN
    propagates identically on both sides (``clip`` passes it through), so a
    claim says what holds wherever the value is a number."""
    kind = node.kind
    ops = node.operands
    if isinstance(node, Const):
        literal = node.literal
        if isinstance(literal, bool) or not isinstance(literal, (int, float)):
            return _UNKNOWN
        return _POSITIVE if literal > 0 else _NONNEG if literal == 0 else _UNKNOWN
    least = _LOWER[min(_RANK[s] for s in signs)] if signs else _UNKNOWN
    if kind in ("exp", "exp2"):
        return _NONNEG          # IEEE underflow: exp(-1000) is +0
    if kind in ("norm", "abs") or (kind == "dot" and ops[0] is ops[1]):
        return _NONNEG
    if kind == "sqrt":
        return signs[0]         # sqrt of a negative is NaN: unknown stays unknown
    if kind == "mul" and ops[0] is ops[1]:
        return _POSITIVE if signs[0] == _POSITIVE else _NONNEG
    if kind in ("mul", "min", "splat", "primitive"):
        return least if kind != "primitive" else signs[0]
    if kind == "add":
        if least == _UNKNOWN:
            return _UNKNOWN
        return _POSITIVE if _POSITIVE in signs else _NONNEG
    if kind == "div":
        return signs[0] if signs[1] == _POSITIVE else _UNKNOWN
    if kind == "max":
        return _LOWER[max(_RANK[s] for s in signs)]
    if kind == "clip":
        return signs[1]
    if kind == "select":
        return _LOWER[min(_RANK[signs[1]], _RANK[signs[2]])]
    return _UNKNOWN


def _sign(node, memo: dict) -> str:
    """``positive``/``nonneg``/``unknown`` of ``node``, bottom-up over its
    operand subgraph, memoised by identity; iterative (deep chains)."""
    stack = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if id(current) in memo:
            continue
        kids = current.operands if current.kind in _FROM_OPERANDS else ()
        if not expanded:
            stack.append((current, True))
            stack.extend((k, False) for k in kids if id(k) not in memo)
            continue
        memo[id(current)] = _own_sign(current, [memo[id(k)] for k in kids])
    return memo[id(node)]


def _unit_clip(node) -> bool:
    if not (isinstance(node, Op) and node.kind == "clip"):
        return False
    lo, hi = node.operands[1], node.operands[2]
    return (isinstance(lo, Const) and isinstance(hi, Const)
            and -1 <= lo.literal and hi.literal <= 1)


def _fp_site(node, memo: dict) -> dict | None:
    if not (isinstance(node, Op) and node.kind in _SITES
            and node.ttype.dtype in _FLOATS):
        return None
    which, need = _SITES[node.kind]
    operand = node.operands[which]
    if need == "unit":
        safe = _unit_clip(operand)
        return {"op": node.kind, "status": "safe" if safe else "unknown",
                "reason": ("the operand is clipped to [-1, 1]" if safe else
                           f"{node.kind} needs an operand clipped to literal [-1, 1]")}
    if need is None:
        return {"op": node.kind, "status": "unknown",
                "reason": f"no safe form of the {node.kind} operand is recognised"}
    sign = _sign(operand, memo)
    if _RANK[sign] >= _RANK[need]:
        return {"op": node.kind, "status": "safe", "reason": f"the operand is {sign}"}
    return {"op": node.kind, "status": "unknown",
            "reason": f"{node.kind} needs a {need} operand; its sign is {sign}"}


def _fp_domain(nodes) -> list:
    memo: dict = {}
    out = []
    for pos, node, _ in nodes:
        site = _fp_site(node, memo)
        if site is not None:
            out.append({"node": pos, **site})
    return out


# --------------------------------------------------------------------------
# d. interface contract


def _guard_masks(walk, kind) -> list[str]:
    """The fields of the masks that guard every store, exactly as the renderer
    computes them (:meth:`hawk.emit.render._Renderer._guard_slot`): each mask
    the kind's guard names counts only if bound in the ``terminated`` role. A
    finishing kernel binds its mask there too, so it is guarded the same way."""
    kind = DEFAULT_KIND if kind is None else kind
    return [binding_name("terminated", m) for m in kind.guard.names
            if ("terminated", m) in walk.slot_of]


def _interface(walk, nodes, kind) -> list:
    scattered = {(n.role, n.name) for _, n, _ in nodes
                 if isinstance(n, Sink) and n.index is not None}
    counters = {FINISHED_PLANE} | {f"{d.name}_mix" for d in walk.dispatches
                                   if d.policy == "auto"}
    single = set(walk.param_planes) | counters
    mask = walk.finish[0] if walk.finish else None
    # a masked sample never reaches its own-column store and keeps the prior
    # value (the host's design), so a guarded own column is caller-initialised
    bound = _guard_masks(walk, kind)
    guard = f"guard: {', '.join(bound)}"
    rows = []
    for role, name in walk.arg_spec:
        ttype = walk.slot_types[(role, name)]
        layout = "scalar" if ttype.shape == () else "soa"
        if role in ("uniform", "nsamples"):
            layout = "value"
        if role == "lookup" and name in counters:
            direction, init, reason = "inout", True, "counter (the caller zeroes it)"
        elif role in _INPUT_ROLES:
            direction, init, reason = "in", True, "input"
        elif role == "mutable" and name in walk.prior_reads:
            direction, init, reason = "inout", True, "prior read"
        elif role == "out":
            direction, init = "inout", True
            reason = "one row written per launch, rest kept"
        elif role == "terminated":
            direction, init = ("inout" if name == mask else "in"), True
            reason = "mask (the finish sets it)" if name == mask else "mask"
        elif (role, name) in scattered:
            direction, init = "inout", True
            reason = ("+= into prior (the caller zeroes it)" if role == "accum_out"
                      else "scattered rows written, rest kept")
        elif role == "accum_out" and bound:
            direction, init, layout = "inout", True, "partials"
            reason = ("one partial per sample, length n_samples; masked samples "
                      f"keep theirs ({guard}): the caller zeroes it, then reduces")
        elif role == "accum_out":
            direction, init, layout = "out", False, "partials"
            reason = "one partial per sample, length n_samples; the caller reduces it"
        elif bound:
            direction, init = "inout", True
            reason = f"masked samples keep the prior value ({guard})"
        else:
            direction, init, reason = "out", False, "written every sample"
        rows.append({
            "field": binding_name(role, name), "role": role, "name": name,
            "direction": direction, "dtype": ttype.dtype, "shape": list(ttype.shape),
            "layout": layout,
            "runtime_length": role in _LENGTH_ROLES and name not in single,
            "init_by_caller": init, "guard": list(bound), "reason": reason})
    return rows


# --------------------------------------------------------------------------
# the report and the caps


def ir_checks(name: str, sinks, walk, kind, profile) -> dict:
    """The verify report of one kernel (never refuses; see :func:`apply_caps`)."""
    top = _top(sinks)
    nodes = list(_every_node(top))
    max_trips, loops = _max_trips(top)
    cost = kernel_cost(walk)
    ints = INDEX_INTS + INT_PLANE_DTYPES
    return {
        "interface": _interface(walk, nodes, kind),
        "budgets": {
            "max_trips": max_trips,
            "loops": loops,
            "ops": {field: getattr(cost, field) for field in _OPS},
            # kernel_cost's loop arm already counts each body x trip.
            "fp64_ops": fp64_ops(top),
            "static_bytes": _static_bytes(nodes),
        },
        "indices": _index_sites(nodes),
        "fp_domain": _fp_domain(nodes),
        "fp_equality_sites": [
            pos for pos, n, _ in nodes if isinstance(n, Op) and n.kind in ("eq", "ne")
            and any(o.ttype.dtype in _FLOATS for o in n.operands)],
        "int_division_sites": [
            pos for pos, n, _ in nodes if isinstance(n, Op)
            and n.kind in ("div", "floordiv", "mod") and n.ttype.dtype in ints],
        "int_ub_sites": [
            pos for pos, n, _ in nodes if isinstance(n, Op)
            and n.kind in _INT_UB and n.ttype.dtype in INDEX_INTS],
        "unsupported_checked": sorted(UNSUPPORTED),
    }


def apply_caps(name: str, report: dict, profile) -> None:
    """Refuse the first cap ``profile`` sets that ``report`` exceeds."""
    budgets = report["budgets"]
    if profile.max_trips is not None and budgets["max_trips"] > profile.max_trips:
        worst = max(budgets["loops"], key=lambda entry: entry["trips"],
                    default={"node": "kernel", "trips": budgets["max_trips"]})
        raise _refusal(
            name, worst["node"], "loop", "budget.trips",
            f"{budgets['max_trips']} worst-case loop trips per sample exceed "
            f"max_trips={profile.max_trips} ({worst['node']} runs {worst['trips']})",
            "shorten the range or raise the cap")
    ops = sum(budgets["ops"][field] for field in _ARITH)
    if profile.max_ops is not None and ops > profile.max_ops:
        raise _refusal(name, "kernel", "ops", "budget.ops",
                       f"{ops:g} arith + special ops per sample exceed "
                       f"max_ops={profile.max_ops}",
                       "split the kernel or raise the cap")
    static = budgets["static_bytes"]
    used = static["total"] if static["total"] is not None else static["estimate"]
    if profile.max_stack_bytes is not None and used > profile.max_stack_bytes:
        raise _refusal(name, "kernel", "stack", "budget.stack",
                       f"{used} static bytes (tapes {static['tapes']}, carries "
                       f"{static['carries']}) exceed max_stack_bytes="
                       f"{profile.max_stack_bytes}",
                       "shorten the taped loop or raise the cap")
    if profile.index_policy == "refuse":
        for site in report["indices"]:
            if site["status"] == "clamped":
                raise _refusal(name, site["node"], site["kind"], "index.unproven",
                               f"the index is not provably in range ({site['reason']})",
                               "index with a static loop index or a literal, or "
                               "export with index_policy='clamp'")
    if profile.fp_domain == "refuse":
        for site in report["fp_domain"]:
            if site["status"] == "unknown":
                raise _refusal(name, site["node"], site["op"], "fp.domain",
                               site["reason"],
                               "bound the operand into the domain (abs, a positive "
                               "offset, clip) or export with fp_domain='record'")
