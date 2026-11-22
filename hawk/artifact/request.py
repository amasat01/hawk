# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The request one kernel sends to a lowering library, built from the kernel, the settings of the build and the machines the
libraries stated.

Every member is read from the objects the build holds; nothing is guessed from a record and nothing is retried. The members the
library defaults (a group of one, no planes, settings at their defaults) are left out, as the library's own writer leaves them
out. The machine answers travel as given. :func:`sinks_of` lists the nodes a request names; their
positions in the dump come from :func:`hawk._ir.dump_numbers`."""

from __future__ import annotations

from types import MappingProxyType

from .. import _ir
from ..errors import HawkError
from ..ext import PLAIN, DEFAULT_GUARD, DATA_ONLY, DEFAULT_KIND
from ..trace import steps as _steps
from ..trace.decl import Plane
from ..targets import TARGETS
from ..trace.kernel import _declared


def _camel(word: str) -> str:
    """``plane_in`` as ``PlaneIn``: the way the request spells a name of a closed set."""
    return "".join(part.capitalize() for part in word.split("_"))


def _slot(slot, axes=()) -> dict:
    """The request's spelling of a declared slot; ``axes`` are those of the plane the slot binds."""
    out = {"role": _camel(slot.role), "name": slot.name, "dtype": _camel(slot.ttype.dtype), "shape": list(slot.ttype.shape)}
    if slot.ttype.tag is not None:
        out["tag"] = slot.ttype.tag
    if axes:
        out["axes"] = _axes(axes)
    return out


def _leaf_axes(walk) -> dict:
    """``{(role, name): the axes the walk's plane leaf states}`` for each plane with axes."""
    leaves = (n for n in _ir.walk_order(walk) if _ir.is_a(n, "leaf"))
    return {(n.field_text("role"), n.field_text("name")): _ir.field_pairs(n, "axes") for n in leaves
            if _ir.field_pairs(n, "axes")}


def _mixes(body) -> tuple:
    """The branch-mix plane of each ``auto`` dispatch the body's walk binds."""
    bound = _ir.slot_of(body.walk)
    found = (("lookup", name) for name in _ir.mix_slots(body.walk).values())
    return tuple(_ir.Slot(role, name, _ir.slot_type(body.walk, role, name)) for role, name in found if (role, name) in bound)


def _member(body, name: str, position: dict) -> dict:
    """A body of the request: the nodes it writes and how its walk is taken. A kernel declares its terminated planes and the
    branch-mix plane of each ``auto`` dispatch (the library derives the companion plane of a compensated sum from the walk); a
    body with no planes of its own declares the slots its walk binds."""
    planes = getattr(body, "planes", None)
    if planes is not None:
        declared = [_slot(slot) for slot in _declared(planes)]
        declared += [_slot(slot) for slot in _mixes(body)]
    else:
        axes = _leaf_axes(body.walk)
        declared = [_slot(_ir.Slot(role, slot, _ir.slot_type(body.walk, role, slot)), axes.get((role, slot), ()))
                    for role, slot in _ir.arg_spec(body.walk) if role in _ir.LEAF_ROLES]
    return {
        "name": name,
        "sinks": [position[sink] for sink in body.sinks],
        "declared": declared,
        "exempt": [_ir.FUSED_STEPS_PLANE],
    }


def _kind(kind) -> dict:
    """The kind of the kernel as the request states it; a policy of another form is refused."""
    guard = kind.guard
    if guard is DEFAULT_GUARD:
        out = "Default"
    elif guard is DATA_ONLY:
        out = "DataOnly"
    else:
        out = {"Masks": list(guard.names)}
    if kind.sink is PLAIN or (kind.sink.id == "plain" and kind.sink.compensation is None):
        sink = "Plain"
    elif kind.sink.compensation is None:
        raise HawkError(f"kind {kind.slug!r}: a sink policy of another form ({kind.sink.id!r}) is not built")
    else:
        sink = {"Compensated": {"plane": kind.sink.compensation, "output_names": list(kind.output.names),
                                **({} if kind.output.slot is None else {"output_slot": kind.output.slot})}}
    return {"guard": out, "active_set": bool(guard.active_set), "sink": sink,
            "lanes": None if kind.lanes is None else {"bits": kind.lanes.bits, "prefix": kind.lanes.prefix}}


def _axes(pairs) -> list:
    return [{**({} if label is None else {"label": label}), **({} if extent is None else {"extent": extent})} for label, extent, *_ in pairs]


def _table_payload(plane, pattern) -> dict:
    """The ``payload`` member of a broadcast row whose plane is a table: the extents of the table's own leading dimensions.

    A broadcast table is stored as (its own components, then the lane axes it varies along), so its dimensions beyond those
    lane axes are the payload: empty for a table of scalars. A plane that is not a table states none.
    """
    if not isinstance(plane, Plane) or plane.form != "table":
        return {}
    kept = sum(flag is True for flag in pattern)
    own = plane.axes[:max(len(plane.axes) - kept, 0)]
    return {"payload": [extent for _, extent, *_ in own]}


def _settings(settings, sizes) -> dict:
    """The settings of the build that differ from the library's defaults; ``sizes`` are the layout sizes a test states."""
    out = {}
    for key, value in (("fold_form", settings.form), ("wide_form", settings.wide_form), ("linalg_form", settings.linalg_form)):
        if value != "auto":
            out[key] = _camel(value)
    if settings.launch_bounds is not None:
        out["launch_bounds"] = [int(n) for n in settings.launch_bounds]
    if settings.fold_rows:
        out["fold_rows"] = settings.fold_rows
    if settings.warps != 1:
        out["warps"] = settings.warps
    if settings.team_threads:
        out["team_threads"] = settings.team_threads
    if settings.fast_entries:
        out["fast_entries"] = True
    if sizes is not None:
        out["exports"] = [int(n) for n in sizes]
    return out


def sinks_of(kernel) -> list:
    """Every node the kernel's request can name or the lowering reads: the sinks of the kernel, of its single step and of its
    members, and of theirs in turn."""
    out = list(kernel.sinks)
    one = getattr(kernel, "one_step", None)
    if one is not None:
        out += sinks_of(one)
    for member in getattr(kernel, "members", None) or ():
        if hasattr(member, "sinks") and hasattr(member, "walk"):
            out += sinks_of(member)
    return out


#: The request's spelling of each role of a derived unit.
_ROLE_WORDS = MappingProxyType({"tape": "Tape", "tape_fused": "TapeFused", "step_vjp": "StepVjp", "fused_vjp": "FusedVjp"})


def _part(kernel, *, settings, sizes, position: dict) -> dict:
    """The members of a kernel's request that belong to the kernel alone, which a derived kernel states for itself as well: its
    single step, group, planes, build settings, steps, lane space, shared planes and broadcast pattern."""
    out = {}
    one = getattr(kernel, "one_step", None)
    if one is not None:
        out["one_step"] = _member(one, one.name, position)
    if getattr(kernel, "group", 1) != 1:
        out["group"] = kernel.group
    planes = getattr(kernel, "planes", None) or {}
    if planes:
        out["planes"] = list(planes)
    stated = _settings(settings, sizes)
    if stated:
        out["settings"] = stated
    steps = getattr(kernel, "steps", None)
    if isinstance(steps, int) and steps >= 1:
        out["steps"] = steps
    elif steps == "auto":
        out["steps"], out["steps_max"] = "auto", _steps.AUTO_K_MAX
    lanes = _axes(getattr(kernel, "lane_space", ((None, None),)))   # the default the sidecar writer takes
    if lanes:
        out["lane_space"] = lanes
    shared = [n for n, p in planes.items() if isinstance(p, Plane) and p.shared]
    if shared:
        out["shared"] = shared
    bound = {name for _, name in _ir.arg_spec(kernel.walk)}
    broadcast = [{"plane": n, "axes": [flag is True for flag in pattern], **_table_payload(planes.get(n), pattern)}
                 for plane, pattern in (getattr(kernel, "broadcast", None) or {}).items()
                 for n in (plane, *(t for t, p in kernel.tangents.items() if p == plane)) if n in bound]
    if broadcast:
        out["broadcast"] = broadcast
    return out


def request(kernel, *, name: str, scalar: str, kind, targets, settings, position: dict, machines: dict,
            sizes=None, exports_block: bool = True, derived=()) -> dict:
    """The request of ``kernel`` for ``targets`` as a dict: ``name`` the unit name wanted, ``scalar`` the scalar mode's name,
    ``kind`` the kind the kernel is built under, ``settings`` the :class:`hawk.build_settings.Settings` of the build, ``position``
    the handle of each node (:func:`hawk._ir.dump_numbers`), ``machines`` ``{target: the machine answer of its OSPREY}`` and ``sizes`` the layout sizes a test states. The texts carry the
    self-check block of a library unit unless ``exports_block`` is false, which a build asks: its units are linked with the text of the
    set, which carries the block once. ``derived`` are the :class:`hawk.diff.recording.DerivedUnit` built with ``kernel`` in one
    library, each stated by its own kernel members and its facts."""
    kind = kind if kind is not None else DEFAULT_KIND
    members = getattr(kernel, "members", None)
    if members and hasattr(members[0], "sinks"):
        body = {"Chain": [_member(m, m.name, position) for m in members]}
    else:
        body = {"Plain": _member(kernel, name, position)}
    rows = {t: TARGETS[t] for t in targets}
    out = {"v": 1, "name": name, "scalar": _camel(scalar), "targets": [row.target for row in rows.values()], "kind": _kind(kind),
           "body": body}
    out |= _part(kernel, settings=settings, sizes=sizes, position=position)
    if derived:
        out["derived"] = [_derived_entry(unit, kind=kind, settings=settings, sizes=sizes, position=position) for unit in derived]
    out["machine"] = {row.target: machines[t] for t, row in rows.items()}
    if not exports_block:
        out["exports_block"] = False
    return out


def _derived_entry(unit, *, kind, settings, sizes, position: dict) -> dict:
    """One entry of the request's ``derived`` list: the unit's own body and kernel members, its role and its facts."""
    entry = {"member": _member(unit.kernel, unit.kernel.name, position), "role": _ROLE_WORDS[unit.role], "kind": _kind(kind)}
    entry |= _part(unit.kernel, settings=settings, sizes=sizes, position=position)
    return entry | {key: value for key, value in unit.facts.items() if value is not None}
