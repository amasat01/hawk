# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The cell layout of a reduction over samples: how a :class:`~raptor.ir.nodes.ReduceSamples` is stored to a plane. The core
states one scalar ``sample_sum`` (or one scalar accumulate); splitting a vector or matrix into one scalar plane per component is
a layout choice of this wing, and so are the compensation plane a summing sample sum carries and the branch-mix plane of an
``auto`` dispatch; :func:`walk` is the one door to the
canonical form."""

from __future__ import annotations

from dataclasses import replace

from raptor.errors import RaptorError
from raptor.ir.nodes import AccumWrite, Const, Finish, Loop, Op, ReduceSamples, SampleSum, Sink, has_feather
from raptor.ir.ops import make
from raptor.ir.types import Slot, TensorType
from raptor.ir.walk import canonical, dispatch_groups

from .dispatch import strategy

#: The reserved ``lookup`` plane a finishing kernel counts its newly finished samples into: ONE ``uint32`` cell.
FINISHED_PLANE = "finished_count"

#: The reserved ``lookup`` plane a ``steps="auto"`` kernel reads its trip count from: set on the device by the runner, never
#: by the host. The access class skips the read of it.
FUSED_STEPS_PLANE = "fused_steps"

#: The suffix of the companion plane a summing ``sample_sum`` keeps beside its target.
SAMPLE_SUM_COMP = "_comp"

#: The offsets plane rides the SAME role every other run-time-addressed
#: plane does (``lookup``), read like any other lookup table.
OFFSETS_ROLE = "lookup"

#: K+1 exclusive-prefix entries, the language index type's width (HAWK types an
#: integer leaf ``i64`` regardless of the emitted wire width).
OFFSETS_DTYPE = "i64"

def offsets_name(dispatch_name: str) -> str:
    """The offsets plane's slot name for dispatch group ``dispatch_name``."""
    return f"{dispatch_name}_offsets"


def mix_name(dispatch_name: str) -> str:
    """The K-counter ``lookup`` plane an ``auto`` dispatch counts the classes
    it picks into (one uint32 per branch, integer-atomic): the branch mix a
    launching runtime's chooser reads."""
    return f"{dispatch_name}_mix"


def class_name(dispatch_name: str) -> str:
    """The per-sample class plane a segmented unit writes (the clamped
    selector: the next sort's key; its bincount is the branch mix)."""
    return f"{dispatch_name}_class"


def stale_name(dispatch_name: str) -> str:
    """The per-sample mask a segmented unit sets where the selector no
    longer picks the unit's own branch (the sample sits in a stale run)."""
    return f"{dispatch_name}_stale"


def classify_name(kernel_name: str) -> str:
    """The classify kernel's name for segmented kernel ``kernel_name``."""
    return f"{kernel_name}__cls"


def _components(value) -> tuple:
    """The row-major component indices of a vector or matrix ``value``."""
    shape = value.ttype.shape
    return tuple([(r,) if len(shape) == 1 else (r, c)
                  for r in range(shape[0]) for c in range(shape[-1] if len(shape) == 2 else 1)])


def stored(reduction: ReduceSamples, name: str, ttype: TensorType | None = None) -> tuple:
    """The cell sinks that store ``reduction`` to the plane ``name``: one for a scalar, one per component (planes ``name_0``,
    ``name_1``, ...) for a vector or matrix."""
    value = reduction.operands[0]
    shape = value.ttype.shape
    if reduction.key is not None:
        return _keyed(reduction, name)
    if not shape:
        return (SampleSum(name, value, ttype, reduction.op),)
    scalar = TensorType((), value.ttype.dtype)
    index = _components(value)
    width = len(index)
    return tuple(SampleSum(f"{name}_{k}", Op("component", (value,), scalar, ix if len(shape) == 2 else ix[0]),
                           scalar, reduction.op, cell=name, columns=width)
                 for k, ix in enumerate(index))


def _keyed(reduction: ReduceSamples, name: str) -> tuple:
    """The accumulates of a reduction over some lane axes: entry ``k`` of the payload (row-major) adds into ``name`` at
    ``key + k * count``, so the plane is ``(*payload, *kept)``, payload first."""
    value = reduction.operands[0]
    shape = value.ttype.shape
    scalar = TensorType((), value.ttype.dtype)
    if not shape:
        return (AccumWrite(name, value, reduction.key, scalar, reduction.contract, reduction),)
    out = []
    for k, ix in enumerate(_components(value)):
        entry = Op("component", (value,), scalar, ix if len(shape) == 2 else ix[0])
        at = reduction.key if k == 0 else make("add", (reduction.key, make(
            "mul", (Const(k, reduction.count.ttype), reduction.count))))
        out.append(AccumWrite(name, entry, at, scalar, reduction.contract))
    return tuple(out)


def _companions(sinks) -> tuple:
    """The compensation planes of ``sinks``: one slot ``<name>_comp`` (the sum's role and type) per summing ``sample_sum``, to be
    handed to :func:`raptor.ir.walk.canonical` as declared slots. Refused when a sink already holds that plane."""
    held, stack = set(), list(sinks)
    while stack:
        node = stack.pop()
        if isinstance(node, Loop):
            stack.extend(node.body_sinks)
        elif isinstance(node, Sink):
            held.add((node.role, node.name))
    out = []
    for sink in sinks:
        if isinstance(sink, SampleSum) and sink.op == "sum":
            slot = Slot(sink.role, f"{sink.name}{SAMPLE_SUM_COMP}", sink.ttype)
            if (slot.role, slot.name) in held:
                raise RaptorError(f"sample_sum {sink.name!r}: its compensation plane {slot.name!r} collides with a plane of the same name")
            out.append(slot)
    return tuple(out)


def walk(sinks, declared=(), **kw):
    """THE way to take the canonical form of a set of sinks: :func:`raptor.ir.walk.canonical` over ``declared``, the compensation
    planes of the summing sample sums among ``sinks`` (:func:`_companions`) and the branch-mix plane of every ``auto`` dispatch, with
    the strategy word of each dispatch group filled in; a finishing kernel's finished-count plane is declared last, and the
    access class skips the read of the fused-steps word. ``kw`` is ``force_access``."""
    sinks = tuple(sinks)
    declared = (*declared, *_companions(sinks))
    kw = {**kw, "exempt_planes": (FUSED_STEPS_PLANE,)}
    finished = ((Slot("lookup", FINISHED_PLANE, TensorType((), "uint32")),)
                if any(isinstance(s, Finish) for s in sinks) else ())
    found = canonical(sinks, (*declared, *finished), **kw)
    groups = dispatch_groups(found)
    mixes = tuple(Slot("lookup", mix_name(name), TensorType((), "uint32"))
                  for name, node in groups.items() if has_feather(node, "counts_branch_mix"))
    if mixes:
        found = canonical(sinks, (*declared, *mixes, *finished), **kw)
    return replace(found, dispatches=tuple(replace(info, strategy=strategy(node)) for info, node in zip(found.dispatches, groups.values())))
