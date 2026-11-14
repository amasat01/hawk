# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Fills :class:`raptor.protocols.unit.LoweredUnit` from what lowering produced (:class:`~hawk.backends.Lowered`)."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from raptor.ir import IR_VERSION
from .._errors import HawkError
from raptor.ir.nodes import Node
from raptor.ir.types import TensorType
from raptor.protocols.unit import LoweredUnit, PlaneDecl
from raptor.ir.walk import canonical_nodes


def _plain(value, where: str):
    """``value`` as plain data: a dataclass becomes a mapping of its fields; a hawk object that is neither data nor a
    raptor object is a refusal naming where it sits."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, TensorType, Node)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_plain(v, where) for v in value)
    if isinstance(value, dict):
        return {k: _plain(v, where) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name), f"{where}.{f.name}") for f in fields(value)}
    raise HawkError(f"lowered unit: {where} holds a {type(value).__name__}, which is neither plain data nor a raptor object")


def frame_of(lowered) -> LoweredUnit:
    """The unit of ``lowered``: filled from the kernel, the scalar mode, the kind and the specs lowering made, and from the
    build settings in force. Builds no text and imports no backend."""
    from . import unit_cache
    from aether.wings.base.writer_aether import working_set_bytes
    from ..wings import view
    from ..trace.decl import Plane

    kernel, name, smode, kind = lowered.kernel, lowered.name, lowered.smode, lowered.kind
    members = getattr(kernel, "members", None)
    one_step = getattr(lowered, "one_step", None)
    sub = None if one_step is None else frame_of(type(lowered)(one_step, name, smode, kind, None, None, None, None, None, 1, lowered.spec))
    planes = tuple(PlaneDecl(n, p.form, p.ttype, *(_plain(getattr(p, f), f"planes.{n}.{f}") for f in
                   ("dims", "staged", "grid", "sorted", "shared", "meta")))
                   for n, p in kernel.planes.items() if isinstance(p, Plane))
    walk = kernel.walk

    spec = lowered.spec
    cfg = spec.settings
    values = {"form": cfg.form, "linalg_form": cfg.linalg_form, "wide_form": cfg.wide_form,
              "contraction": cfg.contraction, "launch_bounds": cfg.launch_bounds, "scalar": cfg.scalar,
              "device_only": cfg.device_only}

    # the cost is plain data, worked out here under the settings the build's spec holds
    cost = view("raptor", "base", "cost")
    branch = cost.kernel_cost(walk)
    costs = {"ops": branch.arith32 + branch.arith64 + branch.special32 + branch.special64,
             "fp64_ops": cost.fp64_ops(canonical_nodes(tuple(kernel.sinks))[0]), "working_set_bytes": working_set_bytes(kernel.sinks, spec.primary.get("form", {}).get("rolled_fold_min"), cfg.form)}

    return LoweredUnit(
        ir_version=IR_VERSION, name=name, walk_digest=walk.digest,
        source_digest=getattr(kernel, "source_digest", None), declaration_digest=unit_cache.declaration_digest(kernel),
        sinks=tuple(kernel.sinks), walk=walk, planes=planes, returns=_plain(getattr(kernel, "returns", None), "returns"),
        mode={"id": smode.id, "real": smode.real_spelling, "cells": smode.cells},
        kind={"slug": kind.slug, "guard": _plain(kind.guard, "kind.guard"), "sink": _plain(kind.sink, "kind.sink"),
              "lanes": _plain(kind.lanes, "kind.lanes")},
        form={"rows": lowered.rows, "group": getattr(kernel, "group", 1), "segment": _plain(lowered.segment, "segment"),
              "active": _plain(lowered.active, "active"), "box": _plain(lowered.box, "box"),
              "team": _plain(getattr(kernel, "team", None), "team"), "fused": lowered.body is not None,
              "members": None if members is None else tuple(m.name for m in members), "one_step": sub},
        settings={**values, "wide_class": spec.wide_class, "staged_rows": cfg.staged_rows},
        cost=costs)
