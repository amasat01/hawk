# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The base wing of raptor (its package is lowering until its rename): the neutral lowering of a kernel (thread box, team stages, placement of wide values, ordered logs) as a package of its own.

It declares the node kinds a lowering to that form leaves behind, by DATA (``wing.json``, in this package):
:class:`TeamStore` and its kin are the stage
boundary of a team (:mod:`raptor.wings.base.team`); :class:`BoxStore` is the thread box (:mod:`raptor.wings.base.box`). It imports the standard library
and raptor only; the wing a build runs on arrives as an argument."""

from __future__ import annotations

import json
from importlib import resources

from raptor.ir.kinds import Registry, declare, make_kind
from raptor.ir.nodes import Leaf, SampleIndex
from raptor.ir.types import TensorType

#: The index type of a scratch plane: ``lanes x n^2`` passes 2^31.
INDEX = TensorType((), "i64")

#: The wing's manifest (data): its kinds, its spelling modules and the entry list its user calls.
MANIFEST = json.loads(resources.files(__name__).joinpath("wing.json").read_text(encoding="utf-8"))

#: The registry of the wing's own kinds (a value; the front end merges it into the one it holds).
KINDS = Registry()
for _kind in MANIFEST["kinds"]:
    declare(KINDS, {"ext": MANIFEST["name"], "dialect": MANIFEST.get("dialect", ""), **_kind}, first_party=(MANIFEST["name"],))


def make_node(kind_: str, ttype, operands=(), /, **fields):
    """A node of the declared kind ``kind_``."""
    return make_kind(KINDS, kind_, ttype, operands, **fields)


# Constructors: the presentation of the declared kinds (what the passes call). Each builds a node of the kind through raptor.


def TeamStore(name, value):
    """A per-lane stage's store of ``value`` into its own slice of the scratch plane ``name``."""
    return make_node("team_store", TensorType((), value.ttype.dtype), (value, SampleIndex(INDEX)), name=name)


def TeamCopy(name, value):
    """A :func:`TeamStore` of a plain read of an input plane, in a stage that holds nothing else (a block per lane copies it)."""
    return make_node("team_copy", TensorType((), value.ttype.dtype), (value, SampleIndex(INDEX)), name=name)


def LogStore(name, value, index, target, width, kept):
    """The store of one adjoint contribution of lane ``i`` into slot ``index`` of an ordered log plane."""
    return make_node("log_store", TensorType((), value.ttype.dtype), (value, index), name=name, target=target, width=width,
                     kept=kept)


def ScatterLog(name, value, index, target):
    """The store of lane ``i``'s key or value into slot ``index`` of a log plane of an ordered ``hawk.scatter``."""
    return make_node("scatter_log", TensorType((), value.ttype.dtype), (value, index), name=name, target=target)


def TeamRead(plane, ttype, index=None):
    """The value of ``ttype`` held in slice ``index`` (the lane's own by default) of the scratch plane ``plane``."""
    scalar = TensorType((), ttype.dtype)
    return make_node("team_read", ttype, (Leaf("table_read", "lookup", plane, scalar),
                                          SampleIndex(INDEX) if index is None else index))


def BoxStore(sink, value, lane):
    """The own-column store of one lane of the box, at ``lane``."""
    return make_node("box_store", sink.ttype, (value, lane), name=sink.name, role=sink.role)
