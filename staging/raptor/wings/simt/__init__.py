# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The SIMT wing: the warp form of a lowering, as a package of its own (its own tests, its own distribution).

It declares the node kinds a lowering to that form leaves behind, by DATA (``wing.json``, in this package): :class:`WarpLane` and
:class:`WarpStore` (:mod:`raptor.wings.simt.lane`). It imports the standard library,
raptor and the base wing only; the wing a build runs on arrives as an argument."""

from __future__ import annotations

import json
from importlib import resources

from raptor.ir.kinds import Registry, declare, make_kind
from raptor.ir.types import TensorType

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


def WarpLane():
    """The lane's place in its warp, ``0 .. WARP - 1`` (a terminal: it binds no slot)."""
    return make_node("warp_lane", TensorType((), "i32"))


def WarpStore(name, value, ttype=None):
    """The commit of a value into the Mutable ``name``; a sliced one: element ``j`` goes to ``j * WARP + lane``."""
    return make_node("warp_store", value.ttype if ttype is None else ttype, (value,), name=name)
