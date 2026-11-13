# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Form dialects: node classes that exist only because of a lowered form.

A dialect module is imported by the passes that build its nodes, never by the core IR or the
core renderer. A node class carries its own render entry: ``render_sink(self, r, target)`` for
a sink, ``render_expr(self, r, materialised)`` for an expression, each calling the writer
through the member it names."""

from raptor.ir.kinds import Registry, declare, is_a as is_a, kind_name as kind_name, make_kind, has_feather as has_feather, load_manifest   # noqa: F401
from .._errors import HawkError
from .. import variants

#: The ONE default registry of declared kinds (and primitives) a hawk process uses; raptor holds none.
KINDS = Registry()
FIRST_PARTY = ("hawk", "hawk-c", "hawk-cuda")


def declare_kinds(group: dict) -> tuple:
    """Declare every kind of a manifest ``kinds`` arm (``{name, dialect, kinds: [...]}``; ``name`` is the extension id) into :data:`KINDS`."""
    from .. import components

    name = group.get("name")
    owned = {*FIRST_PARTY, *(ep.name for ep in variants.entry_points(components.GROUP))}
    if name in owned:
        raise HawkError(f"the name {name!r} belongs to a first-party piece: its kinds are declared by its own distribution, not through this door")
    for row in group.get("kinds", ()):
        if isinstance(row, dict) and "." in str(row.get("name", "")) and row["name"].partition(".")[0] != name:
            raise HawkError(f"kind {row['name']!r}: a kind is spelled '<name>.<kind>' with the name of the manifest that declares it ({name!r})")
    return load_manifest(KINDS, group, first_party=FIRST_PARTY)


def make_node(kind_: str, ttype, operands=(), /, **fields):
    """A node of the declared kind ``kind_``."""
    return make_kind(KINDS, kind_, ttype, operands, **fields)
