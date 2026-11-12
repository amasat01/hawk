# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Form dialects: node classes that exist only because of a lowered form.

A dialect module is imported by the passes that build its nodes, never by the core IR or the
core renderer. A node class carries its own render entry: ``render_sink(self, r, target)`` for
a sink, ``render_expr(self, r, materialised)`` for an expression, each calling the writer
through the member it names."""

from raptor.ir.kinds import Registry, declare, is_a as is_a, kind_name as kind_name, make_kind, trait as trait, load_manifest   # noqa: F401
from .._errors import HawkError
from .. import variants
from ..registry import register_spelling, spelling_of

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


def spell_node(r, node, target=None):
    """Spell a declared node through the active writer: the spelling its declaration names, found by name; the node itself
    carries no render method. A sink's text is emitted by the spelling; an expression's is returned."""
    dialect, name, args = node.decl.spelling or (None, None, ())
    if dialect is None:
        raise HawkError(f"kind {node.decl.name!r} declares no spelling")
    given = [node if a == "node" else target if a == "target" else r._ref(node.operands[int(a[3:])]) for a in args]
    return spelling(r, dialect, name)(*given)


def spells(writer: str, dialect: str, name: str):
    """Register ``fn(w, r, *args)`` as ``writer``'s spelling of ``dialect``'s ``name`` (``w`` is the writer, ``r`` the renderer);
    a second definition of the same spelling is an error."""
    def register(fn):
        register_spelling(writer, dialect, name, fn)
        return fn
    return register


def spelling(r, dialect: str, name: str):
    """The active writer's spelling of ``dialect``'s ``name``, as ``fn(w, r, *args)`` bound to the writer; refuses by name.
    A writer's spellings of a dialect load, on the first miss, from the module its extension's manifest names."""
    w = r.writer
    fn = spelling_of(w.id, dialect, name)
    if fn is None:
        raise HawkError(f"the {w.id!r} writer has no spelling for {dialect}.{name}")
    return lambda *args: fn(w, r, *args)
