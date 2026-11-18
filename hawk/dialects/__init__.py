# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Form dialects: kinds that exist only because of a lowered form, declared into a context.

A dialect's kinds are declared in the context whose graph uses them (:func:`declare_kinds`); the
context states them to every call that reads its nodes, so hawk keeps no registry of them."""

from __future__ import annotations

import json

from .. import _ir
from .._errors import HawkError


def _first_party() -> tuple:
    """The names that own first-party kinds: hawk and the first-party OSPREYs (the registry's table)."""
    from .. import targets

    return ("hawk", *(d.removeprefix("osprey-") for d in targets.DISTRIBUTIONS.values()))


def declare_kinds(context, group: dict) -> None:
    """Declare every kind of a manifest ``kinds`` arm (``{name, dialect, kinds: [...]}``; ``name`` is the OSPREY name) in ``context``."""
    from .. import wings

    name = group.get("name")
    owned = {*_first_party(), *(ep.name for family in ("raptor", "aether", "eagle") for ep in wings.entry_points(f"{family}.wings"))}
    if name in owned:
        raise HawkError(f"the name {name!r} belongs to a first-party piece: its kinds are declared by its own distribution, not through this door")
    _ir.declare_dialect(context, json.dumps(group), _first_party())
