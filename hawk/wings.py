# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Wings hawk loads: installed distributions that publish a wing package under the entry-point group ``<family>.wings``
(``raptor.wings``, ``aether.wings``, ``eagle.wings``). A wing is addressed by ``(family, name)`` and printed ``<family>.wings.<name>``.
Its manifest (``wing.json``) starts with ``family``, ``name`` and ``parent`` (a wing of the same family, or ``None``); loading a wing
loads its chain from the root down, and the node kinds of the chain accumulate (a name is defined once in a chain). A wing declares in
its manifest (``MANIFEST["entries"]``, module -> names) what hawk calls; :func:`view` serves exactly those names, so hawk reaches a
wing through its declared entry list and nothing else.

Refused by name at load: a parent that is not installed, a manifest that states another family than the group it is declared under, a
cycle, and two distributions that declare one name in a family."""

from __future__ import annotations

import importlib
import importlib.metadata
from functools import cache
from types import SimpleNamespace

from ._errors import HawkError
from .variants import entry_points  # noqa: F401  (the one door: variants.py holds it, stdlib only)


def address(family: str, name: str) -> str:
    return f"{family}.wings.{name}"


def _label(family: str, name: str) -> str:
    return f"{family} wing {name}"


def _dist(ep) -> str:
    try:
        return str(ep.dist.metadata["Name"])
    except Exception:             # noqa: BLE001
        return "<unreadable distribution>"


def _find(family: str, name: str):
    """The one entry point of the group ``<family>.wings`` that is named ``name``."""
    found = [e for e in entry_points(f"{family}.wings") if e.name == name]
    owners = sorted({_dist(e) for e in found})
    if len(owners) > 1:
        raise HawkError(f"the {_label(family, name)} is declared by more than one distribution ({', '.join(owners)}): refused")
    if not found:
        raise HawkError(f"the {_label(family, name)} is not installed: install the distribution {family}-{name}")
    return found[0]


def _up(family: str, name: str) -> list:
    """``[(name, package)]`` of the wing ``name`` and its parents, leaf first; each manifest is checked against the group it is declared under."""
    out, seen = [], []
    while name is not None:
        if name in seen:
            raise HawkError(f"the {family} wings form a cycle: {' -> '.join(seen + [name])}")
        seen.append(name)
        try:
            ep = _find(family, name)
        except HawkError as e:
            if not out:
                raise
            raise HawkError(f"the {_label(family, out[-1][0])} builds on the {_label(family, name)}: {e}") from e
        mod = ep.load()
        stated = mod.MANIFEST.get("family")
        if stated != family:
            raise HawkError(f"the {_label(family, name)} is declared by {_dist(ep)} under the group {family}.wings, but its manifest states "
                            f"the family {stated!r}: refused")
        out.append((name, mod))
        name = mod.MANIFEST.get("parent")
    return out


@cache
def wing(family: str, name: str):
    """The wing package ``name`` of ``family``; its chain is loaded root to leaf and its kinds join the front end's registry.

    One load per wing (the function's own memo; a refusal is not remembered, so a later install is seen)."""
    from .dialects import KINDS

    up = _up(family, name)
    defined: dict = {}
    chain_kinds = [(each, getattr(mod, "KINDS", None)) for each, mod in reversed(up)]       # a wing that declares no node kinds has none
    for each, declared in chain_kinds:
        for kind in () if declared is None else declared.kinds:
            if kind in defined:
                raise HawkError(f"the kind {kind!r} is defined twice in the chain of the {_label(family, name)}: by the {_label(family, defined[kind])} "
                                f"and by the {_label(family, each)}")
            defined[kind] = each
    for _, declared in chain_kinds:
        if declared is not None:
            KINDS.kinds.update(declared.kinds)
            KINDS.classes.update(declared.classes)
    return up[0][1]


def row_wings(wings, isa: str | None = None) -> list:
    """The ``(family, name)`` leaves a row's ``wings`` map names. A leaf may be a map from the host's isa to a name: for a build on the
    host ``isa`` the leaf of that isa is the one (a build requires, loads and keys only it); with ``isa`` ``None`` no host is in play and
    every value counts (the registry-wide sets, :func:`device_view`)."""
    out = []
    for family, leaf in (wings or {}).items():
        if isinstance(leaf, dict) and isa is not None:
            if isa not in leaf:
                raise HawkError(f"the {family} wing of this row names no wing for isa {isa!r} (it names {sorted(leaf)})")
            leaf = leaf[isa]
        out += [(family, n) for n in ([] if leaf is None else [leaf] if isinstance(leaf, str) else list(leaf.values()))]
    return out


def chain(wing_keys) -> list:
    """``(family, name)`` of the wings in ``wing_keys`` and of each one's parents, each once, parents before their children."""
    out: list = []
    for family, name in wing_keys:
        wing(family, name)
        for each, _ in reversed(_up(family, name)):
            if (family, each) not in out:
                out.append((family, each))
    return out


def require(wing_keys, backend: str) -> None:
    """Refuse by name a build of ``backend`` when a wing it needs, or one that wing builds on, is not installed or cannot be imported."""
    for family, name in wing_keys:
        try:
            chain([(family, name)])
        except (HawkError, ImportError) as e:
            raise HawkError(f"the {backend!r} backend requires the {_label(family, name)}, which is not available ({e}); "
                            f"install the distribution {family}-{name}") from e


class _View:
    """The declared entries of module ``part`` of wing ``name``; the wing loads at the first entry read."""

    def __init__(self, family: str, name: str, part: str) -> None:
        self._family, self._name, self._part, self._ns = family, name, part, None

    def __getattr__(self, entry: str):
        if self._ns is None:
            mod = wing(self._family, self._name)
            declared = mod.MANIFEST["entries"][self._part]
            sub = importlib.import_module(f"{mod.__name__}.{self._part}")
            self._ns = SimpleNamespace(**{n: getattr(sub, n) for n in declared})
        try:
            return getattr(self._ns, entry)
        except AttributeError:
            raise HawkError(f"{entry!r} is not a declared entry of the module {self._part!r} of the {_label(self._family, self._name)}") from None


@cache
def view(family: str, name: str, part: str) -> _View:
    """The declared entries of module ``part`` of wing ``name`` of ``family``, as a namespace (loaded on first use; one view per triple)."""
    return _View(family, name, part)


def device_view(part: str) -> _View:
    """The declared entries of module ``part`` of the wing a device backend names, or one it builds on, beyond the ones the host backends use (the
    form only a device lowering leaves behind); refused by name when no installed default backend names one."""
    from . import registry

    rows = [registry.traits(b) for b in registry.backend_ids(default=True)]
    # no host is in play here: every value of an isa map counts (the shared set), see :func:`row_wings`
    shared = set(chain([k for t in rows if not t.get("device") for k in row_wings(t.get("wings"))]))
    named = [k for t in rows if t.get("device") for k in row_wings(t.get("wings"))]
    if not set(named) - shared:
        raise HawkError("no device backend is installed that names a wing of its own")
    found = sorted(k for k in set(chain(named)) - shared if part in wing(*k).MANIFEST.get("entries", {}))
    if len(found) != 1:
        raise HawkError(f"the device backends' own wings hold the module {part!r} in {len(found)} places: {[address(*k) for k in found]}")
    return view(found[0][0], found[0][1], part)
