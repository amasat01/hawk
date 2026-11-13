# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The loader of variant data: one literal per variant (``data.py``), read without executing it.

A variant is found the way every piece is: an installed distribution declares an entry point of the group ``raptor.variants``
(name -> a package that holds ``data.py`` and ``raptor_variant.h``); the data is read from that installed package, never from a
path relative to hawk. The name is ONE identifier segment (``[a-z][a-z0-9_]*``), and the data's own ``"name"`` equals it.

A variant is DATA. It may name ONE base variant (``"extends"``, one string); the base has none. The merge is one level: a key the
extension holds replaces the base's, a key that is a dict is merged one key deep. The loaded data is READ-ONLY (nested dicts and
lists too): shared code takes the loaded variant as an argument and copies it explicitly (:func:`derive`) to change a number.

THE SCHEMA, checked once at load (:data:`SCHEMA`): ``name`` is required; the other top-level keys are the known groups
below, each of its type; a key that is not in the schema is refused by name, with the file.

REQUIRED BY NAME (the contract for a backend's variant): a number a lowering or a writer reads is stated by the variant it runs on, and a
missing one is refused by name where it is read: the fold threshold ``form.rolled_fold_min`` (``raptor.ir.fold.exceeds``), ``linalg.max_dim``,
``launch.warp_size``, ``host_wide``. There is no silent default ("nothing exceeds it"), no empty variant that stands for one, and no other
backend's number is borrowed. A backend that has no such number builds with ``emit_form='static'`` or ``'rolled'``, which read none."""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.metadata
import re
from pathlib import Path
from types import MappingProxyType

GROUP = "raptor.variants"
NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
#: The top-level keys a variant may hold and their types; a further data file ``<stem>.py`` of the variant is the key ``<stem>``.
SCHEMA = MappingProxyType({"name": str, "extends": str, "isa": str, "default_arch": str, "uniform_run": int,
          "steps_reverse": dict, "linalg": dict, "lanes": dict, "cells": dict, "scatter": dict, "form": dict, "host_wide": dict,
          "thread_box": dict, "launch": dict, "fold": dict, "arch_features": dict, "fusion_host": dict, "wide_classes": dict,
          "cards": dict})


class _Frozen(dict):
    """A dict that refuses every change (a loaded variant is shared by every build of the process)."""

    def _no(self, *a, **k):
        raise TypeError("variant data is read-only: copy it (hawk.variants.derive) to change a number")

    __setitem__ = __delitem__ = update = pop = popitem = clear = setdefault = __ior__ = _no

    def __deepcopy__(self, memo):
        import copy

        return {copy.deepcopy(k, memo): copy.deepcopy(v, memo) for k, v in self.items()}

    def __reduce__(self):
        return (dict, (dict(self),))


class _FrozenList(list):
    def _no(self, *a, **k):
        raise TypeError("variant data is read-only: copy it (hawk.variants.derive) to change a number")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = __iadd__ = __imul__ = _no

    def __deepcopy__(self, memo):
        import copy

        return [copy.deepcopy(v, memo) for v in self]

    def __reduce__(self):
        return (list, (list(self),))


def entry_points(group: str) -> list:
    """The entry points of ``group`` over every installed distribution: the one door every piece (a variant, a component, an
    extension) is found through. Standard library only (``tools/gen_variant_header.py`` loads this file by path)."""
    try:
        found = importlib.metadata.entry_points()
        return list(found.select(group=group)) if hasattr(found, "select") else list(found.get(group, ()))
    except Exception:             # a damaged distribution: read each one alone and skip the ones that cannot be read
        out = []
        for dist in importlib.metadata.distributions():
            try:
                out += [ep for ep in dist.entry_points if ep.group == group]
            except Exception:     # noqa: BLE001 - the registry refuses that distribution by name for its own group
                continue
        return out


def _dist_of(ep) -> str:
    try:
        return str(ep.dist.metadata["Name"])
    except Exception:             # noqa: BLE001
        return "<unreadable distribution>"


def discover(group: str, name: str, kind: str, error=ValueError):
    """The one entry point named ``name`` of ``group``; ``error`` (with a sentence that says what to install) when there is none
    or when two distributions declare it."""
    found = [e for e in entry_points(group) if e.name == name]
    owners = sorted({_dist_of(e) for e in found})
    if len(owners) > 1:
        raise error(f"the {kind} {name!r} is declared by more than one distribution ({', '.join(owners)}) under {group!r}: refused")
    if not found:
        raise error(f"the {name!r} {kind} is not installed: install a distribution that declares the {group!r} entry point {name!r}")
    return found[0]


def _freeze(value):
    if isinstance(value, dict):
        return _Frozen({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(v) for v in value)
    return value


def _segment(name, what: str = "variant name") -> str:
    """``name`` when it is ONE identifier segment; a path, a separator, ``..`` or any other spelling is refused by name."""
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError(f"{what} {name!r} is refused: a variant is named by ONE identifier segment [a-z][a-z0-9_]* "
                         "(no path, no separator, no '..')")
    return name


@functools.lru_cache(maxsize=None)
def variant_home(name: str) -> Path:
    """The installed package directory of variant ``name``: its entry point's value is ``<top package>[.<sub package>]``; the top
    package is located without importing it."""
    from importlib import util

    ep = discover(GROUP, _segment(name), "variant", ValueError)
    parts = ep.value.partition(":")[0].strip().split(".")
    if ":" in ep.value or not all(p.isidentifier() for p in parts):
        raise ValueError(f"variant {name!r}: its entry point value {ep.value!r} is not a package name")
    spec = util.find_spec(parts[0])
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(f"variant {name!r}: the package {parts[0]!r} its entry point names is not importable")
    home = Path(next(iter(spec.submodule_search_locations))).joinpath(*parts[1:])
    if not (home / "data.py").is_file():
        raise ValueError(f"variant {name!r}: no data.py in {home}")
    return home


def _literal(path: Path):
    try:
        return ast.literal_eval(path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"{path}: a variant's data file holds one literal, nothing else (no def, no expression); "
                         f"{type(e).__name__}: {e}") from None


def _read(name: str) -> dict:
    home = variant_home(name)
    path = home / "data.py"
    data = _literal(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the literal is not a dict")
    for extra in sorted(home.glob("*.py")):      # a further data file of the variant: its literal, under its file name
        if extra.name not in ("data.py", "__init__.py"):
            data[extra.stem] = _literal(extra)
    unknown = sorted(set(data) - set(SCHEMA))
    if unknown:
        raise ValueError(f"{path}: unknown top-level key {unknown[0]!r} (the keys are {sorted(SCHEMA)})")
    if "name" not in data:
        raise ValueError(f"{path}: no 'name'")
    for key, kind in SCHEMA.items():
        if key in data and type(data[key]) is not kind:
            raise ValueError(f"{path}: {key!r} is a {type(data[key]).__name__}, the schema says {kind.__name__}"
                             + (" (a variant extends exactly ONE base: a name)" if key == "extends" else ""))
    if data["name"] != name:
        raise ValueError(f"{path}: the data says name {data['name']!r}, but the entry point names it {name!r}: refused")
    if "extends" in data:
        _segment(data["extends"], f"{path}: base")
    return data


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    """The data of variant ``name``, its base's underneath (read-only)."""
    own = _read(name)
    base_name = own.get("extends")
    if base_name is None:
        return _freeze(own)
    base = _read(base_name)
    if "extends" in base:
        raise ValueError(f"variant {name!r}: base {base_name!r} extends another; the depth is one")
    merged = dict(base)
    for key, value in own.items():
        merged[key] = {**base[key], **value} if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return _freeze(merged)


def reset() -> None:
    """Forget what was loaded (the installed distributions changed)."""
    load.cache_clear()
    variant_home.cache_clear()


def location(name: str) -> Path:
    """The installed package directory of variant ``name`` (the one that holds its header)."""
    return variant_home(name)


def header_digest(name: str) -> str:
    """The sha256 of the CONTENT of variant ``name``'s generated header: what a compile key holds instead of the header's path."""
    path = variant_home(name) / "raptor_variant.h"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


class IncludeFlag(str):
    """The ``-I<dir>`` flag of a variant's header directory. It is the flag on the compile line (the path is where the header is
    installed); its :attr:`key_term` is what a compile key hashes in its place: the variant's NAME and the sha256 of the header's
    CONTENT, so the key does not depend on where the installed package lives."""

    key_term: str = ""


def include_flags(variant: dict) -> tuple:
    """The compile flags that hand a unit its variant's generated header (``raptor_variant.h`` in the installed package of
    the variant, written by ``tools/gen_variant_header.py``): one ``-I`` of that directory (an :class:`IncludeFlag`); none for a
    build with no variant."""
    name = variant.get("name") if variant else None
    if not name:
        return ()
    flag = IncludeFlag(f"-I{variant_home(name)}")
    flag.key_term = f"variant:{name}:{header_digest(name)}"
    return (flag,)


def named(entry, isa: str) -> str:
    """The variant name a backend manifest's ``variant`` entry gives for a host of instruction set ``isa``: the entry is a name,
    or a map from isa to name (a host backend)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isa in entry:
        return entry[isa]
    raise ValueError(f"the manifest variant {entry!r} names no variant for isa {isa!r}")


def derive(variant: dict, overrides: dict) -> dict:
    """A copy of ``variant`` with ``overrides`` merged in (one dict level deep, like an extension over its base): how a test or a
    kit run passes other data to code that takes a variant."""
    merged = dict(variant)
    for key, value in overrides.items():
        merged[key] = {**variant[key], **value} if isinstance(value, dict) and isinstance(variant.get(key), dict) else value
    return merged
