# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The loader of facts: the machine numbers a raptor wing states (``facts.py``, one literal), read without executing it.

A raptor wing is found the way every wing is (:mod:`hawk.wings`): an installed distribution declares an entry point of the group
``raptor.wings``; its facts are read as FILES of that installed package (``wing.json``, ``facts.py``, ``raptor_facts.h``), never from
a path relative to hawk, and reading them runs no wing code. The name is ONE identifier segment (``[a-z][a-z0-9_]*``).

The facts of a wing are RESOLVED along its chain, root to leaf (:func:`hawk.wings.lineage`, the one walk of a chain): a key the nearest
wing states wins, a key that is a dict is merged one key deep at each step, and a wing cannot remove a key. A wing states only what it
adds or changes. The loaded facts are READ-ONLY (nested dicts and lists too): shared code takes the loaded facts as an argument and
copies them explicitly (:func:`derive`) to change a number. The resolved facts carry ``name``: the leaf wing's name.

THE SCHEMA, checked once at load (:data:`SCHEMA`): the top-level keys are the known groups below, each of its type; a key that is not in
the schema is refused by name, with the file.

REQUIRED BY NAME (the contract for a backend's facts): a number a lowering or a writer reads is stated by the wing it runs on or by one
of its parents, and a missing one is refused by name where it is read: the fold threshold ``form.rolled_fold_min``
(``raptor.ir.fold.exceeds``), ``linalg.max_dim``, ``launch.warp_size``, ``host_wide``. There is no silent default ("nothing exceeds it"),
no empty wing that stands for one, and no other backend's number is borrowed. A backend that has no such number builds with
``emit_form='static'`` or ``'rolled'``, which read none."""

from __future__ import annotations

import ast
import functools
import hashlib
from pathlib import Path
from types import MappingProxyType

from . import wings

FAMILY = "raptor"
#: The top-level keys a wing's facts may hold and their types.
SCHEMA = MappingProxyType({"isa": str, "default_arch": str, "uniform_run": int,
          "steps_reverse": dict, "linalg": dict, "lanes": dict, "cells": dict, "scatter": dict, "form": dict, "host_wide": dict,
          "thread_box": dict, "launch": dict, "fold": dict, "arch_features": dict, "fusion_host": dict, "wide_classes": dict,
          "cards": dict})
#: A further data file of a wing, ``<stem>.py`` (one literal), is the key ``<stem>``.
DATA_FILES = ("cards",)


class _Frozen(dict):
    """A dict that refuses every change (loaded facts are shared by every build of the process)."""

    def _no(self, *a, **k):
        raise TypeError("facts are read-only: copy them (hawk.facts.derive) to change a number")

    __setitem__ = __delitem__ = update = pop = popitem = clear = setdefault = __ior__ = _no

    def __deepcopy__(self, memo):
        import copy

        return {copy.deepcopy(k, memo): copy.deepcopy(v, memo) for k, v in self.items()}

    def __reduce__(self):
        return (dict, (dict(self),))


class _FrozenList(list):
    def _no(self, *a, **k):
        raise TypeError("facts are read-only: copy them (hawk.facts.derive) to change a number")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = __iadd__ = __imul__ = _no

    def __deepcopy__(self, memo):
        import copy

        return [copy.deepcopy(v, memo) for v in self]

    def __reduce__(self):
        return (list, (list(self),))


def _freeze(value):
    if isinstance(value, dict):
        return _Frozen({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(v) for v in value)
    return value


@functools.lru_cache(maxsize=None)
def home(name: str) -> Path:
    """The installed folder of the raptor wing ``name`` (the one that holds its ``wing.json``, its ``facts.py`` and its header)."""
    return wings.home(FAMILY, name)


def chain(name: str) -> list:
    """The names of the raptor wing ``name`` and its parents, root first."""
    return [each for each, *_ in reversed(wings.lineage(FAMILY, wings.segment(name)))]


def _literal(path: Path):
    try:
        return ast.literal_eval(path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"{path}: a wing's facts file holds one literal, nothing else (no def, no expression); "
                         f"{type(e).__name__}: {e}") from None


def _read(folder: Path) -> dict:
    """The facts one wing states (none: ``{}``), checked against the schema."""
    path = folder / "facts.py"
    if not path.is_file():
        return {}
    data = _literal(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the literal is not a dict")
    for stem in DATA_FILES:
        if (folder / f"{stem}.py").is_file():
            data[stem] = _literal(folder / f"{stem}.py")
    unknown = sorted(set(data) - set(SCHEMA))
    if unknown:
        raise ValueError(f"{path}: unknown top-level key {unknown[0]!r} (the keys are {sorted(SCHEMA)})")
    for key, kind in SCHEMA.items():
        if key in data and type(data[key]) is not kind:
            raise ValueError(f"{path}: {key!r} is a {type(data[key]).__name__}, the schema says {kind.__name__}")
    return data


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    """The facts of the raptor wing ``name``, resolved along its chain from the root (read-only; ``name`` is the leaf's name)."""
    merged: dict = {}
    for each, *_ in reversed(wings.lineage(FAMILY, wings.segment(name))):
        for key, value in _read(home(each)).items():
            merged[key] = {**merged[key], **value} if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    merged["name"] = name
    return _freeze(merged)


def reset() -> None:
    """Forget what was loaded (the installed distributions changed)."""
    load.cache_clear()
    home.cache_clear()


def location(name: str) -> Path:
    """The installed folder of the raptor wing ``name`` (the one that holds its header)."""
    return home(name)


def header_digest(name: str) -> str:
    """The sha256 of the CONTENT of the raptor wing ``name``'s generated header: what a compile key holds instead of the header's path."""
    path = home(name) / "raptor_facts.h"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


class IncludeFlag(str):
    """The ``-I<dir>`` flag of a wing's header directory. It is the flag on the compile line (the path is where the header is
    installed); its :attr:`key_term` is what a compile key hashes in its place: the wing's NAME and the sha256 of the header's
    CONTENT, so the key does not depend on where the installed package lives."""

    key_term: str = ""


def include_flags(facts: dict) -> tuple:
    """The compile flags that hand a unit its facts header (``raptor_facts.h`` in the folder of the raptor wing, written by
    ``tools/gen_facts_header.py``): one ``-I`` of that folder (an :class:`IncludeFlag`); none for a build with no facts."""
    name = facts.get("name") if facts else None
    if not name:
        return ()
    flag = IncludeFlag(f"-I{home(name)}")
    flag.key_term = f"facts:{name}:{header_digest(name)}"
    return (flag,)


def named(entry, isa: str) -> str:
    """The raptor wing name a row's ``wings`` entry gives for a host of instruction set ``isa``: the entry is a name, or a map from
    isa to name (a host backend)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isa in entry:
        return entry[isa]
    raise ValueError(f"the row's raptor wing {entry!r} names no wing for isa {isa!r}")


def derive(facts: dict, overrides: dict) -> dict:
    """A copy of ``facts`` with ``overrides`` merged in (one dict level deep, like a wing over its parent): how a test or a
    kit run passes other numbers to code that takes facts."""
    merged = dict(facts)
    for key, value in overrides.items():
        merged[key] = {**facts[key], **value} if isinstance(value, dict) and isinstance(facts.get(key), dict) else value
    return merged
