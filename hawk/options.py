# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The user's backend options: ``options={"<backend id>.<key>": value}`` on the build entries.

The driver checks them against the manifests (:func:`check`) before it computes a unit key, and the key takes every option
of a built backend by rule (:func:`canon`). It names no backend's option. The old keywords are folded here, in one table."""

from __future__ import annotations

import dataclasses
import json
import warnings

from . import registry
from ._errors import HawkError

#: old keyword -> the option that replaces it; a keyword left at its default (``None``, ``False``, ``""``) is not given
LEGACY = {"literal_table": "cuda.literal_table", "embedded_profile": "c.profile", "device_arch": "cuda.arch",
          "host_profile": "host.profile"}


def merge(options, **old) -> dict:
    """``options`` with the old keywords in ``old`` folded in, each a DeprecationWarning naming its replacement; the same
    option given twice (keyword and ``options``) is a HawkError."""
    merged = dict(options or {})
    for word, value in old.items():
        if value is None or value is False or value == "":
            continue
        new = LEGACY[word]
        if new in merged:
            raise HawkError(f"{word}= and options[{new!r}] both given: use options[{new!r}] only")
        warnings.warn(f"{word}= is deprecated: use options={{{new!r}: ...}}", DeprecationWarning, stacklevel=3)
        merged[new] = value
    return merged


def shared(merged: dict) -> tuple:
    """``(device arch asked, host profile asked)`` of ``merged``: the two options the driver resolves for every backend."""
    return merged.get(LEGACY["device_arch"], ""), merged.get(LEGACY["host_profile"], "")


def check(options) -> None:
    """Refuse, with a sentence and before anything is keyed or emitted, an option whose backend is not installed, whose key its
    manifest does not declare, or whose value cannot enter the unit key."""
    for name, value in options.items():
        prefix, dot, key = name.partition(".")
        if not (dot and key):
            raise HawkError(f"option {name!r} must be spelled '<backend id>.<key>'")
        if not registry.has("backends", prefix):
            raise HawkError(f"option {name!r}: {registry.missing(prefix)}")
        declared = registry.traits(prefix).get("options", ())
        if key not in declared:
            raise HawkError(f"backend {prefix!r} takes no option {name!r}; its options are "
                            f"{[prefix + '.' + k for k in declared] or 'none'}")
        canon(name, value)


def canon(name: str, value) -> str:
    """The text of ``value`` in a unit key: its ``json.dumps(sort_keys=True)``, a frozen dataclass by field name."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        value = dataclasses.asdict(value)
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise HawkError(f"option {name!r}: a {type(value).__name__} cannot enter the unit key; "
                        f"give plain data or a frozen dataclass of plain data ({e})") from None
