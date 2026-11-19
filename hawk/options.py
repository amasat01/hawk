# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The user's target options: ``options={"<target id>.<key>": value}`` on the build entries.

hawk checks the shape and the target and hands the options, as plain data, to the OSPREY that serves the target; the library
refuses a key it does not take. The old keywords are folded here, in one table."""

from __future__ import annotations

import dataclasses
import json
import warnings

from . import ospreys
from .errors import HawkError

#: old keyword -> the option that replaces it; a keyword left at its default (``None``, ``False``, ``""``) is not given
LEGACY = {"literal_table": "cuda.literal_table", "device_arch": "cuda.arch",
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


def check(options) -> None:
    """Refuse, with a sentence and before anything is lowered, an option not spelled ``<target id>.<key>``, one whose target no
    OSPREY serves, or one whose value is not plain data."""
    for name, value in options.items():
        prefix, dot, key = name.partition(".")
        if not (dot and key):
            raise HawkError(f"option {name!r} must be spelled '<target id>.<key>'")
        ospreys.serving(prefix)
        canon(name, value)


def of_target(options, target: str) -> dict:
    """The options under the prefix ``target``, prefix dropped."""
    return {key: v for k, v in options.items() for prefix, _, key in [k.partition(".")] if ospreys.target_id(prefix) == target}


def canon(name: str, value) -> str:
    """The text of ``value`` as plain data: its ``json.dumps(sort_keys=True)``, a frozen dataclass by field name."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        value = dataclasses.asdict(value)
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise HawkError(f"option {name!r}: a {type(value).__name__} is not plain data; "
                        f"give plain data or a frozen dataclass of plain data ({e})") from None
