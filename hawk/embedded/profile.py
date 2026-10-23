# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C target's profile: the opt-in caps and policies one export runs under.

Every cap defaults to ``None`` (no cap) and every policy to the permissive
value, so the default profile refuses only what the C target cannot build.
``repr()`` is a term of the unit key, so it stays the dataclass default:
deterministic, field order fixed."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import HawkError

#: The cap fields: ``None`` (no cap) or a non-negative ``int``.
_CAPS = ("max_trips", "max_ops", "max_stack_bytes")

#: The allowed values of each policy field, in the order a refusal lists them.
_CHOICES = {
    "fp_domain": ("record", "refuse"),
    "index_policy": ("clamp", "refuse"),
    "c_std": ("c99", "c11"),
}


@dataclass(frozen=True)
class EmbeddedProfile:
    """Caps and policies for one C export (see :mod:`hawk.embedded.verify`)."""

    max_trips: int | None = None        # worst-case loop trips per sample
    max_ops: int | None = None          # arith + special ops per sample (kernel_cost)
    max_stack_bytes: int | None = None  # tapes + carries + named temporaries
    fp_domain: str = "record"           # "record" | "refuse"
    index_policy: str = "clamp"         # "clamp" | "refuse"
    require_cppcheck: bool = False
    c_std: str = "c99"                  # the dialect the consumer compiles under

    def __post_init__(self) -> None:
        for field in _CAPS:
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, int) or value < 0):
                raise HawkError(
                    f"EmbeddedProfile({field}={value!r}): {field} is None (no cap) "
                    "or a non-negative int")
        for field, allowed in _CHOICES.items():
            value = getattr(self, field)
            if value not in allowed:
                raise HawkError(
                    f"EmbeddedProfile({field}={value!r}): {field} is one of "
                    f"{', '.join(repr(a) for a in allowed)}")
