# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""BUILD SETTINGS: the knobs that steer code generation and compilation, as ONE explicit value (never ambient).

A :class:`BuildSettings` is passed as ``settings=`` to a public build entry (``build``, ``emit_source``, ``build_bundle``). The entry validates it here, checks that every key it sets is DECLARED by a backend of
the build (a backend's manifest lists the setting keys it accepts under ``"settings"``; an undeclared key is refused by name), and
turns it into the plain named values of the build spec's :class:`Settings`. This type, its validation and its
defaults live in hawk only: below hawk only the plain values travel."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

from .errors import HawkError


@dataclass(frozen=True)
class Settings:
    """The per-build settings a build states to the lowering, none of them a machine number.

    ``form`` the ``emit_form``; ``linalg_form``, ``wide_form`` as the build option of that name;
    ``launch_bounds`` ``(threads, blocks_per_sm)`` or ``None``; ``warps`` the warps of a block of the warp-per-lane kernel;
    ``fold_rows`` the rows per warp forced in every cell-sum kernel (0: the library's own value); ``fast_entries`` whether a
    device unit builds its fast entry points; ``team_threads`` the threads of every team stage when forced (0: the stage's own
    size). The library refuses a value outside what its target allows."""

    form: str = "auto"
    linalg_form: str = "auto"
    wide_form: str = "auto"
    launch_bounds: tuple | None = None
    warps: int = 1
    fold_rows: int = 0
    fast_entries: bool = False
    team_threads: int = 0


#: The forms a build may ask for: ``auto`` picks per site by the threshold below, ``static``
#: spells every site as the unrolled expression (the text before the setting), and ``rolled``
#: spells every site that has a twin as its loop (a fast-compile mode: the compiler does no
#: per-element work, and a fold or result of any size builds).
FORMS = ("auto", "static", "rolled")

#: The forms a build may ask of per-lane linear algebra (``linalg_form``): ``auto`` picks by size
#: (one thread per matrix up to the wing's ``linalg.max_dim``, a team above it), ``thread`` keeps
#: every matrix in one lane (refused above that size, by name) and ``team`` runs every
#: factorisation as a team stage whatever its size.
LINALG_FORMS = ("auto", "thread", "team")

#: The forms a build may ask of a per-lane value wider than a thread's registers (``wide_form``):
#: ``auto`` places each such value where the card keeps it (the library), ``thread``
#: keeps every value in the lane's own registers and local memory (the text before the pass),
#: and ``team`` is a warp per lane (the library), for the kernels it has a rule for; the others build as ``auto``.
WIDE_FORMS = ("auto", "thread", "team")

#: How a dense contraction is run is the device card's choice per scalar type; no build setting states it.
CONTRACTION_REFUSAL = ("contraction is not a build setting: the device card of the OSPREY chooses, per scalar type and architecture, "
                       "whether a dense contraction runs as the kernel's own loop or as a matrix product")


class SettingError(HawkError, ValueError):
    """A build setting outside its allowed values: a :class:`~hawk.HawkError` and a ``ValueError`` (both are caught)."""


#: setting name -> its allowed values: the ONE table every validation of a form set reads.
CHOICES = MappingProxyType({"emit_form": FORMS, "linalg_form": LINALG_FORMS, "wide_form": WIDE_FORMS})


def refuse_choice(key: str, value) -> None:
    """Refuse a value that is not one of the allowed values of the setting ``key``, by name. The one validation of the form sets:
    the settings value, the entry's keywords and the settings check all call it."""
    if value not in CHOICES[key]:
        raise SettingError(f"{key} must be one of {CHOICES[key]}, got {value!r}")


def check(settings: Settings) -> Settings:
    """``settings`` with its choices validated, refused by name."""
    for key, value in (("emit_form", settings.form), ("linalg_form", settings.linalg_form), ("wide_form", settings.wide_form)):
        refuse_choice(key, value)
    return settings


def default_settings() -> Settings:
    """The settings of a build that states none: every field at its default, ``team_threads`` read once from ``HAWK_TEAM_THREADS``."""
    return Settings(team_threads=int(os.environ.get("HAWK_TEAM_THREADS") or 0))


@dataclass(frozen=True)
class BuildSettings:
    """Every field left ``None`` is unset (the build's default, or the entry's own keyword).

    ``emit_form`` how a product, dot, sum or norm is spelled (``auto``, ``static``, ``rolled``); ``linalg_form`` how per-lane linear
    algebra is worked (``auto``, ``thread``, ``team``); ``wide_form`` where a per-lane value wider than a thread's registers
    lives (``auto``, ``thread``, ``team``); ``contraction`` is refused by name (see :data:`CONTRACTION_REFUSAL`); ``launch_bounds``
    ``(threads, blocks_per_sm)`` for a device kernel; ``fold_rows`` the rows per warp forced in every cell-sum kernel;
    ``fast_entries`` (a bool) builds the device unit's fast entry points."""

    emit_form: str | None = None
    linalg_form: str | None = None
    wide_form: str | None = None
    contraction: str | None = None
    launch_bounds: tuple | None = None
    fold_rows: int | None = None
    fast_entries: bool | None = None

    def __post_init__(self):
        for key in CHOICES:
            value = getattr(self, key)
            if value is not None:
                refuse_choice(key, value)
        if self.contraction is not None:
            raise SettingError(CONTRACTION_REFUSAL)
        if self.fast_entries is not None and not isinstance(self.fast_entries, bool):
            raise SettingError(f"fast_entries must be a bool, got {self.fast_entries!r}")

    def keys(self) -> tuple:
        """The names of the fields these settings set."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)


#: field -> the :class:`Settings` field it becomes.
_SETTING = MappingProxyType({"emit_form": "form", "linalg_form": "linalg_form", "wide_form": "wide_form",
                             "launch_bounds": "launch_bounds", "fold_rows": "fold_rows",
                             "fast_entries": "fast_entries"})


def settings_of(build_settings: BuildSettings | None, base):
    """``base`` (a :class:`Settings`) with the fields ``build_settings`` sets laid over it: plain values only."""
    if build_settings is None:
        return base
    if not isinstance(build_settings, BuildSettings):
        raise HawkError(f"settings must be a hawk.build_settings.BuildSettings, got {type(build_settings).__name__}")
    return replace(base, **{_SETTING[k]: getattr(build_settings, k) for k in build_settings.keys()})
