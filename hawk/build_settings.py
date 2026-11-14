# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""BUILD SETTINGS: the knobs that steer code generation and compilation, as ONE explicit value (never ambient).

A :class:`BuildSettings` is passed as ``settings=`` to a public build entry (``build``, ``emit_source``, ``build_bundle``,
``ospreys.c.render``). The entry validates it here, checks that every key it sets is DECLARED by a backend of
the build (a backend's manifest lists the setting keys it accepts under ``"settings"``; an undeclared key is refused by name), and
turns it into the plain named values of the build spec's :class:`raptor.wings.base.settings.Settings`. This type, its validation and its
defaults live in hawk only: below hawk only the plain values travel."""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import TYPE_CHECKING

from dataclasses import dataclass, fields, replace

from ._errors import HawkError
from . import wings as _wings

if TYPE_CHECKING:
    from raptor.wings.base.settings import Settings


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
#: ``auto`` places each such value where the card keeps it (:mod:`raptor.wings.simt.wide`), ``thread``
#: keeps every value in the lane's own registers and local memory (the text before the pass),
#: and ``team`` is a warp per lane (:mod:`raptor.wings.simt.lane`), for the kernels it has a rule for; the others build as ``auto``.
WIDE_FORMS = ("auto", "thread", "team")

#: How a dense contraction is run (``contraction``): ``direct`` is the kernel's own loop; ``gemm`` asks eagle for a
#: matrix-multiply route, for the kernels that have one (opt in: its values are within a few ulp, not the same bits).
CONTRACTION_FORMS = ("direct", "gemm")


def checked_bounds(launch_bounds: tuple[int, int], limits: dict) -> tuple[int, int]:
    """``launch_bounds`` as ints, refused by name when it breaks ``limits`` (the wing's ``launch`` data: ``warp_size``,
    ``max_threads_per_sm``)."""
    threads, blocks = launch_bounds
    warp, cap = limits["warp_size"], limits["max_threads_per_sm"]
    if threads < warp or threads % warp or blocks < 1 or threads * blocks > cap:
        raise ValueError(f"launch_bounds=(threads, blocks) needs threads a multiple of {warp} and threads*blocks <= {cap}, got {launch_bounds!r}")
    return (int(threads), int(blocks))


class SettingError(HawkError, ValueError):
    """A build setting outside its allowed values: a :class:`~hawk.HawkError` and a ``ValueError`` (both are caught)."""


#: setting name -> its allowed values: the ONE table every validation of a form set reads.
CHOICES = MappingProxyType({"emit_form": FORMS, "linalg_form": LINALG_FORMS, "wide_form": WIDE_FORMS, "contraction": CONTRACTION_FORMS})


def refuse_choice(key: str, value) -> None:
    """Refuse a value that is not one of the allowed values of the setting ``key``, by name. The one validation of the form sets:
    the settings value, the entry's keywords and the settings check all call it."""
    if value not in CHOICES[key]:
        raise SettingError(f"{key} must be one of {CHOICES[key]}, got {value!r}")


def check(settings: Settings, limits: dict | None = None) -> Settings:
    """``settings`` with its choices validated (and ``launch_bounds`` checked against ``limits``, the device wing's ``launch``
    data, when given), refused by name."""
    for key, value in (("emit_form", settings.form), ("linalg_form", settings.linalg_form), ("wide_form", settings.wide_form),
                       ("contraction", settings.contraction)):
        refuse_choice(key, value)
    if settings.launch_bounds is not None and limits is not None:
        return replace(settings, launch_bounds=checked_bounds(settings.launch_bounds, limits))
    return settings


def default_settings() -> Settings:
    """The settings of a build that states none: every field at its default, ``team_threads`` read once from ``HAWK_TEAM_THREADS``."""
    return _wings.view("raptor", "base", "settings").Settings(team_threads=int(os.environ.get("HAWK_TEAM_THREADS") or 0))


@dataclass(frozen=True)
class BuildSettings:
    """Every field left ``None`` is unset (the build's default, or the entry's own keyword).

    ``emit_form`` how a product, dot, sum or norm is spelled (``auto``, ``static``, ``rolled``); ``linalg_form`` how per-lane linear
    algebra is worked (``auto``, ``thread``, ``team``); ``wide_form`` where a per-lane value wider than a thread's registers
    lives (``auto``, ``thread``, ``team``); ``contraction`` how a dense contraction is run (``direct``, ``gemm``); ``launch_bounds``
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
        if self.fast_entries is not None and not isinstance(self.fast_entries, bool):
            raise SettingError(f"fast_entries must be a bool, got {self.fast_entries!r}")

    def keys(self) -> tuple:
        """The names of the fields these settings set."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)


#: field -> the :class:`~raptor.wings.base.settings.Settings` field it becomes.
_SETTING = MappingProxyType({"emit_form": "form", "linalg_form": "linalg_form", "wide_form": "wide_form", "contraction": "contraction",
                             "launch_bounds": "launch_bounds", "fold_rows": "fold_k",
                             "fast_entries": "fast_entries"})


def settings_of(build_settings: BuildSettings | None, base):
    """``base`` (a :class:`~raptor.wings.base.settings.Settings`) with the fields ``build_settings`` sets laid over it: plain values only."""
    if build_settings is None:
        return base
    if not isinstance(build_settings, BuildSettings):
        raise HawkError(f"settings must be a hawk.build_settings.BuildSettings, got {type(build_settings).__name__}")
    return replace(base, **{_SETTING[k]: getattr(build_settings, k) for k in build_settings.keys()})


def refuse_undeclared(build_settings: BuildSettings | None, targets) -> None:
    """Refuse by name a key ``build_settings`` sets that no target of the build declares in its manifest (``"settings"``)."""
    if build_settings is None:
        return
    from . import registry

    declared = set()
    for target in targets:
        if registry.has("backends", target):
            declared |= set(registry.traits(target).get("settings", ()))
    for key in build_settings.keys():
        if key not in declared:
            raise HawkError(f"build settings key {key!r} is declared by none of the targets {tuple(targets)}: a backend's manifest "
                            f"names the setting keys it accepts")


def apply(spec, build_settings: BuildSettings | None, targets):
    """``spec`` with ``build_settings`` laid over its settings, after the keys it sets are checked against the targets' manifests."""
    if build_settings is None:
        return spec
    refuse_undeclared(build_settings, targets)
    return replace(spec, settings=settings_of(build_settings, spec.settings))
