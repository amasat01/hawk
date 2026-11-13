# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""BUILD POLICIES: the knobs that steer code generation and compilation, as ONE explicit value (never ambient).

A :class:`BuildPolicy` is passed as ``build_policy=`` to a public build entry (``build``, ``emit_source``, ``build_bundle``,
``hawk_c.render``). The entry validates it here, checks that every key it sets is DECLARED by a backend of
the build (a backend's manifest lists the policy keys it accepts under ``"policy"``; an undeclared key is refused by name), and
turns it into the plain named values of the build spec's :class:`lowering.settings.Settings`. This type, its validation and its
defaults live in hawk only: below hawk only the plain values travel."""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import TYPE_CHECKING

from dataclasses import dataclass, fields, replace

from ._errors import HawkError
from . import components as _components

if TYPE_CHECKING:
    from lowering.settings import Settings


#: The forms a build may ask for: ``auto`` picks per site by the threshold below, ``static``
#: spells every site as the unrolled expression (the text before the policy), and ``rolled``
#: spells every site that has a twin as its loop (a fast-compile mode: the compiler does no
#: per-element work, and a fold or result of any size builds).
FORMS = ("auto", "static", "rolled")

#: The forms a build may ask of per-lane linear algebra (``linalg_form``): ``auto`` picks by size
#: (one thread per matrix up to the variant's ``linalg.max_dim``, a team above it), ``thread`` keeps
#: every matrix in one lane (refused above that size, by name) and ``team`` runs every
#: factorisation as a team stage whatever its size.
LINALG_FORMS = ("auto", "thread", "team")

#: The forms a build may ask of a per-lane value wider than a thread's registers (``wide_form``):
#: ``auto`` places each such value where the card keeps it (:mod:`simt.wide`), ``thread``
#: keeps every value in the lane's own registers and local memory (the text before the pass),
#: and ``team`` is a warp per lane (:mod:`simt.lane`), for the kernels it has a rule for; the others build as ``auto``.
WIDE_FORMS = ("auto", "thread", "team")

#: How a dense contraction is run (``contraction``): ``direct`` is the kernel's own loop; ``gemm`` asks eagle for a
#: matrix-multiply route, for the kernels that have one (opt in: its values are within a few ulp, not the same bits).
CONTRACTION_FORMS = ("direct", "gemm")


def checked_bounds(launch_bounds: tuple[int, int], limits: dict) -> tuple[int, int]:
    """``launch_bounds`` as ints, refused by name when it breaks ``limits`` (the variant's ``launch`` data: ``warp_size``,
    ``max_threads_per_sm``)."""
    threads, blocks = launch_bounds
    warp, cap = limits["warp_size"], limits["max_threads_per_sm"]
    if threads < warp or threads % warp or blocks < 1 or threads * blocks > cap:
        raise ValueError(f"launch_bounds=(threads, blocks) needs threads a multiple of {warp} and threads*blocks <= {cap}, got {launch_bounds!r}")
    return (int(threads), int(blocks))


class SettingError(HawkError, ValueError):
    """A build setting outside its allowed values: a :class:`~hawk.ir.nodes.HawkError` and a ``ValueError`` (both are caught)."""


#: setting name -> its allowed values: the ONE table every validation of a form set reads.
CHOICES = {"emit_form": FORMS, "linalg_form": LINALG_FORMS, "wide_form": WIDE_FORMS, "contraction": CONTRACTION_FORMS}


def refuse_choice(key: str, value) -> None:
    """Refuse a value that is not one of the allowed values of the setting ``key``, by name. The one validation of the form sets:
    the policy value, the entry's keywords and the settings check all call it."""
    if value not in CHOICES[key]:
        raise SettingError(f"{key} must be one of {CHOICES[key]}, got {value!r}")


def check(settings: Settings, limits: dict | None = None) -> Settings:
    """``settings`` with its choices validated (and ``launch_bounds`` checked against ``limits``, the device variant's ``launch``
    data, when given), refused by name."""
    for key, value in (("emit_form", settings.form), ("linalg_form", settings.linalg_form), ("wide_form", settings.wide_form),
                       ("contraction", settings.contraction)):
        refuse_choice(key, value)
    if settings.launch_bounds is not None and limits is not None:
        return replace(settings, launch_bounds=checked_bounds(settings.launch_bounds, limits))
    return settings


def default_settings() -> Settings:
    """The settings of a build that states none: every field at its default, ``team_threads`` read once from ``HAWK_TEAM_THREADS``."""
    return _components.view("lowering", "settings").Settings(team_threads=int(os.environ.get("HAWK_TEAM_THREADS") or 0))


@dataclass(frozen=True)
class BuildPolicy:
    """Every field left ``None`` is unset (the build's default, or the entry's own keyword).

    ``emit_form`` how a product, dot, sum or norm is spelled (``auto``, ``static``, ``rolled``); ``linalg_form`` how per-lane linear
    algebra is worked (``auto``, ``thread``, ``team``); ``wide_form`` where a per-lane value wider than a thread's registers
    lives (``auto``, ``thread``, ``team``); ``contraction`` how a dense contraction is run (``direct``, ``gemm``); ``launch_bounds``
    ``(threads, blocks_per_sm)`` for a device kernel; ``fold_rows`` the rows per warp forced in every cell-sum kernel."""

    emit_form: str | None = None
    linalg_form: str | None = None
    wide_form: str | None = None
    contraction: str | None = None
    launch_bounds: tuple | None = None
    fold_rows: int | None = None

    def __post_init__(self):
        for key in CHOICES:
            value = getattr(self, key)
            if value is not None:
                refuse_choice(key, value)

    def keys(self) -> tuple:
        """The names of the fields this policy sets."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)


#: field -> the :class:`~lowering.settings.Settings` field it becomes.
_SETTING = MappingProxyType({"emit_form": "form", "linalg_form": "linalg_form", "wide_form": "wide_form", "contraction": "contraction",
                             "launch_bounds": "launch_bounds", "fold_rows": "fold_k"})


def settings_of(build_policy: BuildPolicy | None, base):
    """``base`` (a :class:`~lowering.settings.Settings`) with the fields ``build_policy`` sets laid over it: plain values only."""
    if build_policy is None:
        return base
    if not isinstance(build_policy, BuildPolicy):
        raise HawkError(f"build_policy must be a hawk.build_policy.BuildPolicy, got {type(build_policy).__name__}")
    return replace(base, **{_SETTING[k]: getattr(build_policy, k) for k in build_policy.keys()})


def refuse_undeclared(build_policy: BuildPolicy | None, targets) -> None:
    """Refuse by name a key ``build_policy`` sets that no target of the build declares in its manifest (``"policy"``)."""
    if build_policy is None:
        return
    from . import registry

    declared = set()
    for target in targets:
        if registry.has("backends", target):
            declared |= set(registry.traits(target).get("policy", ()))
    for key in build_policy.keys():
        if key not in declared:
            raise HawkError(f"build policy key {key!r} is declared by none of the targets {tuple(targets)}: a backend's manifest "
                            f"names the policy keys it accepts")


def apply(spec, build_policy: BuildPolicy | None, targets):
    """``spec`` with ``build_policy`` laid over its settings, after the keys it sets are checked against the targets' manifests."""
    if build_policy is None:
        return spec
    refuse_undeclared(build_policy, targets)
    return replace(spec, settings=settings_of(build_policy, spec.settings))
