# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The emit policy: which spelling an op site takes.

A site is spelled STATIC (an ``aether`` expression, every element unrolled) or
ROLLED (an ``aether::rt`` twin, a loop over the element index kept rolled,
:mod:`aether/expr/Rt.h`). Both return the same bits; the rolled form keeps one
element live where the static one keeps a fold's worth, so a long fold or a
wide result does not spill. The choice depends only on shapes the IR carries:
the site's fold length ``K`` and its result element count ``E``.
"""

from __future__ import annotations

import os
from .._errors import HawkError
from dataclasses import dataclass, field, replace


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


@dataclass(frozen=True)
class Settings:
    """The per-build settings, carried by the build spec (:class:`hawk.targets.BuildSpec`, ``spec.settings``): none is a machine
    number. Set once by the outermost entry (:meth:`BuildSpec.with_settings` returns a new spec), read from the spec below it.

    ``form`` the ``emit_form``; ``linalg_form``, ``wide_form``, ``contraction`` as the build option of that name;
    ``launch_bounds`` ``(threads, blocks_per_sm)`` or ``None``; ``scalar`` the scalar mode id (``""``: not stated);
    ``device_only`` whether every target of the build is a device; ``placed`` whether the text keeps wide values in their planes
    (the device text; the host renders a placement as its value); ``blend`` whether float selects render as ``hawk_abi::blend``
    (a host row); ``staged_rows`` the samples a block of the warp form shares one output tile for (1: stores go straight to the
    planes); ``warps`` the warps of a block of the warp-per-lane kernel; ``fold_k`` the rows per warp forced in every cell-sum
    kernel (0: the variant's policy); ``block_fold`` whether the body sits in a frame that allows the block-level commit;
    ``route_to_ordered`` whether a keyed accumulate of policy ``auto`` or ``compensated`` takes the ordered stages (a ``cells``
    scalar mode has no atomic add on a packed cell)."""

    form: str = "auto"
    linalg_form: str = "auto"
    wide_form: str = "auto"
    contraction: str = "direct"
    launch_bounds: tuple | None = None
    scalar: str = ""
    device_only: bool = True
    placed: bool = False
    blend: bool = False
    staged_rows: int = 1
    warps: int = 1
    fold_k: int = 0
    block_fold: bool = False
    route_to_ordered: bool = False
    #: The threads of every team stage when forced (``HAWK_TEAM_THREADS``, read when the settings are made; 0: the stage's own size).
    team_threads: int = field(default_factory=lambda: int(os.environ.get("HAWK_TEAM_THREADS") or 0))


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


def fold_shape(width: int, fold: dict, regs: int | None = None, forced: int = 0) -> int:
    """The rows (32 samples each) one warp folds in ONE batched ``sampleSumWarp_`` call, for a cell sink of ``width``
    columns: the largest power of two ``k <= fold["k_max"]`` whose ``k * width`` shuffle columns keep the warp under the residency
    line (``regs_base + regs_column * (k * width - 1) <= regs``; ``regs`` defaults to the line ``residency_regs``), 1 when none
    fits. ``fold`` is the device variant's ``fold`` data (ptxas: 20 registers for 1 row, 26 for 4, 43 for 8). Every ``k`` stores
    the same rows with the same bits; only the shuffle latency per row changes. ``forced`` (the build's ``Settings.fold_k``) wins when set."""
    if forced:
        return forced
    line = fold["residency_regs"] if regs is None else regs
    k = fold["k_max"]
    while k > 1 and fold["regs_base"] + fold["regs_column"] * (k * width - 1) > line:
        k //= 2
    return k
