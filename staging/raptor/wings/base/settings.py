# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-build settings record: plain named values, none a machine number. A front end makes it and passes it in; nothing here
reads the environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """The per-build settings, carried by the build spec (a build spec's ``settings``): none is a machine
    number. Set once by the outermost entry (``with_settings`` on the spec returns a new one), read from the spec below it.

    ``form`` the ``emit_form``; ``linalg_form``, ``wide_form``, ``contraction`` as the build option of that name;
    ``launch_bounds`` ``(threads, blocks_per_sm)`` or ``None``; ``scalar`` the scalar mode id (``""``: not stated);
    ``device_only`` whether every target of the build is a device; ``placed`` whether the text keeps wide values in their planes
    (the device text; the host renders a placement as its value); ``blend`` whether float selects render as ``hawk_abi::blend``
    (a host row); ``staged_rows`` the samples a block of the warp form shares one output tile for (1: stores go straight to the
    planes); ``warps`` the warps of a block of the warp-per-lane kernel; ``fold_k`` the rows per warp forced in every cell-sum
    kernel (0: the wing's own value); ``block_fold`` whether the body sits in a frame that allows the block-level commit;
    ``route_to_ordered`` whether a keyed accumulate of contract ``auto`` or ``compensated`` takes the ordered stages (a ``cells``
    scalar mode has no atomic add on a packed cell); ``fast_entries`` whether a device unit builds its fast entry points."""

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
    fast_entries: bool = False
    #: The threads of every team stage when forced (0: the stage's own size).
    team_threads: int = 0
