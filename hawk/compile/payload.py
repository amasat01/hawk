# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`current_payload()` — the one sealed :class:`aether_dsc.Payload` every
NVRTC compile in this process serves headers from.

Two sources: the shipped blob (`aether_dsc.payload()`) with no dev
override, else a live re-seal of whatever `$HAWK_AETHER_INCLUDE` points
at — the same override that redirects
:func:`hawk.compile.toolchain.aether_include`, so a dev checkout gets a
matching payload."""
from __future__ import annotations

import os
from pathlib import Path

import aether_dsc

from .. import _roots
from . import toolchain as tc

_PAYLOAD_MEMO: aether_dsc.Payload | None = None


def current_payload() -> aether_dsc.Payload:
    """The payload this process's NVRTC compiles serve from. Memoised
    per process — sealing/decompressing is not free — cleared by
    :func:`reset_payload_memo`, which `hawk.compile._reset_artifact_memo`
    calls alongside the content-closure cache's own memos.

    `aether_dsc.payload()` (the shipped blob) is used when
    `$HAWK_AETHER_INCLUDE` is unset and a blob exists; otherwise a fresh
    :func:`aether_dsc.seal` is built from this process's own resolved
    roots: `hawk.compile.toolchain.aether_include()`, the aether
    checkout's `rtc/` shims (via `hawk._roots.WORKSPACE`), and eagle's
    `plugin/gref_layout.h` (`hawk.compile.toolchain.eagle_include()`).
    """
    global _PAYLOAD_MEMO
    if _PAYLOAD_MEMO is not None:
        return _PAYLOAD_MEMO

    if "HAWK_AETHER_INCLUDE" not in os.environ:
        try:
            _PAYLOAD_MEMO = aether_dsc.payload()
            return _PAYLOAD_MEMO
        except FileNotFoundError:
            pass  # no shipped blob in this install -> fall through to seal()

    _PAYLOAD_MEMO = aether_dsc.seal(
        aether_root=tc.aether_include(),
        rtc_dir=_roots.WORKSPACE / "aether" / "rtc",
        layout_header=Path(tc.eagle_include()) / "plugin" / "gref_layout.h",
    )
    return _PAYLOAD_MEMO


def reset_payload_memo() -> None:
    """Forget the memoised :func:`current_payload` — the next call re-reads
    (and, in dev-tree mode, re-seals) from disk."""
    global _PAYLOAD_MEMO
    _PAYLOAD_MEMO = None
