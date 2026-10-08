# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where a run-written card goes.

The cards under ``tests/cards/`` are written BY THE RUN that measures them, and
committed. An ordinary test run must not rewrite them, so by default a card
goes to a per-process scratch directory; set ``HAWK_MINT_CARDS=1`` to write the
committed copy on purpose (then commit it).
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

#: The committed cards.
COMMITTED = Path(__file__).resolve().parent / "cards"

#: Whether this process has already registered the scratch directory's
#: at-exit cleanup — a module-level flag, not a per-call check, because
#: :func:`card_path` is called many times across many test modules in one
#: process and the cleanup must be registered exactly once.
_CLEANUP_REGISTERED = False


def card_path(name: str) -> Path:
    """The file a run writes card ``name`` to: the committed copy when
    ``HAWK_MINT_CARDS=1``, otherwise a scratch copy — self-cleaning: the
    scratch directory is removed when THIS process exits, so an ordinary
    test run leaves nothing behind in ``$TMPDIR`` (the committed copy is
    never touched, by construction: that branch returns before the scratch
    directory is ever created or registered)."""
    if os.environ.get("HAWK_MINT_CARDS") == "1":
        return COMMITTED / name
    global _CLEANUP_REGISTERED
    scratch = Path(tempfile.gettempdir()) / f"hawk-cards-{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    if not _CLEANUP_REGISTERED:
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        _CLEANUP_REGISTERED = True
    return scratch / name
