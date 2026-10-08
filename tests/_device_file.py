# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device artifact a built unit published for one kernel, whatever its format.

Since the default device target follows the installed driver (``cubin`` when the
compiler is newer than the driver, ``ptx`` otherwise), a test must not hard-code
the extension: it asks for the one file the unit holds for that kernel.
"""

from __future__ import annotations

from pathlib import Path


def device_artifact(directory: Path, stem: str) -> Path:
    """The single ``<stem>.ptx`` or ``<stem>.cubin`` in ``directory``."""
    found = [directory / f"{stem}{ext}" for ext in (".ptx", ".cubin")
             if (directory / f"{stem}{ext}").is_file()]
    assert len(found) == 1, (stem, sorted(p.name for p in directory.iterdir()))
    return found[0]
