# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The refusal of a banded build off the device."""

from __future__ import annotations

from raptor.errors import RaptorError


def refuse_cells_off_device(name: str, mode, target_id: str, target) -> None:
    """Refuse a ``cells`` (banded) build for a target that is not a device: banded is a device feature."""
    if mode.cells and not target.device:
        raise RaptorError(f"{name!r}: banded runs on device execution paths; on a CPU use float64 "
                        f"(target {target_id!r} is not a device)")
