# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-device cards: tuning data, read when a kernel is built.

A card picks among lowerings the IR facts already allow
(:func:`hawk.ir.lanes.choose_mapping`); it never changes what a kernel computes.
``thread_box.json`` holds, per (compute capability, device name, scalar type), the
thread-box width ``T``, the contraction choice and an optional block tile; the
defaults (a thread box, ``gemm``, no tile) apply without a row.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from pathlib import Path

from ...ir import HawkError
from ...ir.lanes import CONTRACTIONS

#: The thread-box card (see its ``about`` text).
THREAD_BOX_FILE = Path(__file__).with_name("thread_box.json")

#: The widest thread box: T lanes held in registers per thread.
T_MAX = 16


def _device() -> tuple | None:
    """``(cc, name)`` of the current CUDA device, or ``None`` without one."""
    from ...compile import nvrtc

    try:
        return nvrtc.current_device_arch(), nvrtc.current_device_name()
    except (ImportError, HawkError):
        return None


def thread_box(scalar_type: str) -> dict:
    """The card for ``scalar_type`` (``"float64"``/``"float32"``): ``{"T": width,
    "contraction": "gemm" | "direct", "itemsize": bytes}`` and, on a device whose
    row carries one, ``"tile"`` (see :func:`hawk.ir.lanes.choose_mapping`).

    The device is probed only when the file has a row for that scalar type."""
    doc = json.loads(THREAD_BOX_FILE.read_text())
    card = {"T": doc["defaults"][scalar_type], "contraction": doc["contraction"][scalar_type]}
    rows = [r for r in doc["rows"] if r["dtype"] == scalar_type]
    device = _device() if rows else None
    for r in rows:
        if device == (r["cc"], r["device"]):
            card.update({k: r[k] for k in ("T", "contraction", "tile") if k in r})
            break
    T = card["T"]
    if not (isinstance(T, int) and 2 <= T <= T_MAX):
        raise HawkError(f"{THREAD_BOX_FILE.name}: T must be an integer in 2..{T_MAX} "
                        f"for {scalar_type}, got {T!r}")
    if card["contraction"] not in CONTRACTIONS:
        raise HawkError(f"{THREAD_BOX_FILE.name}: contraction must be one of {CONTRACTIONS} "
                        f"for {scalar_type}, got {card['contraction']!r}")
    card["itemsize"] = 4 if scalar_type == "float32" else 8
    return card


#: The compute arch the running build resolved for its device targets ("" for a
#: host-only build), set by :func:`build_arch` around the emit.
_BUILD_ARCH: contextvars.ContextVar = contextvars.ContextVar("hawk_build_arch", default="")


@contextlib.contextmanager
def build_arch(arch: str):
    """Within the block :func:`select_weight` reads the rows for ``arch``."""
    token = _BUILD_ARCH.set(arch)
    try:
        yield
    finally:
        _BUILD_ARCH.reset(token)


def select_weight(scalar_type: str) -> float:
    """What one add of a runtime-index store written as selects costs against an
    indexed access, for ``scalar_type`` on the build's device arch. The default
    applies without an arch (a host-only build) or a row; this never probes."""
    doc = json.loads(THREAD_BOX_FILE.read_text())["select_weight"]
    weight = doc["defaults"][scalar_type]
    arch = _BUILD_ARCH.get()
    for r in doc["rows"]:
        if r["dtype"] == scalar_type and r["cc"] == arch:
            weight = r["select_weight"]
    if not weight > 0:
        raise HawkError(f"{THREAD_BOX_FILE.name}: select_weight must be positive for "
                        f"{scalar_type}, got {weight!r}")
    return weight
