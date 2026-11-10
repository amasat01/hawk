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
import functools
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


def thread_box(scalar_type: str) -> dict | None:
    """The card for ``scalar_type`` (``"float64"``/``"float32"``; ``None`` for a type no card measures, which
    builds unboxed): ``{"T": width,
    "contraction": "gemm" | "direct", "itemsize": bytes}`` and, on a device whose
    row carries one, ``"tile"`` (see :func:`hawk.ir.lanes.choose_mapping`).

    The device is probed only when the file has a row for that scalar type."""
    doc = json.loads(THREAD_BOX_FILE.read_text())
    if scalar_type not in doc["defaults"]:
        return None
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


#: The wide-value thresholds card (see its ``about`` text).
WIDE_FILE = Path(__file__).with_name("wide.json")

#: The thresholds a row may set, each a positive integer.
WIDE_KEYS = ("SLICE_MIN", "TEAM_MIN", "TC")

#: Thresholds a caller forces for the block (the sweep measures below the committed ones); ``None`` reads the card.
_WIDE_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("hawk_wide_override", default=None)


@contextlib.contextmanager
def wide_override(values: dict):
    """Within the block :func:`wide` answers ``values`` over whatever the card says."""
    token = _WIDE_OVERRIDE.set(dict(values))
    try:
        yield
    finally:
        _WIDE_OVERRIDE.reset(token)


@functools.cache
def _wide_doc(path: str) -> dict:
    doc = json.loads(Path(path).read_text())
    for where, values in [("defaults", doc["defaults"]), *((f"row {r.get('cc')}/{r.get('device')}/{r.get('dtype')}", r)
                                                          for r in doc["rows"])]:
        for key in WIDE_KEYS:
            if key in values and not (isinstance(values[key], int) and not isinstance(values[key], bool) and values[key] > 0):
                raise HawkError(f"{WIDE_FILE.name}: {key} must be a positive integer in {where}, got {values[key]!r}")
        if where != "defaults" and not all(isinstance(values.get(k), str) for k in ("cc", "device", "dtype")):
            raise HawkError(f"{WIDE_FILE.name}: {where} needs cc, device and dtype as strings")
    if any(k not in doc["defaults"] for k in WIDE_KEYS):
        raise HawkError(f"{WIDE_FILE.name}: defaults must set {WIDE_KEYS}")
    if doc["defaults"]["SLICE_MIN"] % 32:
        raise HawkError(f"{WIDE_FILE.name}: SLICE_MIN must be a multiple of 32, got {doc['defaults']['SLICE_MIN']!r}")
    return doc


def team_warps() -> int:
    """The warps a block of the warp-per-lane kernel holds: 1 unless the card kit forces ``wide_override({"WARPS": k})``."""
    return int((_WIDE_OVERRIDE.get() or {}).get("WARPS", 1))


def wide(scalar_type: str) -> dict:
    """``{"SLICE_MIN", "TEAM_MIN", "TC"}`` for ``scalar_type`` (``"float32"``/``"float64"``, or the IR's ``f32``/``f64``)
    on the build's device arch (:func:`build_arch`): the defaults, and over them the row of the scalar type whose arch is the
    nearest by device features (:func:`hawk.ir.wide_class.distance`: FP64 rate, registers, resident threads), so an unmeasured
    arch takes the closest measured row; a row of the build's own arch must also match the device name when one is present.
    A host-only build, or no row for the scalar type, gets the defaults."""
    scalar_type = {"f32": "float32", "f64": "float64"}.get(scalar_type, scalar_type)
    doc = _wide_doc(str(WIDE_FILE))
    card = {k: doc["defaults"][k] for k in WIDE_KEYS}
    card.update(_nearest_row(doc["rows"], scalar_type, WIDE_KEYS))
    card.update({k: v for k, v in (_WIDE_OVERRIDE.get() or {}).items() if k in WIDE_KEYS})
    return card


def _nearest_row(rows: list, scalar_type: str, keys: tuple) -> dict:
    """The ``keys`` of the row of ``scalar_type`` whose arch is nearest the build's by device features (a row of the build's
    own arch must also match the device name when one is present); ``{}`` for a host-only build or no such row."""
    arch = _BUILD_ARCH.get()
    rows = [r for r in rows if arch and r["dtype"] == scalar_type]
    if not rows:
        return {}
    from ...ir.wide_class import distance, features_of

    device = _device() if any(r["cc"] == arch for r in rows) else None
    rows = [r for r in rows if r["cc"] != arch or device is None or device == (r["cc"], r["device"])]
    if not rows:
        return {}
    want = features_of(arch)
    return {k: v for k, v in min(rows, key=lambda r: distance(want, features_of(r["cc"]))).items() if k in keys}


#: The chain fusion card (see its ``about`` text) and the constants a row sets.
FUSION_FILE = Path(__file__).with_name("fusion.json")
FUSION_KEYS = ("R0", "R_file", "L_us", "c_plane_us_per_B", "c_local_over_plane")


def fusion(scalar_type: str, host: str | None = None) -> dict:
    """The constants of :mod:`hawk.ir.fusion` for ``scalar_type``: the device defaults and over them the nearest row, as
    :func:`wide` reads its card; with ``host`` (a vector ISA: ``sse2``, ``avx2``, ``avx512``, ``neon``), that host's row."""
    scalar_type = {"f32": "float32", "f64": "float64"}.get(scalar_type, scalar_type)
    doc = json.loads(FUSION_FILE.read_text())
    if host is not None:
        if host not in doc["host"]:
            raise HawkError(f"{FUSION_FILE.name}: no host row {host!r}; the rows are {sorted(doc['host'])}")
        return dict(doc["host"][host])
    return {**doc["defaults"], **_nearest_row(doc["rows"], scalar_type, FUSION_KEYS)}
