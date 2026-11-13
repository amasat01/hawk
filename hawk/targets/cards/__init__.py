# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-device cards: tuning data, read when a kernel is built.

A card picks among lowerings the IR facts already allow
(:func:`simt.lanes.choose_mapping`); it never changes what a kernel computes.
The cards are VARIANT DATA (``variants/cuda`` ``cards``; the host's own tables in ``variants/cpu``, ``x86``, ``arm``), read through
the build spec. The thread card holds, per (compute capability, device name, scalar type), the thread-box width ``T``, the
contraction choice and an optional block tile; the defaults (a thread box, ``gemm``, no tile) apply without a row.
"""

from __future__ import annotations

from ..._errors import HawkError
from ... import components as _components
_lanes = _components.view("lowering", "lanes")

#: The widest thread box: T lanes held in registers per thread.
T_MAX = 16


def _device() -> tuple | None:
    """``(cc, name)`` of the current CUDA device, or ``None`` without one."""
    from ...backends import default_backends

    for backend in default_backends("device"):
        try:
            return backend.probe_device()
        except (ImportError, HawkError):
            pass
    return None


def thread_box(scalar_type: str, spec) -> dict | None:
    """The card for ``scalar_type`` (``"float64"``/``"float32"``; ``None`` for a type no card measures, which
    builds unboxed): ``{"T": width,
    "contraction": "gemm" | "direct", "itemsize": bytes}`` and, on a device whose
    row carries one, ``"tile"`` (see :func:`simt.lanes.choose_mapping`). The table is ``spec.primary["thread_box"]`` (the host's own)
    or ``spec.primary["cards"]["thread_box"]`` (a device's).

    The device is probed only when the table has a row for that scalar type."""
    doc = spec.primary.get("thread_box") or spec.primary.get("cards", {}).get("thread_box")
    if doc is None or scalar_type not in doc["defaults"]:
        return None
    card = {"T": doc["defaults"][scalar_type], "contraction": doc["contraction"][scalar_type]}
    rows = [r for r in doc.get("rows", ()) if r["dtype"] == scalar_type]
    device = _device() if rows else None
    for r in rows:
        if device == (r["cc"], r["device"]):
            card.update({k: r[k] for k in ("T", "contraction", "tile") if k in r})
            break
    T = card["T"]
    if not (isinstance(T, int) and 2 <= T <= T_MAX):
        raise HawkError(f"thread_box card: T must be an integer in 2..{T_MAX} "
                        f"for {scalar_type}, got {T!r}")
    if card["contraction"] not in _lanes.CONTRACTIONS:
        raise HawkError(f"thread_box card: contraction must be one of {_lanes.CONTRACTIONS} "
                        f"for {scalar_type}, got {card['contraction']!r}")
    card["itemsize"] = 4 if scalar_type == "float32" else 8
    return card


#: The thresholds a wide card row may set, each a positive integer.
WIDE_KEYS = ("SLICE_MIN", "TEAM_MIN", "TC")


def wide_card(spec) -> dict:
    """The device variant's wide card (``cards.wide``: ``defaults`` and ``rows``), validated; refused by name for a build with no
    device variant (the host places by its own ``host_wide`` table)."""
    if spec.device is None or "wide" not in spec.device.get("cards", {}):
        raise HawkError("the build has no device variant with a wide card: a host build places by the host variant's own host_wide table")
    doc = spec.device["cards"]["wide"]
    for where, values in [("defaults", doc["defaults"]), *((f"row {r.get('cc')}/{r.get('device')}/{r.get('dtype')}", r)
                                                          for r in doc["rows"])]:
        for key in WIDE_KEYS:
            if key in values and not (isinstance(values[key], int) and not isinstance(values[key], bool) and values[key] > 0):
                raise HawkError(f"wide card: {key} must be a positive integer in {where}, got {values[key]!r}")
        if where != "defaults" and not all(isinstance(values.get(k), str) for k in ("cc", "device", "dtype")):
            raise HawkError(f"wide card: {where} needs cc, device and dtype as strings")
    if any(k not in doc["defaults"] for k in WIDE_KEYS):
        raise HawkError(f"wide card: defaults must set {WIDE_KEYS}")
    if doc["defaults"]["SLICE_MIN"] % 32:
        raise HawkError(f"wide card: SLICE_MIN must be a multiple of 32, got {doc['defaults']['SLICE_MIN']!r}")
    return doc


def team_warps(spec) -> int:
    """The warps a block of the warp-per-lane kernel holds: 1 unless ``spec`` forces ``{"WARPS": k}`` (the card kit)."""
    return int(spec.wide_override.get("WARPS", 1))


def wide(scalar_type: str, spec) -> dict:
    """``{"SLICE_MIN", "TEAM_MIN", "TC"}`` for ``scalar_type`` (``"float32"``/``"float64"``, or the IR's ``f32``/``f64``)
    on the device arch of ``spec``: the defaults, and over them the row of the scalar type whose arch is the
    nearest by device features (the SIMT component's ``wide_class.distance``: FP64 rate, registers, resident threads), so an unmeasured
    arch takes the closest measured row; a row of the build's own arch must also match the device name when one is present.
    No row for the scalar type gets the defaults. A build with no device variant is refused by name (:func:`wide_card`)."""
    scalar_type = {"f32": "float32", "f64": "float64"}.get(scalar_type, scalar_type)
    doc = wide_card(spec)
    card = {k: doc["defaults"][k] for k in WIDE_KEYS}
    card.update(_nearest_row(doc["rows"], scalar_type, WIDE_KEYS, spec))
    card.update({k: v for k, v in spec.wide_override.items() if k in WIDE_KEYS})
    return card


def _nearest_row(rows: list, scalar_type: str, keys: tuple, spec) -> dict:
    """The ``keys`` of the row of ``scalar_type`` whose arch is nearest the build's by device features (a row of the build's
    own arch must also match the device name when one is present); ``{}`` for a host-only build or no such row."""
    arch = spec.machine.arch
    rows = [r for r in rows if arch and r["dtype"] == scalar_type]
    if not rows:
        return {}
    from ... import components as _components
    _wide_class = _components.device_view("wide_class")

    device = _device() if any(r["cc"] == arch for r in rows) else None
    rows = [r for r in rows if r["cc"] != arch or device is None or device == (r["cc"], r["device"])]
    if not rows:
        return {}
    want = _wide_class.features_of(arch, spec.device)
    return {k: v for k, v in min(rows, key=lambda r: _wide_class.distance(want, _wide_class.features_of(r["cc"], spec.device))).items() if k in keys}


#: The constants a chain-fusion card row sets.
FUSION_KEYS = ("R0", "R_file", "L_us", "c_plane_us_per_B", "c_local_over_plane")


def fusion(scalar_type: str, spec, host: bool = False) -> dict:
    """The constants of :mod:`hawk.passes.fusion` for ``scalar_type``: the device defaults and over them the nearest row, as
    :func:`wide` reads its card; with ``host``, the row of the host's vector ISA (the ``vector_isa`` fact: ``sse2``, ``avx2``, ``avx512``, ``neon``)."""
    scalar_type = {"f32": "float32", "f64": "float64"}.get(scalar_type, scalar_type)
    if host:
        isa = spec.machine.facts.get("vector_isa", "")
        rows = (spec.host or {}).get("fusion_host", {})
        if isa not in rows:
            raise HawkError(f"the host variant has no fusion row {isa!r}; the rows are {sorted(rows)}")
        return dict(rows[isa])
    if spec.device is None or "fusion" not in spec.device.get("cards", {}):
        raise HawkError("the build has no device variant with a fusion card: a host build reads the host's own row (host=True)")
    doc = spec.device["cards"]["fusion"]
    return {**doc["defaults"], **_nearest_row(doc["rows"], scalar_type, FUSION_KEYS, spec)}
