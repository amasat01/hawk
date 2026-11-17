# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The fusion constants of a build: the registers and costs :mod:`hawk.passes.fusion` estimates a kernel with.

They are WING DATA (``wings/cuda`` ``cards``; the host's own rows in ``wings/x86``, ``arm``), read through the build spec. The
numbers the lowering library decides on are not here: a build states those through ``build_spec(cards=...)``.
"""

from __future__ import annotations

from ..._errors import HawkError


def _device() -> tuple | None:
    """``(cc, name)`` of the current CUDA device, or ``None`` without one."""
    from ...backends import default_backends

    for backend in default_backends("device"):
        try:
            return backend.probe_device()
        except (ImportError, HawkError):
            pass
    return None


def _nearest_row(rows: list, scalar_type: str, keys: tuple, spec) -> dict:
    """The ``keys`` of the row of ``scalar_type`` whose arch is nearest the build's by device features (a row of the build's
    own arch must also match the device name when one is present); ``{}`` for a host-only build or no such row."""
    arch = spec.machine.arch
    rows = [r for r in rows if arch and r["dtype"] == scalar_type]
    if not rows:
        return {}
    from ... import wings as _wings
    _wide_class = _wings.device_view("wide_class")

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
            raise HawkError(f"the host wing has no fusion row {isa!r}; the rows are {sorted(rows)}")
        return dict(rows[isa])
    if spec.device is None or "fusion" not in spec.device.get("cards", {}):
        raise HawkError("the build has no device wing with a fusion card: a host build reads the host's own row (host=True)")
    doc = spec.device["cards"]["fusion"]
    return {**doc["defaults"], **_nearest_row(doc["rows"], scalar_type, FUSION_KEYS, spec)}
