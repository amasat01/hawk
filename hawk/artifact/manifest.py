# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The schema-v2 manifest: FLAT exec keys, in the contractual order.

Top-level key ORDER is contractual: eagle's C++ manifest reader is a
string scanner, and raptor enforces the same order Python-side, so HAWK
writes exactly ``raptor.schema.manifest.TOP_LEVEL_KEY_ORDER_V2``. The
keys are flat, not a nested ``execution`` object. ``exec_op`` is present
iff ``exec_access == 'mapreduce'`` and forbidden otherwise.

The manifest gains no new key: ``Walk.digest`` lives in the sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import _contracts
from ..ir import HawkError

TOP_LEVEL_KEY_ORDER_V2 = ("schema_version", "pattern", "aether_abi", "exec_targets",
                          "exec_access", "exec_op", "plugins")
#: Verbatim twins of raptor's v2 execution-axis vocabularies.
EXEC_TARGETS = ("device", "host")
EXEC_ACCESS_CLASSES = ("sample_local", "cross_sample_read", "cross_sample_write",
                       "mapreduce")
EXEC_OPS = ("sum", "times", "max", "land")


def manifest_doc(entries, *, exec_targets, exec_access, exec_op=None,
                 pattern: str = "pure") -> dict:
    """The manifest document for one bundle. ``entries`` are the
    per-plugin records :func:`plugin_entry` builds."""
    bad = [t for t in exec_targets if t not in EXEC_TARGETS]
    if bad or not exec_targets or len(set(exec_targets)) != len(exec_targets):
        raise HawkError(
            f"exec_targets must be a non-empty duplicate-free subset of "
            f"{EXEC_TARGETS}; got {list(exec_targets)}"
        )
    if exec_access not in EXEC_ACCESS_CLASSES:
        raise HawkError(f"exec_access {exec_access!r} is not one of "
                        f"{EXEC_ACCESS_CLASSES}")
    if (exec_access == "mapreduce") != (exec_op is not None):
        raise HawkError(
            f"exec_op is REQUIRED iff exec_access == 'mapreduce' and forbidden "
            f"otherwise; got exec_access={exec_access!r}, exec_op={exec_op!r}"
        )
    if exec_op is not None and exec_op not in EXEC_OPS:
        raise HawkError(f"exec_op {exec_op!r} is not one of {EXEC_OPS}")
    doc = {
        "schema_version": _contracts.MAX_SCHEMA_VERSION,
        "pattern": pattern,
        "aether_abi": _contracts.AETHER_ABI_VERSION,
        "exec_targets": list(exec_targets),
        "exec_access": exec_access,
    }
    if exec_op is not None:
        doc["exec_op"] = exec_op
    doc["plugins"] = list(entries)
    order = [k for k in TOP_LEVEL_KEY_ORDER_V2 if k in doc]
    if list(doc) != order:                    # pragma: no cover - built in order
        raise HawkError(f"manifest key order {list(doc)} != {order}")
    return doc


def plugin_entry(pid: str, order: int, artifact: str, sidecar: str,
                 fmt: str = "ptx") -> dict:
    """One ``plugins[]`` record. ``order`` is the injection order (== index)."""
    return {"id": pid, "order": order, "enabled": True, "artifact": artifact,
            "sidecar": sidecar, "format": fmt}


def write_manifest(directory: Path, doc: dict) -> Path:
    """Write ``doc`` as the bundle's ``manifest.json``."""
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


