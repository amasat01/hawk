# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The sidecar of a kernel's own unit, written the way a build writes it.

The host OSPREY lowers the kernel and :func:`hawk.artifact.sidecar.sidecar_of_unit` reads its unit record.
"""

from __future__ import annotations

from hawk import _ir
from hawk.artifact import bundle
from hawk.artifact.sidecar import sidecar_of_unit
from hawk.build_settings import default_settings


def lowered(kernel, kind=None, *, scalar_type: str = "float64") -> dict:
    """The answer of the host OSPREY for ``kernel`` lowered for the host."""
    targets = ("host",)
    plan = bundle._plans([kernel], [kernel.name], [kind or kernel.kind], mode=scalar_type, targets=targets,
                         settings=default_settings(), machines=bundle._machines(targets, {}, "", bundle.default_cache_dir()),
                         sizes=None)[0]
    return bundle._lower(plan, targets)


def sidecar_of_kernel(kernel, unit_name: str | None = None, *, kind=None, fmt: str = "host",
                      scalar_type: str = "float64", **kw) -> dict:
    """The sidecar of the unit ``unit_name`` (default: the kernel's own) of the answer for ``kernel`` lowered for the host.

    ``kw`` are the keywords of :func:`sidecar_of_unit` (``steps``, ``block``, ...).
    """
    answer = lowered(kernel, kind, scalar_type=scalar_type)
    unit = next(u for u in answer["units"] if u["meta"]["name"] == (unit_name or kernel.name))
    return sidecar_of_unit(kernel, unit, lead_device=False, fmt=fmt, scalar_type=scalar_type, exec_targets=["host"],
                           family=answer["units"], **kw)


def finish_mask(kernel):
    """The name of the mask the kernel finishes its sample on (its ``Finish`` sink's), ``None`` when it never finishes."""
    return next((s.field_text("name") for s in kernel.sinks if _ir.is_a(s, "finish")), None)
