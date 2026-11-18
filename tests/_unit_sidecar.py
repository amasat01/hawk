# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The sidecar of a kernel's own unit, written the way a build writes it: the lowering library lowers the kernel and
:func:`hawk.artifact.sidecar.sidecar_of_unit` reads its unit record."""

from __future__ import annotations

from aether.wings.base.wrapper import scalar_mode

from hawk import _ir
from hawk.artifact import bundle, request
from hawk.artifact.sidecar import sidecar_of_unit
from hawk.targets import build_spec


def lowered(kernel, kind=None, *, scalar_type: str = "float64") -> dict:
    """The library's answer for ``kernel`` lowered for the host."""
    spec = build_spec(("host",), "").with_settings(scalar=scalar_type)
    asked, dump, dialects = bundle._plans(
        [kernel], [kernel.name], [kind or kernel.kind], smode=scalar_mode(scalar_type), targets=("host",), spec=spec,
        options={})[0]
    return request.lower(asked, dump, dialects)


def sidecar_of_kernel(kernel, unit_name: str | None = None, *, kind=None, fmt: str = "host",
                      scalar_type: str = "float64", **kw) -> dict:
    """The sidecar of the unit ``unit_name`` (default: the kernel's own) of the library's answer for ``kernel`` lowered for
    the host. ``kw`` are the keywords of :func:`sidecar_of_unit` (``steps``, ``block``, ``ptxas``, ...)."""
    answer = lowered(kernel, kind, scalar_type=scalar_type)
    unit = next(u for u in answer["units"] if u["meta"]["name"] == (unit_name or kernel.name))
    found = sidecar_of_unit(kernel, unit, lead_device=False, fmt=fmt, scalar_type=scalar_type, exec_targets=["host"],
                            family=answer["units"], **kw)
    found.pop("aether_include")
    found.pop("eagle_include")
    return found


def finish_mask(kernel):
    """The name of the mask the kernel finishes its sample on (its ``Finish`` sink's), ``None`` when it never finishes."""
    return next((s.field_text("name") for s in kernel.sinks if _ir.is_a(s, "finish")), None)
