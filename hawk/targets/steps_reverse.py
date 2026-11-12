# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The chooser of ``steps_reverse="auto"`` over variant rows; the choosing rule is :func:`raptor.ir.steps_class.choose`.

A table, never a constant: the rows are what a probe timed (``tests/_probe_steps_reverse.py --dump-table``), and a target with
no row answers ``per_step``. The regime is the device-normalised batch: lanes per resident thread (``N / (SMs x resident threads
per SM)``, the caller reads both from the device), or the plain ``N`` on the host (whose rows were measured on one 8-core machine)."""

from __future__ import annotations

from raptor.ir.steps_class import choose


def choose_for(variant: dict, width: int, K: int, n: int, resident: int = 0) -> str:
    """``"fused"`` when the nearest measured row of ``variant`` won at ``n`` samples on a target holding ``resident`` threads
    (``SM count x resident threads per SM``: required when the variant's regime is ``per_resident``, ignored otherwise), else
    ``"per_step"``. The rows and the regime come from the variant's ``steps_reverse`` data."""
    table = variant.get("steps_reverse", {})
    per_resident = table.get("regime") == "per_resident"
    if per_resident and resident <= 0:
        raise ValueError("steps_reverse._choose: a device target needs resident = SM count x resident threads per SM")
    return choose(table.get("rows", ()), width, K, n / (resident if per_resident else 1))



def table_for(device: bool) -> dict:
    """The ``steps_reverse`` data (``{"regime": ..., "rows": ...}``) of the host variant, or of the device variant: plain data a
    caller hands to :func:`raptor.ir.steps_class.choose`."""
    from ..variants import load

    return dict(load("cuda" if device else "cpu").get("steps_reverse", {}))


def host_spec():
    """The build spec of the host rows (no device module is imported): what the cost and stage functions read their variant from."""
    from .. import registry
    from . import TARGETS, build_spec

    return build_spec(tuple(t for t in TARGETS if not registry.traits(t).get("device", False)))
