# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The measured rows of ``steps_reverse="auto"``, as plain data a runtime applies.

A table, never a constant: the rows are what a probe timed (``tests/_probe_steps_reverse.py --dump-table``), and a target with
no row answers ``per_step``. The regime is the device-normalised batch: lanes per resident thread (``N / (SMs x resident threads
per SM)``, the caller reads both from the device), or the plain ``N`` on the host (whose rows were measured on one 8-core machine)."""

from __future__ import annotations


def table_for(device: bool) -> dict:
    """The ``steps_reverse`` data (``{"regime": ..., "rows": ...}``) of the host wing, or of the device wing: plain data the
    runtime's ``steps_reverse="auto"`` rule is applied to."""
    from ..facts import load

    return dict(load("cuda" if device else "cpu").get("steps_reverse", {}))


def host_spec():
    """The build spec of the host rows (no device module is imported): what the cost and stage functions read their wing from."""
    from .. import registry
    from . import TARGETS, build_spec

    return build_spec(tuple(t for t in TARGETS if not registry.traits(t).get("device", False)))
