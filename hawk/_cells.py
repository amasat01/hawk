# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The names of the planes a kernel that reduces over samples keeps beside its cells.

A kernel that stores ``hawk.sum(x)`` (or ``hawk.mean``, ``hawk.max``, ``hawk.min``) writes one row per
``cell_row`` samples; the library folds the rows into the cell the caller bound (see :mod:`hawk.runtime`).
"""

from __future__ import annotations

from . import _ir


def comp_name(name: str) -> str:
    """The companion plane a summing sample sum keeps beside its target ``name``."""
    return f"{name}{_ir.SAMPLE_SUM_COMP_SUFFIX}"


def fold_entry_name(kernel: str, target: str) -> str:
    """The entry of ``kernel`` that folds the owned companion into ``target`` (a host entry adds ``_host``)."""
    return f"{kernel}__fold_{target}"
