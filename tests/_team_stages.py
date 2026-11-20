# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Run a split kernel with team stages, stage by stage, through hawk's own verbs.

The library launches every stage of a split kernel, a team stage with one block per lane included, and makes the
planes between the stages; a row binds the declared planes of the kernel and reads its outputs after the wait.
"""

from __future__ import annotations

import pytest
from conftest import _launch


def needs_hawk(verb: str) -> None:
    """Skip the calling device arm: hawk has no verb that gives it what it reads or stops at. ``verb`` states it."""
    pytest.skip(f"needs a hawk verb: {verb}")


def run_staged(bundle, name: str, planes: dict, target: str) -> None:
    """Run kernel ``name`` of ``bundle`` once over ``planes`` (NumPy arrays for ``"host"``, device arrays for ``"cuda"``) and wait.

    A launch over device memory is queued, so the wait is what makes the planes readable."""
    bound = _launch(bundle, name, target, planes)
    bound.launch()
    bound.wait()
