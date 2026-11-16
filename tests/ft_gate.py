# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The pytest gates of the free-threading and interop conformance rows. raptor holds the checks (pure functions that
return the problem text); turning a problem into a skip or a failure is the tests' job."""

import pytest
from raptor.conformance import freethreading as _ft
from raptor.conformance.interop import certified_framework_problem


def require_free_threaded() -> None:
    """Skip on a GIL build; FAIL when the GIL state is forced from outside (``=0`` and ``=1`` both certify nothing)."""
    problem = _ft.free_threaded_problem()
    if problem is not None:
        pytest.skip(problem)
    reason = _ft.masking_problem()
    if reason is not None:
        pytest.fail(
            f"free-threading gate refused: {reason}; this masks a missing "
            "free-threading declaration. Unset it and rerun."
        )


def require_not_vacuous(result=None, min_lost: float = _ft.MIN_LOST_FRACTION):
    """Fail (RED, never skip) unless the canary lost at least ``min_lost``; call after :func:`require_free_threaded`."""
    result, problem = _ft.not_vacuous_problem(result, min_lost)
    if problem is not None:
        pytest.fail(problem)
    return result


def certified_framework(fw: str) -> None:
    problem = certified_framework_problem(fw)
    if problem is not None:
        pytest.fail(problem)
