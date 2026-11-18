# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gates and helpers of the free-threading rows of ``hawk._core``: skip on a GIL build, fail on a forced GIL state."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

CANARY_THREADS = 8
CANARY_ITERATIONS = 200_000
MIN_LOST_FRACTION = 0.10


def require_free_threaded() -> None:
    """Skip on a GIL build; FAIL when the GIL state is forced from outside (both forced values certify nothing)."""
    if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        pytest.skip("not a free-threaded interpreter")
    if os.environ.get("PYTHON_GIL") is not None:
        pytest.fail("free-threading gate refused: PYTHON_GIL is set, which masks a missing declaration; unset it and rerun")


def hammer(fn, threads: int = CANARY_THREADS, iterations: int = CANARY_ITERATIONS) -> None:
    """Run ``fn()`` ``iterations`` times on each of ``threads`` threads started together; re-raise the first error."""
    barrier = threading.Barrier(threads)
    errors: list = []

    def work() -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    pool = [threading.Thread(target=work) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors:
        raise errors[0]


def measure_loss(bump, read, threads: int = CANARY_THREADS, iterations: int = CANARY_ITERATIONS):
    """Hammer a deliberately unsynchronised ``bump``; return the counter's expected value and the fraction of updates lost."""
    before = read()
    hammer(bump, threads, iterations)
    expected = threads * iterations
    return SimpleNamespace(expected=expected, lost_fraction=(expected - (read() - before)) / expected)


def require_not_vacuous(result, min_lost: float = MIN_LOST_FRACTION):
    """Fail (never skip) unless the canary lost at least ``min_lost`` of its updates; return ``result``."""
    if result.lost_fraction < min_lost:
        pytest.fail(f"the canary lost {result.lost_fraction:.1%} of {result.expected} updates, under {min_lost:.0%}: "
                    "the rows prove nothing")
    return result


def assert_gil_free(module: str) -> None:
    """Import ``module`` in a fresh interpreter and require that the GIL stays disabled."""
    out = subprocess.run([sys.executable, "-c", f"import importlib, sys; importlib.import_module({module!r}); "
                          "print('GIL_ENABLED=' + str(sys._is_gil_enabled()))"], capture_output=True, text=True)
    assert "GIL_ENABLED=False" in out.stdout, (out.stdout, out.stderr)
