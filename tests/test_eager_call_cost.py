# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An eager whole-plane call asks the library's manifest through one binding context, made once for the process.

Oracle: a counter around the context constructor, and ``np.cumsum`` for the values.
"""

from __future__ import annotations

import threading

import _osprey
import numpy as np

import hawk
from hawk import ospreys


def test_a_hundred_eager_scans_build_the_manifest_context_once(monkeypatch):
    built = []
    real = _osprey.Ctx

    def counting():
        built.append(1)
        return real()

    x = np.arange(100, dtype=np.float64)
    out = np.zeros(100)
    hawk.scan(
        x, "sum", out=out
    )  # opens the library once, which makes a context of its own
    ospreys._manifest_ctx.cache_clear()
    monkeypatch.setattr(_osprey, "Ctx", counting)
    for _ in range(100):
        hawk.scan(x, "sum", out=out)
        hawk.scan(x, "sum")
    assert len(built) == 1
    np.testing.assert_array_equal(out, np.cumsum(x))


def test_two_threads_of_eager_scans_each_get_the_right_values():
    x = np.arange(1, 101, dtype=np.int32)
    wanted = np.cumsum(x, dtype=np.int32)
    failures = []

    def work():
        for _ in range(50):
            got = hawk.scan(x, "sum")
            if not (got == wanted).all():
                failures.append(got)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
