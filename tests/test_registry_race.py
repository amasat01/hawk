# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading row for hawk's process state: the primitive registry claims a name in one step."""

from __future__ import annotations

import sys
import threading

from hawk import HawkError
from hawk.ext import primitive

REGISTRY_MODULE = sys.modules["hawk.ext.primitive"]


def test_of_sixteen_threads_declaring_one_primitive_name_exactly_one_wins_and_the_rest_are_refused(
    monkeypatch,
):
    """Over 200 rounds each name has one winner whose declaration is the one registered, and fifteen refusals."""
    monkeypatch.setattr(REGISTRY_MODULE, "_REGISTRY", dict(REGISTRY_MODULE._REGISTRY))
    threads, rounds = 16, 200
    barrier = threading.Barrier(threads)
    wins = {r: [] for r in range(rounds)}
    losses = dict.fromkeys(range(rounds), 0)
    lock = threading.Lock()

    def work(t):
        for r in range(rounds):
            barrier.wait()
            try:
                declared = primitive(f"race_prim_{r}")(eval(f"lambda x: x + {t}"))
                with lock:
                    wins[r].append(declared)
            except HawkError:
                with lock:
                    losses[r] += 1

    pool = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    odd = {
        r: (len(wins[r]), losses[r])
        for r in range(rounds)
        if (len(wins[r]), losses[r]) != (1, threads - 1)
    }
    assert odd == {}
    assert all(
        REGISTRY_MODULE._REGISTRY[f"race_prim_{r}"] is wins[r][0] for r in range(rounds)
    )
