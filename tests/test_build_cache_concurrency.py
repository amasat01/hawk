# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Several processes that build the same kernel into one cache at once each end with a bundle that loads and runs.

The build publishes every file whole, to a private sibling and then by one rename, so a process that races another to the same
unit sees the old file or the new one and never a partial one; a second builder of the same unit finds it published and leaves
it. The rows build from separate processes, into separate folders and into one shared folder, and load the result in each."""

from __future__ import annotations

import multiprocessing

import numpy as np
import pytest

PROCESSES = 4
ROUNDS = 3


def _build_and_run(job: tuple) -> bool:
    """Build ``race_kernel`` into ``folder`` under the shared ``cache``, load it and run it; whether the answer is right."""
    cache, folder = job
    import hawk

    @hawk.kernel
    def race_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x * 2.0 + 1.0

    bundle = hawk.build([race_kernel], folder, targets=("host",), cache_dir=cache)
    x, y = np.arange(16.0), np.zeros(16)
    hawk.run(hawk.load(bundle.directory, "race_kernel"), x=x, y=y)
    return bool(np.array_equal(y, 2.0 * x + 1.0))


@pytest.mark.parametrize("shared_folder", [False, True], ids=["own_folders", "one_folder"])
def test_processes_building_the_same_kernel_into_one_cache_each_load_and_run_it(tmp_path, shared_folder):
    context = multiprocessing.get_context("spawn")
    for round_ in range(ROUNDS):
        cache = str(tmp_path / f"cache{round_}")
        jobs = [(cache, str(tmp_path / (f"shared{round_}" if shared_folder else f"p{round_}_{i}"))) for i in range(PROCESSES)]
        with context.Pool(PROCESSES) as pool:
            assert pool.map(_build_and_run, jobs) == [True] * PROCESSES
