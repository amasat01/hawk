# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A host segmented unit visits ``[offsets[j] + base, offsets[j] + base + count)``
of its run, so a team may cut one run across threads: four cuts of a run
visit every sample of it exactly once, and no sample outside it."""

from __future__ import annotations

import json
import pathlib

import _oracle as O
import numpy as np
import pytest

import hawk
import hawk.math as M
from hawk import Index, Mutable, Scalar, Vector
from hawk.artifact import build_bundle

N = 103
K = 3


def _branches(bs):
    return [bs[j] * (j + 1.0) for j in range(K)]


@hawk.kernel(steps=1)
def accumulate(k: Index, bs: Vector[K], y: Mutable[Scalar]):
    y = y + M.dispatch(k, _branches(bs), policy="segmented")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    root = pathlib.Path(tmp_path_factory.mktemp("segsplit"))
    return build_bundle([accumulate], root / "seg", targets=("cuda", "host"), cache_dir=cache_dir)


def _cuts(count: int, parts: int) -> list:
    edges = np.linspace(0, count, parts + 1).astype(int)
    return [(int(a), int(b - a)) for a, b in zip(edges[:-1], edges[1:])]


@pytest.mark.parametrize("parts", [1, 4, 7])
def test_cutting_a_run_across_threads_visits_each_sample_once(bundle, parts):
    rng = np.random.default_rng(5)
    counts = [40, 33, 30]
    k = np.concatenate([np.full(c, j, dtype=np.int64) for j, c in enumerate(counts)])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    bs = rng.uniform(0.5, 2.0, (K, N))
    y0 = rng.uniform(-1.0, 1.0, N)
    want = y0 + np.array([bs[kj, i] * (kj + 1.0) for i, kj in enumerate(k)])
    y = y0.copy()
    cls = np.full(N, -1, dtype=np.int64)
    stale = np.zeros(N, dtype=bool)
    for j in range(K):
        name = f"accumulate__k{j}"
        sidecar = json.loads((bundle.directory / f"{name}.json").read_text())
        kernel = O.load(bundle.directory, name, sidecar)
        for base, count in _cuts(counts[j], parts):
            O.run_kernel(kernel, N, base=base, count=count, k=k, bs=bs, y=y,
                         dispatch0_offsets=offsets, dispatch0_class=cls, dispatch0_stale=stale)
    np.testing.assert_allclose(y, want, rtol=1e-14, atol=0)   # a double visit would add a whole branch value
    assert not stale.any()
