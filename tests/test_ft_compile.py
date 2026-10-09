# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for the compile path (FT-3).

Every assertion is an EXACT count, never "no exception", and none needs a
free-threaded interpreter to mean something: the single-flight rows count real
compiler invocations (a shim around ``subprocess.run`` counts the ones that
build a shared object) and the cache's own hit/miss counters, so they are
red on a GIL build whenever two threads both compile one key.

* 16 threads, ONE source, cold cache -> exactly 1 compile, 1 miss, 15 hits,
  one artifact, no ``.tmp.*`` left in the slot.
* 16 threads, 16 distinct sources -> exactly 16 compiles.
* a failing compile releases every waiter (no hang, nothing left in flight).
* the address-space cap is applied by the exec trampoline, from many threads.
* ``ClosureWatch`` re-sync under contention never tears a row.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from hawk.compile import cache as cache_mod
from hawk.compile import drivers
from hawk.compile.drivers import CompileOptions, compile_source
from hawk.ir import HawkError

ft = pytest.importorskip("raptor.conformance.freethreading")
ft.declare_ft_row("FT-3-HAWK-COMPILE-SINGLE-FLIGHT", "hawk",
                  "16 threads, one cold source -> exactly one compile, no .tmp.*")
ft.declare_ft_row("FT-3-HAWK-COMPILE-DISTINCT", "hawk",
                  "16 threads, 16 distinct sources -> exactly 16 compiles")
ft.declare_ft_row("FT-3-HAWK-CAP-TRAMPOLINE", "hawk",
                  "the address-space cap is applied through the exec trampoline, "
                  "concurrently")
ft.declare_ft_row("FT-3-HAWK-CLOSURE-WATCH", "hawk",
                  "ClosureWatch re-sync under contention keeps every row whole")

THREADS = 16


def _source(i: int) -> str:
    return f'extern "C" int hawk_ft_probe_{i}(void) {{ return {i} + 1; }}\n'


@pytest.fixture
def compiles(monkeypatch):
    """Counts the real compiler invocations (the ones building a ``.so``)."""
    real = subprocess.run
    lock = threading.Lock()
    box = {"n": 0}

    def counting(argv, *a, **kw):
        if "-shared" in argv:
            with lock:
                box["n"] += 1
        return real(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", counting)
    cache_mod.reset_cache_stats()
    cache_mod.reset_artifact_memo()
    return box


def _run_threads(fns):
    """Start every callable at once; return ``(results, errors)`` by index."""
    barrier = threading.Barrier(len(fns))
    results, errors = [None] * len(fns), [None] * len(fns)

    def work(i):
        barrier.wait()
        try:
            results[i] = fns[i]()
        except BaseException as exc:                      # noqa: BLE001
            errors[i] = exc

    pool = [threading.Thread(target=work, args=(i,)) for i in range(len(fns))]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return results, errors


def _no_tmp(root):
    return sorted(str(p) for p in root.rglob(".tmp.*"))


@ft.register_ft_row("FT-3-HAWK-COMPILE-SINGLE-FLIGHT")
def test_same_source_cold_compiles_exactly_once(tmp_path, compiles):
    opts = CompileOptions(backend="host", cache_dir=str(tmp_path / "c"))
    results, errors = _run_threads(
        [lambda: compile_source(_source(0), "ft_same", opts)] * THREADS)
    assert errors == [None] * THREADS
    assert compiles["n"] == 1
    stats = cache_mod.cache_stats()
    assert (stats["misses"], stats["hits"]) == (1, THREADS - 1), stats
    assert sum(1 for r in results if not r.hit) == 1
    assert len({r.key for r in results}) == 1 and len({r.artifact for r in results}) == 1
    assert results[0].artifact.is_file()
    assert _no_tmp(tmp_path) == []
    assert drivers._INFLIGHT == {}


@ft.register_ft_row("FT-3-HAWK-COMPILE-DISTINCT")
def test_distinct_sources_each_compile_once(tmp_path, compiles):
    opts = CompileOptions(backend="host", cache_dir=str(tmp_path / "c"))
    results, errors = _run_threads(
        [lambda i=i: compile_source(_source(i), f"ft_d{i}", opts)
         for i in range(THREADS)])
    assert errors == [None] * THREADS
    assert compiles["n"] == THREADS
    stats = cache_mod.cache_stats()
    assert (stats["misses"], stats["hits"]) == (THREADS, 0), stats
    assert len({r.key for r in results}) == THREADS
    assert all(r.artifact.is_file() and not r.hit for r in results)
    assert _no_tmp(tmp_path) == []
    assert drivers._INFLIGHT == {}


def test_failed_compile_releases_every_waiter(tmp_path, compiles):
    opts = CompileOptions(backend="host", cache_dir=str(tmp_path / "c"))
    bad = "this is not c++\n"
    _, errors = _run_threads([lambda: compile_source(bad, "ft_bad", opts)] * 4)
    assert all(isinstance(e, HawkError) for e in errors), errors
    assert compiles["n"] == 4            # each waiter retried as its own builder
    assert drivers._INFLIGHT == {}
    assert _no_tmp(tmp_path) == []


@ft.register_ft_row("FT-3-HAWK-CAP-TRAMPOLINE")
def test_cap_is_applied_through_the_trampoline_concurrently(monkeypatch):
    cap = 3 * 1024 ** 3
    monkeypatch.setenv("HAWK_COMPILE_ADDRESS_CAP", str(cap))
    probe = [sys.executable, "-c",
             "import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])"]
    argv = drivers.limited(probe)
    assert argv[:3] == [sys.executable, "-I", "-c"] and argv[-3:] == probe

    def once():
        done = subprocess.run(drivers.limited(probe), capture_output=True, text=True)
        return done.returncode, done.stdout.strip()

    results, errors = _run_threads([once] * 8)
    assert errors == [None] * 8
    assert results == [(0, str(cap))] * 8
    monkeypatch.setenv("HAWK_COMPILE_ADDRESS_CAP", "0")
    assert drivers.limited(probe) == probe


@ft.register_ft_row("FT-3-HAWK-CLOSURE-WATCH")
def test_closure_watch_resync_under_contention(tmp_path):
    files = []
    pairs = []
    for i in range(8):
        f = tmp_path / f"h{i}.h"
        f.write_text(f"// {i}\n")
        files.append(f)
        pairs.append((str(f), cache_mod.digest_file(f)))
    watch = cache_mod.ClosureWatch(pairs)
    # same bytes, new mtime: every check must fall through to the content and
    # re-sync the row, from many threads at once
    for f in files:
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    results, errors = _run_threads([lambda: [watch._check() for _ in range(200)]] * 8)
    assert errors == [None] * 8
    assert all(all(r) and len(r) == 200 for r in results)
    assert isinstance(watch._rows, tuple) and all(isinstance(r, tuple) for r in watch._rows)
    assert [r[0] for r in watch._rows] == [p for p, _ in pairs]
    assert all(r[2] == f.stat().st_mtime_ns for r, f in zip(watch._rows, files))
    assert watch.pairs() == tuple(pairs)
