# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for hawk's Python-side process state (FT-3).

Every assertion is an EXACT count or a whole-file check, never "no exception".
None of these rows needs a free-threaded interpreter to be meaningful (the
file-publish rows and the pip-root row are red on a GIL build too, since file
I/O and the widened windows yield the GIL); on 3.14t they additionally cover
the unlocked ``+=`` on the module counters.

* ``COUNTERS["builds"]`` and the publish counters: exact under 8 threads.
* the publish log: a bounded ring that never raises under contention.
* a published unit file / stamp is never observable half-written, and no
  ``.tmp.*`` is left behind.
* the primitive registry: of N threads registering ONE name, exactly one wins
  and the other N-1 raise.
* the pip-only include root is materialised once, with one ``atexit`` hook.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from hawk.artifact import build_bundle, unit_cache
from hawk.artifact import bundle as bundle_mod
from hawk.compile import toolchain as tc
import hawk.ext.primitive  # noqa: F401  (the name on hawk.ext is the function)
from hawk.ir import HawkError
from raptor.conformance import freethreading as ft
from raptor.conformance.freethreading import declare_ft_row, register_ft_row

pytestmark = pytest.mark.ft

declare_ft_row("FT-3-HAWK-BUILDS-EXACT", "hawk",
               "COUNTERS['builds'] and the publish counters are exact under 8 threads")
declare_ft_row("FT-3-HAWK-PUBLISH-LOG-RING", "hawk",
               "the publish log stays a bounded ring and never raises under contention")
declare_ft_row("FT-3-HAWK-UNIT-FILE-ATOMIC", "hawk",
               "a unit file or stamp is never observed partial; no .tmp.* is left")
declare_ft_row("FT-3-HAWK-REGISTRY-ONE-WINNER", "hawk",
               "16 threads registering one primitive name: one winner, 15 raise")
declare_ft_row("FT-3-HAWK-PIP-ROOT-ONCE", "hawk",
               "16 threads needing the pip-only root: one root, one atexit hook")

EXPECTED_TESTS = frozenset({
    "test_manifest_all_tests_collected",
    "test_rows_declared_and_collected",
    "test_builds_counter_is_exact",
    "test_publish_log_is_a_bounded_ring_under_contention",
    "test_unit_files_are_never_observed_partial",
    "test_one_registry_winner_the_losers_raise",
    "test_one_pip_only_root",
})

prim_mod = sys.modules["hawk.ext.primitive"]
THREADS = 8


def test_manifest_all_tests_collected(request):
    here = {i.name for i in request.session.items if i.module is sys.modules[__name__]}
    assert EXPECTED_TESTS <= here, sorted(EXPECTED_TESTS - here)
    assert {n for n in here if "[" not in n} <= EXPECTED_TESTS


def test_rows_declared_and_collected():
    ft.assert_ft_rows_complete("hawk")


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


@register_ft_row("FT-3-HAWK-BUILDS-EXACT")
def test_builds_counter_is_exact(tmp_path):
    per_thread = 20_000
    before = bundle_mod.counters_snapshot()["builds"]

    def calls():
        for _ in range(per_thread):
            # an attempt that raises at once still counts, and costs nothing
            with pytest.raises(HawkError):
                build_bundle([], tmp_path / "none", targets=("host",))

    _, errors = _run_threads([calls] * THREADS)
    assert errors == [None] * THREADS
    assert bundle_mod.counters_snapshot()["builds"] - before == THREADS * per_thread

    unit_cache.reset_unit_memo()
    n = 20_000

    def notes():
        for _ in range(n):
            unit_cache._note("memo_hits", "x")
            unit_cache._note("stamp_hits", "y")
            unit_cache._note("published", "z")

    _, errors = _run_threads([notes] * THREADS)
    assert errors == [None] * THREADS
    assert unit_cache.unit_stats() == {k: THREADS * n for k in
                                       ("memo_hits", "stamp_hits", "published")}


@register_ft_row("FT-3-HAWK-PUBLISH-LOG-RING")
def test_publish_log_is_a_bounded_ring_under_contention():
    unit_cache.reset_unit_memo()
    n = 20_000

    def notes():
        for i in range(n):
            unit_cache._log(f"m{i}")
            unit_cache.publish_log()

    _, errors = _run_threads([notes] * THREADS)
    assert errors == [None] * THREADS
    assert len(unit_cache.publish_log()) == unit_cache._PUBLISH_LOG_LIMIT


@register_ft_row("FT-3-HAWK-UNIT-FILE-ATOMIC")
def test_unit_files_are_never_observed_partial(tmp_path):
    blob = bytes(range(256)) * (2 * 1024 * 1024 // 256)          # 2 MiB
    digest = bundle_mod._digest_bytes(blob)
    inline = bundle_mod._File("payload.bin", digest, blob=blob)
    src = tmp_path / "src.bin"
    src.write_bytes(blob)
    copied = bundle_mod._File("copied.bin", digest, source_path=src)
    out = tmp_path / "unit"
    out.mkdir()
    stop = threading.Event()
    seen_bad: list = []

    def reader():
        while not stop.is_set():
            for name in ("payload.bin", "copied.bin"):
                p = out / name
                try:
                    data = p.read_bytes()
                except FileNotFoundError:
                    continue
                if bundle_mod._digest_bytes(data) != digest:
                    seen_bad.append((name, len(data)))
            try:
                doc = json.loads((out / unit_cache.STAMP_NAME).read_text())
            except FileNotFoundError:
                continue
            except ValueError as exc:
                seen_bad.append(("stamp", str(exc)))
                continue
            if doc.get("unit") != "u":
                seen_bad.append(("stamp-content", doc))

    def writer():
        for _ in range(40):
            inline.write(out)
            copied.write(out)
            unit_cache._write_stamp(out, "u", (inline, copied))

    rd = [threading.Thread(target=reader) for _ in range(3)]
    for t in rd:
        t.start()
    _, errors = _run_threads([writer] * 3)
    stop.set()
    for t in rd:
        t.join()
    assert errors == [None] * 3
    assert seen_bad == []
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["payload.bin", "copied.bin", unit_cache.STAMP_NAME])
    assert list(tmp_path.rglob(".tmp.*")) == []
    assert unit_cache._stamp_digest(out) == "u"


@register_ft_row("FT-3-HAWK-REGISTRY-ONE-WINNER")
def test_one_registry_winner_the_losers_raise(monkeypatch):
    monkeypatch.setattr(prim_mod, "_REGISTRY", dict(prim_mod._REGISTRY))
    threads, rounds = 16, 200
    barrier = threading.Barrier(threads)
    wins = {r: [] for r in range(rounds)}
    losses = {r: 0 for r in range(rounds)}
    lock = threading.Lock()

    def work(t):
        for r in range(rounds):
            barrier.wait()
            try:
                d = prim_mod.primitive(f"ft_state_prim_{r}")(lambda x: x)
                with lock:
                    wins[r].append((t, d))
            except HawkError:
                with lock:
                    losses[r] += 1

    pool = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    bad = {r: (len(wins[r]), losses[r]) for r in range(rounds)
           if (len(wins[r]), losses[r]) != (1, threads - 1)}
    assert bad == {}
    assert all(prim_mod._REGISTRY[f"ft_state_prim_{r}"] is wins[r][0][1]
               for r in range(rounds))


@register_ft_row("FT-3-HAWK-PIP-ROOT-ONCE")
def test_one_pip_only_root(monkeypatch, tmp_path):
    import atexit
    import time

    from hawk.compile import payload as payload_mod

    entered, hooks = [], []

    class _Served:
        def __enter__(self):
            time.sleep(0.05)               # widen the check-then-act window
            entered.append(1)
            return tmp_path / f"root{len(entered)}"

        def __exit__(self, *exc):
            return None

    class _Payload:
        def serve(self):
            return _Served()

    monkeypatch.setattr(payload_mod, "current_payload", lambda: _Payload())
    monkeypatch.setattr(tc, "_PIP_ONLY_ROOT", None)
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: hooks.append(fn))
    results, errors = _run_threads([tc._pip_only_include_root] * 16)
    assert errors == [None] * 16
    assert len(entered) == 1
    assert len(hooks) == 1
    assert len(set(results)) == 1 and results[0] == tc.pip_only_root()
