# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for the compiled seam (``hawk._core``).

Row classes FT-1, FT-2, FT-3 and FT-5 of the family's free-threading
acceptance, run with the helpers of ``tests/ft_gate.py``.

On a GIL build every row that needs real parallelism SKIPS with a reason; the
name manifest below keeps "skipped" from degrading into "silently not
collected". No row ever sets ``-X gil=0`` or ``PYTHON_GIL``: the harness
refuses to run when either is in force.

The shared-object rows run their threads in a CHILD process so that a crash is
a red verdict (non-zero exit, faulthandler traceback) and not the end of the
pytest session.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import ft_gate as gate
import ft_gate as ft
import pytest

from hawk import _core

pytestmark = pytest.mark.ft


EXPECTED_TESTS = frozenset(
    {
        "test_manifest_all_tests_collected",
        "test_core_is_gil_free",
        "test_core_canary_loses_updates",
        "test_crossings_exact_under_threads",
        "test_shared_hostlibrary_entries_stay_intact",
        "test_bindings_never_release_the_gil",
    }
)

_SRC = Path(__file__).resolve().parent.parent / "src"


def test_manifest_all_tests_collected(request):
    here = {
        i.name for i in request.session.items if i.module is sys.modules[__name__]
    }
    # a marked row (repo_local) may be deselected by the leg's -m expression;
    # every unmarked row must be collected
    required = {
        n for n in EXPECTED_TESTS
        if not getattr(globals().get(n), "pytestmark", None)
    }
    assert required <= here, sorted(required - here)
    assert {n for n in here if "[" not in n} <= EXPECTED_TESTS




def test_core_is_gil_free():
    gate.require_free_threaded()
    ft.assert_gil_free("hawk._core")


def test_core_canary_loses_updates():
    gate.require_free_threaded()
    res = gate.require_not_vacuous(
        ft.measure_loss(_core._unsynchronised_bump, _core._unsynchronised_count)
    )
    print(f"hawk._core canary lost {res.lost_fraction:.1%} of {res.expected}")


def test_crossings_exact_under_threads():
    gate.require_free_threaded()
    threads, iterations = ft.CANARY_THREADS, ft.CANARY_ITERATIONS
    before = _core.crossings()
    ft.hammer(_core.crossings, threads, iterations)
    after = _core.crossings()
    # `before` already counts its own crossing; `after` adds one more.
    assert after - before == threads * iterations + 1, (
        after - before, threads * iterations + 1)


# --------------------------------------------------------------------------- #
# FT-5: ONE stateful object shared across threads. Memory-safe, calls serialise.
# --------------------------------------------------------------------------- #
_LIBRARY_CHILD = r"""
import faulthandler, json, sys, threading
from hawk import _core

faulthandler.enable()
faulthandler.dump_traceback_later(60, exit=True)
path, rounds, nthreads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
names = json.loads(sys.argv[4])
errors, bad = [], 0
for _ in range(rounds):
    lib = _core.HostLibrary(path)
    barrier = threading.Barrier(nthreads)
    got = [None] * nthreads

    def work(k):
        try:
            barrier.wait()
            got[k] = [lib.entry(n).name for n in names]
        except (RuntimeError, ValueError) as exc:
            errors.append(repr(exc))

    pool = [threading.Thread(target=work, args=(k,)) for k in range(nthreads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    bad += sum(1 for g in got if g is not None and g != names)
print(json.dumps({"bad": bad, "errors": errors}))
"""

# Exported by libc, so a dlsym through the artifact's dependency chain finds
# each: the library's `entry()` caches by name and never calls what it finds.
def test_shared_hostlibrary_entries_stay_intact(built, tmp_path):
    gate.require_free_threaded()
    # One bundle of many units: a real hawk library (its ABI tag and layout table) that exports many distinct entries,
    # so the threads race on many distinct names.
    import _deployable as D

    from hawk.artifact import build_bundle
    from hawk.trace import Kernel

    units = [Kernel(f"hawk_ft_entry_{i:03d}", D.axpb.sinks) for i in range(32)]
    bundle = build_bundle(units, tmp_path / "many", targets=("host",), cache_dir=str(tmp_path / "cache"))
    names = [f"{unit.name}_host" for unit in units]
    lib_path = bundle.directory / "units.so"
    serial = _core.HostLibrary(str(lib_path))
    assert all(_resolves(serial, n) for n in names)
    proc = subprocess.run(
        [sys.executable, "-c", _LIBRARY_CHILD, str(lib_path), "200", "8",
         json.dumps(names)],
        capture_output=True, text=True, timeout=180, check=False,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert proc.returncode == 0, (
        f"child died (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"bad": 0, "errors": []}, out


def _resolves(lib, name: str) -> bool:
    try:
        lib.entry(name)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# A2 lint: the bindings never detach.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_bindings_never_release_the_gil():
    # nb::lock_self is a critical section that CPython suspends whenever the
    # thread detaches, so a binding that releases the GIL needs a per-object
    # busy state instead (and an allow-list entry naming it). None does.
    pattern = re.compile(r"gil_scoped_release|call_guard|Py_BEGIN_ALLOW_THREADS")
    hits = [
        f"{p.name}:{n}: {line.strip()}"
        for p in sorted(_SRC.glob("*.cpp"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("//")
    ]
    assert hits == [], hits
