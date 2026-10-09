# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for the compiled seam (``hawk._core``).

Row classes FT-1, FT-2, FT-3 and FT-5 of the family's free-threading
acceptance, registered against the conformance harness that ``raptor-core``
hosts (a TEST dependency only: nothing under ``hawk/`` imports it).

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

import pytest
from conftest import sidecar_of

from hawk import _core
from raptor.conformance import freethreading as ft
from raptor.conformance.freethreading import declare_ft_row, register_ft_row

pytestmark = pytest.mark.ft

declare_ft_row("FT-1-GIL-FREE-HAWK-CORE", "hawk",
               "importing hawk._core leaves the GIL disabled")
declare_ft_row("FT-2-CANARY-HAWK-CORE", "hawk",
               "the planted unsynchronised counter in hawk._core loses >= 10% of updates")
declare_ft_row("FT-3-HAWK-CROSSINGS-EXACT", "hawk",
               "crossings() is exact under 8 threads x 200k calls")
declare_ft_row("FT-5-HAWK-ARGBLOCK-SHARED", "hawk",
               "one ArgBlock bound and run from many threads is never torn")
declare_ft_row("FT-5-HAWK-HOSTLIBRARY-SHARED", "hawk",
               "one HostLibrary resolving entries from many threads stays intact")
declare_ft_row("FT-LINT-HAWK-NO-GIL-RELEASE", "hawk",
               "the bindings never detach (no gil_scoped_release / call_guard)")

EXPECTED_TESTS = frozenset(
    {
        "test_manifest_all_tests_collected",
        "test_rows_declared_and_collected",
        "test_core_is_gil_free",
        "test_core_canary_loses_updates",
        "test_crossings_exact_under_threads",
        "test_shared_argblock_is_never_torn",
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


def test_rows_declared_and_collected():
    ft.assert_ft_rows_complete("hawk")


@register_ft_row("FT-1-GIL-FREE-HAWK-CORE")
def test_core_is_gil_free():
    ft.require_free_threaded()
    ft.assert_gil_free("hawk._core")


@register_ft_row("FT-2-CANARY-HAWK-CORE")
def test_core_canary_loses_updates():
    ft.require_free_threaded()
    res = ft.require_not_vacuous(
        ft.measure_loss(_core._unsynchronised_bump, _core._unsynchronised_count)
    )
    print(f"hawk._core canary lost {res.lost_fraction:.1%} of {res.expected}")


@register_ft_row("FT-3-HAWK-CROSSINGS-EXACT")
def test_crossings_exact_under_threads():
    ft.require_free_threaded()
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
_ARGBLOCK_CHILD = r"""
import faulthandler, json, sys, threading
import numpy as np
from hawk import runtime
import _oracle as O

faulthandler.enable()
faulthandler.dump_traceback_later(60, exit=True)
directory, sidecar, binders, runs = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
N = 8
kernel = O.load(directory, "vec3_scale", sidecar)
xa = np.arange(6 * N, dtype=np.float64)              # pitch N in use (3 * N valid)
xb = 1000.0 + np.arange(6 * N, dtype=np.float64)     # pitch 2N
y = np.zeros((3, N))
kernel.bind_all({"x": xa[: 3 * N].reshape(3, N).copy(), "a": 1.0, "y": y}, N)
slot = kernel.slot_of[("vec_in", "x")]
keep = []
pa, pb = runtime.buffer_address(xa, keep), runtime.buffer_address(xb, keep)
want_a = xa[: 3 * N].reshape(3, N).copy()
want_b = xb.reshape(3, 2 * N)[:, :N].copy()
kernel.block.bind(slot, pa, N, 1, 0, 0)
stop = threading.Event()
errors = []

def binder(flip):
    try:
        while not stop.is_set():
            kernel.block.bind(slot, pa, N, 1, 0, 0)
            kernel.block.bind(slot, pb, 2 * N, 1, 0, 0)
    except (RuntimeError, ValueError) as exc:        # the documented outcomes
        errors.append(repr(exc))

pool = [threading.Thread(target=binder, args=(i,)) for i in range(binders)]
for t in pool:
    t.start()
torn = 0
try:
    for _ in range(runs):
        kernel.entry.run(kernel.block, 0, N, N)
        if not (np.array_equal(y, want_a) or np.array_equal(y, want_b)):
            torn += 1
finally:
    stop.set()
    for t in pool:
        t.join()
print(json.dumps({"torn": torn, "runs": runs, "errors": errors}))
"""


@register_ft_row("FT-5-HAWK-ARGBLOCK-SHARED")
def test_shared_argblock_is_never_torn(built):
    ft.require_free_threaded()
    bundle = built["vec3"]
    proc = subprocess.run(
        [sys.executable, "-c", _ARGBLOCK_CHILD, str(bundle.directory),
         json.dumps(sidecar_of(bundle, "vec3_scale")), "7", "20000"],
        capture_output=True, text=True, timeout=180, check=False,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert proc.returncode == 0, (
        f"child died (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["errors"] == [], out["errors"]
    assert out["torn"] == 0, (
        f"{out['torn']} of {out['runs']} runs read a half-written mirror")


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
@register_ft_row("FT-5-HAWK-HOSTLIBRARY-SHARED")
def test_shared_hostlibrary_entries_stay_intact(built, tmp_path):
    ft.require_free_threaded()
    from hawk.compile import host_compiler, host_flags
    from hawk.compile.toolchain import subprocess_env

    # A real hawk artifact (its ABI tag and layout table) with many exported
    # entries, so the threads race on many distinct names. The extra entries
    # are no-ops appended to the kernel's own translation unit and built with
    # hawk's host recipe, so nothing depends on what the toolchain links in.
    names = [f"hawk_ft_entry_{i:03d}" for i in range(128)]
    tu = tmp_path / "many_entries.cpp"
    tu.write_text(
        (built["axpb"].directory / "axpb.cpp").read_text()
        + "\n"
        + "".join(
            f'extern "C" __attribute__((visibility("default"))) '
            f"void {n}(void* const*, long, long, long) {{}}\n"
            for n in names
        )
    )
    lib_path = tmp_path / "many_entries.so"
    done = subprocess.run(
        [host_compiler(), *host_flags(), str(tu), "-o", str(lib_path)],
        capture_output=True, text=True, env=subprocess_env(), check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
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
@register_ft_row("FT-LINT-HAWK-NO-GIL-RELEASE")
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
