# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The serial oracle never threads, and the audit reads what ships.

``hawk/_core`` runs the whole range in ONE call and is the correctness arm every partitioned, tiled or ranked run is compared
against. An oracle that quietly threaded would be comparing a structure against itself, so the audit is on BOTH sides of the
compiler: the sources may name no OpenMP pragma, no ``<thread>``, no pthread call (comments are stripped first, because the
files explain why they do not thread by naming what they do not do); and the BUILT object may reference no OpenMP runtime
symbol, must not reference ``pthread_create`` at all, and must link no OpenMP or pthread library. Weak (``w``) pthread
references are the ONE thing allowed, and only because a statically-linked libstdc++ emits them as inert placeholders: the
object's own strong reference to ``__libc_single_threaded`` is the corroborating evidence.

The audit covers what SHIPS (the package, the ``_core`` sources and the build file) and deliberately not ``tests/``: the tests
launch through the library and run the rank bed under ``mpirun``, which is the contract being kept. The set is asserted before
any verdict is read off it.
"""

from __future__ import annotations

import ast
import io
import re
import shutil
import subprocess
import sys
import sysconfig
import tokenize
from pathlib import Path

import pytest

import hawk
from hawk import _core

REPO = Path(hawk.__file__).resolve().parent.parent
PACKAGE = REPO / "hawk"
SOURCES = REPO / "src"

#: What SHIPS, and therefore what is a claim about. A wheel install has no
#: src/ beside the package; the tests that read this list are repo_local.
_SHIPPED = (
    [p for p in sorted(PACKAGE.rglob("*.py"))]
    + [p for p in (sorted(SOURCES.iterdir()) if SOURCES.is_dir() else []) if p.is_file()]
    + [REPO / "CMakeLists.txt"]
)

#: The source-side vocabulary: what a threading `_core` would have to name.
_THREAD_TOKENS = ("#pragma omp", "<omp.h>", "omp_set_num_threads", "<thread>",
                  "std::thread", "std::jthread", "pthread_create",
                  "std::async", "<future>", "openmp")


def _py_code(text: str) -> str:
    """``text`` with its docstrings and ``#`` comments removed. String literals
    that are NOT docstrings survive: a launch hidden in one is still a launch."""
    # Every span is collected from ONE parse before anything is replaced: a
    # replace-as-you-walk loop shifts the offsets the remaining nodes were
    # located by, and `get_source_segment` then reads off the end of the file.
    spans = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        span = ast.get_source_segment(text, node.body[0])
        if span:
            spans.append(span)
    for span in spans:
        text = text.replace(span, '"<docstring>"')
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append((tok.start[0], tok.string if tok.type != tokenize.NL else ""))
    lines: dict = {}
    for row, s in out:
        lines.setdefault(row, []).append(s)
    return "\n".join(" ".join(lines.get(r, ()))
                      for r in range(1, max(lines, default=1) + 1))


def _c_code(text: str) -> str:
    """``text`` with ``//`` and ``/* */`` comments removed. A ``//`` inside a
    string literal is left alone (an odd number of unescaped quotes before it
    means we are inside one), so ``"aether-abi/2"`` survives intact."""
    out, in_block = [], False
    for line in text.splitlines():
        if in_block:
            end = line.find("*/")
            if end < 0:
                out.append("")
                continue
            line, in_block = line[end + 2:], False
        while "/*" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            if end < 0:
                line, in_block = line[:start], True
                break
            line = line[:start] + " " + line[end + 2:]
        cut = _outside_string(line, "//")
        out.append(line if cut < 0 else line[:cut])
    return "\n".join(out)


def _hash_code(text: str) -> str:
    """CMake: ``#`` to end of line, outside a string literal."""
    out = []
    for line in text.splitlines():
        cut = _outside_string(line, "#")
        out.append(line if cut < 0 else line[:cut])
    return "\n".join(out)


def _outside_string(line: str, marker: str) -> int:
    """The first index of ``marker`` in ``line`` that is not inside a double- or
    single-quoted literal; ``-1`` when there is none."""
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif quote is None and line.startswith(marker, i):
            return i
        i += 1
    return -1


def _code_of(path: Path, text: str) -> str:
    if path.suffix == ".py":
        return _py_code(text)
    if path.suffix in (".cpp", ".h", ".hpp", ".in"):
        return _c_code(text)
    return _hash_code(text)




@pytest.mark.repo_local
def test_the_audit_scans_what_ships_and_nothing_else():
    """An audit is only as good as the set it looked at, so the set is asserted
    before any verdict is read off it (the git-enumeration-goes-blind lesson:
    assert what is on disk, not what an index says)."""
    assert len(_SHIPPED) >= 25, f"the sweep found only {len(_SHIPPED)} shipped files"
    names = {p.name for p in _SHIPPED}
    assert "hawk_core.cpp" in names, (
        "the audit must cover the C++ — it is the ONE file that could plausibly "
        "hold a launch path, and a sweep that quietly covered only .py would miss it"
    )
    assert "CMakeLists.txt" in names and "__init__.py" in names
    for path in _SHIPPED:
        assert path.is_file(), path
        assert "tests" not in path.relative_to(REPO).parts, (
            f"{path}: the audit must not scan the tests — they launch THROUGH eagle "
            "and drive the rank bed under mpirun, which is being kept"
        )


# --------------------------------------------------------------------------- #
# — the serial oracle, on both sides of the compiler.
# --------------------------------------------------------------------------- #
# a wheel install has no src/ beside the package: this test is repo_local and deselected there
@pytest.mark.repo_local
@pytest.mark.parametrize("path", sorted(SOURCES.iterdir()) if SOURCES.is_dir() else [],
                         ids=lambda p: p.name)
def test_the_core_sources_name_no_thread(path):
    text = path.read_text().lower()
    hits = [t for t in _THREAD_TOKENS if t.lower() in _code_of(path, text)]
    assert not hits, (
        f"{path.name}: the serial oracle names an OpenMP / threading token: {hits}. "
        "Threading is eagle's HostTeam; this path is the reference every "
        "partitioned run is judged against"
    )


def _core_so() -> Path:
    path = Path(_core.__file__)
    assert path.is_file(), f"hawk._core has no file on disk: {path}"
    return path


def _nm(*flags) -> list:
    tool = shutil.which("nm")
    assert tool, "the symbol arm needs `nm`; a skipped audit certifies nothing"
    done = subprocess.run([tool, *flags, str(_core_so())], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


@pytest.mark.needs_toolchain("linux")
def test_the_built_core_references_no_openmp_runtime():
    lines = _nm("-D")
    hits = [ln for ln in lines
            if re.search(r"\b(GOMP_|omp_|__kmpc|kmp_)", ln)]
    assert not hits, (
        f"hawk._core references an OpenMP runtime symbol: {hits[:5]} — the serial "
        "oracle opens no parallel region"
    )


# reads the repository build's toolchain markers (glibc 2.32+); a manylinux_2_28 wheel is built against glibc 2.28
@pytest.mark.repo_local
@pytest.mark.needs_toolchain("linux")
def test_the_built_core_spawns_no_thread():
    """``pthread_create`` at all is the bright line, and so is any other symbol
    that starts, joins or detaches a thread. References to the pthread LOCKING
    primitives (mutex, condition variable, once, thread-local keys) are allowed:
    a statically-linked libstdc++ uses them for its own internal locks and they do
    nothing in a single-threaded process. Against glibc < 2.34 they are weak
    (``w``) placeholders; from glibc 2.34 on pthread lives in libc itself and they
    are ordinary (``U``) references, so their strength says nothing about
    threading."""
    lines = _nm("-D")
    if _static_libstdcxx() and sysconfig.get_config_var("Py_GIL_DISABLED"):
        # nanobind's free-threaded build calls std::thread::hardware_concurrency,
        # which a static libstdc++ packs in one object file with std::thread's
        # start routine, so pthread_create is referenced without ever being
        # called. The symbols cannot tell that apart in a stripped object; the
        # behavioural row below (test_the_core_starts_no_thread_when_used) is the
        # evidence for this build.
        pytest.skip("free-threaded build with a static libstdc++: judged by "
                    "test_the_core_starts_no_thread_when_used")
    assert not [ln for ln in lines if "pthread_create" in ln], (
        "hawk._core references pthread_create: the serial oracle spawns a thread "
        ""
    )
    spawning = [ln for ln in lines
                if re.search(r"\b(pthread_(create|join|detach|attr_)|thrd_create)", ln)]
    assert not spawning, (
        f"hawk._core references a thread start/join symbol: {spawning} — only "
        "libstdc++'s internal locking primitives are expected here"
    )
    assert any("__libc_single_threaded" in ln for ln in lines), (
        "the object does not even reference __libc_single_threaded; the weak-pthread "
        "reading above is then unsupported and this row should be re-derived"
    )


def _static_libstdcxx() -> bool:
    """Whether hawk._core carries libstdc++ inside it (no libstdc++.so NEEDED)."""
    done = subprocess.run(["readelf", "-d", str(_core_so())],
                          capture_output=True, text=True, check=True)
    return "libstdc++.so" not in done.stdout


_THREAD_COUNT_CHILD = r"""
import os, sys, tempfile
from hawk import _core

def tasks():
    return len(os.listdir("/proc/self/task"))

before = tasks()
paths = [sys.executable, os.__file__, tempfile.gettempdir()] * 200
for _ in range(50):
    _core.crossings()
    _core.build_info()
    _core.layout_sizes()
    _core.stat_many(paths)
for _ in range(10_000):
    _core._unsynchronised_bump()
after = tasks()
print(before, after)
"""


@pytest.mark.needs_toolchain("linux")
def test_the_core_starts_no_thread_when_used():
    """Behavioural twin of the symbol rows: a fresh process exercises the core
    and its thread count must not change."""
    done = subprocess.run([sys.executable, "-c", _THREAD_COUNT_CHILD],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    before, after = (int(x) for x in done.stdout.split())
    assert after == before, f"hawk._core started threads: {before} -> {after}"


@pytest.mark.needs_toolchain("linux-tools")
def test_the_built_core_links_no_threading_library():
    tool = shutil.which("ldd")
    assert tool, "the link arm needs `ldd`"
    done = subprocess.run([tool, str(_core_so())], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    hits = [ln.strip() for ln in done.stdout.splitlines()
            if re.search(r"lib(gomp|omp|iomp5|pthread)\.", ln)]
    assert not hits, f"hawk._core links an OpenMP/pthread runtime: {hits}"
    assert "libc.so" in done.stdout, (
        f"ldd reported no libc at all — this reading is not of a shared object:\n"
        f"{done.stdout}"
    )
