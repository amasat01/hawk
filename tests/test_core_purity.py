# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Two claims: HAWK launches nothing, and the oracle never threads.

TWO CLAIMS, TWO INSTRUMENTS, ONE SUBJECT.

1 — HAWK HOLDS NO PRIVATE LAUNCH PATH. This forbids HAWK to hold a launch
path, choose a grid or block, open a thread team, call MPI/NCCL, or decide
residency: a device artifact is loaded and launched by ``eagle.registry.
load_manifest`` -> ``eagle.plan``, a host one by ``eagle.exec.HostTeam`` or by
the serial oracle below. The audit is over what SHIPS — the package
``hawk/hawk/``, the ``_core`` sources ``hawk/src/`` and the build file that turns
them into a binary. It deliberately does NOT scan ``hawk/tests/``: the tests
launch through eagle and run the rank bed under ``mpirun``, which is the
contract being kept, not broken — and a scanner that scanned its own token table
would report itself (the proxy-measures-itself failure).

The token rule is the repo's SINGLE spelling of it
(``hawk.emit.spelling.GEOMETRY_TOKENS`` / ``geometry_hits``), extended with the two
launch entry points this audit adds. It matches at a WORD BOUNDARY and not as a bare
substring, for the reason that module already documents: ``omp`` is a substring
of the ABI mirror's own ``compStride_`` field and of the word ``comp``, so a
naive ``in`` reports the view reconstruction — and a local variable called
``comp`` — as launch geometry.

The audit reads CODE, not prose: comments and docstrings are stripped first.
That is not a softening, it is what makes the instrument mean anything in this
repo — every file here explains WHY it does not thread, and it explains it by
NAMING the thing it does not do ("there is no ``#pragma omp`` anywhere and there
never will be", ``hawk/emit/host.py``). A comment cannot launch a kernel; a
``#pragma`` is not a comment and survives the strip; a string literal survives
it too, so an ``os.system("mpirun ...")`` is still caught.
:func:`test_the_auditor_catches_a_planted_launch` calibrates exactly that, by
planting the same token twice — once as code, once as a comment — and requiring
one hit and not two (an instrument needs a known answer).

Two spans are excised before matching, each named and each asserted to have
actually been removed, so neither exception can silently become a hole:

* each backend's ``index_prologue()`` string — the ONE place a launch-geometry
  token may appear as plain wording, not real code (warp-level spelling is a
  second, separate seam: the CUDA backend's persistent and fast entries use
  ``%laneid``/``__activemask``/shuffle reductions, which this list does not audit);
* the ``GEOMETRY_TOKENS`` assignment in ``hawk/emit/spelling.py`` — the audit's own
  vocabulary, which contains ``MPI_`` and ``nccl`` because it is what looks for
  them.

2 — THE ORACLE DOES NOT THREAD. ``hawk/_core`` runs the whole range in ONE
call and is the correctness arm every partitioned/tiled/ranked run is compared
against. An oracle that quietly threaded would be
comparing a structure against itself, so the audit is on BOTH sides of the
compiler: the sources may name no OpenMP pragma, no ``<thread>``, no pthread
call; and the BUILT object may reference no OpenMP runtime symbol, must not
reference ``pthread_create`` at all, and must link no OpenMP or pthread library.
Weak (``w``) pthread references are the ONE thing allowed, and only because a
statically-linked libstdc++ emits them as inert placeholders — the object's own
strong reference to ``__libc_single_threaded`` is the corroborating evidence.

This test previously failed via three plants, all then removed.

(a) ``#pragma omp parallel for`` added above ``hawk._core``'s serial call, with
``-fopenmp`` on the target::

    AssertionError: hawk/src/hawk_core.cpp: the serial oracle names an OpenMP /
    threading token: ['#pragma omp parallel for']
    AssertionError: hawk._core links an OpenMP runtime: ['libgomp.so.1']

(b) a ``std::thread`` spawned around the same call — the source grep caught
``<thread>``, and with the grep's token list emptied the SYMBOL arm still fired::

    AssertionError: hawk._core references pthread_create: the serial oracle
    spawns a thread

(c) the excision of ``GEOMETRY_TOKENS`` widened to excise the whole of
``hawk/emit/spelling.py`` — ``AssertionError: the audit excised a file it was
supposed to scan``, i.e. the exception cannot be widened into a hole.
"""

from __future__ import annotations

import ast
import io
import re
import shutil
import subprocess
import tokenize
from pathlib import Path

import pytest

import hawk
from hawk import _core
from hawk.emit import BACKENDS
from hawk.emit.aether import GEOMETRY_TOKENS

REPO = Path(hawk.__file__).resolve().parent.parent
PACKAGE = REPO / "hawk"
SOURCES = REPO / "src"

#: The tokens: the two launch entry points plus the launch-geometry set the
#: emitter already pins ONE spelling of. Matched at a word boundary (see above).
AUDIT_TOKENS = ("cuLaunchKernel", "cudaLaunch", *GEOMETRY_TOKENS)
_AUDIT_RE = re.compile("|".join(rf"\b{re.escape(t)}" for t in AUDIT_TOKENS))

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


def _excise(path: Path, text: str) -> tuple:
    """Remove the NAMED exceptions from one file's text; returns
    ``(text, [what was removed])`` so each removal can be asserted real."""
    removed = []
    if path.name in ("cuda.py", "host.py") and path.parent.name == "emit":
        span = BACKENDS[path.stem].index_prologue()
        assert span and span in text, (
            f"{path}: the backend's index_prologue() string is not present verbatim "
            "in its own source, so the named exception cannot be located — the "
            "audit would either scan it or excise the wrong thing"
        )
        # The placeholder keeps the file VALID and the same length in LINES: the
        # excised text is scanned as code afterwards, and a placeholder that
        # broke the parse (or moved every following line) would make the audit
        # report on a file it could no longer read.
        text = text.replace(span, "<index_prologue excised>"
                            + "\n" * span.count("\n"))
        removed.append(span)
    if path.name == "spelling.py" and path.parent.name == "emit":
        tree = ast.parse(text, filename=str(path))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "GEOMETRY_TOKENS"):
                span = ast.get_source_segment(text, node)
                assert span, f"{path}: could not locate the GEOMETRY_TOKENS assignment"
                text = text.replace(
                    span, 'GEOMETRY_TOKENS = ("<excised>")'
                    + "\n" * span.count("\n"))
                removed.append(span)
    return text, removed


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


@pytest.mark.repo_local
def test_no_shipped_file_holds_a_launch_path():
    hits = _launch_hits(_SHIPPED)
    assert not hits, (
        "HAWK holds no private launch path — it names no launch, no "
        "grid, no thread team, no MPI and no NCCL. eagle launches.\n"
        + "\n".join(hits)
    )


def _launch_hits(paths, override: dict | None = None) -> list:
    """Every audited token in the CODE of ``paths`` (``override`` supplies a
    file's text in place of what is on disk — the injection door below)."""
    override = override or {}
    hits = []
    for path in paths:
        text, _removed = _excise(path, override.get(path, path.read_text()))
        for lineno, line in enumerate(_code_of(path, text).splitlines(), 1):
            for m in _AUDIT_RE.finditer(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {m.group(0)}: "
                            f"{line.strip()}")
    return hits


@pytest.mark.repo_local
def test_the_auditor_catches_a_planted_launch():
    """The instrument, against a KNOWN answer. The same token is planted twice
    in a real shipped file — once as CODE, once inside a comment — and the audit
    must report the code one and not the comment one. Without this, "no hits"
    could equally mean "the stripper eats everything"."""
    subject = REPO / "src" / "hawk_core.cpp"
    text = subject.read_text()
    planted = text.replace(
        "std::atomic<std::uint64_t> g_crossings{0};",
        "std::atomic<std::uint64_t> g_crossings{0};\n"
        "// a comment that merely NAMES cuLaunchKernel and #pragma omp\n"
        "void hawk_plant() { cuLaunchKernel(nullptr); }\n"
        "#pragma omp parallel for\n")
    assert planted != text, "the plant did not apply; this row is measuring nothing"
    hits = _launch_hits([subject], {subject: planted})
    tokens = sorted({h.split(": ")[1] for h in hits})
    assert tokens == ["cuLaunchKernel", "omp"], hits
    assert len(hits) == 2, (
        f"expected exactly the two CODE plants, got {len(hits)}: {hits}. More means "
        "the comment was scanned; fewer means a directive was stripped"
    )
    assert not _launch_hits([subject]), "the un-planted file must stay clean"


@pytest.mark.repo_local
def test_the_two_named_exceptions_are_real_and_narrow():
    """An exception that excised nothing would be decorative; one that excised a
    whole file would be a hole. Both are pinned."""
    excised = {}
    for path in _SHIPPED:
        original = path.read_text()
        text, removed = _excise(path, original)
        if removed:
            excised[path.name] = removed
            assert len(text) > 0.5 * len(original), (
                f"the audit excised a file it was supposed to scan: {path}"
            )
    assert set(excised) == {"cuda.py", "host.py", "spelling.py"}, sorted(excised)
    # the geometry token table is the excision that actually HIDES tokens; the
    # prologue exception is plain wording and is asserted only to be real.
    assert _AUDIT_RE.search(excised["spelling.py"][0]), (
        "the GEOMETRY_TOKENS excision removes no audited token, so it is not the "
        "exception it claims to be"
    )
    assert "blockIdx" in excised["cuda.py"][0]


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
