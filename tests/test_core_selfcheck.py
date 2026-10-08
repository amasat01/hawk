# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``import hawk`` refuses a ``_core`` whose build digest
or layout table disagrees with these sources, NAMING the field.

THE DEFECT CLASS THIS EXISTS FOR. A stale, wrong-arch or editable-shadowed
binding imports perfectly cleanly. It resolves, it answers ``build_digest()``,
it marshals — and then it decodes a 40-byte mirror as a 32-byte one and returns
plausible numbers. A downstream package carries a consumer-side md5 pin for
exactly this failure BECAUSE eagle's binding has no in-binding digest to
read; HAWK's does, so the check is a comparison between the binding's own stamp
and the sources beside it, and it fires at ``import hawk``.

HOW THE ROW INJECTS IT. Each arm builds a SHADOW package in a temp directory —
the real ``hawk/__init__.py`` and ``hawk/_contracts.py``, a copy of ``src/`` and
``CMakeLists.txt`` (so the digest RECOMPUTE lands on the real value), and a
hand-written ``_core.py`` standing in for the extension. Putting that directory
first on ``sys.path`` in a fresh subprocess is not a simulation of the
editable-shadowing trap; it IS the editable-shadowing trap. One arm's fake is
correct and must import; the others differ in exactly ONE field and must refuse
naming it.

This test previously failed via two plants, both then removed.

(a) ``_self_check()``'s call deleted from ``hawk/__init__.py`` — all FOUR refusal
arms went red together::

    AssertionError: a binding with a foreign build digest imported
    AssertionError: a binding with a 41-byte GRefMirror imported
    AssertionError: a source edit did not move the digest: ...
    AssertionError: a binding with no sources to verify against imported

and ``test_a_correct_core_imports`` stayed green, which is what says the
harness itself was not the thing that broke.

(b) The REAL binding, against a REAL edit: one comment line appended to
``src/hawk_core.cpp`` with no rebuild — the stale-binding class exactly — and
``import hawk`` refused, naming the field and both values::

    ImportError: hawk: hawk._core's build_digest is '1678527c96...c158fcd2' but
    the sources beside it hash to 'c1cae2a209...7dad1332' — this binding was NOT
    built from these sources. Loaded binding: <site-packages>/hawk/
    _core.cpython-312-x86_64-linux-gnu.so.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hawk
from hawk import _contracts, _core

REPO = Path(hawk.__file__).resolve().parent.parent

#: A stand-in for the compiled extension: the four calls ``hawk/__init__``'s
#: self-check makes, and nothing else. Parameterised so an arm can move ONE
#: field and leave the rest correct — a fake that differed in several fields
#: could not tell us WHICH one the refusal named.
_FAKE_CORE = '''\
"""A stand-in hawk._core (row). Not the extension: the four readings the
self-check makes, with the values this arm wants them to have."""

_DIGEST = {digest!r}
_LAYOUT = {layout!r}


def build_digest():
    return _DIGEST


def layout_sizes():
    return list(_LAYOUT)


def build_info():
    return {{"build_digest": _DIGEST, "compiler": "<fake>", "compiler_id": "<fake>",
            "flags": "", "aether_include": "", "eagle_include": "",
            "abi_tag": "aether-abi/2", "index_type_bytes": "4"}}


def crossings():
    return 0
'''


def _shadow(tmp_path: Path, *, digest: str, layout, with_sources: bool = True,
            wheel_stamp: str | None = None) -> Path:
    """Lay out a shadowing ``hawk`` package whose ``_core`` is the fake above.

    ``wheel_stamp`` simulates the WHEEL path without
    moving or deleting any real file: it writes a synthetic
    ``hawk/_build_digest.py`` into the SHADOW root exactly as CMake's
    ``install(FILES ...)`` would into a real wheel, so ``with_sources=False,
    wheel_stamp=...`` is a wheel-shaped root — never the dev tree's own
    ``src/``/``CMakeLists.txt`` touched or hidden."""
    root = tmp_path
    pkg = root / "hawk"
    pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "hawk" / "__init__.py", pkg / "__init__.py")
    shutil.copy2(REPO / "hawk" / "_contracts.py", pkg / "_contracts.py")
    (pkg / "_core.py").write_text(_FAKE_CORE.format(digest=digest, layout=list(layout)))
    if with_sources:
        shutil.copytree(REPO / "src", root / "src", dirs_exist_ok=True)
        shutil.copy2(REPO / "CMakeLists.txt", root / "CMakeLists.txt")
    if wheel_stamp is not None:
        (pkg / "_build_digest.py").write_text(
            '"""Synthetic wheel stamp (row)."""\n'
            f"SOURCE_DIGEST = {wheel_stamp!r}\n")
    return root


def _import_under(root: Path, probe: str = "import hawk") -> subprocess.CompletedProcess:
    """Import the shadow package in a FRESH interpreter — never by juggling this
    session's ``sys.modules``, which would leave the real hawk half-loaded.

    ``-S`` is load-bearing and is itself part of what this row documents. A
    scikit-build-core EDITABLE install works through a ``.pth`` that registers a
    ``sys.meta_path`` finder, and a meta-path finder is consulted BEFORE
    ``sys.path`` — so ``PYTHONPATH`` alone cannot shadow it, and the arms below
    would all have imported the installed hawk and reported a clean pass.
    ``-S`` skips ``site`` processing, which is what leaves ``sys.path`` in
    charge. (The shadow package needs nothing outside the standard library:
    ``hawk/__init__`` imports ``hashlib``, ``pathlib``, ``typing`` and its own
    two modules.)"""
    return subprocess.run(
        [sys.executable, "-S", "-c", probe + "; print('IMPORTED', hawk.__file__)"],
        capture_output=True, text=True, cwd=str(root),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(root),
             "HOME": str(root)},
    )


#: This build's own truth, read from the REAL binding: the correct arm has to be
#: correct by construction, not by a number typed into this file.
TRUE_DIGEST = _core.build_digest()
TRUE_LAYOUT = _core.layout_sizes()


@pytest.mark.repo_local
def test_a_correct_core_imports(tmp_path):
    """Non-vacuity: the SAME harness, with nothing wrong, imports cleanly. Without
    this arm every refusal below could be the harness failing for its own
    reasons."""
    done = _import_under(_shadow(tmp_path, digest=TRUE_DIGEST, layout=TRUE_LAYOUT))
    assert done.returncode == 0, done.stderr
    assert "IMPORTED" in done.stdout, done.stdout
    assert str(tmp_path) in done.stdout, (
        "the shadow package must be the one that imported, or this row is "
        f"certifying the installed hawk: {done.stdout}"
    )


@pytest.mark.repo_local
def test_a_wrong_build_digest_refuses_naming_the_field(tmp_path):
    done = _import_under(_shadow(tmp_path, digest="not-the-digest",
                                 layout=TRUE_LAYOUT))
    assert done.returncode != 0, "a binding with a foreign build digest imported"
    assert "ImportError" in done.stderr, done.stderr
    assert "build_digest" in done.stderr, (
        "the refusal must NAME the field — 'your binding is wrong' is not "
        f"actionable:\n{done.stderr}"
    )
    assert "not-the-digest" in done.stderr and TRUE_DIGEST in done.stderr, (
        f"the refusal must show both values a reader has to reconcile:\n{done.stderr}"
    )


@pytest.mark.repo_local
def test_a_stale_binding_is_caught_by_a_source_EDIT(tmp_path):
    """The digest is a CONTENT hash, so editing a source moves it — which is the
    property that makes a stale binding (built before the edit) detectable at
    all. The fake reports the CURRENT digest and the tree is then changed."""
    root = _shadow(tmp_path, digest=TRUE_DIGEST, layout=TRUE_LAYOUT)
    source = root / "src" / "hawk_core.cpp"
    source.write_text(source.read_text() + "\n// a one-line change\n")
    done = _import_under(root)
    assert done.returncode != 0, (
        "a source edit did not move the digest: the recompute is not hashing "
        "contents, and a stale binding would import cleanly"
    )
    assert "build_digest" in done.stderr, done.stderr


@pytest.mark.repo_local
def test_a_wrong_layout_constant_refuses_naming_the_field(tmp_path):
    """The other reading. A 41-byte ``GRefMirror`` is the failure shape: it
    does not crash, it decodes every following slot at the wrong offset."""
    bad = list(TRUE_LAYOUT)
    bad[0] = 41
    done = _import_under(_shadow(tmp_path, digest=TRUE_DIGEST, layout=bad))
    assert done.returncode != 0, "a binding with a 41-byte GRefMirror imported"
    assert "GRefMirror" in done.stderr and "41" in done.stderr, done.stderr
    assert str(_contracts.GREF_MIRROR_SIZE) in done.stderr, done.stderr


def test_a_binding_with_neither_sources_nor_a_wheel_stamp_refuses(tmp_path):
    """ wheel path, row (iii): with NEITHER a dev tree's
    ``src/``/``CMakeLists.txt`` NOR a shipped ``hawk._build_digest`` beside the
    package, there is nothing left to certify against. A self-check that
    silently disabled itself here would certify nothing exactly where it
    matters most, so the absence is a refusal, not a pass."""
    done = _import_under(_shadow(tmp_path, digest=TRUE_DIGEST, layout=TRUE_LAYOUT,
                                 with_sources=False))
    assert done.returncode != 0, "a binding with no sources to verify against imported"
    assert "build_digest" in done.stderr, done.stderr
    assert "no wheels ship" not in done.stderr, (
        "the self-check now HAS a wheel path; the refusal message must not claim otherwise:\n"
        f"{done.stderr}"
    )


def test_a_binding_with_a_matching_wheel_stamp_imports(tmp_path):
    """ wheel path, row (i): NO dev-tree sources, but a
    shipped ``hawk._build_digest.SOURCE_DIGEST`` that agrees with ``_core``'s
    own build_digest — the exact shape a REAL wheel install is (``src/`` and
    ``CMakeLists.txt`` never ship; ``_build_digest.py`` does, via CMake's
    ``install(FILES ...)``). This must import cleanly: refusing a correctly
    matched wheel would make the wheel-only route unshippable by design."""
    done = _import_under(_shadow(tmp_path, digest=TRUE_DIGEST, layout=TRUE_LAYOUT,
                                 with_sources=False, wheel_stamp=TRUE_DIGEST))
    assert done.returncode == 0, done.stderr
    assert "IMPORTED" in done.stdout, done.stdout
    assert str(tmp_path) in done.stdout, (
        "the shadow package must be the one that imported, or this row is "
        f"certifying the installed hawk: {done.stdout}"
    )


def test_a_wheel_stamp_disagreeing_with_the_core_refuses_naming_both_digests(tmp_path):
    """ wheel path, row (ii): the SAME editable-shadowing
    trap already covers for a dev tree, now caught between a wheel's
    Python files (``hawk._build_digest``) and somebody else's ``_core`` —
    e.g. an older wheel's ``_core.so`` left importable ahead of a newer
    wheel's Python files on ``sys.path``."""
    done = _import_under(_shadow(tmp_path, digest=TRUE_DIGEST, layout=TRUE_LAYOUT,
                                 with_sources=False, wheel_stamp="not-the-stamp"))
    assert done.returncode != 0, "a binding with a foreign wheel stamp imported"
    assert "ImportError" in done.stderr, done.stderr
    assert "build_digest" in done.stderr, (
        "the refusal must NAME the field, exactly as the dev-tree arm does:\n"
        f"{done.stderr}"
    )
    assert "not-the-stamp" in done.stderr and TRUE_DIGEST in done.stderr, (
        f"the refusal must show both values a reader has to reconcile:\n{done.stderr}"
    )


# --------------------------------------------------------------------------- #
# What the REAL binding reports — the values every arm above is built from.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_the_real_binding_agrees_with_the_real_tree():
    """The installed binding passed the check at import (this module imported
    hawk), so this row states the two readings EXPLICITLY rather than leaving
    them implicit in an import that happened to work."""
    assert _core.build_digest() == hawk._source_digest(REPO)
    sizes = _core.layout_sizes()
    assert sizes[0] == _contracts.GREF_MIRROR_SIZE
    assert sizes[1] == _contracts.SCALAR_HANDLE_SIZE
    assert sizes[2] == _contracts.INT_HANDLE_SIZE
    assert sizes[4] == _contracts.PARTITION_TRIPLE_SIZE


def test_the_index_width_field_has_no_constant_twin_by_design():
    """Field 3 is a BUILD AXIS (``-DAETHER_INDEX_T``). Pinning a number for
    it in ``_contracts.py`` would refuse a legitimately-built 64-bit-index
    binding, so it is checked artifact-against-host at load instead — this row
    pins the DESIGN, so nobody 'fixes' the omission later."""
    twins = {field for _i, field, _v in hawk._LAYOUT_TWINS}
    assert "aether::idx_t" not in twins and "idx_t" not in twins
    assert _core.layout_sizes()[3] == int(_core.build_info()["index_type_bytes"])


# the recorded header roots exist only on the machine that built the binding (a wheel names its build container's)
@pytest.mark.repo_local
def test_build_info_names_both_header_roots(tmp_path):
    """(a): the resolved roots are what a card has to name as the axis
    two arms differ on, so the binding must be able to say which ones it was
    built against."""
    info = _core.build_info()
    assert Path(info["eagle_include"], "plugin", "gref_abi.h").is_file(), info
    assert Path(info["aether_include"], "aether", "typedefs.h").is_file(), info
    assert info["abi_tag"] == "aether-abi/2"
    assert json.dumps(info)          # every value is a plain string


def test_the_layout_twins_cover_every_fixed_field():
    """A twin table that quietly lost a field would weaken the check without
    failing anything, so the covered set is pinned here."""
    covered = {index for index, _f, _v in hawk._LAYOUT_TWINS}
    assert covered == {0, 1, 2, 4}, covered


@pytest.mark.parametrize("name", ["build_digest", "layout_sizes", "build_info",
                                  "crossings", "HostLibrary", "ArgBlock",
                                  "HostEntry", "layout_field_name", "stat_many"])
def test_the_core_surface_is_h67s(name):
    """The table plus the instruments add. Pinned as a SET so a new
    per-launch entry point cannot appear without this row noticing."""
    assert hasattr(_core, name), f"hawk._core exports no {name!r}"


def test_the_core_exports_nothing_else_public():
    public = {n for n in dir(_core) if not n.startswith("_")}
    assert public == {"build_digest", "layout_sizes", "layout_field_name",
                      "build_info", "crossings", "HostLibrary", "HostEntry",
                      "ArgBlock", "stat_many"}, sorted(public)


# --------------------------------------------------------------------------- #
# stat_many — the compile cache's batched stat, at the _core boundary.
# --------------------------------------------------------------------------- #
def test_stat_many_reports_a_sentinel_for_a_missing_path_and_never_throws(
        tmp_path):
    """The contract, exercised directly at the binding: a mix of a real file
    and a missing one must come back as data, not an exception.

    ``hawk.compile.cache.ClosureWatch.valid()`` calls this once over a whole
    recorded closure (hundreds of headers on a real closure); if a
    single deleted header raised, one stale header would turn a cheap per-hit
    check into an exception-handling path over every OTHER entry in the same
    closure, instead of the plain miss already treats a vanished file as.

    Both accepted path forms are checked (str and bytes), because
    ``ClosureWatch`` reserves either as the pre-encoded form and this row is
    what pins that the C++ side actually honours both."""
    present = tmp_path / "present.h"
    present.write_text("#define HW6E_PROBE 1\n")
    missing = tmp_path / "does_not_exist.h"

    sizes, mtimes_ns = _core.stat_many([str(present), str(missing)])
    assert list(sizes) == [present.stat().st_size, -1], (
        "a missing path must report the -1 sentinel, not the real stat of "
        "something else or a truncated result")
    assert mtimes_ns[0] > 0
    assert mtimes_ns[1] == 0, "mtime_ns is meaningless once size is -1"

    # The bytes form of the SAME two paths must agree with the str form.
    sizes_b, mtimes_b = _core.stat_many([str(present).encode(),
                                         str(missing).encode()])
    assert list(sizes_b) == list(sizes) and list(mtimes_b) == list(mtimes_ns)


def test_stat_many_counts_as_one_crossing_however_many_paths():
    """At this call specifically, `stat_many` is the compile cache's PER-HIT
    instrument, not a per-path one — the whole point of moving the loop into
    C++ is that Python pays ONE crossing for however many headers are in the
    closure.

    ``crossings()`` increments the SAME counter it reads (calibrated in
    ``test_launch_crossings.py::
    test_the_instrument_costs_exactly_one_crossing``), so the closing reading
    below is one crossing of its own and is subtracted by name rather than
    silently folded into the count being asserted."""
    before = _core.crossings()
    _core.stat_many(["/nonexistent/a", "/nonexistent/b", "/nonexistent/c"])
    after = _core.crossings()
    assert after - before - 1 == 1, (
        f"stat_many crossed the boundary {after - before - 1} time(s) for 3 "
        "paths, not 1")
