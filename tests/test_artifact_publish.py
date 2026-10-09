# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Publishing a deployment unit is CONTENT-ADDRESSED and IDEMPOTENT.

The subject is ``hawk.artifact.build_bundle``'s answer to "this unit already
exists": within one process, a repeat build into the SAME directory touches
the filesystem not at all; into a DIFFERENT directory it never recompiles
either, but COPIES the already-published files there (deliberately — a
caller's own directory must really hold the bytes it asked for), and across
processes it must recognise a target directory that already
holds the same bytes and leave it alone. Each of these is a skip of
RECOMPILATION, never a skip of a DECISION about what must land where — which
is what the rows below are really about, because a publisher that hands back a
stale unit, or names a directory it never wrote to, is the silent-wrong-answer
class, not a performance regression.

WHY THE WRITES ARE COUNTED AND NOT TIMED. "Wrote nothing" is a claim about
behaviour, and a wall clock cannot distinguish a publisher that got faster from
one that stopped writing. So the rows monkeypatch the two calls every write in
:mod:`hawk.artifact.bundle` goes through — ``Path.write_bytes`` and
``shutil.copyfile`` — plus ``Path.mkdir``, and assert the COUNT. A future
publish route that wrote through some third call would make
``test_a_repeat_publish_of_an_unchanged_unit_writes_nothing`` pass while writing;
:func:`test_the_write_counter_sees_a_real_publish` is the known answer that
forbids it, by asserting the same counter is NON-zero for a genuine first
publish.

This test previously failed. Every row here was observed failing before the
feature existed -- ``build_bundle`` re-published unconditionally, so the counting
rows read e.g.::

    AssertionError: the second build of an unchanged unit wrote 4 file(s) and
    made 1 directory: publishing is content-addressed, so a unit this
    process has already published must be returned, not written again
    assert 5 == 0

and the memo rows failed on ``unit_stats()['memo_hits'] == 0``. Then, with the
feature in, each SEMANTIC guard was planted and observed red in turn; the plants
are recorded on the rows they belong to and were all removed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import _deployable as D
import pytest
from _device_file import device_artifact

from hawk.artifact import STAMP_NAME, build_bundle, reset_unit_memo, unit_stats
from hawk.artifact.bundle import publish_log

#: Both targets, because a two-target unit is the shape a consumer deploys and
#: the shape whose closure is the union of two compiles.
BOTH = ("cuda", "host")


@pytest.fixture
def fresh():
    """A publisher with no memory of any unit, before AND after the row.

    After, too: the memo is process-wide, and a row that left its own unit in it
    would decide the next row's answer."""
    reset_unit_memo()
    yield
    reset_unit_memo()


class _Writes:
    """A counter over every route :mod:`hawk.artifact.bundle` writes through.

    ALL FIVE of them, and the completeness is the point rather than an excess of
    caution: the unit's files go out through ``Path.write_bytes`` and
    ``shutil.copyfile``, the STAMP through ``Path.write_text``, and the
    directory through ``Path.mkdir``. Every file is written to a private
    ``.tmp.*`` sibling and renamed into place (so a reader never sees a partial
    file), which makes ``os.replace`` the fifth route: the counter records the
    DESTINATION of each rename and ignores the temporary names. A counter that watched three of the four
    could report "wrote nothing" about a run that wrote — which is the one
    failure an instrument may not have."""

    def __init__(self, monkeypatch):
        self.files: list[str] = []
        self.dirs: list[str] = []
        real_bytes, real_text, real_copy, real_mkdir, real_replace = (
            Path.write_bytes, Path.write_text, shutil.copyfile, Path.mkdir,
            os.replace)

        def note(path):
            if not Path(path).name.startswith(".tmp."):
                self.files.append(str(path))

        def write_bytes(path, data):
            note(path)
            return real_bytes(path, data)

        def write_text(path, data, *a, **kw):
            note(path)
            return real_text(path, data, *a, **kw)

        def copyfile(src, dst, **kw):
            note(dst)
            return real_copy(src, dst, **kw)

        def replace(src, dst, *a, **kw):
            if Path(src).name.startswith(".tmp."):
                self.files.append(str(dst))
            return real_replace(src, dst, *a, **kw)

        def mkdir(path, *a, **kw):
            self.dirs.append(str(path))
            return real_mkdir(path, *a, **kw)

        monkeypatch.setattr(Path, "write_bytes", write_bytes)
        monkeypatch.setattr(Path, "write_text", write_text)
        monkeypatch.setattr(shutil, "copyfile", copyfile)
        monkeypatch.setattr(Path, "mkdir", mkdir)
        monkeypatch.setattr(os, "replace", replace)

    @property
    def total(self) -> int:
        return len(self.files) + len(self.dirs)

    def report(self) -> str:
        return (f"wrote {len(self.files)} file(s) {self.files} and made "
                f"{len(self.dirs)} director(y/ies) {self.dirs}")


def _module():
    """The publisher MODULE, :mod:`hawk.artifact.bundle`."""
    import importlib

    return importlib.import_module("hawk.artifact.bundle")


def _build(directory, cache, kernels=(D.axpb,), **kw):
    return build_bundle(list(kernels), Path(directory), targets=BOTH,
                        cache_dir=cache, **kw)


# --------------------------------------------------------------------------- #
# (a): a repeat build of the same definition writes nothing.
# --------------------------------------------------------------------------- #
def test_the_write_counter_sees_a_real_publish(tmp_path, cache_dir, monkeypatch,
                                               fresh):
    """The counter's KNOWN ANSWER. A first publish writes the whole unit —
    manifest, sidecar, both objects, both sources — so a counter that read zero
    here would be measuring nothing and every "wrote nothing" row below would be
    vacuous."""
    writes = _Writes(monkeypatch)
    bundle = _build(tmp_path / "first", cache_dir)
    assert writes.total >= 8, (
        "a first publish of a two-target unit makes the directory and writes the "
        "manifest, the sidecar, two objects, two sources and the stamp — seven "
        "files; the counter saw only " + writes.report())
    assert any(f.endswith(STAMP_NAME) for f in writes.files), (
        "the stamp is written through a route the counter cannot see, so a row "
        "asserting 'wrote nothing' could be wrong about the stamp and never know")
    assert unit_stats()["published"] == 1
    for path in bundle.files():
        assert path.is_file(), f"{path} was reported published but is not there"


def test_a_repeat_publish_of_an_unchanged_unit_writes_nothing(tmp_path, cache_dir,
                                                              monkeypatch, fresh):
    """The headline, counted.

    Needs the memo lookup in ``build_bundle``::

        AssertionError: the second build of an unchanged unit wrote 7 file(s)
        and made 1 director(y/ies): publishing is content-addressed
        assert 8 == 0
    """
    first = _build(tmp_path / "unit", cache_dir)
    writes = _Writes(monkeypatch)
    again = _build(tmp_path / "unit", cache_dir)
    assert writes.total == 0, (
        "the second build of an unchanged unit " + writes.report() + ": "
        "publishing is content-addressed, so a unit this process has "
        "already published must be returned, not written again")
    assert unit_stats()["memo_hits"] == 1
    assert again.directory == first.directory
    assert again.manifest == first.manifest


def test_a_repeat_build_into_a_DIFFERENT_directory_is_materialised_there(
        tmp_path, cache_dir, monkeypatch, fresh):
    """By design: a unit's identity is its BYTES, not its
    path — but every caller's OWN directory must really hold them: a second
    build of the same definition into a DIFFERENT directory still never
    recompiles (the memo answers that), but copies the unit's published
    files there, and the returned bundle names THAT directory, not the first
    one. A caller that trusted the directory it passed must find it full, not
    empty (this is what eagle's ``tests/test_active_set.py`` /
    ``test_until_done.py`` read straight off disk).

    No recompilation is asserted directly, by COUNTING calls to
    ``compile_source`` — not merely by trusting ``entry.hit`` — because that is
    the one fact a hit/miss flag could get wrong and a call count cannot."""
    B = _module()
    calls: list = []
    real_compile = B.compile_source

    def counting_compile(*a, **kw):
        calls.append(1)
        return real_compile(*a, **kw)

    monkeypatch.setattr(B, "compile_source", counting_compile)

    first = _build(tmp_path / "a", cache_dir)
    assert calls, (
        "the row's own first build must really compile, or the zero-calls "
        "assertion below would prove nothing")
    calls.clear()

    writes = _Writes(monkeypatch)
    second = _build(tmp_path / "b", cache_dir)

    assert not calls, (
        f"the second build into a different directory ran the compiler "
        f"{len(calls)} time(s): a memo hit must never recompile")
    assert unit_stats()["memo_hits"] == 1
    assert second.directory == tmp_path / "b" != first.directory

    # NON-VACUITY: the materialised copy really writes. A row that passed
    # because the second build quietly wrote nothing (e.g. "b" already held
    # the unit, or the copy was skipped) would look identical to one that
    # worked and prove nothing about the new contract.
    assert writes.total > 0, (
        "materialising the unit into a NEW directory wrote nothing: " +
        writes.report())

    for path in second.files():
        assert path.is_file(), (
            f"the materialised unit names {path}, which is not there")
    first_by_name = {p.name: p for p in first.files()}
    for path in second.files():
        twin = first_by_name[path.name]
        assert path.read_bytes() == twin.read_bytes(), (
            f"{path} is not byte-identical to the first build's {twin}")
    # The first directory is untouched by the second build.
    for path in first.files():
        assert path.is_file()


def test_a_memo_whose_directory_vanishes_mid_copy_falls_back_to_a_publish(
        tmp_path, cache_dir, monkeypatch, fresh):
    """The memo's own directory can disappear AFTER the memo is checked and
    BEFORE its files are copied (a temporary directory reclaimed meanwhile,
    as a caller that builds into short-lived directories can do). The build
    must then publish for real, not raise: the memo can no longer answer,
    and the caller's directory still ends up holding the unit."""
    B = _module()
    first = _build(tmp_path / "a", cache_dir)
    real_copy = B.shutil.copyfile
    vanished: list = []

    def copy_after_the_source_vanished(src, dst, *a, **kw):
        if not vanished:
            shutil.rmtree(first.directory)
            vanished.append(src)
        return real_copy(src, dst, *a, **kw)

    monkeypatch.setattr(B.shutil, "copyfile", copy_after_the_source_vanished)
    second = _build(tmp_path / "b", cache_dir)
    monkeypatch.undo()

    # NON-VACUITY: the copy path really ran and really lost its source.
    assert vanished and not first.directory.exists()
    assert unit_stats()["memo_hits"] == 0, (
        "a materialisation that failed must not count as a memo hit")
    assert second.directory == tmp_path / "b"
    assert second.manifest_path.is_file()
    for art in second.artifacts:
        assert Path(art.sidecar_path).is_file()


def test_a_served_unit_is_marked_a_HIT_on_every_entry(tmp_path, tmp_path_factory,
                                                      fresh):
    """A served unit ran no compiler, and its entries must say so.

    The compile-floor rows both assert ``not entry.hit`` to prove
    they are timing a compiler rather than a lookup. If the unit memo ever
    answered one of them, those rows have to go RED — an entry carrying the
    FIRST build's ``hit=False`` would let a row report a compile that did not
    happen, which is the exact shape of a benchmark measuring itself.

    THE CACHE DIRECTORY IS THIS ROW'S OWN, and that is what makes it mean
    something. Against the shared session cache the first build is itself a
    cache HIT, every entry reports ``hit=True`` for a reason that has nothing to
    do with the memo, and the row passes however ``_served`` behaves — it was
    written that way first and a planted ``_served`` = identity stayed GREEN. On
    a never-used cache directory the first build really compiles, so
    ``hit=False`` is the honest answer for it and ``hit=True`` on the second can
    only come from the memo.

    Needs ``_served`` to report an honest hit/miss::

        AssertionError: a unit served from the memo ran no compiler, so every
        entry must report a hit; got hit=False for 'cuda'
        assert False
    """
    cold = str(tmp_path_factory.mktemp("hawk_publish_served"))
    first = _build(tmp_path / "u", cold)
    assert not first.artifacts[0].entries["cuda"].hit, (
        "this row needs a genuine compile first, or the hit it asserts below "
        "could have come from the compile cache instead of the unit memo")
    served = _build(tmp_path / "u", cold)
    assert unit_stats()["memo_hits"] == 1
    for artifact in served.artifacts:
        for target, entry in artifact.entries.items():
            assert entry.hit, (
                "a unit served from the memo ran no compiler, so every entry "
                f"must report a hit; got hit=False for {target!r}")
            assert entry.seconds == 0.0


def test_a_changed_body_publishes_a_FRESH_unit(tmp_path, cache_dir, fresh):
    """The other direction: two different kernels are two different units, and
    the second one is really written even though the first is memoised."""
    first = _build(tmp_path / "axpb", cache_dir, kernels=(D.axpb,))
    second = _build(tmp_path / "vec3", cache_dir, kernels=(D.vec3_scale,))
    assert second.directory == tmp_path / "vec3" != first.directory
    assert unit_stats()["published"] == 2 and unit_stats()["memo_hits"] == 0
    assert device_artifact(tmp_path / "vec3", "vec3_scale").is_file()


def test_the_cache_directory_is_part_of_a_unit_s_identity(tmp_path, cache_dir,
                                                          tmp_path_factory, fresh):
    """COLD compiles are measured by pointing a build at a cache directory
    that has never been used, and both ASSERT the miss. A unit memo blind to the
    cache directory would answer the second sample of those rows from the first
    sample's unit — the row would then report a compile that never ran, and
    would look FASTER for it.

    Needs ``cache_dir`` to be a term of ``_unit_key``::

        AssertionError: a build against an unused cache directory must be a
        genuine publish, not a memo hit
        assert 1 == 0
    """
    other = tmp_path_factory.mktemp("hawk_publish_cold")
    _build(tmp_path / "warm", cache_dir)
    cold = _build(tmp_path / "cold", str(other))
    assert unit_stats()["memo_hits"] == 0, (
        "a build against an unused cache directory must be a genuine publish, "
        "not a memo hit")
    assert cold.directory == tmp_path / "cold"
    assert not cold.artifacts[0].entries["cuda"].hit


# --------------------------------------------------------------------------- #
# (b): what a memo hit still has to check.
# --------------------------------------------------------------------------- #
def test_a_corrupted_unit_is_detected_and_re_published(tmp_path, cache_dir, fresh):
    """The stamp says the unit is there; the bytes say otherwise.

    A memo that trusted its own record would serve a unit whose ``.ptx`` is no
    longer the one it compiled — and the loader would run whatever is in the
    file. So the hit path re-hashes the unit's OWN published files by content
    and, on a disagreement, publishes again and NAMES the digest it was talking
    about.

    Needs ``_first_moved`` to detect the corruption::

        AssertionError: a corrupted unit was served from the memo: the .ptx on
        disk is not the object the unit was published as
        assert 'corrupt' not in '...'
    """
    bundle = _build(tmp_path / "u", cache_dir)
    ptx = device_artifact(bundle.directory, "axpb")
    good = ptx.read_bytes()
    # Appended, never substring-replaced: the device artifact's own bytes are
    # PTX text OR a CUBIN (#31, hawk.compile.nvrtc.select_target picks
    # whichever loads on the driver present) depending on the box's own
    # compiler/driver versions, and a CUBIN (an ELF image) carries no "//"
    # for a text-only replace to find. Appending a byte is a corruption
    # under EITHER format.
    ptx.write_bytes(good + b"\x00hawk-corruption-probe")
    assert ptx.read_bytes() != good, "the corruption did not take"

    again = _build(tmp_path / "u", cache_dir)
    assert again.directory == tmp_path / "u"
    assert ptx.read_bytes() == good, (
        "a corrupted unit was served from the memo: the .ptx on disk is not the "
        "object the unit was published as")
    assert unit_stats()["published"] == 2
    assert any("invalidated" in line and str(ptx) in line for line in publish_log()), (
        "a re-publish must say WHICH file moved and under which unit digest; "
        f"the log says {publish_log()}")


def test_a_deleted_unit_is_not_served_from_the_memo(tmp_path, cache_dir, fresh):
    """A memo entry outlives a directory somebody deleted. The memo remembers a
    unit's identity, not its existence."""
    bundle = _build(tmp_path / "u", cache_dir)
    shutil.rmtree(bundle.directory)
    again = _build(tmp_path / "u", cache_dir)
    assert unit_stats()["memo_hits"] == 0
    for path in again.files():
        assert path.is_file()


def test_a_changed_reached_header_re_publishes_the_unit(tmp_path, cache_dir,
                                                        monkeypatch, fresh):
    """ survives the unit memo.

    A published unit is only correct while the headers its compile OPENED are
    the bytes it compiled against, so a memo hit re-verifies the whole recorded
    closure by content before it serves anything. The change is simulated at the
    closure level rather than by editing a real header: HAWK's aether tree is
    shared with every other row in the session and editing it would be a
    side-effect no row could undo safely, while the property under test —
    "a closure that no longer verifies is not served" — is the same either way.

    Needs the ``memo.closure.valid()`` guard in ``build_bundle``'s memo
    branch::

        AssertionError: a unit whose compile closure no longer verifies was
        served from the memo
        assert 1 == 0
    """
    _build(tmp_path / "u", cache_dir)
    monkeypatch.setattr(_module().ClosureWatch, "valid", lambda self: False)
    _build(tmp_path / "u", cache_dir)
    assert unit_stats()["memo_hits"] == 0, (
        "a unit whose compile closure no longer verifies was served from the memo")
    # Having refused the memo, the publisher re-derives the unit from the
    # content-closure cache and finds the SAME bytes (only the artifact-level guard was
    # disturbed, not a header), so the idempotent publish leaves the directory
    # alone. That is the two layers agreeing, not the check being skipped.
    assert unit_stats()["stamp_hits"] == 1
    assert any("invalidated" in line for line in publish_log()), publish_log()


def test_the_memo_re_verifies_the_closure_it_recorded(tmp_path, cache_dir, fresh):
    """And the closure it re-verifies is a REAL one — the union of both targets'
    compiler-reported dependency records, not an empty tuple that would make the
    row above pass by vacuity."""
    _build(tmp_path / "u", cache_dir)
    (unit,) = _module()._UNIT_MEMO.values()
    recorded = unit.closure.pairs()
    assert len(recorded) > 50, (
        f"the unit recorded {len(recorded)} closure entries; an nvcc TU over "
        "aether reaches hundreds, so this record cannot be the one the compiler "
        "reported")
    paths = {p for p, _d in recorded}
    assert all(isinstance(d, str) and len(d) == 64 for _p, d in recorded), (
        "every closure entry must carry a sha256 content digest")
    assert any(p.endswith(".cu") for p in paths) and any(p.endswith(".cpp")
                                                         for p in paths), (
        "a two-target unit's closure is the union of BOTH compiles")


# --------------------------------------------------------------------------- #
# (c): the cross-process half — the stamp.
# --------------------------------------------------------------------------- #
def test_the_unit_carries_a_digest_stamp_that_is_not_part_of_the_unit(
        tmp_path, cache_dir, fresh):
    """The stamp records the unit's identity and every member's digest — and is
    not itself a member: a file cannot hash itself, and the audits, the MPI
    bed's deployed digest and ``Bundle.files`` all enumerate the unit."""
    bundle = _build(tmp_path / "u", cache_dir)
    stamp = json.loads((bundle.directory / STAMP_NAME).read_text())
    assert len(stamp["unit"]) == 64
    names = {p.name for p in bundle.files()}
    assert set(stamp["files"]) == names, (
        f"the stamp records {sorted(stamp['files'])} but the unit publishes "
        f"{sorted(names)}")
    assert STAMP_NAME not in names and STAMP_NAME not in stamp["files"]


def test_a_publish_into_a_directory_that_already_holds_the_unit_writes_nothing(
        tmp_path, cache_dir, monkeypatch, fresh):
    """The ACROSS-PROCESS half, exercised the only way one process can: publish,
    forget everything the process knows, and publish again into the same place.

    Without the memo this is exactly what a second interpreter does, and it must
    leave the tree byte-identical without writing a byte.

    Needs the stamp comparison, or phase 3 always writes::

        AssertionError: re-publishing an identical unit into its own directory
        wrote 7 file(s) ...: the stamp makes a publish idempotent
        assert 7 == 0
    """
    _build(tmp_path / "u", cache_dir)
    reset_unit_memo()                       # what a fresh interpreter knows
    writes = _Writes(monkeypatch)
    again = _build(tmp_path / "u", cache_dir)
    assert writes.total == 0, (
        "re-publishing an identical unit into its own directory " + writes.report()
        + ": the stamp makes a publish idempotent")
    assert unit_stats()["stamp_hits"] == 1 and unit_stats()["published"] == 0
    assert again.directory == tmp_path / "u"


def test_a_stamped_directory_whose_bytes_moved_is_re_published(tmp_path, cache_dir,
                                                               fresh):
    """A stamp is a claim, not a proof. If the directory it stamps no longer
    holds those bytes, the writes happen — the stamp cannot certify a file it
    does not describe any more.

    Needs the ``_first_moved`` half of the stamp comparison, not the digest
    test alone::

        AssertionError: a stamped directory whose sidecar was overwritten was
        accepted as already published
    """
    bundle = _build(tmp_path / "u", cache_dir)
    reset_unit_memo()
    sidecar = bundle.directory / "axpb.json"
    good = sidecar.read_text()
    sidecar.write_text(good.replace('"kernel"', '"kernel_TAMPERED"', 1))
    _build(tmp_path / "u", cache_dir)
    assert sidecar.read_text() == good, (
        "a stamped directory whose sidecar was overwritten was accepted as "
        "already published")
    assert unit_stats()["published"] == 1


def test_the_stamp_survives_a_re_publish_and_still_describes_the_unit(
        tmp_path, cache_dir, fresh):
    """After a re-publish the stamp must describe what is NOW there, or the next
    process's idempotence check would be answering about the previous unit."""
    bundle = _build(tmp_path / "u", cache_dir)
    reset_unit_memo()
    device_artifact(bundle.directory, "axpb").write_bytes(b"// not the object\n")
    _build(tmp_path / "u", cache_dir)
    stamp = json.loads((bundle.directory / STAMP_NAME).read_text())
    from hawk.compile import digest_file

    for name, want in stamp["files"].items():
        assert digest_file(bundle.directory / name) == want, (
            f"the stamp says {name} hashes to {want}, and it does not")


def test_a_renderer_patched_INSIDE_a_process_is_the_documented_exposure(
        tmp_path, cache_dir, monkeypatch, fresh):
    """The unit key's STATED exposure, asserted so it is known and not silent —
    the same treatment gives the include-path shadowing false hit.

    ``_unit_key`` names the render's INPUTS (the walk digest, the mode, the
    kind, the targets, the arch, the defines, the ABI tag) and not its OUTPUT,
    which buys a warm build the whole cost of a render. What it cannot see in
    exchange is a change to the RENDERER made inside a live process: patch
    ``render_source`` between two builds of one kernel and the second is served
    the first's unit. The realistic way an emitter changes is an EDIT to a
    source file, which is a new interpreter and therefore a cold memo, and no
    row in this suite patches ``hawk.emit`` at runtime — so the exposure is
    accepted rather than paid for on every warm build.

    :func:`reset_unit_memo` is the door out of it, and the second half of this
    row is what proves the door works: with the memo forgotten, the patched
    renderer's text really is published.

    If this row ever goes RED because the second build MISSED, the trade has
    been changed and the docstring on ``_unit_key`` is now wrong — update it
    with the mechanism rather than deleting this row.
    """
    first = _build(tmp_path / "u", cache_dir)
    original = (first.directory / "axpb.cu").read_text()
    real = _module().render_source

    def patched(*a, **kw):
        source = real(*a, **kw)
        return type(source)(**{**vars(source),
                               "text": source.text + "\n// PATCHED RENDERER\n"})

    monkeypatch.setattr(_module(), "render_source", patched)
    served = _build(tmp_path / "u", cache_dir)
    assert unit_stats()["memo_hits"] == 1, (
        "the renderer patch became visible to the unit key; if that is now "
        "deliberate, _unit_key's stated exposure must be rewritten with the "
        "mechanism that closed it")
    assert (served.directory / "axpb.cu").read_text() == original

    reset_unit_memo()
    fresh_unit = _build(tmp_path / "u", cache_dir)
    assert "// PATCHED RENDERER" in (fresh_unit.directory / "axpb.cu").read_text(), (
        "reset_unit_memo() is the documented door back to the disk path, and it "
        "did not open")


# --------------------------------------------------------------------------- #
# The manifest/sidecar/file name say the REAL compiled format, not a
# static "ptx" guess -- the ``cuda`` leg can land PTX *or* CUBIN depending
# on the box's compiler/driver versions (hawk.compile.nvrtc.select_target),
# and the deployment unit's own text must describe whichever one actually
# landed. Forced deterministically the same way test_compile_device_target.py
# forces it (monkeypatching driver_cuda_version), one row per direction, so
# neither depends on which versions this particular box happens to have.
# --------------------------------------------------------------------------- #
def _skip_without_nvcc():
    from hawk.compile import toolchain as tc
    from hawk.ir import HawkError

    try:
        tc.device_compiler()
    except HawkError:
        pytest.skip("no nvcc on this box")
    if tc.device_compiler_kind() != "nvcc":
        pytest.skip("$HAWK_DEVICE_COMPILER forces nvrtc on this box")


def test_a_cuda_artifact_says_cubin_everywhere_when_the_compiler_outruns_the_driver(
        tmp_path, cache_dir, monkeypatch, fresh):
    """The manifest's ``plugins[].format``, the artifact's own file name, the
    published bytes, AND the sidecar's ``format`` field must all agree, and
    all four must say ``cubin`` the moment the compiler outruns the driver --
    not the old hard-coded ``ptx``/``.ptx`` that merely happened to still be
    openable (the driver auto-detects a module's container regardless of its
    file name or claimed format; the manifest is a separate, cross-repo
    CONTRACT that was lying about it, #33's own gap)."""
    _skip_without_nvcc()
    from hawk.compile import nvrtc as hnvrtc

    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (1, 0))
    # a real arch, so a box with no GPU (where the arch would be a virtual,
    # PTX-only one) still exercises the compiler-outruns-the-driver rule
    from _toolchain import real_arch
    monkeypatch.setenv("HAWK_CUDA_ARCH", real_arch())
    bundle = _build(tmp_path / "u", cache_dir)

    manifest = json.loads(bundle.manifest_path.read_text())
    entry = next(e for e in manifest["plugins"] if e["id"] == "axpb")
    assert entry["format"] == "cubin", entry
    assert entry["artifact"] == "axpb.cubin", entry

    artifact_path = bundle.directory / "axpb.cubin"
    assert artifact_path.is_file(), sorted(p.name for p in bundle.directory.iterdir())
    assert artifact_path.read_bytes()[:4] == b"\x7fELF", (
        "the manifest/file name say cubin but the bytes are not an ELF image")

    sidecar = json.loads((bundle.directory / "axpb.json").read_text())
    assert sidecar["format"] == "cubin"


def test_a_cuda_artifact_says_ptx_everywhere_when_the_driver_is_current(
        tmp_path, cache_dir, monkeypatch, fresh):
    """The other direction (and today's ordinary case, an artifact from a
    driver-new-enough setup): the manifest, file name, bytes and sidecar all
    stay ``ptx``, asserted rather than merely assumed."""
    _skip_without_nvcc()
    from hawk.compile import nvrtc as hnvrtc

    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (999, 9))
    bundle = _build(tmp_path / "u", cache_dir)

    manifest = json.loads(bundle.manifest_path.read_text())
    entry = next(e for e in manifest["plugins"] if e["id"] == "axpb")
    assert entry["format"] == "ptx", entry
    assert entry["artifact"] == "axpb.ptx", entry

    artifact_path = bundle.directory / "axpb.ptx"
    assert artifact_path.is_file(), sorted(p.name for p in bundle.directory.iterdir())
    assert artifact_path.read_bytes()[:4] != b"\x7fELF", (
        "the manifest/file name say ptx but the bytes are an ELF (cubin) image")

    sidecar = json.loads((bundle.directory / "axpb.json").read_text())
    assert sidecar["format"] == "ptx"


# --------------------------------------------------------------------------- #
# opt_level, threaded through build_bundle exactly like host_profile.
# --------------------------------------------------------------------------- #
def test_build_bundle_opt_level_never_shares_a_unit_with_a_different_level(
        tmp_path, fresh):
    """``build_bundle(..., opt_level=...)`` is a unit-key term (not only a
    compile-cache term): an O0 build and an O3 build of the same kernel must
    publish as two DIFFERENT units, never one served for the other.

    A PRIVATE cache directory, never the session-wide ``cache_dir`` fixture:
    that one is shared with every other row in this file (and the session),
    so an axpb/host compile at the default profile/level may already sit in
    it by the time this row runs, and this row's own "both really compiled"
    check must not depend on what ran before it."""
    cache = str(tmp_path / "cache")
    o3 = build_bundle([D.axpb], tmp_path / "o3", targets=("host",),
                      cache_dir=cache, opt_level="O3")
    o0 = build_bundle([D.axpb], tmp_path / "o0", targets=("host",),
                      cache_dir=cache, opt_level="O0")
    assert o3.digest != o0.digest
    entry_o3 = o3.artifacts[0].entries["host"]
    entry_o0 = o0.artifacts[0].entries["host"]
    assert entry_o3.key != entry_o0.key
    assert not entry_o3.hit and not entry_o0.hit, "two distinct levels: both compile"
    # the default (no opt_level given) is O3, so it must hit the O3 unit's
    # own per-process memo, not recompile.
    default = build_bundle([D.axpb], tmp_path / "default", targets=("host",),
                           cache_dir=cache)
    assert default.digest == o3.digest
    assert default.artifacts[0].entries["host"].hit
