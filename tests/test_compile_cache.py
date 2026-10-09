# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The content-closure compile cache, case by case.

Every invalidation case is asserted as a HIT or a MISS by reading the
cache's own counters — never inferred from a wall time — plus the
include-path SHADOWING case, asserted as the documented false HIT it is
(a stated, ACCEPTED exposure) rather than left silent.

The subject is a tiny hand-written TU over a private header tree, not an
emitted kernel: the property under test is the cache's, and a two-line header
this row OWNS is the only way to change a reached header's bytes, touch it
without changing them, and add a shadowing file earlier on the search path.

This test previously failed, via two plants, one per direction:

* ``digest_file`` reverted to hashing ``(path, size, mtime_ns)`` instead of the
  bytes -- ``AssertionError: content, never mtime, decides: touching a
  reached header must not invalidate / assert 'miss' == 'hit'``;
* the validity re-check emptied (``for path, want in ``) --
  ``AssertionError: assert 'hit' == 'miss'`` on the reached-header CONTENT
  change.

Both plants were then removed.

THE PER-PROCESS MEMOS (rows at the end of this file). Two answers were being
recomputed on every lookup even when the key had already been answered in
this process: the compiler's IDENTITY (a ``<compiler> --version`` SUBPROCESS,
~3.9 ms) and the validity RECORD (read and JSON-parsed twice, once by
``check`` and once by ``record``). Together they were ~6 ms of a ~7 ms warm
build, which is what made re-authoring one kernel in one process cost more
than a JIT toolchain's in-memory module hit. Both are now memoised per
process, and neither memo may change an ANSWER:

  * the compiler memo is keyed on the binary's ``(path, size, mtime_ns)`` --
    the same shape, and the same accepted exposure, as :func:`digest_file`'s
    content memo;
  * the artifact memo is keyed on the LOOKUP KEY and stores the record's
    ``(path, digest)`` pairs, which are re-verified BY CONTENT on every hit. It
    replaces the two JSON reads, never the check they carry -- which is why
    ``test_a_content_change_of_a_reached_header_is_a_miss`` above now runs
    THROUGH the memo (its two compiles share a key) and must still say "miss".

Earlier failures for those: plants in ``hawk/compile/drivers.py`` and
``hawk/compile/toolchain.py``, both removed -- pasted at the rows below.

THE WARM CHECK ITSELF (the last two rows). With both memos in, what a warm
lookup costs IS the closure re-verification: 317 recorded headers on a real
fixture's shape. It is now one function, :func:`hawk.compile.closure_unchanged`,
and the rows below pin the two properties that make it both cheap and honest --
it stats each recorded header AT MOST ONCE and reads the CONTENT of none of
them, and it stops at the first entry that disagrees.

WHAT IT IS NOT, and why. An O(1) warm check, batching the stats per
directory, was tried and measured on this box as a LOSS; the numbers are in
``closure_unchanged``'s own docstring: aggregating each directory's entries
through ``os.scandir`` costs 6.213 ms against 0.62 ms for the per-file stats,
because the 33 directories the closure reaches hold 1500+ files between them and
``DirEntry.stat()`` is a syscall per entry like any other. The cost is not the
filesystem either -- 317 stats of THE SAME file cost 0.687 ms against 0.593 ms
for 317 distinct ones -- so what is being paid for is the interpreter frame, and
one stat per reached header is the floor for a check that reads content rather
than trusting a change notification. The rows below therefore assert the floor
is REACHED, not that it was beaten.

The floor above is a floor for ``closure_unchanged``'s Python loop; it is not
a floor for the CROSSING COUNT. ``ClosureWatch`` (the object the
artifact-level unit memo actually watches, ``hawk/artifact/
build.py``'s ``_closure_watch``) now does its whole warm stat pass through ONE
``hawk._core.stat_many`` call instead of one Python ``os.stat`` per row --
``closure_unchanged`` itself is UNCHANGED and stays the Python loop, because it
is a different call site (``drivers.py``'s per-TU ``_memo_still_valid``) with
its own row above pinning its shape. The two rows at the end of this file are
``ClosureWatch``'s: one pins the crossing count ("no per-path Python
work" made a number, not a claim), one is a microbench comparing the same
Python loop against ``stat_many`` on a real, hundreds-of-headers
closure -- not a gate, printed, and only the DIRECTION is asserted.

This test previously failed via one plant, then removed. ``ClosureWatch.valid``
reverted to its pre-fix per-row ``os.stat`` loop (calling no ``_core``
function at all)::

    AssertionError: the warm check crossed the boundary 0 time(s), not 1 --
    ClosureWatch.valid() must do the whole stat pass in ONE _core.stat_many
    call
    assert 0 == 1
"""

from __future__ import annotations

import json

import pytest

from hawk.compile import (
    CompileOptions,
    cache_stats,
    compile_source,
    lookup_key,
    reset_cache_stats,
)

SOURCE = """#include "reached.h"

extern "C" int hawk_cache_probe(void) { return HAWK_PROBE_VALUE; }
"""


@pytest.fixture
def tree(tmp_path):
    """``(early, late, cache)`` include dirs; ``reached.h`` lives in ``late``,
    an UNREACHED header beside it, and ``early`` starts empty so the shadowing
    case can add a file to it WITHOUT changing the flag string."""
    early, late, cache = (tmp_path / "early", tmp_path / "late", tmp_path / "cache")
    for d in (early, late, cache):
        d.mkdir()
    (late / "reached.h").write_text("#define HAWK_PROBE_VALUE 1\n")
    (late / "unreached.h").write_text("#define HAWK_UNREACHED 1\n")
    return early, late, cache


def _opts(tree, *, defines=(), extra=()):
    early, late, cache = tree
    return CompileOptions(backend="host", cache_dir=str(cache), defines=tuple(defines),
                          extra=("-I", str(early), "-I", str(late), *extra))


def _compile(tree, **kw):
    reset_cache_stats()
    result = compile_source(SOURCE, "cache_probe", _opts(tree, **kw))
    stats = cache_stats()
    return result, ("hit" if stats["hits"] else "miss")


def test_first_compile_is_a_miss_and_the_second_is_a_hit(tree):
    result, answer = _compile(tree)
    assert answer == "miss", "a first compile of a key is a MISS by definition"
    assert result.artifact.is_file()
    again, answer = _compile(tree)
    assert answer == "hit"
    assert again.key == result.key and again.artifact == result.artifact


def test_a_touch_of_a_reached_header_is_a_hit(tree):
    _early, late, _cache = tree
    _compile(tree)
    header = late / "reached.h"
    text = header.read_text()
    header.write_text(text)                       # same BYTES, new mtime
    assert header.stat().st_mtime_ns
    _result, answer = _compile(tree)
    assert answer == "hit", (
        "content, never mtime, decides: touching a reached header must "
        "not invalidate"
    )


@pytest.fixture
def sealed_path():
    """Only meaningful on the sealed-payload path (aether-dsc, no
    ``$HAWK_AETHER_INCLUDE``)."""
    from hawk.compile.drivers import _host_uses_sealed_payload

    if not _host_uses_sealed_payload():
        pytest.skip("source-header path ($HAWK_AETHER_INCLUDE is set): this row "
                    "covers the sealed-payload path")


@pytest.mark.usefixtures("sealed_path")
def test_sealed_path_a_changed_user_header_rebuilds_and_an_unchanged_one_hits(
        tree, monkeypatch):
    """On the sealed path the key carries the payload digest for the payload's
    own headers, but a USER header reached through ``-I`` must still decide
    validity by content, exactly as on the source-header path. Counts real
    compiler invocations: unchanged and touched-only -> no rebuild, changed ->
    exactly one."""
    import os
    import subprocess

    real, box = subprocess.run, {"n": 0}

    def counting(argv, *a, **kw):
        if "-shared" in argv:
            box["n"] += 1
        return real(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", counting)
    _early, late, _cache = tree
    header = late / "reached.h"

    _r, first = _compile(tree)
    assert (first, box["n"]) == ("miss", 1)
    _r, again = _compile(tree)
    assert (again, box["n"]) == ("hit", 1)
    st = header.stat()
    os.utime(header, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    _r, touched = _compile(tree)
    assert (touched, box["n"]) == ("hit", 1)        # same bytes: no spurious rebuild

    header.write_text("#define HAWK_PROBE_VALUE 2\n")
    _r, changed = _compile(tree)
    assert (changed, box["n"]) == ("miss", 2), "a changed user header was served stale"
    _r, settled = _compile(tree)
    assert (settled, box["n"]) == ("hit", 2)


def test_a_content_change_of_a_reached_header_is_a_miss(tree):
    _early, late, _cache = tree
    _compile(tree)
    (late / "reached.h").write_text("#define HAWK_PROBE_VALUE 2\n")
    _result, answer = _compile(tree)
    assert answer == "miss"


def test_a_content_change_of_an_UNREACHED_header_is_a_hit(tree):
    _early, late, _cache = tree
    _compile(tree)
    (late / "unreached.h").write_text("#define HAWK_UNREACHED 999\n")
    _result, answer = _compile(tree)
    assert answer == "hit", (
        "the closure is what the COMPILER opened; a header no kernel "
        "reaches must not invalidate — that is the whole cost a "
        "full-tree walk paid"
    )


def test_a_flag_change_is_a_key_change_and_therefore_a_miss(tree):
    first, _ = _compile(tree)
    second, answer = _compile(tree, defines=("HAWK_ROW_18=1",))
    assert answer == "miss" and second.key != first.key


def test_an_include_root_change_is_a_key_change(tree, tmp_path):
    """Both header roots ride inside the flag string, so MOVING either is
    a key change rather than a validity question."""
    first, _ = _compile(tree)
    other = tmp_path / "other"
    other.mkdir()
    second, answer = _compile(tree, extra=("-I", str(other)))
    assert answer == "miss" and second.key != first.key


def test_a_compiler_change_is_a_key_change(tree, monkeypatch):
    from hawk import compile as hcompile
    from hawk.compile import drivers

    first, _ = _compile(tree)
    monkeypatch.setattr(drivers.tc, "compiler_identity",
                        lambda c: hcompile.compiler_identity(c) + "||pretend-15.0")
    _second, answer = _compile(tree)
    assert answer == "miss"


# --------------------------------------------------------------------------- #
# This was found failing: ``compiler_identity`` keyed on ONLY the first
# ``--version`` line, which for nvcc is version-INVARIANT ("nvcc: NVIDIA (R)
# Cuda compiler driver" — the actual release lives on a LATER line, "Cuda
# compilation tools, release 12.6, V12.6.77"). Two nvcc installs that differ
# ONLY past line one — exactly what two real nvcc releases look like — gave
# the SAME identity, so the cache served one version's PTX to the other's
# run. Fixed by folding a digest of the WHOLE ``--version`` text into the
# identity (``hawk/compile/toolchain.py``'s ``compiler_identity``), not by
# special-casing nvcc's "release" line — this row's two fakes are deliberately
# NOT called nvcc, to pin the general mechanism rather than the one compiler
# that surfaced it.
# --------------------------------------------------------------------------- #
def _fake_compiler(tmp_path, name: str, release_line: str):
    """A tiny ``#!/bin/sh`` script answering ``--version`` with a FIRST line
    every fake shares and a LATER line (``release_line``) that distinguishes
    them — the exact shape ``nvcc --version`` has."""
    script = tmp_path / name
    script.write_text(
        "#!/bin/sh\n"
        'echo "fake: NOT a real compiler driver"\n'
        'echo "Copyright (c) 2026 nobody"\n'
        f'echo "{release_line}"\n'
    )
    script.chmod(0o755)
    return str(script)


def test_two_compilers_differing_only_past_the_first_version_line_get_different_identities(
        tmp_path):
    """The direct row: same first ``--version`` line, different RELEASE
    line — ``compiler_identity`` must still tell them apart."""
    from hawk.compile import compiler_identity

    a = _fake_compiler(tmp_path, "fake_cc_a", "release 12.6, V12.6.77")
    b = _fake_compiler(tmp_path, "fake_cc_b", "release 12.9, V12.9.10")
    ident_a, ident_b = compiler_identity(a), compiler_identity(b)
    assert ident_a != ident_b, (
        f"two compilers differing only past line one of --version must get "
        f"DIFFERENT identities, got the same: {ident_a!r}"
    )


def test_two_such_compilers_give_different_cache_lookup_keys(tree, tmp_path):
    """End to end: the SAME source/backend/mode/flags, through two fake
    compilers differing only past the first ``--version`` line, must land on
    two DIFFERENT cache keys — never share a slot."""
    from hawk.compile import compiler_identity, host_flags, lookup_key

    a = _fake_compiler(tmp_path, "fake_cc_a", "release 12.6, V12.6.77")
    b = _fake_compiler(tmp_path, "fake_cc_b", "release 12.9, V12.9.10")
    flags = host_flags(extra=_opts(tree).extra)
    key_a = lookup_key(SOURCE, "host", "float64", compiler_identity(a), flags)
    key_b = lookup_key(SOURCE, "host", "float64", compiler_identity(b), flags)
    assert key_a != key_b


def test_include_path_shadowing_is_the_DOCUMENTED_false_hit(tree):
    """The stated, ACCEPTED exposure. A ``-MD`` closure records the files
    that WERE opened; a NEW file appearing EARLIER on the search path shadows a
    recorded header, so the compile would change while every recorded file's
    content is unchanged — a false HIT. It is accepted because the realistic
    trigger (an include-path change) is a FLAG change and therefore already a
    key change; this row exists so the case is a known exposure, not a silent
    one."""
    early, _late, _cache = tree
    first, _ = _compile(tree)
    (early / "reached.h").write_text("#define HAWK_PROBE_VALUE 424242\n")
    second, answer = _compile(tree)
    assert answer == "hit", (
        "documents this as a false HIT; if it has become a miss the "
        "documentation is now wrong and must be updated with the mechanism"
    )
    assert second.key == first.key


def test_the_key_and_the_record_never_mention_mtime(tree):
    from hawk.compile import Cache

    result, _ = _compile(tree)
    record = Cache(_opts(tree).cache_dir).record(result.key)
    blob = json.dumps(record)
    assert "mtime" not in blob
    assert record["closure"], "the validity record must carry the closure"
    assert all(len(pair) == 2 and pair[1] for pair in record["closure"])


def test_the_lookup_key_is_computable_before_any_compile(tree):
    """ whole point: every term of the key is available BEFORE the first
    compile, so a first compile stores its artifact under the same key any later
    lookup computes. The record carries the key's terms: the compile flags
    plus the host profile and, for ``native``, the target CPU's identity."""
    result, _ = _compile(tree)
    record = __import__("hawk.compile", fromlist=["Cache"]).Cache(
        _opts(tree).cache_dir).record(result.key)
    assert record["key_terms"][:len(record["flags"])] == record["flags"]
    assert lookup_key(SOURCE, "host", "float64", record["compiler"],
                      record["key_terms"]) == result.key


# --------------------------------------------------------------------------- #
# The per-process memos: they may skip WORK, never a decision.
# --------------------------------------------------------------------------- #
def test_a_repeat_lookup_in_one_process_does_not_re_read_the_record(tree,
                                                                    monkeypatch):
    """The artifact memo's whole job, counted rather than timed.

    A second lookup at a key this process has already answered must not reach
    ``Cache.check`` at all — that call is the record read the memo exists to
    replace. Counted, because a memo that merely got faster and still read the
    record would be indistinguishable from this one on a wall clock.

    Needs the memo lookup in ``compile_source``::

        AssertionError: the second lookup at an already-answered key read the
        validity record again (Cache.check called 1 time(s))
    """
    from hawk.compile import drivers

    _compile(tree)                                   # a first, cold compile
    calls = []
    real = drivers.Cache.check
    monkeypatch.setattr(drivers.Cache, "check",
                        lambda self, key: calls.append(key) or real(self, key))
    _result, answer = _compile(tree)
    assert answer == "hit"
    assert not calls, (
        "the second lookup at an already-answered key read the validity record "
        f"again (Cache.check called {len(calls)} time(s))")


def test_the_memo_cannot_serve_a_stale_answer(tree):
    """A changed reached header MISSES through the memo, not only on disk.

    The two compiles here share a key, so the second one is answered by the
    memo — and the memo re-verifies the recorded closure BY CONTENT, which is
    what makes this the same answer the disk record would give.

    Needs ``_memo_still_valid`` to actually check::

        AssertionError: a memoised answer survived a content change in a reached
        header — the memo replaced the validity check instead of the record read
        assert 'hit' == 'miss'

    The same plant also turned ``test_a_content_change_of_a_reached_header_is_a
    _miss`` and ``test_a_deleted_artifact_is_not_served_from_the_memo`` red,
    which is the point: there is no way to make the memo stale that only this
    row can see.
    """
    _early, late, _cache = tree
    _compile(tree)
    (late / "reached.h").write_text("#define HAWK_PROBE_VALUE 7\n")
    _result, answer = _compile(tree)
    assert answer == "miss", (
        "a memoised answer survived a content change in a reached header — the "
        "memo replaced the validity check instead of the record read")


def test_a_deleted_artifact_is_not_served_from_the_memo(tree):
    """A memo entry outlives a cache directory somebody deleted.

    The memo remembers a key's answer, not the bytes; if the artifact it names
    is gone the answer is worthless and the compile has to happen again."""
    import shutil

    result, _ = _compile(tree)
    shutil.rmtree(result.artifact.parent)
    again, answer = _compile(tree)
    assert answer == "miss" and again.artifact.is_file()


def test_the_compiler_is_identified_once_per_process(tree, monkeypatch):
    """``<compiler> --version`` is a SUBPROCESS, and the answer cannot change
    under a binary this process has already stat'd at the same size and mtime.

    Needs the compiler-identity memo::

        AssertionError: the compiler was re-identified by subprocess 3 time(s);
        its identity is a term of every lookup key, so that runs on every
        compile — cached or not
        assert 3 == 1
    """
    import subprocess as sp

    from hawk.compile import toolchain
    from hawk.compile.toolchain import reset_identity_memo

    reset_identity_memo()
    runs = []
    real = sp.run
    monkeypatch.setattr(toolchain.subprocess, "run",
                        lambda argv, **kw: runs.append(tuple(argv)) or real(argv, **kw))
    compiler = toolchain.host_compiler()
    for _ in range(3):
        toolchain.compiler_identity(compiler)
    versions = [a for a in runs if a[1:] == ("--version",)]
    assert len(versions) == 1, (
        f"the compiler was re-identified by subprocess {len(versions)} time(s); "
        "its identity is a term of every lookup key, so that runs on every "
        "compile — cached or not")


# --------------------------------------------------------------------------- #
# The warm check itself: one stat per reached header, no content read.
# --------------------------------------------------------------------------- #
def test_a_warm_hit_stats_each_reached_header_once_and_reads_none(tree, monkeypatch):
    """What a warm lookup is allowed to cost, counted rather than timed.

    A hit re-verifies the recorded closure BY CONTENT, and the
    ``(path, size, mtime_ns) -> digest`` memo is what makes "by content" cost a
    stat instead of a read. So a warm hit must: stat every recorded header (that
    is the check), stat none of them TWICE (that is the memo working), and open
    none of them at all (that is the memo being consulted rather than bypassed).

    Needs the ``memo.get`` lookup in ``closure_unchanged``, or every entry
    falls through to ``digest_file``::

        AssertionError: a warm hit stat'd 3 header(s) more than once
        (['.../stdc-predef.h', '.../cache_probe.cpp', '.../late/reached.h']);
        the check is the floor already and re-stat'ing doubles it

    The DOUBLE STAT is what the plant is caught by rather than the re-read, and
    the difference is worth keeping: ``digest_file`` stats again before it reads,
    so a bypassed memo shows up first as twice the syscalls. The content-read
    assertion fires on the same plant and is kept because it is the property
    that is actually load-bearing.
    """
    import os

    from hawk.compile import cache as C

    _compile(tree)                                   # a first, cold compile
    recorded = [p for p, _d in C.artifact_memo_get(_compile(tree)[0].key)]
    assert len(recorded) >= 3, (
        f"only {len(recorded)} closure entries to check; this row's TU must "
        "reach its own header, the source and at least one system header, or "
        "'stat'd once, read never' is a claim about nothing")

    stats: dict = {}
    reads: list = []
    real_stat, real_hash = os.stat, C._hash_bytes
    interesting = set(recorded)

    def counting_stat(path, *a, **kw):
        if path in interesting:
            stats[path] = stats.get(path, 0) + 1
        return real_stat(path, *a, **kw)

    def counting_hash(path):
        if path in interesting:
            reads.append(path)
        return real_hash(path)

    monkeypatch.setattr(os, "stat", counting_stat)
    monkeypatch.setattr(C, "_hash_bytes", counting_hash)
    _result, answer = _compile(tree)
    assert answer == "hit"

    assert not reads, (
        f"a warm hit read the CONTENT of {len(reads)} recorded header(s) "
        f"{reads[:3]}; the memo exists so a hit costs a stat, not a read")
    assert set(stats) == interesting, (
        "a warm hit must stat EVERY recorded header — the ones it skipped are "
        f"{sorted(interesting - set(stats))}")
    twice = sorted(p for p, n in stats.items() if n > 1)
    assert not twice, (
        f"a warm hit stat'd {len(twice)} header(s) more than once ({twice[:3]}); "
        "the check is the floor already and re-stat'ing doubles it")


def test_the_warm_closure_check_short_circuits_on_the_first_change(tree):
    """A closure that has already disagreed does not need to be finished.

    The verdict is the same either way, so this is only about cost — but it is
    the difference between a changed header costing one stat and costing 317,
    and a check that reads on regardless is the one shape that looks identical
    on a wall clock.

    Needs ``closure_unchanged`` to return as soon as it disagrees::

        AssertionError: the check stat'd 3 entries after the first disagreement;
        the verdict was settled at entry 0
    """
    import os

    from hawk.compile.cache import closure_unchanged

    result, _ = _compile(tree)
    from hawk.compile import cache as C

    recorded = list(C.artifact_memo_get(result.key))
    assert len(recorded) > 2
    spoiled = [(recorded[0][0], "0" * 64)] + recorded[1:]

    seen: list = []
    real_stat = os.stat

    def counting_stat(path, *a, **kw):
        seen.append(path)
        return real_stat(path, *a, **kw)

    os.stat = counting_stat
    try:
        assert closure_unchanged(spoiled) is False
    finally:
        os.stat = real_stat
    after = [p for p in seen if p in {q for q, _d in recorded[1:]}]
    assert not after, (
        f"the check stat'd {len(after)} entries after the first disagreement; "
        "the verdict was settled at entry 0")


def test_a_closure_watch_gives_the_same_verdict_as_the_content_rule(tree):
    """``ClosureWatch``'s REAL check (``._check()``) denormalises the memo;
    it must not denormalise the RULE: a reached header TOUCHED (same bytes,
    new mtime) is still valid.

    This row calls ``._check()`` rather than the public ``.valid()``: ``.valid()`` now trusts a PER-PROCESS verdict memo after its
    first call for a given closure content, so a second ``.valid()`` on the
    same watch would not re-run the rule this row is pinning at all — see
    ``test_a_content_changed_header_stays_a_HIT_until_reset_then_a_MISS``
    below for THAT behaviour, which is actual subject.

    Needs the content fall-through, or the ``(size, mtime_ns)`` comparison
    decides alone::

        AssertionError: content, never mtime: a touched header must leave
        the watch valid
        assert False
    """
    _early, late, _cache = tree
    from hawk.compile import cache as C
    from hawk.compile.cache import ClosureWatch

    result, _ = _compile(tree)
    watch = ClosureWatch(C.artifact_memo_get(result.key))
    assert len(watch) >= 3
    assert watch._check()

    header = late / "reached.h"
    text = header.read_text()
    header.write_text(text)                       # same BYTES, new mtime
    assert watch._check(), (
        "content, never mtime: a touched header must leave the watch valid")

    header.write_text("#define HAWK_PROBE_VALUE 11\n")
    assert not watch._check(), (
        "a reached header whose CONTENT changed must invalidate the watch")


# --------------------------------------------------------------------------- #
# The PUBLIC `.valid()` trusts a per-process verdict memo.
# --------------------------------------------------------------------------- #
def test_a_content_changed_header_stays_a_HIT_until_reset_then_a_MISS(tree):
    """ actual subject, re-owned from the earlier shape of the row above
    (which pinned "every ``.valid()`` call re-verifies by content" — no longer
    true of the PUBLIC method now that a warm hit's 0.6 ms/hit
    ``ClosureWatch.valid()`` cost is a per-process verdict memo, keyed on the
    closure's own content and trusted until :func:`reset_artifact_memo`).

    RED against pre-L2 ``valid()`` (which always re-checks): the SECOND
    assertion below fails, because today's code answers the post-edit call
    honestly (``False``) instead of trusting the first call's ``True``::

        AssertionError: the FIRST .valid() in this process already answered
        this closure's content; a later call on the SAME content is trusted
        until reset_artifact_memo(), not re-stat'd
        assert False
    """
    _early, late, _cache = tree
    from hawk.compile import _reset_artifact_memo as reset_artifact_memo
    from hawk.compile import cache as C
    from hawk.compile.cache import ClosureWatch

    result, _ = _compile(tree)
    watch = ClosureWatch(C.artifact_memo_get(result.key))
    assert watch.valid()

    (late / "reached.h").write_text("#define HAWK_PROBE_VALUE 11\n")
    assert watch.valid(), (
        "the FIRST .valid() in this process already answered this closure's "
        "content; a later call on the SAME content is trusted until "
        "reset_artifact_memo(), not re-stat'd (L2)")

    reset_artifact_memo()
    assert not watch.valid(), (
        "reset_artifact_memo() is the door back to an honest content check; "
        "a reached header whose CONTENT changed must invalidate the watch "
        "once that door has been used")


def test_a_fresh_process_still_misses_on_a_changed_reached_header(tree):
    """ ACROSS processes is exactly what it was — NOT a
    ``ClosureWatch``-level row, and deliberately so: a ``ClosureWatch`` never
    crosses a process boundary in the first place (each process builds its
    own, always baselined at its OWN construction moment, right after ITS OWN
    real compile — verdict memo only ever shortcuts a LATER call on an
    instance THIS process already built). The only place a second process
    COULD inherit a stale belief is the ON-DISK validity record
    ``compile_source`` reads through ``Cache.check``/``closure_unchanged``,
    and this row proves that boundary is still honest: a header edited
    between two processes' compiles is a
    real MISS (a compiler subprocess actually runs) in the second one, not a
    hit inherited from the first process's record."""
    import json
    import subprocess
    import sys

    early, late, _cache = tree
    _compile(tree)                                   # process 1 records content #1
    (late / "reached.h").write_text("#define HAWK_PROBE_VALUE 11\n")  # content #2

    probe = (
        "import json\n"
        "from hawk.compile import CompileOptions, compile_source, cache_stats\n"
        f"opts = CompileOptions(backend='host', cache_dir={str(_cache)!r}, "
        f"extra=('-I', {str(early)!r}, '-I', {str(late)!r}))\n"
        f"compile_source({SOURCE!r}, 'cache_probe', opts)\n"
        "print(json.dumps(cache_stats()['hits'] > 0))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, check=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) is False, (
        "a fresh process compiling against an edited header must be a real "
        "MISS, not a hit inherited from the first process's stale record")


def test_a_closure_watch_over_a_vanished_file_is_not_valid(tree):
    """The "a closure file is missing or unreadable => miss", through the
    watch: the record cannot certify a file that is not there."""
    _early, late, _cache = tree
    from hawk.compile import cache as C
    from hawk.compile.cache import ClosureWatch

    result, _ = _compile(tree)
    watch = ClosureWatch(C.artifact_memo_get(result.key))
    (late / "reached.h").unlink()
    assert not watch.valid()


# --------------------------------------------------------------------------- #
# — ClosureWatch's warm stat pass through hawk._core.stat_many.
# --------------------------------------------------------------------------- #
def test_the_warm_check_makes_exactly_one_core_crossing_for_the_stat_pass(tree):
    """``ClosureWatch.valid()``'s stat pass is ONE ``_core`` crossing,
    however many rows the recorded closure has -- not one Python ``os.stat``
    per row, which is the shape it replaced (see the module docstring's RED
    history: the earlier shape called no ``_core`` function at all, so the
    plant read 0, not some larger number)."""
    from hawk import _core
    from hawk.compile import cache as C
    from hawk.compile.cache import ClosureWatch

    result, _ = _compile(tree)
    watch = ClosureWatch(C.artifact_memo_get(result.key))
    assert len(watch) >= 3

    # crossings() increments the SAME counter it reads (calibrated in
    # test_launch_crossings.py::
    # test_the_instrument_costs_exactly_one_crossing), so the CLOSING reading
    # below is one crossing of its own and is subtracted by name.
    before = _core.crossings()
    assert watch.valid()
    after = _core.crossings()
    crossed = after - before - 1
    assert crossed == 1, (
        f"the warm check crossed the boundary {crossed} time(s), not "
        "1 -- ClosureWatch.valid() must do the whole stat pass in ONE "
        "_core.stat_many call (HW6e)")


def test_stat_many_does_not_slow_down_the_real_closure(built):
    """A microbench, not a gate: ``_core.stat_many`` must not be
    SLOWER than the Python ``os.stat`` loop it replaced, over the REAL
    closure one of the session fixture's cuda compiles recorded -- hundreds
    of headers, not the handful the toy ``tree`` fixture above reaches. Both
    numbers are printed; only the DIRECTION is asserted, because a regression
    here means the binding crossing is not paying for itself and should
    be reverted, not tuned further.

    Both arms call ``watch._check()`` — the CROSSING itself, bypassing 
    per-process verdict memo (:meth:`ClosureWatch.valid`) on purpose: this
    row measures the stat-pass MECHANISM (Python loop vs. one ``stat_many``
    crossing), and the memo would flatten a warm ``.valid()`` to a dict
    lookup after the first call, measuring nothing for 49 of 50 reps."""
    import os
    import statistics
    import time

    from hawk.compile.cache import ClosureWatch, validity_digest

    entry = built["axpb"].artifacts[0].entries["cuda"]
    pairs = validity_digest(entry.closure)
    assert len(pairs) > 100, (
        f"only {len(pairs)} closure entries; this row's claim is about the "
        "real, hundreds-of-headers A1 stage-4 shape, not a toy TU")

    watch = ClosureWatch(pairs)
    assert watch._check(), "the fixture's own compile must still verify"

    def python_loop():
        # The EXACT shape ClosureWatch._check() ran before : one os.stat
        # per row, content read only on a size/mtime disagreement (none here).
        stat = os.stat
        for row in watch._rows:
            st = stat(row[0])
            assert st.st_size == row[1] and st.st_mtime_ns == row[2]

    def core_call():
        assert watch._check()

    def _median_seconds(call, reps=50):
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            call()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    # Interleaved-ish by running one warmup of each before either is timed, so
    # neither arm pays a first-touch cost (page cache, branch prediction) the
    # other one does not.
    python_loop()
    core_call()
    python_s = _median_seconds(python_loop)
    core_s = _median_seconds(core_call)
    print(f"\nstat_many microbench ({len(pairs)} headers): python loop "
         f"{python_s * 1000:.3f} ms/check, stat_many {core_s * 1000:.3f} "
         "ms/check")
    assert core_s <= python_s, (
        f"stat_many ({core_s * 1000:.3f} ms) is SLOWER than the Python "
        f"os.stat loop it replaced ({python_s * 1000:.3f} ms) over "
        f"{len(pairs)} headers -- the binding crossing is not paying for "
        "itself; HW6e should be reverted, not tuned further")


def test_cache_counters_survive_concurrent_builds():
    """hawk builds a bundle's host and device targets on two threads: the
    hit/miss counters lose no update when both record at once."""
    import threading

    from hawk.compile import cache as cache_mod

    cache_mod.reset_cache_stats()
    per_thread = 20_000

    def record(hit):
        for _ in range(per_thread):
            cache_mod.Cache.note(hit)

    threads = [threading.Thread(target=record, args=(hit,)) for hit in (True, False) * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache_mod.cache_stats() == {"hits": 2 * per_thread, "misses": 2 * per_thread}
    cache_mod.reset_cache_stats()
