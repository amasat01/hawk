# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A toolchain probe is paid once per toolchain, not once per process; a changed
toolchain never reads an old answer.

``--version`` / ``-dM -E`` / ``--list-gpu-arch`` answers persist in the cache directory
keyed by the binary's path, size and mtime; a validity record carries each closure file's
``(size, mtime_ns)`` so a fresh process trusts an unchanged header without re-reading it.
"""

from __future__ import annotations

import stat

import pytest

from hawk.compile import cache as cc
from hawk.compile import toolchain as tc


def _fake(path, version: str, log):
    path.write_text(f"#!/bin/sh\necho x >> {log}\necho 'fake {version}'\necho '#define __GNUC__ 13'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _spawns(log) -> int:
    return len(log.read_text().split()) if log.exists() else 0


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def test_a_probe_answers_from_disk_in_a_fresh_process(cache_dir):
    log, cc_ = cache_dir / "log", cache_dir / "fakecc"
    _fake(cc_, "1.0", log)
    first = tc.compiler_identity(str(cc_))
    macros = tc.compiler_macros(str(cc_))
    assert _spawns(log) == 2 and "__GNUC__" in macros
    # a new process: every per-process memo is empty
    tc.reset_identity_memo()
    tc._MACRO_MEMO.clear()
    tc._GCC_FAMILY_MEMO.clear()
    assert tc.compiler_identity(str(cc_)) == first
    assert tc.compiler_is_gcc(str(cc_)) is True
    assert _spawns(log) == 2, "a persisted probe must not respawn the compiler"


def test_a_replaced_compiler_never_reads_the_old_answer(cache_dir):
    log, cc_ = cache_dir / "log", cache_dir / "fakecc"
    _fake(cc_, "1.0", log)
    old = tc.compiler_identity(str(cc_))
    _fake(cc_, "2.0-with-a-longer-banner", log)          # same path, new size and mtime
    tc.reset_identity_memo()
    new = tc.compiler_identity(str(cc_))
    assert new != old and "2.0" in new


def test_a_different_toolchain_or_flags_is_a_different_key():
    a = cc.lookup_key("src", "cuda", "float64", "nvcc 12.8", ("-O3",))
    assert a != cc.lookup_key("src", "cuda", "float64", "nvcc 12.9", ("-O3",))
    assert a != cc.lookup_key("src", "cuda", "float64", "nvcc 12.8", ("-O2",))


def test_a_fresh_process_trusts_an_unchanged_closure_without_hashing(cache_dir, monkeypatch):
    header = cache_dir / "h.h"
    header.write_text("int x;\n")
    cache = cc.Cache(cache_dir / "slots")
    art = cache.slot("ab" + "0" * 62) / "k.cubin"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"x")
    key = "ab" + "0" * 62
    cache.store(key, artifact=art, closure=[str(header)], meta={})
    cc._MEMO.clear()                                    # a new process
    monkeypatch.setattr(cc, "_hash_bytes", lambda p: pytest.fail("re-hashed an unchanged header"))
    assert cache.check(key)
    monkeypatch.undo()
    header.write_text("int y_is_longer;\n")             # size changes -> content is read -> miss
    cc._MEMO.clear()
    assert not cache.check(key)


def test_a_changed_nvrtc_library_is_a_different_probe_key(tmp_path, monkeypatch):
    """The NVRTC version is read from a persisted probe keyed on the candidate
    libraries' path/size/mtime: a new or replaced ``libnvrtc.so`` under the search path
    is a new key, so a stale NVRTC never answers."""
    from hawk.compile import nvrtc as n

    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    before = n._library_fingerprint()
    (tmp_path / "libnvrtc.so.99").write_bytes(b"x")
    after = n._library_fingerprint()
    (tmp_path / "libnvrtc.so.99").write_bytes(b"xyz")
    assert len({before, after, n._library_fingerprint()}) == 3
